#!/usr/bin/env python3
"""
Verify per-position regime switching in TCG routing.
Loads the checkpoint and visualizes routing at each time step.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
src_dir = os.path.join(project_root, 'src')
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from basicts.data import BasicTSForecastingDataset
from basicts.utils.constants import BasicTSMode
from torch.utils.data import DataLoader
from basicts.models.PatchTST import PatchTSTForForecasting, PatchTSTConfig
from basicts.configs import TCGConfig
from basicts.modules.tcg import TemporalContextualGating

# ============== CONFIG ==============
CHECKPOINT_PATH = 'checkpoints/test_logits/54f446e863da5c776561570df1d81d14/PatchTSTForForecasting_best_val_MAE.pt'
OUTPUT_DIR = 'vis/'
DATASET = 'Illness'
INPUT_LEN = 24
OUTPUT_LEN = 24
NUM_FEATURES = 7
NUM_PATTERNS = 8
PATCH_LEN = 16
PATCH_STRIDE = 8
BATCH_SIZE = 32
# ==================================

os.makedirs(OUTPUT_DIR, exist_ok=True)


class RoutingCaptor:
    def __init__(self, model):
        self.model = model
        self.routing_probs = []
        self._orig_forwards = {}
        
    def _patch(self, m):
        orig = m.forward
        def patched(x, return_aux=False):
            res, aux = orig(x, return_aux=True)
            if isinstance(aux, dict) and 'routing_probs' in aux:
                self.routing_probs.append(aux['routing_probs'].detach().cpu())
            return res
        self._orig_forwards[id(m)] = orig
        m.forward = patched
        
    def __enter__(self):
        for _, m in self.model.named_modules():
            if isinstance(m, TemporalContextualGating):
                self._patch(m)
        return self
    
    def __exit__(self, *args):
        for _, m in self.model.named_modules():
            if isinstance(m, TemporalContextualGating) and id(m) in self._orig_forwards:
                m.forward = self._orig_forwards[id(m)]


def load_model_and_capture():
    ckpt = torch.load(CHECKPOINT_PATH, map_location='cpu')
    
    model_cfg = PatchTSTConfig(
        input_len=INPUT_LEN,
        output_len=OUTPUT_LEN,
        num_features=NUM_FEATURES,
    )
    model_cfg.tcg = TCGConfig(
        enabled=True,
        num_patterns=NUM_PATTERNS,
        orth_lambda=0.01,
    )
    
    model = PatchTSTForForecasting(model_cfg)
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    model.eval()
    
    dataset = BasicTSForecastingDataset(
        dataset_name=DATASET,
        input_len=INPUT_LEN,
        output_len=OUTPUT_LEN,
        mode=BasicTSMode.TEST,
    )
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=0, shuffle=False)
    
    all_inputs = []
    with RoutingCaptor(model) as captor:
        for batch in dataloader:
            inputs = batch['inputs']
            with torch.no_grad():
                _ = model(inputs)
            all_inputs.append(inputs.numpy())
    
    inputs = np.concatenate(all_inputs, axis=0)
    routing = np.concatenate(captor.routing_probs, axis=0)
    
    print(f'Inputs shape: {inputs.shape}')
    print(f'Routing shape: {routing.shape}')
    
    return routing, inputs


def patch_to_timestep_routing(routing, input_len, patch_len, patch_stride):
    """Convert patch-level routing [B, num_patches, K] to timestep-level [B, input_len, K].
    
    Each patch covers patch_len consecutive timesteps.
    We weight-average the patch routing probabilities onto timesteps.
    """
    B, num_patches, K = routing.shape
    L = input_len
    
    # timestep_routing[b, t, k] = sum of routing[b, p, k] weighted by overlap
    timestep_routing = np.zeros((B, L, K))
    weight = np.zeros((B, L, 1))
    
    for p_idx in range(num_patches):
        start = p_idx * patch_stride
        end = min(start + patch_len, L)
        overlap = end - start
        if overlap > 0:
            timestep_routing[:, start:end, :] += routing[:, p_idx:p_idx+1, :].numpy() * overlap
            weight[:, start:end, :] += overlap
    
    timestep_routing = timestep_routing / (weight + 1e-6)
    return timestep_routing


def compute_regime_switches(timestep_routing):
    """Compute number of regime switches per sample.
    
    timestep_routing: [B, L, K] routing prob at each timestep
    Returns: dominant_patterns [B, L], switches_per_sample [B]
    """
    B, L, K = timestep_routing.shape
    dominant = np.argmax(timestep_routing, axis=-1)  # [B, L]
    switches = np.sum(np.diff(dominant, axis=1) != 0, axis=1)  # [B]
    return dominant, switches


def visualize(routing, inputs, output_path):
    B_total = inputs.shape[0]
    
    # routing: [B_total, num_patches, K]
    # inputs: [B_total, input_len, num_features]
    num_patches = routing.shape[1]
    
    # Map patch routing to timestep routing
    timestep_routing = patch_to_timestep_routing(routing, INPUT_LEN, PATCH_LEN, PATCH_STRIDE)
    dominant, switches = compute_regime_switches(timestep_routing)
    
    # Per-feature analysis (use feature 0)
    feat_idx = 0
    timestep_routing_feat = timestep_routing[:, :, :]  # [B, L, K]
    dominant_feat = dominant  # [B, L]
    switches_feat = switches  # [B]
    
    # Select sample with most switches for visualization
    best_sample = np.argmax(switches_feat)
    print(f'\n=== Regime Switching Analysis ===')
    print(f'Total samples: {B_total}')
    print(f'Avg switches per sample: {switches_feat.mean():.2f}')
    print(f'Max switches: {switches_feat.max()}')
    print(f'Samples with 0 switches: {(switches_feat == 0).sum()}')
    print(f'Samples with >=3 switches: {(switches_feat >= 3).sum()}')
    
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.size"] = 10
    
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 2, hspace=0.4, wspace=0.35)
    
    # (a) Heatmap of routing probabilities over time for best sample
    ax1 = fig.add_subplot(gs[0, :])
    heat_data = timestep_routing_feat[best_sample].T  # [K, L]
    sns.heatmap(heat_data, cmap='YlOrRd', vmin=0, vmax=1,
                xticklabels=range(0, INPUT_LEN, 4),
                yticklabels=[f'P{i}' for i in range(NUM_PATTERNS)],
                ax=ax1, cbar_kws={'label': 'Routing Probability', 'shrink': 0.6})
    ax1.set_xlabel('Time Step', fontsize=11)
    ax1.set_ylabel('Pattern ID', fontsize=11)
    ax1.set_title(f'(a) Routing Probability Heatmap (Sample {best_sample}, Feature {feat_idx})\n'
                  f'Regime switches: {switches_feat[best_sample]} | '
                  f'Pattern usage std: {np.std(timestep_routing_feat[best_sample].mean(axis=0)):.3f}',
                  fontsize=11, fontweight='bold', loc='left')
    
    # (b) Input signal + dominant pattern regions for best sample
    ax2 = fig.add_subplot(gs[1, 0])
    x = np.arange(INPUT_LEN)
    inp = inputs[best_sample, :, feat_idx]
    inp_norm = (inp - inp.min()) / (inp.max() - inp.min() + 1e-6)
    ax2.plot(x, inp_norm, 'k-', linewidth=1.5, label='Input (normalized)', alpha=0.7)
    dom = dominant_feat[best_sample]
    colors = plt.cm.tab10(np.linspace(0, 1, NUM_PATTERNS))
    for k in range(NUM_PATTERNS):
        mask = (dom == k)
        if mask.any():
            ax2.fill_between(x, 0, 1, where=mask, alpha=0.3, color=colors[k], label=f'P{k}')
    ax2.set_xlabel('Time Step', fontsize=11)
    ax2.set_ylabel('Normalized Input', fontsize=11)
    ax2.set_title(f'(b) Dominant Pattern Over Time\nColored regions = active pattern',
                  fontsize=11, fontweight='bold', loc='left')
    ax2.legend(loc='upper right', ncol=4, fontsize=8)
    ax2.set_ylim(-0.05, 1.05)
    ax2.grid(True, linestyle='--', alpha=0.3)
    
    # (c) Histogram of regime switches
    ax3 = fig.add_subplot(gs[1, 1])
    bins = np.arange(-0.5, switches_feat.max() + 1.5, 1)
    ax3.hist(switches_feat, bins=bins, color='steelblue', alpha=0.8, edgecolor='black')
    ax3.axvline(switches_feat.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean={switches_feat.mean():.2f}')
    ax3.set_xlabel('Number of Regime Switches per Sample', fontsize=11)
    ax3.set_ylabel('Count', fontsize=11)
    ax3.set_title(f'(c) Distribution of Regime Switches\n'
                  f'{(switches_feat >= 3).sum()} samples ({(switches_feat >= 3).mean()*100:.1f}%) have >=3 switches',
                  fontsize=11, fontweight='bold', loc='left')
    ax3.legend()
    ax3.grid(axis='y', linestyle='--', alpha=0.3)
    
    fig.suptitle(f'TCG Per-Position Regime Switching Verification\n'
                f'Dataset: {DATASET} | K={NUM_PATTERNS} patterns | Input L={INPUT_LEN} | Checkpoint: {os.path.basename(CHECKPOINT_PATH)}',
                fontsize=12, fontweight='bold', y=0.98)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'\nSaved: {output_path}')


def main():
    print('Loading checkpoint and capturing routing...')
    routing, inputs = load_model_and_capture()
    
    print('Computing per-position regime analysis...')
    output_path = os.path.join(OUTPUT_DIR, 'regime_switching_verify.pdf')
    visualize(routing, inputs, output_path)


if __name__ == '__main__':
    main()
