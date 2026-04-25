#!/usr/bin/env python3
"""
RQ4 Visualization: Extract and visualize routing probabilities from TCG modules.

Adapted for new TCG architecture (route_centroids + cosine similarity).
Only requires routing_probs - no intensity_gate needed.

Usage:
    python run_rq4_visualization.py --checkpoint <path> --dataset <name>

Recommended (multi-regime, longer context):
    python run_rq4_visualization.py \
        --checkpoint checkpoints/test_ablation/a849f714f87d07857a262cf8bd4b6e68/PatchTSTForForecasting_best_val_MAE.pt \
        --dataset ExchangeRate --input_len 96 --output_len 96 --k 8 \
        --samples 16 --fig_w 10 --fig_h 2.6
"""
import argparse
import os
import sys

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(script_dir, "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)


DATASET_CONFIG = {
    "Sunspots": {"input_len": 36, "output_len": 12, "num_features": 1},
    "Illness": {"input_len": 24, "output_len": 24, "num_features": 7},
    "ExchangeRate": {"input_len": 336, "output_len": 96, "num_features": 8},
}


def patch_to_timestep_range(patch_idx, patch_len=16, stride=8, input_len=96):
    start = patch_idx * stride
    end = min(start + patch_len, input_len)
    return start, end


def aggregate_routing_to_timestep(routing, input_len, patch_len=16, stride=8, method="weighted"):
    b, n, p, k = routing.shape
    timestep_routing = np.zeros((b, n, input_len, k))
    for feat_idx in range(n):
        for patch_idx in range(p):
            start, end = patch_to_timestep_range(patch_idx, patch_len, stride, input_len)
            if method == "weighted":
                timestep_routing[:, feat_idx, start:end, :] += routing[:, feat_idx, patch_idx:patch_idx+1, :] * (end - start)
            else:
                timestep_routing[:, feat_idx, start:end, :] += routing[:, feat_idx, patch_idx, :]
    if method == "weighted":
        total_weight = np.zeros((b, n, input_len, 1))
        for patch_idx in range(p):
            start, end = patch_to_timestep_range(patch_idx, patch_len, stride, input_len)
            total_weight[:, :, start:end, :] += (end - start)
        timestep_routing = timestep_routing / (total_weight + 1e-6)
    return timestep_routing

def visualize_routing_interpretability(
    raw_input,
    routing_probs,
    save_path,
    figsize=(14.5, 3.5),
    num_patterns=8,
    enhance_contrast=True,
):
    import pandas as pd 

    l = raw_input.shape[0]
    k = routing_probs.shape[0]
    time_steps = np.arange(l)
    dominant_pattern = np.argmax(routing_probs, axis=0)

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 11,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.dpi": 300,
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "axes.linewidth": 1.2,    # 加粗外边框
    })

    # 【关键修改3】增加 wspace 解决标题重叠，调整 width_ratios 给热力图留足空间
    fig, axes = plt.subplots(
        1, 3,
        figsize=figsize,
        width_ratios=[1.0, 1.0, 1.25], 
        gridspec_kw={"wspace": 0.35},
    )

    premium_dom_palette = [
        "#FAD2CD", "#CDEAF3", "#BDE4DE", "#E2E6F2", 
        "#FCECE5", "#CED6E5", "#E3F4F1", "#FCE4C5"
    ]

    changes = np.where(np.diff(dominant_pattern) != 0)[0] + 1
    boundaries = [0] + list(changes) + [l - 1]

    def draw_regime_background(ax):
        for i in range(len(boundaries) - 1):
            start_idx = boundaries[i]
            end_idx = boundaries[i+1]
            regime = dominant_pattern[start_idx]
            bg_color = premium_dom_palette[regime % len(premium_dom_palette)]

            ax.axvspan(start_idx, end_idx, facecolor=bg_color, alpha=0.35, zorder=1, edgecolor="none")

            if start_idx > 0:
                ax.axvline(start_idx, color="#9CA3AF", linewidth=1.2, alpha=0.8, linestyle=(0, (3, 3)), zorder=2)

    # ==========================================
    # (a) The "What": Input Time Series & Regimes
    # ==========================================
    ax = axes[0]
    draw_regime_background(ax)
    
    ax.plot(time_steps, raw_input, color="#111827", linewidth=1.8, zorder=3)
    
    ax.set_xlim(0, l - 1)
    y_min, y_max = raw_input.min(), raw_input.max()
    ax.set_ylim(y_min - (y_max-y_min)*0.05, y_max + (y_max-y_min)*0.05)
    
    ax.set_xticks(np.arange(0, l, max(1, l // 4)))
    ax.set_xticklabels(np.arange(0, l, max(1, l // 4)))
    ax.set_xlabel("Time Steps")
    ax.set_ylabel("Input Signal")
    ax.set_title("(a) Input Signal & Regimes", pad=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="in", length=4, width=1.0, colors="#374151")

    # ==========================================
    # (b) The "Why": Rolling Volatility & Regimes
    # ==========================================
    ax = axes[1]
    draw_regime_background(ax)
    
    rolling_std = pd.Series(raw_input).rolling(window=16, min_periods=1, center=True).std().fillna(method='bfill').values
    
    vol_color = "#DC2626"
    ax.plot(time_steps, rolling_std, color=vol_color, linewidth=1.8, zorder=3)
    ax.fill_between(time_steps, rolling_std, 0, color=vol_color, alpha=0.08, zorder=2) 
    
    ax.set_xlim(0, l - 1)
    ax.set_ylim(0, rolling_std.max() * 1.15)
    
    ax.set_xticks(np.arange(0, l, max(1, l // 4)))
    ax.set_xticklabels(np.arange(0, l, max(1, l // 4)))
    ax.set_xlabel("Time Steps")
    ax.set_ylabel("Local Volatility (Std)")
    ax.set_title("(b) Regime Driver (Volatility)", pad=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="in", length=4, width=1.0, colors="#374151")

    # ==========================================
    # (c) The "How": Routing Probabilities
    # ==========================================
    ax = axes[2]
    
    cmap_heat = mcolors.LinearSegmentedColormap.from_list(
        "CustomDeepBlue", ["#FAFAFA", "#D2E0F0", "#7DA7D9", "#245B9D", "#0A254D"]
    )
    
    if enhance_contrast:
        rp_min = float(np.percentile(routing_probs, 2))
        rp_max = float(np.percentile(routing_probs, 98))
        if rp_max - rp_min < 1e-6:
            rp_min = float(np.min(routing_probs))
            rp_max = float(np.max(routing_probs))
        if rp_max - rp_min < 1e-6:
            rp_min, rp_max = 0.0, 1.0
        show_routing = np.clip(routing_probs, rp_min, rp_max)
    else:
        rp_min, rp_max = 0.0, 1.0
        show_routing = routing_probs

    im = ax.imshow(show_routing, aspect="auto", cmap=cmap_heat, vmin=rp_min, vmax=rp_max)
    
    ax.set_xlabel("Time Steps")
    ax.set_xticks(np.arange(0, l, max(1, l // 4)))
    ax.set_xticklabels(np.arange(0, l, max(1, l // 4)))
    ax.set_yticks(np.arange(0, k, max(1, k // 4)))
    ax.set_ylabel("Latent Basis ID")
    ax.set_title("(c) Routing Probabilities", pad=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="in", length=4, width=1.0, colors="#374151")

    # Colorbar排版优化
    cbar = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.04, aspect=20)
    cbar.outline.set_linewidth(0.8)
    cbar.outline.set_edgecolor('#9CA3AF')
    cbar_label = r"$\pi_k$"
    if enhance_contrast:
        cbar_label += " (Stretched)"
    cbar.set_label(cbar_label, fontsize=11, labelpad=6)
    cbar.ax.tick_params(labelsize=9, length=3, width=0.8, direction="out")

    # 【关键修改5】确保保存时使用白底，去掉外层可能存在的黑框
    plt.tight_layout()
    plt.savefig(f"{save_path}.pdf", dpi=300, bbox_inches="tight", pad_inches=0.05, facecolor='white', edgecolor='none')
    plt.close()
    print(f"Saved: {save_path}.pdf")

def select_best_sample(routing_timestep, time_series_batch, num_features=1):
    b, n, l, k = routing_timestep.shape
    best_score = -1
    best_idx = 0
    best_feat = 0
    for batch_idx in range(b):
        for feat_idx in range(min(n, num_features)):
            rp = routing_timestep[batch_idx, feat_idx]
            dom_pat = np.argmax(rp, axis=-1)
            pattern_changes = np.sum(np.diff(dom_pat) != 0)
            confidence = np.mean(np.max(rp, axis=-1))
            score = pattern_changes * 2.0 + confidence * 1.0
            if score > best_score:
                best_score = score
                best_idx = batch_idx
                best_feat = feat_idx
                print(f"  batch={batch_idx}, feat={feat_idx}, "
                      f"switches={pattern_changes}, confidence={confidence:.3f}, score={score:.3f}")
    print(f"Selected: batch={best_idx}, feat={best_feat}")
    return best_idx, best_feat


class RoutingCaptor:
    def __init__(self, model):
        self.model = model
        self.routing_probs = []
        self.hooks = []
        self._orig_forwards = {}

    def _patch_tcg_forward(self, tcg_module):
        from basicts.modules.tcg import TemporalContextualGating
        orig_forward = tcg_module.forward

        def patched(x, return_aux=False):
            result, aux = orig_forward(x, return_aux=True)
            if isinstance(aux, dict) and "routing_probs" in aux:
                self.routing_probs.append(aux["routing_probs"].detach().cpu())
            if return_aux:
                return result, aux
            return result

        self._orig_forwards[id(tcg_module)] = orig_forward
        tcg_module.forward = patched

    def _restore(self):
        from basicts.modules.tcg import TemporalContextualGating
        for name, module in self.model.named_modules():
            if isinstance(module, TemporalContextualGating) and id(module) in self._orig_forwards:
                module.forward = self._orig_forwards[id(module)]
        self._orig_forwards.clear()

    def __enter__(self):
        from basicts.modules.tcg import TemporalContextualGating
        for name, module in self.model.named_modules():
            if isinstance(module, TemporalContextualGating):
                self._patch_tcg_forward(module)
        return self

    def __exit__(self, *args):
        self._restore()


def main():
    parser = argparse.ArgumentParser(description="RQ4 Visualization for TCG Routing")
    parser.add_argument("--checkpoint", type=str,
                        default="checkpoints/test_logits/9378b912cebedfed27f6fff60f402473/PatchTSTForForecasting_best_val_MAE.pt",
                        help="Path to model checkpoint")
    parser.add_argument("--dataset", type=str, default="Sunspots",
                        choices=["Sunspots", "Illness", "ExchangeRate"])
    parser.add_argument("--output", type=str, default="./vis",
                        help="Output directory")
    parser.add_argument("--split", type=str, default="test",
                        choices=["train", "val", "test"])
    parser.add_argument("--k", type=int, default=8, help="TCG num_patterns")
    parser.add_argument("--input_len", type=int, default=None)
    parser.add_argument("--output_len", type=int, default=None)
    parser.add_argument("--samples", type=int, default=8,
                        help="Number of samples to visualize")
    parser.add_argument("--fig_w", type=float, default=11.4,
                        help="Figure width in inches (3 columns ≈ 3/4 of 15.2 inch row)")
    parser.add_argument("--fig_h", type=float, default=3.2,
                        help="Figure height in inches (match parameter_sensitivity row height)")
    args = parser.parse_args()

    dataset_cfg = DATASET_CONFIG[args.dataset]
    input_len = args.input_len or dataset_cfg["input_len"]
    output_len = args.output_len or dataset_cfg["output_len"]
    num_features = dataset_cfg["num_features"]

    os.makedirs(args.output, exist_ok=True)

    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    if "model" in ckpt:
        state_dict = ckpt["model"]
    elif "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    from basicts.models.PatchTST import PatchTSTForForecasting, PatchTSTConfig
    from basicts.configs import TCGConfig

    cfg = PatchTSTConfig(
        input_len=input_len,
        output_len=output_len,
        num_features=num_features,
        patch_len=16,
        patch_stride=8,
    )
    cfg.tcg = TCGConfig(
        enabled=True,
        num_patterns=args.k,
        orth_lambda=0.0001,
        use_multiscale=True,
        identity_init=True,
        discrete_topk=1,
    )

    model = PatchTSTForForecasting(cfg)
    try:
        model.load_state_dict(state_dict, strict=False)
        print("Checkpoint loaded successfully")
    except Exception as e:
        print(f"Warning: Could not load strictly: {e}")

    model.to(device)
    model.eval()

    print(f"Loading dataset: {args.dataset}")
    from basicts.data import BasicTSForecastingDataset
    from torch.utils.data import DataLoader
    from basicts.utils.constants import BasicTSMode

    dataset = BasicTSForecastingDataset(
        dataset_name=args.dataset,
        input_len=input_len,
        output_len=output_len,
        mode=BasicTSMode.TEST,
    )
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=args.samples,
        num_workers=0,
        shuffle=False,
    )

    print("Running inference to capture routing probabilities...")
    all_routing = []
    all_ts = []

    with RoutingCaptor(model) as captor:
        for batch_idx, batch in enumerate(dataloader):
            inputs = batch["inputs"].to(device)
            with torch.no_grad():
                _ = model(inputs)
            if captor.routing_probs:
                rp = torch.cat(captor.routing_probs, dim=0)
                all_routing.append(rp)
                captor.routing_probs.clear()
            all_ts.append(inputs.cpu().numpy())

    if not all_routing:
        print("ERROR: No routing captured. Is TCG enabled?")
        return

    routing_flat = torch.cat(all_routing, dim=0).numpy()
    time_series_batch = np.concatenate(all_ts, axis=0)

    print(f"Routing shape: {routing_flat.shape}")
    print(f"Time series shape: {time_series_batch.shape}")

    B_total, L, N = time_series_batch.shape
    K = args.k
    P = routing_flat.shape[1]

    N_actual = routing_flat.shape[0] // (B_total * P)
    if N_actual * B_total * P != routing_flat.shape[0]:
        N_actual = N  
    routing_reshaped = routing_flat.reshape(B_total, N_actual, P, K)

    routing_timestep = aggregate_routing_to_timestep(
        routing_reshaped, input_len, patch_len=16, stride=8, method="weighted"
    )

    print("\nScoring samples by routing dynamics...")
    best_b, best_f = select_best_sample(routing_timestep, time_series_batch, num_features=1)

    raw_sel = time_series_batch[best_b, :, best_f]
    rp_sel = routing_timestep[best_b, best_f]
    routing_vis = rp_sel.T

    raw_min, raw_max = raw_sel.min(), raw_sel.max()
    if raw_max - raw_min > 1e-6:
        raw_norm = (raw_sel - raw_min) / (raw_max - raw_min)
    else:
        raw_norm = raw_sel

    prefix_map = {
        "Sunspots": "sunspots",
        "Illness": "ili",
        "ExchangeRate": "exchange",
    }
    prefix = prefix_map.get(args.dataset, args.dataset.lower())
    save_path = os.path.join(args.output, f"{prefix}_tcm_routing")

    visualize_routing_interpretability(
        raw_norm,
        routing_vis,
        save_path,
        figsize=(args.fig_w, args.fig_h),
        num_patterns=K,
    )

    print(f"\n=== Visualization Complete ===")
    print(f"Outputs: {save_path}.pdf")


if __name__ == "__main__":
    main()