#!/usr/bin/env python3
"""
RQ4 Visualization: Extract and visualize routing probabilities from TCG modules.

Adapted for new TCG architecture (route_centroids + cosine similarity).
Only requires routing_probs - no intensity_gate needed.

Usage:
    python run_rq4_visualization.py --checkpoint <path> --dataset <name> --output <dir>
"""
import argparse
import os
import sys

import numpy as np
import torch

script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(script_dir, "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)


DATASET_CONFIG = {
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
    figsize=(7, 2.0),
    num_patterns=8,
):
    l = raw_input.shape[0]
    k = routing_probs.shape[0]
    time_steps = np.arange(l)
    dominant_pattern = np.argmax(routing_probs, axis=0)

    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "figure.dpi": 300,
    })

    fig, axes = plt.subplots(1, 3, figsize=figsize,
                              width_ratios=[2.0, 0.8, 2.2],
                              gridspec_kw={"wspace": 0.25})

    ax = axes[0]
    ax.plot(time_steps, raw_input, color="#333333", linewidth=1.0)
    ax.set_xlim(0, l - 1)
    ax.set_xticks(np.arange(0, l, max(1, l // 4)))
    ax.set_xticklabels(np.arange(0, l, max(1, l // 4)))
    ax.set_xlabel("Time Steps", fontsize=8)
    ax.set_ylabel("Input", fontsize=8)
    ax.set_title("(a) Input Time Series", fontsize=8, pad=4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="in", length=1.5)

    ax = axes[1]
    cmap = plt.cm.tab10
    for pid in range(num_patterns):
        mask = dominant_pattern == pid
        if mask.any():
            ax.fill_between(time_steps, 0, 1, where=mask,
                          color=cmap(pid % 10), alpha=0.7)
    ax.set_xlim(0, l - 1)
    ax.set_xticks(np.arange(0, l, max(1, l // 4)))
    ax.set_xticklabels(np.arange(0, l, max(1, l // 4)))
    ax.set_yticks([])
    ax.set_xlabel("Time Steps", fontsize=8)
    ax.set_ylabel("Pattern", fontsize=8)
    ax.set_title("(b) Dominant Pattern", fontsize=8, pad=4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="in", length=1.5)

    ax = axes[2]
    import seaborn as sns
    cmap_heat = sns.color_palette("YlGnBu", as_cmap=True)
    im = ax.imshow(routing_probs, aspect="auto", cmap=cmap_heat, vmin=0, vmax=1)
    ax.set_xlabel("Time Steps", fontsize=8)
    ax.set_xticks(np.arange(0, l, max(1, l // 5)))
    ax.set_xticklabels(np.arange(0, l, max(1, l // 5)))
    ax.set_yticks(np.arange(0, k, 2))
    ax.set_ylabel("Pattern ID", fontsize=8)
    ax.set_title("(c) Routing Probabilities", fontsize=8, pad=4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="in", length=1.5)

    cbar = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02, aspect=15)
    cbar.set_label(r"$\pi_k$", fontsize=8)
    cbar.ax.tick_params(labelsize=6)

    plt.savefig(f"{save_path}.pdf", dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.savefig(f"{save_path}.png", dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close()
    print(f"Saved: {save_path}.pdf / .png")


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
    parser.add_argument("--dataset", type=str, default="Illness",
                        choices=["Illness", "ExchangeRate"])
    parser.add_argument("--output", type=str, default="./rq4_visualizations",
                        help="Output directory")
    parser.add_argument("--split", type=str, default="test",
                        choices=["train", "val", "test"])
    parser.add_argument("--k", type=int, default=8, help="TCG num_patterns")
    parser.add_argument("--input_len", type=int, default=None)
    parser.add_argument("--output_len", type=int, default=None)
    parser.add_argument("--samples", type=int, default=8,
                        help="Number of samples to visualize")
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

    # routing_flat: (B_total * N, P, K) if single TCG
    # Infer N_actual from data
    N_actual = routing_flat.shape[0] // (B_total * P)
    if N_actual * B_total * P != routing_flat.shape[0]:
        N_actual = N  # fallback to dataset num_features
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

    prefix = "ili" if args.dataset == "Illness" else "exchange"
    save_path = os.path.join(args.output, f"{prefix}_tcm_routing")

    visualize_routing_interpretability(
        raw_norm,
        routing_vis,
        save_path,
        num_patterns=K,
    )

    print(f"\n=== Visualization Complete ===")
    print(f"Outputs: {save_path}.pdf / .png")


if __name__ == "__main__":
    main()
