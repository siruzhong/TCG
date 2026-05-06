#!/usr/bin/env python3
"""
RQ4 Visualization: RAW vs DPR prediction comparison + regime analysis.

Layout (1 × 4):
  (a) RAW baseline prediction   (b) DPR prediction
  (c) Input signal + volatility (merged)   (d) Routing probabilities

Usage:
    python run_rq4_visualization.py \
    --checkpoint checkpoints/test_ablation/a849f714f87d07857a262cf8bd4b6e68/PatchTSTForForecasting_best_val_MAE.pt \
    --dataset ExchangeRate --input_len 96 --output_len 96 --k 8 --samples 16 \
    --fig_w 18 --fig_h 3.0 \
    --before_npy checkpoints_old/Informer/ExchangeRate_100_96_96/1814d14edc8c6ef91fb05b73b0b47a82 \
    --after_npy checkpoints_old/Informer/ExchangeRate_100_96_96/a6a6366d9c46c0535fd10bfe825b747b --indices 1010 1041  
"""
import argparse
import os
import sys
from pathlib import Path
from collections import OrderedDict

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


def visualize_combined(
    raw_input,
    input_seq,
    output_gt,
    raw_pred,
    dpr_pred,
    routing_probs,
    save_path,
    figsize=(18, 3.0),
    num_patterns=8,
    enhance_contrast=True,
    output_len=96,
    input_len_val=96,
    sample2_input=None,
    sample2_gt=None,
    sample2_raw=None,
    sample2_dpr=None,
):
    import pandas as pd

    l = raw_input.shape[0]
    k = routing_probs.shape[0]
    time_steps = np.arange(l)
    dominant_pattern = np.argmax(routing_probs, axis=0)

    sns.set_theme(style="whitegrid", font_scale=1.0, rc={
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
        "axes.linewidth": 1.2,
    })

    fig, axes = plt.subplots(
        1, 4,
        figsize=figsize,
        width_ratios=[1.0, 1.0, 1.0, 1.25],
        gridspec_kw={"wspace": 0.18},
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
            end_idx = boundaries[i + 1]
            regime = dominant_pattern[start_idx]
            bg_color = premium_dom_palette[regime % len(premium_dom_palette)]
            ax.axvspan(start_idx, end_idx, facecolor=bg_color, alpha=0.35, zorder=1, edgecolor="none")
            if start_idx > 0:
                ax.axvline(start_idx, color="#9CA3AF", linewidth=1.2, alpha=0.8, linestyle=(0, (3, 3)), zorder=2)

    t_in = np.arange(input_len_val)
    t_out = np.arange(input_len_val, input_len_val + output_len)

    def style_pred_ax(ax, title):
        ax.set_xlim(0, input_len_val + output_len)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(direction="in", length=4, width=1.0, colors="#374151")
        ax.set_title(title, pad=12)
        ax.set_xlabel("Time Steps")

    def style_ax(ax, title):
        ax.set_xlabel("Time Steps")
        ax.set_xlim(0, l - 1)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(direction="in", length=4, width=1.0, colors="#374151")
        ax.set_title(title, pad=12)

    def pred_panel(ax, inp, gt, base, ours, title):
        ax.plot(t_in, inp, color="#7F8C8D", linewidth=1.5, alpha=0.5, label="Input", zorder=3)
        ax.plot(t_out, gt, color="#27AE60", linewidth=2.0, label="GT", zorder=4)
        if base is not None:
            ax.plot(t_out, base, color="#D62728", linewidth=2.0, ls="--", label="Baseline", zorder=5)
        ax.plot(t_out, ours, color="#1F77B4", linewidth=2.0, ls="-", label="DPR", zorder=6)
        ax.axvline(x=input_len_val - 1, color="#34495E", linestyle=":", lw=1.2, zorder=3)
        style_pred_ax(ax, title)
        ax.legend(loc="upper left", ncol=1, framealpha=0.9, edgecolor="gray", fontsize=10)

    # ==========================================
    # (a) Sample 1: Baseline + DPR overlaid
    # ==========================================
    ax = axes[0]
    pred_panel(ax, input_seq, output_gt, raw_pred, dpr_pred, "(a) Prediction Comparison #1")

    # ==========================================
    # (b) Sample 2: Baseline + DPR overlaid
    # ==========================================
    ax = axes[1]
    inp2 = sample2_input if sample2_input is not None else input_seq
    gt2 = sample2_gt if sample2_gt is not None else output_gt
    raw2 = sample2_raw if sample2_raw is not None else raw_pred
    dpr2 = sample2_dpr if sample2_dpr is not None else dpr_pred
    pred_panel(ax, inp2, gt2, raw2, dpr2, "(b) Prediction Comparison #2")

    # ==========================================
    # (c) Input signal & volatility (merged)
    # ==========================================
    ax = axes[2]
    draw_regime_background(ax)

    rolling_std = pd.Series(raw_input).rolling(window=16, min_periods=1, center=True).std().bfill().values

    vol_color = "#DC2626"
    ax.fill_between(time_steps, rolling_std, 0, color=vol_color, alpha=0.08, zorder=2)
    ax.plot(time_steps, rolling_std, color=vol_color, linewidth=1.5, alpha=0.35, zorder=3)

    ax.plot(time_steps, raw_input, color="#111827", linewidth=1.8, zorder=4)

    ax.set_xlim(0, l - 1)
    y_min, y_max = raw_input.min(), raw_input.max()
    ax.set_ylim(y_min - (y_max - y_min) * 0.05, y_max + (y_max - y_min) * 0.05)
    ax.set_xticks(np.arange(0, l, max(1, l // 4)))
    ax.set_xticklabels(np.arange(0, l, max(1, l // 4)))
    style_ax(ax, "(c) Signal & Volatility")
    ax.set_ylabel("Normalized Value")
    from matplotlib.lines import Line2D
    ax.legend(
        handles=[Line2D([0], [0], color="#111827", lw=1.8, label="Signal"),
                 Line2D([0], [0], color=vol_color, lw=1.5, alpha=0.35, label="Volatility")],
        loc="upper left", ncol=1, framealpha=0.9, edgecolor="gray", fontsize=10,
    )

    # ==========================================
    # (d) Routing Probabilities
    # ==========================================
    ax = axes[3]

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
    ax.set_title("(d) Routing Probabilities", pad=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="in", length=4, width=1.0, colors="#374151")

    cbar = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.04, aspect=20)
    cbar.outline.set_linewidth(0.8)
    cbar.outline.set_edgecolor("#9CA3AF")
    cbar_label = r"$\pi_k$"
    if enhance_contrast:
        cbar_label += " (Stretched)"
    cbar.set_label(cbar_label, fontsize=11, labelpad=6)
    cbar.ax.tick_params(labelsize=9, length=3, width=0.8, direction="out")

    plt.savefig(f"{save_path}.pdf", dpi=300, bbox_inches="tight", pad_inches=0.05, facecolor="white", edgecolor="none")
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
    return best_idx, best_feat


class RoutingCaptor:
    def __init__(self, model):
        self.model = model
        self.routing_probs = []
        self._orig_forwards = {}

    def _patch_dpr_forward(self, dpr_module):
        from basicts.modules.dpr import TemporalContextualGating
        orig_forward = dpr_module.forward

        def patched(x, return_aux=False):
            result, aux = orig_forward(x, return_aux=True)
            if isinstance(aux, dict) and "routing_probs" in aux:
                self.routing_probs.append(aux["routing_probs"].detach().cpu())
            if return_aux:
                return result, aux
            return result

        self._orig_forwards[id(dpr_module)] = orig_forward
        dpr_module.forward = patched

    def _restore(self):
        from basicts.modules.dpr import TemporalContextualGating
        for name, module in self.model.named_modules():
            if isinstance(module, TemporalContextualGating) and id(module) in self._orig_forwards:
                module.forward = self._orig_forwards[id(module)]
        self._orig_forwards.clear()

    def __enter__(self):
        from basicts.modules.dpr import TemporalContextualGating
        for name, module in self.model.named_modules():
            if isinstance(module, TemporalContextualGating):
                self._patch_dpr_forward(module)
        return self

    def __exit__(self, *args):
        self._restore()


def dpr_bypass_context(model):
    class _BypassCtx:
        def __init__(self):
            self.orig_forwards = {}
        def __enter__(self):
            from basicts.modules.dpr import TemporalContextualGating
            for name, module in model.named_modules():
                if isinstance(module, TemporalContextualGating):
                    self.orig_forwards[id(module)] = module.forward
                    module.forward = lambda x, return_aux=False: (x, {})
            return self
        def __exit__(self, *args):
            from basicts.modules.dpr import TemporalContextualGating
            for name, module in model.named_modules():
                if isinstance(module, TemporalContextualGating) and id(module) in self.orig_forwards:
                    module.forward = self.orig_forwards[id(module)]
    return _BypassCtx()
    def __init__(self, model):
        self.model = model
        self.routing_probs = []
        self._orig_forwards = {}

    def _patch_dpr_forward(self, dpr_module):
        from basicts.modules.dpr import TemporalContextualGating
        orig_forward = dpr_module.forward

        def patched(x, return_aux=False):
            result, aux = orig_forward(x, return_aux=True)
            if isinstance(aux, dict) and "routing_probs" in aux:
                self.routing_probs.append(aux["routing_probs"].detach().cpu())
            if return_aux:
                return result, aux
            return result

        self._orig_forwards[id(dpr_module)] = orig_forward
        dpr_module.forward = patched

    def _restore(self):
        from basicts.modules.dpr import TemporalContextualGating
        for name, module in self.model.named_modules():
            if isinstance(module, TemporalContextualGating) and id(module) in self._orig_forwards:
                module.forward = self._orig_forwards[id(module)]
        self._orig_forwards.clear()

    def __enter__(self):
        from basicts.modules.dpr import TemporalContextualGating
        for name, module in self.model.named_modules():
            if isinstance(module, TemporalContextualGating):
                self._patch_dpr_forward(module)
        return self

    def __exit__(self, *args):
        self._restore()


def load_model_for_inference(checkpoint_path, input_len, output_len, num_features, k,
                              dpr_enabled=True, device="cpu"):
    from basicts.models.PatchTST import PatchTSTForForecasting, PatchTSTConfig
    from basicts.configs import DPRConfig

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    if "model" in ckpt:
        state_dict = ckpt["model"]
    elif "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    elif "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    else:
        state_dict = ckpt

    cfg = PatchTSTConfig(
        input_len=input_len, output_len=output_len, num_features=num_features,
        patch_len=16, patch_stride=8, hidden_size=256,
    )

    if dpr_enabled:
        cfg.dpr = DPRConfig(
            enabled=True, num_patterns=k, orth_lambda=0.0001,
            use_multiscale=False, identity_init=True, discrete_topk=1,
        )

    model = PatchTSTForForecasting(cfg)

    new_sd = OrderedDict()
    for k_sd, v in state_dict.items():
        new_k = k_sd
        if k_sd.startswith("model.") and not k_sd.startswith("model.model."):
            new_k = k_sd
        new_sd[new_k] = v

    try:
        model.load_state_dict(new_sd, strict=False)
    except Exception as e:
        print(f"Warning: non-strict load: {e}")

    model.to(device)
    model.eval()
    return model


def run_inference_batch(model, dataloader, device, capture_routing=False):
    import torch
    all_preds = []
    all_ts = []
    all_targets = []
    routing_captor = RoutingCaptor(model) if capture_routing else None

    ctx = routing_captor if routing_captor else torch.no_grad()
    if capture_routing:
        with routing_captor:
            for batch in dataloader:
                inputs = batch["inputs"].to(device)
                targets = batch.get("targets", batch.get("target"))
                with torch.no_grad():
                    output = model(inputs)
                pred = output["prediction"].cpu().numpy() if isinstance(output, dict) else output.cpu().numpy()
                all_preds.append(pred)
                all_ts.append(inputs.cpu().numpy())
                if targets is not None:
                    all_targets.append(targets.cpu().numpy())
        routing = torch.cat(routing_captor.routing_probs, dim=0).numpy() if routing_captor.routing_probs else None
    else:
        with torch.no_grad():
            for batch in dataloader:
                inputs = batch["inputs"].to(device)
                targets = batch.get("targets", batch.get("target"))
                output = model(inputs)
                pred = output["prediction"].cpu().numpy() if isinstance(output, dict) else output.cpu().numpy()
                all_preds.append(pred)
                all_ts.append(inputs.cpu().numpy())
                if targets is not None:
                    all_targets.append(targets.cpu().numpy())
        routing = None

    ts = np.concatenate(all_ts, axis=0)
    preds = np.concatenate(all_preds, axis=0)
    tgts = np.concatenate(all_targets, axis=0) if all_targets else None
    return ts, preds, tgts, routing


def main():
    parser = argparse.ArgumentParser(description="RQ4 Visualization with prediction comparison")
    parser.add_argument("--checkpoint", type=str,
                        default="checkpoints/test_ablation/a849f714f87d07857a262cf8bd4b6e68/PatchTSTForForecasting_best_val_MAE.pt",
                        help="Path to DPR model checkpoint")
    parser.add_argument("--before_ckpt", type=str, default=None,
                        help="Path to RAW/baseline checkpoint for comparison")
    parser.add_argument("--before_npy", type=str, default=None,
                        help="Path to RAW directory with test_results/*.npy")
    parser.add_argument("--after_npy", type=str, default=None,
                        help="Path to DPR directory with test_results/*.npy")
    parser.add_argument("--indices", type=int, nargs="+", default=None,
                        help="Sample indices for prediction panels (e.g. 1010 1210)")
    parser.add_argument("--dataset", type=str, default="ExchangeRate",
                        choices=["Sunspots", "Illness", "ExchangeRate"])
    parser.add_argument("--output", type=str, default="./vis", help="Output directory")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--k", type=int, default=8, help="DPR num_patterns")
    parser.add_argument("--input_len", type=int, default=None)
    parser.add_argument("--output_len", type=int, default=None)
    parser.add_argument("--samples", type=int, default=16,
                        help="Number of samples to visualize (batch size)")
    parser.add_argument("--fig_w", type=float, default=18.0, help="Figure width")
    parser.add_argument("--fig_h", type=float, default=3.0, help="Figure height")
    args = parser.parse_args()

    dataset_cfg = DATASET_CONFIG[args.dataset]
    input_len = args.input_len or dataset_cfg["input_len"]
    output_len = args.output_len or dataset_cfg["output_len"]
    num_features = dataset_cfg["num_features"]

    os.makedirs(args.output, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    from basicts.data import BasicTSForecastingDataset
    from torch.utils.data import DataLoader
    from basicts.utils.constants import BasicTSMode

    dataset = BasicTSForecastingDataset(
        dataset_name=args.dataset, input_len=input_len, output_len=output_len,
        mode=BasicTSMode.TEST,
    )
    dataloader = DataLoader(
        dataset=dataset, batch_size=args.samples, num_workers=0, shuffle=False,
    )

    # Load DPR model
    print(f"Loading DPR checkpoint: {args.checkpoint}")
    dpr_model = load_model_for_inference(
        args.checkpoint, input_len, output_len, num_features, args.k,
        dpr_enabled=True, device=device,
    )

    print("Running DPR inference...")
    ts_batch, dpr_preds, tgts_batch, routing_flat = run_inference_batch(
        dpr_model, dataloader, device, capture_routing=True,
    )

    if routing_flat is None:
        print("ERROR: No routing captured. Is DPR enabled?")
        return

    B_total, L, N = ts_batch.shape
    K = args.k
    P = routing_flat.shape[1]
    N_actual = routing_flat.shape[0] // (B_total * P)
    if N_actual * B_total * P != routing_flat.shape[0]:
        N_actual = N
    routing_reshaped = routing_flat.reshape(B_total, N_actual, P, K)
    routing_timestep = aggregate_routing_to_timestep(
        routing_reshaped, input_len, patch_len=16, stride=8, method="weighted",
    )

    print("Scoring samples by routing dynamics...")
    best_b, best_f = select_best_sample(routing_timestep, ts_batch, num_features=1)

    raw_sel = ts_batch[best_b, :, best_f]
    rp_sel = routing_timestep[best_b, best_f]
    dpr_pred_sel = dpr_preds[best_b, :, best_f]
    input_seq_sel = ts_batch[best_b, :, best_f]
    output_gt_sel = tgts_batch[best_b, :, best_f] if tgts_batch is not None else np.zeros(output_len)
    routing_vis = rp_sel.T

    # Load RAW / DPR predictions from npy files (if provided), else fallback to bypass
    raw_pred_sel = None
    raw_preds = None
    if args.before_npy and args.after_npy:
        from pathlib import Path
        def _load_preds_npy(npy_dir, input_len, output_len, num_features):
            d = Path(npy_dir) / "test_results"
            def _load(name):
                p = d / f"{name}.npy"
                try:
                    data = np.load(p)
                    if data.ndim == 2:
                        L = input_len if name == "inputs" else output_len
                        data = data.reshape(-1, L, num_features)
                    return data
                except Exception:
                    L = input_len if name == "inputs" else output_len
                    data = np.memmap(p, mode='r', dtype='float32')
                    B = len(data) // (L * num_features)
                    return data.reshape(B, L, num_features)
            return _load("inputs"), _load("prediction"), _load("targets")

        inp_npy, raw_preds_npy, _ = _load_preds_npy(args.before_npy, input_len, output_len, num_features)
        _, dpr_preds_npy, tgt_npy = _load_preds_npy(args.after_npy, input_len, output_len, num_features)
        best_f = 0
        raw_preds = raw_preds_npy
        dpr_preds = dpr_preds_npy
        raw_input_data = inp_npy

        mae_b = np.mean(np.abs(raw_preds[:, :, best_f] - tgt_npy[:, :, best_f]), axis=1)
        mae_a = np.mean(np.abs(dpr_preds[:, :, best_f] - tgt_npy[:, :, best_f]), axis=1)
        improvement = (mae_b - mae_a) / (mae_b + 1e-9)

        if args.indices:
            sel_indices = args.indices
        else:
            top = np.argsort(improvement)[-min(2, len(improvement)):]
            sel_indices = top.tolist()

        if len(sel_indices) >= 2:
            best_b, best_b2 = int(sel_indices[0]), int(sel_indices[1])
        else:
            best_b = best_b2 = int(sel_indices[0])

        raw_pred_sel = raw_preds[best_b, :, best_f]
        raw_pred2 = raw_preds[best_b2, :, best_f] if best_b2 != best_b else raw_pred_sel
        dpr_pred_sel = dpr_preds[best_b, :, best_f]
        dpr_pred2 = dpr_preds[best_b2, :, best_f] if best_b2 != best_b else dpr_pred_sel
        output_gt_sel = tgt_npy[best_b, :, best_f]
        output_gt2 = tgt_npy[best_b2, :, best_f] if best_b2 != best_b else output_gt_sel
        input_seq_sel = raw_input_data[best_b, :, best_f]
        input_seq2 = raw_input_data[best_b2, :, best_f] if best_b2 != best_b else input_seq_sel
    else:
        print("Running RAW inference (DPR bypassed)...")
        all_raw_preds = []
        with dpr_bypass_context(dpr_model):
            for batch in dataloader:
                inputs = batch["inputs"].to(device)
                with torch.no_grad():
                    output = dpr_model(inputs)
                pred = output["prediction"].cpu().numpy()
                all_raw_preds.append(pred)
        raw_preds = np.concatenate(all_raw_preds, axis=0)
        raw_pred_sel = raw_preds[best_b, :, best_f]

    remaining = np.setdiff1d(np.arange(routing_timestep.shape[0]), [best_b])
    if len(remaining) > 0:
        best_b2 = np.random.choice(remaining)
    else:
        best_b2 = best_b
    input_seq2 = ts_batch[best_b2, :, best_f]
    output_gt2 = tgts_batch[best_b2, :, best_f] if tgts_batch is not None else np.zeros(output_len)
    dpr_pred2 = dpr_preds[best_b2, :, best_f]
    raw_pred2 = raw_preds[best_b2, :, best_f] if raw_preds is not None else None

    raw_min, raw_max = raw_sel.min(), raw_sel.max()
    if raw_max - raw_min > 1e-6:
        raw_norm = (raw_sel - raw_min) / (raw_max - raw_min)
    else:
        raw_norm = raw_sel

    prefix_map = {
        "Sunspots": "sunspots", "Illness": "ili", "ExchangeRate": "exchange",
    }
    prefix = prefix_map.get(args.dataset, args.dataset.lower())
    save_path = os.path.join(args.output, f"{prefix}_dpr_routing")

    visualize_combined(
        raw_norm,
        input_seq_sel,
        output_gt_sel,
        raw_pred_sel,
        dpr_pred_sel,
        routing_vis,
        save_path,
        figsize=(args.fig_w, args.fig_h),
        num_patterns=K,
        output_len=output_len,
        input_len_val=input_len,
        sample2_input=input_seq2,
        sample2_gt=output_gt2,
        sample2_raw=raw_pred2,
        sample2_dpr=dpr_pred2,
    )

    print(f"\n=== Visualization Complete ===")
    print(f"Output: {save_path}.pdf")


if __name__ == "__main__":
    main()
