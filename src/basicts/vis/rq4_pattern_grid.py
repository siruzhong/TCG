"""Visualization utilities for RQ4 pattern analysis - DPR Interpretability."""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional

COLORS = {
    "raw_input": "dimgray",
    "regime_shift": "lightcoral",
    "gate": "royalblue",
    "heatmap": "YlGnBu",
    "grid": "#cccccc",
}


def patch_to_timestep_range(patch_idx: int, patch_len: int = 16, stride: int = 8, input_len: int = 96) -> Tuple[int, int]:
    start = patch_idx * stride
    end = min(start + patch_len, input_len)
    return start, end


def aggregate_routing_to_timestep(
    routing: np.ndarray,
    input_len: int,
    patch_len: int = 16,
    stride: int = 8,
    method: str = "weighted"
) -> np.ndarray:
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


def extract_top_segments(
    routing_timestep: np.ndarray,
    time_series: np.ndarray,
    N: int = 3
) -> List[Dict]:
    b, n, l, k = routing_timestep.shape
    segments = []
    for batch_idx in range(min(b, 10)):
        for feat_idx in range(min(n, 2)):
            routing_slice = routing_timestep[batch_idx, feat_idx]
            dominant_pattern = np.argmax(routing_slice, axis=-1)
            pattern_changes = np.diff(dominant_pattern)
            change_points = np.where(np.abs(pattern_changes) > 0)[0] + 1
            if len(change_points) == 0:
                continue
            change_points = np.concatenate([[0], change_points, [l]])
            for i in range(len(change_points) - 1):
                start, end = change_points[i], change_points[i+1]
                if end - start < 10:
                    continue
                seg_routing = routing_slice[start:end]
                seg_ts = time_series[batch_idx, start:end, feat_idx]
                pattern = dominant_pattern[start]
                segments.append({
                    "batch_idx": batch_idx,
                    "feat_idx": feat_idx,
                    "start": start,
                    "end": end,
                    "routing": seg_routing,
                    "time_series": seg_ts,
                    "dominant_pattern": pattern,
                    "variability": np.std(seg_routing[:, pattern]) if end - start > 1 else 0,
                })
    segments.sort(key=lambda x: x["variability"], reverse=True)
    return segments[:N]


def compute_pattern_stats(segments: List[Dict]) -> Dict:
    if not segments:
        return {}
    all_patterns = set()
    for seg in segments:
        all_patterns.add(seg["dominant_pattern"])
    pattern_lengths = {p: [] for p in all_patterns}
    for seg in segments:
        pattern_lengths[seg["dominant_pattern"]].append(seg["end"] - seg["start"])
    stats = {}
    for pattern, lengths in pattern_lengths.items():
        stats[pattern] = {
            "count": len(lengths),
            "avg_length": np.mean(lengths) if lengths else 0,
            "std_length": np.std(lengths) if lengths else 0,
        }
    return stats


def visualize_pattern_segment_grid(
    segments: List[Dict],
    stats: Dict,
    save_path: str,
    dataset_name: str,
    patch_len: int = 16,
    stride: int = 8,
) -> None:
    if not segments:
        return
    n_segs = len(segments)
    fig, axes = plt.subplots(n_segs, 1, figsize=(14, 3 * n_segs), sharex=True)
    if n_segs == 1:
        axes = [axes]
    for idx, seg in enumerate(segments):
        ax = axes[idx]
        seg_len = seg["end"] - seg["start"]
        time_idx = np.arange(seg_len)
        ax.plot(time_idx, seg["time_series"], color=COLORS["raw_input"], linewidth=1.2, label="Input")
        ax.set_ylabel(f"Feature {seg['feat_idx']}", fontsize=10)
        title = f"Segment {idx+1}: Pattern {seg['dominant_pattern']} (len={seg_len})"
        ax.set_title(title, loc="left", fontsize=11)
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(f"{save_path}.pdf", dpi=300, bbox_inches="tight")
    plt.close()


def visualize_activation_timeline(
    time_series: np.ndarray,
    routing_timestep: np.ndarray,
    save_path: str,
    dataset_name: str,
) -> None:
    pass


def visualize_transition_matrix(
    routing_timestep: np.ndarray,
    save_path: str,
    dataset_name: str,
) -> None:
    pass


def visualize_dpr_interpretability(
    raw_input: np.ndarray,
    intensity_gate: np.ndarray,
    routing_probs: np.ndarray,
    save_path: str,
    title: str = "(a) DPR Interpretability: Unsupervised Regime Discovery",
    regime_shift_start: Optional[int] = None,
    regime_shift_end: Optional[int] = None,
    figsize: Tuple[float, float] = (12, 9),
    patch_len: int = 16,
    stride: int = 8,
    normalize_input: bool = True,
) -> None:
    l = raw_input.shape[0]
    k = routing_probs.shape[0]
    time_steps = np.arange(l)
    
    # Normalize input for visualization if needed
    if normalize_input:
        input_min = np.min(raw_input)
        input_max = np.max(raw_input)
        if input_max - input_min > 1e-6:
            raw_input_norm = (raw_input - input_min) / (input_max - input_min)
        else:
            raw_input_norm = raw_input
    else:
        raw_input_norm = raw_input
    
    plt.rcParams["font.family"] = "serif"
    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True, gridspec_kw={"height_ratios": [1.5, 1, 1.8]})
    ax0 = axes[0]
    ax0.plot(time_steps, raw_input_norm, color=COLORS["raw_input"], linewidth=1.5, label="Raw Sequence")
    if regime_shift_start is not None and regime_shift_end is not None:
        ax0.axvspan(regime_shift_start, regime_shift_end, color=COLORS["regime_shift"], alpha=0.2, label="Regime Shift")
    ax0.set_ylabel("Input Value (Normalized)", fontsize=12)
    ax0.set_title(f"{title}", loc="left", fontweight="bold", fontsize=13)
    ax0.legend(loc="upper left", fontsize=10)
    ax0.grid(True, linestyle="--", alpha=0.4)
    ax1 = axes[1]
    ax1.plot(time_steps, intensity_gate, color=COLORS["gate"], linewidth=2)
    ax1.fill_between(time_steps, intensity_gate, 0, color=COLORS["gate"], alpha=0.2)
    ax1.set_ylim(0, 1.05)
    ax1.set_ylabel("Gate $g$", fontsize=12)
    ax1.set_title("(b) Intensity Gate Activation", loc="left", fontweight="bold", fontsize=12)
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax2 = axes[2]
    sns.heatmap(routing_probs, cmap=COLORS["heatmap"], cbar_kws={"label": "Probability"}, ax=ax2, vmin=0, vmax=1)
    ax2.set_ylabel("Pattern ID ($K$)", fontsize=12)
    ax2.set_xlabel("Time Steps", fontsize=12)
    ax2.set_title(r"(c) Routing Probabilities $\mathbf{\pi}$", loc="left", fontweight="bold", fontsize=12)
    xticks = np.arange(0, l + 1, max(1, l // 10))
    ax2.set_xticks(xticks)
    ax2.set_xticklabels(xticks, rotation=0)
    plt.tight_layout()
    plt.savefig(f"{save_path}.pdf", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}.pdf")


def find_best_sample_for_visualization(
    time_series_batch: np.ndarray,
    routing_timestep: np.ndarray,
    intensity_gate: np.ndarray,
    num_features: int = 1,
    prefer_stable_then_spike: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    b, l, n = time_series_batch.shape
    _, _, _, k = routing_timestep.shape
    best_score = -1
    best_idx = 0
    best_feat = 0
    for batch_idx in range(b):
        for feat_idx in range(min(n, num_features)):
            ts = time_series_batch[batch_idx, :, feat_idx]
            ig = intensity_gate[batch_idx, feat_idx]
            rp = routing_timestep[batch_idx, feat_idx]
            
            # Score 1: Gate variation (model responding to dynamics)
            gate_var = np.std(ig)
            
            # Score 2: Gate activation level (mean gate value)
            gate_mean = np.mean(ig)
            
            # Score 3: Pattern switching (regime discovery)
            dominant_pattern = np.argmax(rp, axis=-1)
            pattern_changes = np.sum(np.diff(dominant_pattern) != 0)
            
            # Score 4: Input variability
            input_var = np.std(ts)
            
            # Combined score prioritizing clear gate dynamics and pattern switching
            score = gate_var * 2.0 + gate_mean * 0.5 + pattern_changes * 0.3 + input_var * 0.01
            
            if score > best_score:
                best_score = score
                best_idx = batch_idx
                best_feat = feat_idx
                print(f"  New best: batch={batch_idx}, feat={feat_idx}, score={score:.4f}, gate_var={gate_var:.4f}, gate_mean={gate_mean:.4f}, pattern_changes={pattern_changes}")
    
    print(f"Selected: batch={best_idx}, feat={best_feat}, score={best_score:.4f}")
    return (
        time_series_batch[best_idx, :, best_feat],
        intensity_gate[best_idx, best_feat],
        routing_timestep[best_idx, best_feat],
        best_idx,
    )


def create_demo_visualization(save_dir: str = "./") -> None:
    l, k = 100, 8
    time_steps = np.arange(l)
    raw_input = np.sin(time_steps * 0.1) * 2
    raw_input[50:] += np.random.normal(0, 0.5, 50) + 1.5
    regime_start, regime_end = 50, 65
    gate_g = np.full(l, 0.12) + np.random.normal(0, 0.02, l)
    gate_g[regime_start:regime_end] = 0.85 + np.random.normal(0, 0.05, regime_end - regime_start)
    gate_g[regime_end:] = 0.5 + np.random.normal(0, 0.05, l - regime_end)
    pi_probs = np.random.dirichlet(np.ones(k), size=l)
    pi_probs[:50, 0] += 0.4
    pi_probs[50:, 3] += 0.4
    pi_probs = np.clip(pi_probs, 0, 1)
    pi_probs = pi_probs / pi_probs.sum(axis=1, keepdims=True)
    pi_probs = pi_probs.T
    save_path = f"{save_dir}/DPR_interpretability_demo"
    visualize_dpr_interpretability(
        raw_input,
        gate_g,
        pi_probs,
        save_path,
        regime_shift_start=regime_start,
        regime_shift_end=regime_end,
    )