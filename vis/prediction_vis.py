import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse

# ==========================================
# DPR Visualization Script
# ==========================================
# Visualize DPR model forecasts vs ground truth (and RAW vs DPR comparisons).
#
# Example dataset / model pairs (see dpr_result.md):
#   1. TimesNet + Illness — strong seasonality, large gains
#   2. TimeMixer + Weather — stable gains, many features for plots
#   3. Informer + ETTm2 / ETTh2 — large gains but noisier series
#
# Examples:
#   # Comparison: RAW vs DPR (random samples with >30% improvement)
#   python prediction_vis.py \
#     --before ../checkpoints_old/Informer/ExchangeRate_100_96_96/1814d14edc8c6ef91fb05b73b0b47a82 \
#     --after ../checkpoints_old/Informer/ExchangeRate_100_96_96/a6a6366d9c46c0535fd10bfe825b747b \
#     --feat 0 --threshold 30
#
#   # Comparison: fixed sample indices
#   python prediction_vis.py \
#     --before ../checkpoints_old/Informer/ExchangeRate_100_96_96/1814d14edc8c6ef91fb05b73b0b47a82 \
#     --after ../checkpoints_old/Informer/ExchangeRate_100_96_96/a6a6366d9c46c0535fd10bfe825b747b \
#     --feat 0 --indices 1010 1210 1041 711
#
#   # Single run: DPR predictions vs targets
#   python prediction_vis.py --dpr ../checkpoints/TimesNetForForecasting/ETTm1_100_96_720/xxx
#
# ==========================================
# 1. Global plot style (matches run_rq5_visualization.py)
# ==========================================
def set_academic_style():
    """Font family and sizes aligned with run_rq5_visualization.py."""
    sns.set_theme(style="whitegrid", font_scale=1.0, rc={
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 11,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "legend.title_fontsize": 11,
        "figure.titlesize": 12,
        "axes.edgecolor": ".15",
        "grid.linestyle": "--",
        "axes.linewidth": 1.2,
        "figure.dpi": 300,
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "savefig.bbox": "tight",
    })


def subplot_legend_kw(ncol=1, loc="upper left"):
    """Legend kwargs for subplots; font size comes from rcParams['legend.fontsize']."""
    return {
        "loc": loc,
        "ncol": ncol,
        "framealpha": 0.9,
        "edgecolor": "gray",
    }


# ==========================================
# 2. Load saved test tensors
# ==========================================
def load_test_results(result_dir):
    """Load inputs, prediction, and targets from a checkpoint result folder."""
    result_dir = Path(result_dir)
    test_results_dir = result_dir / "test_results"
    
    cfg_file = result_dir / "cfg.json"
    input_len, output_len, num_features = 96, 96, 7
    model_name = "Unknown"
    dpr_enabled = "False"
    num_patterns = "N/A"
    
    if cfg_file.exists():
        with open(cfg_file, 'r') as f:
            cfg = json.load(f)
            model_cfg = cfg.get("model_config", {})
            input_len = model_cfg.get("input_len", 96)
            output_len = model_cfg.get("output_len", 96)
            num_features = model_cfg.get("num_features", 7)
            model_name = cfg.get("model", {}).get("name", "Unknown")
            dpr_cfg = model_cfg.get("dpr", {})
            dpr_params = dpr_cfg.get("params", {})
            dpr_enabled = dpr_params.get("enabled", "False")
            num_patterns = dpr_params.get("num_patterns", "N/A")
    
    def load_npy(name):
        path = test_results_dir / f"{name}.npy"
        try:
            data = np.load(path)
            # Ensure shape (B, L, C)
            if len(data.shape) == 2:  # Some exports are flattened
                L = input_len if name == "inputs" else output_len
                B = len(data) // (L * num_features)
                data = data.reshape(B, L, num_features)
            return data
        except:
            data_mm = np.memmap(path, mode='r', dtype='float32')
            L = input_len if name == "inputs" else output_len
            B = len(data_mm) // (L * num_features)
            return data_mm.reshape(B, L, num_features)

    return load_npy("inputs"), load_npy("prediction"), load_npy("targets"), {
        'model_name': model_name,
        'dpr_enabled': dpr_enabled,
        'num_patterns': num_patterns,
        'input_len': input_len,
        'output_len': output_len,
        'num_features': num_features
    }

# ==========================================
# 3. Plot DPR predictions
# ==========================================
def plot_dpr_predictions(tc_dir, output_path, feat_idx=0, num_samples=4, 
                        selection='high_error', show_error_band=True):
    """
    Plot DPR forecasts for selected test samples.

    Args:
        tc_dir: Checkpoint directory with test_results/*.npy
        output_path: Directory for PDF output
        feat_idx: Feature channel index
        num_samples: Number of samples to show
        selection: How to pick samples:
            - 'high_error': highest per-sample MAE
            - 'low_error': lowest MAE
            - 'random': random
            - 'diverse': spread across low / mid / high MAE
        show_error_band: If True, shade |y - y_hat| around the prediction
    """
    set_academic_style()
    
    print(f"Loading data from: {tc_dir}")
    inputs, predictions, targets, info = load_test_results(tc_dir)
    
    print(f"Model: {info['model_name']}")
    print(f"DPR enabled: {info['dpr_enabled']}, num_patterns: {info['num_patterns']}")
    print(f"Data shape: inputs={inputs.shape}, predictions={predictions.shape}, targets={targets.shape}")
    
    mae_per_sample = np.mean(np.abs(predictions[:, :, feat_idx] - targets[:, :, feat_idx]), axis=1)

    if selection == 'high_error':
        indices = np.argsort(mae_per_sample)[-num_samples:]
    elif selection == 'low_error':
        indices = np.argsort(mae_per_sample)[:num_samples]
    elif selection == 'random':
        indices = np.random.choice(len(mae_per_sample), num_samples, replace=False)
    elif selection == 'diverse':
        # Pick low / mid / high / max MAE samples
        sorted_idx = np.argsort(mae_per_sample)
        n = len(sorted_idx)
        if n >= 4:
            indices = [sorted_idx[0], sorted_idx[n//3], sorted_idx[2*n//3], sorted_idx[-1]]
        else:
            indices = sorted_idx
    else:
        indices = np.arange(min(num_samples, len(mae_per_sample)))
    
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.8), sharex=True, constrained_layout=True)
    axes = axes.flatten()
    
    colors = {
        'input': '#7F8C8D',      # input
        'gt': '#27AE60',         # ground truth
        'pred': '#1F77B4',       # prediction
        'error': '#E74C3C',      # error band
        'vline': '#34495E'       # input/output split
    }
    
    L_in = info['input_len']
    L_out = info['output_len']
    t_in = np.arange(L_in)
    t_out = np.arange(L_in, L_in + L_out)

    for i, idx in enumerate(indices):
        ax = axes[i]
        seq_in = inputs[idx, :, feat_idx]
        seq_gt = targets[idx, :, feat_idx]
        seq_pred = predictions[idx, :, feat_idx]
        
        sample_mae = mae_per_sample[idx]

        ax.plot(t_in, seq_in, color=colors['input'], label='Input', lw=1.5, alpha=0.6)
        ax.plot(t_out, seq_gt, color=colors['gt'], label='GT', lw=2.0)
        ax.plot(t_out, seq_pred, color=colors['pred'], label='Prediction', lw=2.0, ls='-')

        if show_error_band:
            abs_error = np.abs(seq_gt - seq_pred)
            ax.fill_between(t_out, seq_pred - abs_error, seq_pred + abs_error, 
                           color=colors['error'], alpha=0.2, label='Error Band')

        ax.axvline(x=L_in-1, color=colors['vline'], linestyle=':', lw=1.2)

        title = f"({chr(97+i)}) Sample #{idx} | MAE: {sample_mae:.4f}"
        ax.set_title(title, loc="center")
        ax.grid(True, linestyle='--', alpha=0.25)
        
        ax.set_xlabel("Time Steps")
        if i == 0:
            ax.set_ylabel("Normalized Value")
        
        ax.set_xlim(0, L_in + L_out)

        leg_loc = "lower left" if i in (1, 3) else "upper left"
        ax.legend(**subplot_legend_kw(loc=leg_loc))

    model_info_text = f"Model: {info['model_name']}"
    if info['dpr_enabled'].lower() == 'true':
        model_info_text += f" | DPR (K={info['num_patterns']})"
    fig.suptitle(model_info_text, y=1.02, style="italic")

    out_dir = Path(output_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    filename = f'dpr_prediction_{info["model_name"]}_feat{feat_idx}.pdf'
    plt.savefig(out_dir / filename, bbox_inches='tight', dpi=300)
    print(f"Success! Visualized samples: {indices}")
    print(f"Results saved to: {out_dir / filename}")
    
    return info

# ==========================================
# 4. Error analysis plots
# ==========================================
def plot_error_analysis(tc_dir, output_path, feat_idx=0, num_bins=50):
    """Histograms and per-timestep MAE for one feature."""
    set_academic_style()
    
    print(f"Loading data for error analysis: {tc_dir}")
    inputs, predictions, targets, info = load_test_results(tc_dir)
    
    mae_per_sample = np.mean(np.abs(predictions[:, :, feat_idx] - targets[:, :, feat_idx]), axis=1)
    mse_per_sample = np.mean((predictions[:, :, feat_idx] - targets[:, :, feat_idx])**2, axis=1)
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), constrained_layout=True)
    
    ax1 = axes[0]
    ax1.hist(mae_per_sample, bins=num_bins, color='steelblue', edgecolor='navy', alpha=0.7)
    ax1.axvline(mae_per_sample.mean(), color='red', linestyle='--', lw=2, 
                label=f'Mean: {mae_per_sample.mean():.4f}')
    ax1.set_xlabel('MAE')
    ax1.set_ylabel('Count')
    ax1.set_title('MAE Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2 = axes[1]
    ax2.hist(mse_per_sample, bins=num_bins, color='coral', edgecolor='darkred', alpha=0.7)
    ax2.axvline(mse_per_sample.mean(), color='red', linestyle='--', lw=2,
                label=f'Mean: {mse_per_sample.mean():.4f}')
    ax2.set_xlabel('MSE')
    ax2.set_ylabel('Count')
    ax2.set_title('MSE Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    ax3 = axes[2]
    abs_errors = np.abs(predictions[:, :, feat_idx] - targets[:, :, feat_idx])
    mean_error_per_timestep = np.mean(abs_errors, axis=0)
    ax3.plot(range(len(mean_error_per_timestep)), mean_error_per_timestep, 
             color='seagreen', lw=1.5)
    ax3.fill_between(range(len(mean_error_per_timestep)), 0, mean_error_per_timestep,
                     alpha=0.3, color='seagreen')
    ax3.axvline(info['input_len']-1, color='gray', linestyle=':', lw=1.5, 
                label='Input/Output boundary')
    ax3.set_xlabel('Time Step (relative to output)')
    ax3.set_ylabel('Mean Absolute Error')
    ax3.set_title('Error per Timestep')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    fig.suptitle(f"Error Analysis: {info['model_name']}", fontweight="bold")
    
    out_dir = Path(output_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    filename = f'dpr_error_analysis_{info["model_name"]}_feat{feat_idx}.pdf'
    plt.savefig(out_dir / filename, bbox_inches='tight', dpi=300)
    print(f"Error analysis saved to: {out_dir / filename}")

# ==========================================
# 5. Multi-feature comparison (one median-MAE sample)
# ==========================================
def plot_multifeature_comparison(tc_dir, output_path, num_features_shown=4):
    """Plot several channels for the sample with median overall MAE."""
    set_academic_style()
    
    print(f"Loading data: {tc_dir}")
    inputs, predictions, targets, info = load_test_results(tc_dir)
    
    num_total_features = min(info['num_features'], num_features_shown)
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.8), sharex=True, constrained_layout=True)
    axes = axes.flatten()
    
    colors = {'input': '#7F8C8D', 'gt': '#27AE60', 'pred': '#1F77B4'}
    L_in = info['input_len']
    t_in = np.arange(L_in)
    t_out = np.arange(L_in, L_in + info['output_len'])
    
    mae_per_sample = np.mean(np.abs(predictions - targets), axis=(1, 2))
    mid_idx = np.argsort(mae_per_sample)[len(mae_per_sample)//2]
    
    for i in range(4):
        ax = axes[i]
        feat_idx = i % num_total_features
        
        seq_in = inputs[mid_idx, :, feat_idx]
        seq_gt = targets[mid_idx, :, feat_idx]
        seq_pred = predictions[mid_idx, :, feat_idx]
        
        ax.plot(t_in, seq_in, color=colors['input'], label='Input', lw=1.2, alpha=0.5)
        ax.plot(t_out, seq_gt, color=colors['gt'], label='GT', lw=1.5)
        ax.plot(t_out, seq_pred, color=colors['pred'], label='Pred', lw=1.5, ls='--')
        ax.axvline(L_in-1, color='#34495E', linestyle=':', lw=1.0)
        
        feat_mae = np.mean(np.abs(seq_gt - seq_pred))
        ax.set_title(f"Feature {feat_idx} | MAE: {feat_mae:.4f}")
        ax.grid(True, alpha=0.25)
        
        ax.set_xlabel("Time Steps")
        if i == 0:
            ax.set_ylabel("Value")

        leg_loc = "lower left" if i in (1, 3) else "upper left"
        ax.legend(**subplot_legend_kw(loc=leg_loc))
    
    fig.suptitle(f"Multi-Feature Prediction: {info['model_name']}", fontweight="bold")
    
    out_dir = Path(output_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = f'dpr_multifeature_{info["model_name"]}.pdf'
    plt.savefig(out_dir / filename, bbox_inches='tight', dpi=300)
    print(f"Multi-feature visualization saved to: {out_dir / filename}")

# ==========================================
# 6. RAW vs DPR comparison
# ==========================================
def plot_comparison(before_dir, after_dir, output_path, feat_idx=0, threshold=30, fixed_indices=None):
    """Compare baseline (before) vs DPR (after); sample by relative MAE improvement."""
    set_academic_style()
    
    print(f"Loading RAW from: {before_dir}")
    print(f"Loading DPR from: {after_dir}")
    
    in_b, pred_b, tar_b, info_b = load_test_results(before_dir)
    in_a, pred_a, tar_a, info_a = load_test_results(after_dir)
    
    print(f"RAW: {info_b['model_name']}, DPR: {info_a['model_name']}")
    print(f"RAW shape: inputs={in_b.shape}, pred={pred_b.shape}, tar={tar_b.shape}")
    print(f"DPR shape: inputs={in_a.shape}, pred={pred_a.shape}, tar={tar_a.shape}")
    
    mae_b = np.mean(np.abs(pred_b[:, :, feat_idx] - tar_b[:, :, feat_idx]), axis=1)
    mae_a = np.mean(np.abs(pred_a[:, :, feat_idx] - tar_a[:, :, feat_idx]), axis=1)
    
    improvement = (mae_b - mae_a) / (mae_b + 1e-9)
    
    if fixed_indices is not None:
        indices = np.array(fixed_indices)
        print(f"Using fixed indices: {indices.tolist()}")
        print(f"Improvements: {[f'{improvement[i]*100:.1f}%' for i in indices]}")
    else:
        threshold_frac = threshold / 100.0
        qualified = np.where(improvement > threshold_frac)[0]
        if len(qualified) < 4:
            print(f"Warning: Only {len(qualified)} samples with improvement > {threshold:.0f}%. Using top 4.")
            indices = qualified[np.argsort(improvement[qualified])[-4:]]
        else:
            indices = np.random.choice(qualified, 4, replace=False)
    
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.8), sharex=True, constrained_layout=True)
    axes = axes.flatten()
    
    colors = {'in': '#7F8C8D', 'gt': '#27AE60', 'base': '#D62728', 'ours': '#1F77B4'}
    L_in, L_out = in_b.shape[1], tar_b.shape[1]
    t_in = np.arange(L_in)
    t_out = np.arange(L_in, L_in + L_out)

    for i, idx in enumerate(indices):
        ax = axes[i]
        seq_in = in_b[idx, :, feat_idx]
        seq_gt = tar_b[idx, :, feat_idx]
        seq_base = pred_b[idx, :, feat_idx]
        seq_ours = pred_a[idx, :, feat_idx]
        
        ax.plot(t_in, seq_in, color=colors['in'], label='Input', lw=1.5, alpha=0.5)
        ax.plot(t_out, seq_gt, color=colors['gt'], label='GT', lw=2.0)
        ax.plot(t_out, seq_base, color=colors['base'], label='Baseline', lw=2.0, ls='--')
        ax.plot(t_out, seq_ours, color=colors['ours'], label='DPR', lw=2.0, ls='-')
        
        ax.axvline(x=L_in-1, color='#34495E', linestyle=':', lw=1.2)
        imp_pct = improvement[idx] * 100
        ax.set_title(
            f"({chr(97+i)}) Sample #{idx} | Imp: {imp_pct:+.1f}%",
            loc="center",
        )
        ax.grid(True, linestyle='--', alpha=0.25)
        
        ax.set_xlabel("Time Steps")
        if i == 0:
            ax.set_ylabel("Normalized Value")

        leg_loc = "lower left" if i in (1, 3) else "upper left"
        ax.legend(**subplot_legend_kw(loc=leg_loc))

    out_dir = Path(output_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = 'prediction_vis.pdf'
    plt.savefig(out_dir / filename, bbox_inches='tight', dpi=300)
    print(f"Saved comparison to: {out_dir / filename}")
    print(f"Samples: {indices.tolist()}, Improvements: {[f'{x:.1f}' for x in improvement[indices]*100]}")

# ==========================================
# Main
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DPR Visualization Tool')
    
    parser.add_argument('--before', type=str, help='RAW/Baseline checkpoint directory')
    parser.add_argument('--after', type=str, help='DPR checkpoint directory')

    parser.add_argument('--dpr', type=str, help='DPR checkpoint directory (single-model mode)')
    
    parser.add_argument('--output', type=str, default='./', help='Output directory')
    parser.add_argument('--feat', type=int, default=0, help='Feature index to visualize')
    parser.add_argument('--samples', type=int, default=4, help='Number of samples to show')
    parser.add_argument('--selection', type=str, default='diverse',
                       choices=['high_error', 'low_error', 'random', 'diverse'],
                       help='Sample selection strategy')
    parser.add_argument('--threshold', type=float, default=30,
                       help='Min improvement threshold for comparison mode (%%)')
    parser.add_argument('--indices', type=int, nargs='+', default=None,
                       help='Fixed sample indices to visualize (e.g., --indices 1036 757 1041 711)')
    parser.add_argument('--show_error_band', action='store_true', default=True,
                       help='Show prediction error band')
    parser.add_argument('--no_error_band', action='store_false', dest='show_error_band',
                       help='Hide prediction error band')
    
    args = parser.parse_args()
    
    if args.before and args.after:
        plot_comparison(args.before, args.after, args.output, args.feat, args.threshold, args.indices)
    elif args.dpr:
        plot_dpr_predictions(args.dpr, args.output, args.feat, args.samples, args.selection, 
                            args.show_error_band)
    else:
        parser.print_help()
        print("\nUsage:")
        print("  1. Compare: --before <raw_dir> --after <dpr_dir>")
        print("  2. Single model: --dpr <dpr_dir>")