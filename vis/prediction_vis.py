import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse

# ==========================================
# TCG Visualization Script
# ==========================================
# 用于可视化 TCG 模型的预测效果
#
# 推荐数据集/模型组合（基于 tcg_result.md 分析）：
#   1. TimesNet + Illness (ILI) - 季节性模式明显，提升26.8%
#   2. TimeMixer + Weather - 稳定提升8-9%，多特征便于展示
#   3. Informer + ETTm2/ETTh2 - 巨大提升(48-61%)但波动较大
#
# 示例用法：
#   # 对比模式：RAW vs TCG（随机筛选 >30% 提升的样本）
#   python prediction_vis.py \
#     --before ../checkpoints_old/Informer/ExchangeRate_100_96_96/1814d14edc8c6ef91fb05b73b0b47a82 \
#     --after ../checkpoints_old/Informer/ExchangeRate_100_96_96/a6a6366d9c46c0535fd10bfe825b747b \
#     --feat 0 --threshold 30
#
#   # 对比模式：指定样本（固定样本 1036 757 1041 711）
#   python prediction_vis.py \
#     --before ../checkpoints_old/Informer/ExchangeRate_100_96_96/1814d14edc8c6ef91fb05b73b0b47a82 \
#     --after ../checkpoints_old/Informer/ExchangeRate_100_96_96/a6a6366d9c46c0535fd10bfe825b747b \
#     --feat 0 --indices 1010 1210 1041 711
#
#   # 单模型模式：可视化 TCG 预测 vs 真值
#   python prediction_vis.py --tcg ../checkpoints/TimesNetForForecasting/ETTm1_100_96_720/xxx
#
# ==========================================
# 1. 学术风格全局配置
# ==========================================
def set_academic_style():
    """设置ICML学术风格（统一格式）"""
    sns.set_theme(style="whitegrid", font_scale=1.0, rc={
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "axes.edgecolor": ".15",
        "grid.linestyle": "--",
        "axes.linewidth": 1.2,
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 8.5,
        "legend.title_fontsize": 10,
        "figure.dpi": 300,
        "savefig.bbox": "tight"
    })


def subplot_legend_kw(ncol=1, loc="upper left"):
    """子图图例：单列竖排，风格与 parameter_sensitivity_analysis.py 一致。"""
    return {
        "loc": loc,
        "ncol": ncol,
        "framealpha": 0.9,
        "edgecolor": "gray",
        "fontsize": 8.5,
    }


# ==========================================
# 2. 数据加载逻辑
# ==========================================
def load_test_results(result_dir):
    """加载测试结果数据"""
    result_dir = Path(result_dir)
    test_results_dir = result_dir / "test_results"
    
    cfg_file = result_dir / "cfg.json"
    input_len, output_len, num_features = 96, 96, 7
    model_name = "Unknown"
    tcg_enabled = "False"
    num_patterns = "N/A"
    
    if cfg_file.exists():
        with open(cfg_file, 'r') as f:
            cfg = json.load(f)
            model_cfg = cfg.get("model_config", {})
            input_len = model_cfg.get("input_len", 96)
            output_len = model_cfg.get("output_len", 96)
            num_features = model_cfg.get("num_features", 7)
            model_name = cfg.get("model", {}).get("name", "Unknown")
            tcg_cfg = model_cfg.get("tcg", {})
            tcg_params = tcg_cfg.get("params", {})
            tcg_enabled = tcg_params.get("enabled", "False")
            num_patterns = tcg_params.get("num_patterns", "N/A")
    
    def load_npy(name):
        path = test_results_dir / f"{name}.npy"
        try:
            data = np.load(path)
            # 确保形状正确 (B, L, C)
            if len(data.shape) == 2: # 某些情况可能是平铺的
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
        'tcg_enabled': tcg_enabled,
        'num_patterns': num_patterns,
        'input_len': input_len,
        'output_len': output_len,
        'num_features': num_features
    }

# ==========================================
# 3. 核心绘图函数：TCG 预测可视化
# ==========================================
def plot_tcg_predictions(tc_dir, output_path, feat_idx=0, num_samples=4, 
                        selection='high_error', show_error_band=True):
    """
    可视化 TCG 模型的预测效果
    
    参数:
        tc_dir: TCG checkpoint 目录
        output_path: 输出路径
        feat_idx: 要可视化的特征索引
        num_samples: 展示的样本数量
        selection: 样本选择策略
            - 'high_error': 选择误差最大的样本
            - 'low_error': 选择误差最小的样本
            - 'random': 随机选择
            - 'diverse': 选择误差分散的样本
        show_error_band: 是否显示预测误差带
    """
    set_academic_style()
    
    print(f"Loading data from: {tc_dir}")
    inputs, predictions, targets, info = load_test_results(tc_dir)
    
    print(f"Model: {info['model_name']}")
    print(f"TCG enabled: {info['tcg_enabled']}, num_patterns: {info['num_patterns']}")
    print(f"Data shape: inputs={inputs.shape}, predictions={predictions.shape}, targets={targets.shape}")
    
    # 计算每个样本的 MAE
    mae_per_sample = np.mean(np.abs(predictions[:, :, feat_idx] - targets[:, :, feat_idx]), axis=1)
    
    # 根据选择策略选择样本
    if selection == 'high_error':
        indices = np.argsort(mae_per_sample)[-num_samples:]
    elif selection == 'low_error':
        indices = np.argsort(mae_per_sample)[:num_samples]
    elif selection == 'random':
        indices = np.random.choice(len(mae_per_sample), num_samples, replace=False)
    elif selection == 'diverse':
        # 选择误差分散的样本：低、中、高、超高
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
    
    # 配色方案
    colors = {
        'input': '#7F8C8D',      # 灰色 - 输入
        'gt': '#27AE60',         # 绿色 - 真值
        'pred': '#1F77B4',       # 蓝色 - 预测
        'error': '#E74C3C',     # 红色 - 误差带
        'vline': '#34495E'       # 深灰 - 分界线
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
        
        # 计算该样本的误差
        sample_mae = mae_per_sample[idx]
        
        # 绘制输入序列
        ax.plot(t_in, seq_in, color=colors['input'], label='Input', lw=1.5, alpha=0.6)
        
        # 绘制真值
        ax.plot(t_out, seq_gt, color=colors['gt'], label='GT', lw=2.0)
        
        # 绘制预测
        ax.plot(t_out, seq_pred, color=colors['pred'], label='Prediction', lw=2.0, ls='-')
        
        # 显示误差带（可选）
        if show_error_band:
            abs_error = np.abs(seq_gt - seq_pred)
            # 使用半透明填充显示误差范围
            ax.fill_between(t_out, seq_pred - abs_error, seq_pred + abs_error, 
                           color=colors['error'], alpha=0.2, label='Error Band')
        
        # 绘制输入/输出分界线
        ax.axvline(x=L_in-1, color=colors['vline'], linestyle=':', lw=1.2)
        
        # 标题
        title = f"({chr(97+i)}) Sample #{idx} | MAE: {sample_mae:.4f}"
        ax.set_title(title, fontweight='bold', fontsize=11, loc='center')
        ax.grid(True, linestyle='--', alpha=0.25)
        
        ax.set_xlabel("Time Steps")
        if i == 0:
            ax.set_ylabel("Normalized Value")
        
        # 设置x轴范围
        ax.set_xlim(0, L_in + L_out)

        leg_loc = "lower left" if i in (1, 3) else "upper left"
        ax.legend(**subplot_legend_kw(loc=leg_loc))

    # 添加模型信息标题
    model_info_text = f"Model: {info['model_name']}"
    if info['tcg_enabled'].lower() == 'true':
        model_info_text += f" | TCG (K={info['num_patterns']})"
    fig.suptitle(model_info_text, fontsize=10, y=1.02, style='italic')

    out_dir = Path(output_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    filename = f'tcg_prediction_{info["model_name"]}_feat{feat_idx}.pdf'
    plt.savefig(out_dir / filename, bbox_inches='tight', dpi=300)
    print(f"Success! Visualized samples: {indices}")
    print(f"Results saved to: {out_dir / filename}")
    
    return info

# ==========================================
# 4. 误差分析可视化
# ==========================================
def plot_error_analysis(tc_dir, output_path, feat_idx=0, num_bins=50):
    """分析预测误差的分布"""
    set_academic_style()
    
    print(f"Loading data for error analysis: {tc_dir}")
    inputs, predictions, targets, info = load_test_results(tc_dir)
    
    # 计算每个样本的 MAE 和 MSE
    mae_per_sample = np.mean(np.abs(predictions[:, :, feat_idx] - targets[:, :, feat_idx]), axis=1)
    mse_per_sample = np.mean((predictions[:, :, feat_idx] - targets[:, :, feat_idx])**2, axis=1)
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), constrained_layout=True)
    
    # 1. MAE 分布直方图
    ax1 = axes[0]
    ax1.hist(mae_per_sample, bins=num_bins, color='steelblue', edgecolor='navy', alpha=0.7)
    ax1.axvline(mae_per_sample.mean(), color='red', linestyle='--', lw=2, 
                label=f'Mean: {mae_per_sample.mean():.4f}')
    ax1.set_xlabel('MAE')
    ax1.set_ylabel('Count')
    ax1.set_title('MAE Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. MSE 分布直方图
    ax2 = axes[1]
    ax2.hist(mse_per_sample, bins=num_bins, color='coral', edgecolor='darkred', alpha=0.7)
    ax2.axvline(mse_per_sample.mean(), color='red', linestyle='--', lw=2,
                label=f'Mean: {mse_per_sample.mean():.4f}')
    ax2.set_xlabel('MSE')
    ax2.set_ylabel('Count')
    ax2.set_title('MSE Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 时间步级别误差热力图
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
    
    fig.suptitle(f"Error Analysis: {info['model_name']}", fontsize=12, fontweight='bold')
    
    out_dir = Path(output_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    filename = f'tcg_error_analysis_{info["model_name"]}_feat{feat_idx}.pdf'
    plt.savefig(out_dir / filename, bbox_inches='tight', dpi=300)
    print(f"Error analysis saved to: {out_dir / filename}")

# ==========================================
# 5. 多特征对比可视化
# ==========================================
def plot_multifeature_comparison(tc_dir, output_path, num_features_shown=4):
    """可视化多个特征的预测效果"""
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
    
    # 选择一个典型样本（误差适中的）
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
        ax.set_title(f"Feature {feat_idx} | MAE: {feat_mae:.4f}", fontsize=10)
        ax.grid(True, alpha=0.25)
        
        ax.set_xlabel("Time Steps")
        if i == 0:
            ax.set_ylabel("Value")

        leg_loc = "lower left" if i in (1, 3) else "upper left"
        ax.legend(**subplot_legend_kw(loc=leg_loc))
    
    fig.suptitle(f"Multi-Feature Prediction: {info['model_name']}", fontsize=11, fontweight='bold')
    
    out_dir = Path(output_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = f'tcg_multifeature_{info["model_name"]}.pdf'
    plt.savefig(out_dir / filename, bbox_inches='tight', dpi=300)
    print(f"Multi-feature visualization saved to: {out_dir / filename}")

# ==========================================
# 6. RAW vs TCG 对比可视化
# ==========================================
def plot_comparison(before_dir, after_dir, output_path, feat_idx=0, threshold=30, fixed_indices=None):
    """对比 RAW (before) 和 TCG (after) 的预测效果
    
    筛选策略：选择 TCG 相比 RAW 提升最大的样本
    """
    set_academic_style()
    
    print(f"Loading RAW from: {before_dir}")
    print(f"Loading TCG from: {after_dir}")
    
    in_b, pred_b, tar_b, info_b = load_test_results(before_dir)
    in_a, pred_a, tar_a, info_a = load_test_results(after_dir)
    
    print(f"RAW: {info_b['model_name']}, TCG: {info_a['model_name']}")
    print(f"RAW shape: inputs={in_b.shape}, pred={pred_b.shape}, tar={tar_b.shape}")
    print(f"TCG shape: inputs={in_a.shape}, pred={pred_a.shape}, tar={tar_a.shape}")
    
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
        ax.plot(t_out, seq_ours, color=colors['ours'], label='TCM', lw=2.0, ls='-')
        
        ax.axvline(x=L_in-1, color='#34495E', linestyle=':', lw=1.2)
        imp_pct = improvement[idx] * 100
        ax.set_title(f"({chr(97+i)}) Sample #{idx} | Imp: {imp_pct:+.1f}%", 
                    fontweight='bold', fontsize=11, loc='center')
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
    parser = argparse.ArgumentParser(description='TCG Visualization Tool')
    
    # 对比模式
    parser.add_argument('--before', type=str, help='RAW/Baseline checkpoint directory')
    parser.add_argument('--after', type=str, help='TCG checkpoint directory')
    
    # 单模型模式
    parser.add_argument('--tcg', type=str, help='TCG checkpoint directory (for single model mode)')
    
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
    elif args.tcg:
        plot_tcg_predictions(args.tcg, args.output, args.feat, args.samples, args.selection, 
                            args.show_error_band)
    else:
        parser.print_help()
        print("\n请使用以下方式之一:")
        print("  1. 对比模式: --before <raw_dir> --after <tcg_dir>")
        print("  2. 单模型模式: --tcg <tcg_dir>")