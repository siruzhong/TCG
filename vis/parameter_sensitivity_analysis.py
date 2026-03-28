import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# 1. 设置 ICML 学术风格（统一格式）
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
})

# 2. 读取数据 - 使用 Informer_old（趋势更明显）
df = pd.read_csv('../checkpoints/Informer_old/table2_all_experiments_detailed.csv')

# 3. 数据筛选与预处理
df_synth = df[df['Dataset'].str.contains('SyntheticTS') & 
              (df['Has_DropoutTS'] == 'Yes') & 
              (df['sparsity_weight'] == 0.00)].copy()

# 提取噪声水平
df_synth['Noise Level'] = df_synth['Dataset'].str.extract(r'noise(\d+\.\d+)').astype(float)

# 创建字符串格式的标签
df_synth['Noise Label'] = df_synth['Noise Level'].apply(lambda x: f'{x:.1f}')
df_synth['Sensitivity Label'] = df_synth['init_sensitivity'].apply(lambda x: f'{x:.1f}')

# 4. 创建画布（适合ICML单栏）
fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.0), constrained_layout=True)

# 定义颜色方案
unique_ws = sorted(df_synth['init_sensitivity'].unique())
unique_noises = sorted(df_synth['Noise Level'].unique())

# 左图：3种sensitivity的颜色
palette_ws = ['#8B4789', '#F4A460', '#4682B4']  # 紫、橙、蓝
# 右图：5种噪声级别的颜色（蓝到绿渐变）
palette_noise = ['#4A5899', '#6B8E99', '#7FA99A', '#A5C4A5', '#C8D9B4']

# ======================== 左图: Impact of w_s across Noise Levels ========================
# 对不同预测长度求平均
df_avg = df_synth.groupby(['Noise Level', 'init_sensitivity']).agg({
    'MSE': ['mean', 'std']
}).reset_index()
df_avg.columns = ['Noise Level', 'init_sensitivity', 'MSE_mean', 'MSE_std']

# 绘制线图
for i, ws in enumerate(unique_ws):
    data_ws = df_avg[df_avg['init_sensitivity'] == ws]
    axes[0].plot(
        data_ws['Noise Level'], 
        data_ws['MSE_mean'],
        marker='o',
        markersize=7,
        linewidth=2.5,
        color=palette_ws[i],
        label=f'{ws:.1f}',
        zorder=3
    )

axes[0].set_title(r'(a) Sensitivity $\gamma$ vs Noise Level', fontweight='bold', fontsize=11, loc='center')
axes[0].set_xlabel(r'Noise Level $\sigma$')
axes[0].set_ylabel('MSE')
axes[0].set_xticks([0.1, 0.3, 0.5, 0.7, 0.9])
axes[0].set_xticklabels(['0.1', '0.3', '0.5', '0.7', '0.9'])
axes[0].legend(title=r'Sensitivity $\gamma$', loc='upper left', framealpha=0.9, edgecolor='gray')
axes[0].set_ylim(top=axes[0].get_ylim()[1] * 1.1)
axes[0].grid(True, linestyle='--', alpha=0.25)

# ======================== 右图: Scatter Plot + Mean Lines ========================
# 方案：散点图显示所有原始数据点 + 均值线连接

# 为每个sensitivity创建一个子位置（类似分组）
sensitivity_positions = {1.0: 0, 5.0: 1, 10.0: 2}
noise_offset = {0.1: -0.15, 0.3: -0.075, 0.5: 0, 0.7: 0.075, 0.9: 0.15}

# 绘制散点（所有原始数据点）
for i, noise in enumerate(unique_noises):
    for j, sens in enumerate(unique_ws):
        data_subset = df_synth[(df_synth['Noise Level'] == noise) & 
                               (df_synth['init_sensitivity'] == sens)]
        
        if len(data_subset) > 0:
            # 计算x位置：sensitivity基准位置 + 噪声级别偏移
            x_pos = sensitivity_positions[sens] + noise_offset[noise]
            
            # 绘制原始数据点（半透明）
            axes[1].scatter(
                [x_pos] * len(data_subset),
                data_subset['MSE'],
                color=palette_noise[i],
                alpha=0.6,
                s=50,
                edgecolors='white',
                linewidth=0.5,
                zorder=2
            )
            
            # 绘制均值标记（实心，更显眼）
            mean_val = data_subset['MSE'].mean()
            axes[1].scatter(
                x_pos,
                mean_val,
                color=palette_noise[i],
                marker='D',  # 菱形
                s=80,
                edgecolors='black',
                linewidth=1.2,
                zorder=3,
                label=f'{noise:.1f}' if j == 0 else None  # 只在第一个sensitivity添加图例
            )

# 为每个噪声级别绘制连接均值的线
for i, noise in enumerate(unique_noises):
    x_positions = []
    y_means = []
    
    for sens in unique_ws:
        data_subset = df_synth[(df_synth['Noise Level'] == noise) & 
                               (df_synth['init_sensitivity'] == sens)]
        if len(data_subset) > 0:
            x_pos = sensitivity_positions[sens] + noise_offset[noise]
            x_positions.append(x_pos)
            y_means.append(data_subset['MSE'].mean())
    
    # 绘制连线
    if len(x_positions) > 1:
        axes[1].plot(x_positions, y_means, 
                    color=palette_noise[i], 
                    linewidth=1.5, 
                    alpha=0.4,
                    linestyle='--',
                    zorder=1)

# 设置X轴
axes[1].set_xticks([0, 1, 2])
axes[1].set_xticklabels(['1.0', '5.0', '10.0'])
axes[1].set_xlabel(r'Sensitivity $\gamma$')
axes[1].set_ylabel('MSE')
axes[1].set_title(r'(b) MSE Distribution by $\gamma$', fontweight='bold', fontsize=11, loc='center')

# 图例：显示噪声级别（菱形图标）
axes[1].legend(title=r'Noise $\sigma$', 
               loc='upper left', 
               framealpha=0.9, 
               edgecolor='gray',
               ncol=1,
               handletextpad=0.5,
               columnspacing=0.8)

axes[1].set_ylim(top=axes[1].get_ylim()[1] * 1.1)
axes[1].grid(True, linestyle='--', alpha=0.25)

# 保存高分辨率图片
plt.savefig('parameter_sensitivity_analysis.pdf', bbox_inches='tight', dpi=300)
plt.show()