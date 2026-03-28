import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

# ==========================================
# 1. 设置 ICML 学术风格（统一格式）
# ==========================================
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

# ==========================================
# 2. 数据生成配置 (Data Generation)
# ==========================================
np.random.seed(42)  # 固定种子以保证结果可复现
t = np.linspace(0, 3, 1000)  # 时间轴 0-3秒

# --- 基础信号参数 ---
freq = 1.0  # 基础频率 1Hz

# --- 定义四种信号模式 (Signal Regimes) ---
# 1. Stationary (Periodic)
y_stationary = 4 * np.sin(2 * np.pi * freq * t)

# 2. Non-stationary Mean (Trend)
# 线性趋势 + 正弦波
y_trend = 0.5 * t * 2 + 2 * np.sin(2 * np.pi * freq * t) - 1

# 3. Non-stationary Frequency (Chirp)
# 频率随时间线性增加: f(t) = f0 + k*t
f0 = 0.5
k = 2.0
phase = 2 * np.pi * (f0 * t + 0.5 * k * t**2)
y_chirp = 4 * np.sin(phase)

# 4. Non-stationary Variance (AM - Amplitude Modulation)
# 载波 * 包络
carrier_freq = 10.0
mod_freq = 0.5
envelope = 1 + 0.5 * np.sin(2 * np.pi * mod_freq * t)
y_am = 2.5 * envelope * np.sin(2 * np.pi * carrier_freq * t)

# --- 定义三种噪声模式 (Noise Profiles) ---
# 基底信号用于展示噪声
y_base = y_stationary.copy()

# 1. Gaussian Noise (Aleatoric)
noise_gaussian = np.random.normal(0, 0.5, size=t.shape)
y_gaussian = y_base + noise_gaussian

# 2. Heavy-tail Noise (Student-t)
# 使用 t-分布 (df=2.5) 模拟极值
noise_heavy = np.random.standard_t(df=2.5, size=t.shape) * 0.3
y_heavy = y_base + noise_heavy

# 3. Missing Values (Failures)
# 随机 mask 掉 40% 的数据
mask = np.random.choice([0, 1], size=t.shape, p=[0.4, 0.6])
y_missing = y_base.copy()
y_missing[mask == 0] = np.nan  # 设置为 NaN，matplotlib 会自动断开连线

# ==========================================
# 3. 绘图设置 (Plotting)
# ==========================================
fig = plt.figure(figsize=(14, 6))

# 使用 GridSpec 创建两行布局
# 高度比例 1:1，行间距 hspace 适当加大
gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1], hspace=0.35)

# --- 第一行：4个信号模式 ---
gs_row1 = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=gs[0], wspace=0.25)
regime_titles = [
    "1. Stationary (Periodic)", 
    "2. Non-stationary (Mean)", 
    "3. Non-stationary (Frequency)", 
    "4. Non-stationary (Variance)"
]
regime_data = [y_stationary, y_trend, y_chirp, y_am]

for i in range(4):
    ax = fig.add_subplot(gs_row1[0, i])
    ax.plot(t, regime_data[i], color='#1F77B4', lw=1.8)
    ax.set_title(regime_titles[i], fontweight='bold', fontsize=10)
    ax.set_xlabel("Time (t)")
    ax.set_ylabel("Amplitude")
    ax.grid(True, linestyle='--', alpha=0.25)
    ax.set_ylim(-4.5, 4.5)

# --- 第二行：3个噪声模式 ---
gs_row2 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[1], wspace=0.25)
noise_titles = [
    "Gaussian Noise (Aleatoric)", 
    "Heavy-tail Noise (Student-t)", 
    "Missing Values (Failures)"
]

# 准备绘图数据 (Clean, Noisy)
noise_plot_data = [
    (y_base, y_gaussian),
    (y_base, y_heavy),
    (y_base, y_missing)
]

for i in range(3):
    ax = fig.add_subplot(gs_row2[0, i])
    clean_sig, noisy_sig = noise_plot_data[i]
    
    # 特殊处理 Missing Values 的图例和画法
    if i == 2:
        # 画完整的浅色线表示原始信号
        ax.plot(t, y_base, color='#1F77B4', alpha=0.4, lw=2.0, label='Clean Signal')
        # 画断断续续的深色线表示观测值
        ax.plot(t, noisy_sig, color='#2C3E50', lw=2.0, label='Observed (Corrupted)')
    else:
        # 对于前两个，画 Clean (Blue) 和 Noise (Orange)
        ax.plot(t, clean_sig, color='#1F77B4', alpha=0.7, lw=2.0, label='Clean Signal')
        ax.plot(t, noisy_sig, color='#FF7F0E', alpha=0.85, lw=1.2, label='Corrupted Noise')
    
    ax.set_title(noise_titles[i], fontweight='bold', fontsize=10)
    ax.set_xlabel("Time (t)")
    ax.set_ylabel("Amplitude")
    ax.grid(True, linestyle='--', alpha=0.25)
    ax.legend(loc='upper right', framealpha=0.9, edgecolor='gray')
    
    # 调整 Heavy-tail 的 Y 轴范围以显示尖峰，但不要太夸张
    if i == 1:
        ax.set_ylim(-6, 6)
    else:
        ax.set_ylim(-4.5, 4.5)

# 保存和显示
plt.savefig('signal_noise_def.pdf', bbox_inches='tight', dpi=300)
plt.show()