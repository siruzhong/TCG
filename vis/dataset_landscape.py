import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from io import StringIO
from matplotlib.lines import Line2D

# 1. 载入数据集数据
csv_data = """rank,dataset,domain,frequency,T,C,ADF_p,spectral_entropy,VoV,rank_H_s,rank_VoV,score
1,Illness,Healthcare,1 Week,966,7,0.0722,0.5176,0.9995,10,9,19
2,BeijingAirQuality,Environment,1 Hour,36000,7,0.0000,0.6089,0.9100,11,8,19
3,COVID19,Epidemiology,1 Day,1143,8,0.0438,0.5016,1.4648,8,11,19
4,Weather,Climatology,10 Min,52696,21,0.0033,0.4514,1.6813,5,12,17
5,VIX,Finance,1 Day,9165,1,0.0000,0.4965,0.8722,7,7,14
6,NABCPU,Cloud Ops,5 Min,4031,3,0.1374,0.7754,0.3955,12,1,13
7,Sunspots,Solar Physics,1 Month,3327,1,0.0002,0.5030,0.5573,9,4,13
8,ExchangeRate,Finance,1 Day,7588,8,0.5499,0.2067,1.3221,1,10,11
9,ETTh1,Energy,1 Hour,14400,7,0.0165,0.4686,0.4786,6,3,9
10,ETTh2,Energy,1 Hour,14400,7,0.0249,0.3586,0.6536,3,6,9
11,ETTm2,Energy,15 Min,57600,7,0.0249,0.3126,0.6411,2,5,7
12,ETTm1,Energy,15 Min,57600,7,0.0165,0.4114,0.4743,4,2,6"""

df = pd.read_csv(StringIO(csv_data))

# 2. 统一论文风格
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# 3. 创建画布 
fig, ax = plt.subplots(figsize=(8.0, 5), dpi=300)

# 计算气泡大小：适度缩放
t_log = np.log10(df['T'])
sizes = 150 + ((t_log - t_log.min()) / (t_log.max() - t_log.min())) * 500

# 配色：使用更明亮的调色板
domain_order = sorted(df['domain'].unique())
palette_colors = sns.color_palette("Set2", n_colors=len(domain_order)).as_hex()
domain_palette = {d: c for d, c in zip(domain_order, palette_colors)}

x = df['spectral_entropy']
y = df['VoV']

# --- 赋予空白区域意义的十字交叉线与象限水印 ---
x_mid = x.median()
y_mid = y.median()

# 画十字参考线
ax.axvline(x_mid, color='#AAAAAA', linestyle='--', linewidth=1.2, zorder=1, alpha=0.8)
ax.axhline(y_mid, color='#AAAAAA', linestyle='--', linewidth=1.2, zorder=1, alpha=0.8)

# 恢复对称的水印坐标：因为图例移出去了，四个角落可以完美对称了
watermark_kwargs = dict(fontsize=13, color='#999999', alpha=0.18, fontweight='bold', ha='center', va='center', zorder=0)
ax.text(0.25, 0.85, 'Low Complexity\nHigh Non-stationarity', transform=ax.transAxes, **watermark_kwargs)
ax.text(0.75, 0.85, 'High Complexity\nHigh Non-stationarity', transform=ax.transAxes, **watermark_kwargs)
ax.text(0.25, 0.15, 'Low Complexity\nLow Non-stationarity', transform=ax.transAxes, **watermark_kwargs)
ax.text(0.75, 0.15, 'High Complexity\nLow Non-stationarity', transform=ax.transAxes, **watermark_kwargs)
# ---------------------------------------------------

# 4. 绘制散点图 
scatter = ax.scatter(
    x, y,
    s=sizes,
    c=df['domain'].map(domain_palette),
    alpha=0.85,
    edgecolor='white', 
    linewidth=1.2,
    zorder=3,
)

# 紧凑边界
x_pad = (x.max() - x.min()) * 0.1
y_pad = (y.max() - y.min()) * 0.1
ax.set_xlim(x.min() - x_pad, x.max() + x_pad)
ax.set_ylim(y.min() - y_pad, y.max() + y_pad)

# 5. 优化标签
for i in range(df.shape[0]):
    ax.annotate(
        df['dataset'].iloc[i],
        (x.iloc[i], y.iloc[i]),
        xytext=(0, 9), 
        textcoords='offset points',
        fontsize=9.5,
        fontweight='bold',
        color='#222222',
        ha='center',
        va='bottom',
        zorder=4
    )

# 6. 美化轴标签和标题
ax.set_xlabel('Spectral Entropy (Complexity)', fontsize=12, fontweight='bold')
ax.set_ylabel('Volatility of Volatility (Non-stationarity)', fontsize=12, fontweight='bold')
# 注意这里增加了 pad=45，专门为了给下方的图例留出空间，防止标题和图例打架
ax.set_title('Dataset Diversity Landscape: Complexity vs. Non-stationarity', fontsize=14, fontweight='bold', pad=60)

ax.grid(axis='both', alpha=0.3, zorder=0, color='#E0E0E0')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(1.2)
ax.spines['bottom'].set_linewidth(1.2)

# 7. 修改图例：移至图表正上方，采用横向排布
legend_handles = [
    Line2D([0], [0], marker='o', color='none', label=domain, 
           markerfacecolor=domain_palette[domain], markeredgecolor='white', 
           markersize=9, markeredgewidth=1)
    for domain in domain_order
]

leg = ax.legend(
    handles=legend_handles,
    loc='lower center',             # 以图例底部中心为锚点
    bbox_to_anchor=(0.5, 1.02),     # 放置在图表上方 (y=1.02代表刚刚超出上边界)
    ncol=4,                         # 8个类别分为4列2行
    frameon=False,                  # 放在外面的图例去掉边框会更显高级
    fontsize=10.5,
    columnspacing=1.5,              # 增加列与列之间的间距
    handletextpad=0.3               # 图标和文字的间距
)

plt.tight_layout()

# 8. 保存
plt.savefig('dataset_landscape.pdf', format='pdf', bbox_inches='tight')
plt.show()