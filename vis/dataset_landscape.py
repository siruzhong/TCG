import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from io import StringIO
from matplotlib.lines import Line2D

# 1. Embedded CSV (heterogeneity ranking snapshot)
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

# 2. Matplotlib defaults for publication PDFs
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
    "axes.edgecolor": ".15",
    "grid.linestyle": "--",
    "axes.linewidth": 1.2,
    "figure.dpi": 300,
    "axes.facecolor": "white",
    "figure.facecolor": "white",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# 3. Figure
fig, ax = plt.subplots(figsize=(6.0, 5), dpi=300)

# Bubble area scales with log10(T)
t_log = np.log10(df['T'])
sizes = 150 + ((t_log - t_log.min()) / (t_log.max() - t_log.min())) * 500

# Color by domain (Set2)
domain_order = sorted(df['domain'].unique())
palette_colors = sns.color_palette("Set2", n_colors=len(domain_order)).as_hex()
domain_palette = {d: c for d, c in zip(domain_order, palette_colors)}

x = df['spectral_entropy']
y = df['VoV']

# Median crosshairs + quadrant watermark labels
x_mid = x.median()
y_mid = y.median()

ax.axvline(x_mid, color='#AAAAAA', linestyle='--', linewidth=1.2, zorder=1, alpha=0.8)
ax.axhline(y_mid, color='#AAAAAA', linestyle='--', linewidth=1.2, zorder=1, alpha=0.8)

# Legend is above the axes, so corners stay symmetric in axes coordinates
watermark_kwargs = dict(fontsize=11, color='#999999', alpha=0.18, ha='center', va='center', zorder=0)
ax.text(0.25, 0.85, 'Low Complexity\nHigh Non-stationarity', transform=ax.transAxes, **watermark_kwargs)
ax.text(0.75, 0.85, 'High Complexity\nHigh Non-stationarity', transform=ax.transAxes, **watermark_kwargs)
ax.text(0.25, 0.15, 'Low Complexity\nLow Non-stationarity', transform=ax.transAxes, **watermark_kwargs)
ax.text(0.75, 0.15, 'High Complexity\nLow Non-stationarity', transform=ax.transAxes, **watermark_kwargs)
# ---------------------------------------------------

# 4. Scatter
scatter = ax.scatter(
    x, y,
    s=sizes,
    c=df['domain'].map(domain_palette),
    alpha=0.85,
    edgecolor='white', 
    linewidth=1.2,
    zorder=3,
)

# Tight limits with padding
x_pad = (x.max() - x.min()) * 0.1
y_pad = (y.max() - y.min()) * 0.1
ax.set_xlim(x.min() - x_pad, x.max() + x_pad)
ax.set_ylim(y.min() - y_pad, y.max() + y_pad)

# 5. Dataset name labels (per-dataset positions)
label_positions = {
    "Weather":        {"xytext": (10, 0), "ha": "left", "va": "center"},
    "COVID19":        {"xytext": (6, 0), "ha": "left", "va": "center"},
    "ExchangeRate":   {"xytext": (10, 0), "ha": "left", "va": "center"},
    "VIX":            {"xytext": (-6, 0), "ha": "right", "va": "center"},
    "BeijingAirQuality": {"xytext": (10, 0), "ha": "left", "va": "center"},
    "ETTm2":          {"xytext": (-9, 0), "ha": "right", "va": "center"},
    "ETTm1":          {"xytext": (0, -12), "ha": "center", "va": "top"},
    "Sunspots":       {"xytext": (10, 0), "ha": "left", "va": "center"},
}
default_style = {"xytext": (0, 9), "ha": "center", "va": "bottom"}

for i in range(df.shape[0]):
    ds = df['dataset'].iloc[i]
    style = label_positions.get(ds, default_style)
    ax.annotate(
        ds,
        (x.iloc[i], y.iloc[i]),
        xytext=style["xytext"],
        textcoords='offset points',
        fontsize=9,
        fontweight='normal',
        color='#222222',
        ha=style["ha"],
        va=style["va"],
        zorder=4
    )

# 6. Axis labels and title (extra pad leaves room for the legend below the title)
ax.set_xlabel('Spectral Entropy (Complexity)')
ax.set_ylabel('Volatility of Volatility (Non-stationarity)')
# ax.set_title('Dataset Diversity Landscape: Complexity vs. Non-stationarity', pad=60)

ax.grid(axis='both', alpha=0.3, zorder=0, color='#E0E0E0')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(1.2)
ax.spines['bottom'].set_linewidth(1.2)

# 7. Legend above the plot, horizontal layout
legend_handles = [
    Line2D([0], [0], marker='o', color='none', label=domain, 
           markerfacecolor=domain_palette[domain], markeredgecolor='white', 
           markersize=9, markeredgewidth=1)
    for domain in domain_order
]

leg = ax.legend(
    handles=legend_handles,
    loc='lower center',
    bbox_to_anchor=(0.5, 1.02),
    ncol=4,
    frameon=False,
    fontsize=10,
    columnspacing=1.5,
    handletextpad=0.3,
)

plt.tight_layout()

# 8. Save
plt.savefig('dataset_landscape.pdf', format='pdf', bbox_inches='tight')
plt.show()