"""
Parameter Scaling vs DPR — grouped bar chart version.

Replaces the scatter plot with grouped bars for clearer MSE comparison.
One row, three columns = ETTh1, Illness, Exchange.
"""

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
import seaborn as sns

# Unify style with other visualisations (whitegrid + sans-serif, used in
# efficiency_analysis.py, parameter_sensitivity.py, dataset_landscape.py, etc.)
sns.set_theme(
    style="whitegrid",
    font_scale=1.0,
    rc={
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "axes.edgecolor": "#333333",
        "axes.linewidth": 1.0,
        "xtick.color": "#333333",
        "ytick.color": "#333333",
    },
)

MODEL_COLORS = {
    "PatchTST": "#74c476",
    "TimesNet": "#e377c2",
    "TimeFilter": "#c49c94",
}

_MODEL_VARIANT_PARAMS = {
    "PatchTST": {
        "Raw": 1.08,
        "2xW": 3.75,
        "2xD": 1.87,
        "2xB": 6.90,
        "PM": 1.08,
        "+DPR": 1.10,
    },
    "TimesNet": {
        "Raw": 18.5,
        "2xW": 73.7,
        "2xD": 36.8,
        "2xB": 147.0,
        "PM": 18.5,
        "+DPR": 18.5,
    },
    "TimeFilter": {
        "Raw": 0.19,
        "2xW": 2.99,
        "2xD": 1.50,
        "2xB": 5.39,
        "PM": 0.19,
        "+DPR": 0.19,
    },
}

_MSE = {
    "ETTh1": {
        "PatchTST":  {"Raw": 0.394, "2xW": 0.405, "2xD": 0.401, "2xB": 0.398, "PM": 0.394, "+DPR": 0.394},
        "TimesNet":  {"Raw": 0.488, "2xW": 0.561, "2xD": 0.503, "2xB": 0.598, "PM": 0.488, "+DPR": 0.475},
        "TimeFilter": {"Raw": 0.390, "2xW": 0.399, "2xD": 0.397, "2xB": 0.403, "PM": 0.391, "+DPR": 0.390},
    },
    "Illness": {
        "PatchTST":  {"Raw": 3.633, "2xW": 3.415, "2xD": 3.199, "2xB": 3.114, "PM": 3.633, "+DPR": 3.108},
        "TimesNet":  {"Raw": 9.241, "2xW": 5.859, "2xD": 6.827, "2xB": 3.515, "PM": 9.345, "+DPR": 3.445},
        "TimeFilter": {"Raw": 1.991, "2xW": 3.264, "2xD": 3.285, "2xB": 2.656, "PM": 2.808, "+DPR": 1.821},
    },
    "Exchange": {
        "PatchTST":  {"Raw": 0.106, "2xW": 0.110, "2xD": 0.110, "2xB": 0.111, "PM": 0.108, "+DPR": 0.104},
        "TimesNet":  {"Raw": 0.137, "2xW": 0.132, "2xD": 0.145, "2xB": 0.138, "PM": 0.129, "+DPR": 0.128},
        "TimeFilter": {"Raw": 0.107, "2xW": 0.106, "2xD": 0.104, "2xB": 0.112, "PM": 0.105, "+DPR": 0.103},
    },
}

_IMPROVEMENT = {
    "ETTh1": {
        "PatchTST":   None,
        "TimesNet":   "2.7%",
        "TimeFilter": None,
    },
    "Illness": {
        "PatchTST":   "14.5%",
        "TimesNet":   "62.7%",
        "TimeFilter": "8.5%",
    },
    "Exchange": {
        "PatchTST":   "1.9%",
        "TimesNet":   "6.6%",
        "TimeFilter": "3.7%",
    },
}

dataset_order = ["ETTh1", "Illness", "Exchange"]
model_order = ["PatchTST", "TimesNet", "TimeFilter"]
variant_order = ["Raw", "2xW", "2xD", "2xB", "PM", "+DPR"]

bar_colors = {
    "Raw":   "#BDBDBD",
    "2xW":   "#9E9E9E",
    "2xD":   "#9E9E9E",
    "2xB":   "#9E9E9E",
    "PM":    "#757575",
}

bar_edge_colors = {
    "Raw":   "none",
    "2xW":   "none",
    "2xD":   "none",
    "2xB":   "none",
    "PM":    "none",
}

n_models = len(model_order)
n_variants = len(variant_order)
group_width = 0.8
bar_width = group_width / n_variants

FIG_W, FIG_H = 14.0, 4.2
fig, axes = plt.subplots(1, 3, figsize=(FIG_W, FIG_H), layout="constrained", dpi=300)

for ax, dname in zip(axes, dataset_order):
    x_positions = np.arange(n_models)
    
    for v_idx, variant in enumerate(variant_order):
        offsets = (v_idx - n_variants / 2 + 0.5) * bar_width
        xs = x_positions + offsets
        ys = [_MSE[dname][m][variant] for m in model_order]
        if variant == "+DPR":
            colors = [MODEL_COLORS[m] for m in model_order]
            edgecolors = ["#333333"] * n_models
            linewidth = 1.0
        else:
            colors = [bar_colors[variant]] * n_models
            edgecolors = [bar_edge_colors[variant]] * n_models
            linewidth = 0
        
        bars = ax.bar(
            xs,
            ys,
            width=bar_width * 0.92,
            color=colors,
            edgecolor=edgecolors,
            linewidth=linewidth,
            label=variant,
            zorder=3,
        )
        
        # Value labels on +DPR bars
        if variant == "+DPR":
            for x, y, m in zip(xs, ys, model_order):
                imp = _IMPROVEMENT[dname][m]
                label_text = f"{y:.3f}"
                if imp is not None:
                    label_text += f"\n↑{imp}"
                ax.text(
                    x,
                    y + ax.get_ylim()[1] * 0.01,
                    label_text,
                    ha="center",
                    va="bottom",
                    fontsize=7.5,
                    fontweight="bold",
                    color=MODEL_COLORS[m],
                    path_effects=[pe.withStroke(linewidth=1.5, foreground="white")],
                    zorder=5,
                )
    
    # Add Raw baseline horizontal dashed line for each model
    for m_idx, model in enumerate(model_order):
        raw_mse = _MSE[dname][model]["Raw"]
        ax.hlines(
            raw_mse,
            m_idx - group_width / 2,
            m_idx + group_width / 2,
            colors="#E0E0E0",
            linestyles="--",
            linewidths=1.0,
            zorder=1,
        )
    
    all_vals = [_MSE[dname][m][v] for m in model_order for v in variant_order]
    y_min, y_max = min(all_vals), max(all_vals)
    padding = (y_max - y_min) * 0.15
    ax.set_ylim(y_min - padding, y_max + padding)

    ax.set_xticks(x_positions)
    ax.set_xticklabels(model_order, rotation=15, ha="right")
    ax.set_title(dname, fontweight="bold", fontsize=12)
    ax.set_ylabel("MSE", fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.3, zorder=0)
    ax.set_axisbelow(True)
    
    # Legend
    from matplotlib.patches import Patch
    legend_handles = []
    for v in variant_order:
        if v == "+DPR":
            facecolor = "#555555"
            edgecolor = "#333333"
        else:
            facecolor = bar_colors[v]
            edgecolor = bar_edge_colors[v]
        legend_handles.append(
            Patch(facecolor=facecolor, edgecolor=edgecolor, linewidth=1.0, label=v)
        )
    ax.legend(
        handles=legend_handles,
        loc="upper right",
        frameon=True,
        edgecolor="black",
        fontsize=8,
        title="Variant",
        title_fontsize=9,
        framealpha=0.92,
        ncol=2,
    )

plt.savefig("scaling_vs_dpr_bar.pdf", format="pdf", bbox_inches="tight", pad_inches=0.03)
plt.show()
