import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D

# ==========================================
# 1. Global matplotlib style
# ==========================================
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams["axes.labelsize"] = 12
plt.rcParams["xtick.labelsize"] = 10
plt.rcParams["ytick.labelsize"] = 10
plt.rcParams["legend.fontsize"] = 9

# ==========================================
# 2. Data: one row, three columns = three datasets.
#    Within each panel: three models (color) × variants (marker / arrows).
#    Tuple: (params in millions, MSE). Params are shared; only MSE varies by dataset.
# ==========================================
_MODEL_VARIANT_PARAMS = {
    "PatchTST": {
        "Raw": 1.08,
        "2xW": 3.75,
        "2xD": 1.87,
        "2xB": 6.90,
        "+TCM": 1.10,
    },
    "iTransformer": {
        "Raw": 1.50,
        "2xW": 3.20,
        "2xD": 2.80,
        "2xB": 8.50,
        "+TCM": 1.55,
    },
    "DLinear": {
        "Raw": 0.05,
        "2xW": 0.12,
        "2xD": 0.09,
        "2xB": 0.20,
        "+TCM": 0.06,
    },
}

# Replace MSE numbers with your experiment results (per dataset).
_MSE = {
    "ETTh1": {
        "PatchTST": {
            "Raw": 0.394,
            "2xW": 0.405,
            "2xD": 0.401,
            "2xB": 0.398,
            "+TCM": 0.385,
        },
        "iTransformer": {
            "Raw": 0.382,
            "2xW": 0.384,
            "2xD": 0.381,
            "2xB": 0.380,
            "+TCM": 0.370,
        },
        "DLinear": {
            "Raw": 0.410,
            "2xW": 0.412,
            "2xD": 0.411,
            "2xB": 0.415,
            "+TCM": 0.395,
        },
    },
    "Illness": {
        "PatchTST": {
            "Raw": 0.65,
            "2xW": 0.68,
            "2xD": 0.66,
            "2xB": 0.67,
            "+TCM": 0.56,
        },
        "iTransformer": {
            "Raw": 0.62,
            "2xW": 0.64,
            "2xD": 0.63,
            "2xB": 0.64,
            "+TCM": 0.52,
        },
        "DLinear": {
            "Raw": 0.72,
            "2xW": 0.75,
            "2xD": 0.74,
            "2xB": 0.76,
            "+TCM": 0.58,
        },
    },
    "Exchange": {
        "PatchTST": {
            "Raw": 0.12,
            "2xW": 0.14,
            "2xD": 0.13,
            "2xB": 0.14,
            "+TCM": 0.10,
        },
        "iTransformer": {
            "Raw": 0.11,
            "2xW": 0.12,
            "2xD": 0.11,
            "2xB": 0.12,
            "+TCM": 0.09,
        },
        "DLinear": {
            "Raw": 0.15,
            "2xW": 0.16,
            "2xD": 0.15,
            "2xB": 0.16,
            "+TCM": 0.12,
        },
    },
}

dataset_order = list(_MSE.keys())
model_order = list(_MODEL_VARIANT_PARAMS.keys())

# data[dataset][model][variant] = (params_M, mse)
def _build_data() -> dict:
    out: dict = {}
    for d in dataset_order:
        out[d] = {}
        for m in model_order:
            out[d][m] = {
                v: (
                    _MODEL_VARIANT_PARAMS[m][v],
                    _MSE[d][m][v],
                )
                for v in _MODEL_VARIANT_PARAMS[m]
            }
    return out


data = _build_data()

# Color = model, marker = variant
colors = {"PatchTST": "#1f77b4", "iTransformer": "#2ca02c", "DLinear": "#d62728"}
markers = {"Raw": "o", "2xW": "^", "2xD": "v", "2xB": "s", "+TCM": "*"}
sizes = {"Raw": 50, "2xW": 50, "2xD": 50, "2xB": 50, "+TCM": 160}

# Text labels "Model (variant)" next to points; set False if panels get crowded
POINT_LABELS = True
POINT_LABEL_FONTSIZE = 4.8

# Two in-panel legends: Model (left), Variant (right)
LEGEND_FRAMEALPHA = 0.92
LEGEND_FONTSIZE_MODEL = 7.0
LEGEND_FONTSIZE_VARIANT = 6.0
VARIANT_LEGEND_NCOL = 1  # use 2 for a flatter legend

model_legend_handles = [
    Line2D(
        [0],
        [0],
        marker="o",
        linestyle="",
        color=colors[m],
        markerfacecolor=colors[m],
        markersize=6,
        label=m,
    )
    for m in model_order
]

variant_legend_handles = [
    Line2D(
        [0], [0], marker=markers["Raw"], color="w", markerfacecolor="gray", markersize=6, label="Raw Baseline"
    ),
    Line2D(
        [0], [0], marker=markers["2xW"], color="w", markerfacecolor="gray", markersize=6, label="2xW (width)"
    ),
    Line2D(
        [0], [0], marker=markers["2xD"], color="w", markerfacecolor="gray", markersize=6, label="2xD (depth)"
    ),
    Line2D(
        [0], [0], marker=markers["2xB"], color="w", markerfacecolor="gray", markersize=6, label="2xB (both)"
    ),
    Line2D(
        [0],
        [0],
        marker=markers["+TCM"],
        color="w",
        markerfacecolor="gold",
        markeredgecolor="black",
        markersize=7,
        label="Ours (+TCM)",
    ),
]

# MSE scales often differ across datasets: default independent y-axes per column.
# Set True if MSE is normalized or comparable across columns.
SHARE_Y = False

# Figure width (adjust for page / column width)
FIG_W, FIG_H = 14.0, 3.4
fig, axes = plt.subplots(
    1,
    3,
    sharey=SHARE_Y,
    figsize=(FIG_W, FIG_H),
    layout="constrained",
    dpi=300,
)

# ==========================================
# 3. One dataset per column; all three models in each
# ==========================================
for ax, dname in zip(axes, dataset_order):
    for model in model_order:
        color = colors[model]
        variants = data[dname][model]
        raw_p, raw_m = variants["Raw"]

        for var_name, (p, m) in variants.items():
            if var_name == "+TCM":
                ax.scatter(
                    p,
                    m,
                    color=color,
                    marker=markers[var_name],
                    s=sizes[var_name],
                    edgecolor="black",
                    linewidth=0.9,
                    zorder=5,
                )
            else:
                ax.scatter(
                    p,
                    m,
                    color=color,
                    marker=markers[var_name],
                    s=sizes[var_name],
                    alpha=0.85,
                    zorder=4,
                )

            if var_name in ["2xW", "2xD", "2xB"]:
                ax.annotate(
                    "",
                    xy=(p, m),
                    xytext=(raw_p, raw_m),
                    arrowprops=dict(
                        arrowstyle="->",
                        color="gray",
                        linestyle="dashed",
                        alpha=0.45,
                        shrinkA=3,
                        shrinkB=3,
                    ),
                )
            elif var_name == "+TCM":
                ax.annotate(
                    "",
                    xy=(p, m),
                    xytext=(raw_p, raw_m),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=1.5, shrinkA=3, shrinkB=3),
                )

            if POINT_LABELS:
                tag = "TCM" if var_name == "+TCM" else var_name
                ox, oy = (3, 2.5) if p < 2.0 else (-11, 2.5)
                ax.annotate(
                    f"{model} ({tag})",
                    (p, m),
                    textcoords="offset points",
                    xytext=(ox, oy),
                    fontsize=POINT_LABEL_FONTSIZE,
                    color=color,
                    path_effects=[pe.withStroke(linewidth=1.2, foreground="white")],
                    zorder=6,
                )

    ax.set_xscale("log")
    ax.set_title(dname, fontweight="bold", fontsize=12)
    ax.grid(True, which="both", ls="--", alpha=0.3)

    if dname == dataset_order[0]:
        ax.text(
            0.04,
            0.06,
            "Optimal\nRegion",
            transform=ax.transAxes,
            fontsize=8,
            fontweight="bold",
            color="#2E7D32",
            style="italic",
            path_effects=[pe.withStroke(linewidth=2, foreground="white")],
            zorder=10,
        )

    leg_m = ax.legend(
        handles=model_legend_handles,
        title="Model",
        loc="upper left",
        frameon=True,
        edgecolor="black",
        fontsize=LEGEND_FONTSIZE_MODEL,
        title_fontsize=LEGEND_FONTSIZE_MODEL + 0.5,
        framealpha=LEGEND_FRAMEALPHA,
    )
    ax.add_artist(leg_m)
    ax.legend(
        handles=variant_legend_handles,
        title="Variant",
        loc="upper right",
        frameon=True,
        edgecolor="black",
        fontsize=LEGEND_FONTSIZE_VARIANT,
        title_fontsize=LEGEND_FONTSIZE_VARIANT + 1.0,
        framealpha=LEGEND_FRAMEALPHA,
        ncol=VARIANT_LEGEND_NCOL,
    )

axes[0].set_ylabel("MSE", fontweight="bold")
for ax in axes:
    ax.set_xlabel("Parameters (M) — log scale", fontweight="bold")

plt.savefig("scaling_vs_tcm.pdf", format="pdf", bbox_inches="tight", pad_inches=0.03)
plt.show()
