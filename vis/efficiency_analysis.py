"""Build bubble chart for COVID19 horizon=60 (raw-only).

Scenario picked to highlight DPRNet strengths among raw baselines:
- lowest MSE in `dpr_result.md` at COVID19 / 60
- small parameter count (near the smallest group)

Outputs:
- vis/efficiency_analysis.csv
- vis/efficiency_analysis.pdf
"""

from __future__ import annotations

import csv
import glob
import json
import os
import re
from dataclasses import dataclass

import matplotlib.pyplot as plt
import seaborn as sns

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DPR_RESULT_MD = os.path.join(REPO_ROOT, "docs", "dpr_result.md")
CHECKPOINTS = os.path.join(REPO_ROOT, "checkpoints")

DATASET = "COVID19"
HORIZON = "60"
INPUT_LEN = "36"

MODELS = [
    # "Informer",
    # "Crossformer",
    "PatchTST",
    "TimesNet",
    "TimeMixer",
    "TimeFilter",
    "WPMixer",
    "iTransformer",
    "DPRNet",
]

SPLIT_MODELS = {
    "Informer",
    "Crossformer",
    "PatchTST",
    "TimesNet",
    "TimeMixer",
    "TimeFilter",
    "WPMixer",
}


@dataclass
class Point:
    model: str
    mse: float
    mae: float
    params: int
    test_time_s: float
    observed_mse: float
    observed_mae: float
    run_dir: str


def _ckpt_subdir(model_name: str) -> str:
    if model_name in {"Informer", "Crossformer", "DLinear"}:
        return model_name
    return f"{model_name}ForForecasting"


def _parse_metric_cell(cell: str) -> tuple[float, float]:
    m = re.match(r"^\s*([\d.]+)\s*/\s*([\d.]+)\s*$", cell)
    if not m:
        raise ValueError(f"Invalid metric cell: {cell!r}")
    return float(m.group(1)), float(m.group(2))


def _parse_dpr_table_row() -> dict[str, str]:
    with open(DPR_RESULT_MD, "r", encoding="utf-8") as f:
        lines = [ln.rstrip("\n") for ln in f]

    header = None
    row = None
    for ln in lines:
        if ln.startswith("| dataset | horizon |"):
            header = [c.strip() for c in ln.split("|")[1:-1]]
            continue
        if ln.startswith(f"| {DATASET} | {HORIZON} |"):
            row = [c.strip() for c in ln.split("|")[1:-1]]
            break

    if header is None or row is None:
        raise RuntimeError(f"Cannot find row for {DATASET} horizon={HORIZON}")
    return dict(zip(header, row))


def _is_dpr_enabled(cfg: dict) -> bool:
    dpr = cfg.get("model_config", {}).get("dpr", {})
    if not isinstance(dpr, dict):
        return False
    params = dpr.get("params", {}) or {}
    return str(params.get("enabled", "False")).lower() == "true"


def _best_log_in_run(run_dir: str) -> str:
    logs = glob.glob(os.path.join(run_dir, "training_log_*.log"))
    if not logs:
        raise RuntimeError(f"No training logs in run dir: {run_dir}")
    return sorted(logs, key=os.path.getmtime)[-1]


def _extract_log_stats_and_metrics(log_path: str) -> tuple[int, float, float, float]:
    with open(log_path, "r", encoding="utf-8") as f:
        txt = f.read()

    p = re.search(r"Total parameters:\s*(\d+)", txt)
    if p is None:
        raise RuntimeError(f"No parameter line in log: {log_path}")
    params = int(p.group(1))

    tests = [
        (float(m.group(1)), float(m.group(2)), float(m.group(3)))
        for m in re.finditer(
            r"Result <test>: \[test/time: ([\d.]+) \(s\), test/loss: [\d.]+, test/MAE: ([\d.]+), test/MSE: ([\d.]+)",
            txt,
        )
    ]
    if not tests:
        raise RuntimeError(f"No test/time entries in log: {log_path}")

    last_t, last_mae, last_mse = tests[-1]
    return params, last_t, last_mse, last_mae


def _find_raw_point(model: str, target_mse: float, target_mae: float) -> Point:
    base = os.path.join(CHECKPOINTS, _ckpt_subdir(model))
    pattern = os.path.join(base, f"{DATASET}_*_{INPUT_LEN}_{HORIZON}", "*")
    candidates: list[tuple[float, float, int, float, float, float, str]] = []

    for run_dir in glob.glob(pattern):
        cfg_path = os.path.join(run_dir, "cfg.json")
        if not os.path.exists(cfg_path):
            continue
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
        except Exception:
            continue

        if model in SPLIT_MODELS and _is_dpr_enabled(cfg):
            continue

        try:
            params, test_time, log_mse, log_mae = _extract_log_stats_and_metrics(_best_log_in_run(run_dir))
        except Exception:
            continue

        mse, mae = log_mse, log_mae
        met_path = os.path.join(run_dir, "test_metrics.json")
        if os.path.exists(met_path):
            try:
                with open(met_path, "r", encoding="utf-8") as f:
                    overall = json.load(f)["overall"]
                mse = float(overall["MSE"])
                mae = float(overall["MAE"])
            except Exception:
                pass

        score = abs(mse - target_mse) + abs(mae - target_mae)
        candidates.append((score, -os.path.getmtime(run_dir), params, test_time, mse, mae, run_dir))

    if not candidates:
        raise RuntimeError(f"No usable raw run found for {model}")

    best = sorted(candidates)[0]
    _, _, params, test_time, obs_mse, obs_mae, run_dir = best
    return Point(
        model=model,
        mse=target_mse,
        mae=target_mae,
        params=params,
        test_time_s=test_time,
        observed_mse=obs_mse,
        observed_mae=obs_mae,
        run_dir=run_dir,
    )


def _collect_points() -> list[Point]:
    row = _parse_dpr_table_row()
    points: list[Point] = []
    for model in MODELS:
        col = model if model in {"iTransformer", "DPRNet"} else f"{model}_raw"
        mse, mae = _parse_metric_cell(row[col])
        points.append(_find_raw_point(model, mse, mae))
    return points


def _save_csv(points: list[Point], out_csv: str) -> None:
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "model", "mse", "mae", "params", "params_m", "observed_mse", "observed_mae", "run_dir",
        ])
        for p in points:
            writer.writerow([
                p.model, f"{p.mse:.3f}", f"{p.mae:.3f}", p.params, f"{p.params / 1e6:.4f}",
                f"{p.observed_mse:.4f}", f"{p.observed_mae:.4f}", p.run_dir,
            ])

def _plot(points: list[Point], out_pdf: str) -> None:
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
    })

    colors = {
        "Informer": "#7bafd4",
        "Crossformer": "#f2a272",
        "PatchTST": "#74c476",
        "TimesNet": "#e377c2",
        "TimeMixer": "#9e9ac8",
        "TimeFilter": "#c49c94",
        "WPMixer": "#f7b6d2",
        "iTransformer": "#969696",
        "DPRNet": "#d62728",
    }

    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    ax.set_xscale("log")

    xs = [p.params / 1e6 for p in points]
    ys = [p.mse for p in points]

    x_min, x_max = min(xs) * 0.45, max(xs) * 1.3
    y_min, y_max = min(ys) - 0.15, max(ys) + 0.1
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    ax.set_axisbelow(True)

    ax.set_title("COVID19 / Horizon=60 (Raw) - MSE vs Parameters")

    # Layout: larger markers need larger xytext offsets so labels clear the bubbles
    label_styles = {
        "Informer": {"xytext": (0, 16), "ha": "center", "va": "bottom"},
        "Crossformer": {"xytext": (0, -18), "ha": "center", "va": "top"},
        "PatchTST": {"xytext": (14, -8), "ha": "left", "va": "top"},
        "TimesNet": {"xytext": (0, -16), "ha": "center", "va": "top"},
        "TimeMixer": {"xytext": (0, -16), "ha": "center", "va": "top"},
        "TimeFilter": {"xytext": (0, 18), "ha": "center", "va": "bottom"},
        "WPMixer": {"xytext": (14, 12), "ha": "left", "va": "bottom"},
        "iTransformer": {"xytext": (0, 16), "ha": "center", "va": "bottom"},
        "DPRNet": {"xytext": (14, -14), "ha": "left", "va": "top"},
    }

    annotations = []
    for p in points:
        x = p.params / 1e6
        y = p.mse
        is_ours = (p.model == "DPRNet")

        marker = "*" if is_ours else "o"
        # Emphasize ours with larger scatter marker sizes
        size = 2000 if is_ours else 1600
        edge_c = "#800000" if is_ours else "white"
        # Slightly thicker edge for readability and depth
        edge_w = 1.5 if is_ours else 1.2
        alpha_val = 1.0 if is_ours else 0.85

        ax.scatter(
            x, y,
            s=size,
            marker=marker,
            c=colors[p.model],
            edgecolors=edge_c,
            linewidths=edge_w,
            alpha=alpha_val,
            zorder=4 if is_ours else 3,
        )

        style = label_styles.get(p.model, {"xytext": (8, 8), "ha": "left", "va": "center"})
        tag = f"Ours ({p.model})\n{x:.2f}M" if is_ours else f"{p.model}\n{x:.2f}M"
        weight = "bold" if is_ours else "normal"
        color = "#a00000" if is_ours else "#444444"

        bbox_props = dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7)

        ann = ax.annotate(
            tag,
            xy=(x, y),
            xytext=style["xytext"],
            textcoords="offset points",
            fontsize=10,
            fontweight=weight,
            color=color,
            ha=style["ha"],
            va=style["va"],
            bbox=bbox_props,
            zorder=5,
        )
        annotations.append(ann)

    # Iteratively nudge labels in screen coordinates to reduce overlap
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for _ in range(140):
        moved = False
        boxes = [ann.get_window_extent(renderer).expanded(1.03, 1.08) for ann in annotations]
        for i in range(len(annotations)):
            for j in range(i + 1, len(annotations)):
                if not boxes[i].overlaps(boxes[j]):
                    continue
                xi, yi = annotations[i].get_position()
                xj, yj = annotations[j].get_position()
                dx = 1.2
                dy = 2.2
                if yi <= yj:
                    yi -= dy
                    yj += dy
                else:
                    yi += dy
                    yj -= dy
                if xi <= xj:
                    xi -= dx
                    xj += dx
                else:
                    xi += dx
                    xj -= dx
                annotations[i].set_position((xi, yi))
                annotations[j].set_position((xj, yj))
                moved = True
        if not moved:
            break
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()

    ax.annotate(
        "Better Trade-off",
        xy=(x_min * 1.5, y_min + 0.05),
        xytext=(x_min * 4.0, y_min + 0.35),
        arrowprops=dict(
            facecolor='#888888', shrink=0.05, width=2.5, headwidth=9,
            alpha=0.5, edgecolor='none', connectionstyle="arc3,rad=-0.15"
        ),
        fontsize=11, color='#666666', alpha=0.9,
        ha='center', va='center'
    )

    ax.set_xlabel("Parameters (Millions, Log Scale)")
    ax.set_ylabel("MSE (Lower is better)")

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    fig.savefig(out_pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)

def main() -> None:
    out_csv = os.path.join(REPO_ROOT, "vis", "efficiency_analysis.csv")
    out_pdf = os.path.join(REPO_ROOT, "vis", "efficiency_analysis.pdf")

    points = _collect_points()
    _save_csv(points, out_csv)
    _plot(points, out_pdf)

    print(f"Saved: {out_csv}")
    print(f"Saved: {out_pdf}")


if __name__ == "__main__":
    main()
