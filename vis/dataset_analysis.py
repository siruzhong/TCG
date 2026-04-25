"""Unified local-heterogeneity analysis for all benchmark datasets.

Running this file produces two artefacts in ``vis/``:

  1. ``dataset_analysis.pdf``  -- per-dataset signal + rolling-std figure,
     panels ordered by the composite heterogeneity score (most heterogeneous
     first).
  2. ``dataset_analysis.csv`` -- stats table, also sorted by the composite
     score in descending order.

Metrics (computed from ALL channels per dataset, after z-normalising each
channel):

  * ``ADF_p``            : Augmented Dickey--Fuller p-value. Higher => more
                           non-stationary.
  * ``spectral_entropy`` : Shannon entropy of the power spectrum (DC excluded)
                           normalised to [0, 1]. ~0 => concentrated periodicity,
                           ~1 => broadband / chaotic.
  * ``VoV``              : std / mean of the rolling-window standard deviation
                           (~1 day window). Directly measures heterogeneity in
                           local second-order moments.
  * ``score``            : ``rank(H_s) + rank(VoV)``, a non-parametric
                           composite index where each dataset is ranked 1..N
                           on each metric (N = highest). The sum therefore
                           lives in ``[2, 2N]``; larger => more locally
                           heterogeneous.

Scoring design choices
----------------------
1. **We do NOT fold ADF_p into the score.** The ADF p-value measures *global*
   stationarity: a high p-value means we cannot reject a unit root, i.e. the
   series is close to a random walk with a drifting mean. That is emphatically
   NOT the same as "rich local regime structure"; a pure random walk is
   globally non-stationary but locally homogeneous (only white-noise
   increments). Including ADF_p would spuriously reward random-walk-like
   series (e.g., ExchangeRate) whose predictability is actually low rather
   than whose regime structure is rich. ADF_p is therefore kept in the CSV
   and the figure as a diagnostic only.

2. **We use rank-sum rather than z-score sum.** VoV has a heavy right tail
   (Weather's ~2.0 versus PEMS08's ~0.1), so a z-score composite lets the one
   outlier dominate the ranking and compresses every other dataset into a
   narrow band near zero. Non-parametric rank aggregation is outlier-immune
   (Borda-count / Wilcoxon style), produces a natively bounded score in
   ``[2, 2N]``, and separates the middle datasets more cleanly.
"""
from __future__ import annotations

import csv
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

try:
    from statsmodels.tsa.stattools import adfuller
    _HAS_SM = True
except Exception:  # pragma: no cover - statsmodels is expected to be present
    _HAS_SM = False

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_ROOT = os.path.join(THIS_DIR, os.pardir, "datasets")
OUT_PDF = os.path.join(THIS_DIR, "dataset_analysis.pdf")
OUT_CSV = os.path.join(THIS_DIR, "dataset_analysis.csv")

# (name, domain, freq_str, freq_min)
DATASETS = [
    ("ETTh1",             "Energy",        "1 Hour",   60),
    ("ETTh2",             "Energy",        "1 Hour",   60),
    ("ETTm1",             "Energy",        "15 Min",   15),
    ("ETTm2",             "Energy",        "15 Min",   15),
    ("Weather",           "Climatology",   "10 Min",   10),
    ("Illness",           "Healthcare",    "1 Week",   7 * 24 * 60),
    ("ExchangeRate",      "Finance",       "1 Day",    24 * 60),
    ("BeijingAirQuality", "Environment",   "1 Hour",   60),
    ("COVID19",           "Epidemiology",  "1 Day",    24 * 60),
    ("VIX",               "Finance",       "1 Day",    24 * 60),
    ("NABCPU",            "Cloud Ops",     "5 Min",    5),
    ("Sunspots",          "Solar Physics", "1 Month",  30 * 24 * 60),
]


# --------------------------------------------------------------------------- #
# Data loading + per-dataset metrics
# --------------------------------------------------------------------------- #

def _load_full(name: str) -> np.ndarray:
    """Concatenate train/val/test into the full (T, C) array."""
    parts = [np.load(os.path.join(DATASET_ROOT, name, f"{split}_data.npy"))
             for split in ("train", "val", "test")]
    return np.concatenate(parts, axis=0).astype(np.float64)


def _window_size(freq_minutes: int) -> int:
    """Rolling window ~= 1 day, capped at 256 for low-frequency datasets."""
    if freq_minutes <= 0:
        return 24
    w = max(12, int(round(24 * 60 / freq_minutes)))
    return min(w, 256)


def _normalize_channel(x: np.ndarray) -> np.ndarray:
    mu = np.nanmean(x)
    sigma = np.nanstd(x) + 1e-8
    return (x - mu) / sigma


def _spectral_entropy(x: np.ndarray) -> float:
    x = x - np.mean(x)
    spec = np.abs(np.fft.rfft(x)) ** 2
    spec = spec[1:]  # drop DC
    if spec.sum() <= 0:
        return 0.0
    p = spec / spec.sum()
    h = -np.sum(p * np.log(p + 1e-12))
    return float(h / np.log(len(p)))  # normalised to [0, 1]


def _rolling_std(x: np.ndarray, w: int) -> np.ndarray:
    if len(x) <= w:
        return np.array([np.std(x)])
    x2 = np.convolve(x ** 2, np.ones(w) / w, mode="valid")
    x1 = np.convolve(x,       np.ones(w) / w, mode="valid")
    return np.sqrt(np.maximum(x2 - x1 ** 2, 0.0))


def compute_stats(name: str, freq_minutes: int,
                  adf_max_channels: int = 20,
                  adf_len: int = 2000) -> dict:
    """Compute the three diagnostics for one dataset.

    H_s and VoV are averaged over ALL channels (cheap). ADF is only run on the
    first ``adf_max_channels`` channels after decimating the series to
    ``adf_len`` points, which keeps runtime linear in #datasets.
    """
    data = _load_full(name)
    T, C = data.shape
    w = _window_size(freq_minutes)

    hs_list, vov_list, adf_list = [], [], []
    for c in range(C):
        x = data[:, c]
        if np.isnan(x).any():
            x = np.nan_to_num(x, nan=np.nanmean(x))
        xn = _normalize_channel(x)

        hs_list.append(_spectral_entropy(xn))

        r = _rolling_std(xn, w)
        if len(r) > 1 and r.mean() > 0:
            vov_list.append(float(r.std() / r.mean()))

        if _HAS_SM and c < adf_max_channels:
            xs = xn if len(xn) <= adf_len else xn[:: max(1, len(xn) // adf_len)][:adf_len]
            try:
                adf_list.append(adfuller(xs, autolag=None, maxlag=10)[1])
            except Exception:
                pass

    return {
        "name": name,
        "T": int(T),
        "C": int(C),
        "window": w,
        "adf_p": float(np.mean(adf_list)) if adf_list else float("nan"),
        "spec_entropy": float(np.mean(hs_list)),
        "vov": float(np.mean(vov_list)) if vov_list else float("nan"),
    }


# --------------------------------------------------------------------------- #
# Composite score + sort
# --------------------------------------------------------------------------- #

def _rank_ascending(values: np.ndarray) -> np.ndarray:
    """Rank values in ascending order (1 = smallest, N = largest).

    Ties are resolved by average rank (standard Wilcoxon convention). We do
    this in pure NumPy to avoid an extra scipy dependency at call time.
    """
    values = np.asarray(values, dtype=float)
    n = len(values)
    order = np.argsort(values, kind="stable")
    ranks = np.empty(n, dtype=float)
    # First pass: assign dense 1..N ranks.
    ranks[order] = np.arange(1, n + 1, dtype=float)
    # Second pass: average ranks across tied groups.
    sorted_vals = values[order]
    i = 0
    while i < n:
        j = i + 1
        while j < n and sorted_vals[j] == sorted_vals[i]:
            j += 1
        if j - i > 1:
            avg = (i + j + 1) / 2.0  # mean of (i+1 .. j) in 1-based ranks
            ranks[order[i:j]] = avg
        i = j
    return ranks


def attach_composite_score(stats_list: list[dict]) -> list[dict]:
    """Add ``score = rank(H_s) + rank(VoV)`` to every dict, in place.

    Each metric is ranked 1..N in ascending order (ties broken by average
    rank). Higher composite score => more locally heterogeneous. See module
    docstring for why we pick rank-sum over z-score-sum and why ADF_p is
    excluded.
    """
    hs  = np.array([s["spec_entropy"] for s in stats_list])
    vov = np.array([s["vov"]          for s in stats_list])
    rank_hs  = _rank_ascending(hs)
    rank_vov = _rank_ascending(vov)
    for s, rh, rv in zip(stats_list, rank_hs, rank_vov):
        s["rank_hs"]  = float(rh)
        s["rank_vov"] = float(rv)
        s["score"]    = float(rh + rv)  # in [2, 2N]
    return stats_list


# --------------------------------------------------------------------------- #
# PDF figure (panels ordered same as the sorted stats list)
# --------------------------------------------------------------------------- #

def plot_panels(sorted_stats: list[dict], info_by_name: dict) -> None:
    rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 11,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "axes.titleweight": "normal",
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.titlesize": 12,
        "figure.dpi": 300,
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "axes.linewidth": 0.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    n = len(sorted_stats)
    ncols = 4
    nrows = int(np.ceil(n / ncols))
    # Tight grid; pad_inches on save further trims PDF margins.
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(ncols * 3.9, nrows * 2.55),
    )
    axes = np.array(axes).reshape(-1)

    for ax, st in zip(axes, sorted_stats):
        name = st["name"]
        domain, freq_str, _ = info_by_name[name]
        data = _load_full(name)
        rng = np.random.default_rng(1)
        ch = int(rng.integers(0, data.shape[1]))
        x = np.nan_to_num(data[:, ch], nan=np.nanmean(data[:, ch]))
        xn = _normalize_channel(x)
        t = np.arange(len(xn))

        ax.plot(t, xn, color="#2b6cb0", linewidth=0.6, alpha=0.8)
        w = st["window"]
        if len(xn) > w:
            rs = _rolling_std(xn, w)
            tr = np.arange(len(rs)) + w // 2
            ax2 = ax.twinx()
            ax2.plot(tr, rs, color="#e53e3e", linewidth=0.9, alpha=0.9)
            ax2.set_yticks([])
            ax2.spines["top"].set_visible(False)

        # Three short lines avoid horizontal overlap between adjacent columns.
        ax.set_title(
            f"{name} ({domain}, {freq_str})\n"
            f"$T$={st['T']:,}  $C$={st['C']}  ADF $p$={st['adf_p']:.2f}\n"
            f"$H_s$={st['spec_entropy']:.2f}  VoV={st['vov']:.2f}  "
            f"score={st['score']:.1f}",
            fontsize=8.5,
            fontweight="normal",
            pad=1,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        _lw = 0.5
        for s in ("left", "bottom"):
            ax.spines[s].set_linewidth(_lw)
        if len(xn) > w:
            ax2.spines["right"].set_linewidth(_lw)

    for ax in axes[n:]:
        ax.axis("off")

    fig.suptitle(
        "Local heterogeneity across benchmark datasets "
        "(panels ordered by composite score, highest first)",
        y=0.98,
    )
    fig.subplots_adjust(
        left=0.025,
        right=0.985,
        top=0.875,
        bottom=0.025,
        wspace=0.08,
        hspace=0.40,
    )
    fig.savefig(OUT_PDF, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"Saved figure to {OUT_PDF}")


# --------------------------------------------------------------------------- #
# CSV + stdout table
# --------------------------------------------------------------------------- #

def save_csv(sorted_stats: list[dict], info_by_name: dict) -> None:
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["rank", "dataset", "domain", "frequency", "T", "C",
                         "ADF_p", "spectral_entropy", "VoV",
                         "rank_H_s", "rank_VoV", "score"])
        for rank, s in enumerate(sorted_stats, 1):
            name = s["name"]
            domain, freq_str, _ = info_by_name[name]
            writer.writerow([rank, name, domain, freq_str, s["T"], s["C"],
                             f"{s['adf_p']:.4f}",
                             f"{s['spec_entropy']:.4f}",
                             f"{s['vov']:.4f}",
                             f"{s['rank_hs']:g}",
                             f"{s['rank_vov']:g}",
                             f"{s['score']:g}"])
    print(f"Saved stats to {OUT_CSV}")


def print_ranking(sorted_stats: list[dict]) -> None:
    n = len(sorted_stats)
    print(f"\nComposite local-heterogeneity ranking  "
          f"(score = rank(H_s) + rank(VoV), in [2, {2 * n}]; ADF_p diagnostic only)")
    print("-" * 92)
    print(f"{'rank':>4s} {'dataset':20s} {'T':>8s} {'C':>5s} "
          f"{'ADF p':>7s} {'H_s':>6s} {'VoV':>6s} "
          f"{'r(H)':>5s} {'r(V)':>5s} {'score':>6s}")
    print("-" * 92)
    for rank, s in enumerate(sorted_stats, 1):
        print(f"{rank:>4d} {s['name']:20s} {s['T']:>8d} {s['C']:>5d} "
              f"{s['adf_p']:>7.3f} {s['spec_entropy']:>6.3f} {s['vov']:>6.3f} "
              f"{s['rank_hs']:>5g} {s['rank_vov']:>5g} {s['score']:>6g}")


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def main() -> None:
    info_by_name = {name: (domain, freq_str, freq_min)
                    for name, domain, freq_str, freq_min in DATASETS}

    stats_list = [compute_stats(name, freq_min) for name, _, _, freq_min in DATASETS]
    attach_composite_score(stats_list)

    stats_list.sort(key=lambda s: -s["score"])   # high -> low

    save_csv(stats_list, info_by_name)
    print_ranking(stats_list)
    plot_panels(stats_list, info_by_name)


if __name__ == "__main__":
    main()
