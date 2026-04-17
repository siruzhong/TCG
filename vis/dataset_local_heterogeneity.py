"""Quantify and visualize local heterogeneity for all benchmark datasets.

Produces:
  - vis/dataset_local_heterogeneity.pdf : 11-panel figure (time series + rolling std)
  - vis/dataset_statistics.csv          : per-dataset summary table used in the appendix

Metrics (per dataset, averaged across channels):
  - ADF p-value: Augmented Dickey-Fuller test; higher p => more non-stationary.
  - Spectral entropy (normalized): Shannon entropy of the power spectrum.
    Values close to 1 indicate broadband/chaotic dynamics, values close to 0
    indicate concentrated periodicity.
  - VoV (volatility-of-volatility): std / mean of rolling-window std, a direct
    measure of heterogeneity in local dynamics (how much local variance itself
    changes over time).
"""
from __future__ import annotations

import os
import csv

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

try:
    from statsmodels.tsa.stattools import adfuller
    _HAS_SM = True
except Exception:
    _HAS_SM = False

DATASET_ROOT = os.path.join(os.path.dirname(__file__), os.pardir, "datasets")
OUT_PDF = os.path.join(os.path.dirname(__file__), "dataset_local_heterogeneity.pdf")
OUT_CSV = os.path.join(os.path.dirname(__file__), "dataset_statistics.csv")

DATASETS = [
    ("ETTh1",             "Energy",        "1 Hour",   60),
    ("ETTh2",             "Energy",        "1 Hour",   60),
    ("ETTm1",             "Energy",        "15 Min",   15),
    ("ETTm2",             "Energy",        "15 Min",   15),
    ("Weather",           "Climatology",   "10 Min",   10),
    ("Illness",           "Healthcare",    "1 Week",   7 * 24 * 60),
    ("ExchangeRate",      "Finance",       "1 Day",    24 * 60),
    ("Solar",             "Energy",        "10 Min",   10),
    ("PEMS08",            "Traffic",       "5 Min",    5),
    ("BeijingAirQuality", "Environment",   "1 Hour",   60),
    ("COVID19",           "Epidemiology",  "1 Day",    24 * 60),
    ("VIX",               "Finance",       "1 Day",    24 * 60),
    ("NABCPU",            "Cloud Ops",     "5 Min",    5),
    ("Sunspots",          "Solar Physics", "1 Month",  30 * 24 * 60),
]


def _load_full(name: str) -> np.ndarray:
    """Concatenate train/val/test into the full time series array of shape (T, C)."""
    parts = [np.load(os.path.join(DATASET_ROOT, name, f"{split}_data.npy"))
             for split in ("train", "val", "test")]
    return np.concatenate(parts, axis=0).astype(np.float64)


def _window_size(freq_minutes: int) -> int:
    """Pick a rolling window ~ 1 day of data (capped for very coarse datasets)."""
    if freq_minutes <= 0:
        return 24
    w = max(12, int(round(24 * 60 / freq_minutes)))
    return min(w, 256)


def _normalize_channel(x: np.ndarray) -> np.ndarray:
    mu, sigma = np.nanmean(x), np.nanstd(x) + 1e-8
    return (x - mu) / sigma


def _spectral_entropy(x: np.ndarray) -> float:
    x = x - np.mean(x)
    spec = np.abs(np.fft.rfft(x)) ** 2
    spec = spec[1:]  # drop DC
    if spec.sum() <= 0:
        return 0.0
    p = spec / spec.sum()
    h = -np.sum(p * np.log(p + 1e-12))
    return float(h / np.log(len(p)))  # normalized to [0, 1]


def _rolling_std(x: np.ndarray, w: int) -> np.ndarray:
    if len(x) <= w:
        return np.array([np.std(x)])
    x2 = np.convolve(x ** 2, np.ones(w) / w, mode="valid")
    x1 = np.convolve(x, np.ones(w) / w, mode="valid")
    var = np.maximum(x2 - x1 ** 2, 0.0)
    return np.sqrt(var)


def compute_stats(name: str, freq_minutes: int,
                  max_channels: int = 16,
                  adf_len: int = 2000) -> dict:
    data = _load_full(name)
    T, C = data.shape
    w = _window_size(freq_minutes)

    rng = np.random.default_rng(0)
    idx = rng.choice(C, size=min(max_channels, C), replace=False)
    sub = data[:, idx]

    adf_ps, spec_hs, vovs = [], [], []
    for c in range(sub.shape[1]):
        x = sub[:, c]
        if np.isnan(x).any():
            x = np.nan_to_num(x, nan=np.nanmean(x))
        xn = _normalize_channel(x)

        if _HAS_SM:
            xs = xn if len(xn) <= adf_len else xn[:: max(1, len(xn) // adf_len)][:adf_len]
            try:
                adf_ps.append(adfuller(xs, autolag=None, maxlag=10)[1])
            except Exception:
                pass

        spec_hs.append(_spectral_entropy(xn))

        r = _rolling_std(xn, w)
        if len(r) > 1 and r.mean() > 0:
            vovs.append(float(r.std() / r.mean()))

    return {
        "name": name,
        "T": int(T),
        "C": int(C),
        "window": w,
        "adf_p": float(np.mean(adf_ps)) if adf_ps else float("nan"),
        "spec_entropy": float(np.mean(spec_hs)),
        "vov": float(np.mean(vovs)) if vovs else float("nan"),
    }


def plot_panels(stats_list: list[dict]) -> None:
    rcParams.update({"font.size": 9, "pdf.fonttype": 42, "ps.fonttype": 42})
    n = len(DATASETS)
    ncols = 7
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.8, nrows * 2.1),
                             constrained_layout=True)
    axes = np.array(axes).reshape(-1)

    for ax, (name, domain, freq_str, freq_min), st in zip(axes, DATASETS, stats_list):
        data = _load_full(name)
        rng = np.random.default_rng(1)
        ch = int(rng.integers(0, data.shape[1]))
        x = data[:, ch]
        x = np.nan_to_num(x, nan=np.nanmean(x))
        xn = _normalize_channel(x)
        t = np.arange(len(xn))

        w = st["window"]
        ax.plot(t, xn, color="#2b6cb0", linewidth=0.6, alpha=0.8, label="signal (z)")
        if len(xn) > w:
            rs = _rolling_std(xn, w)
            tr = np.arange(len(rs)) + w // 2
            ax2 = ax.twinx()
            ax2.plot(tr, rs, color="#e53e3e", linewidth=0.9, alpha=0.9,
                     label=f"rolling std (w={w})")
            ax2.set_yticks([])
            ax2.spines["top"].set_visible(False)

        ax.set_title(
            f"{name}  ({domain}, {freq_str})\n"
            f"T={st['T']:,}  C={st['C']}  "
            f"ADF$p$={st['adf_p']:.2f}  $H_s$={st['spec_entropy']:.2f}  "
            f"VoV={st['vov']:.2f}",
            fontsize=8.5,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    for ax in axes[len(DATASETS):]:
        ax.axis("off")

    fig.suptitle(
        "Local heterogeneity across benchmark datasets: "
        "signal (blue) and rolling std (red) reveal regime shifts and volatility-of-volatility",
        fontsize=10,
    )
    fig.savefig(OUT_PDF, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to {OUT_PDF}")


def main() -> None:
    stats_list = [compute_stats(name, freq_min) for name, _, _, freq_min in DATASETS]

    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "domain", "frequency", "T", "C",
                         "ADF_p", "spectral_entropy", "VoV"])
        for (name, domain, freq_str, _), st in zip(DATASETS, stats_list):
            writer.writerow([name, domain, freq_str, st["T"], st["C"],
                             f"{st['adf_p']:.4f}", f"{st['spec_entropy']:.4f}",
                             f"{st['vov']:.4f}"])
    print(f"Saved stats to {OUT_CSV}")

    print(f"\n{'Dataset':20s} {'T':>8s} {'C':>5s} {'ADF p':>7s} {'H_s':>6s} {'VoV':>6s}")
    print("-" * 60)
    for (name, _, _, _), st in zip(DATASETS, stats_list):
        print(f"{name:20s} {st['T']:>8d} {st['C']:>5d} "
              f"{st['adf_p']:>7.3f} {st['spec_entropy']:>6.3f} {st['vov']:>6.3f}")

    plot_panels(stats_list)


if __name__ == "__main__":
    main()
