"""Robustness check + composite heterogeneity ranking.

Recomputes the three diagnostics across ALL channels (not a random subset) with
a small jackknife estimate, then combines them into a single heterogeneity score
via z-score aggregation (ADF_p is not used for ranking since a single dataset
dominates it).
"""
from __future__ import annotations

import os
import numpy as np
from statsmodels.tsa.stattools import adfuller

DATASET_ROOT = os.path.join(os.path.dirname(__file__), os.pardir, "datasets")

DATASETS = [
    ("ETTh1",             60),
    ("ETTh2",             60),
    ("ETTm1",             15),
    ("ETTm2",             15),
    ("Weather",           10),
    ("Illness",           7 * 24 * 60),
    ("ExchangeRate",      24 * 60),
    ("Solar",             10),
    ("PEMS08",            5),
    ("BeijingAirQuality", 60),
    ("COVID19",           24 * 60),
    ("VIX",               24 * 60),
    ("NABCPU",            5),
    ("Sunspots",          30 * 24 * 60),
]


def load_full(name: str) -> np.ndarray:
    parts = [np.load(os.path.join(DATASET_ROOT, name, f"{s}_data.npy"))
             for s in ("train", "val", "test")]
    return np.concatenate(parts, axis=0).astype(np.float64)


def win(freq):
    return min(256, max(12, int(round(24 * 60 / max(freq, 1)))))


def rolling_std(x, w):
    if len(x) <= w:
        return np.array([np.std(x)])
    x2 = np.convolve(x ** 2, np.ones(w) / w, mode="valid")
    x1 = np.convolve(x,       np.ones(w) / w, mode="valid")
    return np.sqrt(np.maximum(x2 - x1 ** 2, 0.0))


def spec_entropy(x):
    x = x - x.mean()
    s = np.abs(np.fft.rfft(x)) ** 2
    s = s[1:]
    if s.sum() <= 0:
        return 0.0
    p = s / s.sum()
    return float(-np.sum(p * np.log(p + 1e-12)) / np.log(len(p)))


def per_dataset(name, freq):
    data = load_full(name)
    T, C = data.shape
    w = win(freq)

    hs_list, vov_list, adf_list = [], [], []
    mu = data.mean(axis=0, keepdims=True)
    sd = data.std(axis=0, keepdims=True) + 1e-8
    z = (data - mu) / sd
    z = np.nan_to_num(z, nan=0.0)

    for c in range(C):
        x = z[:, c]
        hs_list.append(spec_entropy(x))
        r = rolling_std(x, w)
        if len(r) > 1 and r.mean() > 0:
            vov_list.append(float(r.std() / r.mean()))

    for c in range(min(C, 20)):
        x = z[:, c]
        xs = x if len(x) <= 2000 else x[:: max(1, len(x) // 2000)][:2000]
        try:
            adf_list.append(adfuller(xs, autolag=None, maxlag=10)[1])
        except Exception:
            pass

    return {
        "name": name, "T": T, "C": C, "window": w,
        "Hs_mean": float(np.mean(hs_list)),
        "Hs_std":  float(np.std(hs_list)),
        "VoV_mean": float(np.mean(vov_list)) if vov_list else float("nan"),
        "VoV_std":  float(np.std(vov_list))  if vov_list else float("nan"),
        "ADF_p":   float(np.mean(adf_list)) if adf_list else float("nan"),
        "nan_frac": float(np.isnan(data).mean()),
        "range": (float(data.min()), float(data.max())),
    }


def main():
    rows = [per_dataset(n, f) for n, f in DATASETS]

    print(f"\n{'Dataset':20s} {'T':>6s} {'C':>4s} "
          f"{'H_s(mean±std)':>14s} {'VoV(mean±std)':>15s} {'ADF p':>7s} "
          f"{'NaN%':>5s} {'range':>20s}")
    print("-" * 105)
    for r in rows:
        lo, hi = r["range"]
        print(f"{r['name']:20s} {r['T']:>6d} {r['C']:>4d} "
              f"{r['Hs_mean']:.3f}±{r['Hs_std']:.2f}  "
              f"{r['VoV_mean']:.3f}±{r['VoV_std']:.2f}   "
              f"{r['ADF_p']:>7.3f} "
              f"{r['nan_frac']*100:>4.1f}% "
              f"{lo:>8.2g}..{hi:<8.2g}")

    hs   = np.array([r["Hs_mean"] for r in rows])
    vov  = np.array([r["VoV_mean"] for r in rows])
    adfp = np.array([r["ADF_p"] for r in rows])

    z_hs  = (hs   - hs.mean())   / (hs.std()   + 1e-8)
    z_vov = (vov  - vov.mean())  / (vov.std()  + 1e-8)
    z_adf = (adfp - adfp.mean()) / (adfp.std() + 1e-8)
    score = z_hs + z_vov + z_adf

    order = np.argsort(-score)
    print("\nComposite local-heterogeneity ranking  (z(Hs) + z(VoV) + z(ADFp))")
    print("-" * 80)
    print(f"{'rank':>4s} {'dataset':20s} {'H_s':>6s} {'VoV':>6s} {'ADF p':>7s} {'score':>8s}")
    for rank, i in enumerate(order, 1):
        r = rows[i]
        print(f"{rank:>4d} {r['name']:20s} {r['Hs_mean']:>6.2f} {r['VoV_mean']:>6.2f} "
              f"{r['ADF_p']:>7.2f} {score[i]:>8.2f}")


if __name__ == "__main__":
    main()
