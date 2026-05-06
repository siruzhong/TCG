# Adaptive Feature Recalibration for Non-Stationary Time Series Forecasting

Official code for the work **“Adaptive Feature Recalibration for Non-Stationary Time Series Forecasting”**, centered on **Dynamic Pattern Routing (DPR)** and the minimalist forecaster **DPRNet**.

**Terminology.** **DPR** is implemented as [`TemporalContextualGating`](src/basicts/modules/dpr.py) (`Perceive → Route → Modulate`), configured via [`DPRConfig`](src/basicts/configs/dpr_config.py). **DPRNet** is the dedicated model stack in [`src/basicts/models/DPRNet/`](src/basicts/models/DPRNet/). The same DPR adapter can be attached to other backbones for plug-and-play experiments.

## Overview

Non-stationary series shift across regimes; standard backbones apply **globally shared** feature transforms to every token, which creates a **static mapping bottleneck** and a compromised average representation. **DPR** addresses this with **token-level feature recalibration**: local temporal context drives **soft routing** over a learned basis of dynamical motifs, then **residual Hadamard modulation** (with identity-style initialization so training starts near the backbone’s original map). An **orthogonal regularizer** on the basis discourages collapsed or redundant routing.

### Contributions (paper)

- **Mechanism:** DPR for continuous, regime-aware recalibration at the token level.
- **Architecture:** **DPRNet** — a deliberately simple patch-MLP stack where gains come from recalibration rather than macroscopic parameter scaling.
- **Plug-and-play:** The same DPR module lifts diverse backbones (attention, convolution, MLP-style) with small overhead.

## Installation

### Requirements

- Python ≥ 3.9  
- PyTorch (see your CUDA/CPU setup)  
- Dependencies in [`pyproject.toml`](pyproject.toml) / [`requirements.txt`](requirements.txt) if present  

### Install from source

```bash
git clone <repository-url> DPR
cd DPR
pip install -e .
```

The importable Python package is `basicts`.

## Quick start

### DPRNet

Train the minimalist DPRNet model:

```bash
python run_dprnet.py
```

Adjust dataset, horizons, and checkpoints inside [`run_dprnet.py`](run_dprnet.py) as needed.

### Plug-and-play DPR on standard backbones

[`run_baselines.py`](run_baselines.py) schedules **baseline vs. +DPR** runs over datasets and a hyperparameter grid (`num_patterns`, `orth_lambda`, `conv_kernels`). It uses multiprocessing and expects multiple GPUs; edit `AVAILABLE_GPUS` and `JOBS_PER_GPU` at the top of the file for your hardware.

```bash
python run_baselines.py
```

Patch-based models (`PatchTST`, `WPMixer`, `TimeFilter`) default to **pointwise** perception only (`conv_kernels = (1,)`) so multi-scale depthwise convs do not fight patch tokenization.

### Other experiment drivers (paper RQs)

| Script | Role |
|--------|------|
| [`run_rq2_scaling.py`](run_rq2_scaling.py) | Scaling vs. +DPR (and related configs) |
| [`run_rq3_ablation.py`](run_rq3_ablation.py) | Ablations (multiscale, orthogonality, routing, init, …) |
| [`run_rq4_visualization.py`](run_rq4_visualization.py) | Regime / routing visualizations |
| [`run_rq5_sensitivity.py`](run_rq5_sensitivity.py) | Sensitivity analysis |
| [`run_moe_vs_dpr.py`](run_moe_vs_dpr.py) | DPRNet vs. MoEDPRNet variants |
| [`run_baseline_raw.py`](run_baseline_raw.py) | Baseline runs without DPR (utility entry) |
| [`scripts/extract_scaling_costs.py`](scripts/extract_scaling_costs.py) | After RQ2: read `checkpoints/test_scaling` logs + compute Raw / +DPR params (optional `--flops` via `thop`) |

Run the scaling helper from the repo root: `python scripts/extract_scaling_costs.py [--flops]`.

## Method (implementation)

1. **Perceive:** Multi-scale **depthwise** 1D convolutions along the sequence (or `k=1` when multiscale is off), yielding context features under **channel independence**.
2. **Route:** MLP bottleneck → cosine **softmax** over `K` learnable centroids (basis size / `num_patterns` in [`DPRConfig`](src/basicts/configs/dpr_config.py)).
3. **Modulate:** Convex combination of `K` rows in the **modulation matrix**, then \(\mathbf{h} \odot (\mathbf{1} + \gamma \mathbf{m})\) with learnable \(\gamma\) (zero init for identity start).
4. **Train:** Forecasting loss + \(\lambda_{\mathrm{orth}}\) · orthogonal loss on the normalized basis (see `dpr_orthogonal_loss` in [`dpr.py`](src/basicts/modules/dpr.py)).

[`DPRConfig`](src/basicts/configs/dpr_config.py) exposes `num_patterns`, `orth_lambda`, `use_multiscale` / `conv_kernels`, `identity_init`, and `discrete_topk` (soft vs. hard routing).

## Datasets

Twelve real-world benchmarks (energy, finance, weather, health, epidemiology, etc.), aligned with the paper and [`run_baselines.py`](run_baselines.py): **ETTh1, ETTh2, ETTm1, ETTm2, Weather, Illness, ExchangeRate, BeijingAirQuality, COVID19, VIX, NABCPU, Sunspots**. Per-dataset input/horizon presets follow the paper’s protocol (see `DATASET_CONFIGS` in `run_baselines.py`).

For download and preprocessing, see [`datasets/README.md`](datasets/README.md) where applicable.

## Models

- **DPRNet:** [`src/basicts/models/DPRNet/`](src/basicts/models/DPRNet/)  
- **MoE variant:** [`MoEDPRNet`](src/basicts/models/MoEDPRNet/)  
- **Backbones in [`run_baselines.py`](run_baselines.py):** Informer, Crossformer, PatchTST, TimesNet, TimeMixer, TimeFilter, WPMixer, plus **DPRNet**  
- **Other forecasters** (e.g. iTransformer) live under [`src/basicts/models/`](src/basicts/models/) and can be wired into the same `DPRConfig` pattern for custom runs  

## Code structure

```
DPR/
├── src/basicts/
│   ├── modules/
│   │   └── dpr.py                 # TemporalContextualGating (DPR) + orth loss
│   ├── configs/dpr_config.py      # DPRConfig
│   ├── models/DPRNet/
│   ├── models/MoEDPRNet/          # MoE variant
│   └── runners/callback/          # Training callbacks (aux losses, early stopping, …)
├── run_dprnet.py
├── run_baselines.py               # Main plug-and-play + grid search driver
├── run_rq*.py                     # Paper RQ scripts
├── scripts/
│   ├── aggregate_results.py       # Refresh dpr_result.md from checkpoints/
│   ├── extract_scaling_costs.py   # RQ2 param / FLOP extraction
│   └── data_preparation/          # Dataset build helpers
├── vis/                           # Figures / plotting helpers
└── datasets/                      # Dataset notes and paths
```

## Results

Empirical claims (DPRNet against strong baselines, consistent plug-in gains from DPR, scaling vs. recalibration, ablations) are reported in the paper and reproduced with the scripts above.

After training, you can regenerate the aggregated metric table in [`dpr_result.md`](dpr_result.md) from `checkpoints/` with:

```bash
python scripts/aggregate_results.py        # or: --dry-run
```

## License

This project is licensed under the Apache License 2.0 — see the [LICENSE](LICENSE) file.

## Acknowledgments

Built on an open time-series forecasting codebase; we thank the community for the foundations this work extends.
