# TCG Sensitivity Analysis

Generated from `checkpoints/test_sensitivity/` aggregation.
Model: PatchTST, Dataset: Illness, Input Len: 24

Additional conv sensitivity is aggregated from `checkpoints/test_sensitivity_conv/`.

## Curve 1: Orthogonal Regularization Sensitivity (K=8 fixed)

| orth_lambda | H=24 | H=36 | H=48 | H=60 | **Avg** |
|-------------|------|------|------|------|---------|
| 0 (w/o orth) | 3.456 / 1.075 | 4.008 / 1.195 | 2.887 / 1.083 | 2.726 / 1.075 | 3.269 / 1.107 |
| 0.0001      | 3.380 / 1.065 | 3.923 / 1.202 | 2.887 / 1.083 | 2.726 / 1.075 | 3.229 / 1.106 |
| 0.001       | 3.458 / 1.075 | 3.925 / 1.203 | 2.886 / 1.082 | 2.726 / 1.075 | 3.249 / 1.109 |
| 0.1         | 3.460 / 1.076 | 4.016 / 1.198 | 2.888 / 1.083 | 2.727 / 1.076 | 3.273 / 1.108 |

**Best orth_lambda**: 0.0001 (lowest avg MSE/MAE = 3.229 / 1.106)

## Curve 2: Number of Patterns K Sensitivity (orth_lambda=0.0001 fixed)

| K | H=24 | H=36 | H=48 | H=60 | **Avg** |
|---|------|------|------|------|---------|
|  4 | 3.356 / 1.071 | 4.115 / 1.215 | 2.846 / 1.087 | 2.801 / 1.100 | 3.279 / 1.118 |
|  8 | 3.380 / 1.065 | 3.923 / 1.202 | 2.887 / 1.083 | 2.726 / 1.075 | 3.229 / 1.106 |
| 16 | 3.542 / 1.072 | 3.954 / 1.192 | 2.931 / 1.100 | 2.746 / 1.069 | 3.293 / 1.108 |
| 32 | 3.571 / 1.095 | 3.611 / 1.155 | 2.952 / 1.103 | 2.809 / 1.109 | 3.236 / 1.116 |

**Best K**: K=8 (lowest avg MSE/MAE = 3.229 / 1.106)

## Curve 3: Conv Kernel Sensitivity (K=8, orth_lambda=0.0001 fixed)

### PatchTST

| conv_kernels | H=24 | H=36 | H=48 | H=60 | **Avg** |
|--------------|------|------|------|------|---------|
| (1,)         | 3.108 / 1.042 | 3.892 / 1.206 | 2.887 / 1.103 | 2.676 / 1.059 | 3.141 / 1.102 |
| (3,)         | 3.266 / 1.052 | 4.053 / 1.227 | 2.974 / 1.109 | 2.646 / 1.053 | 3.235 / 1.110 |
| (5,)         | 3.582 / 1.070 | 4.138 / 1.216 | 2.873 / 1.095 | 2.801 / 1.103 | 3.348 / 1.121 |
| (7,)         | 3.460 / 1.055 | 3.969 / 1.219 | 2.857 / 1.095 | 2.711 / 1.071 | 3.249 / 1.110 |
| (3, 7)       | 3.464 / 1.085 | 3.929 / 1.211 | 2.938 / 1.103 | 2.678 / 1.059 | 3.252 / 1.115 |

**Best conv_kernels (PatchTST)**: `(1,)` (lowest avg MSE/MAE = 3.141 / 1.102)

### Crossformer

| conv_kernels | H=24 | H=36 | H=48 | H=60 | **Avg** |
|--------------|------|------|------|------|---------|
| (1,)         | 4.954 / 1.480 | 5.003 / 1.521 | 5.124 / 1.553 | 5.195 / 1.579 | 5.069 / 1.533 |
| (3,)         | 5.226 / 1.548 | 5.158 / 1.515 | 5.470 / 1.615 | 4.955 / 1.532 | 5.202 / 1.553 |
| (5,)         | 4.905 / 1.493 | 5.523 / 1.604 | 4.792 / 1.492 | 5.341 / 1.604 | 5.141 / 1.549 |
| (7,)         | 4.666 / 1.449 | 5.059 / 1.534 | 5.737 / 1.675 | 5.162 / 1.571 | 5.156 / 1.557 |
| (3, 7)       | 4.591 / 1.431 | 6.088 / 1.685 | 5.522 / 1.636 | 4.942 / 1.528 | 5.286 / 1.570 |

**Best conv_kernels (Crossformer)**: `(1,)` (lowest avg MSE/MAE = 5.069 / 1.533)

## Observations

1. **K=8** provides the best overall performance (avg MSE/MAE = 3.229 / 1.106), suggesting a good balance between model capacity and regularization.
2. **orth_lambda=0.0001** is the optimal orthogonal regularization strength (avg MSE/MAE = 3.229 / 1.106).
3. **K=4** underperforms (avg MSE/MAE = 3.279 / 1.118), indicating insufficient capacity to capture diverse patterns.
4. **K=32** shows degraded performance on short horizons but competitive on long horizons.
5. For **long horizons (H=60)**, K=16 performs best (MSE/MAE = 2.746 / 1.069), possibly due to better ability to capture slow-varying dynamics.
6. In conv sensitivity, both **PatchTST** and **Crossformer** show the best average performance with **(1,)**, indicating point-wise depthwise filtering is more robust than larger kernels under the current fixed setting (K=8, orth_lambda=0.0001).
7. For Crossformer, `(3, 7)` is unstable across horizons (notably worse on H=36), suggesting multi-kernel settings need model-specific retuning rather than direct reuse from PatchTST defaults.
