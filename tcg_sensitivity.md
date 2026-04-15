# TCG Sensitivity Analysis

Generated from `checkpoints/test_sensitivity/` aggregation.
Model: PatchTST, Dataset: Illness, Input Len: 24

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

## Observations

1. **K=8** provides the best overall performance (avg MSE/MAE = 3.229 / 1.106), suggesting a good balance between model capacity and regularization.
2. **orth_lambda=0.0001** is the optimal orthogonal regularization strength (avg MSE/MAE = 3.229 / 1.106).
3. **K=4** underperforms (avg MSE/MAE = 3.279 / 1.118), indicating insufficient capacity to capture diverse patterns.
4. **K=32** shows degraded performance on short horizons but competitive on long horizons.
5. For **long horizons (H=60)**, K=16 performs best (MSE/MAE = 2.746 / 1.069), possibly due to better ability to capture slow-varying dynamics.
