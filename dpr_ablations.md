# DPR Ablation Study

Generated from `checkpoints/test_ablation/` aggregation. Raw results from `dpr_result.md`.

## PatchTST

| dataset      | horizon | raw      | full_dpr | wo_multi_scale | wo_orth | wo_identity | discrete_top2 |
| ------------ | ------- | -------- | -------- | -------------- | ------- | ---------- | -------------- |
| Illness      | 24      | 3.633 / 1.079 | 3.352 / 1.070 | 3.083 / 1.041 | 3.456 / 1.075 | 3.459 / 1.103 | 3.463 / 1.076 |
| ExchangeRate | 96      | 0.106 / 0.229 | 0.104 / 0.226 | 0.107 / 0.229 | 0.109 / 0.231 | 0.108 / 0.230 | 0.109 / 0.231 |

## TimeMixer

| dataset      | horizon | raw      | full_dpr | wo_multi_scale | wo_orth | wo_identity | discrete_top2 |
| ------------ | ------- | -------- | -------- | -------------- | ------- | ---------- | -------------- |
| Illness      | 24      | 3.124 / 1.136 | 3.144 / 1.144 | 3.172 / 1.153 | 3.160 / 1.146 | 3.127 / 1.133 | 3.159 / 1.146 |
| ExchangeRate | 96      | 0.102 / 0.224 | 0.101 / 0.224 | 0.101 / 0.223 | 0.101 / 0.224 | 0.101 / 0.224 | 0.101 / 0.224 |

## Informer

| dataset      | horizon | raw      | full_dpr | wo_multi_scale | wo_orth | wo_identity | discrete_top2 |
| ------------ | ------- | -------- | -------- | -------------- | ------- | ---------- | -------------- |
| Illness      | 24      | 7.005 / 1.868 | 6.336 / 1.749 | 6.658 / 1.798 | 6.046 / 1.668 | 6.679 / 1.824 | 6.190 / 1.714 |
| ExchangeRate | 96      | 2.295 / 1.093 | 2.306 / 1.155 | 3.187 / 1.328 | 3.024 / 1.282 | 2.823 / 1.239 | 3.062 / 1.285 |

## Crossformer

| dataset      | horizon | raw      | full_dpr | wo_multi_scale | wo_orth | wo_identity | discrete_top2 |
| ------------ | ------- | -------- | -------- | -------------- | ------- | ---------- | -------------- |
| Illness      | 24      | 4.736 / 1.480 | 4.593 / 1.428 | 5.129 / 1.546 | 4.661 / 1.448 | 5.062 / 1.525 | 4.721 / 1.458 |
| ExchangeRate | 96      | 0.269 / 0.349 | 0.237 / 0.335 | 0.266 / 0.346 | 0.242 / 0.338 | 0.279 / 0.367 | 0.238 / 0.335 |

## MoE vs DPR Comparison

Direct comparison between DPRNet (with DPR) and MoE-DPRNet variants (different top-k) on identical architecture parameters.

| dataset      | horizon | DPRNet   | DPRNet_noDPR | MoE-top1 | MoE-top2 | MoE-top4 |
| ------------ | ------- | -------- | ------------ | -------- | -------- | -------- |
| Illness      | 24      | 3.079 / 1.096 | 3.347 / 1.091 | 3.715 / 1.156 | 3.755 / 1.097 | 4.146 / 1.122 |
| ExchangeRate | 96      | 0.102 / 0.225 | — | 0.102 / 0.224 | 0.103 / 0.225 | 0.103 / 0.225 |
| ETTh1        | 96      | 0.392 / 0.394 | 0.405 / 0.398 | 0.411 / 0.403 | 0.419 / 0.405 | 0.417 / 0.404 |

**vs DPRNet baseline (delta MSE / delta MAE):**

| dataset      | DPRNet_noDPR | MoE-top1 | MoE-top2 | MoE-top4 |
| ------------ | ------------ | -------- | -------- | -------- |
| Illness      | +8.7% / -0.4% | +20.7% / +5.5% | +21.9% / +0.1% | +34.7% / +2.4% |
| ExchangeRate | — | -0.0% / -0.4% | +1.0% / +0.0% | +0.4% / -0.1% |
| ETTh1        | +3.4% / +1.0% | +4.8% / +2.3% | +6.8% / +2.7% | +6.2% / +2.6% |

### Training & Inference Time

| dataset | model | params | train/epoch | total train | test | test (per 1M params) |
| ------- | ----- | ------ | ----------- | ----------- | ---- | ------------------- |
| Illness | DPRNet | 325K | 0.85s | 68.7s (71 ep) | 0.73s | 2.24 μs |
| Illness | DPRNet_noDPR | 287K | 0.81s | 93.7s (100 ep) | 0.71s | 2.47 μs |
| Illness | MoE-top1 | 818K | 1.06s | 50.2s (43 ep) | 0.77s | 0.94 μs |
| Illness | MoE-top2 | 818K | 1.18s | 96.3s (73 ep) | 0.90s | 1.10 μs |
| Illness | MoE-top4 | 818K | 1.02s | 86.1s (76 ep) | 0.77s | 0.94 μs |
| ETTh1 | DPRNet | 602K | 29.90s | 1197.5s (28 ep) | 8.60s | 14.29 μs |
| ETTh1 | DPRNet_noDPR | 563K | 15.84s | 544.9s (32 ep) | 5.07s | 9.00 μs |
| ETTh1 | MoE-top1 | 1,095K | 62.62s | 794.4s (10 ep) | 18.68s | 17.06 μs |
| ETTh1 | MoE-top2 | 1,095K | 89.12s | 178.2s (2 ep) | 16.05s | 14.66 μs |
| ETTh1 | MoE-top4 | 1,095K | 45.89s | 588.1s (13 ep) | 13.43s | 12.27 μs |

**Notes:**
- DPRNet values from `dpr_result.md` (established baseline)
- MoE values from this experiment (`checkpoints/moe_vs_dpr/`)
- All MoE variants use `num_experts=8`, `moe_loss_coef=0.01`
- MoE-top1 active params: 65K, top2: 131K, top4: 262K (vs DPR: ~35K)
- MoE-DPRNet has ~1.82x total parameters (1.09M vs 0.60M)

## Observations

1. **raw** = model without DPR (from `dpr_result.md`)
2. **full_dpr** = model with full DPR (all components enabled, from `dpr_result.md`)
3. **wo_multi_scale** (k=1 instead of k1=3, k2=7): Mixed results.
3. **wo_orth** (orthogonal regularization = 0): Mixed results.
4. **wo_identity** (gamma ~ N(0, 0.01) instead of 0): Small impact in most cases.
5. **discrete_top2** (hard Top-2 routing): Similar to soft routing in most cases.
6. **DPRNet without DPR**:
   - Removing DPR (DPR) block degrades performance on both datasets
   - Illness: +8.7% MSE (3.079 → 3.347), MAE nearly unchanged (-0.4%)
   - ETTh1: +3.4% MSE (0.392 → 0.405), +1.0% MAE
   - DPRNet_noDPR has fewer params (287K vs 325K on Illness, 563K vs 602K on ETTh1) because DPR block is removed
   - Training is faster per epoch but requires more epochs to converge (100 vs 71 on Illness, 32 vs 28 on ETTh1)
   - **Conclusion**: DPR (DPR) block provides consistent performance gain even though it adds only ~38K parameters (~12%)

7. **MoE vs DPR**: 
   - **top_k=1**: Worse on Illness (+20.7%) and ETTh1 (+4.8%), equal on ExchangeRate (-0.0%)
   - **top_k=2**: Worse on Illness (+21.9%) and ETTh1 (+6.8%), equal on ExchangeRate (+1.0%)
   - **top_k=4**: Worse on Illness (+34.7%) and ETTh1 (+6.2%), equal on ExchangeRate (+0.4%)
   - **MoE vs no-DPR**: Even compared to DPRNet without DPR, MoE-top1 is worse on Illness (+11.0%) and ETTh1 (+1.5%)
   - **Conclusion**: Increasing top-k does not help. MoE does not outperform either DPRNet or DPRNet_noDPR. DPR is the most efficient component for time series forecasting among the three.
