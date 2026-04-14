# TCG Ablation Study

Generated from `checkpoints/test_ablation/` aggregation. Raw results from `tcg_result.md`.

## PatchTST

| dataset      | horizon | raw      | full_tcg | wo_multi_scale | wo_orth | wo_identity | discrete_top2 |
| ------------ | ------- | -------- | -------- | -------------- | ------- | ---------- | -------------- |
| Illness      | 24      | 3.633 / 1.079 | 3.352 / 1.070 | 3.083 / 1.041 | 3.456 / 1.075 | 3.459 / 1.103 | 3.463 / 1.076 |
| ExchangeRate | 96      | 0.106 / 0.229 | 0.104 / 0.226 | 0.107 / 0.229 | 0.109 / 0.231 | 0.108 / 0.230 | 0.109 / 0.231 |

## TimeMixer

| dataset      | horizon | raw      | full_tcg | wo_multi_scale | wo_orth | wo_identity | discrete_top2 |
| ------------ | ------- | -------- | -------- | -------------- | ------- | ---------- | -------------- |
| Illness      | 24      | 3.124 / 1.136 | 3.144 / 1.144 | 3.172 / 1.153 | 3.160 / 1.146 | 3.127 / 1.133 | 3.159 / 1.146 |
| ExchangeRate | 96      | 0.102 / 0.224 | 0.101 / 0.224 | 0.101 / 0.223 | 0.101 / 0.224 | 0.101 / 0.224 | 0.101 / 0.224 |

## Informer

| dataset      | horizon | raw      | full_tcg | wo_multi_scale | wo_orth | wo_identity | discrete_top2 |
| ------------ | ------- | -------- | -------- | -------------- | ------- | ---------- | -------------- |
| Illness      | 24      | 7.005 / 1.868 | 6.336 / 1.749 | 6.658 / 1.798 | 6.046 / 1.668 | 6.679 / 1.824 | 6.190 / 1.714 |
| ExchangeRate | 96      | 2.295 / 1.093 | 2.306 / 1.155 | 3.187 / 1.328 | 3.024 / 1.282 | 2.823 / 1.239 | 3.062 / 1.285 |

## Crossformer

| dataset      | horizon | raw      | full_tcg | wo_multi_scale | wo_orth | wo_identity | discrete_top2 |
| ------------ | ------- | -------- | -------- | -------------- | ------- | ---------- | -------------- |
| Illness      | 24      | 4.736 / 1.480 | 4.593 / 1.428 | 5.129 / 1.546 | 4.661 / 1.448 | 5.062 / 1.525 | 4.721 / 1.458 |
| ExchangeRate | 96      | 0.269 / 0.349 | 0.237 / 0.335 | 0.266 / 0.346 | 0.242 / 0.338 | 0.279 / 0.367 | 0.238 / 0.335 |

## Observations

1. **raw** = model without TCG (from `tcg_result.md`)
2. **full_tcg** = model with full TCG (all components enabled, from `tcg_result.md`)
3. **wo_multi_scale** (k=1 instead of k1=3, k2=7): Mixed results.
3. **wo_orth** (orthogonal regularization = 0): Mixed results.
4. **wo_identity** (gamma ~ N(0, 0.01) instead of 0): Small impact in most cases.
5. **discrete_top2** (hard Top-2 routing): Similar to soft routing in most cases.
