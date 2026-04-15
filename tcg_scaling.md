# TCG Wide Scaling Study

Generated from `checkpoints/test_scaling/` aggregation. Base results from `tcg_result.md`.

## Configurations

- **base**: hidden=256, intermediate=1024, num_layers=1 (original)
- **w2_d1**: hidden=512, intermediate=2048, num_layers=1 (2x width only)
- **w1_d2**: hidden=256, intermediate=1024, num_layers=2 (2x depth only)
- **w2_d2**: hidden=512, intermediate=2048, num_layers=2 (2x width + depth)

## PatchTST

| dataset      | horizon | base_raw | base_tcg | w2_d1     | w1_d2     | w2_d2     |
| ------------ | ------- | -------- | -------- | --------- | --------- | --------- |
| ETTh1        | 96      | 0.394 / 0.392 | 0.394 / 0.392 | 0.405 / 0.397 | 0.401 / 0.395 | 0.398 / 0.394 |
| Illness      | 24      | 3.633 / 1.079 | 3.352 / 1.070 | 3.415 / 1.073 | 3.199 / 1.041 | 3.094 / 1.014 |
| ExchangeRate | 96      | 0.106 / 0.229 | 0.104 / 0.226 | 0.110 / 0.232 | 0.110 / 0.231 | 0.111 / 0.233 |

## TimesNet

| dataset      | horizon | base_raw | base_tcg | w2_d1     | w1_d2     | w2_d2     |
| ------------ | ------- | -------- | -------- | --------- | --------- | --------- |
| ETTh1        | 96      | 0.488 / 0.475 | 0.475 / 0.466 | 0.561 / 0.509 | 0.503 / 0.481 | 0.598 / 0.518 |
| Illness      | 24      | 9.241 / 1.389 | 6.769 / 1.355 | 5.859 / 1.250 | 6.827 / 1.242 | 3.515 / 1.081 |
| ExchangeRate | 96      | 0.137 / 0.266 | 0.128 / 0.260 | 0.132 / 0.263 | 0.145 / 0.277 | 0.138 / 0.270 |

## TimeMixer

| dataset      | horizon | base_raw | base_tcg | w2_d1     | w1_d2     | w2_d2     |
| ------------ | ------- | -------- | -------- | --------- | --------- | --------- |
| ETTh1        | 96      | 0.401 / 0.395 | 0.397 / 0.393 | 0.388 / 0.391 | 0.407 / 0.399 | 0.396 / 0.393 |
| Illness      | 24      | 3.124 / 1.136 | 3.144 / 1.144 | 3.188 / 1.169 | 3.333 / 1.148 | 3.278 / 1.172 |
| ExchangeRate | 96      | 0.102 / 0.224 | 0.101 / 0.224 | 0.103 / 0.224 | 0.101 / 0.224 | 0.103 / 0.225 |

## Observations

1. **base_raw** = model without TCG, base config (from `tcg_result.md`)
2. **base_tcg** = model with TCG, base config (from `tcg_result.md`)
3. **w2_d1** (2x width): Performance varies by dataset; sometimes improves, sometimes degrades
4. **w1_d2** (2x depth): Mixed results; depth scaling shows limited benefit
5. **w2_d2** (2x width + depth): Generally similar to width-only scaling, not compounding benefits
