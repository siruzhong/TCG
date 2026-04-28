# TCG Wide Scaling Study

Generated from `checkpoints/test_scaling/` aggregation. Base results from `tcg_result.md`.

## Configurations

- **base**: hidden=256, intermediate=1024, num_layers=1 (original)
- **w2_d1**: hidden=512, intermediate=2048, num_layers=1 (2x width only)
- **w1_d2**: hidden=256, intermediate=1024, num_layers=2 (2x depth only)
- **w2_d2**: hidden=512, intermediate=2048, num_layers=2 (2x width + depth)
- **param_match**: width auto-tuned so base (no TCG) params ≈ base_tcg params; skipped if deviation >50%

## PatchTST

| dataset      | horizon | base_raw | base_tcg | w2_d1     | w1_d2     | w2_d2     | param_match |
| ------------ | ------- | -------- | -------- | --------- | --------- | --------- | ----------- |
| ETTh1        | 96      | 0.394 / 0.392 | 0.394 / 0.392 | 0.405 / 0.397 | 0.401 / 0.395 | 0.398 / 0.394 | 0.394 / 0.392 |
| Illness      | 24      | 3.633 / 1.079 | 3.108 / 1.042 | 3.415 / 1.073 | 3.199 / 1.041 | 3.094 / 1.014 | 3.633 / 1.079 |
| ExchangeRate | 96      | 0.106 / 0.229 | 0.104 / 0.226 | 0.110 / 0.232 | 0.110 / 0.231 | 0.111 / 0.233 | 0.108 / 0.231 |

## TimesNet

| dataset      | horizon | base_raw | base_tcg | w2_d1     | w1_d2     | w2_d2     | param_match |
| ------------ | ------- | -------- | -------- | --------- | --------- | --------- | ----------- |
| ETTh1        | 96      | 0.488 / 0.475 | 0.475 / 0.466 | 0.561 / 0.509 | 0.503 / 0.481 | 0.598 / 0.518 | running |
| Illness      | 24      | 9.241 / 1.389 | 6.769 / 1.355 | 5.859 / 1.250 | 6.827 / 1.242 | 3.515 / 1.081 | 9.345 / 1.392 |
| ExchangeRate | 96      | 0.137 / 0.266 | 0.128 / 0.260 | 0.132 / 0.263 | 0.145 / 0.277 | 0.138 / 0.270 | 0.126 / 0.257 |

## TimeMixer

| dataset      | horizon | base_raw | base_tcg | w2_d1     | w1_d2     | w2_d2     | param_match |
| ------------ | ------- | -------- | -------- | --------- | --------- | --------- | ----------- |
| ETTh1        | 96      | 0.401 / 0.395 | 0.397 / 0.393 | 0.388 / 0.391 | 0.407 / 0.399 | 0.396 / 0.393 | - |
| Illness      | 24      | 3.124 / 1.136 | 3.123 / 1.142 | 3.188 / 1.169 | 3.333 / 1.148 | 3.278 / 1.172 | - |
| ExchangeRate | 96      | 0.102 / 0.224 | 0.101 / 0.224 | 0.103 / 0.224 | 0.101 / 0.224 | 0.103 / 0.225 | - |

## WPMixer

| dataset      | horizon | base_raw | base_tcg | w2_d1     | w1_d2     | w2_d2     | param_match |
| ------------ | ------- | -------- | -------- | --------- | --------- | --------- | ----------- |
| ETTh1        | 96      | 0.382 / 0.388 | 0.381 / 0.387 | 0.382 / 0.390 | 0.382 / 0.389 | 0.382 / 0.390 | - |
| Illness      | 24      | 3.173 / 1.022 | 2.796 / 1.046 | 2.818 / 0.981 | 2.601 / 0.977 | 2.818 / 0.981 | - |
| ExchangeRate | 96      | 0.102 / 0.224 | 0.102 / 0.223 | 0.110 / 0.230 | 0.106 / 0.227 | 0.110 / 0.230 | - |

## TimeFilter

| dataset      | horizon | base_raw | base_tcg | w2_d1     | w1_d2     | w2_d2     | param_match |
| ------------ | ------- | -------- | -------- | --------- | --------- | --------- | ----------- |
| ETTh1        | 96      | 0.390 / 0.390 | 0.391 / 0.390 | 0.399 / 0.397 | 0.397 / 0.395 | 0.403 / 0.398 | 0.391 / 0.390 |
| Illness      | 24      | 1.991 / 0.873 | 1.821 / 0.848 | 3.264 / 0.975 | 3.285 / 0.913 | 2.656 / 0.960 | 2.808 / 0.986 |
| ExchangeRate | 96      | 0.107 / 0.230 | 0.103 / 0.226 | 0.106 / 0.229 | 0.104 / 0.229 | 0.112 / 0.237 | 0.105 / 0.228 |

## Observations

1. **base_raw** = model without TCG, base config (from `tcg_result.md`)
2. **base_tcg** = model with TCG, base config (from `tcg_result.md`)
3. **param_match** = base model sans TCG, width auto-tuned to equal base_tcg's parameter count (fair budget comparison). For models where TCG adds negligible params (e.g. PatchTST: 18.5M → 18.5M), param_match ≈ base_raw by design.
4. **w2_d1** (2x width): Performance is dataset/model dependent; gains are not consistent.
5. **w1_d2** (2x depth): Usually limited benefit, and can be worse than base_tcg.
6. **w2_d2** (2x width + depth): Often does not compound gains; some models regress.
7. **WPMixer note**: due to config mapping (`hidden_size/intermediate_size/num_layers -> d_model/tfactor/dfactor`), `w2_d1` and `w2_d2` collapse to the same effective config, so their numbers are identical.
8. **param_match vs base_tcg (fair comparison)**:
   - PatchTST: param_match ≈ base_raw ≈ base_tcg on all datasets; TCG doesn't help but also doesn't hurt.
   - TimesNet: On ExchangeRate, param_match (0.126) ≈ base_tcg (0.128) — at equal params, TCG roughly ties with scaling. On Illness, base_tcg (6.77) ≫ param_match (9.35) — TCG substantially outperforms pure width scaling.
   - TimeFilter: On ETTh1 and ExchangeRate, param_match ≈ base_tcg ≈ base_raw. On Illness, base_tcg (1.82) beats param_match (2.81).
