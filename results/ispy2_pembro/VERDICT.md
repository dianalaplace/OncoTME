# I-SPY2 TME × Pembrolizumab interaction

- Arms: **Paclitaxel** (N=179) vs **Paclitaxel + Pembrolizumab** (N=69)
- pCR: ctrl 17.3%, trt 44.9% (Δ = +27.6%)
- Baseline OR = 3.895, Fisher p = 0.0000

## Top 10 TME × arm interactions (BH-adjusted)

| Signature | n | OR_int | p | q |
|---|---|---|---|---|
| sig__CAF_proxy | 248 | 0.418 | 0.0099 | 0.2269 |
| sig__IFN_gamma | 248 | 2.895 | 0.0348 | 0.3172 |
| sig__HLA_I_score | 248 | 2.368 | 0.0414 | 0.3172 |
| sig__checkpoint_score | 248 | 2.198 | 0.0703 | 0.3704 |
| sig__B_cell | 248 | 1.882 | 0.0840 | 0.3704 |
| sig__NK_cell | 248 | 1.777 | 0.1335 | 0.3704 |
| sig__TLS_signature | 248 | 1.843 | 0.1373 | 0.3704 |
| sig__TGFb_activity | 248 | 0.613 | 0.1389 | 0.3704 |
| sig__cytolytic_score | 248 | 1.807 | 0.1449 | 0.3704 |
| sig__IL6_STAT3_axis | 248 | 1.675 | 0.1801 | 0.3920 |

## Subgroup benefit (median-dichotomized for p<0.10 signatures)

| Signature | Low ctrl→trt (ARR) | High ctrl→trt (ARR) | Δ ARR |
|---|---|---|---|
| sig__IFN_gamma | 10.5%→17.2% (+6.7%) | 25.0%→65.0% (+40.0%) | **+0.333** |
| sig__checkpoint_score | 10.0%→23.5% (+13.5%) | 24.7%→65.7% (+41.0%) | **+0.275** |
| sig__B_cell | 13.3%→29.4% (+16.1%) | 21.3%→60.0% (+38.7%) | **+0.226** |
| sig__HLA_I_score | 12.6%→27.6% (+15.0%) | 22.6%→57.5% (+34.9%) | **+0.199** |
| sig__CAF_proxy | 20.2%→57.1% (+36.9%) | 14.4%→32.4% (+17.9%) | **-0.190** |

## Verdict

🟡 **Strong hypothesis-generating signal**: min p = 0.0099 (signature = sig__CAF_proxy, OR_int=0.42). Does not survive BH with 23 tests; replicate on KEYNOTE-522, TONIC, or other pembro cohorts.