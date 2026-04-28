# I-SPY2 durvalumab+olaparib treatment × TME interaction

- N = 100 (control=29, durva+olap=71)
- HER2- only; HR+: 62 / HR-: 38
- Baseline pCR rate: control 24.1%, durva+olap 40.8% (absolute Δ = +16.7%)
- Baseline OR (Fisher) = 2.170, p = 0.1679

## Interaction results (per signature)

| Signature | n | β_int | OR_int | p | q (BH) |
|---|---|---|---|---|---|
| Mitotic_sig | 100 | 1.390 | 4.013 | 0.0308 | 0.4309 |
| TAMsurr_TcClassII_ratio_sig | 100 | 0.528 | 1.696 | 0.3184 | 0.9580 |
| PDL1 | 100 | -0.598 | 0.550 | 0.3293 | 0.9580 |
| Mast.cells_sig | 100 | -0.532 | 0.588 | 0.3373 | 0.9580 |
| STAT1_sig | 100 | 0.434 | 1.544 | 0.4658 | 0.9580 |
| PARPi7_sig. | 100 | 0.220 | 1.246 | 0.6887 | 0.9580 |
| SET.index | 99 | -0.497 | 0.608 | 0.6977 | 0.9580 |
| B.cells_sig | 100 | -0.236 | 0.790 | 0.7022 | 0.9580 |
| CD68 | 100 | -0.232 | 0.793 | 0.7160 | 0.9580 |
| T.cells_sig | 100 | -0.173 | 0.841 | 0.7602 | 0.9580 |
| Dendritic.cells_sig | 100 | 0.117 | 1.124 | 0.8221 | 0.9580 |
| PD1 | 100 | 0.079 | 1.082 | 0.8893 | 0.9580 |
| ESR1_PGR_ave | 100 | 0.130 | 1.139 | 0.8896 | 0.9580 |
| TIS_sig | 100 | -0.023 | 0.977 | 0.9691 | 0.9691 |

## Subgroup ORR (median-dichotomized, interaction p < 0.10)

| Signature | Low: ctrl → trt (ARR) | High: ctrl → trt (ARR) | Δ ARR (high−low) |
|---|---|---|---|
| Mitotic_sig | 15.4% → 29.7% (+14.3%) | 31.2% → 52.9% (+21.7%) | **+0.073** |

## Verdict

🟡 **Nominally significant (before multiple testing)**: min interaction p = 0.0308, but does not survive BH correction (min q = 0.4309). Treat as hypothesis-generating; replicate in CALGB 40601 / NeoALTTO / larger I-SPY2.