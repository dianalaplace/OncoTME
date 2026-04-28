# I-SPY2 targeted therapy × TME composite score spectrum

## Composite definition (pre-specified ICI biology)

```
composite = z(IFN_gamma) + z(HLA_I_score) + z(B_cell) + z(checkpoint_score)
           - z(CAF_proxy)
```

## Treatment arm × composite interaction (primary hypothesis per arm)

| Arm | Biology | n_ctrl / n_trt | pCR_ctrl → pCR_trt | OR_interaction [95% CI] | p_int |
|---|---|---|---|---|---|
| **Pembrolizumab** | anti-PD1 | 179 / 69 | 17.3% → 44.9% | 2.97 [1.19, 7.43] | **0.0197** |
| **AMG-386** | anti-Angiopoietin-1/2 | 179 / 115 | 17.3% → 28.7% | 1.56 [0.82, 2.99] | **0.1763** |
| **Ganitumab** | anti-IGF-1R mAb | 179 / 105 | 17.3% → 22.9% | 1.44 [0.75, 2.75] | **0.2758** |
| **ABT888+Carboplatin** | PARPi+Platinum | 179 / 71 | 17.3% → 38.0% | 1.28 [0.65, 2.52] | **0.4741** |
| **Neratinib** | pan-HER TKI | 179 / 114 | 17.3% → 36.8% | 1.22 [0.68, 2.20] | **0.5047** |
| **Trastuzumab-based** | HER2 mAb combos | 179 / 128 | 17.3% → 44.5% | 1.18 [0.66, 2.10] | **0.5704** |
| **Ganetespib** | HSP90 inhibitor | 179 / 93 | 17.3% → 28.0% | 1.19 [0.62, 2.30] | **0.6054** |
| **T-DM1+Pertuzumab** | HER2 ADC + mAb | 179 / 52 | 17.3% → 57.7% | 1.06 [0.48, 2.33] | **0.8915** |
| **MK-2206** | AKT inhibitor | 179 / 60 | 17.3% → 30.0% | 1.01 [0.48, 2.12] | **0.9864** |

## Key clinical take-aways

1. **Strongest composite × therapy interaction**: Pembrolizumab (anti-PD1), OR_int = 2.97, p = 0.0197.
2. **ICI-specificity**: Pembrolizumab interaction p = 0.0197; next-most-significant non-ICI arm has p = 0.1763.

## Arm-specific dominant TME signatures (top-1 each)

| Arm | Top signature | OR_int | p | q_global (BH across all tests) |
|---|---|---|---|---|
| Pembrolizumab | CAF_proxy | 0.42 | 0.0099 | 0.9535 |
| AMG-386 | M2_polarization | 2.19 | 0.0365 | 0.9535 |
| Ganitumab | CAF_proxy | 0.50 | 0.0226 | 0.9535 |
| ABT888+Carboplatin | IL6_STAT3_axis | 2.34 | 0.0387 | 0.9535 |
| Neratinib | angiogenesis | 1.61 | 0.0989 | 0.9628 |
| Trastuzumab-based | TGFb_activity | 0.57 | 0.0507 | 0.9535 |
| Ganetespib | Dendritic | 0.61 | 0.1318 | 0.9628 |
| T-DM1+Pertuzumab | hypoxia_score | 2.79 | 0.0225 | 0.9535 |
| MK-2206 | immune_exclusion | 0.58 | 0.1167 | 0.9628 |