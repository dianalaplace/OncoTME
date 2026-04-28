# I-SPY2 Composite ICI-TME-readiness score × Pembrolizumab

## Composite definition (pre-specified by ICI biology)

```
composite = z(IFN_gamma) + z(HLA_I_score) + z(B_cell) + z(checkpoint_score)
           - z(CAF_proxy)
```

- N = 248 (ctrl=179, pembro=69), pCR events = 62

## Main interaction test (single pre-specified hypothesis)

| Model | OR interaction | 95% CI | p | BH-adjusted (1 test) |
|---|---|---|---|---|
| Composite × arm | 2.977 | [1.191, 7.445] | 0.0196 | 0.0196 |
| TIS-proxy × arm (comparator) | 2.895 | [1.079, 7.768] | 0.0348 | — |

## Tertile-stratified pembrolizumab effect

| Tertile | Ctrl ORR | Pembro ORR | ARR | OR (95% CI) |
|---|---|---|---|---|
| **low** | 10.8% (n=65) | 5.6% (n=18) | **-5.2%** | 0.67 [0.11, 4.18] |
| **medium** | 15.1% (n=53) | 48.3% (n=29) | **+33.2%** | 5.01 [1.80, 13.95] |
| **high** | 26.2% (n=61) | 72.7% (n=22) | **+46.5%** | 7.00 [2.41, 20.37] |

## Verdict

✅ **Pre-specified composite score formally interacts with pembrolizumab** (p = 0.0196, OR_interaction = 2.98). This is a single pre-specified test, so no multiple-testing correction required. Independent replication remains necessary (GeparNuevo, KEYNOTE-522 biomarker sub-study, IMpassion130).

Clinically: pembrolizumab ARR is **46.5%** in composite-**high** patients vs **-5.2%** in composite-**low** (difference +51.7 pp).