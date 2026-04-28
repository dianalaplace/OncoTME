# TCGA-BRCA rigorous benchmark — OS

- N (non-NaN outcome): 1096
- Events: 151

## Nested 5-fold CV C-index

| Model | C-index | 95% CI | ΔC vs clinical |
|---|---|---|---|
| clinical_only | 0.6854 | [0.642, 0.733] | +0.000 |
| TIS_18 | 0.5298 | [0.479, 0.587] | -0.156 |
| CYT | 0.5425 | [0.485, 0.602] | -0.143 |
| sig_IFN_gamma | 0.5496 | [0.497, 0.607] | -0.136 |
| random_feat_s0 | 0.5602 | [0.504, 0.613] | -0.125 |
| random_feat_s1 | 0.5578 | [0.503, 0.614] | -0.128 |
| random_feat_s2 | 0.5286 | [0.471, 0.583] | -0.157 |
| NMF_risk_k6 | 0.5515 | [0.497, 0.610] | -0.134 |
| NMF_plus_clinical | 0.6894 | [0.640, 0.738] | +0.004 |

## Permutation test (null distribution C-index)

- **NMF_risk_k6**: observed C=0.5515 | null mean=0.489 | null p95=0.543 | **p=0.0297**
- **NMF_plus_clinical**: observed C=0.6894 | null mean=0.494 | null p95=0.543 | **p=0.0099**

## Diagnostics

- **Schoenfeld PH test**: p=0.4284 | HR=1.45 [1.20, 1.74] | PH violated: **False**
- **Spearman ρ (risk vs time | observed)**: raw=-0.160 → partial (control clinical)=-0.143 | proxy warning: **False**

## NMF init sensitivity

- init=`nndsvda`: cophenetic=1.000, stability=1.000, explained_var=0.986
- init=`random`: cophenetic=0.761, stability=0.744, explained_var=0.966

## Verdict (auto-generated)

🔴 **Stop; reconsider**: ΔC = -0.134 — TME risk is not distinguishable from clinical baseline. Revisit feature set, k, cohort scope.