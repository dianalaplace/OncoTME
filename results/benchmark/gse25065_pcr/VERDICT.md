# gse25065 rigorous benchmark — pCR

- N: 182 | positives: 42 (23.1%)
- Regimen: neoadjuvant_chemo; T-FAC or FEC

## Nested 5-fold CV AUC

| Model | AUC | 95% CI | ΔAUC vs clinical | AP |
|---|---|---|---|---|
| clinical_only | 0.6018 | [0.515, 0.695] | +0.000 | 0.267 |
| TIS_18 | 0.5638 | [0.459, 0.676] | -0.038 | 0.292 |
| CYT | 0.4852 | [0.392, 0.587] | -0.117 | 0.219 |
| sig_IFN_gamma | 0.5682 | [0.469, 0.676] | -0.034 | 0.286 |
| sig_APM | 0.5393 | [0.451, 0.639] | -0.062 | 0.269 |
| sig_CD8 | 0.5497 | [0.456, 0.649] | -0.052 | 0.259 |
| random_feat_s0 | 0.5776 | [0.492, 0.677] | -0.024 | 0.252 |
| random_feat_s1 | 0.6005 | [0.500, 0.697] | -0.001 | 0.272 |
| NMF_logit_k5 | 0.5376 | [0.445, 0.639] | -0.064 | 0.274 |
| ElasticNet_all | 0.7100 | [0.607, 0.795] | +0.108 | 0.406 |
| ElasticNet_all+clinical | 0.6560 | [0.562, 0.742] | +0.054 | 0.358 |
| NMF+clinical | 0.5854 | [0.495, 0.685] | -0.016 | 0.276 |

## Permutation test
- **NMF_logit_k5**: observed AUC=0.5376 | null mean=0.488 | null p95=0.569 | **p=0.1935**
- **ElasticNet_all+clinical**: observed AUC=0.6560 | null mean=0.517 | null p95=0.607 | **p=0.0323**
- **NMF+clinical**: observed AUC=0.5854 | null mean=0.490 | null p95=0.597 | **p=0.1290**

## Verdict
✅ **Clinically meaningful signal**: 'ElasticNet_all' beats clinical by ΔAUC = +0.108. Proceed to external validation + subtype-stratified analysis.