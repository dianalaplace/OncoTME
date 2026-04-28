# External validation — train gse25065 → test gse25066

- Train: N=182  pos=42 (23.1%)
- Test:  N=488  pos=99 (20.3%)
- Endpoint: pCR

## Test-set AUC

| Model | AUC | 95% CI | ΔvsClinical | AP | Permutation p |
|---|---|---|---|---|---|
| NMF+clinical | 0.7533 | [0.701, 0.797] | +0.008 | 0.393 | 0.0020 |
| clinical_only | 0.7455 | [0.695, 0.792] | +0.000 | 0.375 | 0.0020 |
| ElasticNet_all+clinical | 0.7447 | [0.689, 0.797] | -0.001 | 0.422 | 0.0020 |
| ElasticNet_all | 0.7446 | [0.693, 0.796] | -0.001 | 0.402 | 0.0020 |
| NMF_logit_k5 | 0.6531 | [0.593, 0.711] | -0.092 | 0.342 | 0.0020 |
| random_feat_s1 | 0.6342 | [0.582, 0.697] | -0.111 | 0.282 | 0.0020 |
| sig_IFN_gamma | 0.6229 | [0.562, 0.683] | -0.123 | 0.298 | 0.0020 |
| random_feat_s0 | 0.6116 | [0.554, 0.671] | -0.134 | 0.267 | 0.0020 |
| sig_APM | 0.5933 | [0.526, 0.659] | -0.152 | 0.285 | 0.0040 |
| TIS_18 | 0.5902 | [0.530, 0.656] | -0.155 | 0.270 | 0.0060 |
| sig_CD8 | 0.5510 | [0.498, 0.610] | -0.195 | 0.223 | 0.0778 |
| CYT | 0.5253 | [0.470, 0.586] | -0.220 | 0.204 | 0.1996 |

## Verdict

🔴 **Does NOT replicate**: best ΔAUC = +0.008. Internal signal was cohort-specific; reconsider feature set or treatment-context homogeneity.