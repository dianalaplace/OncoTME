# TCGA-BRCA NMF discovery — итоги

Дата: 2026-04-20. Когорта: TCGA-BRCA, N=1097 пациенток, 20530 генов, 23 TME-сигнатуры.

## Ключевые цифры

| Метрика | OS | DFS |
|---|---|---|
| N (с non-NaN endpoint + clinical) | 740 | 69 |
| C-index | **0.611** [0.550–0.673] | 0.619 [0.535–0.695] |
| Tertile log-rank p | **0.0044** | 0.022 (WARN) |
| Median survival (low risk) | 3959 дней (≈10.8 лет) | 44 мес |
| Median survival (high risk) | 3409 дней (≈9.3 года) | 24 мес |
| Δmedian low − high | ≈550 дней / 1.5 года | 20 мес |

**Вывод**: TME-risk score статистически дискриминирует low/medium/high группы по общей выживаемости (OS), C-index CI lower bound 0.55 > 0.5, log-rank p=0.004 с BH поправкой. DFS результаты ограничены мощностью (n=69), но направление подтверждается.

## NMF factorization

| | |
|---|---|
| Оптимальное k | **6** |
| Cophenetic correlation | 1.000 |
| Explained variance | 98.6% |
| Scan PAC (consensus clustering) | провален ( ≥0.42 — TME is continuous, а не discrete) |

Факторы (k=6):
- `factor_1`, `factor_4` — **fibrotic-stromal** (CAF / TGFβ / angiogenesis / hypoxia) — два близких фактора, следующая итерация с k=5 парсимоничнее
- `factor_2` — **immunosuppressive-hypoxic** (CXCL1-MDSC axis, hypoxia, MDSC-proxy)
- `factor_3` — **TLS-rich** (B_cell, TLS signature, CD4 T-cell)
- `factor_5` — **antigen-presenting** (APM, HLA-I, HLA-II) — ключевой immune-evasion axis
- `factor_6` — **immune-desert** (NK / M2-macrophage mix)

## Treatment × factor interaction

На TCGA: **0 факторов с q < 0.10**. Ожидаемо по двум причинам:
1. `targeted_therapy_received` на Xena — только бинарный YES/NO без drug-specific аннотации
2. 530 из 576 annotated пациенток (**92%**) получили YES → группы сильно несбалансированы для interaction-Cox

**→** Этот анализ будет мощным только на клинических trials (CALGB 40601, NeoALTTO, I-SPY2), где известен arm + drug class.

## Decision Curve Analysis

Max net benefit OS: 0.092 при threshold 0.05 (1090 пациенток flagged).  
Score превосходит "treat-all" и "treat-none" в низкопороговой области — клиническая польза для screening-решений.

## Артефакты

```
results/discovery_nmf/tcga_brca/
├── discovery_summary.json         # машинно-читаемый свод
├── SUMMARY.md                     # этот файл
├── factor_loadings_W.csv          # per-patient (1097 × 6)
├── factor_basis_H.csv             # factor × signature (6 × 23)
├── factor_names.csv               # автоматические имена
├── nmf_stability_by_k.csv         # scan результаты
├── risk_{os,dfs}.csv              # per-patient risk + тертиль
├── risk_coefs_{os,dfs}.csv        # Cox log-HR каждого фактора
├── tertile_km_{os,dfs}.csv        # данные для KM-кривых
├── tertile_pairwise_{os,dfs}.csv  # BH-corrected log-rank
├── treatment_interaction_{os,dfs}.csv  # factor × targeted
├── dca_{os,dfs}.csv               # decision curve данные
├── nmf_model.pkl                  # NMF basis для переноса на METABRIC
└── figures/
    ├── fig_nmf_stability.{pdf,png}
    ├── fig_factor_basis_heatmap.{pdf,png}
    ├── fig_factor_loadings_boxplot.{pdf,png}
    ├── fig_tertile_km_{os,dfs}.{pdf,png}
    ├── fig_dca_{os,dfs}.{pdf,png}
    └── fig_risk_coefs_{os,dfs}.{pdf,png}
```

## Что проверено validation suite

25 проверок на этапе build_features (все PASS).
Добавлены и прошли новые проверки discovery:
- `discovery.nmf_stability.k=6`: PASS (cophenetic=1.00, stability=1.00)
- `discovery.risk_score_discrimination` OS: PASS (CI lower 0.55 > 0.5)
- `discovery.risk_score_discrimination` DFS: PASS
- `discovery.tertile_separation` OS: PASS (p=0.004)
- `discovery.tertile_separation` DFS: WARN (p=0.022, above 0.01)
- `discovery.event_rate` OS + DFS: PASS

## Следующие шаги

1. **Sensitivity**: stability=1.0 — проверить на разных initialization methods (nndsvd / random / nndsvda); возможно искусственно высокий показатель из-за детерминистичности nndsvda. Запустить `--init random` и сравнить.
2. **k=5 parsimony**: factor_1 и factor_4 дублируют fibrotic-stromal axis; итерация с k=5 даст чище.
3. **METABRIC replication**: главная валидация. Применить `nmf_model.pkl` через `project_new_samples()` и повторить tertile KM. Если C-index > 0.55 — риск-скор переносится.
4. **Treatment-centric trials**: переход к GeparNuevo, CALGB 40601, I-SPY2 для реального treatment × factor interaction analysis.
5. **Subtype-stratified risk**: проверить C-index внутри TNBC / HER2+ / HR+ отдельно — H3.
6. **Publication figure**: объединить KM + DCA + forest plot в composite figure для статьи.
