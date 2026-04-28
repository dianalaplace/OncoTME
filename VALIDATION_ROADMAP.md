# Validation Roadmap

Этот файл — «список на вырост». Здесь перечислены проверки, которые нужно
реализовать по мере наполнения пайплайна, и данные/метрики, которые помогут
поднять уверенность в результатах.

Живой индекс. Обновляется вместе с кодом.

---

## Уже реализовано

### Атомарные (`validation/checks.py`)
- [x] NaN / Inf / shape / range
- [x] Уникальность индекса
- [x] Выравнивание индексов между таблицами
- [x] Class balance для бинарного таргета
- [x] Ожидаемый знак корреляции между сигнатурами

### Когортные (`validation/cohort_checks.py`)
- [x] Expression unit plausibility (log-space vs counts)
- [x] HGNC-like gene symbols (ловит Ensembl/probe IDs)
- [x] Endpoint encoding (pCR бинарный, time_* ≥ 0, event_* ∈ {0,1})
- [x] Disjoint sample_id между когортами
- [x] Присутствие ключевых клинических колонок (age/ER/HER2)

### Feature-level (`validation/feature_checks.py`)
- [x] NaN / Inf / fully-NaN columns
- [x] Constant columns
- [x] Near-duplicate columns (|r|>0.98)
- [x] Биологическая sanity: CD8↔IFNγ, HLA_I↔APM, TLS↔B_cell, TGFβ↔CD8 и т.п.
- [x] Центрированность z-score сигнатур

### Model-level (`validation/model_checks.py`)
- [x] CV leakage: overlap индексов и групп между train/test
- [x] Permutation sanity (AUC ≈ 0.5 на перемешанном y)
- [x] Seed stability (std метрики по ≥3 seed'ам)
- [x] Calibration: Brier + ECE

---

## Что добавить в следующей итерации

### Cohort-level
- [ ] **Housekeeping genes expression sanity** — ACTB, GAPDH, PPIA, RPL13A должны быть экспрессированы во всех образцах; образец с <10-й перцентиль housekeeping — технический outlier
- [ ] **Subtype consistency** — если есть IHC ER/PR/HER2 и PAM50-call, проверить пересечение (>80% ER+ IHC == Luminal A/B PAM50); большое несоответствие → кривая аннотация
- [ ] **PCA outlier detection** — Mahalanobis distance по первым 10 PC; самплы с p<0.001 помечаются как suspected outliers
- [ ] **Sex-chromosome sanity** — BRCA-когорты должны быть почти все female (XIST+, RPS4Y-); если нет — смешение с ccRCC/другой тканью
- [ ] **Duplicate / related samples** — матрица cosine-similarity между образцами; пары >0.98 — возможные технические реплики или pedigree-related (должны быть в одном CV fold)
- [ ] **Batch effect detection** — silhouette по `platform`/`batch`-metadata до ComBat; если silhouette > 0.3 после — не сработало
- [ ] **Tumor purity sanity** — ESTIMATE tumor purity vs pathology-reported покрытие (если доступно); purity <0.3 помечать для sensitivity
- [ ] **Missing treatment context** — для predictive-когорт обязательно наличие arm/regimen колонки; None → WARN

### Feature-level
- [ ] **Leakage-of-target в feature engineering** — корреляция каждой фичи с y; если >0.99 на train — почти наверняка случайно попавший в фичи сам таргет или его прокси
- [ ] **Cross-platform stability of TME-панели** — сравнение распределений сигнатур между RNA-seq и microarray когортами; KS-test p<0.001 после ComBat → проблема
- [ ] **Signature size sanity** — минимум 3 гена сигнатуры присутствует в экспрессии; иначе скор нерепрезентативен
- [ ] **Time-consistency** (если pre/on-treatment как в Bassez) — pre-sample того же пациента не должен быть далёк от on-sample по non-immune генам (housekeeping), иначе технический artefact
- [ ] **Replication of published signature distributions** — TIS на TCGA-BRCA должна давать медиану, совпадающую с Ayers 2017 Fig. 2 в пределах 5% по подтипам
- [ ] **Monotonicity PAM50 → immune** — Basal > Luminal в immune scores; если нет — перепутанные подтипы

### Model-level
- [ ] **Nested CV honesty** — feature-selection / scaler / ComBat обучаются только внутри train-fold; автоматическая проверка через «отравленный» ген (вводим специально leaky фичу → модель не должна её видеть при правильной изоляции)
- [ ] **Temporal leakage** — если даты диагноза/биопсии доступны, train/test разбиение по времени хотя бы в sensitivity analysis
- [ ] **Cohort leakage in LOCO** — автоматически проверять, что ни один sample из held-out когорты не встречался в train (дубликаты по ID/expression)
- [ ] **Subgroup performance tracking** — AUC/C-index отдельно в TNBC/HER2+/HR+; если в подгруппе AUC < 0.5 — модель в ней работает антикоррелированно
- [ ] **Shapley additivity invariant** — сумма SHAP по фичам + base value ≈ prediction (в пределах ε); sanity для TreeSHAP
- [ ] **Interaction plausibility** — топ SHAP-interactions проверяются против известной биологии (CD8 × TGFβ должна быть отрицательной и т.д.)
- [ ] **Decision Curve Analysis validity** — Net benefit > treat-all и > treat-none хотя бы в диапазоне threshold 10-40%; если нет — клинически модель бесполезна
- [ ] **Fairness / bias slicing** — performance по age-binам, расе (где есть), институтам/странам; большие разрывы → biased модель
- [ ] **Bootstrap CI coverage** — width 95% CI для AUC/C-index должна быть < 0.15 на валидационной когорте; иначе сигнал статистически нестабилен

### Report / reproducibility
- [ ] **Hash-check of input artefacts** — SHA256 всех processed parquet; при пересборке изменение хэша отражается в report
- [ ] **Environment freeze** — автоматический дамп `pip freeze` + system info в `results/environment.txt`
- [ ] **Re-run byte-identity** — критические метрики должны воспроизводиться до 6 знаков при том же seed и config
- [ ] **Figure sanity** — все PNG в report/ имеют DPI ≥ 300 и осями без усечённых подписей (pytest-plot style)
- [ ] **Data-license check** — когорты с ограниченным доступом (controlled-access) не попадают в public-репо

---

## Данные, которые стоит добавить (повышение power и generalisability)

### Первый уровень (высокий приоритет)
- **SCAN-B** (GSE81540 / GSE96058, ~3200 образцов) — долгосрочный survival-follow-up, независимая популяционная валидация
- **KM Plotter breast meta-dataset** (~4900 образцов, mixed array+RNA-seq) — массив для мета-anal survival
- **TransATAC** (GSE59515) — endocrine-only, заполняет HR+ предиктивную нишу
- **GeparSixto** (GSE87455) — bevacizumab + chemo в TNBC/HER2+, заполняет «targeted, но не ICI» нишу

### Второй уровень
- **Single-cell BRCA атласы** — Wu 2021 (SRP223272), Bassez 2021 (GSE169246) для валидации deconvolution
- **TCGA-BRCA multi-omics** — CNV, methylation, mutations; добавить APM-loss-of-function mutations как отдельный HLA-evasion маркер
- **Spatial transcriptomics** — 10x Visium BRCA (SPATIAL-project); верификация, что TME-архетипы действительно соответствуют spatially defined регионам
- **CPTAC BRCA proteomics** — подтвердить, что mRNA-сигнатуры корреспондируют с белковой экспрессией (особенно HLA-I, PD-L1)

### Для валидации на реальной клинике
- **METABRIC replication с clinical arms** — если найти разбиение по treatment
- **NeoALTTO** (GSE50948) — anti-HER2 arms (trast/lapat/both)
- **SAFIR02-IMMUNO** (если доступно) — metastatic, multiple targeted arms
- **Real-world data** (институциональные когорты) — sanity-check generalisability за пределами trial-квалифицированных пациенток

---

## Метрики, которые стоит добавить к отчёту

- [ ] **Integrated Discrimination Improvement (IDI)** — сравнение моделей на survival/classification
- [ ] **Net Reclassification Improvement (NRI)** — клинически важная метрика "сколько пациенток переклассифицируем"
- [ ] **Harrell + Uno C-index** вместе (Uno устойчивее к цензурированию)
- [ ] **Time-dependent AUC в конкретные клинически значимые точки** (1, 3, 5, 10 лет)
- [ ] **Decision Curve Analysis** с per-subtype net benefit curves
- [ ] **Stratified C-index** — Harrell внутри каждой страты (cohort, subtype)

---

## Автоматизация

- [ ] **Pre-commit hook**: запуск быстрой части suite перед git commit
- [ ] **CI pipeline** (GitHub Actions): smoke-prогон validation на каждом PR
- [ ] **Nightly full run**: полный suite на всех данных с отчётом в `results/validation/YYYY-MM-DD.json`
- [ ] **Regression dashboard**: trend PASS/WARN/FAIL во времени, ловля деградаций
