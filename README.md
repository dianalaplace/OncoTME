# OncoTME

**Опухолевое микроокружение как прогностический и предиктивный биомаркер при раке молочной железы.**

ML-фреймворк, который использует молекулярный профиль TME (иммунная инфильтрация, строма, антиген-презентация) для двух параллельных задач:

1. **Prognostic** — прогноз безрецидивной / общей выживаемости (RFS / DFS / OS)
2. **Predictive** — прогноз ответа на терапию (pCR в неоадъюванте, ORR) по классам: chemo, endocrine, anti-HER2, PARP, CDK4/6, ICI

## Гипотезы

| # | |
|---|---|
| **H1** | TME-архетип — независимый предиктор RFS/OS при контроле на клинику и подтип |
| **H2** | Роль TME-компонентов зависит от класса терапии (chemo / anti-HER2 / PARP / ICI / endocrine) |
| **H3** | TME-эффект подтип-специфичен (TNBC / HER2+ / HR+) |
| **H4** | Дефект антиген-презентации (APM/HLA-low) — маркер плохого RFS независимо от инфильтрации |
| **H5** | TME-hot профиль → больший relative benefit от добавления таргетной / ICI поверх chemo |

## Данные (публичные)

- **Discovery / prognostic**: TCGA-BRCA, METABRIC, SCAN-B
- **Predictive / response**: GSE25066 (MDACC), I-SPY1/2 (GSE22226, GSE194040), GeparSixto (GSE87455), GeparNuevo (GSE173839), CALGB 40601 (GSE181574), NeoALTTO (GSE50948), TransATAC (GSE59515), Bassez 2021 (GSE169246)

## Структура

```
src/oncotme/
├── cohorts/    # loader'ы каждой когорты → CohortBundle
├── preprocess/ # нормализация, gene harmonization, ComBat, QC
├── features/   # ssGSEA сигнатуры, TME-панель, baselines (TIS, CYT, ESTIMATE)
├── models/     # response.py (clf), survival.py (Cox/RSF), blocks.py (ablation)
├── explain/    # SHAP, PD, interactions
├── stats/      # DeLong, compareC, bootstrap, calibration, DCA
└── report/     # Quarto integration
```

## Ключевые результаты (flagship figures)

- **TME × treatment interaction heatmap** — `feature × treatment → {pCR-OR, RFS-HR}`
- **Subtype-stratified SHAP** — отдельные ранжирования в TNBC / HER2+ / HR+
- **Block-importance** — % |SHAP| по модулям (immune / stromal / APM / clinical)
- **Survival по TME-архетипам** — KM-curves с multivariable Cox

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .[dev]

# когортные artefacts
python scripts/build_features.py --config config/cohorts.yaml

# модели
python scripts/train_survival.py --cohort tcga_brca
python scripts/train_response.py --cohort pooled_neoadjuvant

# отчёт
quarto render report/
```

## Переиспользуемое

Проект наследует ряд модулей из сестринского `DrugResponce/` (ssGSEA-обёртка, ComBat, nested-CV каркас, TME-сигнатуры GMT).

## Лицензия / статус

Research use. In development.
