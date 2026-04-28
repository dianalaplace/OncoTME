# scripts/

Executable entrypoints для OncoTME-пайплайна.

Планируемые:

| Скрипт | Что делает |
|---|---|
| `build_features.py` | Читает `config/cohorts.yaml`, запускает loader каждой когорты → ssGSEA → TME-панель → `data/processed/<cohort>_features.parquet` |
| `train_survival.py` | Обучает survival-модели (ElasticNet-Cox + RSF) по cohort-конфигу; артефакты в `results/metrics/` + `results/shap/` |
| `train_response.py` | Обучает response-классификатор на pooled neoadjuvant; LOCO-CV по cohort-labels |
| `run_ablation.py` | Запускает feature-block ablation (E1, E2) через `models/blocks.ablation_grid` |
| `run_interaction.py` | Treatment × TME interaction анализ (E4); flagship heatmap `feature × treatment` |
| `make_report.py` | Собирает фигуры/таблицы и триггерит `quarto render report/` |

Все скрипты должны быть чистыми CLI-обёртками — вся логика в `src/oncotme/`.
