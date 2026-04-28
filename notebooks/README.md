# notebooks/

Exploratory / разведочные ноутбуки. Ничего production — продакшн-логика уходит в `src/oncotme/`.

Планируемая нумерация:

- `01_eda.ipynb` — базовый EDA по каждой когорте (N, пропуски, подтип-распределение)
- `02_tme_phenotypes.ipynb` — unsupervised TME-кластеризация на TCGA + METABRIC
- `03_survival_discovery.ipynb` — KM + Cox по TME-архетипам (шаг 4→5)
- `04_response_discovery.ipynb` — базовые univariate скрининги перед обучением
- `05_interpretation.ipynb` — SHAP + treatment × TME interaction matrix
