"""Response-классификатор (pCR / ORR) с nested CV и leave-one-cohort-out.

Скелет. Опирается на паттерн из DrugResponce/src/best_response_model.py, но:
- убрана I-SPY2-specific логика (она становится частным случаем treatment_context)
- добавлена LOCO-CV по cohort-меткам
- treatment_context фигурирует и как feature, и как stratum

Основные сущности:
- ``ResponseTrainer`` — класс с конфигом (model, hyperparam grids, CV scheme)
- ``fit_nested_cv(X, y, groups=cohort_labels)`` → OOF predictions + fold metrics
- ``fit_loco(X, y, cohorts)`` → per-cohort held-out predictions
- ``stack_ensemble(model_list)`` — logistic meta-learner
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import pandas as pd


@dataclass
class ResponseTrainer:
    """Конфигурация response-тренинга.

    Attributes
    ----------
    models
        Список базовых моделей: ``["elastic_net", "xgboost", "lightgbm"]``.
    outer_cv
        n-folds для внешнего CV (по умолчанию 5).
    inner_cv
        n-folds для подбора гиперпараметров (по умолчанию 5).
    stratify_by
        Колонки в meta для стратификации (cohort, subtype).
    loco
        Если True — leave-one-cohort-out схема; иначе StratifiedKFold.
    random_seed
        Для воспроизводимости.
    """

    models: List[str] = field(default_factory=lambda: ["elastic_net", "xgboost", "lightgbm"])
    outer_cv: int = 5
    inner_cv: int = 5
    stratify_by: List[str] = field(default_factory=lambda: ["cohort", "subtype"])
    loco: bool = True
    random_seed: int = 42

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        groups: Optional[pd.Series] = None,
    ) -> dict:
        """Обучение с nested CV (или LOCO). Возвращает OOF pred + fold metrics + SHAP.

        TODO: портировать основу из DrugResponce/src/best_response_model.py.
        """
        raise NotImplementedError(
            "Port nested CV + XGBoost + SHAP block from "
            "DrugResponce/src/best_response_model.py; add LOCO branch."
        )
