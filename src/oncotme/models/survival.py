"""Survival-модели: ElasticNet-Cox, Random Survival Forest, DeepSurv (optional).

Скелет. Используем:
- ``sksurv.linear_model.CoxnetSurvivalAnalysis`` — Elastic-Net Cox
- ``sksurv.ensemble.RandomSurvivalForest`` — ensemble
- ``sksurv.metrics.concordance_index_censored``, ``brier_score``
- ``lifelines`` — для diagnostic plots (KM, PH assumption tests)

DeepSurv (pycox) — optional, добавляем только если лучше RSF на валидации.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd


@dataclass
class SurvivalTrainer:
    """Конфигурация survival-тренинга.

    Attributes
    ----------
    models
        Базовые модели: ``["coxnet", "rsf"]`` (``"deepsurv"`` optional).
    outer_cv
        n-folds для внешнего CV (по умолчанию 5, stratified by event).
    inner_cv
        n-folds для гиперпараметрического поиска.
    time_points
        Точки для time-dependent AUC / Brier (лет): например [1, 3, 5, 10].
    endpoint
        Имя event в CohortBundle.endpoints: ``rfs``, ``dfs``, ``os``.
    random_seed
        Для воспроизводимости.
    """

    models: List[str] = field(default_factory=lambda: ["coxnet", "rsf"])
    outer_cv: int = 5
    inner_cv: int = 5
    time_points: List[int] = field(default_factory=lambda: [1, 3, 5, 10])
    endpoint: str = "rfs"
    random_seed: int = 42

    def fit(
        self,
        X: pd.DataFrame,
        time: pd.Series,
        event: pd.Series,
        groups: Optional[pd.Series] = None,
    ) -> dict:
        """Обучение Cox/RSF с nested CV.

        Возвращает dict:
          - ``oof_risk`` : pd.Series of out-of-fold risk scores
          - ``c_index_harrell`` : bootstrap mean + CI
          - ``c_index_uno``
          - ``time_auc`` : DataFrame time × cohort-fold
          - ``brier_integrated`` : scalar
          - ``shap`` : per-model SHAP (RSF via TreeExplainer)

        TODO: implement.
        """
        raise NotImplementedError("SurvivalTrainer.fit to be implemented.")

    @staticmethod
    def to_structured_array(time: pd.Series, event: pd.Series) -> np.ndarray:
        """Конвертирует в ``sksurv`` structured array ``[(event_bool, time_float), ...]``."""
        dtype = [("event", bool), ("time", float)]
        arr = np.empty(len(time), dtype=dtype)
        arr["event"] = event.astype(bool).values
        arr["time"] = time.astype(float).values
        return arr
