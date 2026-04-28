"""Permutation test для null distribution C-index.

Идея: мы shuffle outcome (time + event pair перемешиваем согласованно), запускаем
полный nested CV pipeline → получаем null C-index. Повторяем n_perm раз →
эмпирическая p-value = доля null >= observed.

Важно: **перемешиваем time-event пары вместе** (иначе ломаем связь между
цензурированием и временем).
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def permutation_test(
    model_factory: Callable,
    X: pd.DataFrame,
    time: pd.Series,
    event: pd.Series,
    observed_c_index: float,
    clinical: Optional[pd.DataFrame] = None,
    *,
    n_permutations: int = 100,
    n_folds: int = 5,
    seed: int = 42,
) -> dict:
    """Empirical permutation p-value для C-index.

    Для каждой перестановки:
    1. Перемешиваем (time, event) пары вместе между пациентками
    2. Запускаем nested CV с permuted outcome
    3. Записываем C-index

    Returns
    -------
    dict:
        - ``observed``: observed C-index (передаётся как input)
        - ``null_c_indices``: list[float] — null distribution
        - ``p_value``: доля null >= observed
        - ``null_mean``, ``null_p95``: null summary
    """
    from .nested_cv import nested_cv_evaluate

    rng = np.random.default_rng(seed)
    null_c_indices: list[float] = []

    common = sorted(set(X.index) & set(time.index) & set(event.index))
    pairs = pd.DataFrame({
        "time": time.loc[common].values,
        "event": event.loc[common].values,
    }, index=common)

    for i in range(n_permutations):
        perm_idx = rng.permutation(len(pairs))
        t_perm = pd.Series(pairs["time"].values[perm_idx], index=pairs.index)
        e_perm = pd.Series(pairs["event"].values[perm_idx], index=pairs.index)
        try:
            res = nested_cv_evaluate(
                model_factory, X, t_perm, e_perm, clinical,
                n_folds=n_folds, seed=seed + i, n_bootstrap=0,
            )
            null_c_indices.append(res.mean_c_index)
        except Exception as exc:
            logger.warning("Permutation %d failed: %s", i, exc)
            continue
        if (i + 1) % 10 == 0:
            logger.info(
                "  perm %d/%d: null C mean=%.3f, max=%.3f",
                i + 1, n_permutations,
                float(np.mean(null_c_indices)), float(np.max(null_c_indices)),
            )

    null_arr = np.asarray(null_c_indices)
    null_arr = null_arr[np.isfinite(null_arr)]
    if len(null_arr) == 0:
        return {"observed": observed_c_index, "null_c_indices": [],
                "p_value": float("nan"), "n_success": 0}

    # empirical one-sided p
    p_value = float((null_arr >= observed_c_index).sum() + 1) / (len(null_arr) + 1)
    return {
        "observed": float(observed_c_index),
        "null_c_indices": null_arr.tolist(),
        "null_mean": float(np.mean(null_arr)),
        "null_p95": float(np.quantile(null_arr, 0.95)),
        "null_max": float(np.max(null_arr)),
        "p_value": p_value,
        "n_success": len(null_arr),
    }
