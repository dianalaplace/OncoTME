"""Нормализация и гармонизация gene expression между когортами.

Функции:
- ``log2_tpm(x)`` — log2(TPM + 1) для RNA-seq TPM
- ``harmonize_genes(list_of_exprs)`` — пересечение HGNC symbols
- ``rank_normalize(expr)`` — per-sample rank normalization (platform-agnostic)
- ``combat_fit_transform / combat_transform`` — ComBat batch correction
  (требует ``pycombat`` или ``combat-py``; при отсутствии — логирует и возвращает
  вход без изменений с WARN)

ComBat работает только на матрицах с постоянным набором генов между train
и test. Пайплайн обязан сначала harmonize_genes по всем когортам, потом
применять ComBat.
"""

from __future__ import annotations

import logging
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    from pycombat import Combat  # noqa: F401
    _COMBAT_AVAILABLE = True
except Exception as exc:  # noqa: BLE001
    Combat = None  # type: ignore[assignment]
    _COMBAT_AVAILABLE = False
    _COMBAT_IMPORT_ERROR = exc


def log2_tpm(tpm: pd.DataFrame, pseudocount: float = 1.0) -> pd.DataFrame:
    """Преобразование TPM → log2(TPM + pseudocount)."""
    return np.log2(tpm + pseudocount)


def harmonize_genes(exprs: Iterable[pd.DataFrame]) -> List[pd.DataFrame]:
    """Пересечение HGNC-символов между всеми матрицами → все усекаются до общего набора.

    Подразумевается, что индекс каждого DataFrame — gene symbols. Пустой результат —
    ошибка (вероятно, разные namespace: HGNC vs Ensembl vs probe IDs).
    """
    exprs = list(exprs)
    if not exprs:
        return []
    common: set[str] = set(map(str, exprs[0].index))
    for e in exprs[1:]:
        common &= set(map(str, e.index))
    if not common:
        raise ValueError(
            "harmonize_genes: intersection is empty. "
            "Check that all matrices use the same gene namespace (e.g., HGNC)."
        )
    common_sorted = sorted(common)
    return [e.loc[common_sorted] for e in exprs]


def rank_normalize(expr: pd.DataFrame) -> pd.DataFrame:
    """Per-sample rank normalization to percentiles [0, 1].

    Используется как platform-agnostic альтернатива ComBat для сигнатурных
    скоров. Не заменяет ComBat для модельного обучения.
    """
    return expr.rank(axis=0, pct=True)


def combat_fit_transform(
    expr: pd.DataFrame, batches: pd.Series
) -> Tuple[pd.DataFrame, object]:
    """Применяет ComBat батч-коррекцию; возвращает (corrected_expr, fitted_model).

    Вход: gene × sample матрица + Series того же размера, что и columns,
    со значениями batch-label (cohort, platform).

    При отсутствии ``pycombat`` — возвращает вход без изменений и модель = None
    (логирует warning). Это осознанный deferred failure: хочется, чтоб пайплайн
    хоть как-то проходил без batch correction, но validation-suite должен
    поймать batch-эффект через silhouette/PCA check.
    """
    if not _COMBAT_AVAILABLE:
        logger.warning(
            "pycombat unavailable (%s); returning expression unchanged. "
            "Install pycombat or combat-py before cross-cohort pooling.",
            _COMBAT_IMPORT_ERROR,
        )
        return expr.copy(), None

    if batches.index.equals(expr.columns):
        aligned_batches = batches
    else:
        aligned_batches = batches.reindex(expr.columns)
    if aligned_batches.isna().any():
        missing = aligned_batches.index[aligned_batches.isna()].tolist()
        raise ValueError(
            f"combat_fit_transform: {len(missing)} samples missing batch label: {missing[:3]}"
        )

    model = Combat()  # type: ignore[operator]
    corrected = model.fit_transform(expr.T.values, aligned_batches.to_numpy())
    corrected_df = pd.DataFrame(corrected.T, index=expr.index, columns=expr.columns)
    # sanity: число NaN не должно вырасти
    if corrected_df.isna().sum().sum() > expr.isna().sum().sum():
        raise RuntimeError("combat_fit_transform produced unexpected NaNs.")
    return corrected_df, model


def combat_transform(
    expr: pd.DataFrame, combat_model: object, batches: pd.Series
) -> pd.DataFrame:
    """Применяет уже обученный ComBat model к новой когорте."""
    if combat_model is None:
        logger.warning("combat_transform: model is None; returning input unchanged.")
        return expr.copy()
    aligned = batches.reindex(expr.columns)
    if aligned.isna().any():
        raise ValueError("combat_transform: batch labels missing for some samples.")
    corrected = combat_model.transform(expr.T.values, aligned.to_numpy())  # type: ignore[attr-defined]
    return pd.DataFrame(corrected.T, index=expr.index, columns=expr.columns)
