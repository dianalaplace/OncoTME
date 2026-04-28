"""Подготовка feature-frame для discovery-кластеризации.

Ключевые решения:
- Используем только сигнатурные скоры (``sig__*``), не индивидуальные гены:
  они высоко скоррелированы внутри сигнатур → добавляют шум, не информацию.
- Клинические переменные (``clin__*``) НЕ участвуют в discovery:
  кластеризация должна быть "слепой" к фенотипу; связь архетипа с подтипом —
  результат, а не вход.
- Robust z-score (median / MAD): устойчивее к outliers ssGSEA ES, чем обычный z-score.
  Matches Șenbabaoğlu 2014 и RNA-seq best practice.
"""

from __future__ import annotations

import logging
from typing import List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _robust_zscore(x: pd.Series) -> pd.Series:
    """Robust z-score: (x - median) / (1.4826 · MAD). Устойчив к outliers."""
    med = x.median()
    mad = (x - med).abs().median()
    if mad == 0:
        # fallback на обычный z-score
        sd = x.std()
        if sd == 0:
            return pd.Series(0.0, index=x.index)
        return (x - x.mean()) / sd
    return (x - med) / (1.4826 * mad)


def preprocess_for_clustering(
    features: pd.DataFrame,
    *,
    robust: bool = True,
    drop_low_variance: float = 0.01,
) -> pd.DataFrame:
    """Подготавливает sample × signature frame для кластеризации.

    Parameters
    ----------
    features
        Выход ``features/tme_panel`` или ``build_features.py``.
        Индекс = ``sample_id``; колонки = ``sig__*`` / ``gene__*`` / ``clin__*``.
    robust
        Использовать robust z-score (median/MAD) вместо стандартного. По умолчанию True.
    drop_low_variance
        Порог std (после scaling) для отбрасывания почти-константных сигнатур.
        Значение 0.01 отбросит сигнатуру только если её std после z-score
        действительно вырожден (почти всегда — баг препроцессинга).

    Returns
    -------
    sample × signature DataFrame, zero-NaN, уже центрированный и отмасштабированный.
    """
    sig_cols = [c for c in features.columns if c.startswith("sig__")]
    if not sig_cols:
        raise ValueError(
            "No 'sig__' columns found. Discovery clusters on signature scores; "
            "run build_features first."
        )
    X = features[sig_cols].copy()

    if X.isna().any().any():
        n_nan = int(X.isna().sum().sum())
        logger.warning("Found %d NaN in signature matrix — imputing column median.", n_nan)
        X = X.fillna(X.median(numeric_only=True))

    if robust:
        X_scaled = X.apply(_robust_zscore, axis=0)
    else:
        X_scaled = (X - X.mean()) / X.std().replace(0, 1.0)

    # отбрасываем вырожденные колонки (std ≈ 0 после scaling — это баг)
    stds = X_scaled.std()
    drop = stds[stds < drop_low_variance].index.tolist()
    if drop:
        logger.warning("Dropping %d near-constant scaled signatures: %s", len(drop), drop)
        X_scaled = X_scaled.drop(columns=drop)

    # убираем экстремальные outliers (|z| > 10) — клипируем, но не удаляем sample
    X_scaled = X_scaled.clip(lower=-10, upper=10)

    return X_scaled


def signature_module_map(columns: List[str]) -> dict[str, str]:
    """Разбивка сигнатур по модулям (immune / stromal / apm) для характеризации."""
    immune = {"B_cell", "CD4_T_cell", "CD8_T_cell", "NK_cell", "Treg", "Dendritic",
              "M2_macrophage", "MDSC_proxy", "TLS_signature", "IFN_gamma"}
    stromal = {"CAF_proxy", "TGFb", "angiogenesis", "hypoxia",
               "CXCL1_MDSC_axis", "M2_polarization"}
    apm = {"HLA_I_score", "HLA_II_score", "APM_score", "checkpoint_score"}

    mapping: dict[str, str] = {}
    for col in columns:
        name = col.replace("sig__", "")
        if name in immune:
            mapping[col] = "immune"
        elif name in stromal:
            mapping[col] = "stromal"
        elif name in apm:
            mapping[col] = "apm"
        else:
            mapping[col] = "other"
    return mapping
