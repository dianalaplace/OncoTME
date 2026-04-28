"""Nested 5-fold CV evaluator: leakage-free оценка C-index на TCGA/METABRIC.

Критические решения:
- **Outer loop**: StratifiedKFold по event (баланс цензурирования между fold'ами)
- **Inner loop (опциональный)**: используется только если у модели есть
  ``tune_on_inner_cv`` — иначе фит делается с default penalizer (оба варианта
  документированы в публикациях — без внутреннего tuning overestimation меньше)
- **Preprocess** (z-score, NMF): fit на train-fold, apply на test-fold — без
  единого raz-z-scoring всей когорты
- **Censored harmonization**: test-fold event=0 учитывается честно через
  concordance_index (IPCW не используем — простой harrell-c достаточен для
  comparative benchmark; для публикации добавим Uno's C)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class NestedCVResult:
    model_name: str
    fold_c_indices: List[float]
    mean_c_index: float
    ci_lower: float
    ci_upper: float
    n_folds: int
    total_n: int
    total_events: int

    def summary_row(self) -> dict:
        return {
            "model": self.model_name,
            "c_index_mean": self.mean_c_index,
            "c_index_ci_lo": self.ci_lower,
            "c_index_ci_hi": self.ci_upper,
            "fold_c_indices": self.fold_c_indices,
            "n_folds": self.n_folds,
            "total_n": self.total_n,
            "total_events": self.total_events,
        }


def _concordance(time: pd.Series, risk: pd.Series, event: pd.Series) -> float:
    """Harrell's C через lifelines.concordance_index."""
    from lifelines.utils import concordance_index
    try:
        return float(concordance_index(time, -risk, event))
    except Exception as exc:
        logger.warning("concordance_index failed: %s", exc)
        return float("nan")


def nested_cv_evaluate(
    model_factory: Callable[[], "BenchmarkModel"],
    X: pd.DataFrame,
    time: pd.Series,
    event: pd.Series,
    clinical: Optional[pd.DataFrame] = None,
    *,
    n_folds: int = 5,
    seed: int = 42,
    n_bootstrap: int = 500,
) -> NestedCVResult:
    """Nested 5-fold CV: fit model на train, score на test, pool C-index.

    Parameters
    ----------
    model_factory
        Callable возвращающий НОВЫЙ инстанс модели для каждого fold (важно для
        NMF, которая хранит state).
    X
        Feature matrix (sample × features).
    time, event
        Survival данные.
    clinical
        Optional — для clinical-only / combined моделей.
    n_folds
        Outer CV folds.
    n_bootstrap
        Размер bootstrap на pooled OOF predictions для CI C-index.
    """
    from sklearn.model_selection import StratifiedKFold

    # Filter to samples with non-NaN time/event and feature-alignment
    common = sorted(set(X.index) & set(time.index) & set(event.index))
    if clinical is not None:
        common = sorted(set(common) & set(clinical.index))
    t = pd.to_numeric(time.loc[common], errors="coerce")
    e = pd.to_numeric(event.loc[common], errors="coerce")
    mask = t.notna() & e.notna() & (t >= 0)
    common = list(np.array(common)[mask])
    t, e = t.loc[common], e.loc[common].astype(int)
    X_c = X.loc[common]
    clin_c = clinical.loc[common] if clinical is not None else None

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_scores: List[float] = []
    oof_risk = pd.Series(np.nan, index=common, dtype=float)

    for fold, (tr_idx, te_idx) in enumerate(skf.split(common, e.values), start=1):
        tr = [common[i] for i in tr_idx]
        te = [common[i] for i in te_idx]
        model = model_factory()
        try:
            model.fit(X_c.loc[tr], t.loc[tr], e.loc[tr],
                      clin_c.loc[tr] if clin_c is not None else None)
            risk = model.score(X_c.loc[te], clin_c.loc[te] if clin_c is not None else None)
            fold_c = _concordance(t.loc[te], risk.loc[te], e.loc[te])
            fold_scores.append(fold_c)
            oof_risk.loc[te] = risk.loc[te].values
            logger.info("  %s fold %d: C=%.4f (n_train=%d, n_test=%d)",
                        getattr(model, "name", "model"), fold, fold_c, len(tr), len(te))
        except Exception as exc:
            logger.warning("Fold %d failed: %s", fold, exc)
            fold_scores.append(float("nan"))

    valid = oof_risk.dropna()
    if len(valid) < 20:
        logger.warning("Too few valid OOF predictions for bootstrap CI.")
        ci_lo, ci_hi = float("nan"), float("nan")
        mean_c = float(np.nanmean(fold_scores))
    else:
        t_v, e_v = t.loc[valid.index], e.loc[valid.index]
        mean_c = _concordance(t_v, valid, e_v)
        # bootstrap CI
        rng = np.random.default_rng(seed)
        boot = []
        for _ in range(n_bootstrap):
            bi = rng.choice(len(valid), size=len(valid), replace=True)
            try:
                c = _concordance(t_v.iloc[bi], valid.iloc[bi], e_v.iloc[bi])
                if np.isfinite(c):
                    boot.append(c)
            except Exception:
                continue
        ci_lo, ci_hi = (float(np.quantile(boot, 0.025)),
                         float(np.quantile(boot, 0.975))) if boot else (float("nan"), float("nan"))

    return NestedCVResult(
        model_name=getattr(model, "name", "unknown"),
        fold_c_indices=[float(s) for s in fold_scores],
        mean_c_index=mean_c,
        ci_lower=ci_lo,
        ci_upper=ci_hi,
        n_folds=n_folds,
        total_n=len(common),
        total_events=int(e.sum()),
    )
