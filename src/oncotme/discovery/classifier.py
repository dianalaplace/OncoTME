"""Centroid NN classifier для переноса архетипов на внешние когорты.

Работает поверх z-scored signature matrix. Для каждого кластера хранится
центроид (mean z-score). Новая sample classified по минимуму correlation distance
(устойчивее к масштабированию, чем euclidean).
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class CentroidClassifier:
    """Centroid-based классификатор для переноса TME-архетипов.

    Attributes
    ----------
    centroids_
        DataFrame cluster_id × feature, mean z-score.
    feature_names_
        Имена колонок, в том порядке, в котором применяется distance.
    metric
        ``correlation`` (default) | ``euclidean`` | ``cosine``.
    """

    def __init__(self, metric: str = "correlation") -> None:
        if metric not in {"correlation", "euclidean", "cosine"}:
            raise ValueError(f"Unknown metric {metric!r}")
        self.metric = metric
        self.centroids_: Optional[pd.DataFrame] = None
        self.feature_names_: Optional[list[str]] = None

    def fit(self, X: pd.DataFrame, labels: pd.Series) -> "CentroidClassifier":
        """Вычисляет центроиды для каждого кластера."""
        df = X.copy()
        df["_cluster"] = labels.reindex(df.index)
        df = df.dropna(subset=["_cluster"])
        self.centroids_ = df.groupby("_cluster").mean()
        self.feature_names_ = [c for c in X.columns]
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Присваивает cluster_id каждой sample в X."""
        if self.centroids_ is None:
            raise RuntimeError("Classifier not fitted.")
        common = [f for f in self.feature_names_ if f in X.columns]
        missing = set(self.feature_names_) - set(X.columns)
        if missing:
            logger.warning(
                "predict: %d/%d features missing in X; imputing zeros.",
                len(missing), len(self.feature_names_),
            )
        X_aligned = X.reindex(columns=self.feature_names_, fill_value=0.0)

        dist = self._pairwise(X_aligned.values, self.centroids_.values)
        preds = self.centroids_.index[dist.argmin(axis=1)]
        return pd.Series(preds, index=X.index, name="archetype")

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        """Soft-assignment: инвертируем distance → softmax."""
        if self.centroids_ is None:
            raise RuntimeError("Classifier not fitted.")
        X_aligned = X.reindex(columns=self.feature_names_, fill_value=0.0)
        dist = self._pairwise(X_aligned.values, self.centroids_.values)
        # softmax(-dist)
        neg = -dist
        neg -= neg.max(axis=1, keepdims=True)  # numerical stability
        exp = np.exp(neg)
        prob = exp / exp.sum(axis=1, keepdims=True)
        return pd.DataFrame(prob, index=X.index, columns=self.centroids_.index)

    def _pairwise(self, X: np.ndarray, C: np.ndarray) -> np.ndarray:
        if self.metric == "euclidean":
            return np.linalg.norm(X[:, None, :] - C[None, :, :], axis=2)
        if self.metric == "cosine":
            Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
            Cn = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-12)
            return 1.0 - Xn @ Cn.T
        # correlation
        Xc = X - X.mean(axis=1, keepdims=True)
        Cc = C - C.mean(axis=1, keepdims=True)
        Xn = Xc / (np.linalg.norm(Xc, axis=1, keepdims=True) + 1e-12)
        Cn = Cc / (np.linalg.norm(Cc, axis=1, keepdims=True) + 1e-12)
        return 1.0 - Xn @ Cn.T

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str | Path) -> "CentroidClassifier":
        with Path(path).open("rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise TypeError(f"Loaded object is not {cls.__name__}")
        return obj
