"""Унифицированный интерфейс survival-моделей для benchmark.

Каждая модель реализует:
- ``fit(X, time, event, clinical)`` — обучение на train-fold
- ``score(X, clinical)`` → ``pd.Series`` — return linear risk predictor
  (больше = более рискованно). C-index использует ``-score`` как ordering.

Модели:
- ``NMFRiskModel`` — fit NMF на train, project test через NNLS, Cox на loadings
- ``ClinicalCoxModel`` — Cox только на clinical covariates
- ``SignatureCoxModel`` — Cox на одной signature column
- ``TISModel``, ``CYTModel`` — fixed-formula baselines (Ayers 2017, Rooney 2015)
- ``RandomFeaturesModel`` — Cox на K случайных фичах (null control)
- ``CombinedModel`` — объединение двух моделей (NMF + clinical)
"""

from __future__ import annotations

import logging
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# --- базовый интерфейс ----------------------------------------------------


class BenchmarkModel(ABC):
    """Minimal interface для survival-модели в benchmark."""

    name: str = "abstract"

    @abstractmethod
    def fit(
        self,
        X: pd.DataFrame,
        time: pd.Series,
        event: pd.Series,
        clinical: Optional[pd.DataFrame] = None,
    ) -> "BenchmarkModel": ...

    @abstractmethod
    def score(
        self, X: pd.DataFrame, clinical: Optional[pd.DataFrame] = None
    ) -> pd.Series: ...


def _cox_fit(
    df: pd.DataFrame,
    duration: str,
    event: str,
    penalizer: float,
    l1_ratio: float = 0.5,
):
    from lifelines import CoxPHFitter
    cph = CoxPHFitter(penalizer=penalizer, l1_ratio=l1_ratio)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cph.fit(df, duration_col=duration, event_col=event, show_progress=False)
    return cph


def _standardize_fit(df: pd.DataFrame, cols: List[str]) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Fit mean/std и возвращает стандартизованные values + means/stds для inference."""
    m = df[cols].mean()
    s = df[cols].std().replace(0, 1.0)
    out = df.copy()
    out[cols] = (df[cols] - m) / s
    return out, m, s


def _standardize_apply(
    df: pd.DataFrame, cols: List[str], m: pd.Series, s: pd.Series
) -> pd.DataFrame:
    out = df.copy()
    out[cols] = (df[cols] - m) / s.replace(0, 1.0)
    return out


# --- Clinical-only baseline -----------------------------------------------


@dataclass
class ClinicalCoxModel(BenchmarkModel):
    covariates: List[str] = field(
        default_factory=lambda: ["age", "stage", "subtype_er", "subtype_her2"]
    )
    penalizer: float = 0.01
    name: str = "clinical_only"
    _fitted_cols: List[str] = field(default_factory=list, init=False)
    _mean: Optional[pd.Series] = field(default=None, init=False)
    _std: Optional[pd.Series] = field(default=None, init=False)
    _beta: Optional[pd.Series] = field(default=None, init=False)

    def fit(self, X, time, event, clinical=None):
        if clinical is None:
            raise ValueError(f"{self.name} requires clinical frame.")
        common = sorted(set(time.index) & set(event.index) & set(clinical.index))
        df = pd.DataFrame({
            "_time": pd.to_numeric(time.loc[common], errors="coerce"),
            "_event": pd.to_numeric(event.loc[common], errors="coerce"),
        }, index=common)
        for c in self.covariates:
            if c in clinical.columns:
                df[c] = pd.to_numeric(clinical.loc[common, c], errors="coerce")
        df = df.dropna()
        feat_cols = [c for c in df.columns if c not in {"_time", "_event"}]
        # drop constant columns
        feat_cols = [c for c in feat_cols if df[c].nunique() >= 2]
        if not feat_cols:
            raise RuntimeError(f"{self.name}: no usable clinical features.")
        df_std, m, s = _standardize_fit(df, feat_cols)
        cph = _cox_fit(df_std[["_time", "_event"] + feat_cols],
                       "_time", "_event", self.penalizer)
        self._fitted_cols = feat_cols
        self._mean, self._std = m, s
        self._beta = cph.params_.loc[feat_cols]
        return self

    def score(self, X, clinical=None):
        if clinical is None:
            raise ValueError(f"{self.name} requires clinical frame.")
        df = pd.DataFrame(index=clinical.index)
        for c in self._fitted_cols:
            df[c] = pd.to_numeric(clinical[c], errors="coerce") if c in clinical.columns else np.nan
        df = df.fillna(df.median(numeric_only=True))
        df_std = _standardize_apply(df, self._fitted_cols, self._mean, self._std)
        return pd.Series(df_std[self._fitted_cols].values @ self._beta.values,
                         index=df.index, name=self.name)


# --- Signature baselines ---------------------------------------------------


@dataclass
class SignatureCoxModel(BenchmarkModel):
    """Cox только на одной сигнатурной колонке (``sig__XYZ``)."""
    signature: str = "sig__IFN_gamma"
    penalizer: float = 0.01
    name: str = "signature"
    _mean: Optional[float] = field(default=None, init=False)
    _std: Optional[float] = field(default=None, init=False)
    _beta: Optional[float] = field(default=None, init=False)

    def __post_init__(self):
        self.name = f"sig_{self.signature.replace('sig__', '')}"

    def fit(self, X, time, event, clinical=None):
        common = sorted(set(X.index) & set(time.index) & set(event.index))
        sig = X.loc[common, self.signature] if self.signature in X.columns else None
        if sig is None:
            raise ValueError(f"{self.signature} missing in X")
        df = pd.DataFrame({
            "_time": pd.to_numeric(time.loc[common], errors="coerce"),
            "_event": pd.to_numeric(event.loc[common], errors="coerce"),
            "_sig": sig.values,
        }).dropna()
        self._mean = float(df["_sig"].mean())
        self._std = float(df["_sig"].std()) or 1.0
        df["_sig"] = (df["_sig"] - self._mean) / self._std
        cph = _cox_fit(df, "_time", "_event", self.penalizer)
        self._beta = float(cph.params_["_sig"])
        return self

    def score(self, X, clinical=None):
        s = pd.to_numeric(X[self.signature], errors="coerce") if self.signature in X.columns \
            else pd.Series(0.0, index=X.index)
        z = (s - self._mean) / (self._std or 1.0)
        return pd.Series(self._beta * z.values, index=X.index, name=self.name)


class TISModel(BenchmarkModel):
    """Ayers 2017 TIS-18 как single composite baseline."""
    name = "TIS_18"

    def __init__(self, penalizer: float = 0.01):
        self.penalizer = penalizer
        self._mean = None
        self._std = None
        self._beta = None

    def _compute(self, features_or_expr: pd.DataFrame) -> pd.Series:
        # ищем в features_or_expr все TIS-18 гены
        from ..features.baselines import TIS_GENES, tis_18gene_score
        # если у нас features-frame (sample × cols) — поищем gene__*
        gene_cols = {c.replace("gene__", ""): c for c in features_or_expr.columns if c.startswith("gene__")}
        matches = [c for g in TIS_GENES if (c := gene_cols.get(g)) is not None]
        if len(matches) >= 5:
            sub = features_or_expr[matches]
            z = sub.sub(sub.mean(axis=0), axis=1).div(sub.std(axis=0).replace(0, 1), axis=1)
            return z.mean(axis=1)
        # fallback — нули (sanity — должен быть gene__ префикс)
        return pd.Series(0.0, index=features_or_expr.index)

    def fit(self, X, time, event, clinical=None):
        common = sorted(set(X.index) & set(time.index) & set(event.index))
        sig = self._compute(X.loc[common])
        df = pd.DataFrame({
            "_time": pd.to_numeric(time.loc[common], errors="coerce"),
            "_event": pd.to_numeric(event.loc[common], errors="coerce"),
            "_sig": sig.values,
        }).dropna()
        self._mean = float(df["_sig"].mean())
        self._std = float(df["_sig"].std()) or 1.0
        df["_sig"] = (df["_sig"] - self._mean) / self._std
        cph = _cox_fit(df, "_time", "_event", self.penalizer)
        self._beta = float(cph.params_["_sig"])
        return self

    def score(self, X, clinical=None):
        sig = self._compute(X)
        z = (sig - self._mean) / (self._std or 1.0)
        return pd.Series(self._beta * z.values, index=X.index, name=self.name)


class CYTModel(BenchmarkModel):
    """Rooney 2015 CYT = sqrt(GZMA · PRF1)."""
    name = "CYT"

    def __init__(self, penalizer: float = 0.01):
        self.penalizer = penalizer
        self._mean = None
        self._std = None
        self._beta = None

    def _compute(self, X: pd.DataFrame) -> pd.Series:
        gzma = X["gene__GZMA"] if "gene__GZMA" in X.columns else None
        prf1 = X["gene__PRF1"] if "gene__PRF1" in X.columns else None
        if gzma is None or prf1 is None:
            return pd.Series(0.0, index=X.index)
        return np.sqrt(np.clip(gzma, 0, None) * np.clip(prf1, 0, None))

    def fit(self, X, time, event, clinical=None):
        common = sorted(set(X.index) & set(time.index) & set(event.index))
        sig = self._compute(X.loc[common])
        df = pd.DataFrame({
            "_time": pd.to_numeric(time.loc[common], errors="coerce"),
            "_event": pd.to_numeric(event.loc[common], errors="coerce"),
            "_sig": sig.values,
        }).dropna()
        self._mean = float(df["_sig"].mean())
        self._std = float(df["_sig"].std()) or 1.0
        df["_sig"] = (df["_sig"] - self._mean) / self._std
        cph = _cox_fit(df, "_time", "_event", self.penalizer)
        self._beta = float(cph.params_["_sig"])
        return self

    def score(self, X, clinical=None):
        sig = self._compute(X)
        z = (sig - self._mean) / (self._std or 1.0)
        return pd.Series(self._beta * z.values, index=X.index, name=self.name)


# --- Random features null model -------------------------------------------


@dataclass
class RandomFeaturesModel(BenchmarkModel):
    """Случайные k фичей из ``sig__*`` — negative control.

    Каждый re-fit берёт НОВЫЕ случайные фичи, определяемые seed.
    """
    n_features: int = 6
    seed: int = 0
    penalizer: float = 0.05
    name: str = "random_features"
    _cols: List[str] = field(default_factory=list, init=False)
    _mean: Optional[pd.Series] = field(default=None, init=False)
    _std: Optional[pd.Series] = field(default=None, init=False)
    _beta: Optional[pd.Series] = field(default=None, init=False)

    def fit(self, X, time, event, clinical=None):
        sig_cols = [c for c in X.columns if c.startswith("sig__")]
        rng = np.random.default_rng(self.seed)
        if len(sig_cols) <= self.n_features:
            self._cols = sig_cols
        else:
            self._cols = list(rng.choice(sig_cols, size=self.n_features, replace=False))
        common = sorted(set(X.index) & set(time.index) & set(event.index))
        df = pd.DataFrame({
            "_time": pd.to_numeric(time.loc[common], errors="coerce"),
            "_event": pd.to_numeric(event.loc[common], errors="coerce"),
        }, index=common)
        for c in self._cols:
            df[c] = X.loc[common, c]
        df = df.dropna()
        df_std, m, s = _standardize_fit(df, self._cols)
        cph = _cox_fit(df_std[["_time", "_event"] + self._cols],
                       "_time", "_event", self.penalizer)
        self._mean, self._std = m, s
        self._beta = cph.params_.loc[self._cols]
        return self

    def score(self, X, clinical=None):
        df = X[self._cols].copy().fillna(0.0)
        df_std = _standardize_apply(df, self._cols, self._mean, self._std)
        return pd.Series(df_std.values @ self._beta.values, index=X.index, name=self.name)


# --- NMF + Cox risk model -------------------------------------------------


@dataclass
class NMFRiskModel(BenchmarkModel):
    """NMF factorization → Cox на loadings. Leakage-free.

    - Preprocess z-scoring: fit на train (medians/MADs), apply на test
    - NMF fit на train только
    - project(test) через NNLS на уже fitted H
    - Cox fit на train W
    - score(test) = linear combination loadings × β
    """
    k: int = 6
    n_runs: int = 10           # для benchmark — 10 достаточно, 30 в final run
    init: str = "nndsvda"
    penalizer: float = 0.05
    l1_ratio: float = 0.5
    seed: int = 42
    name: str = "NMF_risk"
    _H: Optional[pd.DataFrame] = field(default=None, init=False)
    _sig_medians: Optional[pd.Series] = field(default=None, init=False)
    _sig_mads: Optional[pd.Series] = field(default=None, init=False)
    _mean: Optional[pd.Series] = field(default=None, init=False)
    _std: Optional[pd.Series] = field(default=None, init=False)
    _beta: Optional[pd.Series] = field(default=None, init=False)
    _factor_cols: List[str] = field(default_factory=list, init=False)

    def _preprocess_fit(self, X_sig: pd.DataFrame) -> pd.DataFrame:
        med = X_sig.median(axis=0)
        mad = (X_sig - med).abs().median(axis=0)
        std_fallback = X_sig.std(axis=0)
        mad = mad.where(mad != 0, std_fallback).replace(0, 1.0)
        self._sig_medians = med
        self._sig_mads = mad
        scaled = (X_sig - med) / (1.4826 * mad)
        return scaled.clip(-10, 10)

    def _preprocess_apply(self, X_sig: pd.DataFrame) -> pd.DataFrame:
        scaled = (X_sig - self._sig_medians) / (1.4826 * self._sig_mads.replace(0, 1.0))
        return scaled.clip(-10, 10)

    @staticmethod
    def _shift_nonnegative(X: pd.DataFrame) -> pd.DataFrame:
        shift = X.min(axis=0).clip(upper=0).abs() + 1e-6
        return X.add(shift, axis=1)

    def fit(self, X, time, event, clinical=None):
        from sklearn.decomposition import NMF
        sig_cols = [c for c in X.columns if c.startswith("sig__")]
        X_sig = X[sig_cols]
        X_scaled = self._preprocess_fit(X_sig)
        X_nn = self._shift_nonnegative(X_scaled).values

        # multi-init → best-by-reconstruction-error
        best_err = np.inf
        best_W, best_H = None, None
        rng = np.random.default_rng(self.seed)
        for _ in range(self.n_runs):
            seed = int(rng.integers(0, 2**31 - 1))
            model = NMF(
                n_components=self.k, init=self.init, solver="cd",
                max_iter=500, random_state=seed, tol=1e-4,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                W = model.fit_transform(X_nn)
            if model.reconstruction_err_ < best_err:
                best_err = float(model.reconstruction_err_)
                best_W, best_H = W, model.components_

        self._H = pd.DataFrame(best_H, columns=sig_cols,
                                index=[f"factor_{i+1}" for i in range(self.k)])
        self._factor_cols = list(self._H.index)

        # нормализация W per-factor как в NMFResult
        W_df = pd.DataFrame(best_W, index=X.index, columns=self._factor_cols)
        W_min = W_df.min(axis=0)
        W_max = W_df.max(axis=0)
        denom = (W_max - W_min).replace(0, 1.0)
        W_norm = (W_df - W_min) / denom
        # сохраняем training-level min/max для inference
        self._W_min_train, self._W_max_train = W_min, W_max

        # Cox на W
        common = sorted(set(W_norm.index) & set(time.index) & set(event.index))
        df = pd.DataFrame({
            "_time": pd.to_numeric(time.loc[common], errors="coerce"),
            "_event": pd.to_numeric(event.loc[common], errors="coerce"),
        }, index=common)
        for c in self._factor_cols:
            df[c] = W_norm.loc[common, c]
        df = df.dropna()
        df_std, m, s = _standardize_fit(df, self._factor_cols)
        cph = _cox_fit(df_std[["_time", "_event"] + self._factor_cols],
                       "_time", "_event", self.penalizer, self.l1_ratio)
        self._mean, self._std = m, s
        self._beta = cph.params_.loc[self._factor_cols]
        return self

    def _project(self, X: pd.DataFrame) -> pd.DataFrame:
        """Test-fold projection через NNLS на fitted H."""
        from scipy.optimize import nnls
        sig_cols = list(self._H.columns)
        X_sig = X.reindex(columns=sig_cols, fill_value=0.0)
        X_scaled = self._preprocess_apply(X_sig)
        X_nn = self._shift_nonnegative(X_scaled).values
        H = self._H.values
        W_new = np.zeros((X_nn.shape[0], self.k))
        for i in range(X_nn.shape[0]):
            w, _ = nnls(H.T, X_nn[i], maxiter=200)
            W_new[i] = w
        W_df = pd.DataFrame(W_new, index=X.index, columns=self._factor_cols)
        denom = (self._W_max_train - self._W_min_train).replace(0, 1.0)
        return (W_df - self._W_min_train) / denom

    def score(self, X, clinical=None):
        W = self._project(X)
        df_std = _standardize_apply(W, self._factor_cols, self._mean, self._std)
        return pd.Series(df_std.values @ self._beta.values,
                         index=X.index, name=self.name)


# --- Combined model -------------------------------------------------------


class CombinedModel(BenchmarkModel):
    """Stacking: linear combination of two model scores via Cox refit.

    fit: fit A и B на train, получаем два score, refit Cox на [score_A, score_B].
    score: возвращает refit-based combination.
    """

    def __init__(self, model_a: BenchmarkModel, model_b: BenchmarkModel,
                 penalizer: float = 0.01, name: Optional[str] = None):
        self.model_a = model_a
        self.model_b = model_b
        self.penalizer = penalizer
        self.name = name or f"{model_a.name}+{model_b.name}"
        self._beta = None
        self._mean = None
        self._std = None

    def fit(self, X, time, event, clinical=None):
        self.model_a.fit(X, time, event, clinical)
        self.model_b.fit(X, time, event, clinical)
        sa = self.model_a.score(X, clinical)
        sb = self.model_b.score(X, clinical)
        common = sorted(
            set(sa.index) & set(sb.index) & set(time.index) & set(event.index)
        )
        df = pd.DataFrame({
            "_time": pd.to_numeric(time.loc[common], errors="coerce"),
            "_event": pd.to_numeric(event.loc[common], errors="coerce"),
            "a": sa.loc[common].values,
            "b": sb.loc[common].values,
        }).dropna()
        df_std, self._mean, self._std = _standardize_fit(df, ["a", "b"])
        cph = _cox_fit(df_std[["_time", "_event", "a", "b"]],
                       "_time", "_event", self.penalizer)
        self._beta = cph.params_.loc[["a", "b"]]
        return self

    def score(self, X, clinical=None):
        sa = self.model_a.score(X, clinical)
        sb = self.model_b.score(X, clinical)
        df = pd.DataFrame({"a": sa, "b": sb}).loc[X.index]
        df_std = _standardize_apply(df, ["a", "b"], self._mean, self._std)
        return pd.Series(df_std.values @ self._beta.values, index=df.index,
                         name=self.name)
