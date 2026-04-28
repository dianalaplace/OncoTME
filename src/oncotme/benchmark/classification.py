"""Classification benchmark — параллель survival-benchmark.

Изменения:
- ``.fit(X, y, clinical)`` вместо ``(X, time, event, clinical)``
- ``.predict_proba(X, clinical)`` → p(y=1) вместо risk-score
- Метрика: **AUC** (primary) + **AP** (average precision, robust to imbalance)
- CV: ``StratifiedKFold`` по y (не по censoring)
- Modely: ElasticNet-logistic, XGBoost, NMF+logistic, Clinical-only, Signature-only

Для этого этапа проекта (treatment-homogeneous chemo cohort с pCR)
достаточно тонкого wrapper'а над sklearn, без собственных Cox-heroics.
"""

from __future__ import annotations

import logging
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ClassificationModel(ABC):
    name: str = "abstract_clf"

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series,
            clinical: Optional[pd.DataFrame] = None) -> "ClassificationModel": ...

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame,
                      clinical: Optional[pd.DataFrame] = None) -> pd.Series: ...


# --- базовые helpers ------------------------------------------------------


def _clinical_design(clinical: pd.DataFrame, covariates: List[str]) -> pd.DataFrame:
    """Select + impute + align clinical covariates."""
    df = pd.DataFrame(index=clinical.index)
    for c in covariates:
        if c in clinical.columns:
            df[c] = pd.to_numeric(clinical[c], errors="coerce")
    return df.fillna(df.median(numeric_only=True))


def _signature_columns(X: pd.DataFrame) -> List[str]:
    return [c for c in X.columns if c.startswith("sig__")]


# --- Clinical-only ---------------------------------------------------------


@dataclass
class ClinicalLogitModel(ClassificationModel):
    covariates: List[str] = field(default_factory=lambda: [
        "age", "stage", "grade", "subtype_er", "subtype_pr", "subtype_her2",
    ])
    C: float = 1.0
    name: str = "clinical_only"

    def __post_init__(self):
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        self._scaler = StandardScaler()
        self._model = LogisticRegression(
            penalty="l2", C=self.C, solver="lbfgs", max_iter=1000,
            class_weight="balanced",
        )
        self._cols: List[str] = []

    def fit(self, X, y, clinical=None):
        if clinical is None:
            raise ValueError("ClinicalLogitModel requires clinical.")
        D = _clinical_design(clinical.loc[y.index], self.covariates)
        # drop constant columns
        self._cols = [c for c in D.columns if D[c].nunique() >= 2]
        if not self._cols:
            raise RuntimeError("No usable clinical covariates.")
        Z = self._scaler.fit_transform(D[self._cols].values)
        self._model.fit(Z, y.values.astype(int))
        return self

    def predict_proba(self, X, clinical=None):
        if clinical is None:
            raise ValueError("ClinicalLogitModel requires clinical.")
        D = _clinical_design(clinical, self.covariates)
        for c in self._cols:
            if c not in D.columns:
                D[c] = 0.0
        Z = self._scaler.transform(D[self._cols].values)
        p = self._model.predict_proba(Z)[:, 1]
        return pd.Series(p, index=D.index, name=self.name)


# --- TIS / CYT / signature baselines --------------------------------------


class _SingleFeatureLogit(ClassificationModel):
    """Generic: one scalar feature → logistic."""

    def __init__(self, name: str, compute_fn: Callable[[pd.DataFrame], pd.Series],
                 C: float = 1.0):
        self.name = name
        self._compute = compute_fn
        from sklearn.linear_model import LogisticRegression
        self._model = LogisticRegression(C=C, solver="lbfgs", max_iter=1000,
                                         class_weight="balanced")
        self._mean = None
        self._std = None

    def fit(self, X, y, clinical=None):
        s = self._compute(X.loc[y.index]).astype(float)
        self._mean = float(s.mean())
        self._std = float(s.std()) or 1.0
        Z = ((s - self._mean) / self._std).values.reshape(-1, 1)
        self._model.fit(Z, y.values.astype(int))
        return self

    def predict_proba(self, X, clinical=None):
        s = self._compute(X).astype(float)
        Z = ((s - self._mean) / self._std).values.reshape(-1, 1)
        return pd.Series(self._model.predict_proba(Z)[:, 1],
                         index=X.index, name=self.name)


def _tis_compute(X: pd.DataFrame) -> pd.Series:
    from ..features.baselines import TIS_GENES
    gene_cols = {c.replace("gene__", ""): c for c in X.columns if c.startswith("gene__")}
    matches = [gene_cols[g] for g in TIS_GENES if g in gene_cols]
    if len(matches) < 5:
        return pd.Series(0.0, index=X.index)
    sub = X[matches]
    z = sub.sub(sub.mean(axis=0), axis=1).div(sub.std(axis=0).replace(0, 1), axis=1)
    return z.mean(axis=1)


def _cyt_compute(X: pd.DataFrame) -> pd.Series:
    gzma = X.get("gene__GZMA")
    prf1 = X.get("gene__PRF1")
    if gzma is None or prf1 is None:
        return pd.Series(0.0, index=X.index)
    return np.sqrt(np.clip(gzma, 0, None) * np.clip(prf1, 0, None))


def TIS_Clf(): return _SingleFeatureLogit("TIS_18", _tis_compute)
def CYT_Clf(): return _SingleFeatureLogit("CYT", _cyt_compute)


def Signature_Clf(signature_col: str) -> ClassificationModel:
    def compute(X: pd.DataFrame) -> pd.Series:
        if signature_col not in X.columns:
            return pd.Series(0.0, index=X.index)
        return X[signature_col].astype(float)
    return _SingleFeatureLogit(f"sig_{signature_col.replace('sig__', '')}", compute)


# --- Random features ------------------------------------------------------


@dataclass
class RandomFeaturesClf(ClassificationModel):
    n_features: int = 6
    seed: int = 0
    C: float = 1.0
    name: str = "random_features"

    def __post_init__(self):
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        self._scaler = StandardScaler()
        self._model = LogisticRegression(
            penalty="l2", C=self.C, solver="lbfgs", max_iter=1000,
            class_weight="balanced",
        )
        self._cols: List[str] = []

    def fit(self, X, y, clinical=None):
        sig_cols = _signature_columns(X)
        rng = np.random.default_rng(self.seed)
        if len(sig_cols) <= self.n_features:
            self._cols = sig_cols
        else:
            self._cols = list(rng.choice(sig_cols, size=self.n_features, replace=False))
        Z = self._scaler.fit_transform(X.loc[y.index, self._cols].values)
        self._model.fit(Z, y.values.astype(int))
        return self

    def predict_proba(self, X, clinical=None):
        Z = self._scaler.transform(X[self._cols].values)
        return pd.Series(self._model.predict_proba(Z)[:, 1],
                         index=X.index, name=self.name)


# --- NMF + logistic --------------------------------------------------------


@dataclass
class NMFLogitModel(ClassificationModel):
    """NMF-loadings + logistic. Fit/project leakage-free, как NMFRiskModel."""
    k: int = 5
    n_runs: int = 5
    init: str = "nndsvda"
    C: float = 1.0
    seed: int = 42
    name: str = "NMF_logit"
    _H: Optional[pd.DataFrame] = field(default=None, init=False)
    _sig_medians: Optional[pd.Series] = field(default=None, init=False)
    _sig_mads: Optional[pd.Series] = field(default=None, init=False)
    _W_min: Optional[pd.Series] = field(default=None, init=False)
    _W_max: Optional[pd.Series] = field(default=None, init=False)
    _scaler: object = field(default=None, init=False)
    _model: object = field(default=None, init=False)

    def _preprocess_fit(self, X_sig):
        med = X_sig.median(axis=0)
        mad = (X_sig - med).abs().median(axis=0)
        std = X_sig.std(axis=0)
        mad = mad.where(mad != 0, std).replace(0, 1.0)
        self._sig_medians = med
        self._sig_mads = mad
        return ((X_sig - med) / (1.4826 * mad)).clip(-10, 10)

    def _preprocess_apply(self, X_sig):
        return ((X_sig - self._sig_medians) / (1.4826 * self._sig_mads.replace(0, 1.0))).clip(
            -10, 10
        )

    @staticmethod
    def _shift_nn(X):
        shift = X.min(axis=0).clip(upper=0).abs() + 1e-6
        return X.add(shift, axis=1)

    def fit(self, X, y, clinical=None):
        from sklearn.decomposition import NMF
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        sig_cols = _signature_columns(X)
        X_sig = X.loc[y.index, sig_cols]
        X_scaled = self._preprocess_fit(X_sig)
        X_nn = self._shift_nn(X_scaled).values

        best_err = np.inf
        best_W, best_H = None, None
        rng = np.random.default_rng(self.seed)
        for _ in range(self.n_runs):
            seed = int(rng.integers(0, 2**31 - 1))
            m = NMF(n_components=self.k, init=self.init, solver="cd",
                    max_iter=500, random_state=seed, tol=1e-4)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                W = m.fit_transform(X_nn)
            if m.reconstruction_err_ < best_err:
                best_err = float(m.reconstruction_err_)
                best_W, best_H = W, m.components_

        self._H = pd.DataFrame(best_H, columns=sig_cols,
                               index=[f"factor_{i+1}" for i in range(self.k)])
        W_df = pd.DataFrame(best_W, index=y.index, columns=self._H.index)
        self._W_min = W_df.min(axis=0)
        self._W_max = W_df.max(axis=0)
        denom = (self._W_max - self._W_min).replace(0, 1.0)
        W_norm = (W_df - self._W_min) / denom
        self._scaler = StandardScaler()
        Z = self._scaler.fit_transform(W_norm.values)
        self._model = LogisticRegression(
            penalty="l2", C=self.C, solver="lbfgs", max_iter=1000,
            class_weight="balanced",
        )
        self._model.fit(Z, y.values.astype(int))
        return self

    def _project(self, X):
        from scipy.optimize import nnls
        sig_cols = list(self._H.columns)
        X_sig = X.reindex(columns=sig_cols, fill_value=0.0)
        X_nn = self._shift_nn(self._preprocess_apply(X_sig)).values
        H = self._H.values
        W_new = np.zeros((X_nn.shape[0], self._H.shape[0]))
        for i in range(X_nn.shape[0]):
            w, _ = nnls(H.T, X_nn[i], maxiter=200)
            W_new[i] = w
        W_df = pd.DataFrame(W_new, index=X.index, columns=self._H.index)
        denom = (self._W_max - self._W_min).replace(0, 1.0)
        return (W_df - self._W_min) / denom

    def predict_proba(self, X, clinical=None):
        W = self._project(X)
        Z = self._scaler.transform(W.values)
        return pd.Series(self._model.predict_proba(Z)[:, 1],
                         index=X.index, name=self.name)


# --- Stacking ElasticNet on all signatures + clinical --------------------


@dataclass
class ElasticNetLogitModel(ClassificationModel):
    """ElasticNet logistic на всех ``sig__*`` + clinical.

    Более честная альтернатива NMF на маленьких когортах: penalized logistic
    выбирает relevant features сама. Это standard-ML baseline.
    """
    C: float = 0.5
    l1_ratio: float = 0.5
    use_clinical: bool = True
    clinical_covariates: List[str] = field(default_factory=lambda: [
        "age", "stage", "grade", "subtype_er", "subtype_pr", "subtype_her2",
    ])
    name: str = "elastic_net_logit"
    _cols: List[str] = field(default_factory=list, init=False)
    _scaler: object = field(default=None, init=False)
    _model: object = field(default=None, init=False)

    def fit(self, X, y, clinical=None):
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        sig_cols = _signature_columns(X)
        gene_cols = [c for c in X.columns if c.startswith("gene__")]
        feat_cols = sig_cols + gene_cols
        D = X.loc[y.index, feat_cols].copy()
        if self.use_clinical and clinical is not None:
            clin = _clinical_design(clinical.loc[y.index], self.clinical_covariates)
            # don't duplicate col names
            for c in clin.columns:
                D[f"clin_{c}"] = clin[c].values
        self._cols = [c for c in D.columns if D[c].nunique() >= 2]
        D = D[self._cols].fillna(0.0)
        self._scaler = StandardScaler()
        Z = self._scaler.fit_transform(D.values)
        self._model = LogisticRegression(
            penalty="elasticnet", l1_ratio=self.l1_ratio, C=self.C,
            solver="saga", max_iter=2000, class_weight="balanced",
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._model.fit(Z, y.values.astype(int))
        return self

    def predict_proba(self, X, clinical=None):
        D = pd.DataFrame(index=X.index)
        for c in self._cols:
            if c.startswith("clin_"):
                raw = c[len("clin_"):]
                if clinical is not None and raw in clinical.columns:
                    D[c] = pd.to_numeric(clinical[raw], errors="coerce")
                else:
                    D[c] = 0.0
            else:
                D[c] = X[c] if c in X.columns else 0.0
        D = D.fillna(0.0)
        Z = self._scaler.transform(D[self._cols].values)
        return pd.Series(self._model.predict_proba(Z)[:, 1],
                         index=X.index, name=self.name)


# --- Combined: clinical + TME-derived score -------------------------------


class ClfCombinedModel(ClassificationModel):
    """Stack two classifiers via second-level logistic on their p(y=1)."""

    def __init__(self, a: ClassificationModel, b: ClassificationModel,
                 C: float = 1.0, name: Optional[str] = None):
        from sklearn.linear_model import LogisticRegression
        self.a = a
        self.b = b
        self.name = name or f"{a.name}+{b.name}"
        self._meta = LogisticRegression(C=C, solver="lbfgs", max_iter=1000,
                                         class_weight="balanced")

    def fit(self, X, y, clinical=None):
        self.a.fit(X, y, clinical)
        self.b.fit(X, y, clinical)
        pa = self.a.predict_proba(X.loc[y.index], clinical)
        pb = self.b.predict_proba(X.loc[y.index], clinical)
        Z = np.vstack([pa.values, pb.values]).T
        self._meta.fit(Z, y.values.astype(int))
        return self

    def predict_proba(self, X, clinical=None):
        pa = self.a.predict_proba(X, clinical)
        pb = self.b.predict_proba(X, clinical)
        Z = np.vstack([pa.values, pb.values]).T
        return pd.Series(self._meta.predict_proba(Z)[:, 1],
                         index=X.index, name=self.name)


# --- Nested CV + permutation ---------------------------------------------


@dataclass
class ClfCVResult:
    model_name: str
    fold_auc: List[float]
    fold_ap: List[float]
    mean_auc: float
    auc_ci: tuple
    mean_ap: float
    n: int
    n_pos: int


def nested_cv_classification(
    factory: Callable[[], ClassificationModel],
    X: pd.DataFrame,
    y: pd.Series,
    clinical: Optional[pd.DataFrame] = None,
    *,
    n_folds: int = 5,
    seed: int = 42,
    n_bootstrap: int = 500,
) -> ClfCVResult:
    from sklearn.metrics import average_precision_score, roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    common = sorted(set(X.index) & set(y.dropna().index))
    if clinical is not None:
        common = sorted(set(common) & set(clinical.index))
    y_c = y.loc[common].astype(int)
    X_c = X.loc[common]
    clin_c = clinical.loc[common] if clinical is not None else None

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    aucs, aps = [], []
    oof = pd.Series(np.nan, index=common, dtype=float)
    name = "?"
    for fold, (tr, te) in enumerate(skf.split(common, y_c.values), start=1):
        tr_idx = [common[i] for i in tr]
        te_idx = [common[i] for i in te]
        m = factory()
        name = getattr(m, "name", "model")
        try:
            m.fit(X_c.loc[tr_idx], y_c.loc[tr_idx],
                  clin_c.loc[tr_idx] if clin_c is not None else None)
            p = m.predict_proba(X_c.loc[te_idx],
                                clin_c.loc[te_idx] if clin_c is not None else None)
            oof.loc[te_idx] = p.loc[te_idx].values
            a = float(roc_auc_score(y_c.loc[te_idx], p.loc[te_idx]))
            ap = float(average_precision_score(y_c.loc[te_idx], p.loc[te_idx]))
            aucs.append(a)
            aps.append(ap)
            logger.info("  %s fold %d: AUC=%.4f  AP=%.4f  n_test=%d",
                        name, fold, a, ap, len(te_idx))
        except Exception as exc:
            logger.warning("%s fold %d failed: %s", name, fold, exc)

    valid = oof.dropna()
    if len(valid) < 20:
        return ClfCVResult(name, aucs, aps, float(np.nanmean(aucs)),
                           (float("nan"), float("nan")), float(np.nanmean(aps)),
                           len(y_c), int(y_c.sum()))
    y_v = y_c.loc[valid.index]
    mean_auc = float(roc_auc_score(y_v, valid))
    # bootstrap CI
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(n_bootstrap):
        idx = rng.choice(len(valid), size=len(valid), replace=True)
        try:
            b = roc_auc_score(y_v.iloc[idx], valid.iloc[idx])
            boot.append(float(b))
        except Exception:
            continue
    ci = (float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))) \
        if boot else (float("nan"), float("nan"))
    mean_ap = float(average_precision_score(y_v, valid))
    return ClfCVResult(
        model_name=name, fold_auc=aucs, fold_ap=aps, mean_auc=mean_auc,
        auc_ci=ci, mean_ap=mean_ap, n=len(y_c), n_pos=int(y_c.sum()),
    )


def permutation_test_auc(
    factory: Callable,
    X: pd.DataFrame,
    y: pd.Series,
    observed_auc: float,
    clinical: Optional[pd.DataFrame] = None,
    *,
    n_permutations: int = 100,
    n_folds: int = 5,
    seed: int = 42,
) -> dict:
    rng = np.random.default_rng(seed)
    null = []
    common = sorted(set(X.index) & set(y.dropna().index))
    y_c = y.loc[common].astype(int)
    for i in range(n_permutations):
        y_perm = pd.Series(rng.permutation(y_c.values), index=common)
        try:
            r = nested_cv_classification(
                factory, X, y_perm, clinical,
                n_folds=n_folds, seed=seed + i, n_bootstrap=0,
            )
            null.append(r.mean_auc)
        except Exception:
            continue
        if (i + 1) % 10 == 0:
            logger.info("  perm %d/%d: null mean=%.3f max=%.3f",
                        i + 1, n_permutations, float(np.mean(null)), float(np.max(null)))
    null = np.array([n for n in null if np.isfinite(n)])
    if len(null) == 0:
        return {"observed": observed_auc, "p_value": float("nan")}
    return {
        "observed": float(observed_auc),
        "null_mean": float(np.mean(null)),
        "null_p95": float(np.quantile(null, 0.95)),
        "null_max": float(np.max(null)),
        "p_value": float(((null >= observed_auc).sum() + 1) / (len(null) + 1)),
        "n_success": int(len(null)),
        "null_aucs": null.tolist(),
    }
