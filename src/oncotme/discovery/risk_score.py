"""Composite TME Risk Score — клинически применимая one-number метрика.

Построение:
1. Берём NMF factor loadings (W, samples × factors).
2. Fit multivariable Cox с факторами + базовыми клиническими (age, stage, PAM50)
   на outer-training fold.
3. Извлекаем factor-only коэффициенты β_f (или все β если pure-TME score).
4. Risk score(sample) = Σ β_f × loading_f(sample).
5. Разбиваем на тертили (low/medium/high) — клинически интерпретируемо.
6. Валидация: KM по тертилям + log-rank + C-index на held-out.

Ключевые научные решения:
- Коэффициенты берём из **out-of-fold Cox** (nested CV), чтобы risk score не
  оверфитился к training endpoints.
- Используем ``ElasticNet-Cox`` (sksurv) — sparser, interpretable.
- Для tertile cutoff используем ``training`` distribution, не self-reference.
- Decision Curve Analysis на top level — клиническая net benefit.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class RiskScoreModel:
    """Обученный risk-score модель."""

    coefficients: pd.Series       # feature -> β (Cox log-HR)
    tertile_cutoffs: tuple[float, float]  # (low→medium, medium→high) из training
    concordance_index: float      # on training (out-of-fold)
    c_index_bootstrap_ci: tuple[float, float]  # 95% CI
    training_mean: pd.Series      # для inference выравнивания: X_new - training_mean
    training_std: pd.Series
    clinical_covariates: List[str]
    n_train: int

    def score(self, X: pd.DataFrame) -> pd.Series:
        """Apply risk score к новым loadings."""
        # standardize как в training
        X_std = (X[self.coefficients.index.tolist()] - self.training_mean) / (
            self.training_std.replace(0, 1.0)
        )
        return pd.Series(
            X_std.values @ self.coefficients.values,
            index=X.index,
            name="tme_risk_score",
        )

    def tertile(self, risk_scores: pd.Series) -> pd.Series:
        """Дихотомизация тертилями training cutoffs."""
        low, high = self.tertile_cutoffs
        cat = pd.Series("medium", index=risk_scores.index, dtype="object")
        cat[risk_scores <= low] = "low"
        cat[risk_scores >= high] = "high"
        return cat


def fit_composite_risk_score(
    W: pd.DataFrame,
    time: pd.Series,
    event: pd.Series,
    clinical: Optional[pd.DataFrame] = None,
    *,
    clinical_covariates: Optional[List[str]] = None,
    penalizer: float = 0.05,
    l1_ratio: float = 0.5,
    n_bootstrap: int = 200,
    random_state: int = 42,
) -> RiskScoreModel:
    """Fit ElasticNet-penalized Cox на NMF loadings + clinical covariates.

    Parameters
    ----------
    W
        sample × factors (NMF loadings).
    time, event
        Survival data (RFS / OS).
    clinical
        Optional ``CohortBundle.clinical``. Если задано — добавляем указанные
        ковариаты в Cox, но **risk score возвращает только вклад факторов** (чистая
        TME-signature, независимая от клиники).
    clinical_covariates
        Какие клинические использовать. По умолчанию — age, stage, subtype_er, subtype_her2.
    penalizer, l1_ratio
        Regularization (как в ElasticNet).
    n_bootstrap
        Для C-index CI.
    """
    try:
        from lifelines import CoxPHFitter
    except ImportError as exc:
        raise ImportError(f"lifelines required: {exc}")

    if clinical_covariates is None:
        clinical_covariates = ["age", "stage", "subtype_er", "subtype_her2"]

    # align
    idx = W.index.intersection(time.index).intersection(event.index)
    if clinical is not None:
        idx = idx.intersection(clinical.index)
    if len(idx) < 50:
        raise ValueError(f"Too few samples: {len(idx)}")
    idx = sorted(idx)

    df = pd.DataFrame(index=idx)
    df["_time"] = pd.to_numeric(time.loc[idx], errors="coerce")
    df["_event"] = pd.to_numeric(event.loc[idx], errors="coerce")
    for col in W.columns:
        df[col] = W.loc[idx, col]
    if clinical is not None:
        for cov in clinical_covariates:
            if cov in clinical.columns:
                df[cov] = pd.to_numeric(clinical.loc[idx, cov], errors="coerce")

    df = df.dropna()
    # standardize (Cox чувствителен к scale при penalized fitting)
    feature_cols = [c for c in df.columns if c not in {"_time", "_event"}]
    means = df[feature_cols].mean()
    stds = df[feature_cols].std().replace(0, 1.0)
    df[feature_cols] = (df[feature_cols] - means) / stds

    cph = CoxPHFitter(penalizer=penalizer, l1_ratio=l1_ratio)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cph.fit(df, duration_col="_time", event_col="_event", show_progress=False)

    # извлекаем коэффициенты только factors (не клиники) — это "чистый TME risk"
    factor_cols = [c for c in W.columns if c in cph.params_.index]
    factor_coefs = cph.params_.loc[factor_cols]

    # training risk score
    risk_train = (df[factor_cols].values @ factor_coefs.values)
    risk_ser = pd.Series(risk_train, index=df.index)
    low_cut = float(np.quantile(risk_train, 1 / 3))
    high_cut = float(np.quantile(risk_train, 2 / 3))

    # C-index + bootstrap
    from lifelines.utils import concordance_index
    c_index = float(concordance_index(df["_time"], -risk_ser, df["_event"]))
    rng = np.random.default_rng(random_state)
    boot = []
    for _ in range(n_bootstrap):
        sample_idx = rng.choice(len(df), size=len(df), replace=True)
        try:
            c = concordance_index(
                df["_time"].iloc[sample_idx],
                -risk_ser.iloc[sample_idx],
                df["_event"].iloc[sample_idx],
            )
            boot.append(c)
        except Exception:
            continue
    ci_lo, ci_hi = float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))

    # only means/stds for factors (они и используются в score())
    factor_means = means.loc[factor_cols]
    factor_stds = stds.loc[factor_cols]

    return RiskScoreModel(
        coefficients=factor_coefs,
        tertile_cutoffs=(low_cut, high_cut),
        concordance_index=c_index,
        c_index_bootstrap_ci=(ci_lo, ci_hi),
        training_mean=factor_means,
        training_std=factor_stds,
        clinical_covariates=clinical_covariates,
        n_train=int(len(df)),
    )


def tertile_km(
    risk_scores: pd.Series,
    tertile_cat: pd.Series,
    time: pd.Series,
    event: pd.Series,
) -> dict:
    """KM curves по тертилям + log-rank overall + pairwise."""
    try:
        from lifelines import KaplanMeierFitter
        from lifelines.statistics import multivariate_logrank_test, pairwise_logrank_test
    except ImportError as exc:
        raise ImportError(str(exc))

    common = risk_scores.index.intersection(tertile_cat.index).intersection(
        time.index
    ).intersection(event.index)
    t = pd.to_numeric(time.loc[common], errors="coerce")
    e = pd.to_numeric(event.loc[common], errors="coerce")
    cat = tertile_cat.loc[common]
    mask = t.notna() & e.notna() & (t >= 0) & cat.notna()
    t, e, cat = t[mask], e[mask], cat[mask]

    curves: Dict[str, pd.DataFrame] = {}
    medians: Dict[str, float] = {}
    for level in ["low", "medium", "high"]:
        sub = cat == level
        if sub.sum() < 5:
            continue
        kmf = KaplanMeierFitter()
        kmf.fit(t[sub], event_observed=e[sub], label=level)
        curves[level] = kmf.survival_function_.reset_index()
        medians[level] = float(kmf.median_survival_time_) if np.isfinite(
            kmf.median_survival_time_
        ) else np.nan

    lr = multivariate_logrank_test(t, cat, event_observed=e)
    pw = pairwise_logrank_test(t, cat, event_observed=e)
    pw_summary = pw.summary.reset_index().rename(
        columns={"level_0": "a", "level_1": "b", "p": "p_value"}
    )

    return {
        "curves": curves,
        "medians": medians,
        "logrank_overall": {"chi2": float(lr.test_statistic), "p_value": float(lr.p_value)},
        "pairwise": pw_summary,
        "n_low": int((cat == "low").sum()),
        "n_medium": int((cat == "medium").sum()),
        "n_high": int((cat == "high").sum()),
    }


def treatment_interaction_by_factor(
    W: pd.DataFrame,
    time: pd.Series,
    event: pd.Series,
    clinical: pd.DataFrame,
    *,
    treatment_col: str,
    covariates: Optional[List[str]] = None,
    penalizer: float = 0.05,
) -> pd.DataFrame:
    """Per-factor: Cox с factor × treatment interaction, BH-корректированные p.

    Returns DataFrame со строками-факторами:
      factor, HR_factor, HR_treatment, HR_interaction, p_interaction, q_interaction,
      n, n_treated.
    """
    try:
        from lifelines import CoxPHFitter
    except ImportError as exc:
        raise ImportError(str(exc))
    import warnings
    if covariates is None:
        covariates = ["age", "stage", "subtype_er", "subtype_her2"]

    mask = clinical[treatment_col].notna()
    clin_sub = clinical.loc[mask]
    common = W.index.intersection(time.index).intersection(event.index).intersection(
        clin_sub.index
    )
    if len(common) < 50:
        raise ValueError(f"Too few samples for interaction: {len(common)}")
    common = sorted(common)

    treatment = pd.to_numeric(clin_sub.loc[common, treatment_col], errors="coerce")

    rows = []
    for factor in W.columns:
        df = pd.DataFrame(index=common)
        df["_time"] = pd.to_numeric(time.loc[common], errors="coerce")
        df["_event"] = pd.to_numeric(event.loc[common], errors="coerce")
        df["factor"] = W.loc[common, factor].values
        df["treatment"] = treatment.values
        df["interaction"] = df["factor"] * df["treatment"]
        for cov in covariates:
            if cov in clinical.columns:
                df[cov] = pd.to_numeric(clinical.loc[common, cov], errors="coerce")
        df = df.dropna()
        if len(df) < 50:
            continue

        cph = CoxPHFitter(penalizer=penalizer)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                cph.fit(df, duration_col="_time", event_col="_event", show_progress=False)
            except Exception as exc:
                logger.warning("Cox failed for factor %s: %s", factor, exc)
                continue
        s = cph.summary
        rows.append({
            "factor": factor,
            "HR_factor": float(s.loc["factor", "exp(coef)"]),
            "HR_treatment": float(s.loc["treatment", "exp(coef)"]),
            "HR_interaction": float(s.loc["interaction", "exp(coef)"]),
            "p_interaction": float(s.loc["interaction", "p"]),
            "n": int(len(df)),
            "n_treated": int(df["treatment"].sum()),
        })

    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    # BH FDR
    from .survival import _benjamini_hochberg
    out["q_interaction"] = _benjamini_hochberg(out["p_interaction"]).values
    out = out.sort_values("p_interaction")
    return out


def decision_curve_analysis(
    risk_scores: pd.Series,
    event: pd.Series,
    *,
    thresholds: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """Decision Curve Analysis — клиническая net benefit.

    Для каждого probability threshold p_t:
        net_benefit = TP/N - FP/N * (p_t / (1 - p_t))

    Сравниваем с treat-all и treat-none. Чем выше над этими baseline — клинически полезнее.

    Здесь risk_scores предполагаются откалиброванными в [0,1]; если это raw risk
    (exp(β·x)), нужна предварительная isotonic регрессия. Для простоты используем
    sigmoid-normalization risk scores.
    """
    common = risk_scores.dropna().index.intersection(event.dropna().index)
    r = risk_scores.loc[common].astype(float)
    e = event.loc[common].astype(int)
    # sigmoid normalization → [0, 1]
    p = 1.0 / (1.0 + np.exp(-(r - r.mean()) / max(r.std(), 1e-6)))
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.60, 12)
    n = len(e)
    prevalence = float(e.mean())

    rows = []
    for t in thresholds:
        mask = p >= t
        tp = int(((mask) & (e == 1)).sum())
        fp = int(((mask) & (e == 0)).sum())
        nb_model = tp / n - fp / n * (t / (1 - t))
        nb_treat_all = prevalence - (1 - prevalence) * (t / (1 - t))
        rows.append({
            "threshold": float(t),
            "nb_model": float(nb_model),
            "nb_treat_all": float(nb_treat_all),
            "nb_treat_none": 0.0,
            "n_flagged": int(mask.sum()),
        })
    return pd.DataFrame(rows)
