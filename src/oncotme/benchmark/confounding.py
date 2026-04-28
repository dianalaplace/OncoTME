"""Confounding-проверки:
- ``partial_correlation_check``: корреляция risk-score vs endpoint при контроле на
  клинические переменные. Если частичная корреляция → 0, наш score — прокси клиники.
- ``schoenfeld_check``: PH assumption через Schoenfeld residuals.
- ``nmf_init_sensitivity``: как меняется cophenetic при init='random' vs 'nndsvda'.
"""

from __future__ import annotations

import logging
import warnings
from typing import List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def partial_correlation_check(
    risk_score: pd.Series,
    time: pd.Series,
    event: pd.Series,
    clinical: pd.DataFrame,
    confounders: Optional[List[str]] = None,
) -> dict:
    """Partial Spearman корреляция risk vs (log) survival time, при контроле на confounders.

    Low partial corr + high raw corr → risk-score проксирует confounders, а не TME.
    """
    from scipy import stats
    if confounders is None:
        confounders = ["age", "stage", "subtype_er", "subtype_her2"]

    common = sorted(
        set(risk_score.index) & set(time.index) & set(event.index)
        & set(clinical.index)
    )
    rs = risk_score.loc[common].astype(float)
    t = pd.to_numeric(time.loc[common], errors="coerce")
    e = pd.to_numeric(event.loc[common], errors="coerce").astype(int)

    # убираем цензурированных для raw correlation
    obs = (e == 1)
    if obs.sum() < 20:
        return {"error": "Too few events for partial correlation."}

    raw_rho, raw_p = stats.spearmanr(rs[obs], t[obs])

    # partial rho через residuals (OLS)
    from statsmodels.api import OLS, add_constant
    confound_df = clinical.loc[common, [c for c in confounders if c in clinical.columns]]
    confound_df = confound_df.apply(pd.to_numeric, errors="coerce")
    confound_df = confound_df.fillna(confound_df.median(numeric_only=True))

    def _resid(y: pd.Series):
        Xc = add_constant(confound_df.loc[y.index])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = OLS(y, Xc).fit()
        return res.resid

    rs_res = _resid(rs[obs])
    t_res = _resid(t[obs])
    partial_rho, partial_p = stats.spearmanr(rs_res, t_res)

    return {
        "raw_spearman_rho": float(raw_rho),
        "raw_spearman_p": float(raw_p),
        "partial_spearman_rho": float(partial_rho),
        "partial_spearman_p": float(partial_p),
        "proxy_warning": bool(abs(partial_rho) < 0.5 * abs(raw_rho)),
        "n_events_used": int(obs.sum()),
    }


def schoenfeld_check(
    risk_score: pd.Series,
    time: pd.Series,
    event: pd.Series,
) -> dict:
    """Fit univariable Cox на risk score → PH test через Schoenfeld residuals."""
    from lifelines import CoxPHFitter
    from lifelines.statistics import proportional_hazard_test

    common = sorted(set(risk_score.index) & set(time.index) & set(event.index))
    df = pd.DataFrame({
        "_time": pd.to_numeric(time.loc[common], errors="coerce"),
        "_event": pd.to_numeric(event.loc[common], errors="coerce"),
        "risk": risk_score.loc[common].astype(float),
    }).dropna()
    # standardize risk for comparability
    df["risk"] = (df["risk"] - df["risk"].mean()) / (df["risk"].std() or 1.0)
    cph = CoxPHFitter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cph.fit(df, duration_col="_time", event_col="_event", show_progress=False)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ph = proportional_hazard_test(cph, df, time_transform="rank")
        p = float(ph.summary["p"].iloc[0]) if not ph.summary.empty else float("nan")
    except Exception as exc:
        logger.warning("Schoenfeld test failed: %s", exc)
        p = float("nan")
    return {
        "hr": float(cph.summary.loc["risk", "exp(coef)"]),
        "hr_ci_lo": float(cph.summary.loc["risk", "exp(coef) lower 95%"]),
        "hr_ci_hi": float(cph.summary.loc["risk", "exp(coef) upper 95%"]),
        "coefficient_p": float(cph.summary.loc["risk", "p"]),
        "ph_test_p": p,
        "ph_violated": bool(p < 0.01) if np.isfinite(p) else False,
    }


def nmf_init_sensitivity(
    X: pd.DataFrame,
    k: int,
    *,
    n_runs: int = 30,
    seed: int = 42,
) -> dict:
    """Cophenetic + stability при init='nndsvda' vs 'random'.

    Если cophenetic падает сильно → наш результат переоценён.
    """
    from ..discovery.nmf_factors import nmf_with_stability
    # ХAC: переписываем init через monkey-patch — быстро для sensitivity
    from ..discovery import nmf_factors as nmf_mod

    def _run(init_method: str):
        orig = nmf_mod._run_single_nmf
        def patched(X_nn, k, random_state, max_iter=500):
            from sklearn.decomposition import NMF
            model = NMF(
                n_components=k, init=init_method,
                solver="cd", beta_loss="frobenius",
                max_iter=max_iter, random_state=random_state, tol=1e-4,
            )
            W = model.fit_transform(X_nn)
            return W, model.components_, float(model.reconstruction_err_)
        nmf_mod._run_single_nmf = patched
        try:
            res = nmf_with_stability(X, k=k, n_runs=n_runs, random_state=seed)
            return {"cophenetic": res.cophenetic, "stability": res.stability,
                    "explained_variance": res.explained_variance}
        finally:
            nmf_mod._run_single_nmf = orig

    return {
        "nndsvda": _run("nndsvda"),
        "random": _run("random"),
    }
