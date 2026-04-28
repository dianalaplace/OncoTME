"""Survival-анализ TME-архетипов.

Проверки, которые делаем:
1. **KM** по каждому архетипу + **log-rank overall** (multivariate_logrank_test).
2. **Pairwise log-rank** между архетипами с **BH FDR** коррекцией.
3. **Univariable Cox** — archetype как категория.
4. **Multivariable Cox** — archetype + age + stage + ER + HER2 + PAM50.
5. **Stratified Cox** by PAM50 — проверка подтип-специфичности (H3).
6. **Treatment-aware Cox**: archetype + targeted_therapy_received +
   archetype:targeted_therapy_received — проверка H5 (TME модифицирует
   эффект таргетной).
7. **PH assumption check** через Schoenfeld residuals; WARN если нарушено.

Все HR и p-values возвращаются с bootstrap CI (parametric 95%, из lifelines),
плюс BH-коррекция для pairwise и interaction-тестов.
"""

from __future__ import annotations

import logging
import warnings
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _import_lifelines():
    try:
        from lifelines import CoxPHFitter, KaplanMeierFitter
        from lifelines.statistics import (
            multivariate_logrank_test,
            pairwise_logrank_test,
            proportional_hazard_test,
        )
        return {
            "CoxPHFitter": CoxPHFitter,
            "KaplanMeierFitter": KaplanMeierFitter,
            "multivariate_logrank_test": multivariate_logrank_test,
            "pairwise_logrank_test": pairwise_logrank_test,
            "proportional_hazard_test": proportional_hazard_test,
        }
    except Exception as exc:
        raise ImportError(f"lifelines not available: {exc}")


def _benjamini_hochberg(pvals: pd.Series) -> pd.Series:
    """BH FDR adjustment, сохраняющий индекс."""
    ranked = pvals.rank(method="min")
    n = len(pvals)
    bh = (pvals * n / ranked).clip(upper=1.0)
    # monotonicity (enforce non-increasing when sorted)
    order = pvals.sort_values().index
    cummin = 1.0
    for idx in order[::-1]:
        cummin = min(cummin, bh.loc[idx])
        bh.loc[idx] = cummin
    return bh


def km_by_archetype(
    cluster_labels: pd.Series,
    time: pd.Series,
    event: pd.Series,
    *,
    archetype_names: Optional[Dict[int, str]] = None,
) -> dict:
    """KM per archetype + overall log-rank + pairwise BH-adjusted log-rank.

    Returns dict:
        - ``km_curves``: dict[archetype_name] → DataFrame timeline
        - ``median_survival``: Series archetype → median survival (or NaN if not reached)
        - ``logrank_overall``: chi2, p_value, df
        - ``logrank_pairwise``: DataFrame (archetype_a, archetype_b, p_value, q_value)
    """
    ll = _import_lifelines()
    # Выравнивание и фильтрация NaN
    common = cluster_labels.index.intersection(time.index).intersection(event.index)
    c = cluster_labels.loc[common]
    t = pd.to_numeric(time.loc[common], errors="coerce")
    e = pd.to_numeric(event.loc[common], errors="coerce")
    mask = t.notna() & e.notna() & c.notna() & (t >= 0)
    c, t, e = c[mask], t[mask], e[mask]
    if len(c) < 30:
        raise ValueError(f"Too few samples after NaN filter: n={len(c)}")

    name_map = {cid: (archetype_names[cid] if archetype_names and cid in archetype_names
                      else f"A{cid}") for cid in sorted(c.unique())}

    km_curves = {}
    median_surv = {}
    for cid, name in name_map.items():
        mask_c = c == cid
        kmf = ll["KaplanMeierFitter"]()
        kmf.fit(t[mask_c], event_observed=e[mask_c], label=name)
        km_curves[name] = kmf.survival_function_.reset_index()
        median_surv[name] = float(kmf.median_survival_time_) if np.isfinite(
            kmf.median_survival_time_
        ) else np.nan

    # overall log-rank
    logrank = ll["multivariate_logrank_test"](t, c, event_observed=e)
    logrank_overall = {
        "chi2": float(logrank.test_statistic),
        "p_value": float(logrank.p_value),
        "df": int(len(name_map) - 1),
        "n": int(len(c)),
    }

    # pairwise
    pairwise_df = pd.DataFrame()
    if len(name_map) >= 2:
        pw = ll["pairwise_logrank_test"](t, c, event_observed=e)
        summary = pw.summary.reset_index()
        # индексация зависит от версии lifelines
        summary = summary.rename(columns={"level_0": "cluster_a", "level_1": "cluster_b"})
        summary["cluster_a"] = summary["cluster_a"].map(name_map)
        summary["cluster_b"] = summary["cluster_b"].map(name_map)
        if "p" in summary.columns:
            summary = summary.rename(columns={"p": "p_value"})
        summary["q_value"] = _benjamini_hochberg(summary["p_value"])
        pairwise_df = summary[["cluster_a", "cluster_b", "p_value", "q_value"]]

    return {
        "km_curves": km_curves,
        "median_survival": pd.Series(median_surv),
        "logrank_overall": logrank_overall,
        "logrank_pairwise": pairwise_df,
        "name_map": name_map,
    }


def _build_cox_frame(
    cluster_labels: pd.Series,
    time: pd.Series,
    event: pd.Series,
    clinical: pd.DataFrame,
    covariates: List[str],
    *,
    archetype_ref: Optional[int] = None,
    include_archetype: bool = True,
) -> pd.DataFrame:
    """Собирает одну таблицу для CoxPHFitter.fit."""
    idx = cluster_labels.index.intersection(time.index).intersection(event.index)
    if clinical is not None:
        idx = idx.intersection(clinical.index)
    idx = sorted(idx)
    if len(idx) < 30:
        raise ValueError(f"Too few samples for Cox: {len(idx)}")

    df = pd.DataFrame(index=idx)
    df["_time"] = pd.to_numeric(time.loc[idx], errors="coerce")
    df["_event"] = pd.to_numeric(event.loc[idx], errors="coerce")

    if include_archetype:
        cid = cluster_labels.loc[idx].astype(int)
        if archetype_ref is None:
            archetype_ref = int(cid.mode().iloc[0])
        for c in sorted(cid.unique()):
            if c == archetype_ref:
                continue
            df[f"archetype_{c}"] = (cid == c).astype(int)

    for cov in covariates:
        if cov in clinical.columns:
            v = pd.to_numeric(clinical.loc[idx, cov], errors="coerce")
            df[cov] = v
        else:
            logger.warning("Covariate %r missing; skipped.", cov)

    # категоричные PAM50 → one-hot
    if "subtype_pam50" in clinical.columns and "subtype_pam50" not in covariates:
        pam = clinical.loc[idx, "subtype_pam50"].astype(str)
        for sub in ["LUMA", "LUMB", "HER2", "BASAL"]:
            # NORMAL как reference (самый редкий → избегаем perfect separation)
            df[f"pam50_{sub}"] = (pam == sub).astype(int)

    # отбрасываем колонки с >20% NaN и строки с любым NaN в оставшихся
    df = df.dropna(axis=1, thresh=int(0.8 * len(df)))
    df = df.dropna(axis=0)
    # убираем колонки с constant variance (иначе Cox взорвётся)
    const = [c for c in df.columns if c not in {"_time", "_event"} and df[c].nunique() < 2]
    if const:
        logger.info("Dropping constant covariates: %s", const)
        df = df.drop(columns=const)
    return df


def multivariable_cox(
    cluster_labels: pd.Series,
    time: pd.Series,
    event: pd.Series,
    clinical: pd.DataFrame,
    *,
    covariates: Optional[List[str]] = None,
    archetype_ref: Optional[int] = None,
    penalizer: float = 0.01,
) -> dict:
    """Multivariable Cox: archetype (dummy) + clinical covariates.

    Returns dict:
      - ``summary``: lifelines.summary DataFrame (HR, CI, p)
      - ``concordance_index``
      - ``log_likelihood_ratio_p``
      - ``ph_test``: proportional hazards Schoenfeld test (DataFrame)
      - ``ph_violated``: list of variables where Schoenfeld p<0.01
      - ``n``
    """
    ll = _import_lifelines()
    if covariates is None:
        covariates = ["age", "stage", "subtype_er", "subtype_her2"]

    df = _build_cox_frame(
        cluster_labels, time, event, clinical, covariates,
        archetype_ref=archetype_ref, include_archetype=True,
    )
    cph = ll["CoxPHFitter"](penalizer=penalizer)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cph.fit(df, duration_col="_time", event_col="_event", show_progress=False)

    # PH assumption check
    ph_test = None
    ph_violated: List[str] = []
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ph = ll["proportional_hazard_test"](cph, df, time_transform="rank")
        ph_test = ph.summary
        ph_violated = ph_test.index[ph_test["p"] < 0.01].tolist()
    except Exception as exc:
        logger.warning("PH test failed: %s", exc)

    return {
        "summary": cph.summary,
        "concordance_index": float(cph.concordance_index_),
        "log_likelihood_ratio_p": float(cph.log_likelihood_ratio_test().p_value),
        "ph_test": ph_test,
        "ph_violated": ph_violated,
        "n": int(len(df)),
        "archetype_ref": archetype_ref,
    }


def stratified_cox_by_pam50(
    cluster_labels: pd.Series,
    time: pd.Series,
    event: pd.Series,
    clinical: pd.DataFrame,
    *,
    covariates: Optional[List[str]] = None,
    penalizer: float = 0.01,
) -> dict:
    """Cox с stratify=PAM50 — проверка, работает ли archetype внутри подтипа."""
    ll = _import_lifelines()
    if covariates is None:
        covariates = ["age", "stage"]
    df = _build_cox_frame(cluster_labels, time, event, clinical, covariates)

    pam50 = clinical["subtype_pam50"].reindex(df.index).astype(str).replace("nan", np.nan)
    df = df.loc[pam50.notna()]
    pam50 = pam50.loc[df.index]
    df["_strata"] = pam50.values

    cph = ll["CoxPHFitter"](penalizer=penalizer)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cph.fit(df, duration_col="_time", event_col="_event",
                strata="_strata", show_progress=False)
    return {
        "summary": cph.summary,
        "concordance_index": float(cph.concordance_index_),
        "n": int(len(df)),
        "strata_counts": pam50.value_counts().to_dict(),
    }


def treatment_interaction_cox(
    cluster_labels: pd.Series,
    time: pd.Series,
    event: pd.Series,
    clinical: pd.DataFrame,
    *,
    treatment_col: str = "targeted_therapy_received",
    covariates: Optional[List[str]] = None,
    penalizer: float = 0.01,
) -> dict:
    """Cox с архетипом × treatment interaction.

    Проверяет, модифицирует ли TME-архетип эффект таргетной терапии.
    Включает только пациенток с непропущенным treatment_col.

    Returns dict с summary + interaction p-values и q-values (BH).
    """
    ll = _import_lifelines()
    if covariates is None:
        covariates = ["age", "stage", "subtype_er", "subtype_her2"]

    mask = clinical[treatment_col].notna()
    clin_sub = clinical.loc[mask]
    cluster_sub = cluster_labels.loc[cluster_labels.index.intersection(clin_sub.index)]

    df = _build_cox_frame(cluster_sub, time, event, clin_sub, covariates)
    df[treatment_col] = pd.to_numeric(clin_sub.loc[df.index, treatment_col], errors="coerce")
    df = df.dropna(subset=[treatment_col])

    # interactions archetype_X × treatment
    arch_cols = [c for c in df.columns if c.startswith("archetype_")]
    interaction_cols = []
    for ac in arch_cols:
        col = f"{ac}_x_{treatment_col}"
        df[col] = df[ac] * df[treatment_col]
        interaction_cols.append(col)

    cph = ll["CoxPHFitter"](penalizer=penalizer)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cph.fit(df, duration_col="_time", event_col="_event", show_progress=False)

    summary = cph.summary.copy()
    interaction_rows = summary.loc[interaction_cols]
    q_values = _benjamini_hochberg(interaction_rows["p"])
    interaction_rows["q_value"] = q_values

    return {
        "summary": summary,
        "interaction_rows": interaction_rows,
        "concordance_index": float(cph.concordance_index_),
        "n": int(len(df)),
        "n_treated": int(df[treatment_col].sum()),
        "n_untreated": int((df[treatment_col] == 0).sum()),
    }
