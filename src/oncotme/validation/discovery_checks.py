"""Проверки специфичные для discovery-пайплайна.

Ловят научные ошибки, на которых рассыпаются статьи:
- Trivial кластеризация (один огромный + micro-outliers)
- Архетип, полностью редуцируемый к подтипу (cluster ≈ PAM50)
- Нестабильный выбор k (PAC слишком высокий)
- Нарушение proportional hazards assumption в Cox
- Недостаточная event-rate для survival инференса
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from .checks import _fail, _pass, _warn
from .result import CheckResult, CheckStatus


def check_consensus_stability(
    pac: float,
    k: int,
    *,
    max_pac: float = 0.25,
) -> CheckResult:
    """PAC ≤ max_pac → стабильная кластеризация.

    Șenbabaoğlu 2014: типичный порог для publication-grade = 0.2..0.3.
    """
    name = f"discovery.consensus_stability.k={k}"
    if pac > max_pac:
        return _fail(
            name,
            f"PAC={pac:.3f} > {max_pac}. Clustering unstable at k={k}; "
            "try different k or base clustering method.",
            pac=pac,
        )
    return _pass(name, f"PAC={pac:.3f} ≤ {max_pac} (stable).", pac=pac)


def check_cluster_sizes(
    sizes: Dict[int, int],
    *,
    min_fraction: float = 0.05,
) -> CheckResult:
    """Минимальный кластер ≥ min_fraction от N — иначе trivial outlier-кластер."""
    name = "discovery.cluster_sizes"
    total = sum(sizes.values())
    min_size = min(sizes.values())
    min_frac = min_size / max(total, 1)
    if min_frac < min_fraction:
        return _warn(
            name,
            f"Smallest cluster {min_size}/{total} ({min_frac:.1%}) < {min_fraction:.0%}. "
            "Likely outlier cluster; consider k-1 or merging.",
            sizes=sizes, min_fraction=min_frac,
        )
    return _pass(name, f"All clusters ≥ {min_fraction:.0%} of N (min={min_frac:.1%}).",
                 sizes=sizes)


def check_subtype_redundancy(
    cluster_labels: pd.Series,
    subtype: pd.Series,
    *,
    max_single_subtype_fraction: float = 0.95,
) -> CheckResult:
    """Если кластер полностью состоит из одного подтипа (PAM50) — он редуцируем к подтипу."""
    name = "discovery.subtype_redundancy"
    common = cluster_labels.index.intersection(subtype.dropna().index)
    if len(common) < 20:
        return _warn(name, "Too few samples with PAM50 label to assess redundancy.")
    c = cluster_labels.loc[common]
    s = subtype.loc[common].astype(str)
    redundant: List[dict] = []
    for cid in sorted(c.unique()):
        sub = s[c == cid]
        if len(sub) < 5:
            continue
        top_frac = sub.value_counts(normalize=True).iloc[0]
        top_sub = sub.value_counts(normalize=True).index[0]
        if top_frac > max_single_subtype_fraction:
            redundant.append({"cluster": int(cid), "subtype": top_sub,
                              "fraction": float(top_frac)})
    if redundant:
        return _warn(
            name,
            f"{len(redundant)} cluster(s) dominated >95% by single PAM50 subtype — "
            f"potentially redundant with subtype: {redundant}",
            redundant=redundant,
        )
    return _pass(name, "No cluster is a mere subtype re-labeling.")


def check_event_rate(
    event: pd.Series,
    *,
    min_events: int = 20,
    min_event_rate: float = 0.05,
) -> CheckResult:
    """Для Cox/KM: достаточное число событий для надёжного инференса."""
    name = "discovery.event_rate"
    e = pd.to_numeric(event, errors="coerce").dropna().astype(int)
    if e.empty:
        return _fail(name, "Event column is empty.")
    n_events = int(e.sum())
    rate = n_events / len(e)
    if n_events < min_events or rate < min_event_rate:
        return _warn(
            name,
            f"Only {n_events}/{len(e)} events ({rate:.1%}). "
            f"Survival inference under-powered; HR estimates unreliable.",
            n_events=n_events, event_rate=rate,
        )
    return _pass(name, f"{n_events}/{len(e)} events ({rate:.1%}).")


def check_ph_assumption(ph_violated: List[str]) -> CheckResult:
    """PH assumption: WARN если есть нарушения (Schoenfeld p<0.01)."""
    name = "discovery.cox_ph_assumption"
    if not ph_violated:
        return _pass(name, "Schoenfeld residuals: no PH violation (all p≥0.01).")
    return _warn(
        name,
        f"{len(ph_violated)} covariates violate PH assumption: {ph_violated}. "
        "Consider time-varying Cox, stratification, or non-linear terms.",
        violated=ph_violated,
    )


def check_nmf_stability(
    cophenetic: float,
    stability: float,
    k: int,
    *,
    min_cophenetic: float = 0.90,
    min_stability: float = 0.80,
) -> CheckResult:
    """Brunet 2004 + практические пороги стабильности NMF.

    Если cophenetic < min_cophenetic — НЕ FAIL (NMF всё ещё интерпретируем
    как soft factors), но WARN. Коммуникация клинически результата требует
    четкого сообщения.
    """
    name = f"discovery.nmf_stability.k={k}"
    if cophenetic >= min_cophenetic and stability >= min_stability:
        return _pass(
            name,
            f"Stable: cophenetic={cophenetic:.3f}, stability={stability:.3f}.",
            cophenetic=cophenetic, stability=stability,
        )
    return _warn(
        name,
        f"Low stability at k={k}: cophenetic={cophenetic:.3f} (<{min_cophenetic}), "
        f"stability={stability:.3f} (<{min_stability}). Factors are interpretable "
        f"but between-run agreement is imperfect; report CI for downstream inference.",
        cophenetic=cophenetic, stability=stability,
    )


def check_risk_score_discriminates(
    c_index: float,
    ci_lo: float,
    *,
    min_c_index: float = 0.55,
) -> CheckResult:
    """Risk score C-index должен быть существенно >0.5 с нижним CI >0.5."""
    name = "discovery.risk_score_discrimination"
    if c_index < min_c_index or ci_lo < 0.5:
        return _fail(
            name,
            f"C-index={c_index:.3f} (95% CI lower={ci_lo:.3f}). "
            f"Risk score does not discriminate better than random.",
            c_index=c_index, ci_lo=ci_lo,
        )
    return _pass(
        name,
        f"C-index={c_index:.3f} (95% CI lower={ci_lo:.3f}) > 0.5.",
        c_index=c_index, ci_lo=ci_lo,
    )


def check_tertile_separation(
    logrank_p: float,
    n_low: int, n_med: int, n_high: int,
    *,
    max_p: float = 0.01,
    min_per_tertile: int = 30,
) -> CheckResult:
    """Тертили должны: (а) быть ≥30 в каждой группе (статистическая мощность),
    (б) иметь log-rank p < 0.01 (клинически значимая separation)."""
    name = "discovery.tertile_separation"
    if min(n_low, n_med, n_high) < min_per_tertile:
        return _warn(
            name,
            f"Small tertile sizes: low={n_low}, med={n_med}, high={n_high} "
            f"(<{min_per_tertile}). Statistical power low.",
            sizes=[n_low, n_med, n_high],
        )
    if logrank_p > max_p:
        return _warn(
            name,
            f"Tertile log-rank p={logrank_p:.3g} > {max_p}. "
            f"Weak survival separation; consider alternative cutoffs or more factors.",
            p_value=logrank_p,
        )
    return _pass(
        name,
        f"Log-rank p={logrank_p:.2e}; tertile sizes: {n_low}/{n_med}/{n_high}.",
        p_value=logrank_p,
    )


def check_classifier_transfer_sanity(
    classifier,
    X_internal: pd.DataFrame,
    labels_internal: pd.Series,
    *,
    min_accuracy: float = 0.85,
) -> CheckResult:
    """Self-prediction accuracy: ``predict(train)`` должна совпадать с исходными labels.

    Если нет — classifier учит не то, что должен.
    """
    name = "discovery.classifier_selfpredict_accuracy"
    preds = classifier.predict(X_internal)
    common = preds.index.intersection(labels_internal.index)
    acc = float((preds.loc[common] == labels_internal.loc[common]).mean())
    if acc < min_accuracy:
        return _fail(
            name,
            f"Classifier self-predict accuracy {acc:.3f} < {min_accuracy}. "
            "Centroids don't capture clusters; likely non-convex or poorly-separated.",
            accuracy=acc,
        )
    return _pass(name, f"Self-predict accuracy {acc:.3f} ≥ {min_accuracy}.",
                 accuracy=acc)
