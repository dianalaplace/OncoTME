"""Характеризация TME-архетипов.

Три шага:
1. Для каждого кластера вычисляем **mean z-scored signature profile** — это
   «фингерпринт» архетипа, готовый к heatmap-визуализации.
2. Автоматическое именование архетипа по топ-2 upregulated и топ-1 downregulated
   сигнатурам с использованием априорного каталога "immune / stromal / apm".
3. Распределения clinical / subtype / treatment по архетипам — contingency tables
   с χ² и Cramér's V, а также ANOVA для непрерывных.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


# Эвристика именования: какие сигнатуры определяют какой архетип.
# Используется только если доминируют в топ-3 upregulated.
_ARCHETYPE_HEURISTICS: List[tuple[frozenset[str], str]] = [
    (frozenset({"CD8_T_cell", "IFN_gamma", "TLS_signature"}), "inflamed"),
    (frozenset({"CD8_T_cell", "IFN_gamma", "checkpoint_score"}), "inflamed-checkpoint"),
    (frozenset({"TLS_signature", "B_cell"}), "TLS-rich"),
    (frozenset({"HLA_I_score", "APM_score"}), "antigen-presenting"),
    (frozenset({"CAF_proxy", "TGFb"}), "fibrotic-excluded"),
    (frozenset({"CAF_proxy", "hypoxia"}), "desmoplastic"),
    (frozenset({"M2_macrophage", "MDSC_proxy"}), "immunosuppressive"),
    (frozenset({"angiogenesis", "hypoxia"}), "angiogenic-hypoxic"),
]


@dataclass
class ArchetypeSummary:
    archetype_id: int
    name: str
    size: int
    top_up: List[str]
    top_down: List[str]
    profile: pd.Series            # mean z-score per signature
    clinical_table: pd.DataFrame  # descriptive stats by clinical/subtype/treatment


def _top_signatures(profile: pd.Series, k: int = 3) -> tuple[list[str], list[str]]:
    """Топ-k наиболее высоких и топ-k наиболее низких сигнатур."""
    sorted_sig = profile.sort_values()
    down = [s.replace("sig__", "") for s in sorted_sig.head(k).index]
    up = [s.replace("sig__", "") for s in sorted_sig.tail(k).index[::-1]]
    return up, down


def _name_cluster(top_up: List[str], top_down: List[str]) -> str:
    """Автоматическое имя по heuristics; fallback — сборное из топ-признаков."""
    up_set = set(top_up)
    # прямой match по эвристике
    for sig_set, name in _ARCHETYPE_HEURISTICS:
        if sig_set.issubset(up_set):
            # проверяем, что APM не подавлен (важно для "inflamed" и т.п.)
            if "APM_score" in top_down or "HLA_I_score" in top_down:
                return f"APM-low_{name}"
            return name
    # спец-случаи для desert / APM-low
    if ("APM_score" in top_down or "HLA_I_score" in top_down) and not set(top_up) & {
        "CD8_T_cell", "IFN_gamma", "TLS_signature", "B_cell"
    }:
        return "immune-desert_APM-low"
    if not set(top_up) & {"CD8_T_cell", "IFN_gamma", "TLS_signature", "B_cell",
                           "CAF_proxy", "TGFb", "angiogenesis"}:
        return "immune-desert"
    # generic fallback
    return f"{top_up[0]}-high_{top_down[0]}-low"


def name_archetypes(
    profiles: pd.DataFrame,
) -> pd.DataFrame:
    """Применяет автоматическое именование для каждого кластера.

    Parameters
    ----------
    profiles
        DataFrame cluster × signature с mean z-score.

    Returns
    -------
    DataFrame с колонками: cluster_id, name, top_up, top_down.
    """
    rows = []
    for cid in profiles.index:
        prof = profiles.loc[cid]
        up, down = _top_signatures(prof, k=3)
        name = _name_cluster(up, down)
        rows.append({
            "cluster_id": cid,
            "name": name,
            "top_up": ",".join(up),
            "top_down": ",".join(down),
        })
    # make names unique, append suffix if collision
    df = pd.DataFrame(rows)
    seen: dict[str, int] = {}
    renamed = []
    for n in df["name"]:
        if n in seen:
            seen[n] += 1
            renamed.append(f"{n}_{seen[n]}")
        else:
            seen[n] = 1
            renamed.append(n)
    df["name_unique"] = renamed
    return df


def _categorical_distribution(
    cluster_labels: pd.Series,
    col: pd.Series,
) -> tuple[pd.DataFrame, float, float]:
    """Contingency table + χ² + Cramér's V."""
    ct = pd.crosstab(cluster_labels, col.astype(str))
    if ct.shape[1] < 2:
        return ct, np.nan, np.nan
    chi2, p, dof, _ = stats.chi2_contingency(ct)
    n = ct.values.sum()
    min_dim = min(ct.shape) - 1
    cramer_v = np.sqrt(chi2 / (n * min_dim)) if n > 0 and min_dim > 0 else np.nan
    return ct, float(p), float(cramer_v)


def _continuous_distribution(
    cluster_labels: pd.Series,
    col: pd.Series,
) -> tuple[pd.DataFrame, float]:
    """Descriptive stats + one-way ANOVA p-value."""
    s = col.dropna()
    l = cluster_labels.loc[s.index]
    stats_df = s.groupby(l).agg(["mean", "std", "median", "count"])
    # ANOVA
    groups = [s[l == c].values for c in stats_df.index]
    if len(groups) < 2 or any(len(g) < 2 for g in groups):
        return stats_df, np.nan
    try:
        _, p = stats.f_oneway(*groups)
    except Exception:
        p = np.nan
    return stats_df, float(p)


def characterize_archetypes(
    features: pd.DataFrame,
    cluster_labels: pd.Series,
    clinical: pd.DataFrame,
) -> dict[str, pd.DataFrame | pd.Series]:
    """Полный профиль TME-архетипов.

    Parameters
    ----------
    features
        Z-scored signature matrix (выход ``preprocess_for_clustering``).
    cluster_labels
        Series sample_id → cluster_id (из ``ConsensusResult.labels``).
    clinical
        ``CohortBundle.clinical`` — для subtype/treatment distribution.

    Returns
    -------
    dict с ключами:
    - ``profiles`` — DataFrame cluster × signature (mean z-score)
    - ``names`` — DataFrame с автоматическими именами
    - ``clinical_distribution`` — descriptive stats и p-values по каждой клин. переменной
    - ``pairwise_distances`` — евклидово расстояние между центроидами
    """
    sig_cols = [c for c in features.columns if c.startswith("sig__")] or list(features.columns)
    df = features[sig_cols].copy()
    df["_cluster"] = cluster_labels.reindex(df.index).values

    profiles = df.groupby("_cluster").mean()
    profiles.index.name = "cluster_id"

    names = name_archetypes(profiles)

    # pairwise distance между центроидами
    from scipy.spatial.distance import squareform, pdist
    dist_mat = squareform(pdist(profiles.values, metric="euclidean"))
    pairwise = pd.DataFrame(dist_mat, index=profiles.index, columns=profiles.index)

    # Clinical / subtype / treatment distributions
    clin_results: dict[str, dict] = {}
    binary_vars = [
        "subtype_er", "subtype_pr", "subtype_her2",
        "targeted_therapy_received", "radiation_received", "neoadjuvant_received",
    ]
    continuous_vars = ["age", "stage", "ki67"]
    categorical_vars = ["subtype_pam50"]

    clin_aligned = clinical.reindex(cluster_labels.index)
    for var in binary_vars + categorical_vars:
        if var in clin_aligned.columns:
            ct, p, v = _categorical_distribution(cluster_labels, clin_aligned[var])
            clin_results[var] = {
                "table": ct,
                "p_value": p,
                "cramer_v": v,
                "kind": "categorical",
            }
    for var in continuous_vars:
        if var in clin_aligned.columns:
            stats_df, p = _continuous_distribution(cluster_labels, clin_aligned[var])
            clin_results[var] = {
                "stats": stats_df,
                "p_value": p,
                "kind": "continuous",
            }

    # BH FDR correction
    pvals = {k: v["p_value"] for k, v in clin_results.items() if not np.isnan(v.get("p_value", np.nan))}
    if pvals:
        keys = list(pvals.keys())
        raw_p = np.array([pvals[k] for k in keys])
        order = np.argsort(raw_p)
        n = len(raw_p)
        ranks = np.empty(n, dtype=int)
        ranks[order] = np.arange(1, n + 1)
        bh = np.minimum.accumulate((raw_p[order] * n / ranks[order])[::-1])[::-1]
        for i, k in enumerate(keys):
            clin_results[k]["q_value"] = float(min(1.0, bh[ranks[i] - 1]))

    return {
        "profiles": profiles,
        "names": names,
        "clinical_distribution": clin_results,
        "pairwise_distances": pairwise,
    }
