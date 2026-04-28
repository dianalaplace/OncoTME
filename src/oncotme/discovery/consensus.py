"""Consensus clustering (Monti et al. 2003) с выбором k через PAC (Șenbabaoğlu 2014).

Метод:
1. Для каждого k в заданном диапазоне:
   a. ``n_bootstrap`` раз: subsample 80% объектов, применяем base clustering
      (Ward hierarchical по correlation distance), записываем кто с кем попал
      в один кластер.
   b. ``consensus_matrix[i, j]`` = доля бутстрапов, в которых i и j были в одном кластере
      (среди тех бутстрапов, где оба объекта присутствовали).
2. Извлекаем финальные лейблы через hierarchical clustering по ``1 - consensus_matrix``.
3. Выбираем оптимальный k:
   - **PAC** (Proportion of Ambiguous Clustering) — доля записей консенсус-матрицы
     в "серой зоне" (0.1, 0.9). Минимум PAC → самая стабильная партиция.
   - **ΔCDF area** (Monti) — изменение площади под CDF с ростом k; плато → k*.
   - Silhouette по consensus distance — побочный sanity-check.

Референс:
- Monti S, et al. Machine Learning 2003; 52: 91-118.
- Șenbabaoğlu Y, et al. Scientific Reports 2014; 4: 6207.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

logger = logging.getLogger(__name__)


@dataclass
class ConsensusResult:
    """Результат consensus clustering для одного значения k."""

    k: int
    consensus_matrix: np.ndarray   # n × n
    labels: np.ndarray              # длина n, значения 1..k
    pac: float                      # Proportion of Ambiguous Clustering [0,1]; low = stable
    cdf_area: float                 # Area под CDF консенсус-матрицы
    silhouette: float               # По consensus distance; [-1,1]; high = stable
    cluster_sizes: Dict[int, int]   # размер каждого кластера

    @property
    def min_cluster_fraction(self) -> float:
        total = sum(self.cluster_sizes.values())
        return min(self.cluster_sizes.values()) / max(total, 1)


def _correlation_distance(X: pd.DataFrame) -> np.ndarray:
    """1 - Pearson correlation между строками (samples). Condensed form."""
    # (X - mean) / std по строке
    centered = X.sub(X.mean(axis=1), axis=0)
    std = X.std(axis=1, ddof=0).replace(0, np.nan)
    norm = centered.div(std, axis=0).fillna(0.0)
    # корреляция через матричное умножение
    n = norm.shape[0]
    corr = (norm.values @ norm.values.T) / norm.shape[1]
    np.fill_diagonal(corr, 1.0)
    corr = np.clip(corr, -1.0, 1.0)
    dist = 1.0 - corr
    # condensed form для scipy.linkage
    return squareform(dist, checks=False)


def _one_bootstrap_labels(
    X: pd.DataFrame,
    k: int,
    subsample_idx: np.ndarray,
    linkage_method: str = "average",
) -> np.ndarray:
    """Один бутстрап — hierarchical clustering на subsample.

    Выбор distance:
    - ``ward`` — euclidean (scipy требование)
    - ``average`` / ``complete`` — correlation distance (классика для gene expression).
    """
    Xs = X.iloc[subsample_idx]
    if linkage_method == "ward":
        Z = linkage(Xs.values, method="ward")
    else:
        cond = _correlation_distance(Xs)
        Z = linkage(cond, method=linkage_method)
    labels = fcluster(Z, t=k, criterion="maxclust")
    return labels


def _pac(consensus: np.ndarray, low: float = 0.1, high: float = 0.9) -> float:
    """Proportion of Ambiguous Clustering (Șenbabaoğlu 2014).

    Доля off-diagonal записей consensus matrix, попадающих в "серую зону"
    (low, high). Низкий PAC = стабильная кластеризация (всё либо 0 либо 1).
    """
    n = consensus.shape[0]
    iu = np.triu_indices(n, k=1)
    values = consensus[iu]
    return float(np.mean((values > low) & (values < high)))


def _cdf_area(consensus: np.ndarray) -> float:
    """Площадь под CDF off-diagonal элементов (Monti 2003)."""
    n = consensus.shape[0]
    iu = np.triu_indices(n, k=1)
    values = np.sort(consensus[iu])
    # trapezoidal rule: x = values, y = linearly spaced empirical CDF [1/m..1]
    m = len(values)
    if m == 0:
        return 0.0
    y = np.arange(1, m + 1) / m
    trap = getattr(np, "trapezoid", None) or getattr(np, "trapz")
    area = trap(y, values)
    return float(area)


def _silhouette_from_consensus(consensus: np.ndarray, labels: np.ndarray) -> float:
    """Silhouette score на distance = 1 - consensus."""
    try:
        from sklearn.metrics import silhouette_score
    except Exception:
        return float("nan")
    dist = 1.0 - consensus
    np.fill_diagonal(dist, 0.0)
    try:
        return float(silhouette_score(dist, labels, metric="precomputed"))
    except ValueError:
        return float("nan")


def consensus_cluster(
    X: pd.DataFrame,
    k_range: Iterable[int] = range(2, 7),
    *,
    n_bootstrap: int = 100,
    subsample_fraction: float = 0.8,
    linkage_method: str = "ward",
    random_state: int = 42,
) -> Dict[int, ConsensusResult]:
    """Consensus clustering для каждого k в ``k_range``.

    Returns
    -------
    dict {k: ConsensusResult}.
    """
    k_list = list(k_range)
    n = X.shape[0]
    if n < 20:
        raise ValueError(f"Too few samples ({n}) for consensus clustering.")

    rng = np.random.default_rng(random_state)
    results: Dict[int, ConsensusResult] = {}

    for k in k_list:
        logger.info("Consensus clustering k=%d (bootstrap=%d) …", k, n_bootstrap)
        M = np.zeros((n, n), dtype=np.float64)   # number of co-assignments
        I = np.zeros((n, n), dtype=np.float64)   # number of times pair was co-sampled

        subsample_n = int(subsample_fraction * n)
        for b in range(n_bootstrap):
            idx = rng.choice(n, size=subsample_n, replace=False)
            labels = _one_bootstrap_labels(X, k, idx, linkage_method=linkage_method)
            # co-assignment update
            for c in np.unique(labels):
                members = idx[labels == c]
                M[np.ix_(members, members)] += 1.0
            I[np.ix_(idx, idx)] += 1.0

        consensus = np.divide(M, I, out=np.zeros_like(M), where=I > 0)
        np.fill_diagonal(consensus, 1.0)

        # финальные labels из consensus: hierarchical на 1 - consensus
        cond = squareform(1.0 - consensus, checks=False)
        Z = linkage(cond, method="average")
        final_labels = fcluster(Z, t=k, criterion="maxclust")

        sizes = {int(c): int((final_labels == c).sum()) for c in np.unique(final_labels)}
        results[k] = ConsensusResult(
            k=k,
            consensus_matrix=consensus,
            labels=final_labels,
            pac=_pac(consensus),
            cdf_area=_cdf_area(consensus),
            silhouette=_silhouette_from_consensus(consensus, final_labels),
            cluster_sizes=sizes,
        )
        logger.info(
            "  k=%d: PAC=%.3f, CDF_area=%.3f, silhouette=%.3f, sizes=%s",
            k, results[k].pac, results[k].cdf_area,
            results[k].silhouette, sizes,
        )

    return results


def select_k_by_pac(
    results: Dict[int, ConsensusResult],
    *,
    min_cluster_fraction: float = 0.03,
    prefer_higher_k_if_ties: bool = False,
) -> int:
    """Выбирает оптимальное k по min PAC, с требованием min_cluster_fraction.

    Parameters
    ----------
    min_cluster_fraction
        Если наименьший кластер при данном k < min_cluster_fraction * N,
        этот k не рассматривается (trivial: один микро-кластер из outliers).
    prefer_higher_k_if_ties
        При равных PAC — выбрать большее k (более детальная кластеризация).
    """
    candidates: List[tuple[int, float]] = []
    for k, res in results.items():
        if res.min_cluster_fraction < min_cluster_fraction:
            logger.info(
                "k=%d rejected: min cluster fraction %.3f < %.3f (sizes=%s)",
                k, res.min_cluster_fraction, min_cluster_fraction, res.cluster_sizes,
            )
            continue
        candidates.append((k, res.pac))

    if not candidates:
        raise ValueError(
            "No k candidate passed min_cluster_fraction filter. "
            "Try larger subsample or different linkage."
        )

    min_pac = min(c[1] for c in candidates)
    best = [k for k, p in candidates if abs(p - min_pac) < 1e-6]
    if prefer_higher_k_if_ties:
        return max(best)
    return min(best)


def delta_cdf_curve(results: Dict[int, ConsensusResult]) -> pd.DataFrame:
    """Δ CDF area по Monti: (A(k) - A(k-1)) / A(k-1)."""
    ks = sorted(results.keys())
    rows = []
    for i, k in enumerate(ks):
        a_k = results[k].cdf_area
        if i == 0:
            rows.append({"k": k, "cdf_area": a_k, "delta_k": np.nan})
        else:
            a_prev = results[ks[i - 1]].cdf_area
            delta = (a_k - a_prev) / a_prev if a_prev > 0 else np.nan
            rows.append({"k": k, "cdf_area": a_k, "delta_k": delta})
    return pd.DataFrame(rows)
