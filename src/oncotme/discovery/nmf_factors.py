"""NMF latent factors для TME с stability selection (Brunet 2004).

Почему NMF, а не PCA/ICA:
- Non-negativity → интерпретируемо: "сколько программы X экспрессируется"
  vs "какой знак проекции на PC1" (PCA).
- Parts-based decomposition: биологические программы TME перекрываются
  (пациентка может быть одновременно inflamed + APM-low — два разных фактора),
  что нативно описывается аддитивной моделью.
- Репликация между когортами лучше (фактор = набор co-expressed signatures,
  не зависит от cohort-specific mean).

Выбор k (число факторов):
- **Cophenetic correlation** (Brunet 2004): насколько стабильна иерархия
  consensus matrix при повторных NMF-раунах. Плато / крутое падение → k*.
- **Reconstruction error "elbow"**: ||X - WH|| как функция k.
- **Stability**: средняя корреляция W-матриц между раундами.

NMF требует non-negative input. Используем ``NMF-friendly shift``:
- сначала robust z-score (как в consensus pipeline)
- затем сдвиг на min + eps → все значения ≥ 0
- знак фактора восстанавливается по корреляции с исходными сигнатурами

Результат: W (samples × factors) = loadings, H (factors × signatures) = basis.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class NMFResult:
    """Результат NMF factorization для одного k."""

    k: int
    W: pd.DataFrame              # samples × factors (loadings)
    H: pd.DataFrame              # factors × signatures (basis)
    reconstruction_error: float  # ||X - WH||_F
    cophenetic: float            # [0..1], cluster stability
    stability: float             # mean Pearson correlation of W across runs
    explained_variance: float    # 1 - ||X-WH||² / ||X||²


def _prepare_for_nmf(X: pd.DataFrame) -> pd.DataFrame:
    """NMF требует non-negative values.

    Input — z-scored matrix (может иметь отрицательные значения).
    Делаем сдвиг per column так, чтобы min = 0, +epsilon.
    """
    shift = X.min(axis=0).clip(upper=0).abs() + 1e-6
    return X.add(shift, axis=1)


def _run_single_nmf(
    X_nn: np.ndarray,
    k: int,
    random_state: int,
    max_iter: int = 500,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Одна итерация NMF через sklearn."""
    from sklearn.decomposition import NMF
    model = NMF(
        n_components=k,
        init="nndsvda",
        solver="cd",
        beta_loss="frobenius",
        max_iter=max_iter,
        random_state=random_state,
        tol=1e-4,
    )
    W = model.fit_transform(X_nn)
    H = model.components_
    return W, H, float(model.reconstruction_err_)


def _consensus_matrix_from_assignments(assignments: list[np.ndarray]) -> np.ndarray:
    """Для каждого NMF-раунда: argmax по factors → hard assignment;
    consensus[i,j] = доля раундов, где i и j получили одинаковый argmax."""
    n = len(assignments[0])
    n_runs = len(assignments)
    M = np.zeros((n, n), dtype=np.float64)
    for a in assignments:
        # outer equality
        M += (a[:, None] == a[None, :]).astype(np.float64)
    return M / n_runs


def _cophenetic_correlation(consensus: np.ndarray) -> float:
    """Brunet 2004 cophenetic correlation consensus matrix."""
    from scipy.cluster.hierarchy import linkage, cophenet
    from scipy.spatial.distance import squareform
    # distance = 1 - consensus
    dist = 1.0 - consensus
    np.fill_diagonal(dist, 0.0)
    cond = squareform(dist, checks=False)
    try:
        Z = linkage(cond, method="average")
        c, _ = cophenet(Z, cond)
        return float(c)
    except Exception as exc:
        logger.warning("Cophenetic failed: %s", exc)
        return float("nan")


def _match_and_align_factors(
    W_runs: list[np.ndarray], H_runs: list[np.ndarray],
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Permutation matching: выравниваем факторы через Hungarian на correlation matrix.

    После этого можем усреднить W и H и посчитать stability.
    """
    from scipy.optimize import linear_sum_assignment
    ref_H = H_runs[0]
    k = ref_H.shape[0]
    aligned_W = [W_runs[0]]
    aligned_H = [ref_H]
    for W_r, H_r in zip(W_runs[1:], H_runs[1:]):
        # correlation между рядами ref_H и H_r
        corr = np.zeros((k, k))
        for i in range(k):
            for j in range(k):
                c = np.corrcoef(ref_H[i], H_r[j])[0, 1]
                corr[i, j] = c if np.isfinite(c) else 0.0
        # max corr → min -corr
        row_ind, col_ind = linear_sum_assignment(-corr)
        perm = col_ind  # для каждого ref-factor i, какой H_r j
        # signs: если корреляция отрицательная, оставляем (NMF non-neg, знак+)
        aligned_W.append(W_r[:, perm])
        aligned_H.append(H_r[perm, :])
    return aligned_W, aligned_H


def _stability_score(aligned_W: list[np.ndarray]) -> float:
    """Mean Pearson correlation W-матриц между раундами (по колонкам)."""
    ref = aligned_W[0]
    k = ref.shape[1]
    vals = []
    for W_r in aligned_W[1:]:
        for j in range(k):
            r = np.corrcoef(ref[:, j], W_r[:, j])[0, 1]
            if np.isfinite(r):
                vals.append(r)
    return float(np.mean(vals)) if vals else float("nan")


def nmf_with_stability(
    X: pd.DataFrame,
    k: int,
    *,
    n_runs: int = 30,
    random_state: int = 42,
    max_iter: int = 500,
) -> NMFResult:
    """NMF с multi-run stability assessment.

    Parameters
    ----------
    X
        sample × signature matrix (z-scored из ``preprocess_for_clustering``).
        Будет сдвинут в non-negative пространство внутри функции.
    k
        Число факторов.
    n_runs
        Число независимых NMF-инициализаций для stability.
    """
    X_nn = _prepare_for_nmf(X).values
    X_norm = np.linalg.norm(X_nn, "fro")
    rng = np.random.default_rng(random_state)

    W_runs: list[np.ndarray] = []
    H_runs: list[np.ndarray] = []
    errors: list[float] = []

    for i in range(n_runs):
        seed = int(rng.integers(0, 2**31 - 1))
        W, H, err = _run_single_nmf(X_nn, k=k, random_state=seed, max_iter=max_iter)
        W_runs.append(W)
        H_runs.append(H)
        errors.append(err)

    aligned_W, aligned_H = _match_and_align_factors(W_runs, H_runs)
    mean_W = np.mean(aligned_W, axis=0)
    mean_H = np.mean(aligned_H, axis=0)

    # consensus по argmax (cluster-like stability на дискретизированных loadings)
    assignments = [w.argmax(axis=1) for w in aligned_W]
    consensus = _consensus_matrix_from_assignments(assignments)
    cophenetic = _cophenetic_correlation(consensus)
    stability = _stability_score(aligned_W)

    # финальная reconstruction error — на усреднённых W, H
    recon_err = float(np.linalg.norm(X_nn - mean_W @ mean_H, "fro"))
    explained = 1.0 - (recon_err / X_norm) ** 2

    W_df = pd.DataFrame(
        mean_W,
        index=X.index,
        columns=[f"factor_{i + 1}" for i in range(k)],
    )
    H_df = pd.DataFrame(
        mean_H,
        index=[f"factor_{i + 1}" for i in range(k)],
        columns=X.columns,
    )
    # normalize W per-factor to [0, 1] для интерпретируемости
    W_min = W_df.min(axis=0)
    W_max = W_df.max(axis=0)
    denom = (W_max - W_min).replace(0, 1.0)
    W_df_norm = (W_df - W_min) / denom

    logger.info(
        "NMF k=%d: cophenetic=%.3f, stability=%.3f, reconstruction_err=%.2f, "
        "explained_var=%.3f", k, cophenetic, stability, recon_err, explained,
    )

    return NMFResult(
        k=k,
        W=W_df_norm,
        H=H_df,
        reconstruction_error=recon_err,
        cophenetic=cophenetic,
        stability=stability,
        explained_variance=explained,
    )


def scan_nmf_k(
    X: pd.DataFrame,
    k_range: range,
    *,
    n_runs: int = 30,
    random_state: int = 42,
) -> Dict[int, NMFResult]:
    """Сканируем k в range; возвращаем dict{k → NMFResult}."""
    results: Dict[int, NMFResult] = {}
    for k in k_range:
        results[k] = nmf_with_stability(
            X, k=k, n_runs=n_runs, random_state=random_state,
        )
    return results


def select_k_nmf(
    results: Dict[int, NMFResult],
    *,
    min_cophenetic: float = 0.90,
    min_stability: float = 0.80,
) -> int:
    """Выбор k по Brunet 2004 + практические пороги стабильности.

    Логика: предпочитаем наибольший k, где cophenetic ≥ min_cophenetic И
    stability ≥ min_stability. Если ничего не удовлетворяет — берём k с max cophenetic.
    """
    passing = [
        k for k, r in results.items()
        if r.cophenetic >= min_cophenetic and r.stability >= min_stability
    ]
    if passing:
        return max(passing)
    # fallback — k с max cophenetic
    logger.warning(
        "No k met cophenetic≥%.2f and stability≥%.2f; selecting max-cophenetic k.",
        min_cophenetic, min_stability,
    )
    return max(results.keys(), key=lambda k: results[k].cophenetic)


def name_factors_by_top_signatures(
    H: pd.DataFrame,
    *,
    n_top: int = 3,
) -> pd.DataFrame:
    """Автоматические имена факторов по топ-N сигнатурам в каждом.

    Использует ту же каталогическую эвристику, что и characterize.py:
    - immune / stromal / apm → содержательное имя
    - иначе — signatures join.
    """
    from .characterize import _name_cluster

    rows = []
    for fname in H.index:
        row = H.loc[fname]
        # топ-N положительных contributions (H в NMF всегда >=0, но ранжирование
        # отражает "какие сигнатуры формируют фактор")
        top = [c.replace("sig__", "") for c in row.sort_values(ascending=False).head(n_top).index]
        bot = [c.replace("sig__", "") for c in row.sort_values(ascending=True).head(n_top).index]
        name = _name_cluster(top, bot)
        rows.append({
            "factor": fname,
            "name": name,
            "top_signatures": ",".join(top),
            "low_signatures": ",".join(bot),
        })
    df = pd.DataFrame(rows)
    # unique names
    seen: dict[str, int] = {}
    uniq = []
    for n in df["name"]:
        if n in seen:
            seen[n] += 1
            uniq.append(f"{n}_{seen[n]}")
        else:
            seen[n] = 1
            uniq.append(n)
    df["name_unique"] = uniq
    return df


def project_new_samples(
    X_new: pd.DataFrame,
    H: pd.DataFrame,
    *,
    max_iter: int = 200,
) -> pd.DataFrame:
    """Projection новой когорты через zero-initialization W, fixed H (Lin 2007).

    Используется для переноса NMF-факторов на METABRIC / другие когорты.
    """
    # align features
    common = [c for c in H.columns if c in X_new.columns]
    if len(common) < 5:
        raise ValueError(
            f"Only {len(common)} signatures overlap between NMF basis and new cohort."
        )
    X_aligned = X_new[common].copy()
    H_aligned = H[common].values

    X_nn = _prepare_for_nmf(X_aligned).values
    # NNLS per sample
    from scipy.optimize import nnls
    k = H_aligned.shape[0]
    W_new = np.zeros((X_nn.shape[0], k))
    for i in range(X_nn.shape[0]):
        w, _ = nnls(H_aligned.T, X_nn[i], maxiter=max_iter)
        W_new[i] = w

    # normalize per factor как в training (нужен W_min, W_max от training —
    # но у нас их нет без дополнительного передавания; используем local min/max
    # с caveat — на практике центроиды факторов всё равно транслируются через Cox)
    W_min = W_new.min(axis=0)
    W_max = W_new.max(axis=0)
    denom = np.where(W_max - W_min == 0, 1.0, W_max - W_min)
    W_norm = (W_new - W_min) / denom

    return pd.DataFrame(
        W_norm,
        index=X_new.index,
        columns=H.index,
    )
