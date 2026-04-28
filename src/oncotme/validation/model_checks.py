"""Проверки на уровне ML-пайплайна.

Самый важный блок. Сюда входят:
- ``check_cv_no_leakage`` — нет пересечения train/test fold по sample_id, cohort,
  пациенту (если есть patient_id), и по time (для longitudinal)
- ``check_permutation_sanity`` — при перемешанном y модель не должна учиться:
  AUC ≈ 0.5 ± 0.05; C-index ≈ 0.5 ± 0.05. Если хуже — есть утечка.
- ``check_seed_stability`` — стандартное отклонение метрики по 3+ seed'ам
  должно быть мало (< 0.03 для AUC, < 0.02 для C-index).
- ``check_calibration`` — Brier, expected calibration error (ECE), slope.
"""

from __future__ import annotations

from typing import Callable, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .checks import _fail, _pass, _warn
from .result import CheckResult


# --------------------------------------------------------------------- leakage

def check_cv_no_leakage(
    train_test_splits: Iterable[Tuple[np.ndarray, np.ndarray]],
    sample_ids: Sequence[str],
    *,
    groups: Optional[Sequence[str]] = None,
    group_name: str = "sample_id",
) -> CheckResult:
    """Проверяет, что train и test индексы в каждом фолде непересекающиеся,
    а если передан ``groups`` (cohort, patient_id) — что группа ни одна не
    появляется в train и test одновременно.
    """
    name = f"cv.no_leakage_by_{group_name}"
    sample_ids = np.asarray(sample_ids)
    group_arr = np.asarray(groups) if groups is not None else None

    bad_folds: List[dict] = []
    n_folds = 0
    for fold_idx, (tr, te) in enumerate(train_test_splits):
        n_folds += 1
        if len(set(tr) & set(te)) > 0:
            bad_folds.append({"fold": fold_idx, "reason": "index overlap"})
            continue
        if group_arr is not None:
            gtr = set(group_arr[tr])
            gte = set(group_arr[te])
            shared = gtr & gte
            if shared:
                bad_folds.append(
                    {"fold": fold_idx, "reason": "group overlap", "groups": list(shared)[:5]}
                )

    if n_folds == 0:
        return _fail(name, "No folds to check.")
    if bad_folds:
        return _fail(
            name,
            f"{len(bad_folds)}/{n_folds} folds leak {group_name}. Example: {bad_folds[0]}",
            bad_folds=bad_folds[:5],
        )
    return _pass(name, f"All {n_folds} folds disjoint by {group_name}.")


# --------------------------------------------------------------------- permutation sanity

def check_permutation_sanity(
    fit_predict_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_permutations: int = 5,
    task: str = "binary",
    tol: float = 0.08,
    random_state: int = 42,
) -> CheckResult:
    """Если перемешать y случайно, модель не должна учиться.

    Parameters
    ----------
    fit_predict_fn
        Функция ``(X, y) -> y_pred_out_of_fold`` (например, cross_val_predict).
    task
        ``"binary"`` — измеряем AUC; ``"survival"`` — C-index (ожидается двухколонный y
        с полями ``event`` и ``time``).
    tol
        Допуск |metric - 0.5|. Выше tol → FAIL (signature for leakage).
    """
    name = f"model.permutation_sanity.{task}"
    rng = np.random.default_rng(random_state)

    try:
        from sklearn.metrics import roc_auc_score
    except Exception as exc:
        return _fail(name, f"sklearn not available: {exc}")

    deltas: List[float] = []
    for i in range(n_permutations):
        y_perm = y.copy()
        rng.shuffle(y_perm)
        try:
            pred = fit_predict_fn(X, y_perm)
        except Exception as exc:  # noqa: BLE001
            return _fail(name, f"fit_predict_fn raised on perm {i}: {exc}")
        if task == "binary":
            try:
                auc = float(roc_auc_score(y_perm, pred))
            except ValueError as exc:
                return _fail(name, f"AUC undefined on perm {i}: {exc}")
            deltas.append(abs(auc - 0.5))
        elif task == "survival":
            # простая C-index-проверка (Kendall tau-like); подмена — пользоваться
            # lifelines.concordance_index в реальном usage
            try:
                from lifelines.utils import concordance_index
            except Exception as exc:
                return _fail(name, f"lifelines not available: {exc}")
            time, event = y_perm["time"], y_perm["event"]
            c = float(concordance_index(time, -np.asarray(pred), event))
            deltas.append(abs(c - 0.5))
        else:
            return _fail(name, f"Unknown task {task!r}.")

    max_delta = max(deltas)
    if max_delta > tol:
        return _fail(
            name,
            f"Max |metric - 0.5| = {max_delta:.3f} > tol {tol}. Suspected leakage or bug.",
            deltas=deltas,
        )
    return _pass(
        name,
        f"Max |metric - 0.5| = {max_delta:.3f} (<= {tol}) across {n_permutations} perms.",
        deltas=deltas,
    )


# --------------------------------------------------------------------- seed stability

def check_seed_stability(
    fit_and_score_fn: Callable[[int], float],
    seeds: Sequence[int] = (0, 1, 2, 3, 4),
    *,
    max_std: float = 0.03,
    metric_name: str = "AUC",
) -> CheckResult:
    """Стабильность метрики относительно random seed.

    fit_and_score_fn(seed) -> scalar (AUC, C-index, AP…).
    """
    name = f"model.seed_stability.{metric_name.lower()}"
    scores: List[float] = []
    for s in seeds:
        try:
            scores.append(float(fit_and_score_fn(s)))
        except Exception as exc:  # noqa: BLE001
            return _fail(name, f"fit_and_score_fn raised at seed={s}: {exc}")

    arr = np.asarray(scores)
    sd = float(arr.std(ddof=0))
    mean = float(arr.mean())
    if sd > max_std:
        return _fail(
            name,
            f"Std({metric_name})={sd:.3f} across seeds {list(seeds)} > max_std {max_std}. "
            f"Model too sensitive to seed; increase CV folds, shrink, or ensemble.",
            mean=mean,
            std=sd,
            scores=scores,
        )
    return _pass(
        name,
        f"{metric_name}: mean={mean:.3f} ± {sd:.3f} over seeds {list(seeds)}.",
        mean=mean,
        std=sd,
        scores=scores,
    )


# --------------------------------------------------------------------- calibration

def check_calibration(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    n_bins: int = 10,
    max_ece: float = 0.10,
    max_brier: float = 0.25,
) -> CheckResult:
    """Expected Calibration Error (ECE) + Brier score.

    ECE считается как взвешенная по bin'ам |mean(y) - mean(p)|.
    """
    name = "model.calibration"
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    if len(y_true) != len(y_prob):
        return _fail(name, f"Length mismatch: {len(y_true)} vs {len(y_prob)}.")
    if np.isnan(y_prob).any() or np.isnan(y_true).any():
        return _fail(name, "NaN in y_true or y_prob.")

    brier = float(np.mean((y_prob - y_true) ** 2))

    # binned ECE
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (y_prob >= lo) & (y_prob < hi) if hi < 1.0 else (y_prob >= lo) & (y_prob <= hi)
        if mask.sum() == 0:
            continue
        frac_pos = y_true[mask].mean()
        mean_p = y_prob[mask].mean()
        ece += (mask.sum() / n) * abs(frac_pos - mean_p)

    evidence = {"ece": ece, "brier": brier, "n": int(n), "n_bins": n_bins}

    if np.isnan(brier):
        return _fail(name, "Brier is NaN.", **evidence)
    if ece > max_ece or brier > max_brier:
        return _warn(
            name,
            f"ECE={ece:.3f} (>{max_ece}) or Brier={brier:.3f} (>{max_brier}). "
            f"Fit isotonic/Platt calibration.",
            **evidence,
        )
    return _pass(name, f"ECE={ece:.3f}, Brier={brier:.3f}.", **evidence)
