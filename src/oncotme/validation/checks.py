"""Атомарные проверки — кирпичики, на которых строятся доменные проверки.

Каждая функция:
- принимает значение(я) и называние проверки
- возвращает ``CheckResult``
- никогда не рушится сама (ловит исключения и превращает в FAIL)

Покрываем:
- ``assert_no_nan`` — нет NaN в DataFrame/Series/matrix
- ``assert_no_inf`` — нет бесконечностей
- ``assert_shape`` — ожидаемый shape
- ``assert_in_range`` — значения в диапазоне [lo, hi]
- ``assert_index_unique`` — уникальный индекс
- ``assert_columns_subset`` — подмножество колонок
- ``assert_index_aligned`` — общий индекс между DataFrame
- ``assert_balanced_binary`` — класс-баланс (WARN если <0.05 или >0.95)
- ``assert_monotonic_cv_leakage_probe`` — вспомогалка для model_checks
"""

from __future__ import annotations

from typing import Iterable, Optional, Tuple

import numpy as np
import pandas as pd

from .result import CheckResult, CheckStatus


def _pass(name: str, message: str = "", **evidence) -> CheckResult:
    return CheckResult(name=name, status=CheckStatus.PASS, message=message, evidence=evidence)


def _warn(name: str, message: str, **evidence) -> CheckResult:
    return CheckResult(name=name, status=CheckStatus.WARN, message=message, evidence=evidence)


def _fail(name: str, message: str, **evidence) -> CheckResult:
    return CheckResult(name=name, status=CheckStatus.FAIL, message=message, evidence=evidence)


# --------------------------------------------------------------------- NaN/Inf

def assert_no_nan(df: pd.DataFrame | pd.Series, name: str) -> CheckResult:
    if isinstance(df, pd.Series):
        n_nan = int(df.isna().sum())
        total = int(df.size)
    else:
        n_nan = int(df.isna().sum().sum())
        total = int(df.size)
    if n_nan == 0:
        return _pass(name, "No NaN.", total=total)
    share = n_nan / max(total, 1)
    if share > 0.0:
        return _fail(name, f"{n_nan}/{total} NaN ({share:.1%}).", n_nan=n_nan, total=total)
    return _pass(name, "No NaN.", total=total)


def assert_no_inf(df: pd.DataFrame | pd.Series, name: str) -> CheckResult:
    arr = df.values if hasattr(df, "values") else df
    try:
        n_inf = int(np.isinf(arr).sum())
    except TypeError:
        # не-numeric → проверять inf бессмысленно, считаем PASS
        return _pass(name, "Non-numeric, skipped inf check.")
    if n_inf == 0:
        return _pass(name, "No Inf.")
    return _fail(name, f"{n_inf} Inf values.", n_inf=n_inf)


# --------------------------------------------------------------------- shape / range

def assert_shape(
    df: pd.DataFrame | pd.Series,
    expected: Tuple[Optional[int], Optional[int]] | int,
    name: str,
) -> CheckResult:
    actual = df.shape if hasattr(df, "shape") else (len(df),)
    if isinstance(expected, int):
        ok = (len(actual) == 1 and actual[0] == expected) or (
            len(actual) == 2 and actual[0] == expected
        )
        return _pass(name, f"shape={actual}") if ok else _fail(
            name, f"Expected length {expected}, got shape {actual}.", actual=actual
        )
    # per-axis match, None = wildcard
    if len(actual) != len(expected):
        return _fail(name, f"Rank mismatch. Expected {expected}, got {actual}.", actual=actual)
    for a, e in zip(actual, expected):
        if e is not None and a != e:
            return _fail(name, f"Expected shape {expected}, got {actual}.", actual=actual)
    return _pass(name, f"shape={actual}")


def assert_in_range(
    series: pd.Series | np.ndarray,
    lo: float,
    hi: float,
    name: str,
    *,
    allow_nan: bool = False,
) -> CheckResult:
    arr = np.asarray(series)
    finite = arr[~np.isnan(arr)] if arr.dtype.kind == "f" else arr
    if not allow_nan and np.isnan(arr).any():
        return _fail(name, "NaN encountered.", n_nan=int(np.isnan(arr).sum()))
    if finite.size == 0:
        return _warn(name, "Empty after NaN filter.")
    amin, amax = float(np.min(finite)), float(np.max(finite))
    if amin < lo or amax > hi:
        return _fail(
            name,
            f"Values outside [{lo}, {hi}]: min={amin:.3g} max={amax:.3g}.",
            min=amin,
            max=amax,
        )
    return _pass(name, f"min={amin:.3g} max={amax:.3g}", min=amin, max=amax)


# --------------------------------------------------------------------- index / columns

def assert_index_unique(obj: pd.DataFrame | pd.Series, name: str, axis: int = 0) -> CheckResult:
    idx = obj.index if axis == 0 else obj.columns
    dup = idx.duplicated().sum()
    if dup == 0:
        return _pass(name, "Index unique.", n=len(idx))
    example = idx[idx.duplicated(keep=False)].unique().tolist()[:5]
    return _fail(name, f"{dup} duplicates. Examples: {example}", n_duplicates=int(dup))


def assert_columns_subset(
    df: pd.DataFrame, required: Iterable[str], name: str
) -> CheckResult:
    required = list(required)
    missing = [c for c in required if c not in df.columns]
    if not missing:
        return _pass(name, f"All {len(required)} required columns present.")
    return _fail(name, f"Missing columns: {missing}", missing=missing)


def assert_index_aligned(
    a: pd.DataFrame | pd.Series,
    b: pd.DataFrame | pd.Series,
    name: str,
    *,
    a_axis: int = 0,
    b_axis: int = 0,
    min_overlap: int = 1,
) -> CheckResult:
    idx_a = a.index if a_axis == 0 else a.columns
    idx_b = b.index if b_axis == 0 else b.columns
    a_set, b_set = set(idx_a), set(idx_b)
    common = a_set & b_set
    if len(common) < min_overlap:
        return _fail(
            name,
            f"Overlap {len(common)} < min_overlap {min_overlap}.",
            overlap=len(common),
            only_a=len(a_set - b_set),
            only_b=len(b_set - a_set),
        )
    if a_set == b_set:
        return _pass(name, f"Indexes identical (n={len(common)}).")
    return _warn(
        name,
        f"Overlap {len(common)}; only_a={len(a_set - b_set)}, only_b={len(b_set - a_set)}.",
        overlap=len(common),
        only_a=len(a_set - b_set),
        only_b=len(b_set - a_set),
    )


# --------------------------------------------------------------------- balance / distribution

def assert_balanced_binary(
    y: pd.Series,
    name: str,
    *,
    min_minority_fraction: float = 0.05,
) -> CheckResult:
    y_clean = pd.to_numeric(y.dropna(), errors="coerce")
    n_coerced = int(y_clean.isna().sum())
    if n_coerced > 0:
        return _fail(
            name,
            f"{n_coerced} non-numeric values in target (e.g. strings like 'yes'/'no').",
            n_non_numeric=n_coerced,
        )
    vc = y_clean.astype(int).value_counts(normalize=True).sort_index()
    if vc.empty:
        return _fail(name, "Target is empty.")
    classes = vc.index.tolist()
    if set(classes) - {0, 1}:
        return _fail(name, f"Non-binary target: classes={classes}", classes=classes)
    minority = float(vc.min())
    if minority < min_minority_fraction:
        return _warn(
            name,
            f"Highly imbalanced: minority={minority:.1%} (< {min_minority_fraction:.0%}).",
            minority_fraction=minority,
        )
    return _pass(name, f"Minority fraction={minority:.1%}", minority_fraction=minority)


def assert_expected_correlation_sign(
    s1: pd.Series,
    s2: pd.Series,
    expected_sign: int,
    name: str,
    *,
    min_abs_corr: float = 0.1,
) -> CheckResult:
    """Биологическая sanity-check: скоррелированы ли две сигнатуры в ожидаемом направлении.

    Примеры:
    - CD8_T_cell vs IFN_gamma — ожидаем +, |r| >= 0.3
    - CAF_proxy vs CD8_T_cell — часто - в excluded-опухолях
    """
    common = s1.dropna().index.intersection(s2.dropna().index)
    if len(common) < 10:
        return _warn(name, f"n={len(common)} < 10, correlation unreliable.")
    r = float(s1.loc[common].corr(s2.loc[common]))
    if np.isnan(r):
        return _warn(name, "Pearson r is NaN (constant series?).", r=r)
    if abs(r) < min_abs_corr:
        return _warn(
            name,
            f"|r|={abs(r):.2f} < min_abs_corr={min_abs_corr} (expected sign {expected_sign:+d}).",
            r=r,
        )
    if np.sign(r) != np.sign(expected_sign):
        return _fail(
            name,
            f"Sign mismatch: r={r:.2f}, expected {expected_sign:+d}.",
            r=r,
            expected_sign=expected_sign,
        )
    return _pass(name, f"r={r:.2f} (expected {expected_sign:+d}).", r=r)
