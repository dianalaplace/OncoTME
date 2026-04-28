"""Проверки на уровне когорты — выполняются сразу после того, как loader
вернул CohortBundle. Ловят самые коварные ошибки: неверное выравнивание
sample_id, смешивание единиц измерения, кривое кодирование endpoint,
скрытые дубликаты.
"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

from ..cohorts.base import CohortBundle
from .checks import (
    _fail,
    _pass,
    _warn,
    assert_balanced_binary,
    assert_columns_subset,
    assert_index_aligned,
    assert_index_unique,
    assert_no_inf,
    assert_no_nan,
    assert_in_range,
)
from .result import CheckResult, CheckStatus


def _check_expr_unit_plausibility(cohort: CohortBundle) -> CheckResult:
    """log2(TPM+1) обычно в [0, 20]; log2-intensity в [3, 16]; counts — огромные.

    Ловим кейс, когда loader вернул raw counts, а пайплайн ожидает log-space.
    """
    name = f"cohort.{cohort.name}.expr_unit_plausible"
    arr = cohort.expr.values
    if arr.size == 0:
        return _fail(name, "Empty expression matrix.")
    # сэмплируем до 1M точек, чтобы не сжечь RAM на больших когортах
    flat = arr.ravel()
    if flat.size > 1_000_000:
        rng = np.random.default_rng(0)
        flat = rng.choice(flat, size=1_000_000, replace=False)
    q99 = float(np.nanquantile(flat, 0.99))
    q01 = float(np.nanquantile(flat, 0.01))

    expected_q99 = {
        "tpm": (0, 30),
        "log2_intensity": (6, 18),
        "counts": (100, 1e9),
        "log2_tpm": (1, 20),
    }.get(cohort.expr_kind)

    if expected_q99 is None:
        return _warn(
            name,
            f"Unknown expr_kind={cohort.expr_kind!r}; can't sanity-check range.",
            q01=q01,
            q99=q99,
        )
    lo, hi = expected_q99
    if q99 < lo or q99 > hi:
        return _fail(
            name,
            f"99-th percentile {q99:.2f} outside expected range [{lo}, {hi}] for expr_kind={cohort.expr_kind!r}. "
            f"Probable wrong units (counts vs log-space).",
            q99=q99,
            expected=(lo, hi),
        )
    return _pass(name, f"q01={q01:.2f} q99={q99:.2f} consistent with {cohort.expr_kind!r}.",
                 q01=q01, q99=q99)


def _check_gene_symbols_look_like_hgnc(cohort: CohortBundle) -> CheckResult:
    name = f"cohort.{cohort.name}.gene_symbols_hgnc_like"
    idx = cohort.expr.index.astype(str)
    # эвристика: HGNC-подобные символы — UPPERCASE/цифры/-, не "ENSG..." и не "probe_..."
    ens = idx.str.startswith("ENSG").sum()
    probes = idx.str.match(r"^\d+_at").sum() + idx.str.match(r"^ILMN").sum()
    lowercase = (idx != idx.str.upper()).sum()
    issues = []
    if ens > 0:
        issues.append(f"{ens} Ensembl IDs")
    if probes > 0:
        issues.append(f"{probes} probe IDs")
    if lowercase > 0:
        issues.append(f"{lowercase} non-uppercase symbols")
    if issues:
        return _fail(
            name,
            f"Gene index does not look like HGNC symbols: {', '.join(issues)}. "
            f"Run harmonize_genes() with mygene mapping before downstream steps.",
            ens=int(ens), probes=int(probes), lowercase=int(lowercase),
        )
    return _pass(name, f"All {len(idx)} gene symbols look HGNC-like.")


def _check_endpoints_encoding(cohort: CohortBundle) -> CheckResult:
    name = f"cohort.{cohort.name}.endpoints_encoding"
    ep = cohort.endpoints
    issues: List[str] = []
    # response endpoints
    for col in ["pCR", "ORR"]:
        if col in ep.columns:
            vals = set(ep[col].dropna().unique())
            if not vals.issubset({0, 1, True, False, "0", "1"}):
                issues.append(f"{col}: unexpected values {sorted(map(str, vals))[:5]}")
    # survival endpoints: time_* должны быть positive, event_* бинарные
    for col in ep.columns:
        if col.startswith("time_"):
            if (ep[col].dropna() < 0).any():
                issues.append(f"{col}: negative times present")
            if (ep[col].dropna() == 0).sum() > 0.5 * len(ep):
                issues.append(f"{col}: >50% zero times (suspicious)")
        if col.startswith("event_"):
            vals = set(ep[col].dropna().unique())
            if not vals.issubset({0, 1, True, False}):
                issues.append(f"{col}: non-binary event {sorted(map(str, vals))[:5]}")
    if issues:
        return _fail(name, "; ".join(issues))
    return _pass(name, f"Endpoints OK ({list(ep.columns)}).")


def _check_sample_id_disjoint_with_other(cohort: CohortBundle, other: CohortBundle) -> CheckResult:
    """Для пулинга когорт: нельзя, чтобы sample_id пересекался между когортами —
    иначе LOCO-CV утечёт."""
    name = f"cohort.disjoint_sample_ids.{cohort.name}_vs_{other.name}"
    a, b = set(cohort.expr.columns), set(other.expr.columns)
    overlap = a & b
    if not overlap:
        return _pass(name, f"Disjoint ({len(a)} vs {len(b)} samples).")
    return _fail(
        name,
        f"{len(overlap)} overlapping sample_id between {cohort.name} and {other.name} "
        f"(first: {sorted(overlap)[:3]}). Add cohort-prefix to sample_id.",
        overlap=list(sorted(overlap))[:10],
    )


def check_cohort_bundle(cohort: CohortBundle) -> List[CheckResult]:
    """Набор проверок после загрузки одной когорты."""
    results: List[CheckResult] = []

    # 1. базовые инварианты структуры
    results.append(assert_index_unique(cohort.expr, f"cohort.{cohort.name}.expr_gene_index_unique"))
    results.append(
        assert_index_unique(
            pd.DataFrame(index=cohort.expr.columns),
            f"cohort.{cohort.name}.expr_sample_columns_unique",
        )
    )
    results.append(assert_index_unique(cohort.clinical, f"cohort.{cohort.name}.clinical_index_unique"))
    results.append(assert_index_unique(cohort.endpoints, f"cohort.{cohort.name}.endpoints_index_unique"))

    # 2. выравнивание sample_id
    results.append(
        assert_index_aligned(
            pd.DataFrame(index=cohort.expr.columns),
            cohort.clinical,
            f"cohort.{cohort.name}.expr_clinical_aligned",
            min_overlap=max(10, int(0.5 * cohort.n_samples)),
        )
    )
    results.append(
        assert_index_aligned(
            cohort.clinical,
            cohort.endpoints,
            f"cohort.{cohort.name}.clinical_endpoints_aligned",
            min_overlap=max(10, int(0.5 * cohort.n_samples)),
        )
    )

    # 3. NaN / Inf в экспрессии
    results.append(assert_no_nan(cohort.expr, f"cohort.{cohort.name}.expr_no_nan"))
    results.append(assert_no_inf(cohort.expr, f"cohort.{cohort.name}.expr_no_inf"))

    # 4. семантические проверки
    results.append(_check_expr_unit_plausibility(cohort))
    results.append(_check_gene_symbols_look_like_hgnc(cohort))
    results.append(_check_endpoints_encoding(cohort))

    # 5. баланс pCR (если есть и содержит данные)
    if "pCR" in cohort.endpoints.columns:
        pcr_non_nan = cohort.endpoints["pCR"].dropna()
        if len(pcr_non_nan) >= 10:
            results.append(
                assert_balanced_binary(
                    cohort.endpoints["pCR"],
                    f"cohort.{cohort.name}.pcr_class_balance",
                )
            )
        else:
            results.append(
                CheckResult(
                    name=f"cohort.{cohort.name}.pcr_class_balance",
                    status=CheckStatus.SKIP,
                    message=f"pCR has {len(pcr_non_nan)} non-NaN values — insufficient for balance check.",
                )
            )

    # 6. клинические ключевые колонки (не обязательны, поэтому WARN, а не FAIL)
    clinical_expected = ["age", "subtype_er", "subtype_her2"]
    missing = [c for c in clinical_expected if c not in cohort.clinical.columns]
    if missing:
        results.append(
            _warn(
                f"cohort.{cohort.name}.clinical_expected_cols",
                f"Soft-missing clinical columns: {missing}. "
                f"Subtype-stratified and treatment-adjusted models will degrade.",
                missing=missing,
            )
        )
    else:
        results.append(
            _pass(
                f"cohort.{cohort.name}.clinical_expected_cols",
                f"Key clinical columns present: {clinical_expected}.",
            )
        )

    return results


def check_cohorts_disjoint(cohorts: List[CohortBundle]) -> List[CheckResult]:
    """При пулинге нескольких когорт — sample_id не должны пересекаться."""
    results: List[CheckResult] = []
    for i, a in enumerate(cohorts):
        for b in cohorts[i + 1:]:
            results.append(_check_sample_id_disjoint_with_other(a, b))
    return results
