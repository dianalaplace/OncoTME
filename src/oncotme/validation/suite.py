"""Фасад ``run_suite`` — собирает результаты проверок по всем слоям пайплайна.

Использование::

    from oncotme.validation import run_suite

    report = run_suite(
        cohorts=[tcga, metabric],
        feature_frames={"tcga": X_tcga, "metabric": X_metabric},
        model_artefacts=None,  # пока ещё не обучали
    )
    print(report.to_text())
    report.to_json("results/validation/report.json")
    report.assert_no_failures()  # hard-gate перед следующим шагом
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import pandas as pd

from ..cohorts.base import CohortBundle
from .cohort_checks import check_cohort_bundle, check_cohorts_disjoint
from .feature_checks import check_feature_frame, check_signature_scores
from .result import ValidationReport


def run_suite(
    cohorts: Optional[Iterable[CohortBundle]] = None,
    feature_frames: Optional[Dict[str, pd.DataFrame]] = None,
    model_artefacts: Optional[dict] = None,
) -> ValidationReport:
    """Запускает все релевантные проверки и собирает отчёт.

    Parameters
    ----------
    cohorts
        Список CohortBundle, прошедших loader. Будут прогнаны когортные
        проверки + disjoint-тест попарно.
    feature_frames
        Словарь ``{cohort_name → feature DataFrame}`` после сборки TME-панели.
    model_artefacts
        Заглушка: dict с ``"cv_splits"``, ``"sample_ids"``, ``"groups"``,
        ``"permutation_fn"``, ``"seed_fn"``, ``"y_true"``, ``"y_prob"`` —
        поля, которые заполняются скриптами обучения. Если None, соответствующие
        блоки пропускаются.
    """
    report = ValidationReport()

    # --- cohort-level ------------------------------------------------------
    cohort_list: List[CohortBundle] = list(cohorts or [])
    for c in cohort_list:
        report.extend(check_cohort_bundle(c))
    if len(cohort_list) >= 2:
        report.extend(check_cohorts_disjoint(cohort_list))

    # --- feature-level -----------------------------------------------------
    for name, X in (feature_frames or {}).items():
        ff = check_feature_frame(X)
        # префиксируем имена для группировки в отчёте
        for r in ff:
            r.name = f"{name}::{r.name}"
        report.extend(ff)
        sig = check_signature_scores(X)
        for r in sig:
            r.name = f"{name}::{r.name}"
        report.extend(sig)

    # --- model-level (optional) --------------------------------------------
    if model_artefacts:
        from .model_checks import (
            check_calibration,
            check_cv_no_leakage,
            check_permutation_sanity,
            check_seed_stability,
        )

        if "cv_splits" in model_artefacts:
            report.add(
                check_cv_no_leakage(
                    model_artefacts["cv_splits"],
                    model_artefacts.get("sample_ids", []),
                    groups=model_artefacts.get("groups"),
                    group_name=model_artefacts.get("group_name", "sample_id"),
                )
            )
        if "permutation" in model_artefacts:
            pm = model_artefacts["permutation"]
            report.add(
                check_permutation_sanity(
                    pm["fit_predict_fn"],
                    pm["X"],
                    pm["y"],
                    task=pm.get("task", "binary"),
                    n_permutations=pm.get("n_permutations", 5),
                    tol=pm.get("tol", 0.08),
                )
            )
        if "seed" in model_artefacts:
            sf = model_artefacts["seed"]
            report.add(
                check_seed_stability(
                    sf["fit_and_score_fn"],
                    seeds=sf.get("seeds", (0, 1, 2, 3, 4)),
                    max_std=sf.get("max_std", 0.03),
                    metric_name=sf.get("metric_name", "AUC"),
                )
            )
        if "calibration" in model_artefacts:
            cf = model_artefacts["calibration"]
            report.add(
                check_calibration(
                    cf["y_true"],
                    cf["y_prob"],
                    n_bins=cf.get("n_bins", 10),
                    max_ece=cf.get("max_ece", 0.10),
                    max_brier=cf.get("max_brier", 0.25),
                )
            )

    return report
