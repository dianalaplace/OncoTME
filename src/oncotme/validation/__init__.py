"""Validation / self-check framework для OncoTME.

Идея: каждая крупная операция пайплайна (загрузка когорты, ssGSEA, CV-разбиение,
обучение, калибровка) сопровождается набором автоматических проверок, которые
возвращают ``CheckResult`` со статусом ``PASS`` / ``WARN`` / ``FAIL``.

Публичный API::

    from oncotme.validation import Check, CheckResult, ValidationReport
    from oncotme.validation import (
        check_cohort_bundle,
        check_feature_frame,
        check_cv_no_leakage,
        check_permutation_sanity,
        check_seed_stability,
        check_calibration,
        run_suite,
    )

Все проверки детерминированные и идемпотентные.
"""

from .result import Check, CheckResult, CheckStatus, ValidationReport
from .cohort_checks import check_cohort_bundle
from .feature_checks import check_feature_frame, check_signature_scores
from .model_checks import (
    check_calibration,
    check_cv_no_leakage,
    check_permutation_sanity,
    check_seed_stability,
)
from .discovery_checks import (
    check_consensus_stability,
    check_cluster_sizes,
    check_subtype_redundancy,
    check_event_rate,
    check_ph_assumption,
    check_classifier_transfer_sanity,
    check_nmf_stability,
    check_risk_score_discriminates,
    check_tertile_separation,
)
from .suite import run_suite

__all__ = [
    "Check",
    "CheckResult",
    "CheckStatus",
    "ValidationReport",
    "check_cohort_bundle",
    "check_feature_frame",
    "check_signature_scores",
    "check_cv_no_leakage",
    "check_permutation_sanity",
    "check_seed_stability",
    "check_calibration",
    "check_consensus_stability",
    "check_cluster_sizes",
    "check_subtype_redundancy",
    "check_event_rate",
    "check_ph_assumption",
    "check_classifier_transfer_sanity",
    "check_nmf_stability",
    "check_risk_score_discriminates",
    "check_tertile_separation",
    "run_suite",
]
