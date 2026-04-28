"""Rigorous benchmark для survival-моделей.

Основное назначение:
- Унифицированный интерфейс ``BenchmarkModel`` (.fit / .score) — чтобы nested CV
  одинаково работал с NMF+Cox, clinical-only, single-signature, random-features.
- ``nested_cv_evaluate`` — leakage-free nested 5-fold CV. Для NMF-модели
  factorization делается ТОЛЬКО на train-fold, projection на test-fold через NNLS.
- ``permutation_test`` — permute outcome, refit pipeline, null distribution C-index.
- ``partial_correlation_check`` — risk score vs age при контроле confounders.

Используется ``scripts/run_benchmark.py`` для полного сравнительного прогона.
"""

from .models import (
    BenchmarkModel,
    ClinicalCoxModel,
    NMFRiskModel,
    RandomFeaturesModel,
    SignatureCoxModel,
    TISModel,
    CYTModel,
    CombinedModel,
)
from .nested_cv import nested_cv_evaluate, NestedCVResult
from .permutation import permutation_test
from .confounding import partial_correlation_check, schoenfeld_check

__all__ = [
    "BenchmarkModel",
    "ClinicalCoxModel",
    "NMFRiskModel",
    "RandomFeaturesModel",
    "SignatureCoxModel",
    "TISModel",
    "CYTModel",
    "CombinedModel",
    "nested_cv_evaluate",
    "NestedCVResult",
    "permutation_test",
    "partial_correlation_check",
    "schoenfeld_check",
]
