"""Discovery: unsupervised TME-архетипы + характеризация + survival.

Научная логика:
1. ``preprocess`` — z-score per signature (чтобы разные масштабы ssGSEA ES не
   ломали расстояния).
2. ``consensus`` — Monti 2003 consensus clustering с бутстрапом; выбор k через
   PAC (Șenbabaoğlu 2014) и ΔCDF.
3. ``characterize`` — mean signature-профиль каждого архетипа; автоматическое
   именование по доминирующим модулям; распределение подтипов и таргетной терапии.
4. ``survival`` — KM + log-rank + Cox (univariable / multivariable / stratified /
   treatment-aware interaction); PH-assumption check.
5. ``classifier`` — centroid-based NN для переноса архетипов на внешние когорты
   (METABRIC, SCAN-B).
"""

from .preprocess import preprocess_for_clustering
from .consensus import consensus_cluster, select_k_by_pac, ConsensusResult
from .characterize import characterize_archetypes, name_archetypes
from .survival import km_by_archetype, multivariable_cox, treatment_interaction_cox
from .classifier import CentroidClassifier
from .nmf_factors import (
    NMFResult,
    nmf_with_stability,
    scan_nmf_k,
    select_k_nmf,
    name_factors_by_top_signatures,
    project_new_samples,
)
from .risk_score import (
    RiskScoreModel,
    fit_composite_risk_score,
    tertile_km,
    treatment_interaction_by_factor,
    decision_curve_analysis,
)

__all__ = [
    "preprocess_for_clustering",
    "consensus_cluster",
    "select_k_by_pac",
    "ConsensusResult",
    "characterize_archetypes",
    "name_archetypes",
    "km_by_archetype",
    "multivariable_cox",
    "treatment_interaction_cox",
    "CentroidClassifier",
    "NMFResult",
    "nmf_with_stability",
    "scan_nmf_k",
    "select_k_nmf",
    "name_factors_by_top_signatures",
    "project_new_samples",
    "RiskScoreModel",
    "fit_composite_risk_score",
    "tertile_km",
    "treatment_interaction_by_factor",
    "decision_curve_analysis",
]
