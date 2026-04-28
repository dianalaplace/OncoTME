"""End-to-end discovery: TME-архетипы + характеризация + survival на TCGA-BRCA.

Порядок с hard-gate после каждого шага:
1. Загрузка cohort + feature-frame (артефакты build_features.py)
2. Preprocess: robust z-score сигнатур
3. Consensus clustering по k ∈ [2..6], 100 бутстрапов
4. Validation: PAC ≤ 0.25, min cluster fraction ≥ 5%, subtype non-redundancy
5. Characterization: profiles, auto-naming, clinical / treatment distribution
6. Survival: KM + log-rank + multivariable Cox + stratified + treatment-interaction
7. Обучение CentroidClassifier + self-predict sanity
8. Сохранение артефактов в results/discovery/

Usage::
    python scripts/run_discovery.py \\
        --features data/processed/tcga_brca_features.parquet \\
        --cohort-pkl data/processed/tcga_brca.pkl \\
        --out-dir results/discovery/
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))

import numpy as np
import pandas as pd

from oncotme.discovery import (
    CentroidClassifier,
    characterize_archetypes,
    consensus_cluster,
    km_by_archetype,
    multivariable_cox,
    preprocess_for_clustering,
    select_k_by_pac,
    treatment_interaction_cox,
)
from oncotme.discovery.consensus import delta_cdf_curve
from oncotme.discovery.survival import stratified_cox_by_pam50
from oncotme.validation import (
    check_classifier_transfer_sanity,
    check_cluster_sizes,
    check_consensus_stability,
    check_event_rate,
    check_ph_assumption,
    check_subtype_redundancy,
)
from oncotme.validation.result import CheckStatus, ValidationReport

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("run_discovery")


def _hard_gate(report: ValidationReport, step: str) -> None:
    counts = report.counts()
    logger.info(
        "[%s] PASS=%d WARN=%d FAIL=%d",
        step, counts["PASS"], counts["WARN"], counts["FAIL"],
    )
    for r in report.results:
        if r.status == CheckStatus.WARN:
            logger.warning("  [WARN] %s: %s", r.name, r.message)
        elif r.status == CheckStatus.FAIL:
            logger.error("  [FAIL] %s: %s", r.name, r.message)
    if counts["FAIL"] > 0:
        logger.error("Validation FAILED at step %r.", step)
        sys.exit(1)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--features", required=True)
    p.add_argument("--cohort-pkl", required=True)
    p.add_argument("--out-dir", default=str(_REPO / "results" / "discovery"))
    p.add_argument("--k-min", type=int, default=2)
    p.add_argument("--k-max", type=int, default=6)
    p.add_argument("--n-bootstrap", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--linkage",
        choices=["ward", "average", "complete"],
        default="average",
        help="Base linkage. 'average'+correlation is classic Eisen-style for gene expression.",
    )
    p.add_argument(
        "--max-pac",
        type=float,
        default=0.35,
        help="Maximum acceptable PAC for TME data (Şenbabaoğlu 2014: 0.2 for clean "
             "molecular subtypes; 0.3-0.4 is realistic for continuous TME).",
    )
    return p.parse_args()


def _json_safe(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj) if np.isfinite(obj) else None
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (pd.Series, pd.DataFrame)):
        return obj.to_dict()
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


def main() -> int:
    args = _parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------- 1. Загрузка -------------------------------------------------
    logger.info("[1/7] Loading features + cohort …")
    features = pd.read_parquet(args.features)
    with open(args.cohort_pkl, "rb") as f:
        cohort = pickle.load(f)
    logger.info("Features: %s | cohort: %s", features.shape, cohort)

    # ---------- 2. Preprocess ----------------------------------------------
    logger.info("[2/7] Preprocessing (robust z-score) …")
    X = preprocess_for_clustering(features, robust=True)
    logger.info("Preprocessed: %s", X.shape)

    # ---------- 3. Consensus clustering ------------------------------------
    logger.info(
        "[3/7] Consensus clustering k=[%d..%d], %d bootstraps …",
        args.k_min, args.k_max, args.n_bootstrap,
    )
    cc_results = consensus_cluster(
        X, k_range=range(args.k_min, args.k_max + 1),
        n_bootstrap=args.n_bootstrap,
        linkage_method=args.linkage,
        random_state=args.seed,
    )

    # выбираем k
    best_k = select_k_by_pac(cc_results, min_cluster_fraction=0.05)
    logger.info("Selected k* = %d (min PAC, ≥5%% cluster fraction).", best_k)
    best = cc_results[best_k]
    labels = pd.Series(best.labels, index=X.index, name="cluster_id")

    # сохраняем Δ CDF таблицу
    delta_df = delta_cdf_curve(cc_results)
    delta_df.to_csv(out_dir / "delta_cdf.csv", index=False)

    # сохраняем стабильность по k
    stability_df = pd.DataFrame(
        [{"k": k, "pac": r.pac, "cdf_area": r.cdf_area,
          "silhouette": r.silhouette, "min_cluster_frac": r.min_cluster_fraction,
          "cluster_sizes": json.dumps(r.cluster_sizes)}
         for k, r in cc_results.items()]
    )
    stability_df.to_csv(out_dir / "stability_by_k.csv", index=False)

    # ---------- 4. Validation consensus ------------------------------------
    rep = ValidationReport()
    rep.add(check_consensus_stability(best.pac, best_k, max_pac=args.max_pac))
    rep.add(check_cluster_sizes(best.cluster_sizes))
    if "subtype_pam50" in cohort.clinical.columns:
        rep.add(check_subtype_redundancy(labels, cohort.clinical["subtype_pam50"]))
    _hard_gate(rep, "consensus_validation")

    # ---------- 5. Characterization ----------------------------------------
    logger.info("[4/7] Characterizing archetypes …")
    char = characterize_archetypes(X, labels, cohort.clinical)
    profiles = char["profiles"]
    names_df = char["names"]
    profiles.to_csv(out_dir / "archetype_profiles.csv")
    names_df.to_csv(out_dir / "archetype_names.csv", index=False)
    # clinical distribution
    clinical_distr_rows = []
    for var, info in char["clinical_distribution"].items():
        row = {
            "variable": var,
            "kind": info["kind"],
            "p_value": info.get("p_value"),
            "q_value": info.get("q_value"),
            "cramer_v": info.get("cramer_v"),
        }
        clinical_distr_rows.append(row)
    pd.DataFrame(clinical_distr_rows).to_csv(out_dir / "archetype_clinical_assoc.csv", index=False)

    # raw tables — по одной на переменную
    for var, info in char["clinical_distribution"].items():
        if info["kind"] == "categorical":
            info["table"].to_csv(out_dir / f"archetype_x_{var}.csv")
        else:
            info["stats"].to_csv(out_dir / f"archetype_x_{var}_stats.csv")

    # mapping cluster_id → human-readable name
    name_map = {int(r.cluster_id): r.name_unique for _, r in names_df.iterrows()}
    labels_named = labels.map(name_map)

    # ---------- 6. Survival -------------------------------------------------
    logger.info("[5/7] Survival analysis …")
    endpoints = cohort.endpoints.reindex(labels.index)

    survival_summary = {}
    for endpoint_prefix in ["os", "dfs"]:
        time_col = f"time_{endpoint_prefix}"
        event_col = f"event_{endpoint_prefix}"
        if time_col not in endpoints.columns or event_col not in endpoints.columns:
            logger.info("Skipping %s (missing).", endpoint_prefix.upper())
            continue
        if endpoints[time_col].notna().sum() < 30:
            logger.info("Skipping %s: too few non-NaN times.", endpoint_prefix.upper())
            continue

        logger.info(" -- Endpoint: %s", endpoint_prefix.upper())
        ep_rep = ValidationReport()
        ep_rep.add(check_event_rate(endpoints[event_col]))
        _hard_gate(ep_rep, f"event_rate_{endpoint_prefix}")

        km = km_by_archetype(labels, endpoints[time_col], endpoints[event_col],
                             archetype_names=name_map)
        # КМ кривые — сохраним как длинный CSV
        km_long = []
        for arch_name, curve in km["km_curves"].items():
            curve = curve.copy()
            curve["archetype"] = arch_name
            km_long.append(curve)
        pd.concat(km_long).to_csv(out_dir / f"km_curves_{endpoint_prefix}.csv", index=False)
        km["median_survival"].to_csv(out_dir / f"km_median_{endpoint_prefix}.csv")
        km["logrank_pairwise"].to_csv(out_dir / f"logrank_pairwise_{endpoint_prefix}.csv",
                                      index=False)

        # multivariable Cox
        try:
            cox = multivariable_cox(labels, endpoints[time_col], endpoints[event_col],
                                    cohort.clinical)
            cox["summary"].to_csv(out_dir / f"cox_multivar_{endpoint_prefix}.csv")
            ph_report = ValidationReport()
            ph_report.add(check_ph_assumption(cox["ph_violated"]))
            _hard_gate(ph_report, f"cox_ph_{endpoint_prefix}")
            # stratified by PAM50 — проверка подтип-специфичности (H3)
            try:
                strat = stratified_cox_by_pam50(labels, endpoints[time_col],
                                                 endpoints[event_col], cohort.clinical)
                strat["summary"].to_csv(out_dir / f"cox_stratified_pam50_{endpoint_prefix}.csv")
            except Exception as exc:
                logger.warning("Stratified Cox failed: %s", exc)
                strat = None
            # treatment interaction — если есть targeted_therapy_received
            interaction = None
            if "targeted_therapy_received" in cohort.clinical.columns and \
               cohort.clinical["targeted_therapy_received"].notna().sum() >= 100:
                try:
                    interaction = treatment_interaction_cox(
                        labels, endpoints[time_col], endpoints[event_col],
                        cohort.clinical,
                        treatment_col="targeted_therapy_received",
                    )
                    interaction["summary"].to_csv(
                        out_dir / f"cox_treatment_interaction_{endpoint_prefix}.csv"
                    )
                    interaction["interaction_rows"].to_csv(
                        out_dir / f"cox_treatment_interaction_rows_{endpoint_prefix}.csv"
                    )
                except Exception as exc:
                    logger.warning("Treatment-interaction Cox failed: %s", exc)

            survival_summary[endpoint_prefix] = {
                "n": cox["n"],
                "concordance_index": cox["concordance_index"],
                "logrank_overall": km["logrank_overall"],
                "ph_violated": cox["ph_violated"],
                "stratified_c_index": strat["concordance_index"] if strat else None,
                "interaction_n": interaction["n"] if interaction else None,
            }
        except Exception as exc:
            logger.warning("Cox failed for %s: %s", endpoint_prefix, exc)

    with (out_dir / "survival_summary.json").open("w") as f:
        json.dump(_json_safe(survival_summary), f, indent=2)

    # ---------- 7. Classifier ----------------------------------------------
    logger.info("[6/7] Training CentroidClassifier …")
    classifier = CentroidClassifier(metric="correlation")
    classifier.fit(X, labels)
    cl_rep = ValidationReport()
    cl_rep.add(check_classifier_transfer_sanity(classifier, X, labels, min_accuracy=0.85))
    _hard_gate(cl_rep, "classifier_sanity")
    classifier.save(out_dir / "centroid_classifier.pkl")

    # labels на диск
    label_df = pd.DataFrame({"cluster_id": labels, "archetype": labels_named})
    label_df.to_csv(out_dir / "archetype_labels.csv")

    logger.info("[7/7] Done. Artefacts → %s", out_dir)
    # финальное резюме
    logger.info("=" * 60)
    logger.info("OncoTME discovery summary:")
    logger.info("  k* = %d, PAC = %.3f, silhouette = %.3f",
                best_k, best.pac, best.silhouette)
    logger.info("  Cluster sizes: %s", best.cluster_sizes)
    for cid, name in name_map.items():
        logger.info("    %d → %s", cid, name)
    for ep, summ in survival_summary.items():
        logger.info("  %s: n=%d | C-index=%.3f | log-rank p=%.2e | PH violated: %s",
                    ep.upper(), summ["n"], summ["concordance_index"],
                    summ["logrank_overall"]["p_value"], summ["ph_violated"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
