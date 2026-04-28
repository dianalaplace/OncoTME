"""Clinically-oriented discovery: NMF factors → composite TME risk score → DCA.

Шаги (hard-gate между каждым):
1. Load features + cohort.
2. Preprocess: robust z-score (для NMF — в non-negative пространство внутри).
3. Scan NMF k=[3..6], 30 runs per k, считаем cophenetic + stability.
4. Select k* по Brunet 2004 (≥0.90 cophenetic) или max-cophenetic fallback.
5. Name factors по топ-сигнатурам.
6. Fit composite Cox risk score (ElasticNet-Cox на factor loadings) для RFS/OS.
7. Tertile KM + log-rank + pairwise BH.
8. Treatment × factor interaction Cox, BH FDR на interaction p-values.
9. Decision Curve Analysis.
10. Сохранить всё в results/discovery_nmf/.

Usage::

    python scripts/run_discovery_nmf.py \\
        --features data/processed/tcga_brca_features.parquet \\
        --cohort-pkl data/processed/tcga_brca.pkl \\
        --out-dir results/discovery_nmf/tcga_brca/
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
    decision_curve_analysis,
    fit_composite_risk_score,
    name_factors_by_top_signatures,
    preprocess_for_clustering,
    scan_nmf_k,
    select_k_nmf,
    tertile_km,
    treatment_interaction_by_factor,
)
from oncotme.validation import (
    check_event_rate,
    check_nmf_stability,
    check_risk_score_discriminates,
    check_tertile_separation,
)
from oncotme.validation.result import CheckStatus, ValidationReport

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("run_discovery_nmf")


def _hard_gate(report: ValidationReport, step: str, fail_on_warn: bool = False) -> None:
    counts = report.counts()
    logger.info(
        "[%s] PASS=%d WARN=%d FAIL=%d", step,
        counts["PASS"], counts["WARN"], counts["FAIL"],
    )
    for r in report.results:
        if r.status == CheckStatus.WARN:
            logger.warning("  [WARN] %s: %s", r.name, r.message)
        elif r.status == CheckStatus.FAIL:
            logger.error("  [FAIL] %s: %s", r.name, r.message)
    if counts["FAIL"] > 0 or (fail_on_warn and counts["WARN"] > 0):
        sys.exit(1)


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


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--features", required=True)
    p.add_argument("--cohort-pkl", required=True)
    p.add_argument("--out-dir", default=str(_REPO / "results" / "discovery_nmf"))
    p.add_argument("--k-min", type=int, default=3)
    p.add_argument("--k-max", type=int, default=6)
    p.add_argument("--n-runs", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--penalizer", type=float, default=0.05,
                   help="ElasticNet-Cox penalizer for risk score.")
    p.add_argument("--l1-ratio", type=float, default=0.5)
    return p.parse_args()


def main() -> int:
    args = _parse()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------- 1. Load ----------------------------------------------------
    logger.info("[1/10] Loading artefacts …")
    features = pd.read_parquet(args.features)
    with open(args.cohort_pkl, "rb") as f:
        cohort = pickle.load(f)
    logger.info("features=%s cohort=%s", features.shape, cohort)

    # ---------- 2. Preprocess ---------------------------------------------
    logger.info("[2/10] Robust z-score on signatures …")
    X = preprocess_for_clustering(features, robust=True)
    X.to_csv(out_dir / "preprocessed_signatures.csv")

    # ---------- 3. Scan NMF k ---------------------------------------------
    logger.info("[3/10] Scanning NMF k=[%d..%d], %d runs/k …",
                args.k_min, args.k_max, args.n_runs)
    results = scan_nmf_k(
        X, k_range=range(args.k_min, args.k_max + 1),
        n_runs=args.n_runs, random_state=args.seed,
    )

    stability_df = pd.DataFrame([
        {"k": k, "cophenetic": r.cophenetic, "stability": r.stability,
         "reconstruction_err": r.reconstruction_error,
         "explained_variance": r.explained_variance}
        for k, r in results.items()
    ])
    stability_df.to_csv(out_dir / "nmf_stability_by_k.csv", index=False)

    # ---------- 4. Select k* ----------------------------------------------
    logger.info("[4/10] Selecting k* …")
    k_star = select_k_nmf(results, min_cophenetic=0.90, min_stability=0.80)
    best = results[k_star]
    logger.info(
        "k* = %d (cophenetic=%.3f, stability=%.3f, explained_var=%.3f)",
        k_star, best.cophenetic, best.stability, best.explained_variance,
    )
    rep = ValidationReport()
    rep.add(check_nmf_stability(best.cophenetic, best.stability, k_star))
    _hard_gate(rep, "nmf_stability")

    # ---------- 5. Name factors -------------------------------------------
    logger.info("[5/10] Naming factors by top signatures …")
    names = name_factors_by_top_signatures(best.H, n_top=3)
    names.to_csv(out_dir / "factor_names.csv", index=False)
    for _, row in names.iterrows():
        logger.info("  %s ↦ %s (top: %s)",
                    row["factor"], row["name_unique"], row["top_signatures"])

    # сохраняем W (loadings per patient) и H (basis per signature)
    W = best.W.copy()
    # переименуем колонки в human-readable
    name_map = dict(zip(names["factor"], names["name_unique"]))
    W.rename(columns=name_map, inplace=True)
    best.H.index = [name_map[f] for f in best.H.index]
    W.to_csv(out_dir / "factor_loadings_W.csv")
    best.H.to_csv(out_dir / "factor_basis_H.csv")

    # ---------- 6. Composite risk score (RFS / OS) -------------------------
    logger.info("[6/10] Fitting composite risk score …")
    endpoints = cohort.endpoints.reindex(W.index)
    risk_summary = {"k_star": int(k_star),
                    "factors": list(W.columns),
                    "nmf_cophenetic": best.cophenetic,
                    "nmf_stability": best.stability,
                    "explained_variance": best.explained_variance}

    for endpoint in ["os", "dfs"]:
        tcol, ecol = f"time_{endpoint}", f"event_{endpoint}"
        if tcol not in endpoints.columns or ecol not in endpoints.columns:
            continue
        t = endpoints[tcol]
        e = endpoints[ecol]

        ev_rep = ValidationReport()
        ev_rep.add(check_event_rate(e))
        if ev_rep.failures() or ev_rep.warnings():
            _hard_gate(ev_rep, f"event_rate_{endpoint}", fail_on_warn=False)

        try:
            model = fit_composite_risk_score(
                W, t, e, clinical=cohort.clinical,
                penalizer=args.penalizer, l1_ratio=args.l1_ratio,
                random_state=args.seed,
            )
        except Exception as exc:
            logger.warning("Risk score fit failed for %s: %s", endpoint, exc)
            continue

        logger.info(
            "%s: C-index=%.3f [%.3f, %.3f] (n=%d)",
            endpoint.upper(), model.concordance_index,
            *model.c_index_bootstrap_ci, model.n_train,
        )
        # сохранение
        pd.DataFrame({
            "coefficient": model.coefficients,
        }).to_csv(out_dir / f"risk_coefs_{endpoint}.csv")

        # c-index gate
        rs_rep = ValidationReport()
        rs_rep.add(check_risk_score_discriminates(
            model.concordance_index, model.c_index_bootstrap_ci[0],
        ))
        _hard_gate(rs_rep, f"risk_discrimination_{endpoint}")

        # ---------- 7. Tertile KM -----------------------------------------
        risk = model.score(W)
        cat = model.tertile(risk)
        pd.DataFrame({"risk": risk, "tertile": cat}).to_csv(
            out_dir / f"risk_{endpoint}.csv"
        )

        km = tertile_km(risk, cat, t, e)
        km_long = []
        for level, curve in km["curves"].items():
            c = curve.copy()
            c["tertile"] = level
            km_long.append(c)
        pd.concat(km_long).to_csv(out_dir / f"tertile_km_{endpoint}.csv", index=False)
        km["pairwise"].to_csv(out_dir / f"tertile_pairwise_{endpoint}.csv", index=False)

        tert_rep = ValidationReport()
        tert_rep.add(check_tertile_separation(
            km["logrank_overall"]["p_value"],
            km["n_low"], km["n_medium"], km["n_high"],
        ))
        _hard_gate(tert_rep, f"tertile_separation_{endpoint}", fail_on_warn=False)

        # ---------- 8. Treatment × factor interaction -------------------
        if "targeted_therapy_received" in cohort.clinical.columns:
            n_annot = cohort.clinical["targeted_therapy_received"].notna().sum()
            if n_annot >= 100:
                try:
                    inter = treatment_interaction_by_factor(
                        W, t, e, cohort.clinical,
                        treatment_col="targeted_therapy_received",
                    )
                    if not inter.empty:
                        inter.to_csv(
                            out_dir / f"treatment_interaction_{endpoint}.csv", index=False
                        )
                        top_sig = inter[inter["q_interaction"] < 0.1]
                        logger.info(
                            "%s treatment×factor: %d factor(s) with q<0.10:\n%s",
                            endpoint.upper(), len(top_sig),
                            top_sig.to_string(index=False) if not top_sig.empty else "(none)",
                        )
                except Exception as exc:
                    logger.warning("Interaction analysis failed for %s: %s", endpoint, exc)

        # ---------- 9. Decision Curve Analysis --------------------------
        try:
            dca = decision_curve_analysis(risk, e)
            dca.to_csv(out_dir / f"dca_{endpoint}.csv", index=False)
            best_thresh = dca.iloc[dca["nb_model"].idxmax()]
            logger.info(
                "%s DCA: max NB=%.4f at threshold=%.2f (n_flagged=%d)",
                endpoint.upper(), best_thresh["nb_model"],
                best_thresh["threshold"], int(best_thresh["n_flagged"]),
            )
        except Exception as exc:
            logger.warning("DCA failed for %s: %s", endpoint, exc)

        risk_summary[endpoint] = {
            "c_index": model.concordance_index,
            "c_index_ci": list(model.c_index_bootstrap_ci),
            "n_train": model.n_train,
            "tertile_logrank_p": km["logrank_overall"]["p_value"],
            "tertile_sizes": {"low": km["n_low"], "medium": km["n_medium"], "high": km["n_high"]},
            "tertile_medians": km["medians"],
        }

    # ---------- 10. Save summary -----------------------------------------
    with (out_dir / "discovery_summary.json").open("w") as f:
        json.dump(_json_safe(risk_summary), f, indent=2)

    # Сохраним NMF basis для переноса (METABRIC и т.д.)
    nmf_model = {
        "H": best.H,
        "W_columns": list(W.columns),
        "k": k_star,
        "preprocessing": "robust_zscore",
    }
    with (out_dir / "nmf_model.pkl").open("wb") as f:
        pickle.dump(nmf_model, f)

    logger.info("=" * 60)
    logger.info("Discovery complete. Key artefacts in %s:", out_dir)
    logger.info("  - factor_loadings_W.csv (patient × factor)")
    logger.info("  - factor_basis_H.csv (factor × signature)")
    logger.info("  - risk_*.csv (risk score + tertile per sample)")
    logger.info("  - tertile_km_*.csv (KM curves)")
    logger.info("  - treatment_interaction_*.csv (factor × targeted therapy)")
    logger.info("  - dca_*.csv (decision curve analysis)")
    logger.info("  - nmf_model.pkl (for transfer to METABRIC)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
