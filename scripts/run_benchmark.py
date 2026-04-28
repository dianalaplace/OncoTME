"""Rigorous benchmark: nested 5-fold CV C-index для всех моделей + честный сравнительный вывод.

Steps:
1. Загрузка features + cohort.
2. Nested CV для каждой модели:
   - clinical_only
   - TIS_18 (Ayers 2017)
   - CYT (Rooney 2015)
   - single_signature IFN_gamma
   - random_features (5 разных seed)
   - NMF_risk (k=6)
   - NMF + clinical (stacked)
3. Benchmark table с CI, ΔC vs clinical-only.
4. Permutation test для NMF и NMF+clinical (n_perm=100).
5. Schoenfeld PH check на risk-score (single full-data fit).
6. Partial correlation risk vs age.
7. NMF init sensitivity (nndsvda vs random).
8. Save benchmark_table.csv + benchmark_summary.json + verdict.md.
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

from oncotme.benchmark import (
    CYTModel,
    ClinicalCoxModel,
    CombinedModel,
    NMFRiskModel,
    RandomFeaturesModel,
    SignatureCoxModel,
    TISModel,
    nested_cv_evaluate,
    partial_correlation_check,
    permutation_test,
    schoenfeld_check,
)
from oncotme.benchmark.confounding import nmf_init_sensitivity
from oncotme.discovery.preprocess import preprocess_for_clustering

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("benchmark")


def _json_safe(obj):
    if isinstance(obj, (np.integer,)): return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj) if np.isfinite(obj) else None
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, (pd.Series, pd.DataFrame)): return obj.to_dict()
    if isinstance(obj, dict): return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)): return [_json_safe(v) for v in obj]
    return obj


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--features", required=True)
    p.add_argument("--cohort-pkl", required=True)
    p.add_argument("--out-dir", default=str(_REPO / "results" / "benchmark"))
    p.add_argument("--endpoint", choices=["os", "dfs"], default="os")
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--n-permutations", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--skip-permutation", action="store_true")
    p.add_argument("--skip-sensitivity", action="store_true")
    return p.parse_args()


def main() -> int:
    args = _parse()
    out_dir = Path(args.out_dir) / f"{args.endpoint}"
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("[1/6] Loading …")
    features = pd.read_parquet(args.features)
    with open(args.cohort_pkl, "rb") as f:
        cohort = pickle.load(f)
    endpoints = cohort.endpoints
    t = endpoints[f"time_{args.endpoint}"]
    e = endpoints[f"event_{args.endpoint}"]
    clinical = cohort.clinical
    logger.info("features=%s n_with_outcome=%d n_events=%d",
                features.shape,
                int((t.notna() & e.notna()).sum()),
                int(e.fillna(0).sum()))

    logger.info("[2/6] Nested CV benchmark …")

    # Факторы моделей (каждая модель — factory, возвращающая НОВЫЙ инстанс)
    model_factories = {
        "clinical_only": lambda: ClinicalCoxModel(),
        "TIS_18": lambda: TISModel(),
        "CYT": lambda: CYTModel(),
        "sig_IFN_gamma": lambda: SignatureCoxModel(signature="sig__IFN_gamma"),
        "random_feat_s0": lambda: RandomFeaturesModel(n_features=6, seed=0),
        "random_feat_s1": lambda: RandomFeaturesModel(n_features=6, seed=1),
        "random_feat_s2": lambda: RandomFeaturesModel(n_features=6, seed=2),
        "NMF_risk_k6": lambda: NMFRiskModel(k=6, n_runs=5),  # в nested CV — 5 runs
        "NMF_plus_clinical": lambda: CombinedModel(
            NMFRiskModel(k=6, n_runs=5), ClinicalCoxModel(),
            name="NMF+clinical",
        ),
    }

    rows = []
    nested_results = {}
    for mname, factory in model_factories.items():
        logger.info("  model: %s", mname)
        try:
            res = nested_cv_evaluate(
                factory, features, t, e, clinical,
                n_folds=args.n_folds, seed=args.seed,
                n_bootstrap=500,
            )
            nested_results[mname] = res
            rows.append({
                "model": mname,
                "c_index": res.mean_c_index,
                "ci_lo": res.ci_lower,
                "ci_hi": res.ci_upper,
                "fold_c": res.fold_c_indices,
                "n": res.total_n,
                "n_events": res.total_events,
            })
        except Exception as exc:
            logger.warning("  %s failed: %s", mname, exc)
            rows.append({"model": mname, "c_index": None, "error": str(exc)})

    bench = pd.DataFrame(rows)
    # ΔC vs clinical
    clin_c = bench.loc[bench["model"] == "clinical_only", "c_index"].iloc[0]
    if isinstance(clin_c, (int, float)) and np.isfinite(clin_c):
        bench["delta_vs_clinical"] = bench["c_index"].apply(
            lambda x: (x - clin_c) if (x is not None and np.isfinite(x)) else None
        )
    bench.to_csv(out_dir / "benchmark_table.csv", index=False)
    logger.info("Saved benchmark_table.csv")
    print("\n" + "=" * 70)
    print(bench.drop(columns=["fold_c"], errors="ignore").to_string(index=False,
          float_format=lambda x: f"{x:.4f}" if pd.notna(x) else "—"))
    print("=" * 70)

    # ---------- Permutation test (key models) ------------------------------
    perm_results = {}
    if not args.skip_permutation:
        logger.info("[3/6] Permutation test (n=%d) …", args.n_permutations)
        for key_model in ["NMF_risk_k6", "NMF_plus_clinical"]:
            if key_model not in nested_results:
                continue
            observed = nested_results[key_model].mean_c_index
            logger.info("  permuting %s (observed C=%.4f) …", key_model, observed)
            perm = permutation_test(
                model_factories[key_model], features, t, e, observed, clinical,
                n_permutations=args.n_permutations, n_folds=args.n_folds,
                seed=args.seed,
            )
            perm_results[key_model] = perm
            logger.info(
                "    p=%.4f | null mean=%.3f | null p95=%.3f | n_success=%d",
                perm.get("p_value", float("nan")),
                perm.get("null_mean", float("nan")),
                perm.get("null_p95", float("nan")),
                perm.get("n_success", 0),
            )

    # ---------- Schoenfeld PH + partial correlation -----------------------
    logger.info("[4/6] PH + partial corr (single-fit NMF risk on full data) …")
    diag: dict = {}
    try:
        nmf_model = NMFRiskModel(k=6, n_runs=10).fit(features, t, e, clinical)
        risk = nmf_model.score(features, clinical)
        ph = schoenfeld_check(risk, t, e)
        logger.info("  Schoenfeld p=%.4f | HR=%.3f [%.3f, %.3f] | PH violated=%s",
                    ph["ph_test_p"], ph["hr"], ph["hr_ci_lo"], ph["hr_ci_hi"],
                    ph["ph_violated"])
        pc = partial_correlation_check(risk, t, e, clinical)
        logger.info("  Raw Spearman rho=%.3f (p=%.1e); "
                    "Partial (controlling clinical) rho=%.3f (p=%.1e); proxy_warning=%s",
                    pc["raw_spearman_rho"], pc["raw_spearman_p"],
                    pc["partial_spearman_rho"], pc["partial_spearman_p"],
                    pc["proxy_warning"])
        diag = {"schoenfeld": ph, "partial_correlation": pc}
    except Exception as exc:
        logger.warning("  diagnostic fit failed: %s", exc)

    # ---------- NMF init sensitivity --------------------------------------
    sens = {}
    if not args.skip_sensitivity:
        logger.info("[5/6] NMF init sensitivity (nndsvda vs random) …")
        try:
            X_pre = preprocess_for_clustering(features, robust=True)
            sens = nmf_init_sensitivity(X_pre, k=6, n_runs=30, seed=args.seed)
            logger.info("  nndsvda: cophenetic=%.3f stability=%.3f",
                        sens["nndsvda"]["cophenetic"], sens["nndsvda"]["stability"])
            logger.info("  random:  cophenetic=%.3f stability=%.3f",
                        sens["random"]["cophenetic"], sens["random"]["stability"])
        except Exception as exc:
            logger.warning("  sensitivity failed: %s", exc)

    # ---------- Summary JSON + verdict markdown ---------------------------
    summary = {
        "endpoint": args.endpoint,
        "n_samples": int((t.notna() & e.notna()).sum()),
        "n_events": int(e.fillna(0).sum()),
        "benchmark": bench.to_dict(orient="records"),
        "permutation": perm_results,
        "diagnostics": diag,
        "nmf_init_sensitivity": sens,
    }
    with (out_dir / "benchmark_summary.json").open("w") as f:
        json.dump(_json_safe(summary), f, indent=2)

    # Verdict markdown
    _write_verdict(out_dir / "VERDICT.md", summary)
    logger.info("[6/6] Done. Artefacts → %s", out_dir)
    return 0


def _write_verdict(path: Path, summary: dict) -> None:
    bench = pd.DataFrame(summary["benchmark"])
    clinical_row = bench[bench["model"] == "clinical_only"]
    nmf_row = bench[bench["model"] == "NMF_risk_k6"]
    combined_row = bench[bench["model"] == "NMF_plus_clinical"]

    def _fmt(row, col="c_index"):
        if row.empty:
            return "—"
        v = row[col].iloc[0]
        return f"{v:.4f}" if isinstance(v, (int, float)) and pd.notna(v) else "—"

    def _delta(row):
        return _fmt(row, "delta_vs_clinical")

    verdict_lines = [
        f"# TCGA-BRCA rigorous benchmark — {summary['endpoint'].upper()}",
        "",
        f"- N (non-NaN outcome): {summary['n_samples']}",
        f"- Events: {summary['n_events']}",
        "",
        "## Nested 5-fold CV C-index",
        "",
        "| Model | C-index | 95% CI | ΔC vs clinical |",
        "|---|---|---|---|",
    ]
    for _, row in bench.iterrows():
        c = row.get("c_index")
        lo, hi = row.get("ci_lo"), row.get("ci_hi")
        d = row.get("delta_vs_clinical")
        ci_str = f"[{lo:.3f}, {hi:.3f}]" if lo is not None and pd.notna(lo) else "—"
        c_str = f"{c:.4f}" if c is not None and pd.notna(c) else "—"
        d_str = (f"{d:+.3f}" if d is not None and pd.notna(d) else "—")
        verdict_lines.append(f"| {row['model']} | {c_str} | {ci_str} | {d_str} |")

    perm = summary.get("permutation", {})
    if perm:
        verdict_lines += ["", "## Permutation test (null distribution C-index)", ""]
        for m, d in perm.items():
            verdict_lines.append(
                f"- **{m}**: observed C={d.get('observed', float('nan')):.4f} | "
                f"null mean={d.get('null_mean', float('nan')):.3f} | "
                f"null p95={d.get('null_p95', float('nan')):.3f} | "
                f"**p={d.get('p_value', float('nan')):.4f}**"
            )

    diag = summary.get("diagnostics", {})
    if diag:
        verdict_lines += ["", "## Diagnostics", ""]
        ph = diag.get("schoenfeld", {})
        if ph:
            verdict_lines += [
                f"- **Schoenfeld PH test**: p={ph.get('ph_test_p', float('nan')):.4f}"
                f" | HR={ph.get('hr', float('nan')):.2f} "
                f"[{ph.get('hr_ci_lo', float('nan')):.2f}, "
                f"{ph.get('hr_ci_hi', float('nan')):.2f}] | "
                f"PH violated: **{ph.get('ph_violated', False)}**",
            ]
        pc = diag.get("partial_correlation", {})
        if pc:
            verdict_lines += [
                f"- **Spearman ρ (risk vs time | observed)**: "
                f"raw={pc.get('raw_spearman_rho', float('nan')):.3f} → "
                f"partial (control clinical)={pc.get('partial_spearman_rho', float('nan')):.3f}"
                f" | proxy warning: **{pc.get('proxy_warning', False)}**",
            ]

    sens = summary.get("nmf_init_sensitivity", {})
    if sens:
        verdict_lines += ["", "## NMF init sensitivity", ""]
        for init_name in ("nndsvda", "random"):
            d = sens.get(init_name, {})
            if d:
                verdict_lines.append(
                    f"- init=`{init_name}`: cophenetic={d.get('cophenetic', float('nan')):.3f}, "
                    f"stability={d.get('stability', float('nan')):.3f}, "
                    f"explained_var={d.get('explained_variance', float('nan')):.3f}"
                )

    # Автоматическая рекомендация
    verdict_lines += ["", "## Verdict (auto-generated)", ""]
    if not nmf_row.empty and not clinical_row.empty:
        delta = nmf_row["delta_vs_clinical"].iloc[0]
        perm_p = perm.get("NMF_risk_k6", {}).get("p_value")
        if delta is not None and pd.notna(delta):
            if delta >= 0.03 and (perm_p is None or perm_p < 0.05):
                verdict_lines.append(
                    f"✅ **Go to METABRIC**: NMF risk ΔC = {delta:+.3f} vs clinical; "
                    f"permutation p = {perm_p}."
                )
            elif delta >= 0.01:
                verdict_lines.append(
                    f"🟡 **Incremental signal** (ΔC = {delta:+.3f}). METABRIC validation "
                    f"worthwhile but expect modest results."
                )
            else:
                verdict_lines.append(
                    f"🔴 **Stop; reconsider**: ΔC = {delta:+.3f} — TME risk is not "
                    f"distinguishable from clinical baseline. "
                    f"Revisit feature set, k, cohort scope."
                )
    path.write_text("\n".join(verdict_lines), encoding="utf-8")
    logger.info("Saved %s", path.name)


if __name__ == "__main__":
    raise SystemExit(main())
