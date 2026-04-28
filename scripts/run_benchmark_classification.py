"""Rigorous classification benchmark: nested 5-fold CV AUC для pCR-моделей.

Usage::
    python scripts/run_benchmark_classification.py \\
        --features data/processed/gse25065_features.parquet \\
        --cohort-pkl data/processed/gse25065.pkl \\
        --out-dir results/benchmark/gse25065_pcr/
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

from oncotme.benchmark.classification import (
    ClfCombinedModel,
    ClinicalLogitModel,
    CYT_Clf,
    ElasticNetLogitModel,
    NMFLogitModel,
    RandomFeaturesClf,
    Signature_Clf,
    TIS_Clf,
    nested_cv_classification,
    permutation_test_auc,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("benchmark_clf")


def _json_safe(o):
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.floating,)): return float(o) if np.isfinite(o) else None
    if isinstance(o, np.ndarray): return o.tolist()
    if isinstance(o, (pd.Series, pd.DataFrame)): return o.to_dict()
    if isinstance(o, dict): return {str(k): _json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [_json_safe(v) for v in o]
    return o


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--features", required=True)
    p.add_argument("--cohort-pkl", required=True)
    p.add_argument("--endpoint", default="pCR")
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--n-permutations", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", default=str(_REPO / "results" / "benchmark_clf"))
    p.add_argument("--skip-permutation", action="store_true")
    return p.parse_args()


def main() -> int:
    args = _parse()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    features = pd.read_parquet(args.features)
    with open(args.cohort_pkl, "rb") as f:
        cohort = pickle.load(f)
    y = cohort.endpoints[args.endpoint]
    mask = y.notna()
    y = y.loc[mask].astype(int)
    logger.info(
        "Cohort=%s  endpoint=%s  N=%d  pos=%d (%.1f%%)",
        cohort.name, args.endpoint, len(y), int(y.sum()), 100 * y.mean(),
    )

    models = {
        "clinical_only": lambda: ClinicalLogitModel(),
        "TIS_18": lambda: TIS_Clf(),
        "CYT": lambda: CYT_Clf(),
        "sig_IFN_gamma": lambda: Signature_Clf("sig__IFN_gamma"),
        "sig_APM": lambda: Signature_Clf("sig__APM_score"),
        "sig_CD8": lambda: Signature_Clf("sig__CD8_T_cell"),
        "random_feat_s0": lambda: RandomFeaturesClf(n_features=6, seed=0),
        "random_feat_s1": lambda: RandomFeaturesClf(n_features=6, seed=1),
        "NMF_logit_k5": lambda: NMFLogitModel(k=5, n_runs=3),
        "ElasticNet_all": lambda: ElasticNetLogitModel(use_clinical=False),
        "ElasticNet_all+clinical": lambda: ElasticNetLogitModel(use_clinical=True),
        "NMF+clinical": lambda: ClfCombinedModel(
            NMFLogitModel(k=5, n_runs=3), ClinicalLogitModel(), name="NMF+clinical",
        ),
    }

    rows = []
    results = {}
    for mname, factory in models.items():
        logger.info("  model: %s", mname)
        try:
            r = nested_cv_classification(
                factory, features, y, cohort.clinical,
                n_folds=args.n_folds, seed=args.seed, n_bootstrap=500,
            )
            results[mname] = r
            rows.append({
                "model": mname,
                "auc": r.mean_auc, "auc_ci_lo": r.auc_ci[0], "auc_ci_hi": r.auc_ci[1],
                "ap": r.mean_ap,
                "n": r.n, "n_pos": r.n_pos,
                "fold_auc": r.fold_auc,
            })
        except Exception as exc:
            logger.warning("  %s failed: %s", mname, exc)
            rows.append({"model": mname, "auc": None, "error": str(exc)})

    bench = pd.DataFrame(rows)
    base = bench.loc[bench["model"] == "clinical_only", "auc"]
    if len(base) and pd.notna(base.iloc[0]):
        bench["delta_vs_clinical"] = bench["auc"].apply(
            lambda x: (x - base.iloc[0]) if x is not None and pd.notna(x) else None
        )
    bench.to_csv(out_dir / "benchmark_auc.csv", index=False)
    print("\n" + "=" * 90)
    print(bench.drop(columns=["fold_auc"], errors="ignore").to_string(
        index=False, float_format=lambda x: f"{x:.4f}" if pd.notna(x) else "—",
    ))
    print("=" * 90)

    # Permutation test для ключевых моделей
    perm_results = {}
    if not args.skip_permutation:
        for key in ["NMF_logit_k5", "ElasticNet_all+clinical", "NMF+clinical"]:
            if key not in results:
                continue
            obs = results[key].mean_auc
            logger.info("Permutation test %s (observed=%.4f)", key, obs)
            perm_results[key] = permutation_test_auc(
                models[key], features, y, obs, cohort.clinical,
                n_permutations=args.n_permutations, n_folds=args.n_folds,
                seed=args.seed,
            )
            logger.info(
                "  p=%.4f | null mean=%.3f | null p95=%.3f",
                perm_results[key].get("p_value", float("nan")),
                perm_results[key].get("null_mean", float("nan")),
                perm_results[key].get("null_p95", float("nan")),
            )

    summary = {
        "endpoint": args.endpoint,
        "cohort": cohort.name,
        "n": int(len(y)),
        "n_pos": int(y.sum()),
        "benchmark": bench.to_dict(orient="records"),
        "permutation": perm_results,
    }
    with (out_dir / "benchmark_summary.json").open("w") as f:
        json.dump(_json_safe(summary), f, indent=2)

    # verdict
    verdict = [
        f"# {cohort.name} rigorous benchmark — {args.endpoint}",
        "",
        f"- N: {len(y)} | positives: {int(y.sum())} ({y.mean():.1%})",
        f"- Regimen: {cohort.treatment_context}; "
        f"{cohort.meta.get('regimen', '-')}",
        "",
        "## Nested 5-fold CV AUC",
        "",
        "| Model | AUC | 95% CI | ΔAUC vs clinical | AP |",
        "|---|---|---|---|---|",
    ]
    for _, row in bench.iterrows():
        auc = row.get("auc")
        lo, hi = row.get("auc_ci_lo"), row.get("auc_ci_hi")
        d = row.get("delta_vs_clinical")
        ap = row.get("ap")
        auc_str = f"{auc:.4f}" if auc is not None and pd.notna(auc) else "—"
        ci_str = f"[{lo:.3f}, {hi:.3f}]" if lo is not None and pd.notna(lo) else "—"
        d_str = f"{d:+.3f}" if d is not None and pd.notna(d) else "—"
        ap_str = f"{ap:.3f}" if ap is not None and pd.notna(ap) else "—"
        verdict.append(f"| {row['model']} | {auc_str} | {ci_str} | {d_str} | {ap_str} |")

    if perm_results:
        verdict += ["", "## Permutation test"]
        for k, d in perm_results.items():
            verdict.append(
                f"- **{k}**: observed AUC={d.get('observed', float('nan')):.4f} | "
                f"null mean={d.get('null_mean', float('nan')):.3f} | "
                f"null p95={d.get('null_p95', float('nan')):.3f} | "
                f"**p={d.get('p_value', float('nan')):.4f}**"
            )

    # auto-verdict
    best_row = bench.loc[bench["delta_vs_clinical"].fillna(-1).idxmax()]
    best_name = best_row["model"]
    best_delta = best_row.get("delta_vs_clinical", None)
    verdict += ["", "## Verdict"]
    if best_delta is not None and pd.notna(best_delta):
        if best_delta >= 0.03:
            verdict.append(
                f"✅ **Clinically meaningful signal**: '{best_name}' beats clinical by "
                f"ΔAUC = {best_delta:+.3f}. Proceed to external validation + "
                f"subtype-stratified analysis."
            )
        elif best_delta >= 0.01:
            verdict.append(
                f"🟡 **Incremental**: '{best_name}' +ΔAUC = {best_delta:+.3f} over "
                f"clinical. Consider but not revolutionary."
            )
        else:
            verdict.append(
                f"🔴 **No added value**: best TME-based model only +{best_delta:+.3f} "
                f"over clinical. Likely underpowered on N={len(y)}; need larger cohort "
                f"or richer features."
            )
    (out_dir / "VERDICT.md").write_text("\n".join(verdict), encoding="utf-8")

    logger.info("Done. Artefacts → %s", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
