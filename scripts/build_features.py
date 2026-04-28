"""End-to-end: TCGA-BRCA → log2(TPM+1) → ssGSEA → TME-панель → parquet.

Каждый шаг заканчивается вызовом ``run_suite`` / таргетированной проверки;
при FAIL скрипт завершается с exit code 1, артефакты не перезаписываются.

Usage::

    python scripts/build_features.py \\
        --tcga-rnaseq  ../DrugResponce/data/raw/tcga_brca_rnaseq_tpm.tsv \\
        --tcga-clinical ../DrugResponce/data/raw/tcga_brca_clinical.tsv \\
        --xena-clinical ../DrugResponce/data/raw/BRCA_clinicalMatrix.tsv \\
        --xena-survival ../DrugResponce/data/raw/BRCA_survival.tsv \\
        --out-dir data/processed/
"""

from __future__ import annotations

import argparse
import logging
import pickle
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))

import pandas as pd

from oncotme.cohorts.tcga import load_tcga_brca
from oncotme.features.signatures import load_gmt, ssgsea_scores
from oncotme.validation import (
    run_suite,
    check_cohort_bundle,
    check_feature_frame,
    check_signature_scores,
)
from oncotme.validation.result import CheckStatus, ValidationReport

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("build_features")


def _hard_gate(report: ValidationReport, step: str) -> None:
    """Печатаем отчёт; если FAIL — падаем с exit 1."""
    counts = report.counts()
    logger.info(
        "[%s] PASS=%d WARN=%d FAIL=%d",
        step, counts["PASS"], counts["WARN"], counts["FAIL"],
    )
    if counts["FAIL"] > 0:
        logger.error("Validation FAILED at step %r.", step)
        for r in report.failures():
            logger.error("  [FAIL] %s: %s", r.name, r.message)
        sys.exit(1)
    if counts["WARN"] > 0:
        for r in report.warnings():
            logger.warning("  [WARN] %s: %s", r.name, r.message)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build OncoTME features end-to-end.")
    p.add_argument("--tcga-rnaseq", required=True, help="TPM matrix TSV.")
    p.add_argument("--tcga-clinical", required=True, help="GDC clinical TSV.")
    p.add_argument("--xena-clinical", default=None, help="Xena clinicalMatrix (optional).")
    p.add_argument("--xena-survival", default=None, help="Xena survival TSV (optional).")
    p.add_argument("--gmt", default=str(_REPO / "signatures" / "tme_signatures.gmt"))
    p.add_argument("--out-dir", default=str(_REPO / "data" / "processed"))
    p.add_argument(
        "--validation-out",
        default=str(_REPO / "results" / "validation" / "build_features.json"),
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------- Шаг 1: загрузка TCGA ----------------------------------------
    logger.info("[1/4] Loading TCGA-BRCA …")
    cohort = load_tcga_brca(
        expression_path=args.tcga_rnaseq,
        clinical_path=args.tcga_clinical,
        xena_clinical_path=args.xena_clinical,
        survival_path=args.xena_survival,
    )
    logger.info("Loaded: %s", cohort)

    rep1 = ValidationReport()
    rep1.extend(check_cohort_bundle(cohort))
    _hard_gate(rep1, "cohort_load")

    cohort_pkl = out_dir / f"{cohort.name}.pkl"
    with cohort_pkl.open("wb") as f:
        pickle.dump(cohort, f)
    logger.info("Cohort pickled → %s", cohort_pkl)

    # ---------- Шаг 2: ssGSEA -----------------------------------------------
    logger.info("[2/4] Computing ssGSEA scores …")
    gene_sets = load_gmt(args.gmt)
    logger.info("Loaded %d signatures from GMT.", len(gene_sets))
    scores = ssgsea_scores(cohort.expr, gene_sets=gene_sets)

    if scores.empty or scores.shape[0] != cohort.n_samples:
        logger.error(
            "ssGSEA shape mismatch: got %s, expected %d samples.",
            scores.shape, cohort.n_samples,
        )
        return 1

    scores.columns = [f"sig__{c}" for c in scores.columns]

    # ---------- Шаг 3: сборка TME-панели + individual genes + clinical ------
    logger.info("[3/4] Assembling TME feature frame …")

    # individual genes (HLA/APM/checkpoints, joined из config)
    individual_genes = [
        "HLA-A", "HLA-B", "HLA-C", "B2M", "TAP1", "TAP2", "PSMB8", "PSMB9",
        "NLRC5", "CIITA", "HLA-DRA", "HLA-DRB1", "HLA-DQA1", "HLA-DQB1",
        "CD274", "PDCD1LG2", "CTLA4", "LAG3", "HAVCR2", "TIGIT", "IDO1",
        "GZMA", "PRF1",
    ]
    expr_upper_idx = cohort.expr.index.astype(str).str.upper()
    expr_upper = cohort.expr.copy()
    expr_upper.index = expr_upper_idx
    gene_cols = {}
    for g in individual_genes:
        g_up = g.upper()
        if g_up in expr_upper.index:
            gene_cols[f"gene__{g}"] = expr_upper.loc[g_up].reindex(scores.index).values
    gene_frame = pd.DataFrame(gene_cols, index=scores.index)

    clin_keep = ["age", "stage", "subtype_er", "subtype_pr", "subtype_her2", "ki67"]
    present = [c for c in clin_keep if c in cohort.clinical.columns]
    clin_frame = cohort.clinical[present].copy().reindex(scores.index)
    clin_frame.columns = [f"clin__{c}" for c in clin_frame.columns]

    features = pd.concat([scores, gene_frame, clin_frame], axis=1)
    features.index.name = "sample_id"

    logger.info(
        "Feature frame: %d samples × %d features (sig=%d, gene=%d, clin=%d).",
        *features.shape,
        len([c for c in features.columns if c.startswith("sig__")]),
        len([c for c in features.columns if c.startswith("gene__")]),
        len([c for c in features.columns if c.startswith("clin__")]),
    )

    rep2 = ValidationReport()
    rep2.extend(check_feature_frame(features))
    rep2.extend(check_signature_scores(features))
    _hard_gate(rep2, "feature_build")

    features_pq = out_dir / f"{cohort.name}_features.parquet"
    features.to_parquet(features_pq)
    logger.info("Features → %s", features_pq)

    # ---------- Шаг 4: final suite + endpoints ------------------------------
    logger.info("[4/4] Final validation suite …")
    final = run_suite(
        cohorts=[cohort],
        feature_frames={cohort.name: features},
    )
    print(final.to_text())
    final.to_json(args.validation_out)
    logger.info("Validation report → %s", args.validation_out)

    # также сохраняем endpoints отдельным parquet для удобства моделей
    endpoints_pq = out_dir / f"{cohort.name}_endpoints.parquet"
    cohort.endpoints.reindex(features.index).to_parquet(endpoints_pq)
    logger.info("Endpoints → %s", endpoints_pq)

    return 0 if final.all_passed() else 1


if __name__ == "__main__":
    raise SystemExit(main())
