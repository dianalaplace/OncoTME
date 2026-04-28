"""Build features for GSE25066 validation cohort."""

from __future__ import annotations

import argparse
import logging
import pickle
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))

import pandas as pd

from oncotme.cohorts.gse25066 import load_gse25066
from oncotme.features.signatures import load_gmt, ssgsea_scores
from oncotme.validation import (
    check_cohort_bundle,
    check_feature_frame,
    check_signature_scores,
    run_suite,
)
from oncotme.validation.result import CheckStatus, ValidationReport

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("build_features_gse25066")


def _gate(rep: ValidationReport, step: str) -> None:
    c = rep.counts()
    logger.info("[%s] PASS=%d WARN=%d FAIL=%d", step, c["PASS"], c["WARN"], c["FAIL"])
    for r in rep.results:
        if r.status == CheckStatus.WARN:
            logger.warning("  [WARN] %s: %s", r.name, r.message)
        if r.status == CheckStatus.FAIL:
            logger.error("  [FAIL] %s: %s", r.name, r.message)
    if c["FAIL"] > 0:
        sys.exit(1)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--series-matrix",
                   default=str(_REPO / "data/raw/geo/GSE25066_series_matrix.txt.gz"))
    p.add_argument("--gpl96",
                   default="/Users/dianalysenko/Documents/DrugResponce/data/raw/geo/GPL96_annotation.txt.gz")
    p.add_argument("--gmt", default=str(_REPO / "signatures" / "tme_signatures.gmt"))
    p.add_argument("--out-dir", default=str(_REPO / "data" / "processed"))
    args = p.parse_args()
    out_dir = Path(args.out_dir)

    logger.info("[1/3] Load GSE25066 …")
    cohort = load_gse25066(args.series_matrix, args.gpl96)
    logger.info("Loaded: %s", cohort)

    r1 = ValidationReport()
    r1.extend(check_cohort_bundle(cohort))
    _gate(r1, "cohort_load")

    with (out_dir / f"{cohort.name}.pkl").open("wb") as f:
        pickle.dump(cohort, f)

    logger.info("[2/3] ssGSEA …")
    gmt = load_gmt(args.gmt)
    scores = ssgsea_scores(cohort.expr, gene_sets=gmt)
    scores.columns = [f"sig__{c}" for c in scores.columns]

    individual_genes = [
        "CCL5", "CD27", "CD274", "CD276", "CD8A", "CMKLR1", "CXCL9", "CXCR6",
        "HLA-DQA1", "HLA-DRB1", "HLA-E", "IDO1", "LAG3", "NKG7", "PDCD1LG2",
        "PSMB10", "STAT1", "TIGIT", "GZMA", "PRF1",
        "HLA-A", "HLA-B", "HLA-C", "B2M", "TAP1", "TAP2", "PSMB8", "PSMB9",
        "NLRC5", "CIITA", "HLA-DRA", "CTLA4", "HAVCR2", "CD4", "CD19",
    ]
    expr_up = cohort.expr.copy()
    expr_up.index = expr_up.index.astype(str).str.upper()
    gene_cols = {}
    for g in individual_genes:
        g_up = g.upper()
        if g_up in expr_up.index:
            gene_cols[f"gene__{g}"] = expr_up.loc[g_up].reindex(scores.index).values
    gene_frame = pd.DataFrame(gene_cols, index=scores.index)

    clin_keep = ["age", "stage", "stage_t", "stage_n", "grade",
                 "subtype_er", "subtype_pr", "subtype_her2"]
    present = [c for c in clin_keep if c in cohort.clinical.columns]
    clin_frame = cohort.clinical[present].copy().reindex(scores.index)
    clin_frame.columns = [f"clin__{c}" for c in clin_frame.columns]

    features = pd.concat([scores, gene_frame, clin_frame], axis=1)
    features.index.name = "sample_id"
    logger.info("Features: %s", features.shape)

    r2 = ValidationReport()
    r2.extend(check_feature_frame(features))
    r2.extend(check_signature_scores(features))
    _gate(r2, "feature_build")

    features.to_parquet(out_dir / f"{cohort.name}_features.parquet")
    cohort.endpoints.reindex(features.index).to_parquet(
        out_dir / f"{cohort.name}_endpoints.parquet")

    logger.info("[3/3] Done → %s", out_dir)
    final = run_suite(cohorts=[cohort], feature_frames={cohort.name: features})
    logger.info("Final suite: %s", final.counts())
    return 0 if final.all_passed() else 1


if __name__ == "__main__":
    raise SystemExit(main())
