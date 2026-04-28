"""GSE25066 — Hatzis validation cohort (JAMA 2011), ~410 patients.

Same neoadjuvant T-FAC/FEC regimen, same characteristic keys as GSE25065.
Bonus fields: ``pam50_class``, ``dlda30_prediction``.

Reuses ``load_gse25065`` and extends clinical frame with PAM50.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .base import CohortBundle
from .geo_series import (
    aggregate_probes_to_genes,
    load_gpl96_annotation,
    parse_series_matrix,
)
from .gse25065 import _encode_er, _encode_pcr, _encode_stage_digit

logger = logging.getLogger(__name__)


def _normalise_pam50(value: str) -> str | float:
    if pd.isna(value):
        return np.nan
    v = str(value).strip().upper()
    return {
        "LUMA": "LUMA", "LUMINALA": "LUMA", "LUMINAL A": "LUMA",
        "LUMB": "LUMB", "LUMINALB": "LUMB", "LUMINAL B": "LUMB",
        "HER2": "HER2", "HER2-ENRICHED": "HER2",
        "BASAL": "BASAL", "BASAL-LIKE": "BASAL",
        "NORMAL": "NORMAL", "NORMAL-LIKE": "NORMAL",
    }.get(v, np.nan)


def load_gse25066(
    series_matrix_path: str | Path,
    gpl96_annotation_path: str | Path,
    *,
    name: str = "gse25066",
) -> CohortBundle:
    parsed = parse_series_matrix(series_matrix_path)
    logger.info("GSE25066: %s", parsed["title"])
    probe_to_gene = load_gpl96_annotation(gpl96_annotation_path)
    expr = aggregate_probes_to_genes(parsed["expr"], probe_to_gene, method="max_variance")
    expr.index = expr.index.astype(str).str.upper()
    expr = expr.loc[~expr.index.duplicated(keep="first")]

    char = parsed["characteristics"].copy()

    clinical = pd.DataFrame(index=char.index)
    if "age_years" in char.columns:
        clinical["age"] = pd.to_numeric(char["age_years"], errors="coerce")
    if "er_status_ihc" in char.columns:
        clinical["subtype_er"] = char["er_status_ihc"].map(_encode_er)
    if "pr_status_ihc" in char.columns:
        clinical["subtype_pr"] = char["pr_status_ihc"].map(_encode_er)
    if "her2_status" in char.columns:
        clinical["subtype_her2"] = char["her2_status"].map(_encode_er)
    if "clinical_t_stage" in char.columns:
        clinical["stage_t"] = char["clinical_t_stage"].map(lambda v: _encode_stage_digit(v, "T"))
    if "clinical_nodal_status" in char.columns:
        clinical["stage_n"] = char["clinical_nodal_status"].map(
            lambda v: _encode_stage_digit(v, "N"))
    if "clinical_ajcc_stage" in char.columns:
        clinical["stage"] = char["clinical_ajcc_stage"].map(
            lambda v: {"I": 1, "IIA": 2, "IIB": 2, "IIIA": 3, "IIIB": 3, "IIIC": 3, "IV": 4}.get(
                str(v).strip().upper(), np.nan))
    if "grade" in char.columns:
        clinical["grade"] = pd.to_numeric(char["grade"], errors="coerce")
    # BONUS (доступно в validation cohort)
    if "pam50_class" in char.columns:
        clinical["subtype_pam50"] = char["pam50_class"].map(_normalise_pam50)
    if "dlda30_prediction" in char.columns:
        clinical["dlda30_prediction"] = char["dlda30_prediction"].astype(str)
    for extra in ["set_class", "ggi_class", "chemosensitivity_prediction"]:
        if extra in char.columns:
            clinical[extra] = char[extra].astype(str)

    endpoints = pd.DataFrame(index=char.index)
    if "pathologic_response_pcr_rd" in char.columns:
        endpoints["pCR"] = char["pathologic_response_pcr_rd"].map(_encode_pcr)
    if "drfs_1_event_0_censored" in char.columns:
        endpoints["event_drfs"] = pd.to_numeric(
            char["drfs_1_event_0_censored"], errors="coerce")
    if "drfs_even_time_years" in char.columns:
        endpoints["time_drfs"] = pd.to_numeric(
            char["drfs_even_time_years"], errors="coerce") * 12.0

    common = sorted(set(expr.columns) & set(clinical.index) & set(endpoints.index))
    if len(common) < 100:
        raise ValueError(f"GSE25066: too few overlapping samples ({len(common)}).")
    expr = expr[common]
    clinical = clinical.loc[common]
    endpoints = endpoints.loc[common]

    for col in endpoints.columns:
        logger.info("endpoint %s: %d non-NaN", col, int(endpoints[col].notna().sum()))

    bundle = CohortBundle(
        name=name,
        expr=expr,
        clinical=clinical,
        endpoints=endpoints,
        treatment_context="neoadjuvant_chemo",
        platform="microarray_affy_u133a",
        expr_kind="log2_intensity",
        meta={
            "source": "GEO/GSE25066",
            "regimen": "T-FAC or FEC",
            "n_samples": len(common),
            "reference": "Hatzis C et al, JAMA 2011 (validation cohort)",
        },
    )
    bundle.sanity_check()
    return bundle
