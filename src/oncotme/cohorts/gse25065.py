"""GSE25065 Hatzis validation cohort loader — MDACC neoadjuvant chemo + pCR.

Paper: Hatzis C, et al. JAMA 2011. Genomic predictor of response and survival
following taxane-anthracycline chemotherapy in breast cancer.

Treatment: T-FAC or FEC (taxane + anthracycline), neoadjuvant, homogeneous arm.
Endpoint: pCR (pathologic complete response) — primary; DRFS — secondary.
Platform: GPL96 (Affymetrix U133A), log2 normalized.

Cohort homogeneity → чистый testbed для "TME as predictor of chemo pCR" —
не смешивается с endocrine / targeted.
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

logger = logging.getLogger(__name__)


def _encode_pcr(value: str) -> float:
    if pd.isna(value):
        return np.nan
    v = str(value).strip().upper()
    if v in {"PCR"}:
        return 1.0
    if v in {"RD"}:  # residual disease
        return 0.0
    return np.nan


def _encode_er(value: str) -> float:
    if pd.isna(value):
        return np.nan
    v = str(value).strip().upper()
    if v in {"P", "POSITIVE", "POS", "1"}:
        return 1.0
    if v in {"N", "NEGATIVE", "NEG", "0"}:
        return 0.0
    return np.nan


def _encode_stage_digit(value: str, prefix: str) -> float:
    if pd.isna(value):
        return np.nan
    v = str(value).upper().replace(" ", "")
    if not v.startswith(prefix.upper()):
        return np.nan
    digits = "".join(ch for ch in v if ch.isdigit())
    if not digits:
        return np.nan
    return float(int(digits[0]))


def load_gse25065(
    series_matrix_path: str | Path,
    gpl96_annotation_path: str | Path,
    *,
    name: str = "gse25065",
    treatment_context: str = "neoadjuvant_chemo",
    regimen: str = "T-FAC or FEC",
) -> CohortBundle:
    """Собирает CohortBundle для GSE25065.

    Parameters
    ----------
    series_matrix_path
        Path to ``GSE25065_series_matrix.txt[.gz]``.
    gpl96_annotation_path
        Path to ``GPL96_annotation.txt[.gz]``.
    """
    parsed = parse_series_matrix(series_matrix_path)
    logger.info("GSE25065: %s", parsed["title"])
    if parsed["platform_id"] != "GPL96":
        logger.warning(
            "Expected GPL96 platform, got %s — will still attempt mapping.",
            parsed["platform_id"],
        )

    # probe → gene
    probe_to_gene = load_gpl96_annotation(gpl96_annotation_path)
    expr_genes = aggregate_probes_to_genes(
        parsed["expr"], probe_to_gene, method="max_variance",
    )
    expr_genes.index = expr_genes.index.astype(str).str.upper()
    expr_genes = expr_genes.loc[~expr_genes.index.duplicated(keep="first")]

    # характеристики → clinical + endpoints
    char = parsed["characteristics"].copy()

    clinical = pd.DataFrame(index=char.index)
    # age
    if "age_years" in char.columns:
        clinical["age"] = pd.to_numeric(char["age_years"], errors="coerce")
    # subtype IHC
    if "er_status_ihc" in char.columns:
        clinical["subtype_er"] = char["er_status_ihc"].map(_encode_er)
    if "pr_status_ihc" in char.columns:
        clinical["subtype_pr"] = char["pr_status_ihc"].map(_encode_er)
    if "her2_status" in char.columns:
        clinical["subtype_her2"] = char["her2_status"].map(_encode_er)
    # stage
    if "clinical_t_stage" in char.columns:
        clinical["stage_t"] = char["clinical_t_stage"].map(
            lambda v: _encode_stage_digit(v, "T")
        )
    if "clinical_nodal_status" in char.columns:
        clinical["stage_n"] = char["clinical_nodal_status"].map(
            lambda v: _encode_stage_digit(v, "N")
        )
    if "clinical_ajcc_stage" in char.columns:
        clinical["stage"] = char["clinical_ajcc_stage"].map(
            lambda v: {"I": 1, "IIA": 2, "IIB": 2, "IIIA": 3, "IIIB": 3, "IIIC": 3, "IV": 4}.get(
                str(v).strip().upper(), np.nan
            )
        )
    if "grade" in char.columns:
        clinical["grade"] = pd.to_numeric(char["grade"], errors="coerce")
    # taxane type как treatment stratifier
    if "type_taxane" in char.columns:
        clinical["taxane_type"] = char["type_taxane"].astype(str).str.lower()
    # extra biomarker annotations (Hatzis SET class, GGI, chemosensitivity prediction)
    for extra in ["set_class", "ggi_class", "chemosensitivity_prediction"]:
        if extra in char.columns:
            clinical[extra] = char[extra].astype(str)

    # endpoints
    endpoints = pd.DataFrame(index=char.index)
    if "pathologic_response_pcr_rd" in char.columns:
        endpoints["pCR"] = char["pathologic_response_pcr_rd"].map(_encode_pcr)
    # DRFS (time + event)
    if "drfs_1_event_0_censored" in char.columns:
        endpoints["event_drfs"] = pd.to_numeric(
            char["drfs_1_event_0_censored"], errors="coerce"
        )
    if "drfs_even_time_years" in char.columns:
        # конвертируем годы → месяцы для унификации с TCGA (там дни / месяцы)
        endpoints["time_drfs"] = pd.to_numeric(
            char["drfs_even_time_years"], errors="coerce"
        ) * 12.0

    # align samples between expr и clinical
    common = sorted(set(expr_genes.columns) & set(clinical.index) & set(endpoints.index))
    if len(common) < 50:
        raise ValueError(f"GSE25065: too few overlapping samples ({len(common)}).")
    expr_genes = expr_genes[common]
    clinical = clinical.loc[common]
    endpoints = endpoints.loc[common]

    # лог покрытия ключевых полей
    for col in endpoints.columns:
        n_nn = int(endpoints[col].notna().sum())
        logger.info("endpoint %s: %d non-NaN (of %d)", col, n_nn, len(common))

    bundle = CohortBundle(
        name=name,
        expr=expr_genes,
        clinical=clinical,
        endpoints=endpoints,
        treatment_context=treatment_context,
        platform="microarray_affy_u133a",
        expr_kind="log2_intensity",
        meta={
            "source": "GEO/GSE25065",
            "regimen": regimen,
            "n_samples": len(common),
            "n_genes": expr_genes.shape[0],
            "reference": "Hatzis C et al, JAMA 2011",
        },
    )
    bundle.sanity_check()
    return bundle
