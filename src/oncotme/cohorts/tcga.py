"""TCGA-BRCA loader → CohortBundle.

Поддерживает два источника clinical метаданных (GDC + UCSC Xena),
автоматически выбирая доступный. Выход — ``CohortBundle`` с:
- expr: gene × sample в log2(TPM+1), indexed by HGNC symbols
- clinical: age, subtype (ER/PR/HER2/PAM50), stage, Ki-67
- endpoints: pCR (где есть), time_os/event_os, time_dfs/event_dfs

В OncoTME survival — первичный endpoint, поэтому DFS/OS поля обязательны;
pCR — optional (в TCGA обычно отсутствует кроме подмножеств).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .base import CohortBundle

logger = logging.getLogger(__name__)


# Каноничные → алиасы, которые встречаются в TCGA / Xena / nature2012 dumps.
_CLIN_ALIASES: dict[str, list[str]] = {
    "patient_id": [
        "sampleID", "bcr_patient_barcode", "bcr_sample_barcode",
        "case_submitter_id", "submitter_id", "_PATIENT", "patient_id", "case_id",
    ],
    "subtype_pam50": [
        "subtype_PAM50", "pam50", "PAM50", "PAM50Call_RNAseq", "PAM50_mRNA_nature2012",
    ],
    "age": [
        "age_at_diagnosis", "age", "diagnosis_age",
        "age_at_initial_pathologic_diagnosis", "Age_at_Initial_Pathologic_Diagnosis_nature2012",
    ],
    "stage_T": ["tumor_stage_T", "ajcc_pathologic_t", "pathologic_T", "t_stage", "Tumor_nature2012"],
    "stage_N": ["tumor_stage_N", "ajcc_pathologic_n", "pathologic_N", "n_stage", "Node_nature2012"],
    "subtype_er": ["ER_status", "er_status", "er_status_by_ihc", "ER_Status_nature2012"],
    "subtype_pr": ["PR_status", "pr_status", "pr_status_by_ihc", "PR_Status_nature2012"],
    "subtype_her2": [
        "HER2_status", "her2_status", "her2_status_by_ihc", "HER2_Final_Status_nature2012",
    ],
    "ki67": ["Ki67", "ki67", "proliferation_index"],
    "grade": ["histologic_grade", "neoplasm_histologic_grade", "grade"],
    # survival
    "time_os": ["OS_Time_nature2012", "OS.time", "os_days", "days_to_death", "overall_survival_days"],
    "event_os": ["OS_event_nature2012", "OS", "vital_status", "death_event"],
    "time_dfs": [
        "DFS_months", "disease_free_survival_months",
        "days_to_new_tumor_event_after_initial_treatment", "DFS.time",
    ],
    "event_dfs": [
        "DFS_event", "disease_free_status", "new_tumor_event",
        "recurrence_event", "new_tumor_event_after_initial_treatment", "DFS",
    ],
    "pCR": [
        "pCR", "pathologic_complete_response", "response", "binary_response",
    ],
    # Treatment / therapy flags (из Xena clinicalMatrix)
    "targeted_therapy_received": [
        "targeted_molecular_therapy", "targeted_therapy", "molecular_therapy_received",
    ],
    "radiation_received": ["radiation_therapy", "radiation_therapy_received"],
    "neoadjuvant_received": ["history_of_neoadjuvant_treatment", "neoadjuvant_therapy"],
    "pharmaceutical_additional": ["additional_pharmaceutical_therapy"],
}


def _normalize_patient_id(sample_id: str) -> str:
    if not isinstance(sample_id, str):
        return str(sample_id)
    token = sample_id.strip().replace(".", "-")
    parts = token.split("-")
    if len(parts) >= 3 and parts[0].upper() == "TCGA":
        return "-".join(parts[:3]).upper()
    return token


def _first_present(df: pd.DataFrame, aliases: list[str]) -> Optional[str]:
    for col in aliases:
        if col in df.columns:
            return col
    return None


def _encode_binary(value: object) -> float:
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float, np.integer, np.floating)):
        return 1.0 if float(value) >= 0.5 else 0.0
    v = str(value).strip().lower()
    if v in {"positive", "pos", "1", "1.0", "yes", "true", "amplified", "overexpressed",
             "dead", "deceased", "recurred/progressed"}:
        return 1.0
    if v in {"negative", "neg", "0", "0.0", "no", "false", "not amplified", "normal",
             "alive", "living", "diseasefree", "disease free"}:
        return 0.0
    # Unknown / discrepancy / not reported → NaN (НЕ ошибка, просто пропуск)
    if v in {"[discrepancy]", "[not available]", "[not evaluated]", "[unknown]",
             "not reported", "indeterminate", "equivocal", "nan", ""}:
        return np.nan
    try:
        return 1.0 if float(v) >= 0.5 else 0.0
    except ValueError:
        return np.nan


def _encode_stage_digit(value: object, prefix: str) -> float:
    if pd.isna(value):
        return np.nan
    v = str(value).upper().replace(" ", "")
    if not v.startswith(prefix.upper()):
        return np.nan
    digits = "".join(ch for ch in v if ch.isdigit())
    if not digits:
        return np.nan
    return float(int(digits[0]))


def _prepare_clinical(df: pd.DataFrame) -> pd.DataFrame:
    """Гармонизируем алиасы, кодируем бинарные/стадийные поля."""
    renamed: dict[str, str] = {}
    for canonical, aliases in _CLIN_ALIASES.items():
        hit = _first_present(df, aliases)
        if hit is not None and hit not in renamed.values():
            renamed[hit] = canonical

    out = df.rename(columns=renamed).copy()
    out = out.loc[:, ~out.columns.duplicated(keep="first")]

    for col in _CLIN_ALIASES:
        if col not in out.columns:
            out[col] = np.nan

    out["patient_id"] = out["patient_id"].astype(str).map(_normalize_patient_id)

    for c in [
        "subtype_er", "subtype_pr", "subtype_her2", "event_os", "event_dfs", "pCR",
        "targeted_therapy_received", "radiation_received",
        "neoadjuvant_received", "pharmaceutical_additional",
    ]:
        if c in out.columns:
            out[c] = out[c].map(_encode_binary)
    out["stage_T"] = out["stage_T"].map(lambda v: _encode_stage_digit(v, "T"))
    out["stage_N"] = out["stage_N"].map(lambda v: _encode_stage_digit(v, "N"))
    # Общая стадия (I/II/III/IV → 1..4)
    out["stage"] = out[["stage_T", "stage_N"]].max(axis=1)

    for c in ["age", "ki67", "time_os", "time_dfs"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    # Если time_dfs задано в днях (xena), конвертируем в месяцы
    if out["time_dfs"].dropna().median() > 1000:
        out["time_dfs"] = out["time_dfs"] / 30.44

    pam50_map = {
        "BASAL-LIKE": "BASAL",
        "LUMINAL A": "LUMA", "LUMINALA": "LUMA",
        "LUMINAL B": "LUMB", "LUMINALB": "LUMB",
        "HER2-ENRICHED": "HER2",
        "NORMAL-LIKE": "NORMAL",
        "NAN": None,
    }
    pam50_up = out["subtype_pam50"].astype(str).str.upper()
    out["subtype_pam50"] = pam50_up.map(lambda v: pam50_map.get(v, v) if v != "NAN" else None)

    # dedup по patient_id — берём первую встретившуюся запись
    out = out.drop_duplicates(subset="patient_id", keep="first").set_index("patient_id")
    return out


def _load_expression(path: str | Path) -> pd.DataFrame:
    """Читает RNA-seq TPM TSV, возвращает gene × sample в log2(TPM+1)."""
    src = Path(path)
    logger.info("Loading RNA-seq TPM matrix from %s", src)
    df = pd.read_csv(src, sep="\t", low_memory=False)
    if df.empty:
        raise ValueError(f"Empty RNA-seq matrix at {src}")

    gene_col = df.columns[0]
    df = df.set_index(gene_col)

    # если gene-level metadata есть — фильтруем protein_coding
    if "gene_type" in df.columns:
        mask = df["gene_type"].astype(str).str.lower().eq("protein_coding")
        df = df.loc[mask].drop(columns=["gene_type"])

    # числовые значения
    numeric = df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    # TPM → log2(TPM+1). Если уже в log-space (min<0 или max<30) — НЕ делаем повторно
    arr_max = float(numeric.values.max()) if numeric.size else 0.0
    if arr_max > 100:
        numeric = np.log2(numeric + 1.0)

    # нормализуем sample-id: в TCGA TPM матрицах обычно sample_ids в колонках
    numeric.columns = [_normalize_patient_id(c) for c in numeric.columns]
    # коллапс дубликатов (несколько aliquot → медиана)
    numeric = numeric.T.groupby(level=0).median().T

    # финальная чистка индекса генов — убираем NaN, upper-case
    idx = pd.Index(numeric.index.astype(str).str.upper())
    mask = ~idx.duplicated() & (idx != "NAN")
    numeric = numeric.loc[mask]
    numeric.index = idx[mask]

    if not np.isfinite(numeric.to_numpy()).all():
        raise ValueError("Non-finite values in RNA-seq matrix after cleanup.")
    logger.info("RNA-seq loaded: %d genes × %d samples", *numeric.shape)
    return numeric


def load_tcga_brca(
    expression_path: str | Path,
    clinical_path: str | Path,
    *,
    xena_clinical_path: str | Path | None = None,
    survival_path: str | Path | None = None,
    name: str = "tcga_brca",
    treatment_context: str = "mixed_adjuvant",
) -> CohortBundle:
    """Собирает CohortBundle для TCGA-BRCA.

    Parameters
    ----------
    expression_path
        Путь к TPM matrix (genes × samples, tab-separated).
    clinical_path
        GDC clinical TSV.
    xena_clinical_path
        Дополнительный Xena clinicalMatrix (опционально, merge-source для PAM50 и nature2012-полей).
    survival_path
        Xena survival.tsv с OS/DFS (опционально).
    """
    expr = _load_expression(expression_path)
    clinical = _prepare_clinical(pd.read_csv(clinical_path, sep="\t", low_memory=False))

    # Merge с Xena clinical, если дано (обогащает PAM50, ER/PR/HER2 из nature2012)
    if xena_clinical_path is not None and Path(xena_clinical_path).exists():
        xena = _prepare_clinical(pd.read_csv(xena_clinical_path, sep="\t", low_memory=False))
        clinical = clinical.combine_first(xena)
        logger.info("Enriched clinical with Xena: +%d rows / %d total",
                    len(xena), len(clinical))

    # Merge survival (Xena survival.tsv)
    if survival_path is not None and Path(survival_path).exists():
        surv = pd.read_csv(survival_path, sep="\t", low_memory=False)
        sample_col = _first_present(surv, _CLIN_ALIASES["patient_id"])
        if sample_col:
            surv = surv.rename(columns={sample_col: "patient_id"})
            surv["patient_id"] = surv["patient_id"].astype(str).map(_normalize_patient_id)
            surv = surv.drop_duplicates("patient_id").set_index("patient_id")

            surv_renamed = _prepare_clinical(surv.reset_index())
            # накатываем только survival-колонки — не трогаем остальные
            for col in ["time_os", "event_os", "time_dfs", "event_dfs"]:
                if col in surv_renamed.columns:
                    clinical[col] = surv_renamed[col].reindex(clinical.index).combine_first(
                        clinical[col]
                    )

    # Выравниваем по общему sample_id
    common = sorted(set(expr.columns) & set(clinical.index))
    min_overlap = 10  # toy-cohorts и тесты; для real TCGA обычно ~1000
    if len(common) < min_overlap:
        raise ValueError(
            f"Too few overlapping samples ({len(common)} < {min_overlap}) "
            "between expr and clinical. Check patient_id harmonization."
        )
    expr = expr[common]
    clinical = clinical.loc[common]

    # Собираем endpoints и clinical отдельно.
    # Treatment-флаги остаются в clinical — они используются и как
    # ковариаты в Cox, и как "по чему стратифицировать" в discovery.
    endpoint_cols = ["pCR", "time_os", "event_os", "time_dfs", "event_dfs"]
    endpoints = clinical[[c for c in endpoint_cols if c in clinical.columns]].copy()
    clinical_feat = clinical.drop(columns=[c for c in endpoint_cols if c in clinical.columns])

    # Лог сколько пациенток имеет treatment-annotation
    for c in ["targeted_therapy_received", "radiation_received", "neoadjuvant_received"]:
        if c in clinical_feat.columns:
            n_annot = int(clinical_feat[c].notna().sum())
            n_yes = int((clinical_feat[c] == 1).sum())
            logger.info("%s: %d annotated, %d YES (%.1f%%)",
                        c, n_annot, n_yes, 100 * n_yes / max(n_annot, 1))

    bundle = CohortBundle(
        name=name,
        expr=expr,
        clinical=clinical_feat,
        endpoints=endpoints,
        treatment_context=treatment_context,
        platform="rnaseq",
        expr_kind="log2_tpm",
        meta={
            "source": "tcga_gdc_xena",
            "n_samples": len(common),
            "n_genes": expr.shape[0],
            "endpoints_available": list(endpoints.columns),
        },
    )
    bundle.sanity_check()
    return bundle
