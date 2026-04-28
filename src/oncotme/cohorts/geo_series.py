"""Generic GEO Series Matrix loader.

Parses GEO Series Matrix (``..._series_matrix.txt[.gz]``) files:
- Sample IDs from ``!Sample_geo_accession``
- Clinical characteristics from ``!Sample_characteristics_ch*`` (key:value format)
- Expression matrix from ``!series_matrix_table_begin`` … ``_end``
- Probe → gene mapping from platform annotation (GPL96, GPL570, GPL13534)

Output: raw probe-level matrix (probes × samples) + clinical DataFrame.
Downstream (cohort-specific modules) применяют probe → gene aggregation
и harmonization в ``CohortBundle``.
"""

from __future__ import annotations

import gzip
import logging
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _open(path: Path):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", encoding="latin-1", errors="replace")
    return open(path, "r", encoding="latin-1", errors="replace")


def _parse_characteristics(value: str) -> tuple[Optional[str], Optional[str]]:
    """``"age_years: 32.2"`` → ("age_years", "32.2")."""
    v = value.strip().strip('"')
    if ":" not in v:
        return None, v if v else None
    key, val = v.split(":", 1)
    return key.strip(), val.strip()


def parse_series_matrix(path: str | Path) -> dict:
    """Parse GEO Series Matrix file.

    Returns
    -------
    dict:
        - ``title``: Series title
        - ``platform_id``: e.g. GPL96
        - ``sample_ids``: list of GSM accessions
        - ``characteristics``: DataFrame sample_id × characteristic_key
        - ``expr``: DataFrame probe_id × sample_id (numeric)
    """
    path = Path(path)
    logger.info("Parsing GEO series matrix: %s", path.name)

    title = None
    platform_id = None
    sample_ids: list[str] = []
    # characteristics: list of dicts, один dict на sample_characteristics_ch line
    char_rows: list[list[tuple[Optional[str], Optional[str]]]] = []

    # First pass: metadata
    expr_lines: list[str] = []
    in_table = False
    with _open(path) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith("!series_matrix_table_begin"):
                in_table = True
                continue
            if line.startswith("!series_matrix_table_end"):
                in_table = False
                continue
            if in_table:
                expr_lines.append(line)
                continue
            # metadata
            if line.startswith("!Series_title"):
                parts = line.split("\t", 1)
                if len(parts) == 2:
                    title = parts[1].strip().strip('"')
            elif line.startswith("!Series_platform_id"):
                parts = line.split("\t", 1)
                if len(parts) == 2:
                    platform_id = parts[1].strip().strip('"')
            elif line.startswith("!Sample_geo_accession"):
                parts = line.split("\t")[1:]
                sample_ids = [p.strip().strip('"') for p in parts]
            elif line.startswith("!Sample_characteristics_ch"):
                parts = line.split("\t")[1:]
                row = [_parse_characteristics(p) for p in parts]
                char_rows.append(row)

    # Normalize characteristics → DataFrame sample × key
    if char_rows and sample_ids:
        n = len(sample_ids)
        all_rows: dict[int, dict[str, str]] = {i: {} for i in range(n)}
        for char_row in char_rows:
            # В пределах одной характеристики у разных samples может быть РАЗНЫЙ key
            # (редкий edge-case; чаще один key на всю строку).
            for i, (k, v) in enumerate(char_row[:n]):
                if k is None:
                    continue
                # если уже есть этот key — делаем suffix
                base_k = k
                suffix = 1
                while base_k in all_rows[i] and all_rows[i][base_k] != v:
                    suffix += 1
                    base_k = f"{k}__{suffix}"
                all_rows[i][base_k] = v
        characteristics = pd.DataFrame.from_dict(all_rows, orient="index")
        characteristics.index = sample_ids
    else:
        characteristics = pd.DataFrame(index=sample_ids)

    # Expression
    if not expr_lines:
        raise ValueError(f"No series_matrix_table found in {path}")
    header = expr_lines[0].split("\t")
    header = [h.strip().strip('"') for h in header]
    data_rows: list[list[str]] = []
    for line in expr_lines[1:]:
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        parts = [p.strip().strip('"') for p in parts]
        data_rows.append(parts)

    if not data_rows:
        raise ValueError(f"No expression rows in {path}")
    expr = pd.DataFrame(data_rows, columns=header)
    expr = expr.set_index(header[0])
    # convert to numeric
    expr = expr.apply(pd.to_numeric, errors="coerce")

    # align columns to sample_ids
    keep_cols = [c for c in expr.columns if c in sample_ids]
    expr = expr[keep_cols]

    return {
        "title": title,
        "platform_id": platform_id,
        "sample_ids": sample_ids,
        "characteristics": characteristics,
        "expr": expr,
    }


# -------- probe → gene mapping --------------------------------------------


def load_gpl96_annotation(path: str | Path) -> pd.Series:
    """GPL96 (Affymetrix U133A) probe → HGNC symbol mapping.

    GPL96 annotation has tab-separated columns; needs "ID" and "Gene Symbol".
    Ambiguous gene symbols (``A /// B``) → split to first non-empty.
    """
    path = Path(path)
    # автоматически находим header-строку (начинается с ID\t…)
    with _open(path) as fh:
        rows = []
        header = None
        for line in fh:
            if line.startswith("#") or line.startswith("!"):
                continue
            parts = line.rstrip("\n").split("\t")
            if header is None:
                if parts and parts[0].upper() == "ID":
                    header = [p.strip() for p in parts]
                continue
            if not parts or parts[0] == "":
                continue
            rows.append(parts)
    if not header:
        raise ValueError(f"Could not parse {path} header.")
    df = pd.DataFrame(rows, columns=header).set_index("ID")
    # ищем gene symbol column
    candidates = [c for c in df.columns if "gene symbol" in c.lower() or c.lower() == "gene_symbol"]
    if not candidates:
        # fallback — ищем что-то типа "Symbol"
        candidates = [c for c in df.columns if "symbol" in c.lower()]
    if not candidates:
        raise ValueError(f"No 'Gene Symbol' column in GPL96 annotation; got {df.columns.tolist()[:10]}")
    col = candidates[0]
    symbols = df[col].astype(str).str.split("///").str[0].str.strip().str.upper()
    symbols = symbols.replace({"": np.nan, "NAN": np.nan, "---": np.nan})
    return symbols


def aggregate_probes_to_genes(
    expr_probes: pd.DataFrame,
    probe_to_gene: pd.Series,
    *,
    method: str = "max_variance",
) -> pd.DataFrame:
    """Aggregate probe-level matrix → gene-level.

    method:
    - ``max_variance`` — для каждого гена оставляем probe с максимальной variance
      (стандарт для Affymetrix benchmarks)
    - ``mean`` — усреднение probes на ген
    """
    # map probes → genes
    mapped = expr_probes.copy()
    mapped["_gene"] = probe_to_gene.reindex(mapped.index).values
    mapped = mapped.dropna(subset=["_gene"])

    if method == "max_variance":
        variance = expr_probes.loc[mapped.index].var(axis=1)
        mapped["_var"] = variance.values
        # per gene keep max-variance probe
        mapped = mapped.sort_values("_var", ascending=False)
        mapped = mapped.drop_duplicates(subset="_gene", keep="first")
        genes = mapped.set_index("_gene").drop(columns=["_var"])
        return genes.sort_index()
    if method == "mean":
        return mapped.groupby("_gene").mean(numeric_only=True).sort_index()
    raise ValueError(f"Unknown method {method!r}")
