"""Эталонные сигнатуры для сравнения с TME-моделью.

Реализации:
- ``tis_18gene_score`` — Ayers 2017 Tumor Inflammation Signature (18 genes)
- ``cyt_rooney`` — Rooney 2015 cytolytic activity = sqrt(GZMA * PRF1)
- ``estimate_scores`` — Yoshihara 2013 Immune + Stromal ESTIMATE (нужен внешний пакет)
- ``oncotype_proxy`` — 21-gene RS суррогат (HR+)
"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd


TIS_GENES: List[str] = [
    "CCL5", "CD27", "CD274", "CD276", "CD8A", "CMKLR1", "CXCL9", "CXCR6",
    "HLA-DQA1", "HLA-DRB1", "HLA-E", "IDO1", "LAG3", "NKG7", "PDCD1LG2",
    "PSMB10", "STAT1", "TIGIT",
]


def tis_18gene_score(expr: pd.DataFrame) -> pd.Series:
    """TIS = mean of z-scored expression of 18 genes."""
    present = [g for g in TIS_GENES if g in expr.index]
    if not present:
        return pd.Series(np.nan, index=expr.columns, name="TIS_18gene")
    sub = expr.loc[present]
    z = sub.sub(sub.mean(axis=1), axis=0).div(sub.std(axis=1).replace(0, 1), axis=0)
    return z.mean(axis=0).rename("TIS_18gene")


def cyt_rooney(expr: pd.DataFrame) -> pd.Series:
    """CYT = geometric mean of GZMA and PRF1 expression."""
    required = ["GZMA", "PRF1"]
    if not all(g in expr.index for g in required):
        return pd.Series(np.nan, index=expr.columns, name="CYT")
    return np.sqrt(expr.loc["GZMA"] * expr.loc["PRF1"]).rename("CYT")


def estimate_scores(_expr: pd.DataFrame) -> pd.DataFrame:
    """Yoshihara 2013 ESTIMATE (Immune + Stromal).

    TODO: wrapper — либо портировать ssGSEA на ESTIMATE gene sets,
    либо звать R-пакет ``estimate`` через rpy2 (optional dependency).
    """
    raise NotImplementedError("ESTIMATE scoring: implement after ssGSEA wrapper is ready.")


def oncotype_proxy(_expr: pd.DataFrame) -> pd.Series:
    """Proxy для 21-gene Oncotype DX recurrence score (HR+).

    TODO: implement with published coefficients (Paik 2004).
    """
    raise NotImplementedError("Oncotype proxy: not implemented yet.")
