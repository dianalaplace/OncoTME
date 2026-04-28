"""ssGSEA-скоринг сигнатур + z-score fallback.

Пара ключевых решений:
- Если gseapy недоступен или падает — используем z-score (per-row standardize,
  усреднение по сигнатуре). Детерминированно и устойчиво на малых N.
- GMT-парсер кладёт имена генов в UPPERCASE, чтобы не зависеть от регистра
  между когортами.
- Публичный API — три функции: ``load_gmt``, ``ssgsea_scores``, ``fallback_zscore``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

try:
    import gseapy as gp

    _GSEAPY_AVAILABLE = True
    _GSEAPY_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # noqa: BLE001 — gseapy опциональная зависимость
    gp = None  # type: ignore[assignment]
    _GSEAPY_AVAILABLE = False
    _GSEAPY_IMPORT_ERROR = exc


logger = logging.getLogger(__name__)


def load_gmt(path: str | Path) -> Dict[str, List[str]]:
    """Парсит GMT-файл → dict signature_name → list[str] of UPPERCASE gene symbols.

    GMT format: ``<name>\\t<description>\\t<gene1>\\t<gene2>\\t...``.
    Пустые строки и строки короче 3 полей пропускаются.
    """
    path = Path(path)
    sigs: Dict[str, List[str]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            name, _desc, *genes = parts
            genes = [g.upper() for g in genes if g]
            if name and genes:
                sigs[name] = genes
    return sigs


def _zscore_rows(expr: pd.DataFrame) -> pd.DataFrame:
    """Per-gene z-score через строки (центрирование по среднему sample-уровня).

    Constant rows (std == 0) превращаются в 0 (чтобы не получить NaN через 0/0).
    """
    mean = expr.mean(axis=1)
    std = expr.std(axis=1, ddof=0).replace(0.0, np.nan)
    z = expr.sub(mean, axis=0).div(std, axis=0)
    return z.fillna(0.0)


def fallback_zscore(
    expr: pd.DataFrame,
    gene_sets: Dict[str, List[str]],
) -> pd.DataFrame:
    """Fallback-реализация сигнатурного скоринга: mean z-score по genes in signature.

    Возвращает DataFrame samples × signatures. Если ни один ген сигнатуры не
    найден в expression index — соответствующая колонка = 0.
    """
    if expr.empty or not gene_sets:
        return pd.DataFrame(index=expr.columns)

    expr_upper = expr.copy()
    expr_upper.index = expr_upper.index.astype(str).str.upper()
    z = _zscore_rows(expr_upper)

    rows: Dict[str, pd.Series] = {}
    for name, genes in gene_sets.items():
        matched = [g for g in genes if g in z.index]
        if matched:
            rows[name] = z.loc[matched].mean(axis=0)
        else:
            logger.warning("Signature %r: no genes matched in expression index.", name)
            rows[name] = pd.Series(0.0, index=expr.columns)
    return pd.DataFrame(rows)


def ssgsea_scores(
    expr: pd.DataFrame,
    gene_sets: Dict[str, List[str]] | None = None,
    gmt_path: str | Path | None = None,
    *,
    sample_norm_method: str = "rank",
) -> pd.DataFrame:
    """Compute ssGSEA scores → DataFrame samples × signatures.

    Если ``gseapy`` недоступен или падает — автоматически переключается на
    ``fallback_zscore``. Это делает пайплайн CI-friendly (тесты не требуют gseapy).

    Parameters
    ----------
    expr
        gene × sample matrix; индекс — gene symbols.
    gene_sets
        dict signature_name → list[gene]. Если None, читается из ``gmt_path``.
    gmt_path
        Путь к GMT-файлу. Используется, если ``gene_sets`` не передан.
    sample_norm_method
        Параметр gseapy ssgsea (``rank`` по умолчанию — устойчиво к платформе).
    """
    if gene_sets is None:
        if gmt_path is None:
            raise ValueError("Provide either gene_sets or gmt_path.")
        gene_sets = load_gmt(gmt_path)

    if not _GSEAPY_AVAILABLE:
        logger.info(
            "gseapy unavailable (%s); using z-score fallback.", _GSEAPY_IMPORT_ERROR
        )
        return fallback_zscore(expr, gene_sets)

    try:
        res = gp.ssgsea(
            data=expr,
            gene_sets=gene_sets,
            sample_norm_method=sample_norm_method,
            outdir=None,
            permutation_num=0,
            no_plot=True,
            min_size=1,
            threads=1,
            verbose=False,
        )
        metric = "ES" if "ES" in res.res2d.columns else "NES"
        scores = res.res2d.pivot(index="Term", columns="Name", values=metric)
        return scores.T
    except Exception as exc:  # noqa: BLE001 — gseapy может падать на краевых случаях
        logger.warning("ssGSEA failed (%s); falling back to z-score.", exc)
        return fallback_zscore(expr, gene_sets)
