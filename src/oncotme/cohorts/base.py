"""Базовая структура для когорт OncoTME.

Каждый loader (tcga.py, metabric.py, ispy2.py, …) возвращает CohortBundle —
унифицированный контейнер, который далее использует весь пайплайн
(preprocess → features → models). Это единственный контракт между
cohort-specific кодом и остальной системой.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd


@dataclass
class CohortBundle:
    """Унифицированный контейнер когорты.

    Attributes
    ----------
    name
        Короткий идентификатор (``tcga_brca``, ``metabric``, ``gepar_nuevo``…).
    expr
        Матрица экспрессии, gene × sample. Индекс — HGNC symbols, колонки —
        ``sample_id``. Значения: log2(TPM+1) для RNA-seq или
        log2-intensity для microarray (решение — за loader'ом).
    clinical
        DataFrame со строками на ``sample_id``. Ожидаемые колонки (могут
        отсутствовать — пайплайн ориентируется через config):
        ``age``, ``stage``, ``grade``, ``subtype_er``, ``subtype_pr``,
        ``subtype_her2``, ``intrinsic_pam50``, ``ki67``.
    endpoints
        DataFrame на ``sample_id``. Колонки зависят от когорты:
        response — ``pCR`` (0/1), ``ORR``; survival — ``time_rfs``,
        ``event_rfs``, ``time_os``, ``event_os`` и т.д. Имена событий
        берутся из ``config/cohorts.yaml:cohorts.*.endpoints``.
    treatment_context
        Строка-ярлык: ``neoadjuvant_chemo``, ``neoadjuvant_her2``,
        ``neoadjuvant_chemo_plus_ici``, ``endocrine``, ``mixed_adjuvant`` и т.п.
        Для multi-arm когорт (I-SPY2) — per-sample колонка ``arm`` в
        ``clinical``, а ``treatment_context`` = ``neoadjuvant_multi_arm``.
    platform
        ``rnaseq`` | ``microarray_illumina_ht12`` | ``microarray_affy_u133`` | ``scrna``.
    expr_kind
        ``tpm`` | ``log2_intensity`` | ``counts`` — для выбора правильной нормализации.
    meta
        Свободный словарь для loader-specific заметок.
    """

    name: str
    expr: pd.DataFrame
    clinical: pd.DataFrame
    endpoints: pd.DataFrame
    treatment_context: str
    platform: str
    expr_kind: str
    meta: dict = field(default_factory=dict)

    def sanity_check(self) -> None:
        """Базовая валидация инвариантов, на которые полагается пайплайн."""
        # общий индекс sample_id
        expr_samples = set(self.expr.columns)
        clin_samples = set(self.clinical.index)
        endpt_samples = set(self.endpoints.index)
        common = expr_samples & clin_samples & endpt_samples
        assert common, f"No overlap of sample_id across expr/clinical/endpoints in {self.name}"

        # gene symbols не должны быть пустыми
        assert self.expr.index.notna().all(), f"Gene symbols contain NaN in {self.name}"
        assert (self.expr.index.str.len() > 0).all(), f"Empty gene symbols in {self.name}"

    def aligned(self) -> "CohortBundle":
        """Выравнивает expr/clinical/endpoints по общему set sample_id."""
        common = sorted(
            set(self.expr.columns) & set(self.clinical.index) & set(self.endpoints.index)
        )
        return CohortBundle(
            name=self.name,
            expr=self.expr[common],
            clinical=self.clinical.loc[common],
            endpoints=self.endpoints.loc[common],
            treatment_context=self.treatment_context,
            platform=self.platform,
            expr_kind=self.expr_kind,
            meta={**self.meta, "n_samples": len(common)},
        )

    @property
    def n_samples(self) -> int:
        return self.expr.shape[1]

    @property
    def n_genes(self) -> int:
        return self.expr.shape[0]

    def __repr__(self) -> str:
        return (
            f"CohortBundle(name={self.name!r}, n_samples={self.n_samples}, "
            f"n_genes={self.n_genes}, platform={self.platform}, "
            f"context={self.treatment_context})"
        )


def load_cohort(name: str, config: Optional[dict] = None) -> CohortBundle:
    """Фабричный диспетчер — вызывает правильный loader по имени когорты.

    Конкретные loader'ы реализуются отдельными модулями (tcga.py, metabric.py,
    и т.д.) и регистрируются здесь. На данный момент — скелет.
    """
    raise NotImplementedError(
        f"Loader for cohort {name!r} is not implemented yet. "
        f"See src/oncotme/cohorts/ for planned loaders."
    )
