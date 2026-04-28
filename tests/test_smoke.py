"""Smoke-тесты скелета пакета."""

from __future__ import annotations

import numpy as np
import pandas as pd

from oncotme.cohorts.base import CohortBundle
from oncotme.features.baselines import TIS_GENES, cyt_rooney, tis_18gene_score
from oncotme.features.signatures import fallback_zscore, load_gmt
from oncotme.models.blocks import ablation_grid, select_block


def _toy_cohort(n_samples: int = 20, n_genes: int = 200) -> CohortBundle:
    rng = np.random.default_rng(42)
    samples = [f"S{i:03d}" for i in range(n_samples)]
    genes = TIS_GENES + ["GZMA", "PRF1", "B2M", "HLA-A", "HLA-B", "HLA-C"]
    genes += [f"G{i:04d}" for i in range(n_genes - len(genes))]
    expr = pd.DataFrame(
        rng.normal(5.0, 1.5, size=(len(genes), n_samples)),
        index=genes,
        columns=samples,
    )
    clinical = pd.DataFrame(
        {
            "age": rng.integers(35, 75, size=n_samples),
            "subtype_er": rng.choice([0, 1], size=n_samples),
        },
        index=samples,
    )
    endpoints = pd.DataFrame(
        {"pCR": rng.choice([0, 1], size=n_samples)},
        index=samples,
    )
    return CohortBundle(
        name="toy",
        expr=expr,
        clinical=clinical,
        endpoints=endpoints,
        treatment_context="neoadjuvant_chemo",
        platform="rnaseq",
        expr_kind="tpm",
    )


def test_cohort_bundle_aligns_and_validates():
    c = _toy_cohort()
    c.sanity_check()
    aligned = c.aligned()
    assert aligned.n_samples == c.n_samples
    assert aligned.n_genes == c.n_genes


def test_gmt_loader_reads_real_file():
    gmt = load_gmt("signatures/tme_signatures.gmt")
    # минимум — наши ключевые сигнатуры присутствуют
    assert "CD8_T_cell" in gmt
    assert "HLA_I_score" in gmt
    assert "APM_score" in gmt
    # у каждой хоть один ген
    assert all(len(v) > 0 for v in gmt.values())


def test_fallback_zscore_shape():
    c = _toy_cohort()
    gmt = load_gmt("signatures/tme_signatures.gmt")
    scores = fallback_zscore(c.expr, {k: gmt[k] for k in ["CD8_T_cell", "HLA_I_score"]})
    assert scores.shape == (c.n_samples, 2)
    assert set(scores.columns) == {"CD8_T_cell", "HLA_I_score"}


def test_tis_and_cyt():
    c = _toy_cohort()
    tis = tis_18gene_score(c.expr)
    cyt = cyt_rooney(c.expr)
    assert len(tis) == c.n_samples
    assert len(cyt) == c.n_samples
    assert not tis.isna().all()
    assert not cyt.isna().all()


def test_block_ablation_grid():
    # создаём синтетический feature-frame со всеми префиксами
    X = pd.DataFrame(
        np.random.randn(10, 6),
        columns=[
            "sig__CD8_T_cell",
            "sig__TGFb",
            "sig__HLA_I_score",
            "gene__B2M",
            "baseline__TIS",
            "clin__age",
        ],
    )
    grid = ablation_grid(X)
    assert set(grid.keys()) >= {
        "clinical_only", "immune_only", "stromal_only", "apm_only",
        "tme_full", "tme_plus_clinical", "full",
    }
    assert grid["full"].shape == X.shape
    assert "clin__age" in grid["clinical_only"].columns
    assert "sig__CD8_T_cell" in grid["immune_only"].columns
    assert "sig__HLA_I_score" in select_block(X, "apm").columns
