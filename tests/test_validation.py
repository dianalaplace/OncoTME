"""Тесты самопроверочного фреймворка.

Цель — убедиться, что проверки реально ловят нарушения, а не всегда PASS.
Каждая проверка прогоняется на двух входах:
(a) валидном  → должна PASS
(b) испорченном → должна FAIL / WARN

Запускается через unittest (pytest опционален)::

    python -m unittest tests.test_validation -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))

from oncotme.cohorts.base import CohortBundle
from oncotme.features.signatures import fallback_zscore, load_gmt
from oncotme.validation import run_suite
from oncotme.validation.checks import (
    assert_balanced_binary,
    assert_columns_subset,
    assert_expected_correlation_sign,
    assert_in_range,
    assert_index_aligned,
    assert_index_unique,
    assert_no_inf,
    assert_no_nan,
    assert_shape,
)
from oncotme.validation.cohort_checks import check_cohort_bundle, check_cohorts_disjoint
from oncotme.validation.feature_checks import check_feature_frame, check_signature_scores
from oncotme.validation.model_checks import (
    check_calibration,
    check_cv_no_leakage,
    check_permutation_sanity,
    check_seed_stability,
)
from oncotme.validation.result import CheckStatus, ValidationReport


# ---------- атомарные проверки --------------------------------------------

class TestAtomicChecks(unittest.TestCase):

    def test_no_nan_passes_and_fails(self):
        good = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
        bad = pd.DataFrame({"a": [1.0, np.nan, 3.0]})
        self.assertEqual(assert_no_nan(good, "x").status, CheckStatus.PASS)
        self.assertEqual(assert_no_nan(bad, "x").status, CheckStatus.FAIL)

    def test_no_inf(self):
        bad = pd.Series([1.0, np.inf, 3.0])
        self.assertEqual(assert_no_inf(bad, "x").status, CheckStatus.FAIL)

    def test_shape_mismatch(self):
        df = pd.DataFrame(np.zeros((5, 3)))
        self.assertEqual(assert_shape(df, (5, 3), "x").status, CheckStatus.PASS)
        self.assertEqual(assert_shape(df, (5, 4), "x").status, CheckStatus.FAIL)

    def test_in_range(self):
        s = pd.Series([0.1, 0.5, 0.9])
        self.assertEqual(assert_in_range(s, 0, 1, "x").status, CheckStatus.PASS)
        self.assertEqual(assert_in_range(pd.Series([0.5, 1.2]), 0, 1, "x").status, CheckStatus.FAIL)

    def test_index_uniqueness(self):
        good = pd.DataFrame(index=["a", "b", "c"])
        bad = pd.DataFrame(index=["a", "b", "a"])
        self.assertEqual(assert_index_unique(good, "x").status, CheckStatus.PASS)
        self.assertEqual(assert_index_unique(bad, "x").status, CheckStatus.FAIL)

    def test_columns_subset(self):
        df = pd.DataFrame(columns=["a", "b", "c"])
        self.assertEqual(assert_columns_subset(df, ["a", "b"], "x").status, CheckStatus.PASS)
        self.assertEqual(assert_columns_subset(df, ["a", "z"], "x").status, CheckStatus.FAIL)

    def test_index_aligned(self):
        a = pd.DataFrame(index=["s1", "s2", "s3"])
        b = pd.DataFrame(index=["s1", "s2", "s3"])
        bad = pd.DataFrame(index=["x1", "x2"])
        self.assertEqual(assert_index_aligned(a, b, "x").status, CheckStatus.PASS)
        self.assertEqual(
            assert_index_aligned(a, bad, "x", min_overlap=2).status, CheckStatus.FAIL
        )

    def test_balanced_binary(self):
        balanced = pd.Series([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        imbalanced = pd.Series([0] * 99 + [1])
        non_binary = pd.Series([0, 1, 2])
        self.assertEqual(assert_balanced_binary(balanced, "x").status, CheckStatus.PASS)
        self.assertEqual(assert_balanced_binary(imbalanced, "x").status, CheckStatus.WARN)
        self.assertEqual(assert_balanced_binary(non_binary, "x").status, CheckStatus.FAIL)

    def test_correlation_sign(self):
        rng = np.random.default_rng(0)
        x = pd.Series(rng.normal(size=100), name="x")
        y_pos = x + pd.Series(rng.normal(scale=0.3, size=100))
        y_neg = -x + pd.Series(rng.normal(scale=0.3, size=100))
        # matching sign
        self.assertEqual(
            assert_expected_correlation_sign(x, y_pos, +1, "x", min_abs_corr=0.5).status,
            CheckStatus.PASS,
        )
        # opposite sign → FAIL
        self.assertEqual(
            assert_expected_correlation_sign(x, y_neg, +1, "x", min_abs_corr=0.5).status,
            CheckStatus.FAIL,
        )


# ---------- cohort-level --------------------------------------------------

def _toy_cohort(
    name: str = "toy",
    n_samples: int = 50,
    bad_gene_index: bool = False,
    wrong_units: bool = False,
    bad_pcr_encoding: bool = False,
    sample_prefix: str = "S",
    seed: int = 42,
) -> CohortBundle:
    """Синтетическая когорта с вшитой биологией: скрытый "иммунный" фактор
    ап-регулирует все иммунные и APM-сигнатуры согласованно — чтобы
    корреляционные sanity-проверки проходили на чистых данных."""
    rng = np.random.default_rng(seed)
    gmt = load_gmt(str(_REPO / "signatures" / "tme_signatures.gmt"))
    gene_list = sorted({g for gs in gmt.values() for g in gs})
    if bad_gene_index:
        gene_list = [f"ENSG{i:08d}" for i in range(len(gene_list))]
    samples = [f"{sample_prefix}{i:03d}" for i in range(n_samples)]
    base = rng.normal(5.0, 1.5, size=(len(gene_list), n_samples))

    # вшиваем latent "immune/hot" ось — ап-регулирует иммунные + APM сигнатуры
    if not bad_gene_index:
        latent = rng.normal(0, 1, size=n_samples)
        idx_map = {g: i for i, g in enumerate(gene_list)}
        immune_up = set()
        for s in ["CD8_T_cell", "CD4_T_cell", "NK_cell", "B_cell", "Dendritic",
                  "IFN_gamma", "HLA_I_score", "HLA_II_score", "APM_score",
                  "TLS_signature", "checkpoint_score"]:
            immune_up |= set(gmt.get(s, []))
        stromal_down = set()
        for s in ["CAF_proxy", "TGFb", "hypoxia", "M2_polarization"]:
            stromal_down |= set(gmt.get(s, []))
        for g in immune_up:
            if g in idx_map:
                base[idx_map[g]] += 1.2 * latent
        for g in stromal_down:
            if g in idx_map:
                base[idx_map[g]] -= 0.6 * latent

    if wrong_units:
        base = np.exp2(base) * 1000  # блестящие огромные counts
    expr = pd.DataFrame(base, index=gene_list, columns=samples)
    clinical = pd.DataFrame(
        {
            "age": rng.integers(35, 75, size=n_samples),
            "subtype_er": rng.choice([0, 1], size=n_samples),
            "subtype_her2": rng.choice([0, 1], size=n_samples),
        },
        index=samples,
    )
    pcr_vals = rng.choice([0, 1], size=n_samples)
    if bad_pcr_encoding:
        pcr_vals = pcr_vals.astype(object)
        pcr_vals[:5] = "yes"  # грязное кодирование
    endpoints = pd.DataFrame({"pCR": pcr_vals}, index=samples)
    return CohortBundle(
        name=name,
        expr=expr,
        clinical=clinical,
        endpoints=endpoints,
        treatment_context="neoadjuvant_chemo",
        platform="rnaseq",
        expr_kind="log2_tpm",
    )


class TestCohortChecks(unittest.TestCase):

    def test_valid_cohort_passes(self):
        c = _toy_cohort()
        results = check_cohort_bundle(c)
        fails = [r for r in results if r.status == CheckStatus.FAIL]
        self.assertEqual(fails, [], msg=f"Unexpected failures: {[r.name for r in fails]}")

    def test_ensembl_index_fails(self):
        c = _toy_cohort(bad_gene_index=True)
        results = check_cohort_bundle(c)
        names = {r.name for r in results if r.status == CheckStatus.FAIL}
        self.assertIn(f"cohort.{c.name}.gene_symbols_hgnc_like", names)

    def test_wrong_units_fails(self):
        c = _toy_cohort(wrong_units=True)
        results = check_cohort_bundle(c)
        names = {r.name for r in results if r.status == CheckStatus.FAIL}
        self.assertIn(f"cohort.{c.name}.expr_unit_plausible", names)

    def test_bad_pcr_encoding_fails(self):
        c = _toy_cohort(bad_pcr_encoding=True)
        results = check_cohort_bundle(c)
        names = {r.name for r in results if r.status == CheckStatus.FAIL}
        self.assertTrue(
            any(n.endswith(".endpoints_encoding") or n.endswith(".pcr_class_balance")
                for n in names),
            msg=f"Expected pCR encoding failure, got: {sorted(names)}",
        )

    def test_disjoint_sample_ids(self):
        a = _toy_cohort("A", sample_prefix="X")
        b = _toy_cohort("B", sample_prefix="X")  # одинаковые префиксы → пересечение
        results = check_cohorts_disjoint([a, b])
        self.assertTrue(
            any(r.status == CheckStatus.FAIL for r in results),
            msg="Expected overlap detection to fail.",
        )

    def test_disjoint_sample_ids_ok(self):
        a = _toy_cohort("A", sample_prefix="A_")
        b = _toy_cohort("B", sample_prefix="B_")
        results = check_cohorts_disjoint([a, b])
        self.assertTrue(all(r.status == CheckStatus.PASS for r in results))


# ---------- feature-level ------------------------------------------------

class TestFeatureChecks(unittest.TestCase):

    def _features_from_cohort(self, c: CohortBundle) -> pd.DataFrame:
        gmt = load_gmt(str(_REPO / "signatures" / "tme_signatures.gmt"))
        scores = fallback_zscore(c.expr, gmt)
        scores.columns = [f"sig__{col}" for col in scores.columns]
        scores["gene__B2M"] = c.expr.loc["B2M"].reindex(scores.index).values if "B2M" in c.expr.index else 0.0
        scores["clin__age"] = c.clinical["age"].reindex(scores.index).values
        return scores

    def test_valid_features_pass(self):
        c = _toy_cohort()
        X = self._features_from_cohort(c)
        results = check_feature_frame(X)
        fails = [r for r in results if r.status == CheckStatus.FAIL]
        self.assertEqual(fails, [], msg=f"Unexpected feature failures: {[r.name for r in fails]}")

    def test_all_nan_column_fails(self):
        c = _toy_cohort()
        X = self._features_from_cohort(c)
        X["sig__broken"] = np.nan
        results = check_feature_frame(X)
        names = {r.name for r in results if r.status == CheckStatus.FAIL}
        self.assertIn("features.no_all_nan_columns", names)

    def test_duplicate_column_warns(self):
        c = _toy_cohort()
        X = self._features_from_cohort(c)
        X["sig__duplicate"] = X["sig__CD8_T_cell"]
        results = check_feature_frame(X)
        names = {r.name for r in results if r.status == CheckStatus.WARN}
        self.assertIn("features.no_near_duplicate_columns", names)


# ---------- model-level --------------------------------------------------

class TestModelChecks(unittest.TestCase):

    def test_cv_leakage_detects_overlap(self):
        # sample_id с пересечением в train и test
        splits = [(np.array([0, 1, 2]), np.array([2, 3, 4]))]
        sample_ids = [f"s{i}" for i in range(5)]
        result = check_cv_no_leakage(splits, sample_ids)
        self.assertEqual(result.status, CheckStatus.FAIL)

    def test_cv_leakage_detects_group_overlap(self):
        # разные sample_id, но один patient в train и test
        splits = [(np.array([0, 1]), np.array([2, 3]))]
        sample_ids = ["s0", "s1", "s2", "s3"]
        groups = ["pA", "pB", "pA", "pC"]  # pA в обоих
        result = check_cv_no_leakage(splits, sample_ids, groups=groups, group_name="patient")
        self.assertEqual(result.status, CheckStatus.FAIL)

    def test_cv_no_leakage_ok(self):
        splits = [(np.array([0, 1]), np.array([2, 3]))]
        sample_ids = ["s0", "s1", "s2", "s3"]
        groups = ["pA", "pB", "pC", "pD"]
        result = check_cv_no_leakage(splits, sample_ids, groups=groups, group_name="patient")
        self.assertEqual(result.status, CheckStatus.PASS)

    def test_permutation_sanity_catches_cheating_model(self):
        # "модель", которая знает y — должна получать AUC=1 даже на permuted y → FAIL
        rng = np.random.default_rng(0)
        X = rng.normal(size=(100, 3))
        y = rng.integers(0, 2, size=100)

        def cheating(_X, y_perm):
            return y_perm.astype(float)

        result = check_permutation_sanity(cheating, X, y, n_permutations=3)
        self.assertEqual(result.status, CheckStatus.FAIL)

    def test_permutation_sanity_passes_for_random_model(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(200, 3))
        y = rng.integers(0, 2, size=200)

        def random_model(_X, _y):
            return rng.uniform(size=len(_y))

        result = check_permutation_sanity(random_model, X, y, n_permutations=5, tol=0.15)
        self.assertIn(result.status, (CheckStatus.PASS, CheckStatus.WARN))

    def test_seed_stability_detects_high_variance(self):
        rng = np.random.default_rng(0)

        def unstable(seed):
            return rng.uniform(0.5, 0.9)  # разброс 0.4

        r = check_seed_stability(unstable, seeds=(0, 1, 2, 3, 4), max_std=0.01)
        self.assertEqual(r.status, CheckStatus.FAIL)

    def test_seed_stability_passes_for_stable(self):
        def stable(seed):
            return 0.75 + 0.001 * seed

        r = check_seed_stability(stable, seeds=(0, 1, 2), max_std=0.03)
        self.assertEqual(r.status, CheckStatus.PASS)

    def test_calibration_perfect(self):
        rng = np.random.default_rng(0)
        y_prob = rng.uniform(size=2000)
        y_true = (rng.uniform(size=2000) < y_prob).astype(int)
        r = check_calibration(y_true, y_prob)
        self.assertIn(r.status, (CheckStatus.PASS, CheckStatus.WARN))  # реально близко к passing

    def test_calibration_miscalibrated(self):
        # систематическое занижение вероятности
        rng = np.random.default_rng(0)
        y_true = rng.integers(0, 2, size=500)
        y_prob = np.clip(y_true * 0.3 + 0.05, 0, 1)  # предсказания сильно ниже истинных
        r = check_calibration(y_true, y_prob, max_ece=0.05)
        self.assertEqual(r.status, CheckStatus.WARN)


# ---------- suite end-to-end --------------------------------------------

class TestSuite(unittest.TestCase):

    def test_suite_runs_on_synthetic(self):
        # симулируем мини-сценарий: две когорты + feature-frames
        a = _toy_cohort("tcga_mini", n_samples=60, sample_prefix="TCGA_")
        b = _toy_cohort("meta_mini", n_samples=60, sample_prefix="META_")
        gmt = load_gmt(str(_REPO / "signatures" / "tme_signatures.gmt"))
        fa = fallback_zscore(a.expr, gmt)
        fa.columns = [f"sig__{c}" for c in fa.columns]
        fb = fallback_zscore(b.expr, gmt)
        fb.columns = [f"sig__{c}" for c in fb.columns]
        report: ValidationReport = run_suite(
            cohorts=[a, b],
            feature_frames={"tcga_mini": fa, "meta_mini": fb},
        )
        # гарантируем, что suite что-то проверил и не упал
        self.assertGreater(len(report.results), 10)
        # на чистых синтетических данных FAIL быть не должно
        fails = report.failures()
        self.assertEqual(fails, [], msg=f"Unexpected failures: {[r.name for r in fails]}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
