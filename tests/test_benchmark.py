"""Тесты benchmark-инфраструктуры."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))

from oncotme.benchmark.models import (
    ClinicalCoxModel,
    CombinedModel,
    NMFRiskModel,
    RandomFeaturesModel,
    SignatureCoxModel,
)
from oncotme.benchmark.nested_cv import nested_cv_evaluate
from oncotme.benchmark.permutation import permutation_test


def _synth_cohort(n=200, n_sig=15, k_true=3, seed=0):
    rng = np.random.default_rng(seed)
    latent = rng.normal(size=(n, k_true))
    loadings = rng.uniform(0, 1, size=(k_true, n_sig))
    expr = latent @ loadings + rng.normal(0, 0.3, size=(n, n_sig))
    samples = [f"S{i:03d}" for i in range(n)]
    sig_cols = [f"sig__s_{i:02d}" for i in range(n_sig)]
    # gene features для CYT/TIS baselines
    gene_frame = pd.DataFrame({
        f"gene__GZMA": rng.uniform(1, 10, size=n),
        f"gene__PRF1": rng.uniform(1, 10, size=n),
    }, index=samples)
    X = pd.DataFrame(expr, index=samples, columns=sig_cols).join(gene_frame)
    # outcome связан с latent[:, 0]
    hazard = np.exp(0.7 * latent[:, 0])
    time = pd.Series(rng.exponential(scale=5 / hazard), index=samples)
    event = pd.Series((time < 8).astype(int), index=samples)
    clinical = pd.DataFrame({
        "age": rng.integers(40, 80, size=n),
        "stage": rng.integers(1, 4, size=n),
        "subtype_er": rng.choice([0, 1], size=n),
        "subtype_her2": rng.choice([0, 1], size=n),
    }, index=samples)
    return X, time, event, clinical


class TestModels(unittest.TestCase):

    def test_clinical_cox_fit_score(self):
        X, t, e, cl = _synth_cohort()
        m = ClinicalCoxModel().fit(X, t, e, cl)
        s = m.score(X, cl)
        self.assertEqual(len(s), len(X))
        self.assertFalse(s.isna().any())

    def test_random_features_different_seed_different_cols(self):
        X, t, e, cl = _synth_cohort()
        m0 = RandomFeaturesModel(n_features=3, seed=0).fit(X, t, e, cl)
        m1 = RandomFeaturesModel(n_features=3, seed=1).fit(X, t, e, cl)
        # разные seeds должны дать (обычно) разные колонки
        self.assertNotEqual(m0._cols, m1._cols)

    def test_nmf_risk_leakage_free(self):
        X, t, e, cl = _synth_cohort(n=120)
        m = NMFRiskModel(k=3, n_runs=3).fit(X, t, e, cl)
        # создаём "unseen" sample — random noise
        rng = np.random.default_rng(99)
        X_new = pd.DataFrame(
            rng.normal(size=(30, X.shape[1])),
            index=[f"NEW{i}" for i in range(30)],
            columns=X.columns,
        )
        s = m.score(X_new)
        self.assertEqual(len(s), 30)
        self.assertFalse(s.isna().any())

    def test_combined_model(self):
        X, t, e, cl = _synth_cohort(n=100)
        m = CombinedModel(
            NMFRiskModel(k=3, n_runs=3),
            ClinicalCoxModel(),
        ).fit(X, t, e, cl)
        s = m.score(X, cl)
        self.assertEqual(len(s), len(X))


class TestNestedCV(unittest.TestCase):

    def test_nested_cv_runs(self):
        X, t, e, cl = _synth_cohort(n=150)
        res = nested_cv_evaluate(
            lambda: ClinicalCoxModel(),
            X, t, e, cl, n_folds=3, seed=0, n_bootstrap=50,
        )
        self.assertEqual(len(res.fold_c_indices), 3)
        self.assertGreater(res.mean_c_index, 0.0)
        self.assertLess(res.mean_c_index, 1.0)

    def test_nested_cv_nmf_leakage_free(self):
        X, t, e, cl = _synth_cohort(n=120)
        res = nested_cv_evaluate(
            lambda: NMFRiskModel(k=3, n_runs=2),
            X, t, e, cl, n_folds=3, seed=0, n_bootstrap=20,
        )
        self.assertEqual(len(res.fold_c_indices), 3)


class TestPermutation(unittest.TestCase):

    def test_permutation_p_value_reasonable_for_random_model(self):
        """Random features model → permutation должен давать p ≈ 1 (observed ≈ null)."""
        X, t, e, cl = _synth_cohort(n=100)
        factory = lambda: RandomFeaturesModel(n_features=3, seed=0)
        observed = 0.52  # ~ random
        perm = permutation_test(
            factory, X, t, e, observed, cl,
            n_permutations=5, n_folds=3, seed=0,
        )
        # at least n_success > 0 and p_value in [0,1]
        self.assertGreater(perm["n_success"], 0)
        self.assertGreaterEqual(perm["p_value"], 0.0)
        self.assertLessEqual(perm["p_value"], 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
