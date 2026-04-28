"""Тесты NMF factorization и risk-score модуля."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))

from oncotme.discovery.nmf_factors import (
    name_factors_by_top_signatures,
    nmf_with_stability,
    project_new_samples,
    scan_nmf_k,
    select_k_nmf,
)
from oncotme.discovery.risk_score import (
    decision_curve_analysis,
    fit_composite_risk_score,
    tertile_km,
)
from oncotme.validation.discovery_checks import (
    check_nmf_stability,
    check_risk_score_discriminates,
    check_tertile_separation,
)
from oncotme.validation.result import CheckStatus


def _synth_tme_data(n=200, k_true=3, n_sig=15, seed=0) -> pd.DataFrame:
    """Синтезируем TME-like data с k_true скрытых факторов."""
    rng = np.random.default_rng(seed)
    W_true = rng.uniform(0, 1, size=(n, k_true))
    H_true = rng.uniform(0, 1, size=(k_true, n_sig))
    X = W_true @ H_true + rng.normal(0, 0.1, size=(n, n_sig))
    samples = [f"S{i:03d}" for i in range(n)]
    cols = [f"sig__sig_{i:02d}" for i in range(n_sig)]
    return pd.DataFrame(X, index=samples, columns=cols), W_true


class TestNMF(unittest.TestCase):

    def test_nmf_converges_and_reconstructs(self):
        X, _ = _synth_tme_data(n=100, k_true=3, n_sig=12, seed=0)
        res = nmf_with_stability(X, k=3, n_runs=5, random_state=0)
        self.assertEqual(res.W.shape, (100, 3))
        self.assertEqual(res.H.shape, (3, 12))
        self.assertGreater(res.explained_variance, 0.80)

    def test_scan_and_select_k(self):
        X, _ = _synth_tme_data(n=80, k_true=3, n_sig=12, seed=1)
        results = scan_nmf_k(X, range(2, 5), n_runs=5, random_state=1)
        k = select_k_nmf(results, min_cophenetic=0.5, min_stability=0.5)
        # хотя бы какой-то k выбрался
        self.assertIn(k, results.keys())

    def test_project_new_samples(self):
        X, _ = _synth_tme_data(n=100, k_true=3, n_sig=12, seed=2)
        res = nmf_with_stability(X, k=3, n_runs=3, random_state=2)
        X_new, _ = _synth_tme_data(n=30, k_true=3, n_sig=12, seed=3)
        W_new = project_new_samples(X_new, res.H)
        self.assertEqual(W_new.shape, (30, 3))
        # loadings должны быть неотрицательные после normalize
        self.assertTrue((W_new.values >= 0).all())

    def test_name_factors(self):
        # H: factors (index) × signatures (columns) — как выход NMF
        H = pd.DataFrame(
            {
                "sig__CD8_T_cell": [2.0, 0.1],
                "sig__IFN_gamma": [1.5, 0.1],
                "sig__TLS_signature": [1.2, 0.05],
                "sig__CAF_proxy": [0.1, 2.0],
                "sig__TGFb": [0.1, 1.5],
                "sig__HLA_I_score": [0.8, 0.1],
            },
            index=["factor_1", "factor_2"],
        )
        names = name_factors_by_top_signatures(H)
        self.assertEqual(set(names["factor"]), {"factor_1", "factor_2"})
        joined = " ".join(names["name"].str.lower())
        # factor_1 должен получить inflamed-like или TLS-rich имя
        self.assertTrue(
            "inflamed" in joined or "tls" in joined,
            f"Expected 'inflamed' or 'TLS' in {list(names['name'])}",
        )


class TestRiskScore(unittest.TestCase):

    def test_fit_and_score(self):
        rng = np.random.default_rng(0)
        n = 150
        W = pd.DataFrame(
            rng.uniform(0, 1, size=(n, 3)),
            columns=["factor_1", "factor_2", "factor_3"],
            index=[f"S{i}" for i in range(n)],
        )
        # синтезируем survival, где factor_1 высокий = высокий риск
        hazard = np.exp(1.5 * W["factor_1"].values - 0.5 * W["factor_3"].values)
        time = rng.exponential(scale=5 / hazard)
        event = (time < 8).astype(int)
        clinical = pd.DataFrame({
            "age": rng.integers(40, 80, size=n),
            "stage": rng.integers(1, 4, size=n),
        }, index=W.index)

        model = fit_composite_risk_score(
            W, pd.Series(time, index=W.index),
            pd.Series(event, index=W.index),
            clinical=clinical,
            clinical_covariates=["age", "stage"],
            n_bootstrap=50, random_state=0,
        )
        # factor_1 должен получить положительный коэффициент
        self.assertGreater(model.coefficients["factor_1"], 0,
                           "factor_1 should drive positive risk in this synthetic")
        self.assertGreater(model.concordance_index, 0.55)
        # score на training возвращает Series правильной длины
        rs = model.score(W)
        self.assertEqual(len(rs), n)
        cat = model.tertile(rs)
        self.assertEqual(set(cat.unique()) - {"low", "medium", "high"}, set())

    def test_tertile_km(self):
        rng = np.random.default_rng(0)
        n = 120
        rs = pd.Series(rng.normal(size=n), index=[f"S{i}" for i in range(n)])
        cat = pd.cut(rs, bins=3, labels=["low", "medium", "high"]).astype(str)
        time = pd.Series(rng.exponential(5, size=n), index=rs.index)
        event = pd.Series(rng.choice([0, 1], size=n), index=rs.index)
        km = tertile_km(rs, cat, time, event)
        self.assertIn("logrank_overall", km)
        self.assertEqual(len(km["curves"]), 3)

    def test_dca_shape(self):
        rng = np.random.default_rng(0)
        n = 100
        rs = pd.Series(rng.normal(size=n))
        event = pd.Series(rng.choice([0, 1], size=n))
        dca = decision_curve_analysis(rs, event)
        self.assertIn("nb_model", dca.columns)
        self.assertIn("nb_treat_all", dca.columns)
        self.assertGreater(len(dca), 0)


class TestDiscoveryValidationNMF(unittest.TestCase):

    def test_nmf_stability_pass_warn(self):
        self.assertEqual(check_nmf_stability(0.92, 0.85, 4).status, CheckStatus.PASS)
        self.assertEqual(check_nmf_stability(0.80, 0.70, 4).status, CheckStatus.WARN)

    def test_risk_discrimination(self):
        self.assertEqual(
            check_risk_score_discriminates(0.65, 0.55).status, CheckStatus.PASS
        )
        self.assertEqual(
            check_risk_score_discriminates(0.52, 0.45).status, CheckStatus.FAIL
        )

    def test_tertile_separation(self):
        self.assertEqual(
            check_tertile_separation(0.001, 50, 50, 50).status, CheckStatus.PASS
        )
        self.assertEqual(
            check_tertile_separation(0.5, 50, 50, 50).status, CheckStatus.WARN
        )
        self.assertEqual(
            check_tertile_separation(0.001, 5, 50, 50).status, CheckStatus.WARN
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
