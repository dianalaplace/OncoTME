"""Тесты ssGSEA/fallback и GMT-парсера."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))

from oncotme.features.signatures import fallback_zscore, load_gmt, ssgsea_scores


class TestGMT(unittest.TestCase):

    def test_load_real_gmt(self):
        gmt = load_gmt(_REPO / "signatures" / "tme_signatures.gmt")
        self.assertGreaterEqual(len(gmt), 15)
        for name, genes in gmt.items():
            self.assertGreater(len(genes), 0, f"Signature {name!r} is empty")
            for g in genes:
                self.assertEqual(g, g.upper(), f"Gene {g!r} is not uppercase")

    def test_load_gmt_ignores_short_lines(self):
        with tempfile.NamedTemporaryFile("w", suffix=".gmt", delete=False) as f:
            f.write("good_sig\tdesc\tA\tB\tC\n")
            f.write("too_short_line\n")
            f.write("empty_sig\tdesc\n")
            path = f.name
        try:
            gmt = load_gmt(path)
            self.assertIn("good_sig", gmt)
            self.assertNotIn("too_short_line", gmt)
            self.assertNotIn("empty_sig", gmt)
        finally:
            Path(path).unlink()


class TestFallbackZscore(unittest.TestCase):

    def test_basic_shape_and_dtype(self):
        rng = np.random.default_rng(0)
        expr = pd.DataFrame(
            rng.normal(5, 1, size=(50, 20)),
            index=[f"G{i}" for i in range(50)],
            columns=[f"S{i}" for i in range(20)],
        )
        gmt = {"sig1": ["G0", "G1", "G2"], "sig2": ["G10", "G11"]}
        scores = fallback_zscore(expr, gmt)
        self.assertEqual(scores.shape, (20, 2))
        self.assertListEqual(list(scores.columns), ["sig1", "sig2"])
        self.assertFalse(scores.isna().any().any())

    def test_empty_signature_returns_zero(self):
        expr = pd.DataFrame(np.ones((3, 5)), index=["A", "B", "C"],
                            columns=[f"S{i}" for i in range(5)])
        gmt = {"nonexistent": ["XYZ1", "XYZ2"]}
        scores = fallback_zscore(expr, gmt)
        self.assertEqual(scores.shape, (5, 1))
        self.assertTrue((scores["nonexistent"] == 0).all())

    def test_case_insensitive_gene_matching(self):
        expr = pd.DataFrame(
            [[1.0, 2.0, 3.0, 4.0, 5.0]],
            index=["cd8a"],  # lowercase
            columns=[f"S{i}" for i in range(5)],
        )
        gmt = {"sig": ["CD8A"]}  # uppercase request
        scores = fallback_zscore(expr, gmt)
        self.assertFalse((scores["sig"] == 0).all())

    def test_constant_genes_do_not_produce_nan(self):
        expr = pd.DataFrame(
            [[5.0] * 5, [1.0, 2.0, 3.0, 4.0, 5.0]],
            index=["CONST", "VAR"],
            columns=[f"S{i}" for i in range(5)],
        )
        gmt = {"sig": ["CONST", "VAR"]}
        scores = fallback_zscore(expr, gmt)
        self.assertFalse(scores.isna().any().any())

    def test_latent_factor_preserves_signature_corr(self):
        """Если два набора генов ап-регулированы общим латентным фактором,
        их signature scores должны коррелировать +."""
        rng = np.random.default_rng(0)
        n_samples = 100
        latent = rng.normal(size=n_samples)
        genes_a = [f"A{i}" for i in range(10)]
        genes_b = [f"B{i}" for i in range(10)]
        rows = {}
        for g in genes_a + genes_b:
            rows[g] = rng.normal(size=n_samples) + 1.5 * latent
        expr = pd.DataFrame(rows).T
        expr.columns = [f"S{i}" for i in range(n_samples)]
        scores = fallback_zscore(expr, {"A": genes_a, "B": genes_b})
        r = scores["A"].corr(scores["B"])
        self.assertGreater(r, 0.6)


class TestSsgseaFallback(unittest.TestCase):

    def test_ssgsea_uses_fallback_when_gseapy_unavailable(self):
        # gseapy недоступен в тест-окружении → должен переключиться на fallback
        # без падения; результат — валидный DataFrame
        rng = np.random.default_rng(42)
        expr = pd.DataFrame(
            rng.normal(5, 1, size=(30, 10)),
            index=[f"G{i}" for i in range(30)],
            columns=[f"S{i}" for i in range(10)],
        )
        gmt = {"sig1": ["G0", "G1"], "sig2": ["G5", "G6"]}
        scores = ssgsea_scores(expr, gene_sets=gmt)
        self.assertEqual(scores.shape[0], 10)
        self.assertEqual(scores.shape[1], 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
