"""Тесты normalize / harmonize / ComBat-обёртки."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))

from oncotme.preprocess.normalize import (
    combat_fit_transform,
    combat_transform,
    harmonize_genes,
    log2_tpm,
    rank_normalize,
)


class TestNormalize(unittest.TestCase):

    def test_log2_tpm_monotonic(self):
        x = pd.DataFrame([[0.0, 1.0, 10.0, 100.0]])
        y = log2_tpm(x)
        self.assertTrue((np.diff(y.values.ravel()) > 0).all())

    def test_log2_tpm_zero_stays_finite(self):
        x = pd.DataFrame([[0.0]])
        y = log2_tpm(x)
        self.assertTrue(np.isfinite(y.values).all())
        self.assertEqual(float(y.iloc[0, 0]), 0.0)  # log2(0+1) = 0

    def test_rank_normalize_range(self):
        rng = np.random.default_rng(0)
        x = pd.DataFrame(rng.normal(size=(20, 5)))
        r = rank_normalize(x)
        self.assertTrue((r.values >= 0).all() and (r.values <= 1).all())


class TestHarmonize(unittest.TestCase):

    def test_intersection_common_genes(self):
        a = pd.DataFrame(np.zeros((4, 3)), index=["G1", "G2", "G3", "G4"])
        b = pd.DataFrame(np.zeros((3, 3)), index=["G2", "G3", "G5"])
        out = harmonize_genes([a, b])
        self.assertEqual(list(out[0].index), ["G2", "G3"])
        self.assertEqual(list(out[1].index), ["G2", "G3"])

    def test_empty_intersection_raises(self):
        a = pd.DataFrame(np.zeros((2, 2)), index=["G1", "G2"])
        b = pd.DataFrame(np.zeros((2, 2)), index=["G3", "G4"])
        with self.assertRaises(ValueError):
            harmonize_genes([a, b])


class TestCombatFallback(unittest.TestCase):

    def test_combat_without_pycombat_returns_input(self):
        """Если pycombat недоступен, модуль логирует warning и возвращает вход."""
        expr = pd.DataFrame(
            np.random.normal(size=(50, 10)),
            index=[f"G{i}" for i in range(50)],
            columns=[f"S{i}" for i in range(10)],
        )
        batches = pd.Series(["a"] * 5 + ["b"] * 5, index=expr.columns)
        corrected, model = combat_fit_transform(expr, batches)
        self.assertEqual(corrected.shape, expr.shape)
        # либо ComBat действительно был применён (model != None),
        # либо fallback (model == None и значения не поменялись)
        if model is None:
            pd.testing.assert_frame_equal(corrected, expr)
        # transform тоже не должно падать
        out = combat_transform(expr, model, batches)
        self.assertEqual(out.shape, expr.shape)


if __name__ == "__main__":
    unittest.main(verbosity=2)
