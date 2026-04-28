"""Тесты discovery: preprocess, consensus, characterize, classifier, survival."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))

from oncotme.discovery.classifier import CentroidClassifier
from oncotme.discovery.consensus import consensus_cluster, select_k_by_pac
from oncotme.discovery.characterize import characterize_archetypes, name_archetypes
from oncotme.discovery.preprocess import preprocess_for_clustering
from oncotme.discovery.survival import km_by_archetype, multivariable_cox
from oncotme.validation.discovery_checks import (
    check_classifier_transfer_sanity,
    check_cluster_sizes,
    check_consensus_stability,
    check_subtype_redundancy,
)
from oncotme.validation.result import CheckStatus


def _synthetic_features(n_samples: int = 120, n_sigs: int = 20, k_true: int = 3,
                        seed: int = 0) -> pd.DataFrame:
    """Создаём synthetic signature-frame с явной кластерной структурой."""
    rng = np.random.default_rng(seed)
    per_group = n_samples // k_true
    parts = []
    for g in range(k_true):
        centers = rng.normal(0, 1, size=n_sigs)
        centers[g * (n_sigs // k_true):(g + 1) * (n_sigs // k_true)] += 3.0
        X = rng.normal(centers, 0.6, size=(per_group, n_sigs))
        parts.append(X)
    X = np.vstack(parts)
    samples = [f"S{i:03d}" for i in range(X.shape[0])]
    cols = [f"sig__sig_{i:02d}" for i in range(n_sigs)]
    df = pd.DataFrame(X, index=samples, columns=cols)
    true_labels = pd.Series(
        np.repeat(np.arange(1, k_true + 1), per_group), index=samples, name="true_cluster"
    )
    return df, true_labels


class TestPreprocess(unittest.TestCase):

    def test_preprocess_returns_only_sig_cols(self):
        X = pd.DataFrame({
            "sig__A": [1, 2, 3, 4],
            "sig__B": [5, 6, 7, 8],
            "gene__G1": [9, 10, 11, 12],
            "clin__age": [40, 50, 60, 70],
        }, index=[f"S{i}" for i in range(4)])
        out = preprocess_for_clustering(X)
        self.assertEqual(set(out.columns), {"sig__A", "sig__B"})

    def test_preprocess_scales_to_unit(self):
        rng = np.random.default_rng(0)
        X = pd.DataFrame(
            rng.normal(100, 50, size=(50, 5)) * np.arange(1, 6),  # разные масштабы
            index=[f"S{i}" for i in range(50)],
            columns=[f"sig__s{i}" for i in range(5)],
        )
        out = preprocess_for_clustering(X)
        # все std должны быть в разумном диапазоне после robust z-score
        stds = out.std()
        self.assertLess(stds.max() / stds.min(), 5.0,
                        "Robust z-score did not equalize scales")


class TestConsensusClustering(unittest.TestCase):

    def test_consensus_recovers_known_structure(self):
        X, true = _synthetic_features(n_samples=90, n_sigs=15, k_true=3, seed=0)
        X_pre = preprocess_for_clustering(X)
        results = consensus_cluster(X_pre, k_range=range(2, 5), n_bootstrap=20,
                                    random_state=0)
        # PAC при k=3 должен быть минимальным
        pacs = {k: r.pac for k, r in results.items()}
        best = min(pacs, key=pacs.get)
        self.assertEqual(best, 3, f"Expected k=3, got k={best}. PACs: {pacs}")

    def test_select_k_by_pac_filters_small_clusters(self):
        # конструируем results вручную: k=5 имеет крошечный кластер
        from oncotme.discovery.consensus import ConsensusResult
        results = {
            2: ConsensusResult(2, np.ones((10, 10)), np.array([1]*5+[2]*5),
                               pac=0.2, cdf_area=0.5, silhouette=0.3,
                               cluster_sizes={1: 5, 2: 5}),
            5: ConsensusResult(5, np.ones((10, 10)), np.array([1]*7+[2,3,4,5]+[1]*(-1+1)),
                               pac=0.1, cdf_area=0.5, silhouette=0.3,
                               cluster_sizes={1: 7, 2: 1, 3: 1, 4: 1, 5: 0}),
        }
        # k=5 имеет min_fraction = 0 → должен быть отвергнут
        k = select_k_by_pac(results, min_cluster_fraction=0.1)
        self.assertEqual(k, 2)


class TestCharacterize(unittest.TestCase):

    def test_profile_shape_and_names(self):
        X, true = _synthetic_features(n_samples=60, n_sigs=10, k_true=3, seed=1)
        X_pre = preprocess_for_clustering(X)
        # клинические — пустые
        clinical = pd.DataFrame(index=X.index)
        char = characterize_archetypes(X_pre, true, clinical)
        self.assertEqual(char["profiles"].shape[0], 3)  # 3 кластера
        self.assertEqual(char["profiles"].shape[1], X_pre.shape[1])
        self.assertEqual(len(char["names"]), 3)

    def test_name_archetypes_contains_known_patterns(self):
        # profiles с явным "inflamed" паттерном
        profiles = pd.DataFrame({
            "sig__CD8_T_cell": [2.0, -1.0],
            "sig__IFN_gamma": [2.5, -0.5],
            "sig__TLS_signature": [1.5, -1.0],
            "sig__CAF_proxy": [-1.0, 1.5],
            "sig__TGFb": [-1.0, 1.0],
            "sig__APM_score": [1.0, -0.5],
        }, index=[1, 2])
        names = name_archetypes(profiles)
        # кластер 1: должен получить inflamed-like имя
        name_0 = names.loc[names["cluster_id"] == 1, "name"].iloc[0]
        self.assertIn("inflamed", name_0.lower(), f"Got name: {name_0}")


class TestClassifier(unittest.TestCase):

    def test_centroid_classifier_self_predict(self):
        X, true = _synthetic_features(n_samples=90, n_sigs=15, k_true=3, seed=2)
        X_pre = preprocess_for_clustering(X)
        clf = CentroidClassifier(metric="correlation")
        clf.fit(X_pre, true)
        preds = clf.predict(X_pre)
        acc = float((preds.values == true.values).mean())
        self.assertGreater(acc, 0.85)

    def test_centroid_classifier_transfer(self):
        X, true = _synthetic_features(n_samples=90, n_sigs=15, k_true=3, seed=3)
        X_pre = preprocess_for_clustering(X)
        clf = CentroidClassifier().fit(X_pre, true)
        # создаём "внешнюю" когорту со сдвинутыми значениями
        X2, true2 = _synthetic_features(n_samples=60, n_sigs=15, k_true=3, seed=4)
        X2_pre = preprocess_for_clustering(X2)
        preds = clf.predict(X2_pre)
        # совпадений должно быть больше случайного (33%)
        # ключ в том, что оба датасета — tri-modal, структуры одинаковые
        from scipy.stats import mode
        # сопоставляем предсказанные кластеры с истинными через majority vote
        cm = pd.crosstab(true2, preds)
        best_assign = cm.values.max(axis=1).sum() / cm.values.sum()
        self.assertGreater(best_assign, 0.5,
                           f"Transfer accuracy {best_assign} too low.")


class TestDiscoveryValidation(unittest.TestCase):

    def test_pac_check(self):
        good = check_consensus_stability(0.1, k=3, max_pac=0.25)
        bad = check_consensus_stability(0.4, k=3, max_pac=0.25)
        self.assertEqual(good.status, CheckStatus.PASS)
        self.assertEqual(bad.status, CheckStatus.FAIL)

    def test_cluster_sizes_check(self):
        ok = check_cluster_sizes({1: 50, 2: 30, 3: 20}, min_fraction=0.05)
        bad = check_cluster_sizes({1: 100, 2: 5}, min_fraction=0.1)
        self.assertEqual(ok.status, CheckStatus.PASS)
        self.assertEqual(bad.status, CheckStatus.WARN)

    def test_subtype_redundancy(self):
        # кластер 1 полностью BASAL — redundant
        labels = pd.Series([1]*20 + [2]*20, index=[f"S{i}" for i in range(40)])
        subtype = pd.Series(["BASAL"]*20 + ["LUMA"]*10 + ["LUMB"]*10,
                             index=labels.index)
        result = check_subtype_redundancy(labels, subtype, max_single_subtype_fraction=0.95)
        self.assertEqual(result.status, CheckStatus.WARN)

    def test_classifier_sanity(self):
        X, true = _synthetic_features(n_samples=60, n_sigs=10, k_true=3, seed=5)
        X_pre = preprocess_for_clustering(X)
        clf = CentroidClassifier().fit(X_pre, true)
        good = check_classifier_transfer_sanity(clf, X_pre, true, min_accuracy=0.85)
        self.assertEqual(good.status, CheckStatus.PASS)


class TestSurvival(unittest.TestCase):

    def test_km_and_cox_run(self):
        rng = np.random.default_rng(0)
        n = 120
        labels = pd.Series(rng.choice([1, 2, 3], size=n), index=[f"S{i}" for i in range(n)])
        time = pd.Series(rng.exponential(scale=5, size=n), index=labels.index)
        event = pd.Series(rng.choice([0, 1], size=n, p=[0.4, 0.6]), index=labels.index)
        clinical = pd.DataFrame({
            "age": rng.integers(35, 75, size=n),
            "stage": rng.integers(1, 4, size=n),
            "subtype_er": rng.choice([0, 1], size=n),
            "subtype_her2": rng.choice([0, 1], size=n),
        }, index=labels.index)
        km = km_by_archetype(labels, time, event)
        self.assertIn("logrank_overall", km)
        self.assertIn("p_value", km["logrank_overall"])

        cox = multivariable_cox(labels, time, event, clinical)
        self.assertIn("summary", cox)
        self.assertIn("concordance_index", cox)
        self.assertGreaterEqual(cox["concordance_index"], 0.3)
        self.assertLessEqual(cox["concordance_index"], 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
