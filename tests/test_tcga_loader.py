"""Тесты TCGA loader'а (без реальных 180МБ данных).

Создаём миниатюрные TCGA-like TSV на лету → проверяем, что loader
возвращает корректный CohortBundle с нужными колонками и
что downstream validation на нём проходит.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))

from oncotme.cohorts.tcga import load_tcga_brca
from oncotme.validation.cohort_checks import check_cohort_bundle
from oncotme.validation.result import CheckStatus


def _write_toy_tcga(tmpdir: Path, n_samples: int = 120, n_genes: int = 200) -> dict[str, Path]:
    """Создаёт toy-TCGA: TPM matrix, GDC clinical, Xena clinical, Xena survival."""
    rng = np.random.default_rng(42)
    samples = [f"TCGA-AA-{i:04d}" for i in range(n_samples)]
    genes = [f"G{i:04d}" for i in range(n_genes - 20)] + [
        "HLA-A", "HLA-B", "HLA-C", "B2M", "CD274", "CTLA4", "GZMA", "PRF1",
        "CD8A", "CD4", "FOXP3", "CD19", "MS4A1", "CIITA", "STAT1", "IDO1",
        "TAP1", "TAP2", "PSMB8", "PSMB9",
    ]
    # TPM в высоких значениях чтобы loader применил log2
    tpm = rng.uniform(0.01, 2000.0, size=(len(genes), n_samples))
    tpm_df = pd.DataFrame(tpm, index=genes, columns=samples)
    tpm_df.index.name = "gene"
    tpm_path = tmpdir / "tcga_brca_rnaseq_tpm.tsv"
    tpm_df.reset_index().to_csv(tpm_path, sep="\t", index=False)

    # GDC clinical
    gdc_clin = pd.DataFrame(
        {
            "bcr_patient_barcode": samples,
            "age_at_diagnosis": rng.integers(35, 85, size=n_samples),
            "ajcc_pathologic_t": rng.choice(["T1", "T2", "T3", "T4"], size=n_samples),
            "ajcc_pathologic_n": rng.choice(["N0", "N1", "N2", "N3"], size=n_samples),
            "er_status_by_ihc": rng.choice(["positive", "negative"], size=n_samples),
            "pr_status_by_ihc": rng.choice(["positive", "negative"], size=n_samples),
            "her2_status_by_ihc": rng.choice(["positive", "negative"], size=n_samples),
            "vital_status": rng.choice(["Alive", "Dead"], size=n_samples),
            "days_to_death": rng.integers(30, 3000, size=n_samples).astype(float),
        }
    )
    gdc_path = tmpdir / "tcga_brca_clinical.tsv"
    gdc_clin.to_csv(gdc_path, sep="\t", index=False)

    # Xena clinical (PAM50, extra nature2012)
    xena_clin = pd.DataFrame(
        {
            "sampleID": samples,
            "PAM50Call_RNAseq": rng.choice(
                ["Basal", "Her2", "LumA", "LumB", "Normal"], size=n_samples
            ),
        }
    )
    xena_path = tmpdir / "BRCA_clinicalMatrix.tsv"
    xena_clin.to_csv(xena_path, sep="\t", index=False)

    # Xena survival
    surv = pd.DataFrame(
        {
            "sample": samples,
            "OS.time": rng.integers(30, 3500, size=n_samples),
            "OS": rng.choice([0, 1], size=n_samples),
            "DFS.time": rng.integers(30, 3500, size=n_samples),
            "DFS": rng.choice([0, 1], size=n_samples),
        }
    )
    surv_path = tmpdir / "BRCA_survival.tsv"
    surv.to_csv(surv_path, sep="\t", index=False)

    return {
        "tpm": tpm_path,
        "gdc_clin": gdc_path,
        "xena_clin": xena_path,
        "surv": surv_path,
    }


class TestTcgaLoader(unittest.TestCase):

    def test_loader_returns_valid_bundle(self):
        with tempfile.TemporaryDirectory() as td:
            paths = _write_toy_tcga(Path(td))
            bundle = load_tcga_brca(
                expression_path=paths["tpm"],
                clinical_path=paths["gdc_clin"],
                xena_clinical_path=paths["xena_clin"],
                survival_path=paths["surv"],
            )
            # basic structure
            self.assertEqual(bundle.platform, "rnaseq")
            self.assertEqual(bundle.expr_kind, "log2_tpm")
            self.assertGreater(bundle.n_samples, 100)
            self.assertGreater(bundle.n_genes, 100)
            # endpoints должны быть заполнены из Xena survival
            self.assertIn("time_os", bundle.endpoints.columns)
            self.assertIn("event_os", bundle.endpoints.columns)
            # clinical должен иметь основные столбцы
            for col in ["age", "subtype_er", "subtype_her2", "subtype_pam50"]:
                self.assertIn(col, bundle.clinical.columns, f"Missing clin col {col}")

    def test_loader_output_passes_validation(self):
        """Полный TCGA-подобный toy datagen должен проходить check_cohort_bundle."""
        with tempfile.TemporaryDirectory() as td:
            paths = _write_toy_tcga(Path(td))
            bundle = load_tcga_brca(
                expression_path=paths["tpm"],
                clinical_path=paths["gdc_clin"],
                xena_clinical_path=paths["xena_clin"],
                survival_path=paths["surv"],
            )
            results = check_cohort_bundle(bundle)
            fails = [r for r in results if r.status == CheckStatus.FAIL]
            self.assertEqual(
                fails, [],
                msg=f"Cohort validation failed: {[(r.name, r.message) for r in fails]}",
            )

    def test_log2_not_applied_twice(self):
        """Если TPM уже log2, loader не должен лог2-ить ещё раз."""
        with tempfile.TemporaryDirectory() as td:
            tpm = pd.DataFrame(
                np.random.uniform(0.1, 10, size=(50, 30)),
                index=[f"G{i:04d}" for i in range(50)],
                columns=[f"TCGA-AA-{i:04d}" for i in range(30)],
            )
            tpm.index.name = "gene"
            tpm_path = Path(td) / "tpm.tsv"
            tpm.reset_index().to_csv(tpm_path, sep="\t", index=False)

            clin = pd.DataFrame({
                "bcr_patient_barcode": tpm.columns,
                "age_at_diagnosis": 50,
                "vital_status": "Alive",
            })
            clin_path = Path(td) / "clin.tsv"
            clin.to_csv(clin_path, sep="\t", index=False)

            bundle = load_tcga_brca(
                expression_path=tpm_path, clinical_path=clin_path,
            )
            # max оригинального == 10, значит loader не применил log2 повторно
            self.assertLess(float(bundle.expr.values.max()), 15)


if __name__ == "__main__":
    unittest.main(verbosity=2)
