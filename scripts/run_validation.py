"""CLI: запускает self-validation suite на текущем состоянии проекта.

Примеры::

    # smoke-прогон на синтетических данных (без загруженных когорт)
    python scripts/run_validation.py --synthetic

    # на реальных когортах и фичах (когда они будут загружены)
    python scripts/run_validation.py \\
        --cohorts data/processed/tcga_brca.pkl data/processed/metabric.pkl \\
        --feature-frames tcga=data/processed/tcga_features.parquet \\
                         metabric=data/processed/metabric_features.parquet \\
        --out results/validation/report.json

Exit-коды:
    0 — все проверки PASS/WARN
    1 — хотя бы один FAIL (CI gate)
    2 — ошибка запуска (не удалось загрузить артефакты)
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path
from typing import Dict, List

# работаем из корня репо: добавляем src/ в path при запуске как script
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

import numpy as np
import pandas as pd

from oncotme.cohorts.base import CohortBundle
from oncotme.features.baselines import TIS_GENES
from oncotme.features.signatures import fallback_zscore, load_gmt
from oncotme.validation import run_suite


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="OncoTME self-validation suite")
    p.add_argument("--synthetic", action="store_true", help="Run on synthetic data (smoke).")
    p.add_argument("--cohorts", nargs="*", default=[], help="Paths to pickled CohortBundle files.")
    p.add_argument(
        "--feature-frames",
        nargs="*",
        default=[],
        help="Feature frames as name=path (parquet).",
    )
    p.add_argument("--out", default="results/validation/report.json", help="Path for JSON report.")
    p.add_argument("--strict", action="store_true", help="Exit 1 on any WARN as well as FAIL.")
    return p.parse_args()


def _synthetic_cohorts() -> List[CohortBundle]:
    """Две маленькие синтетические когорты с валидной структурой,
    чтобы прогнать полный suite без реальных данных."""
    rng = np.random.default_rng(42)

    gmt = load_gmt(str(_REPO_ROOT / "signatures" / "tme_signatures.gmt"))
    # все гены из GMT + TIS + служебные
    genes = sorted(
        {g for gs in gmt.values() for g in gs}
        | set(TIS_GENES)
        | {"GZMA", "PRF1", "B2M", "HLA-A", "HLA-B", "HLA-C"}
    )

    def _make(name: str, n_samples: int, prefix: str, platform: str, expr_kind: str) -> CohortBundle:
        samples = [f"{prefix}{i:04d}" for i in range(n_samples)]
        # базовый фон
        base = rng.normal(5.0, 1.5, size=(len(genes), n_samples))
        # вшитая биология: создаём скрытый «иммунный» фактор,
        # ап-регулируем CD8/IFN/HLA/APM вместе, CAF/TGFb анти-коррелируем
        latent = rng.normal(0, 1, size=n_samples)
        idx = {g: i for i, g in enumerate(genes)}
        immune_genes = sum((gmt.get(s, []) for s in
                            ["CD8_T_cell", "IFN_gamma", "HLA_I_score", "APM_score",
                             "TLS_signature", "B_cell", "checkpoint_score"]), [])
        stromal_genes = sum((gmt.get(s, []) for s in ["CAF_proxy", "TGFb", "hypoxia"]), [])
        for g in set(immune_genes):
            if g in idx:
                base[idx[g]] += 1.2 * latent
        for g in set(stromal_genes):
            if g in idx:
                base[idx[g]] -= 0.6 * latent
        expr = pd.DataFrame(base, index=genes, columns=samples)
        clinical = pd.DataFrame(
            {
                "age": rng.integers(35, 75, size=n_samples),
                "subtype_er": rng.choice([0, 1], size=n_samples),
                "subtype_her2": rng.choice([0, 1], size=n_samples),
                "stage": rng.choice(["I", "II", "III"], size=n_samples),
            },
            index=samples,
        )
        # pCR завязан на latent: «горячие» чаще отвечают
        pcr_prob = 1 / (1 + np.exp(-1.2 * latent))
        pcr = (rng.uniform(size=n_samples) < pcr_prob).astype(int)
        # survival: hazard пропорционально -latent (горячие живут дольше)
        time_rfs = rng.exponential(scale=np.exp(-0.5 * latent) * 5 + 1.0)
        event_rfs = (time_rfs < 10).astype(int)
        endpoints = pd.DataFrame(
            {"pCR": pcr, "time_rfs": time_rfs, "event_rfs": event_rfs},
            index=samples,
        )
        return CohortBundle(
            name=name,
            expr=expr,
            clinical=clinical,
            endpoints=endpoints,
            treatment_context="neoadjuvant_chemo",
            platform=platform,
            expr_kind=expr_kind,
        )

    a = _make("synth_tcga", 120, "A_", "rnaseq", "log2_tpm")
    b = _make("synth_metabric", 150, "B_", "microarray_illumina_ht12", "log2_intensity")
    return [a, b]


def _synthetic_features(cohorts: List[CohortBundle]) -> Dict[str, pd.DataFrame]:
    gmt = load_gmt(str(_REPO_ROOT / "signatures" / "tme_signatures.gmt"))
    out = {}
    for c in cohorts:
        scores = fallback_zscore(c.expr, gmt)
        scores.columns = [f"sig__{col}" for col in scores.columns]
        # добавим индивидуальные гены-APM/checkpoint
        for g in ["HLA-A", "HLA-B", "HLA-C", "B2M", "CD274", "CTLA4", "LAG3"]:
            if g in c.expr.index:
                scores[f"gene__{g}"] = c.expr.loc[g].reindex(scores.index).values
        # клинические
        for clin_col in ["age", "subtype_er", "subtype_her2"]:
            if clin_col in c.clinical.columns:
                scores[f"clin__{clin_col}"] = c.clinical[clin_col].reindex(scores.index).values
        out[c.name] = scores
    return out


def _load_cohort_from_pickle(path: str) -> CohortBundle:
    with open(path, "rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, CohortBundle):
        raise TypeError(f"{path} does not contain a CohortBundle (got {type(obj).__name__}).")
    return obj


def _load_feature_frame(spec: str) -> tuple[str, pd.DataFrame]:
    if "=" not in spec:
        raise ValueError(f"Expected name=path, got {spec!r}")
    name, path = spec.split("=", 1)
    return name.strip(), pd.read_parquet(path)


def main() -> int:
    args = _parse_args()

    cohorts: List[CohortBundle] = []
    feature_frames: Dict[str, pd.DataFrame] = {}

    try:
        if args.synthetic:
            cohorts = _synthetic_cohorts()
            feature_frames = _synthetic_features(cohorts)
        else:
            for p in args.cohorts:
                cohorts.append(_load_cohort_from_pickle(p))
            for spec in args.feature_frames:
                name, frame = _load_feature_frame(spec)
                feature_frames[name] = frame
    except Exception as exc:
        print(f"[run_validation] Failed to load artefacts: {exc}", file=sys.stderr)
        return 2

    report = run_suite(cohorts=cohorts, feature_frames=feature_frames)
    print(report.to_text())
    report.to_json(args.out)
    print(f"\nReport saved to {args.out}")

    counts = report.counts()
    if counts["FAIL"] > 0:
        return 1
    if args.strict and counts["WARN"] > 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
