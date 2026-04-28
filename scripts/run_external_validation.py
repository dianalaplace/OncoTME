"""External validation: train on GSE25065, test on GSE25066.

Это центральный момент для публикации: модель, показавшая +0.108 ΔAUC vs clinical
на training cohort, должна воспроизвестись на independent cohort **без re-fit**.

Steps:
1. Load features + endpoints для обеих когорт.
2. Harmonize feature columns (пересечение).
3. Fit всех моделей на GSE25065 train.
4. Predict на GSE25066 test.
5. AUC + 95% bootstrap CI + permutation test.
6. Сравнить с GSE25065 internal CV — если similar → replicates.
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from oncotme.benchmark.classification import (
    ClfCombinedModel,
    ClinicalLogitModel,
    CYT_Clf,
    ElasticNetLogitModel,
    NMFLogitModel,
    RandomFeaturesClf,
    Signature_Clf,
    TIS_Clf,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("external_val")


def _bootstrap_auc_ci(y_true, y_pred, n=500, seed=42):
    rng = np.random.default_rng(seed)
    aucs = []
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n_s = len(y_true)
    for _ in range(n):
        idx = rng.choice(n_s, size=n_s, replace=True)
        try:
            a = roc_auc_score(y_true[idx], y_pred[idx])
            aucs.append(a)
        except Exception:
            continue
    if not aucs:
        return float("nan"), float("nan")
    return float(np.quantile(aucs, 0.025)), float(np.quantile(aucs, 0.975))


def _permutation_p(y_true, y_pred, observed, n_perm=500, seed=42):
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    null = []
    for _ in range(n_perm):
        y_perm = rng.permutation(y_true)
        try:
            null.append(roc_auc_score(y_perm, y_pred))
        except Exception:
            continue
    null = np.asarray(null)
    if len(null) == 0:
        return float("nan")
    return float(((null >= observed).sum() + 1) / (len(null) + 1))


def _parse():
    p = argparse.ArgumentParser()
    p.add_argument("--train-features",
                   default=str(_REPO / "data/processed/gse25065_features.parquet"))
    p.add_argument("--train-cohort",
                   default=str(_REPO / "data/processed/gse25065.pkl"))
    p.add_argument("--test-features",
                   default=str(_REPO / "data/processed/gse25066_features.parquet"))
    p.add_argument("--test-cohort",
                   default=str(_REPO / "data/processed/gse25066.pkl"))
    p.add_argument("--endpoint", default="pCR")
    p.add_argument("--out-dir",
                   default=str(_REPO / "results" / "external_validation"))
    p.add_argument("--n-permutations", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> int:
    args = _parse()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load
    Xtr = pd.read_parquet(args.train_features)
    Xte = pd.read_parquet(args.test_features)
    with open(args.train_cohort, "rb") as f:
        coh_tr = pickle.load(f)
    with open(args.test_cohort, "rb") as f:
        coh_te = pickle.load(f)

    ytr = coh_tr.endpoints[args.endpoint]
    yte = coh_te.endpoints[args.endpoint]
    ytr = ytr.loc[ytr.notna()].astype(int)
    yte = yte.loc[yte.notna()].astype(int)

    logger.info(
        "Train: %s (N=%d, pos=%d)  Test: %s (N=%d, pos=%d)",
        coh_tr.name, len(ytr), int(ytr.sum()),
        coh_te.name, len(yte), int(yte.sum()),
    )

    # Harmonize feature columns — test должен иметь те же sig__/gene__/clin__
    common_cols = sorted(set(Xtr.columns) & set(Xte.columns))
    logger.info("Common feature cols: %d / %d (train) / %d (test)",
                len(common_cols), Xtr.shape[1], Xte.shape[1])
    Xtr = Xtr[common_cols]
    Xte = Xte[common_cols]

    clin_tr = coh_tr.clinical.reindex(Xtr.index)
    clin_te = coh_te.clinical.reindex(Xte.index)

    models = {
        "clinical_only": lambda: ClinicalLogitModel(),
        "TIS_18": lambda: TIS_Clf(),
        "CYT": lambda: CYT_Clf(),
        "sig_IFN_gamma": lambda: Signature_Clf("sig__IFN_gamma"),
        "sig_CD8": lambda: Signature_Clf("sig__CD8_T_cell"),
        "sig_APM": lambda: Signature_Clf("sig__APM_score"),
        "random_feat_s0": lambda: RandomFeaturesClf(n_features=6, seed=0),
        "random_feat_s1": lambda: RandomFeaturesClf(n_features=6, seed=1),
        "NMF_logit_k5": lambda: NMFLogitModel(k=5, n_runs=5),
        "ElasticNet_all": lambda: ElasticNetLogitModel(use_clinical=False),
        "ElasticNet_all+clinical": lambda: ElasticNetLogitModel(use_clinical=True),
        "NMF+clinical": lambda: ClfCombinedModel(
            NMFLogitModel(k=5, n_runs=5), ClinicalLogitModel(), name="NMF+clinical",
        ),
    }

    rows = []
    for name, factory in models.items():
        logger.info("Fit %s on train, predict test …", name)
        try:
            m = factory()
            m.fit(Xtr.loc[ytr.index], ytr, clin_tr.loc[ytr.index])
            p_test = m.predict_proba(Xte.loc[yte.index], clin_te.loc[yte.index])
            auc = float(roc_auc_score(yte, p_test.loc[yte.index]))
            ap = float(average_precision_score(yte, p_test.loc[yte.index]))
            ci_lo, ci_hi = _bootstrap_auc_ci(yte.values, p_test.loc[yte.index].values)
            p_perm = _permutation_p(yte.values, p_test.loc[yte.index].values, auc,
                                    n_perm=args.n_permutations, seed=args.seed)
            rows.append({
                "model": name, "auc": auc, "ap": ap,
                "auc_ci_lo": ci_lo, "auc_ci_hi": ci_hi,
                "permutation_p": p_perm,
            })
            logger.info("  AUC=%.4f [%.3f, %.3f]  AP=%.3f  perm p=%.4f",
                        auc, ci_lo, ci_hi, ap, p_perm)
        except Exception as exc:
            logger.warning("  %s failed: %s", name, exc)
            rows.append({"model": name, "auc": None, "error": str(exc)})

    bench = pd.DataFrame(rows)
    base = bench.loc[bench["model"] == "clinical_only", "auc"]
    if len(base) and pd.notna(base.iloc[0]):
        bench["delta_vs_clinical"] = bench["auc"].apply(
            lambda x: (x - base.iloc[0]) if x is not None and pd.notna(x) else None)
    bench = bench.sort_values("auc", ascending=False, na_position="last")
    bench.to_csv(out_dir / "external_validation_auc.csv", index=False)

    print("\n" + "=" * 90)
    print("External validation: train GSE25065 → test GSE25066 ({})".format(args.endpoint))
    print("=" * 90)
    print(bench.to_string(index=False,
        float_format=lambda x: f"{x:.4f}" if pd.notna(x) else "—"))
    print("=" * 90)

    # Summary JSON
    summary = {
        "train": {"name": coh_tr.name, "n": int(len(ytr)), "pos": int(ytr.sum())},
        "test": {"name": coh_te.name, "n": int(len(yte)), "pos": int(yte.sum())},
        "benchmark": bench.to_dict(orient="records"),
    }
    with (out_dir / "external_validation_summary.json").open("w") as f:
        json.dump(summary, f, indent=2, default=str)

    # Verdict
    lines = [
        f"# External validation — train {coh_tr.name} → test {coh_te.name}",
        "",
        f"- Train: N={len(ytr)}  pos={int(ytr.sum())} ({ytr.mean():.1%})",
        f"- Test:  N={len(yte)}  pos={int(yte.sum())} ({yte.mean():.1%})",
        f"- Endpoint: {args.endpoint}",
        "",
        "## Test-set AUC",
        "",
        "| Model | AUC | 95% CI | ΔvsClinical | AP | Permutation p |",
        "|---|---|---|---|---|---|",
    ]
    for _, r in bench.iterrows():
        auc = r.get("auc")
        lo, hi = r.get("auc_ci_lo"), r.get("auc_ci_hi")
        d = r.get("delta_vs_clinical")
        ap = r.get("ap")
        pp = r.get("permutation_p")
        lines.append(
            f"| {r['model']} | "
            f"{(f'{auc:.4f}' if pd.notna(auc) else '—')} | "
            f"{(f'[{lo:.3f}, {hi:.3f}]' if pd.notna(lo) else '—')} | "
            f"{(f'{d:+.3f}' if pd.notna(d) else '—')} | "
            f"{(f'{ap:.3f}' if pd.notna(ap) else '—')} | "
            f"{(f'{pp:.4f}' if pd.notna(pp) else '—')} |"
        )
    best = bench.loc[bench["delta_vs_clinical"].fillna(-1).idxmax()] \
        if "delta_vs_clinical" in bench.columns else None
    if best is not None and best.get("delta_vs_clinical") is not None:
        d = best["delta_vs_clinical"]
        if d >= 0.05:
            verdict = (
                f"✅ **Replicates clinically**: '{best['model']}' externally validates "
                f"with ΔAUC = {d:+.3f} over clinical-only. This is publication-grade "
                f"evidence for the TME-based predictor.")
        elif d >= 0.02:
            verdict = (
                f"🟡 **Partial replication**: '{best['model']}' ΔAUC = {d:+.3f}. "
                f"Smaller than internal CV — some over-fitting; signal real but modest.")
        else:
            verdict = (
                f"🔴 **Does NOT replicate**: best ΔAUC = {d:+.3f}. Internal signal "
                f"was cohort-specific; reconsider feature set or treatment-context "
                f"homogeneity.")
        lines += ["", "## Verdict", "", verdict]
    (out_dir / "VERDICT.md").write_text("\n".join(lines), encoding="utf-8")
    logger.info("Artefacts → %s", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
