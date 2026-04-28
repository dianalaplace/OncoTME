"""Pre-specified TME composites for each targeted therapy class на I-SPY2.

Протокол научный:
- Каждый composite pre-specified on published biology (см. targeted_composites.py)
- One interaction test per class: pCR ~ composite_z + arm + composite_z:arm + HR + MP
- Family-wise correction: BH FDR over 6 classes (not 23×9 = 207)
- Report tertile ARR/OR per class + forest plot

Это главный анализ "TME × targeted therapy" проекта OncoTME.
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import sys
import warnings
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))

import numpy as np
import pandas as pd

from oncotme.benchmark.targeted_composites import ALL_COMPOSITES, TargetedComposite
from oncotme.features.signatures import load_gmt, ssgsea_scores

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("targeted_composites")


def _parse_meta(path: str) -> pd.DataFrame:
    samples, chars = None, {}
    with gzip.open(path, "rt", encoding="latin-1") as f:
        for line in f:
            if line.startswith("!Sample_geo_accession"):
                samples = [p.strip('"') for p in line.rstrip().split("\t")[1:]]
            elif line.startswith("!Sample_characteristics_ch"):
                parts = line.rstrip().split("\t")[1:]
                kvs = []
                for p in parts:
                    p = p.strip('"')
                    if ":" in p:
                        k, v = p.split(": ", 1)
                        kvs.append((k.strip(), v.strip()))
                    else:
                        kvs.append((None, None))
                key = next((k for k, _ in kvs if k is not None), None)
                if key is None:
                    continue
                chars.setdefault(key, [v for _, v in kvs])
    return pd.DataFrame(chars, index=samples)


def _load_gene(path):
    df = pd.read_csv(path, sep="\t", low_memory=False, index_col=0)
    df = df.apply(pd.to_numeric, errors="coerce")
    df.index = df.index.astype(str).str.upper()
    return df.loc[~df.index.duplicated(keep="first")].dropna(how="all")


def _zscore(s: pd.Series) -> pd.Series:
    std = s.std()
    return (s - s.mean()) / (std if std else 1.0)


def _compute_composite(scores: pd.DataFrame, comp: TargetedComposite) -> pd.Series:
    """Apply pre-specified formula to per-sample ssGSEA scores."""
    terms = []
    for c in comp.components:
        if c.name not in scores.columns:
            logger.warning("Composite %s: missing signature %s (skipping)",
                           comp.class_name, c.name)
            continue
        terms.append(c.sign * _zscore(scores[c.name]))
    if not terms:
        raise ValueError(f"No components found for composite {comp.class_name}")
    return sum(terms)


def _fit_interaction(df: pd.DataFrame, comp_col: str = "composite") -> dict:
    import statsmodels.api as sm
    d = df[["pCR", "arm", comp_col, "HR", "MP"]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(d) < 40 or d["arm"].nunique() < 2:
        return {"error": f"n={len(d)}"}
    covs = [c for c in ["HR", "MP"] if d[c].nunique() >= 2]
    d["score_z"] = _zscore(d[comp_col])
    d["interaction"] = d["score_z"] * d["arm"]
    X = sm.add_constant(d[["score_z", "arm", "interaction"] + covs])
    y = d["pCR"].astype(int)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = sm.Logit(y, X).fit(disp=False, method="bfgs", maxiter=300)
        return {
            "n": int(len(d)),
            "n_pcr": int(y.sum()),
            "beta_score": float(m.params["score_z"]),
            "beta_arm": float(m.params["arm"]),
            "beta_interaction": float(m.params["interaction"]),
            "or_interaction": float(np.exp(m.params["interaction"])),
            "or_ci_lo": float(np.exp(m.conf_int().loc["interaction", 0])),
            "or_ci_hi": float(np.exp(m.conf_int().loc["interaction", 1])),
            "p_interaction": float(m.pvalues["interaction"]),
        }
    except Exception as exc:
        return {"error": str(exc)}


def _tertile_effects(df: pd.DataFrame) -> pd.DataFrame:
    d = df[["composite", "arm", "pCR"]].dropna()
    d["tertile"] = pd.qcut(d["composite"], q=3, labels=["low", "med", "high"],
                            duplicates="drop")
    rows = []
    for t in d["tertile"].cat.categories:
        dd = d[d["tertile"] == t]
        c = dd[dd["arm"] == 0]["pCR"]
        tt = dd[dd["arm"] == 1]["pCR"]
        if len(c) < 3 or len(tt) < 3:
            continue
        a = tt.sum() + 0.5; b = (len(tt) - tt.sum()) + 0.5
        cc = c.sum() + 0.5; e = (len(c) - c.sum()) + 0.5
        or_val = (a / b) / (cc / e)
        se = float(np.sqrt(1/a + 1/b + 1/cc + 1/e))
        rows.append({
            "tertile": t,
            "n_ctrl": int(len(c)), "n_trt": int(len(tt)),
            "orr_ctrl": float(c.mean()), "orr_trt": float(tt.mean()),
            "arr": float(tt.mean() - c.mean()),
            "or": float(or_val),
            "or_ci_lo": float(np.exp(np.log(or_val) - 1.96 * se)),
            "or_ci_hi": float(np.exp(np.log(or_val) + 1.96 * se)),
        })
    return pd.DataFrame(rows)


def _bh(p: np.ndarray) -> np.ndarray:
    n = len(p)
    order = np.argsort(p)
    ranks = np.empty(n, dtype=float)
    ranks[order] = np.arange(1, n + 1)
    q = np.minimum(p * n / ranks, 1.0)
    q_sorted = q[order]
    for i in range(n - 2, -1, -1):
        if q_sorted[i] > q_sorted[i + 1]:
            q_sorted[i] = q_sorted[i + 1]
    q_out = np.empty(n, dtype=float)
    q_out[order] = q_sorted
    return q_out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gene-matrix",
                    default=str(_REPO / "data/raw/geo/GSE194040_gene_level.txt.gz"))
    ap.add_argument("--meta1",
                    default=str(_REPO / "data/raw/geo/GSE194040-GPL20078_series_matrix.txt.gz"))
    ap.add_argument("--meta2",
                    default=str(_REPO / "data/raw/geo/GSE194040-GPL30493_series_matrix.txt.gz"))
    ap.add_argument("--gmt", default=str(_REPO / "signatures" / "tme_signatures.gmt"))
    ap.add_argument("--control-arm", default="Paclitaxel")
    ap.add_argument("--out-dir",
                    default=str(_REPO / "results" / "targeted_composites"))
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    # Load cohort
    meta = pd.concat([_parse_meta(args.meta1).assign(platform="GPL20078"),
                      _parse_meta(args.meta2).assign(platform="GPL30493")])
    meta["resid"] = meta["patient id"].astype(str)
    meta["pcr_int"] = pd.to_numeric(meta["pcr"], errors="coerce")
    meta = meta.dropna(subset=["pcr_int"])
    gene = _load_gene(args.gene_matrix)
    meta = meta[meta["resid"].isin(gene.columns.astype(str))]
    logger.info("I-SPY2 full: %d samples with expression + pCR", len(meta))

    # Precompute all ssGSEA scores once
    gmt = load_gmt(args.gmt)
    scores_full = ssgsea_scores(gene[meta["resid"].tolist()], gene_sets=gmt)
    logger.info("ssGSEA computed for %d signatures × %d samples", *scores_full.shape[::-1])

    meta_i = meta.set_index("resid")

    # Control reference (Paclitaxel-only) in both platforms
    results_rows = []
    tertile_dfs = {}

    for comp in ALL_COMPOSITES:
        # select samples in control ∪ this class arms
        sel_labels = [args.control_arm] + comp.arm_labels
        sel = meta_i[meta_i["arm"].isin(sel_labels)].copy()
        if sel.empty:
            logger.warning("No samples for class %s", comp.class_name)
            continue
        sel["arm_bin"] = (sel["arm"] != args.control_arm).astype(int)
        n_ctrl = int((sel["arm_bin"] == 0).sum())
        n_trt = int((sel["arm_bin"] == 1).sum())
        if n_trt < 30:
            logger.warning("Class %s too small (N_trt=%d), skipping", comp.class_name, n_trt)
            continue

        # Composite on the subset (using full-data ssGSEA scores)
        comp_series = _compute_composite(scores_full.loc[sel.index], comp)
        df = pd.DataFrame(index=sel.index)
        df["composite"] = comp_series
        df["pCR"] = sel["pcr_int"].astype(int)
        df["arm"] = sel["arm_bin"]
        df["HR"] = pd.to_numeric(sel["hr"], errors="coerce")
        df["MP"] = pd.to_numeric(sel["mp"], errors="coerce")

        res = _fit_interaction(df)

        # baseline Fisher (arm only, no covariates)
        from scipy.stats import fisher_exact
        ct = pd.crosstab(df["arm"], df["pCR"])
        if ct.shape == (2, 2):
            or_base, p_base = fisher_exact(ct.values)
        else:
            or_base, p_base = float("nan"), float("nan")

        row = {
            "class_name": comp.class_name,
            "biology": comp.biology,
            "arm_labels": "; ".join(comp.arm_labels),
            "formula": comp.formula(),
            "n_ctrl": n_ctrl, "n_trt": n_trt,
            "pcr_ctrl": float(df[df["arm"] == 0]["pCR"].mean()),
            "pcr_trt": float(df[df["arm"] == 1]["pCR"].mean()),
            "baseline_OR": float(or_base), "baseline_p": float(p_base),
            **{k: v for k, v in res.items() if k != "error"},
        }
        results_rows.append(row)
        logger.info(
            "class=%s  N=%d (ctrl=%d, trt=%d)  baseline OR=%.2f p=%.3f | "
            "composite OR_int=%s p=%s",
            comp.class_name, len(df), n_ctrl, n_trt, or_base, p_base,
            f"{res.get('or_interaction', float('nan')):.3f}" if "or_interaction" in res else "—",
            f"{res.get('p_interaction', float('nan')):.4f}" if "p_interaction" in res else "—",
        )
        # Tertiles
        ter = _tertile_effects(df)
        ter["class_name"] = comp.class_name
        tertile_dfs[comp.class_name] = ter
        ter.to_csv(out / f"tertile_{comp.class_name.replace('+', '_').replace(' ', '_')}.csv",
                   index=False)

    # BH FDR across all class tests
    res_df = pd.DataFrame(results_rows)
    if not res_df.empty and "p_interaction" in res_df.columns:
        res_df["q_bh_class"] = _bh(res_df["p_interaction"].values)
    res_df.to_csv(out / "targeted_composites_results.csv", index=False)

    # Tertile concat
    if tertile_dfs:
        all_tertiles = pd.concat(tertile_dfs.values(), ignore_index=True)
        all_tertiles.to_csv(out / "all_tertile_effects.csv", index=False)

    # Print summary
    print("\n" + "=" * 110)
    print("TME × Targeted therapy class interaction — pre-specified composites (I-SPY2)")
    print("=" * 110)
    disp_cols = ["class_name", "n_ctrl", "n_trt", "pcr_ctrl", "pcr_trt",
                 "baseline_OR", "or_interaction", "or_ci_lo", "or_ci_hi",
                 "p_interaction", "q_bh_class"]
    print(res_df[disp_cols].to_string(
        index=False, float_format=lambda x: f"{x:.3f}" if pd.notna(x) else "—"))

    print("\n" + "=" * 110)
    print("Tertile ARR by class")
    print("=" * 110)
    for cls, ter in tertile_dfs.items():
        print(f"\n  {cls}:")
        disp = ter[["tertile", "n_ctrl", "n_trt", "orr_ctrl", "orr_trt", "arr", "or",
                    "or_ci_lo", "or_ci_hi"]]
        print(disp.to_string(index=False,
            float_format=lambda x: f"{x:.3f}" if pd.notna(x) else "—"))

    # VERDICT.md
    lines = [
        "# OncoTME — TME × targeted therapy class (I-SPY2)",
        "",
        "## Pre-specified composite биомаркеры",
        "",
        "Для каждого класса таргетной терапии composite TME score pre-specified на основе",
        "published biology (см. `src/oncotme/benchmark/targeted_composites.py` для citations).",
        "**One test per class** → BH FDR across 6 classes only, no 23×N hidden tests.",
        "",
        "## Composite formulas",
        "",
    ]
    for c in ALL_COMPOSITES:
        lines.append(f"- **{c.class_name}** [{c.biology}]:")
        lines.append(f"  ```\n  composite = {c.formula()}\n  ```")
        lines.append(f"  Rationale: {'; '.join(c.citations)}")
        lines.append("")

    lines += [
        "## Interaction results",
        "",
        "| Class | n_ctrl / n_trt | pCR ctrl → trt | OR_int [95% CI] | p_int | q_BH |",
        "|---|---|---|---|---|---|",
    ]
    for _, r in res_df.iterrows():
        or_int = r.get("or_interaction"); lo = r.get("or_ci_lo"); hi = r.get("or_ci_hi")
        p = r.get("p_interaction"); q = r.get("q_bh_class")
        marker = " ✅" if (pd.notna(q) and q < 0.10) else \
                 (" 🟡" if (pd.notna(p) and p < 0.05) else "")
        lines.append(
            f"| **{r['class_name']}**{marker} | "
            f"{int(r['n_ctrl'])} / {int(r['n_trt'])} | "
            f"{r['pcr_ctrl']:.1%} → {r['pcr_trt']:.1%} | "
            f"{or_int:.2f} [{lo:.2f}, {hi:.2f}] | "
            f"{p:.4f} | {q:.4f} |"
        )

    lines += ["", "## Tertile ARR per class", ""]
    for cls, ter in tertile_dfs.items():
        lines.append(f"\n### {cls}\n")
        lines.append("| Tertile | n_ctrl / n_trt | ORR ctrl | ORR trt | ARR | OR [95% CI] |")
        lines.append("|---|---|---|---|---|---|")
        for _, r in ter.iterrows():
            lines.append(
                f"| {r['tertile']} | {int(r['n_ctrl'])} / {int(r['n_trt'])} | "
                f"{r['orr_ctrl']:.1%} | {r['orr_trt']:.1%} | "
                f"**{r['arr']:+.1%}** | "
                f"{r['or']:.2f} [{r['or_ci_lo']:.2f}, {r['or_ci_hi']:.2f}] |"
            )

    # Auto-verdict
    lines += ["", "## Verdict", ""]
    sig_classes = res_df[res_df["p_interaction"] < 0.05].sort_values("p_interaction") \
        if "p_interaction" in res_df.columns else pd.DataFrame()
    bh_sig = res_df[res_df["q_bh_class"] < 0.10].sort_values("q_bh_class") \
        if "q_bh_class" in res_df.columns else pd.DataFrame()
    if not bh_sig.empty:
        lines.append(
            f"✅ **BH-significant ({len(bh_sig)} / {len(res_df)} classes)**: "
            + ", ".join([f"{r['class_name']} (q={r['q_bh_class']:.3f}, OR={r['or_interaction']:.2f})"
                        for _, r in bh_sig.iterrows()]) + "."
        )
    elif not sig_classes.empty:
        lines.append(
            f"🟡 **Nominally significant ({len(sig_classes)} / {len(res_df)})** before BH: "
            + ", ".join([f"{r['class_name']} (p={r['p_interaction']:.3f}, OR={r['or_interaction']:.2f})"
                        for _, r in sig_classes.iterrows()])
            + ". Replication on independent targeted-therapy cohorts required."
        )
    else:
        lines.append(
            "🔴 **No class** shows composite × arm interaction at nominal p<0.05. "
            "Either composite formulas need refinement, или TME не модифицирует ответ "
            "этих таргетных терапий в I-SPY2 (possibly due to small N in some arms)."
        )

    (out / "VERDICT.md").write_text("\n".join(lines), encoding="utf-8")

    with (out / "summary.json").open("w") as f:
        json.dump({
            "composites": [{
                "class_name": c.class_name, "formula": c.formula(),
                "biology": c.biology, "citations": c.citations,
            } for c in ALL_COMPOSITES],
            "results": res_df.to_dict(orient="records"),
        }, f, indent=2, default=str)

    logger.info("Artefacts → %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
