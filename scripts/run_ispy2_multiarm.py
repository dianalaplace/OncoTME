"""Treatment × TME interaction spectrum — all I-SPY2 targeted therapy arms.

Центральный клинический вопрос: какой TME-профиль предсказывает benefit от каждого
класса таргетной терапии, и **специфичен ли composite ICI-TME score к ICI** или
работает шире?

Arms (все vs Paclitaxel control):
- **anti-PD-1**:      Paclitaxel + Pembrolizumab
- **PARPi + platinum**: Paclitaxel + ABT-888 + Carboplatin
- **anti-HER2-TKI**:  Paclitaxel + Neratinib
- **HER2 ADC**:       T-DM1 + Pertuzumab (HER2+ only)
- **anti-HER2 mAb**:  Paclitaxel + Trastuzumab (+ Pertuzumab variants)
- **anti-IGF-1R**:    Paclitaxel + Ganitumab
- **HSP90i**:         Paclitaxel + Ganetespib
- **AKTi**:           Paclitaxel + MK-2206
- **anti-Angiopoietin**: Paclitaxel + AMG-386

Per-arm analysis:
1. Composite × arm interaction (pre-specified ICI-readiness score).
2. Каждая individual TME signature × arm (exploratory).
3. Subgroup pCR-rate per tertile of composite.
4. Compare: который arm имеет самую сильную TME-interaction, с какой сигнатурой.

Output: one multi-arm forest table + per-arm VERDICT.md.
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

from oncotme.features.signatures import load_gmt, ssgsea_scores

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("multiarm")


POSITIVE_COMPONENTS = ["IFN_gamma", "HLA_I_score", "B_cell", "checkpoint_score"]
NEGATIVE_COMPONENTS = ["CAF_proxy"]

# Какие arms включать: name → (matching_strings, biological_class, targeted_class)
ARMS_TO_TEST = {
    "Pembrolizumab": (["Paclitaxel + Pembrolizumab"], "anti-PD1", "ICI"),
    "ABT888+Carboplatin": (["Paclitaxel + ABT 888 + Carboplatin"], "PARPi+Platinum", "PARPi"),
    "Neratinib": (["Paclitaxel + Neratinib"], "pan-HER TKI", "HER-kinase"),
    "T-DM1+Pertuzumab": (["T-DM1 + Pertuzumab"], "HER2 ADC + mAb", "HER2"),
    "Trastuzumab-based": (
        ["Paclitaxel + Trastuzumab",
         "Paclitaxel + Pertuzumab + Trastuzumab",
         "Paclitaxel + AMG 386 + Trastuzumab",
         "Paclitaxel + MK-2206 + Trastuzumab"],
        "HER2 mAb combos", "HER2",
    ),
    "Ganitumab": (["Paclitaxel + Ganitumab"], "anti-IGF-1R mAb", "GF-receptor"),
    "Ganetespib": (["Paclitaxel + Ganetespib"], "HSP90 inhibitor", "chaperone"),
    "MK-2206": (["Paclitaxel + MK-2206"], "AKT inhibitor", "PI3K-AKT"),
    "AMG-386": (["Paclitaxel + AMG 386", "Paclitaxel + AMG-386"],
                 "anti-Angiopoietin-1/2", "angiogenesis"),
}


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


def _zscore(s):
    std = s.std()
    return (s - s.mean()) / (std if std else 1.0)


def _fit_interaction(df: pd.DataFrame, score_col: str) -> dict:
    """pCR ~ score_z + arm + interaction + HR + MP. Single arm vs Paclitaxel."""
    import statsmodels.api as sm
    d = df[["pCR", "arm", score_col, "HR", "MP"]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(d) < 40 or d["arm"].nunique() < 2:
        return {"error": f"n={len(d)}"}
    # keep HR/MP only if they vary
    covs = ["HR", "MP"]
    covs = [c for c in covs if d[c].nunique() >= 2]
    d["score_z"] = _zscore(d[score_col])
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
            "p_arm": float(m.pvalues["arm"]),
            "p_score": float(m.pvalues["score_z"]),
        }
    except Exception as exc:
        return {"error": str(exc)}


def _fisher_arm_only(df: pd.DataFrame) -> dict:
    from scipy.stats import fisher_exact
    ct = pd.crosstab(df["arm"], df["pCR"].astype(int))
    if ct.shape != (2, 2):
        return {"or": float("nan"), "p": float("nan")}
    or_val, p = fisher_exact(ct.values)
    return {"or": float(or_val), "p": float(p)}


def _tertile_effects(df: pd.DataFrame, score_col: str) -> pd.DataFrame:
    d = df[[score_col, "arm", "pCR"]].dropna()
    d["tertile"] = pd.qcut(d[score_col], q=3, labels=["low", "med", "high"])
    rows = []
    for t in ["low", "med", "high"]:
        dd = d[d["tertile"] == t]
        c = dd[dd["arm"] == 0]["pCR"]
        tt = dd[dd["arm"] == 1]["pCR"]
        if len(c) < 3 or len(tt) < 3:
            continue
        a = tt.sum() + 0.5; b = (len(tt) - tt.sum()) + 0.5
        cc = c.sum() + 0.5; e = (len(c) - c.sum()) + 0.5
        or_val = (a / b) / (cc / e)
        se = float(np.sqrt(1/a + 1/b + 1/cc + 1/e))
        ci_lo = float(np.exp(np.log(or_val) - 1.96 * se))
        ci_hi = float(np.exp(np.log(or_val) + 1.96 * se))
        rows.append({
            "tertile": t,
            "n_ctrl": int(len(c)), "n_trt": int(len(tt)),
            "orr_ctrl": float(c.mean()), "orr_trt": float(tt.mean()),
            "arr": float(tt.mean() - c.mean()),
            "or": float(or_val), "or_ci_lo": ci_lo, "or_ci_hi": ci_hi,
        })
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gene-matrix",
                    default=str(_REPO / "data/raw/geo/GSE194040_gene_level.txt.gz"))
    ap.add_argument("--meta1",
                    default=str(_REPO / "data/raw/geo/GSE194040-GPL20078_series_matrix.txt.gz"))
    ap.add_argument("--meta2",
                    default=str(_REPO / "data/raw/geo/GSE194040-GPL30493_series_matrix.txt.gz"))
    ap.add_argument("--gmt", default=str(_REPO / "signatures" / "tme_signatures.gmt"))
    ap.add_argument("--out-dir",
                    default=str(_REPO / "results" / "ispy2_multiarm"))
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    # Load metadata + gene
    meta = pd.concat([_parse_meta(args.meta1).assign(platform="GPL20078"),
                      _parse_meta(args.meta2).assign(platform="GPL30493")])
    meta["resid"] = meta["patient id"].astype(str)
    meta["pcr_int"] = pd.to_numeric(meta["pcr"], errors="coerce")
    meta = meta.dropna(subset=["pcr_int"])
    gene = _load_gene(args.gene_matrix)
    meta = meta[meta["resid"].isin(gene.columns.astype(str))]
    logger.info("N with pCR + expression: %d", len(meta))

    # ssGSEA across all samples once
    gmt = load_gmt(args.gmt)
    scores = ssgsea_scores(gene[meta["resid"].tolist()], gene_sets=gmt)

    # Composite
    composite = sum(_zscore(scores[c]) for c in POSITIVE_COMPONENTS) \
                - sum(_zscore(scores[c]) for c in NEGATIVE_COMPONENTS)
    composite.name = "composite"

    # Assemble per-sample frame
    meta_i = meta.set_index("resid")
    base_df = pd.DataFrame(index=scores.index)
    base_df["pCR"] = meta_i.loc[base_df.index, "pcr_int"].values
    base_df["HR"] = pd.to_numeric(meta_i.loc[base_df.index, "hr"], errors="coerce").values
    base_df["HER2"] = pd.to_numeric(meta_i.loc[base_df.index, "her2"], errors="coerce").values
    base_df["MP"] = pd.to_numeric(meta_i.loc[base_df.index, "mp"], errors="coerce").values
    base_df["arm_label"] = meta_i.loc[base_df.index, "arm"].values
    base_df["composite"] = composite.reindex(base_df.index).values
    # каждая individual signature (для exploratory arm-specific biology)
    for c in scores.columns:
        base_df[f"sig_{c}"] = scores[c].reindex(base_df.index).values

    # Per-arm analysis
    multiarm_rows = []
    all_sig_rows = []
    for arm_name, (labels, bio_class, targeted_cls) in ARMS_TO_TEST.items():
        sel = base_df[base_df["arm_label"].isin(["Paclitaxel"] + list(labels))].copy()
        if sel.empty or sel["arm_label"].nunique() < 2:
            continue
        sel["arm"] = (sel["arm_label"] != "Paclitaxel").astype(int)
        n_ctrl = int((sel["arm"] == 0).sum())
        n_trt = int((sel["arm"] == 1).sum())
        if n_trt < 30:
            continue

        # unadjusted arm effect
        fisher_res = _fisher_arm_only(sel)

        # Composite interaction
        comp_res = _fit_interaction(sel, "composite")
        # Individual signature interactions (exploratory)
        sig_rows = []
        for sig in scores.columns:
            col = f"sig_{sig}"
            if col not in sel.columns:
                continue
            r = _fit_interaction(sel, col)
            if "p_interaction" in r:
                r.update({"arm_name": arm_name, "signature": sig,
                          "targeted_class": targeted_cls})
                sig_rows.append(r)
        sig_df = pd.DataFrame(sig_rows)

        logger.info(
            "arm=%s  N=%d (ctrl=%d, trt=%d)  baseline OR=%.2f p=%.4f | "
            "composite OR_int=%s p_int=%s",
            arm_name, len(sel), n_ctrl, n_trt,
            fisher_res.get("or", float('nan')), fisher_res.get("p", float('nan')),
            f"{comp_res.get('or_interaction', float('nan')):.3f}"
            if "or_interaction" in comp_res else "—",
            f"{comp_res.get('p_interaction', float('nan')):.4f}"
            if "p_interaction" in comp_res else "—",
        )

        row = {
            "arm_name": arm_name,
            "biology": bio_class,
            "targeted_class": targeted_cls,
            "n_ctrl": n_ctrl, "n_trt": n_trt,
            "pcr_ctrl": float(sel[sel["arm"] == 0]["pCR"].mean()),
            "pcr_trt": float(sel[sel["arm"] == 1]["pCR"].mean()),
            "baseline_OR": fisher_res.get("or"),
            "baseline_p": fisher_res.get("p"),
            **{f"composite_{k}": v for k, v in comp_res.items() if k != "error"},
        }
        multiarm_rows.append(row)

        # Tertile effects за composite
        ter = _tertile_effects(sel, "composite")
        ter.to_csv(out / f"tertile_composite_{arm_name}.csv", index=False)

        # Save sig-level for this arm
        if not sig_df.empty:
            sig_df = sig_df.sort_values("p_interaction")
            sig_df.to_csv(out / f"signatures_{arm_name}.csv", index=False)
            all_sig_rows.append(sig_df.assign(arm_name=arm_name))

    multi_df = pd.DataFrame(multiarm_rows).sort_values("composite_p_interaction")
    multi_df.to_csv(out / "multiarm_composite.csv", index=False)

    # Concatenate per-sig
    all_sigs = pd.concat(all_sig_rows, ignore_index=True) if all_sig_rows else pd.DataFrame()
    if not all_sigs.empty:
        # BH correction across ALL arm × sig tests (strict)
        from scipy.stats import rankdata
        p = all_sigs["p_interaction"].values
        n = len(p)
        ranks = rankdata(p, method="min")
        q = np.minimum(p * n / ranks, 1.0)
        # monotonicity
        order = np.argsort(p)
        q_sorted = q[order]
        for i in range(n - 2, -1, -1):
            if q_sorted[i] > q_sorted[i + 1]:
                q_sorted[i] = q_sorted[i + 1]
        q_out = np.empty(n, dtype=float)
        q_out[order] = q_sorted
        all_sigs["q_global_bh"] = q_out
        all_sigs.to_csv(out / "signatures_all_arms.csv", index=False)

    # ====== Print multi-arm table ==========================================
    print("\n" + "=" * 100)
    print("Multi-arm spectrum: Composite ICI-TME score × targeted therapy arm")
    print("=" * 100)
    if not multi_df.empty:
        disp_cols = ["arm_name", "biology", "n_ctrl", "n_trt", "pcr_ctrl", "pcr_trt",
                     "baseline_OR", "baseline_p",
                     "composite_or_interaction", "composite_p_interaction"]
        disp = multi_df[disp_cols].copy()
        print(disp.to_string(index=False,
            float_format=lambda x: f"{x:.3f}" if pd.notna(x) else "—"))

    # ====== Per-arm top signature ===========================================
    if not all_sigs.empty:
        print("\n" + "=" * 100)
        print("Per-arm: top signature by interaction p")
        print("=" * 100)
        for arm in multi_df["arm_name"]:
            sub = all_sigs[all_sigs["arm_name"] == arm].sort_values("p_interaction").head(3)
            if sub.empty:
                continue
            print(f"\n  Arm: {arm}")
            for _, r in sub.iterrows():
                print(f"    {r['signature']:20s}  OR_int={r['or_interaction']:.3f}  "
                      f"p={r['p_interaction']:.4f}  q_global={r['q_global_bh']:.3f}")

    # ====== VERDICT.md =====================================================
    lines = [
        "# I-SPY2 targeted therapy × TME composite score spectrum",
        "",
        "## Composite definition (pre-specified ICI biology)",
        "",
        "```",
        "composite = z(IFN_gamma) + z(HLA_I_score) + z(B_cell) + z(checkpoint_score)",
        "           - z(CAF_proxy)",
        "```",
        "",
        "## Treatment arm × composite interaction (primary hypothesis per arm)",
        "",
        "| Arm | Biology | n_ctrl / n_trt | pCR_ctrl → pCR_trt | OR_interaction [95% CI] | p_int |",
        "|---|---|---|---|---|---|",
    ]
    for _, r in multi_df.iterrows():
        or_int = r.get("composite_or_interaction")
        ci_lo = r.get("composite_or_ci_lo")
        ci_hi = r.get("composite_or_ci_hi")
        p_int = r.get("composite_p_interaction")
        lines.append(
            f"| **{r['arm_name']}** | {r['biology']} | "
            f"{int(r['n_ctrl'])} / {int(r['n_trt'])} | "
            f"{r['pcr_ctrl']:.1%} → {r['pcr_trt']:.1%} | "
            f"{or_int:.2f} [{ci_lo:.2f}, {ci_hi:.2f}] | "
            f"**{p_int:.4f}** |"
        )

    lines += ["", "## Key clinical take-aways", ""]
    # Самый сильный arm
    if not multi_df.empty:
        top_arm = multi_df.iloc[0]
        lines.append(
            f"1. **Strongest composite × therapy interaction**: {top_arm['arm_name']} "
            f"({top_arm['biology']}), OR_int = {top_arm['composite_or_interaction']:.2f}, "
            f"p = {top_arm['composite_p_interaction']:.4f}."
        )
        # Specificity: if pembro >> other arms, the biomarker is ICI-specific
        pembro_row = multi_df[multi_df["arm_name"] == "Pembrolizumab"]
        if not pembro_row.empty:
            other = multi_df[multi_df["arm_name"] != "Pembrolizumab"]
            if not other.empty:
                pembro_p = float(pembro_row["composite_p_interaction"].iloc[0])
                min_other_p = float(other["composite_p_interaction"].min())
                lines.append(
                    f"2. **ICI-specificity**: Pembrolizumab interaction p = {pembro_p:.4f}; "
                    f"next-most-significant non-ICI arm has p = {min_other_p:.4f}."
                )
                if pembro_p < 0.05 and min_other_p > 0.20:
                    lines.append(
                        "   → composite is a **specific ICI biomarker**, not a generic "
                        "response predictor. This is exactly the precision medicine profile."
                    )
                elif pembro_p < 0.05 and min_other_p < 0.10:
                    lines.append(
                        "   → composite shows interaction with multiple targeted classes; "
                        "interpret as **general immune-permissive TME** biomarker."
                    )

    # Per-arm top signatures
    if not all_sigs.empty:
        lines += ["", "## Arm-specific dominant TME signatures (top-1 each)", ""]
        lines.append("| Arm | Top signature | OR_int | p | q_global (BH across all tests) |")
        lines.append("|---|---|---|---|---|")
        for arm in multi_df["arm_name"]:
            sub = all_sigs[all_sigs["arm_name"] == arm].sort_values("p_interaction").head(1)
            if sub.empty:
                continue
            r = sub.iloc[0]
            lines.append(
                f"| {arm} | {r['signature']} | {r['or_interaction']:.2f} | "
                f"{r['p_interaction']:.4f} | {r['q_global_bh']:.4f} |"
            )

    (out / "VERDICT.md").write_text("\n".join(lines), encoding="utf-8")

    with (out / "summary.json").open("w") as f:
        json.dump({
            "n_total": int(len(base_df)),
            "arms": multi_df.to_dict(orient="records"),
        }, f, indent=2, default=str)

    logger.info("Artefacts → %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
