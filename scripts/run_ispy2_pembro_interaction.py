"""I-SPY2 Pembrolizumab × TME interaction — headline clinical analysis.

Central question: **which TME phenotype predicts pCR benefit от добавления
pembrolizumab к paclitaxel**?

Design:
- GSE194040 I-SPY2, GPL20078 platform
- Arm A: Paclitaxel alone (control, N=120)
- Arm B: Paclitaxel + Pembrolizumab (N=69)
- Endpoint: pCR

Steps:
1. Load gene-level expression + clinical metadata (series matrix)
2. Select pembro + control samples only (N≈189)
3. Compute ssGSEA TME panel (23 signatures)
4. Per signature: fit pCR ~ sig + arm + sig:arm + HR + MP_class + HER2
5. BH FDR for interactions
6. Subgroup ORR table + forest plot data
7. Sanity: reproduce baseline ORR effect (should match published ~+22%)
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
logger = logging.getLogger("ispy2_pembro")


def _parse_series_matrix_chars(path: str) -> pd.DataFrame:
    """Extract patient × characteristic table from I-SPY2 GSE194040 series matrix."""
    samples = None
    chars: dict[str, list] = {}
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
                if not kvs:
                    continue
                key = None
                for k, _ in kvs:
                    if k is not None:
                        key = k
                        break
                if key is None:
                    continue
                chars.setdefault(key, [v for _, v in kvs])
    df = pd.DataFrame(chars, index=samples)
    df.index.name = "gsm"
    return df


def _load_gene_matrix(path: str, sample_ids: list[str] | None = None) -> pd.DataFrame:
    """Load GSE194040 gene-level matrix.

    Формат файла: header строка — 988 ResID (patient) IDs; каждая data строка —
    gene symbol + 988 numeric values. Header has one fewer field than data row
    (header has no column label for the gene column). Используем ``index_col=0``,
    чтобы pandas корректно взял первую колонку data-строки в index.
    """
    logger.info("Loading gene matrix from %s …", path)
    df = pd.read_csv(path, sep="\t", low_memory=False, index_col=0)
    df = df.apply(pd.to_numeric, errors="coerce")
    # normalize index to upper-case HGNC symbols; drop NaN / duplicates
    df.index = df.index.astype(str).str.upper()
    df = df.loc[~df.index.duplicated(keep="first")]
    df = df.loc[df.index.notna() & (df.index != "NAN")]
    logger.info("Gene matrix: %d genes × %d samples", *df.shape)
    return df


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


def _fit_interaction(df: pd.DataFrame, sig_col: str, covariates: list[str]) -> dict:
    import statsmodels.api as sm
    d = df[["pCR", "arm", sig_col] + covariates].copy().apply(pd.to_numeric, errors="coerce")
    d = d.dropna()
    if d.shape[0] < 50:
        return {"sig": sig_col, "error": f"n={d.shape[0]}"}
    d[sig_col] = (d[sig_col] - d[sig_col].mean()) / d[sig_col].std()
    d["interaction"] = d[sig_col] * d["arm"]
    X = sm.add_constant(d[[sig_col, "arm", "interaction"] + covariates])
    y = d["pCR"].astype(int)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = sm.Logit(y, X).fit(disp=False, method="bfgs", maxiter=300)
        return {
            "sig": sig_col,
            "n": int(d.shape[0]),
            "n_pcr": int(y.sum()),
            "beta_sig": float(m.params[sig_col]),
            "beta_arm": float(m.params["arm"]),
            "beta_interaction": float(m.params["interaction"]),
            "or_interaction": float(np.exp(m.params["interaction"])),
            "p_interaction": float(m.pvalues["interaction"]),
            "p_sig": float(m.pvalues[sig_col]),
            "p_arm": float(m.pvalues["arm"]),
        }
    except Exception as exc:
        return {"sig": sig_col, "error": str(exc)}


def _subgroup_stats(df: pd.DataFrame, sig_col: str, split: str = "median") -> dict:
    d = df[[sig_col, "arm", "pCR"]].dropna()
    cut = d[sig_col].median() if split == "median" else d[sig_col].quantile(0.67)
    d["sub"] = np.where(d[sig_col] >= cut, "high", "low")
    out = {"sig": sig_col, "cut": float(cut)}
    for sub in ["low", "high"]:
        dd = d[d["sub"] == sub]
        c = dd[dd["arm"] == 0]; t = dd[dd["arm"] == 1]
        if len(c) < 5 or len(t) < 5:
            continue
        orr_c = float(c["pCR"].mean()); orr_t = float(t["pCR"].mean())
        # Haldane-Anscombe
        a = t["pCR"].sum() + 0.5; b = (len(t) - t["pCR"].sum()) + 0.5
        cc = c["pCR"].sum() + 0.5; e = (len(c) - c["pCR"].sum()) + 0.5
        or_val = float((a / b) / (cc / e))
        se_log_or = float(np.sqrt(1/a + 1/b + 1/cc + 1/e))
        ci_lo = float(np.exp(np.log(or_val) - 1.96 * se_log_or))
        ci_hi = float(np.exp(np.log(or_val) + 1.96 * se_log_or))
        out[sub] = {
            "n_ctrl": int(len(c)), "n_trt": int(len(t)),
            "orr_ctrl": orr_c, "orr_trt": orr_t,
            "arr": orr_t - orr_c,
            "or": or_val, "or_ci_lo": ci_lo, "or_ci_hi": ci_hi,
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gene-matrix",
                    default=str(_REPO / "data/raw/geo/GSE194040_gene_level.txt.gz"))
    ap.add_argument("--series-matrix-gpl20078",
                    default=str(_REPO / "data/raw/geo/GSE194040-GPL20078_series_matrix.txt.gz"))
    ap.add_argument("--series-matrix-gpl30493",
                    default=str(_REPO / "data/raw/geo/GSE194040-GPL30493_series_matrix.txt.gz"))
    ap.add_argument("--gmt", default=str(_REPO / "signatures" / "tme_signatures.gmt"))
    ap.add_argument("--treatment-arm", default="Paclitaxel + Pembrolizumab",
                    help="Experimental arm (string match in 'arm' metadata).")
    ap.add_argument("--control-arm", default="Paclitaxel")
    ap.add_argument("--out-dir",
                    default=str(_REPO / "results" / "ispy2_pembro"))
    args = ap.parse_args()
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Metadata
    meta1 = _parse_series_matrix_chars(args.series_matrix_gpl20078)
    meta2 = _parse_series_matrix_chars(args.series_matrix_gpl30493)
    meta = pd.concat([meta1.assign(platform="GPL20078"),
                      meta2.assign(platform="GPL30493")])
    logger.info("Combined metadata: %d samples, arms: %s",
                len(meta), sorted(meta["arm"].unique()))

    # 2. Выбираем целевые arms
    mask = meta["arm"].isin([args.control_arm, args.treatment_arm])
    sub = meta.loc[mask].copy()
    sub["arm_bin"] = (sub["arm"] == args.treatment_arm).astype(int)
    sub["pcr_int"] = pd.to_numeric(sub["pcr"], errors="coerce")
    sub = sub.dropna(subset=["pcr_int"])
    logger.info("Selected arms — control='%s' (%d), trt='%s' (%d); both with pCR: %d",
                args.control_arm, (sub["arm_bin"] == 0).sum(),
                args.treatment_arm, (sub["arm_bin"] == 1).sum(),
                len(sub))
    logger.info("pCR rates: control %.1f%%, trt %.1f%% (Δ = %+.1f%%)",
                100 * sub[sub["arm_bin"] == 0]["pcr_int"].mean(),
                100 * sub[sub["arm_bin"] == 1]["pcr_int"].mean(),
                100 * (sub[sub["arm_bin"] == 1]["pcr_int"].mean() -
                       sub[sub["arm_bin"] == 0]["pcr_int"].mean()))

    # 3. Load gene matrix. patient ids in header = ResID; need mapping via meta['patient id']
    gene = _load_gene_matrix(args.gene_matrix)
    sub["resid"] = sub["patient id"].astype(str)
    gene_cols_avail = set(gene.columns.astype(str))
    sub = sub[sub["resid"].isin(gene_cols_avail)]
    logger.info("After matching to gene matrix: N=%d", len(sub))

    # 4. ssGSEA
    gmt = load_gmt(args.gmt)
    expr_sub = gene[sub["resid"].tolist()]
    logger.info("Running ssGSEA on %d samples × %d genes …", *expr_sub.shape[::-1])
    scores = ssgsea_scores(expr_sub, gene_sets=gmt)
    scores.index = scores.index.astype(str)
    scores.columns = [f"sig__{c}" for c in scores.columns]
    # attach clinical: map resid → signature row
    sub_i = sub.set_index("resid")
    scores["pCR"] = pd.to_numeric(sub_i.loc[scores.index, "pcr_int"], errors="coerce")
    scores["arm"] = sub_i.loc[scores.index, "arm_bin"].astype(int)
    scores["HR"] = pd.to_numeric(sub_i.loc[scores.index, "hr"], errors="coerce")
    scores["HER2"] = pd.to_numeric(sub_i.loc[scores.index, "her2"], errors="coerce")
    scores["MP"] = pd.to_numeric(sub_i.loc[scores.index, "mp"], errors="coerce")
    scores = scores.dropna(subset=["pCR", "arm"])
    logger.info("Feature frame shape: %s", scores.shape)

    # 5. Baseline Fisher
    from scipy.stats import fisher_exact
    ct = pd.crosstab(scores["arm"], scores["pCR"])
    or_base, p_base = fisher_exact(ct.values)
    logger.info("Baseline OR (trt vs ctrl) = %.3f, Fisher p = %.4f", or_base, p_base)

    # 6. Interaction per signature
    sig_cols = [c for c in scores.columns if c.startswith("sig__")]
    covariates = ["HR", "MP"]
    rows = [_fit_interaction(scores, s, covariates) for s in sig_cols]
    df_int = pd.DataFrame([r for r in rows if "p_interaction" in r])
    df_int = df_int.sort_values("p_interaction")
    df_int["q_interaction"] = _bh(df_int["p_interaction"].values)
    df_int.to_csv(out_dir / "interaction_by_signature.csv", index=False)

    print("\n" + "=" * 90)
    print(f"I-SPY2: {args.treatment_arm} vs {args.control_arm}")
    print("per-signature arm × TME interaction (pCR ~ sig + arm + sig:arm + HR + MP)")
    print("=" * 90)
    print(df_int[["sig", "n", "n_pcr", "beta_interaction", "or_interaction",
                  "p_interaction", "q_interaction"]].to_string(
        index=False, float_format=lambda x: f"{x:.4f}" if pd.notna(x) else "—"))

    # 7. Subgroup for interaction p<0.10
    sig_significant = df_int[df_int["p_interaction"] < 0.10]
    sub_rows = []
    for sig in sig_significant["sig"]:
        st = _subgroup_stats(scores, sig)
        if "low" in st and "high" in st:
            sub_rows.append({
                "signature": sig,
                "cut": st["cut"],
                "low_n_ctrl": st["low"]["n_ctrl"], "low_n_trt": st["low"]["n_trt"],
                "low_orr_ctrl": st["low"]["orr_ctrl"], "low_orr_trt": st["low"]["orr_trt"],
                "low_ARR": st["low"]["arr"],
                "low_OR": st["low"]["or"],
                "low_OR_ci_lo": st["low"]["or_ci_lo"], "low_OR_ci_hi": st["low"]["or_ci_hi"],
                "high_n_ctrl": st["high"]["n_ctrl"], "high_n_trt": st["high"]["n_trt"],
                "high_orr_ctrl": st["high"]["orr_ctrl"], "high_orr_trt": st["high"]["orr_trt"],
                "high_ARR": st["high"]["arr"],
                "high_OR": st["high"]["or"],
                "high_OR_ci_lo": st["high"]["or_ci_lo"], "high_OR_ci_hi": st["high"]["or_ci_hi"],
                "delta_ARR": st["high"]["arr"] - st["low"]["arr"],
                "delta_OR": st["high"]["or"] - st["low"]["or"],
            })
    sub_df = pd.DataFrame(sub_rows).sort_values("delta_ARR", ascending=False) \
        if sub_rows else pd.DataFrame()
    sub_df.to_csv(out_dir / "subgroup_orr.csv", index=False)

    print("\n" + "=" * 90)
    print("Subgroup ORR for interaction p < 0.10 (sorted by Δ ARR: high − low)")
    print("=" * 90)
    if not sub_df.empty:
        display = [
            "signature",
            "low_orr_ctrl", "low_orr_trt", "low_ARR",
            "high_orr_ctrl", "high_orr_trt", "high_ARR",
            "delta_ARR",
        ]
        print(sub_df[display].to_string(
            index=False, float_format=lambda x: f"{x:.3f}" if pd.notna(x) else "—"))
    else:
        print("no significant interactions")

    # 8. Verdict
    best_p = df_int["p_interaction"].min() if not df_int.empty else 1.0
    best_q = df_int["q_interaction"].min() if not df_int.empty else 1.0
    top = df_int.iloc[0] if not df_int.empty else None

    lines = [
        f"# I-SPY2 TME × Pembrolizumab interaction",
        "",
        f"- Arms: **{args.control_arm}** (N={int((scores['arm']==0).sum())}) vs "
        f"**{args.treatment_arm}** (N={int((scores['arm']==1).sum())})",
        f"- pCR: ctrl {100*scores[scores['arm']==0]['pCR'].mean():.1f}%, "
        f"trt {100*scores[scores['arm']==1]['pCR'].mean():.1f}% "
        f"(Δ = {100*(scores[scores['arm']==1]['pCR'].mean() - scores[scores['arm']==0]['pCR'].mean()):+.1f}%)",
        f"- Baseline OR = {or_base:.3f}, Fisher p = {p_base:.4f}",
        "",
        "## Top 10 TME × arm interactions (BH-adjusted)",
        "",
        "| Signature | n | OR_int | p | q |",
        "|---|---|---|---|---|",
    ]
    for _, r in df_int.head(10).iterrows():
        lines.append(
            f"| {r['sig']} | {int(r['n'])} | {r['or_interaction']:.3f} | "
            f"{r['p_interaction']:.4f} | {r['q_interaction']:.4f} |"
        )
    if not sub_df.empty:
        lines += [
            "",
            "## Subgroup benefit (median-dichotomized for p<0.10 signatures)",
            "",
            "| Signature | Low ctrl→trt (ARR) | High ctrl→trt (ARR) | Δ ARR |",
            "|---|---|---|---|",
        ]
        for _, r in sub_df.iterrows():
            lines.append(
                f"| {r['signature']} | "
                f"{r['low_orr_ctrl']:.1%}→{r['low_orr_trt']:.1%} ({r['low_ARR']:+.1%}) | "
                f"{r['high_orr_ctrl']:.1%}→{r['high_orr_trt']:.1%} ({r['high_ARR']:+.1%}) | "
                f"**{r['delta_ARR']:+.3f}** |"
            )

    lines += ["", "## Verdict", ""]
    if best_q < 0.10:
        lines.append(
            f"✅ **Clinically actionable**: min BH-q = {best_q:.4f} "
            f"(signature = {top['sig']}). Pembro benefit is heterogeneous by TME — "
            f"precision biomarker candidate."
        )
    elif best_p < 0.01:
        lines.append(
            f"🟡 **Strong hypothesis-generating signal**: min p = {best_p:.4f} "
            f"(signature = {top['sig']}, OR_int={top['or_interaction']:.2f}). "
            f"Does not survive BH with {len(df_int)} tests; replicate on KEYNOTE-522, "
            f"TONIC, or other pembro cohorts."
        )
    elif best_p < 0.05:
        lines.append(
            f"🟡 **Nominal signal**: min p = {best_p:.4f}. "
            f"Report as exploratory; requires large prospective cohort."
        )
    else:
        lines.append(
            f"🔴 **No interaction detected** on I-SPY2 alone (min p = {best_p:.4f}). "
            f"Pembro benefit appears homogeneous within this TME panel."
        )
    (out_dir / "VERDICT.md").write_text("\n".join(lines), encoding="utf-8")
    logger.info("Artefacts → %s", out_dir)

    summary = {
        "arm_trt": args.treatment_arm, "arm_ctrl": args.control_arm,
        "n_ctrl": int((scores["arm"] == 0).sum()),
        "n_trt": int((scores["arm"] == 1).sum()),
        "pcr_ctrl": float(scores[scores["arm"] == 0]["pCR"].mean()),
        "pcr_trt": float(scores[scores["arm"] == 1]["pCR"].mean()),
        "baseline_or": float(or_base), "baseline_p": float(p_base),
        "min_interaction_p": float(best_p),
        "min_interaction_q": float(best_q),
        "top_signature": str(top["sig"]) if top is not None else None,
        "interactions": df_int.to_dict(orient="records"),
        "subgroups": sub_df.to_dict(orient="records") if not sub_df.empty else [],
    }
    with (out_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2, default=str)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
