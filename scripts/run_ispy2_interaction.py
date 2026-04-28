"""Treatment × TME interaction analysis в I-SPY2 durvalumab+olaparib arm vs control.

Клинический вопрос: **у каких пациенток по TME-profile добавление
durvalumab+olaparib к paclitaxel даёт benefit в pCR**?

Approach:
1. Load I-SPY2 biomarker CSV (готовые TME signatures, arm labels, pCR).
2. Fit logistic regression per signature:
     pCR ~ sig + arm + sig:arm + HR + MP_class
   Extract interaction β and p-value.
3. BH FDR across signatures.
4. For signatures с q<0.10:
   - Dichotomize at median → ORR(arm, subgroup), OR(durva vs control | subgroup)
   - Absolute risk reduction (ARR) per subgroup
5. Save forest-plot data, VERDICT.md.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("ispy2_interaction")


TME_SIGNATURES = [
    "PD1", "PDL1", "T.cells_sig", "B.cells_sig", "Dendritic.cells_sig",
    "Mast.cells_sig", "CD68", "TIS_sig", "STAT1_sig",
    "TAMsurr_TcClassII_ratio_sig", "PARPi7_sig.", "Mitotic_sig",
    "ESR1_PGR_ave", "SET.index",
]


def _benjamini_hochberg(p: np.ndarray) -> np.ndarray:
    n = len(p)
    order = np.argsort(p)
    ranks = np.empty(n, dtype=float)
    ranks[order] = np.arange(1, n + 1)
    q = p * n / ranks
    q = np.minimum(q, 1.0)
    # enforce monotonicity after sort
    q_sorted = q[order]
    for i in range(n - 2, -1, -1):
        if q_sorted[i] > q_sorted[i + 1]:
            q_sorted[i] = q_sorted[i + 1]
    q_out = np.empty(n, dtype=float)
    q_out[order] = q_sorted
    return q_out


def _fit_interaction(df: pd.DataFrame, sig_col: str,
                     covariates: list[str]) -> dict:
    """Logistic regression: pCR ~ sig + arm + sig:arm + covariates.

    Returns beta, se, z, p for interaction term.
    """
    import statsmodels.api as sm
    d = df[["pCR", "arm", sig_col] + covariates].copy().dropna()
    if d.shape[0] < 30:
        return {"error": f"n={d.shape[0]} < 30"}
    # standardize signature and covariates for interpretability
    d[sig_col] = (d[sig_col] - d[sig_col].mean()) / d[sig_col].std()
    d["interaction"] = d[sig_col] * d["arm"]
    X = d[[sig_col, "arm", "interaction"] + covariates]
    X = sm.add_constant(X)
    y = d["pCR"].astype(int)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = sm.Logit(y, X).fit(disp=False, method="bfgs", maxiter=200)
        inter_beta = float(model.params["interaction"])
        inter_se = float(model.bse["interaction"])
        inter_p = float(model.pvalues["interaction"])
        # for inspection:
        sig_beta = float(model.params[sig_col])
        arm_beta = float(model.params["arm"])
        return {
            "sig_col": sig_col,
            "n": int(d.shape[0]),
            "n_pcr": int(y.sum()),
            "sig_beta": sig_beta,
            "arm_beta": arm_beta,
            "interaction_beta": inter_beta,
            "interaction_se": inter_se,
            "interaction_OR": float(np.exp(inter_beta)),
            "interaction_p": inter_p,
            "converged": bool(model.mle_retvals.get("converged", True)),
        }
    except Exception as exc:
        return {"sig_col": sig_col, "error": str(exc)}


def _subgroup_stats(df: pd.DataFrame, sig_col: str) -> dict:
    """По медиане разбиваем на low/high — ORR по arm × subgroup + OR + ARR."""
    d = df[[sig_col, "arm", "pCR"]].dropna()
    cut = d[sig_col].median()
    d["sub"] = np.where(d[sig_col] >= cut, "high", "low")
    out = {"sig": sig_col, "median_cut": float(cut)}
    for sub in ["low", "high"]:
        dd = d[d["sub"] == sub]
        ctrl = dd[dd["arm"] == 0]
        trt = dd[dd["arm"] == 1]
        if len(ctrl) < 5 or len(trt) < 5:
            continue
        orr_c = float(ctrl["pCR"].mean())
        orr_t = float(trt["pCR"].mean())
        arr = orr_t - orr_c
        # OR with Haldane-Anscombe correction
        a = trt["pCR"].sum() + 0.5
        b = (len(trt) - trt["pCR"].sum()) + 0.5
        c = ctrl["pCR"].sum() + 0.5
        e = (len(ctrl) - ctrl["pCR"].sum()) + 0.5
        odds_t = a / b
        odds_c = c / e
        or_val = float(odds_t / odds_c)
        se_log_or = float(np.sqrt(1 / a + 1 / b + 1 / c + 1 / e))
        or_ci = (float(np.exp(np.log(or_val) - 1.96 * se_log_or)),
                 float(np.exp(np.log(or_val) + 1.96 * se_log_or)))
        out[sub] = {
            "n_control": int(len(ctrl)), "n_trt": int(len(trt)),
            "orr_control": orr_c, "orr_trt": orr_t,
            "arr": arr,
            "OR_trt_vs_control": or_val,
            "OR_ci_lo": or_ci[0], "OR_ci_hi": or_ci[1],
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--biomarker-csv",
                    default="/Users/dianalysenko/Documents/DrugResponce/data/raw/geo/"
                            "GSE173839_ISPY2_DurvalumabOlaparibArm_biomarkers.csv.gz")
    ap.add_argument("--out-dir", default=str(_REPO / "results" / "ispy2_interaction"))
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.biomarker_csv)
    df = df.rename(columns={
        "HR.status..0.HR....1..HR..": "HR_status",
        "HER2.status..0..HER2...1..HER2..": "HER2_status",
        "MP.1.2..class..0..MP1..1..MP2.": "MP_class",
    })
    df["arm"] = (df["Arm"] == "durvalumab/olaparib").astype(int)
    # drop missing pCR
    df = df[df["pCR.status"].isin([0, 1])].rename(columns={"pCR.status": "pCR"}).copy()
    logger.info(
        "N=%d (control=%d, durva+olap=%d); pCR: ctrl=%d/%d (%.1f%%), "
        "trt=%d/%d (%.1f%%)",
        len(df), (df["arm"] == 0).sum(), (df["arm"] == 1).sum(),
        df[df["arm"] == 0]["pCR"].sum(), (df["arm"] == 0).sum(),
        100 * df[df["arm"] == 0]["pCR"].mean(),
        df[df["arm"] == 1]["pCR"].sum(), (df["arm"] == 1).sum(),
        100 * df[df["arm"] == 1]["pCR"].mean(),
    )

    # Baseline unadjusted chi2 + OR per arm (без TME)
    from scipy.stats import fisher_exact
    ct = pd.crosstab(df["arm"], df["pCR"])
    or_base, p_base = fisher_exact(ct.values)
    logger.info("Baseline: OR (durva+olap vs ctrl) = %.3f, Fisher p = %.4f",
                or_base, p_base)

    # Interaction analysis
    covariates = ["HR_status", "MP_class"]
    rows = []
    for sig in TME_SIGNATURES:
        res = _fit_interaction(df, sig, covariates)
        rows.append(res)
    inter_df = pd.DataFrame([r for r in rows if "interaction_p" in r])
    inter_df = inter_df.sort_values("interaction_p")
    # BH FDR
    inter_df["q_interaction"] = _benjamini_hochberg(inter_df["interaction_p"].values)
    inter_df.to_csv(out_dir / "interaction_by_signature.csv", index=False)

    print("\n" + "=" * 90)
    print("Per-signature arm × TME interaction (logistic: pCR ~ sig + arm + sig:arm + HR + MP)")
    print("=" * 90)
    print(inter_df[["sig_col", "n", "n_pcr", "interaction_beta", "interaction_OR",
                    "interaction_p", "q_interaction"]].to_string(
        index=False, float_format=lambda x: f"{x:.4f}" if pd.notna(x) else "—"))

    # Subgroup ORR для signatures с interaction_p < 0.10 (descriptive для top hits)
    significant = inter_df[inter_df["interaction_p"] < 0.10]
    subgroup_rows = []
    for sig in significant["sig_col"]:
        s = _subgroup_stats(df, sig)
        if "low" in s and "high" in s:
            subgroup_rows.append({
                "signature": sig,
                "median_cut": s["median_cut"],
                "low_orr_control": s["low"]["orr_control"],
                "low_orr_trt": s["low"]["orr_trt"],
                "low_ARR": s["low"]["arr"],
                "low_OR": s["low"]["OR_trt_vs_control"],
                "low_OR_ci_lo": s["low"]["OR_ci_lo"],
                "low_OR_ci_hi": s["low"]["OR_ci_hi"],
                "high_orr_control": s["high"]["orr_control"],
                "high_orr_trt": s["high"]["orr_trt"],
                "high_ARR": s["high"]["arr"],
                "high_OR": s["high"]["OR_trt_vs_control"],
                "high_OR_ci_lo": s["high"]["OR_ci_lo"],
                "high_OR_ci_hi": s["high"]["OR_ci_hi"],
                "ARR_difference_high_minus_low":
                    s["high"]["arr"] - s["low"]["arr"],
            })
    subgroup_df = pd.DataFrame(subgroup_rows).sort_values(
        "ARR_difference_high_minus_low", ascending=False
    )
    subgroup_df.to_csv(out_dir / "subgroup_orr.csv", index=False)

    print("\n" + "=" * 90)
    print("Subgroup ORR for signatures with interaction p < 0.10")
    print("(ARR = absolute risk reduction of pCR: durva+olap - control)")
    print("=" * 90)
    if not subgroup_df.empty:
        display_cols = [
            "signature", "low_orr_control", "low_orr_trt", "low_ARR",
            "high_orr_control", "high_orr_trt", "high_ARR",
            "ARR_difference_high_minus_low",
        ]
        print(subgroup_df[display_cols].to_string(
            index=False, float_format=lambda x: f"{x:.3f}" if pd.notna(x) else "—"
        ))
    else:
        print("No signatures with interaction p < 0.10.")

    # VERDICT.md
    lines = [
        "# I-SPY2 durvalumab+olaparib treatment × TME interaction",
        "",
        f"- N = {len(df)} (control={int((df['arm'] == 0).sum())}, "
        f"durva+olap={int((df['arm'] == 1).sum())})",
        f"- HER2- only; HR+: {int((df['HR_status'] == 1).sum())} / "
        f"HR-: {int((df['HR_status'] == 0).sum())}",
        f"- Baseline pCR rate: control {100 * df[df['arm'] == 0]['pCR'].mean():.1f}%, "
        f"durva+olap {100 * df[df['arm'] == 1]['pCR'].mean():.1f}% "
        f"(absolute Δ = {100 * (df[df['arm'] == 1]['pCR'].mean() - df[df['arm'] == 0]['pCR'].mean()):+.1f}%)",
        f"- Baseline OR (Fisher) = {or_base:.3f}, p = {p_base:.4f}",
        "",
        "## Interaction results (per signature)",
        "",
        "| Signature | n | β_int | OR_int | p | q (BH) |",
        "|---|---|---|---|---|---|",
    ]
    for _, r in inter_df.iterrows():
        lines.append(
            f"| {r['sig_col']} | {int(r['n'])} | "
            f"{r['interaction_beta']:.3f} | {r['interaction_OR']:.3f} | "
            f"{r['interaction_p']:.4f} | {r['q_interaction']:.4f} |"
        )

    if not subgroup_df.empty:
        lines += [
            "",
            "## Subgroup ORR (median-dichotomized, interaction p < 0.10)",
            "",
            "| Signature | Low: ctrl → trt (ARR) | High: ctrl → trt (ARR) | Δ ARR (high−low) |",
            "|---|---|---|---|",
        ]
        for _, r in subgroup_df.iterrows():
            lines.append(
                f"| {r['signature']} | "
                f"{r['low_orr_control']:.1%} → {r['low_orr_trt']:.1%} "
                f"({r['low_ARR']:+.1%}) | "
                f"{r['high_orr_control']:.1%} → {r['high_orr_trt']:.1%} "
                f"({r['high_ARR']:+.1%}) | "
                f"**{r['ARR_difference_high_minus_low']:+.3f}** |"
            )

    # Auto-verdict
    best_any_interaction = inter_df["interaction_p"].min() \
        if "interaction_p" in inter_df.columns and not inter_df.empty else 1.0
    best_bh = inter_df["q_interaction"].min() \
        if "q_interaction" in inter_df.columns and not inter_df.empty else 1.0
    lines += ["", "## Verdict", ""]
    if best_bh < 0.10:
        lines.append(
            f"✅ **Clinically actionable signal**: min BH-q = {best_bh:.4f} "
            f"among {len(TME_SIGNATURES)} signatures. "
            "Identify benefiting subgroup for precision application of durva+olaparib."
        )
    elif best_any_interaction < 0.05:
        lines.append(
            f"🟡 **Nominally significant (before multiple testing)**: "
            f"min interaction p = {best_any_interaction:.4f}, "
            f"but does not survive BH correction (min q = {best_bh:.4f}). "
            "Treat as hypothesis-generating; replicate in CALGB 40601 / NeoALTTO / larger I-SPY2."
        )
    else:
        lines.append(
            f"🔴 **No interaction**: best p = {best_any_interaction:.4f}. "
            "Durva+olaparib benefit is homogeneous (or N is too small to detect modifier); "
            "consider larger multi-arm I-SPY2 (GSE194040)."
        )
    (out_dir / "VERDICT.md").write_text("\n".join(lines), encoding="utf-8")
    logger.info("Artefacts → %s", out_dir)

    # JSON summary
    summary = {
        "n_total": int(len(df)),
        "n_control": int((df["arm"] == 0).sum()),
        "n_trt": int((df["arm"] == 1).sum()),
        "pcr_rate_control": float(df[df["arm"] == 0]["pCR"].mean()),
        "pcr_rate_trt": float(df[df["arm"] == 1]["pCR"].mean()),
        "baseline_or": float(or_base),
        "baseline_fisher_p": float(p_base),
        "interactions": inter_df.to_dict(orient="records"),
        "subgroups": subgroup_df.to_dict(orient="records"),
    }
    with (out_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2, default=str)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
