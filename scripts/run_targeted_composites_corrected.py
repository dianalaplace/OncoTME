"""Corrected methodology: TME × targeted therapy composite interactions.

Исправленные проблемы:
1. **HER2+ vs HER2- mismatch**: исключены anti-HER2 arms (нет proper HER2+ control).
2. **Mixed-HER2 arm (Neratinib)**: restrict to HER2- subset.
3. **Covariate balance**: chi2/t-test per clinical variable per comparison.
4. **Permutation test**: 1000 reshuffles outcome within arm+control, computes
   null interaction OR distribution → empirical p-value robust к shared-control
   non-independence.
5. **Bootstrap CI** for OR_interaction.
6. **Pre-specification honesty flags**: какие composites строго biology-based,
   какие informed by earlier per-sig exploration в той же когорте.
7. **Power annotation**: explicit minimum detectable OR_int per arm.

Outputs:
- results/targeted_corrected/VERDICT.md
- results/targeted_corrected/balance_checks.csv  (covariate balance)
- results/targeted_corrected/corrected_results.csv
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
logger = logging.getLogger("corrected")


# Honest pre-specification audit — which composites were truly biology-only,
# which were informed by prior exploratory per-signature analysis on the SAME cohort.
PRESPEC_AUDIT = {
    "anti-HER2": "informed_by_exploratory",         # TGFb was #1 per-sig
    "PARPi+platinum": "informed_by_exploratory",    # IL6-STAT3 was #1 per-sig
    "anti-Angiopoietin": "informed_by_exploratory", # M2 was #1 per-sig
    "anti-IGF-1R": "informed_by_exploratory",       # CAF was #1 per-sig
    "pan-HER TKI": "informed_by_exploratory",       # angiogenesis was #1 per-sig
    "AKT inhibitor": "informed_by_exploratory",     # immune_exclusion
}

# HER2 eligibility requirement per arm (for control matching)
ARM_HER2_STATUS = {
    "Paclitaxel + ABT 888 + Carboplatin": "her2_neg",
    "Paclitaxel + AMG 386": "her2_neg",
    "Paclitaxel + AMG-386": "her2_neg",
    "Paclitaxel + Ganetespib": "her2_neg",
    "Paclitaxel + Ganitumab": "her2_neg",
    "Paclitaxel + MK-2206": "her2_neg",
    "Paclitaxel + Pembrolizumab": "her2_neg",
    "Paclitaxel + Neratinib": "mixed",              # needs subset
    "Paclitaxel + Trastuzumab": "her2_pos_no_ctrl",  # DROP
    "Paclitaxel + Pertuzumab + Trastuzumab": "her2_pos_no_ctrl",  # DROP
    "Paclitaxel + AMG 386 + Trastuzumab": "her2_pos_no_ctrl",     # DROP
    "Paclitaxel + MK-2206 + Trastuzumab": "her2_pos_no_ctrl",     # DROP
    "T-DM1 + Pertuzumab": "her2_pos_no_ctrl",        # DROP
}


def _parse_meta(path):
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
                if key is None: continue
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


def _compute_composite(scores: pd.DataFrame, comp: TargetedComposite) -> pd.Series:
    terms = []
    for c in comp.components:
        if c.name not in scores.columns:
            continue
        terms.append(c.sign * _zscore(scores[c.name]))
    return sum(terms)


def _balance_check(df: pd.DataFrame) -> pd.DataFrame:
    """Per-covariate balance between arm=0 and arm=1 via χ² (categorical) or t-test."""
    from scipy.stats import chi2_contingency, ttest_ind
    rows = []
    for col in ["HR", "MP", "HER2"]:
        if col not in df.columns:
            continue
        d = df[[col, "arm"]].dropna()
        if d[col].nunique() < 2:
            rows.append({"variable": col, "test": "—", "statistic": None, "p_value": None,
                         "note": "constant"})
            continue
        ct = pd.crosstab(d["arm"], d[col])
        if ct.shape == (2, 2) or ct.shape[1] >= 2:
            try:
                chi2, p, _, _ = chi2_contingency(ct)
                rows.append({"variable": col, "test": "chi2",
                             "statistic": float(chi2), "p_value": float(p), "note": ""})
            except Exception as exc:
                rows.append({"variable": col, "test": "chi2_failed",
                             "statistic": None, "p_value": None, "note": str(exc)})
    return pd.DataFrame(rows)


def _fit_interaction(df: pd.DataFrame, comp_col: str = "composite") -> dict:
    import statsmodels.api as sm
    d = df[["pCR", "arm", comp_col, "HR", "MP"]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(d) < 40:
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
            "n": int(len(d)), "n_pcr": int(y.sum()),
            "or_interaction": float(np.exp(m.params["interaction"])),
            "or_ci_lo": float(np.exp(m.conf_int().loc["interaction", 0])),
            "or_ci_hi": float(np.exp(m.conf_int().loc["interaction", 1])),
            "p_interaction": float(m.pvalues["interaction"]),
        }
    except Exception as exc:
        return {"error": str(exc)}


def _permutation_test(df: pd.DataFrame, n_perm: int = 1000, seed: int = 42) -> dict:
    """Shuffle pCR labels within the arm+control pool → null OR_int distribution.

    This is robust к shared-control non-independence across composites (each test's
    null is computed with its own permutation), а также не предполагает
    distributional assumptions of Wald test.
    """
    import statsmodels.api as sm
    observed = _fit_interaction(df)
    if "p_interaction" not in observed:
        return {"error": "observed fit failed"}
    obs_or = observed["or_interaction"]
    obs_p = observed["p_interaction"]

    rng = np.random.default_rng(seed)
    null_or = []
    d = df[["pCR", "arm", "composite", "HR", "MP"]].apply(
        pd.to_numeric, errors="coerce").dropna()
    if len(d) < 40:
        return {"error": f"n={len(d)}"}
    covs = [c for c in ["HR", "MP"] if d[c].nunique() >= 2]
    d["score_z"] = _zscore(d["composite"])

    for _ in range(n_perm):
        y_perm = d["pCR"].sample(frac=1.0, random_state=int(rng.integers(0, 2**31))).values
        d_perm = d.copy()
        d_perm["pCR"] = y_perm
        d_perm["interaction"] = d_perm["score_z"] * d_perm["arm"]
        X = sm.add_constant(d_perm[["score_z", "arm", "interaction"] + covs])
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m = sm.Logit(d_perm["pCR"].astype(int), X).fit(
                    disp=False, method="bfgs", maxiter=200)
            null_or.append(float(np.exp(m.params["interaction"])))
        except Exception:
            continue

    null_arr = np.array([x for x in null_or if np.isfinite(x)])
    if len(null_arr) == 0:
        return {"error": "no successful perms"}
    # two-sided empirical p
    p_perm = float(((np.abs(np.log(null_arr)) >= np.abs(np.log(obs_or))).sum() + 1) /
                    (len(null_arr) + 1))
    return {
        "observed_or": obs_or,
        "observed_p_wald": obs_p,
        "null_n": int(len(null_arr)),
        "null_median_or": float(np.median(null_arr)),
        "null_95_or_lo": float(np.quantile(null_arr, 0.025)),
        "null_95_or_hi": float(np.quantile(null_arr, 0.975)),
        "permutation_p": p_perm,
    }


def _bootstrap_or(df: pd.DataFrame, n_boot: int = 500, seed: int = 42) -> tuple:
    """Bootstrap CI for OR_interaction."""
    import statsmodels.api as sm
    d = df[["pCR", "arm", "composite", "HR", "MP"]].apply(
        pd.to_numeric, errors="coerce").dropna()
    if len(d) < 40:
        return (float("nan"), float("nan"))
    covs = [c for c in ["HR", "MP"] if d[c].nunique() >= 2]
    d["score_z"] = _zscore(d["composite"])
    d["interaction"] = d["score_z"] * d["arm"]
    rng = np.random.default_rng(seed)
    ors = []
    for _ in range(n_boot):
        idx = rng.choice(len(d), size=len(d), replace=True)
        try:
            sub = d.iloc[idx]
            X = sm.add_constant(sub[["score_z", "arm", "interaction"] + covs])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m = sm.Logit(sub["pCR"].astype(int), X).fit(
                    disp=False, method="bfgs", maxiter=150)
            or_int = float(np.exp(m.params["interaction"]))
            if np.isfinite(or_int):
                ors.append(or_int)
        except Exception:
            continue
    if not ors:
        return (float("nan"), float("nan"))
    return (float(np.quantile(ors, 0.025)), float(np.quantile(ors, 0.975)))


def _min_detectable_or(n_arm: int, pcr_ctrl: float = 0.2) -> float:
    """Очень грубая оценка: minimum detectable OR_int для 80% power.

    Formula approximation for logistic interaction (Demidenko 2008).
    """
    if n_arm < 20:
        return float("inf")
    # rough: OR_detectable ~ exp(2.8 * sqrt(2/(n*p*(1-p))))
    # using pcr_ctrl as approximation for variance scale
    sigma = np.sqrt(2 / (n_arm * pcr_ctrl * (1 - pcr_ctrl)))
    return float(np.exp(2.8 * sigma))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gene-matrix",
                    default=str(_REPO / "data/raw/geo/GSE194040_gene_level.txt.gz"))
    ap.add_argument("--meta1",
                    default=str(_REPO / "data/raw/geo/GSE194040-GPL20078_series_matrix.txt.gz"))
    ap.add_argument("--meta2",
                    default=str(_REPO / "data/raw/geo/GSE194040-GPL30493_series_matrix.txt.gz"))
    ap.add_argument("--gmt", default=str(_REPO / "signatures" / "tme_signatures.gmt"))
    ap.add_argument("--n-perm", type=int, default=1000)
    ap.add_argument("--n-boot", type=int, default=500)
    ap.add_argument("--out-dir",
                    default=str(_REPO / "results" / "targeted_corrected"))
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    meta = pd.concat([_parse_meta(args.meta1).assign(platform="GPL20078"),
                      _parse_meta(args.meta2).assign(platform="GPL30493")])
    meta["resid"] = meta["patient id"].astype(str)
    meta["pcr_int"] = pd.to_numeric(meta["pcr"], errors="coerce")
    meta["her2_int"] = pd.to_numeric(meta["her2"], errors="coerce")
    meta = meta.dropna(subset=["pcr_int", "her2_int"])
    gene = _load_gene(args.gene_matrix)
    meta = meta[meta["resid"].isin(gene.columns.astype(str))]

    # All control patients
    control = meta[meta["arm"] == "Paclitaxel"].copy()
    ctrl_her2_neg = int((control["her2_int"] == 0).sum())
    ctrl_her2_pos = int((control["her2_int"] == 1).sum())
    logger.info("Control arm: %d total (HER2- = %d, HER2+ = %d)",
                len(control), ctrl_her2_neg, ctrl_her2_pos)
    if ctrl_her2_pos < 30:
        logger.warning(
            "⚠️ Paclitaxel-only control has only %d HER2+ patients. "
            "Anti-HER2 arms have no valid matched control in this release.",
            ctrl_her2_pos,
        )

    # Precompute ssGSEA on full cohort
    gmt = load_gmt(args.gmt)
    scores_full = ssgsea_scores(gene[meta["resid"].tolist()], gene_sets=gmt)
    meta_i = meta.set_index("resid")

    results = []
    balance_rows = []

    for comp in ALL_COMPOSITES:
        logger.info("\n=== Class: %s ===", comp.class_name)
        arm_labels = comp.arm_labels
        # HER2 eligibility status
        all_arm_her2 = {ARM_HER2_STATUS.get(a, "unknown") for a in arm_labels}

        # DECISION TREE for validity
        if "her2_pos_no_ctrl" in all_arm_her2:
            logger.warning(
                "  ⚠️ Arms in class %s are HER2+. No valid HER2+ control in this "
                "release (Paclitaxel-only = 100%% HER2-). SKIPPED as invalid.",
                comp.class_name,
            )
            results.append({
                "class_name": comp.class_name,
                "status": "INVALID_HER2_MISMATCH",
                "note": "HER2+ arms have no valid HER2- control comparator",
                "prespec_audit": PRESPEC_AUDIT.get(comp.class_name, "unknown"),
            })
            continue

        # Build comparison set: ARM patients (matching HER2 status) + matched controls
        arm_pts = meta_i[meta_i["arm"].isin(arm_labels)].copy()
        if "mixed" in all_arm_her2:
            # restrict to HER2- subset
            before_n = len(arm_pts)
            arm_pts = arm_pts[arm_pts["her2_int"] == 0]
            logger.info("  HER2-mixed arm restricted to HER2-: %d → %d",
                        before_n, len(arm_pts))
        # Control: HER2- only (since all valid arms here are HER2- after filter)
        ctrl_pts = meta_i[(meta_i["arm"] == "Paclitaxel") & (meta_i["her2_int"] == 0)].copy()
        combined = pd.concat([ctrl_pts, arm_pts])
        combined["arm"] = (combined["arm"] != "Paclitaxel").astype(int)

        n_ctrl = int((combined["arm"] == 0).sum())
        n_trt = int((combined["arm"] == 1).sum())
        logger.info("  N: ctrl=%d, trt=%d", n_ctrl, n_trt)
        if n_trt < 30:
            results.append({
                "class_name": comp.class_name,
                "status": "TOO_SMALL",
                "note": f"n_trt={n_trt} < 30",
            })
            continue

        # Composite
        combined["composite"] = _compute_composite(
            scores_full.loc[combined.index], comp).values
        combined["pCR"] = combined["pcr_int"].astype(int)
        combined["HR"] = pd.to_numeric(combined["hr"], errors="coerce")
        combined["MP"] = pd.to_numeric(combined["mp"], errors="coerce")
        combined["HER2"] = combined["her2_int"]

        # Balance check
        balance_df = _balance_check(combined)
        balance_df["class_name"] = comp.class_name
        balance_rows.append(balance_df)
        logger.info("  Covariate balance:\n%s", balance_df.to_string(index=False))

        # Interaction fit
        fit = _fit_interaction(combined)
        # Permutation test
        perm = _permutation_test(combined, n_perm=args.n_perm)
        # Bootstrap
        boot_lo, boot_hi = _bootstrap_or(combined, n_boot=args.n_boot)
        # Power annotation
        min_detect = _min_detectable_or(n_trt)

        # Unadjusted Fisher
        from scipy.stats import fisher_exact
        ct = pd.crosstab(combined["arm"], combined["pCR"])
        or_base, p_base = fisher_exact(ct.values) if ct.shape == (2, 2) else (np.nan, np.nan)

        results.append({
            "class_name": comp.class_name,
            "status": "VALID",
            "prespec_audit": PRESPEC_AUDIT.get(comp.class_name, "unknown"),
            "n_ctrl": n_ctrl, "n_trt": n_trt,
            "pcr_ctrl": float(combined[combined["arm"] == 0]["pCR"].mean()),
            "pcr_trt": float(combined[combined["arm"] == 1]["pCR"].mean()),
            "baseline_OR": float(or_base), "baseline_p": float(p_base),
            "or_interaction": fit.get("or_interaction"),
            "or_ci_wald_lo": fit.get("or_ci_lo"), "or_ci_wald_hi": fit.get("or_ci_hi"),
            "or_ci_boot_lo": float(boot_lo), "or_ci_boot_hi": float(boot_hi),
            "p_wald": fit.get("p_interaction"),
            "p_permutation": perm.get("permutation_p"),
            "null_median_or": perm.get("null_median_or"),
            "min_detectable_or_80power": min_detect,
            "balance_HR_p": balance_df.loc[balance_df["variable"] == "HR", "p_value"].iloc[0]
                if (balance_df["variable"] == "HR").any() else None,
            "balance_MP_p": balance_df.loc[balance_df["variable"] == "MP", "p_value"].iloc[0]
                if (balance_df["variable"] == "MP").any() else None,
        })
        logger.info(
            "  OR_int=%.3f  Wald CI=[%.2f, %.2f]  Boot CI=[%.2f, %.2f]  "
            "p_Wald=%.4f  p_perm=%.4f  min_detectable@80pow=%.2f",
            fit.get("or_interaction", float("nan")),
            fit.get("or_ci_lo", float("nan")), fit.get("or_ci_hi", float("nan")),
            boot_lo, boot_hi,
            fit.get("p_interaction", float("nan")),
            perm.get("permutation_p", float("nan")),
            min_detect,
        )

    res_df = pd.DataFrame(results)
    res_df.to_csv(out / "corrected_results.csv", index=False)
    if balance_rows:
        pd.concat(balance_rows, ignore_index=True).to_csv(
            out / "balance_checks.csv", index=False)

    # BH on permutation p-values across VALID tests
    valid = res_df[res_df["status"] == "VALID"].copy()
    if not valid.empty and "p_permutation" in valid.columns:
        p = valid["p_permutation"].astype(float).values
        order = np.argsort(p)
        ranks = np.empty(len(p))
        ranks[order] = np.arange(1, len(p) + 1)
        q = np.minimum(p * len(p) / ranks, 1.0)
        q_sorted = q[order]
        for i in range(len(p) - 2, -1, -1):
            if q_sorted[i] > q_sorted[i + 1]:
                q_sorted[i] = q_sorted[i + 1]
        q_final = np.empty(len(p))
        q_final[order] = q_sorted
        valid["q_perm_bh"] = q_final
        res_df = pd.concat([valid, res_df[res_df["status"] != "VALID"]])
        res_df.to_csv(out / "corrected_results.csv", index=False)

    # Print + VERDICT
    print("\n" + "=" * 115)
    print("CORRECTED analysis: TME × targeted therapy — HER2-matched controls,"
          " permutation test, bootstrap CI")
    print("=" * 115)
    for _, r in res_df.iterrows():
        if r.get("status") == "VALID":
            print(f"  {r['class_name']:20s}  n_ctrl={r['n_ctrl']:3d} n_trt={r['n_trt']:3d}  "
                  f"OR_int={r.get('or_interaction', float('nan')):.2f}  "
                  f"Boot95%CI=[{r.get('or_ci_boot_lo', float('nan')):.2f}, "
                  f"{r.get('or_ci_boot_hi', float('nan')):.2f}]  "
                  f"p_Wald={r.get('p_wald', float('nan')):.4f}  "
                  f"p_perm={r.get('p_permutation', float('nan')):.4f}  "
                  f"q_BH={r.get('q_perm_bh', float('nan')):.4f}  "
                  f"min_detect_OR={r.get('min_detectable_or_80power', float('nan')):.1f}  "
                  f"audit={r.get('prespec_audit', '')}")
        else:
            print(f"  {r['class_name']:20s}  {r['status']:30s}  {r.get('note', '')}")

    # VERDICT.md
    lines = [
        "# Corrected methodology: TME × targeted therapy",
        "",
        "## Methodology corrections applied",
        "",
        "1. **HER2 matching**: HER2+ arms (trastuzumab, T-DM1+pertuzumab) excluded — "
        "no valid HER2+ control exists in this GSE194040 release "
        f"(Paclitaxel-only control = {ctrl_her2_neg} HER2- / {ctrl_her2_pos} HER2+).",
        "2. **Mixed-HER2 arm (Neratinib)**: restricted to HER2- subset for fair comparison.",
        "3. **Covariate balance** тестирован χ² для HR, MP, HER2 между arm и control.",
        "4. **Permutation test** (n=" + str(args.n_perm) + "): reshuffles pCR labels, "
        "recomputes OR_int, empirical p — robust to distributional assumptions and "
        "to shared-control non-independence.",
        "5. **Bootstrap CI** (n=" + str(args.n_boot) + "): additional CI robust к model assumption.",
        "6. **Pre-specification audit**: отмечено, какие composites были biology-only pre-spec, "
        "а какие informed by prior exploratory per-signature analysis на той же когорте.",
        "7. **Power annotation**: minimum detectable OR_int при 80% power показан.",
        "",
        "## Results (valid tests only)",
        "",
        "| Class | n_ctrl/n_trt | OR_int | Boot 95% CI | p_perm | q_BH | Min detect OR @80pow | Balance OK? | Pre-spec audit |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    valid_results = res_df[res_df["status"] == "VALID"] if "status" in res_df.columns else res_df
    for _, r in valid_results.iterrows():
        b_hr = r.get("balance_HR_p")
        b_mp = r.get("balance_MP_p")
        balance_ok = "✓" if (pd.notna(b_hr) and b_hr > 0.05) and \
                            (pd.notna(b_mp) and b_mp > 0.05) else "⚠"
        lines.append(
            f"| {r['class_name']} | "
            f"{int(r.get('n_ctrl', 0))}/{int(r.get('n_trt', 0))} | "
            f"{r.get('or_interaction', float('nan')):.2f} | "
            f"[{r.get('or_ci_boot_lo', float('nan')):.2f}, "
            f"{r.get('or_ci_boot_hi', float('nan')):.2f}] | "
            f"{r.get('p_permutation', float('nan')):.4f} | "
            f"{r.get('q_perm_bh', float('nan')):.4f} | "
            f"{r.get('min_detectable_or_80power', float('nan')):.1f} | "
            f"{balance_ok} | {r.get('prespec_audit', '—')} |"
        )

    # Invalid
    invalid = res_df[res_df["status"] != "VALID"] if "status" in res_df.columns else pd.DataFrame()
    if not invalid.empty:
        lines += ["", "## Excluded (invalid comparison or too small)", ""]
        for _, r in invalid.iterrows():
            lines.append(f"- **{r['class_name']}**: {r['status']} — {r.get('note', '')}")

    lines += ["", "## Verdict", ""]
    if not valid_results.empty:
        sig_bh = valid_results[valid_results["q_perm_bh"] < 0.10] \
            if "q_perm_bh" in valid_results.columns else pd.DataFrame()
        sig_nom = valid_results[
            (valid_results["p_permutation"] < 0.05) &
            (valid_results["q_perm_bh"] >= 0.10)
        ] if "p_permutation" in valid_results.columns else pd.DataFrame()
        if not sig_bh.empty:
            lines.append(
                f"✅ **BH-significant under permutation test** ({len(sig_bh)} of "
                f"{len(valid_results)}): "
                + ", ".join([
                    f"{r['class_name']} (q={r['q_perm_bh']:.3f}, OR={r['or_interaction']:.2f})"
                    for _, r in sig_bh.iterrows()
                ])
            )
        if not sig_nom.empty:
            lines.append(
                f"🟡 **Nominally significant under permutation** ({len(sig_nom)}): "
                + ", ".join([
                    f"{r['class_name']} (p_perm={r['p_permutation']:.3f})"
                    for _, r in sig_nom.iterrows()
                ])
            )
        if sig_bh.empty and sig_nom.empty:
            lines.append(
                "🔴 No composite survives rigorous permutation test and BH correction. "
                "Previous nominal p-values may have been inflated by Wald-approximation "
                "in small samples."
            )
    lines += [
        "",
        "## Key limitations that remain",
        "",
        "- **Composite pre-specification imperfect**: all composites were informed by "
        "exploratory per-signature analysis on the same I-SPY2 cohort. True "
        "pre-specification requires registration before seeing data, or held-out "
        "validation cohort (GeparOLA, CALGB 40601, TransATAC).",
        "- **Temporal/adaptive-randomization bias** not addressed: "
        "I-SPY2 enrolled patients 2010-2017+ with adaptive assignment. Control arm "
        "includes patients from all enrollment eras, while some experimental arms "
        "were active only in specific years. Enrollment dates are not in metadata.",
        "- **Anti-HER2 analysis not possible** в этом GSE release: no HER2+ control. "
        "Требуется отдельный loader paclitaxel+trastuzumab HER2+ control arm или "
        "separate cohort (CALGB 40601, NeoALTTO).",
        "- **Power limited** для arms с n_trt<100. CI широкие; effect sizes могут быть "
        "inflated winner's-curse.",
    ]
    (out / "VERDICT.md").write_text("\n".join(lines), encoding="utf-8")
    logger.info("Artefacts → %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
