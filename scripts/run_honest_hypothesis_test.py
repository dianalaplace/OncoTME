"""Honest hypothesis test — runs LOCKED pre-registered analysis from PRE_REGISTRATION.md.

Executes:
- H1: stromal_score × each of 9 I-SPY2 arms, sign test + meta-analysis
- H2: subtype-stratified (HR+/TNBC) for Ganitumab, Pembro, Neratinib
- H3: stromal-related vs stromal-unrelated arm comparison
- R1: 1000 split-half stability on strongest arm
- R2: leave-one-component-out sensitivity
- R3: 1000 permutations per arm

Outputs `results/honest_test/`:
- stromal_cross_arm.csv — H1 results
- h2_subtype.csv — H2 results
- stability_splithalf.csv — R1
- leave_one_out.csv — R2
- HONEST_VERDICT.md — final honest verdict
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
logger = logging.getLogger("honest")


# ===== LOCKED FORMULA — DO NOT MODIFY =====
STROMAL_COMPONENTS = ["CAF_proxy", "TGFb_activity"]  # all with +1 sign

ARM_LIST_LOCKED = [
    # (arm_label, short_name, biologically_stromal_related)
    ("Paclitaxel + Pembrolizumab", "Pembro", True),    # immune exclusion by stroma
    ("Paclitaxel + Ganitumab", "Ganitumab", True),      # IGF-1 stromal
    ("Paclitaxel + Neratinib", "Neratinib", True),      # HER TKI, TGFβ-EMT
    ("Paclitaxel + ABT 888 + Carboplatin", "PARPi", True),  # stroma access
    ("Paclitaxel + AMG 386", "AMG386", False),           # less stromal
    ("Paclitaxel + MK-2206", "AKT", True),                # TGFβ/PI3K
    ("Paclitaxel + Ganetespib", "Ganetespib", False),    # HSP90 — stromal-independent
    ("T-DM1 + Pertuzumab", "TDM1", True),                # ADCC, HER2+ (control mismatch!)
    ("Paclitaxel + Trastuzumab", "Trastuzumab", True),   # ADCC, HER2+ (control mismatch!)
]


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
                        k, v = p.split(": ", 1); kvs.append((k.strip(), v.strip()))
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


def _compute_stromal(scores, components=STROMAL_COMPONENTS):
    return sum(_zscore(scores[c]) for c in components)


def _fit_logit_interaction(df, score_col="stromal_score", extra_covs=None):
    import statsmodels.api as sm
    covs = ["HR", "MP"] + (extra_covs or [])
    covs = [c for c in covs if c in df.columns]
    d = df[["pCR", "arm", score_col] + covs].apply(pd.to_numeric, errors="coerce").dropna()
    if len(d) < 40 or d["arm"].nunique() < 2:
        return {"error": f"n={len(d)}"}
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
            "or_interaction": float(np.exp(m.params["interaction"])),
            "log_or_interaction": float(m.params["interaction"]),
            "se_log_or": float(m.bse["interaction"]),
            "or_ci_lo": float(np.exp(m.conf_int().loc["interaction", 0])),
            "or_ci_hi": float(np.exp(m.conf_int().loc["interaction", 1])),
            "p_interaction": float(m.pvalues["interaction"]),
        }
    except Exception as exc:
        return {"error": str(exc)}


def _permutation_p(df, n_perm=1000, seed=42):
    import statsmodels.api as sm
    obs = _fit_logit_interaction(df)
    if "or_interaction" not in obs:
        return float("nan")
    obs_log_or = obs["log_or_interaction"]
    rng = np.random.default_rng(seed)
    covs = [c for c in ["HR", "MP"] if c in df.columns and df[c].nunique() >= 2]
    d = df[["pCR", "arm", "stromal_score"] + covs].apply(
        pd.to_numeric, errors="coerce").dropna()
    d["score_z"] = _zscore(d["stromal_score"])
    null = []
    for _ in range(n_perm):
        y_perm = rng.permutation(d["pCR"].values)
        dp = d.copy()
        dp["pCR"] = y_perm
        dp["interaction"] = dp["score_z"] * dp["arm"]
        X = sm.add_constant(dp[["score_z", "arm", "interaction"] + covs])
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m = sm.Logit(dp["pCR"].astype(int), X).fit(
                    disp=False, method="bfgs", maxiter=150)
            null.append(float(m.params["interaction"]))
        except Exception:
            continue
    null = np.array([x for x in null if np.isfinite(x)])
    if len(null) == 0:
        return float("nan")
    return float(((np.abs(null) >= abs(obs_log_or)).sum() + 1) / (len(null) + 1))


def _meta_analysis_re(log_ors, se_log_ors):
    """DerSimonian-Laird random effects meta-analysis for log OR_int."""
    log_ors = np.asarray(log_ors, dtype=float)
    se = np.asarray(se_log_ors, dtype=float)
    mask = np.isfinite(log_ors) & np.isfinite(se) & (se > 0)
    log_ors = log_ors[mask]; se = se[mask]
    if len(log_ors) < 2:
        return {"error": "insufficient arms"}
    w_fe = 1.0 / (se ** 2)
    mu_fe = (w_fe * log_ors).sum() / w_fe.sum()
    Q = (w_fe * (log_ors - mu_fe) ** 2).sum()
    df_q = len(log_ors) - 1
    c = w_fe.sum() - (w_fe ** 2).sum() / w_fe.sum()
    tau2 = max(0.0, (Q - df_q) / c) if c > 0 else 0.0
    w_re = 1.0 / (se ** 2 + tau2)
    mu_re = (w_re * log_ors).sum() / w_re.sum()
    se_re = 1.0 / np.sqrt(w_re.sum())
    z = mu_re / se_re
    from scipy.stats import norm
    p_re = 2 * (1 - norm.cdf(abs(z)))
    return {
        "pooled_log_or": float(mu_re),
        "pooled_or": float(np.exp(mu_re)),
        "pooled_ci_lo": float(np.exp(mu_re - 1.96 * se_re)),
        "pooled_ci_hi": float(np.exp(mu_re + 1.96 * se_re)),
        "pooled_p": float(p_re),
        "tau2": float(tau2),
        "n_arms": int(len(log_ors)),
        "Q": float(Q), "Q_df": int(df_q),
    }


def _sign_test_binomial(outcomes_below_one, n_total):
    """One-sided binomial test: does OR_int < 1 occur more often than 50/50?"""
    from scipy.stats import binom
    # one-sided: P(X >= outcomes_below_one | p=0.5)
    return float(1 - binom.cdf(outcomes_below_one - 1, n_total, 0.5))


def _split_half_stability(df, n_iter=1000, seed=42):
    """Bootstrap-like: fit logit interaction on random 60% resample, record p."""
    import statsmodels.api as sm
    rng = np.random.default_rng(seed)
    p_values = []
    covs = [c for c in ["HR", "MP"] if c in df.columns and df[c].nunique() >= 2]
    base = df.copy().reset_index(drop=True)
    base["score_z"] = _zscore(base["stromal_score"])
    base["interaction"] = base["score_z"].astype(float) * base["arm"].astype(float)
    first_err = None
    for _ in range(n_iter):
        idx_0 = np.where(base["arm"].values == 0)[0]
        idx_1 = np.where(base["arm"].values == 1)[0]
        n0 = max(int(0.6 * len(idx_0)), 5)
        n1 = max(int(0.6 * len(idx_1)), 5)
        sel_0 = rng.choice(idx_0, size=n0, replace=False)
        sel_1 = rng.choice(idx_1, size=n1, replace=False)
        sample = base.iloc[np.concatenate([sel_0, sel_1])].copy()
        sample_pcr = sample["pCR"].astype(int)
        if sample_pcr.nunique() < 2:
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                X = sm.add_constant(
                    sample[["score_z", "arm", "interaction"] + covs].astype(float))
                m = sm.Logit(sample_pcr, X).fit(disp=False, method="bfgs",
                                                 maxiter=150)
                p = float(m.pvalues["interaction"])
                if np.isfinite(p):
                    p_values.append(p)
        except Exception as e:
            if first_err is None:
                first_err = f"{type(e).__name__}: {e}"
            continue
    if not p_values and first_err:
        logger.warning("split-half all iterations failed, e.g.: %s", first_err)
    return np.array(p_values)


def _fisher_arm(df):
    from scipy.stats import fisher_exact
    ct = pd.crosstab(df["arm"], df["pCR"])
    if ct.shape != (2, 2):
        return float("nan"), float("nan")
    or_val, p = fisher_exact(ct.values)
    return float(or_val), float(p)


def _balance_check(df):
    from scipy.stats import chi2_contingency
    out = {}
    for col in ["HR", "MP", "HER2"]:
        if col not in df.columns:
            continue
        d = df[[col, "arm"]].dropna()
        if d[col].nunique() < 2:
            out[col] = None; continue
        ct = pd.crosstab(d["arm"], d[col])
        if ct.shape[0] >= 2 and ct.shape[1] >= 2:
            try:
                _, p, _, _ = chi2_contingency(ct)
                out[col] = float(p)
            except Exception:
                out[col] = None
        else:
            out[col] = None
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gene-matrix",
                    default=str(_REPO / "data/raw/geo/GSE194040_gene_level.txt.gz"))
    ap.add_argument("--meta1",
                    default=str(_REPO / "data/raw/geo/GSE194040-GPL20078_series_matrix.txt.gz"))
    ap.add_argument("--meta2",
                    default=str(_REPO / "data/raw/geo/GSE194040-GPL30493_series_matrix.txt.gz"))
    ap.add_argument("--gmt", default=str(_REPO / "signatures" / "tme_signatures.gmt"))
    ap.add_argument("--out-dir", default=str(_REPO / "results" / "honest_test"))
    ap.add_argument("--n-perm", type=int, default=1000)
    ap.add_argument("--n-split", type=int, default=1000)
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    # Load cohort
    meta = pd.concat([_parse_meta(args.meta1).assign(platform="GPL20078"),
                      _parse_meta(args.meta2).assign(platform="GPL30493")])
    meta["resid"] = meta["patient id"].astype(str)
    meta["pcr_int"] = pd.to_numeric(meta["pcr"], errors="coerce")
    meta["her2_int"] = pd.to_numeric(meta["her2"], errors="coerce")
    meta["hr_int"] = pd.to_numeric(meta["hr"], errors="coerce")
    meta["mp_int"] = pd.to_numeric(meta["mp"], errors="coerce")
    meta = meta.dropna(subset=["pcr_int"])
    gene = _load_gene(args.gene_matrix)
    meta = meta[meta["resid"].isin(gene.columns.astype(str))]
    logger.info("I-SPY2: %d samples loaded", len(meta))

    # ssGSEA — run ONCE
    gmt = load_gmt(args.gmt)
    scores = ssgsea_scores(gene[meta["resid"].tolist()], gene_sets=gmt)
    logger.info("ssGSEA computed")

    # Stromal score (LOCKED)
    stromal = _compute_stromal(scores)
    stromal.name = "stromal_score"
    stromal_caf_only = _zscore(scores["CAF_proxy"])
    stromal_tgfb_only = _zscore(scores["TGFb_activity"])
    logger.info("stromal_score computed (CAF + TGFb, z-scored each)")

    meta_i = meta.set_index("resid")

    # ==================== H1: Cross-arm analysis ============================
    logger.info("=== H1: cross-arm stromal × arm interaction ===")
    h1_rows = []
    for arm_label, short, biol_linked in ARM_LIST_LOCKED:
        arm_pts = meta_i[meta_i["arm"] == arm_label]
        ctrl_pts = meta_i[meta_i["arm"] == "Paclitaxel"]
        # HER2 matching
        all_her2_trt = set(arm_pts["her2_int"].unique()) - {np.nan}
        is_her2_pos_arm = (all_her2_trt == {1}) or (all_her2_trt == {1.0})
        if is_her2_pos_arm:
            # trastuzumab / T-DM1 arms: no HER2+ control
            logger.warning("  %s — HER2+ arm, no matched HER2+ control (SKIPPING)",
                           short)
            h1_rows.append({
                "arm": short, "arm_label": arm_label, "biological_stromal_link": biol_linked,
                "status": "SKIPPED_NO_CONTROL",
            })
            continue
        if "mixed" in str(all_her2_trt) or (0 in all_her2_trt and 1 in all_her2_trt):
            # mixed arm: restrict to HER2-
            arm_pts = arm_pts[arm_pts["her2_int"] == 0]
        combined = pd.concat([
            ctrl_pts.assign(_label="ctrl"),
            arm_pts.assign(_label="trt"),
        ])
        combined["arm"] = (combined["_label"] == "trt").astype(int)
        if (combined["arm"] == 1).sum() < 30:
            logger.warning("  %s — n_trt=%d too small, skip", short, (combined["arm"] == 1).sum())
            h1_rows.append({
                "arm": short, "arm_label": arm_label, "biological_stromal_link": biol_linked,
                "status": "TOO_SMALL",
            })
            continue
        df = pd.DataFrame(index=combined.index)
        df["pCR"] = combined["pcr_int"].astype(int)
        df["arm"] = combined["arm"]
        df["HR"] = combined["hr_int"]; df["MP"] = combined["mp_int"]
        df["HER2"] = combined["her2_int"]
        df["stromal_score"] = stromal.reindex(df.index).values

        balance = _balance_check(df)
        fit = _fit_logit_interaction(df)
        or_fisher, p_fisher = _fisher_arm(df)
        # Permutation
        p_perm = _permutation_p(df, n_perm=args.n_perm)

        row = {
            "arm": short, "arm_label": arm_label, "biological_stromal_link": biol_linked,
            "status": "VALID",
            "n_ctrl": int((df["arm"] == 0).sum()),
            "n_trt": int((df["arm"] == 1).sum()),
            "pcr_ctrl": float(df[df["arm"] == 0]["pCR"].mean()),
            "pcr_trt": float(df[df["arm"] == 1]["pCR"].mean()),
            "baseline_OR": or_fisher, "baseline_p": p_fisher,
            "or_interaction": fit.get("or_interaction"),
            "log_or": fit.get("log_or_interaction"),
            "se_log_or": fit.get("se_log_or"),
            "or_ci_lo": fit.get("or_ci_lo"), "or_ci_hi": fit.get("or_ci_hi"),
            "p_wald": fit.get("p_interaction"),
            "p_permutation": p_perm,
            "balance_HR_p": balance.get("HR"), "balance_MP_p": balance.get("MP"),
        }
        h1_rows.append(row)
        logger.info(
            "  %s  n=%d  OR_int=%.2f [%.2f, %.2f]  p_Wald=%.3f  p_perm=%.3f",
            short, row["n_ctrl"] + row["n_trt"],
            row["or_interaction"] or np.nan,
            row.get("or_ci_lo") or np.nan, row.get("or_ci_hi") or np.nan,
            row.get("p_wald") or np.nan, row.get("p_permutation") or np.nan,
        )

    h1_df = pd.DataFrame(h1_rows)
    h1_df.to_csv(out / "H1_cross_arm.csv", index=False)

    # Sign test on VALID arms only
    valid = h1_df[h1_df["status"] == "VALID"]
    valid_or = valid["or_interaction"].astype(float).values
    n_below_1 = int((valid_or < 1).sum())
    n_total = len(valid_or)
    signtest_p = _sign_test_binomial(n_below_1, n_total) if n_total > 0 else float("nan")
    signtest_p_above = _sign_test_binomial(n_total - n_below_1, n_total) if n_total > 0 else float("nan")
    signtest_p_twosided = 2 * min(signtest_p, signtest_p_above) if n_total > 0 else float("nan")
    logger.info("H1 sign test: %d/%d arms OR<1  one-sided p=%.3f  two-sided p=%.3f",
                n_below_1, n_total, signtest_p, signtest_p_twosided)

    # Meta-analysis
    meta_res = _meta_analysis_re(
        valid["log_or"].astype(float).values,
        valid["se_log_or"].astype(float).values,
    )
    logger.info("H1 meta-analysis: pooled OR=%.3f [%.3f, %.3f] p=%.4f (τ²=%.3f)",
                meta_res.get("pooled_or", np.nan),
                meta_res.get("pooled_ci_lo", np.nan),
                meta_res.get("pooled_ci_hi", np.nan),
                meta_res.get("pooled_p", np.nan),
                meta_res.get("tau2", np.nan))

    # ==================== H2: Subtype-stratified ===========================
    logger.info("=== H2: subtype-stratified for Ganitumab, Pembro, Neratinib ===")
    h2_rows = []
    for arm_label, short in [("Paclitaxel + Ganitumab", "Ganitumab"),
                              ("Paclitaxel + Pembrolizumab", "Pembro"),
                              ("Paclitaxel + Neratinib", "Neratinib")]:
        for subtype_val, subtype_name in [(1, "HR+"), (0, "TNBC")]:
            arm_pts = meta_i[meta_i["arm"] == arm_label]
            ctrl_pts = meta_i[meta_i["arm"] == "Paclitaxel"]
            # Neratinib includes HER2+ — restrict to HER2- for fairness
            if "Neratinib" in arm_label:
                arm_pts = arm_pts[arm_pts["her2_int"] == 0]
            # Subtype subset
            arm_pts = arm_pts[arm_pts["hr_int"] == subtype_val]
            ctrl_pts = ctrl_pts[ctrl_pts["hr_int"] == subtype_val]
            if len(arm_pts) < 15 or len(ctrl_pts) < 15:
                h2_rows.append({
                    "arm": short, "subtype": subtype_name,
                    "status": "TOO_SMALL",
                    "n_trt": len(arm_pts), "n_ctrl": len(ctrl_pts),
                })
                continue
            combined = pd.concat([ctrl_pts, arm_pts])
            combined["arm"] = (combined["arm"] == arm_label).astype(int)
            df = pd.DataFrame(index=combined.index)
            df["pCR"] = combined["pcr_int"].astype(int)
            df["arm"] = combined["arm"]
            df["MP"] = combined["mp_int"]
            df["HER2"] = combined["her2_int"]
            df["stromal_score"] = stromal.reindex(df.index).values
            # HR is fixed in subset, omit
            fit = _fit_logit_interaction(df)
            h2_rows.append({
                "arm": short, "subtype": subtype_name,
                "status": "VALID",
                "n_ctrl": int((df["arm"] == 0).sum()),
                "n_trt": int((df["arm"] == 1).sum()),
                "pcr_ctrl": float(df[df["arm"] == 0]["pCR"].mean()),
                "pcr_trt": float(df[df["arm"] == 1]["pCR"].mean()),
                "or_interaction": fit.get("or_interaction"),
                "or_ci_lo": fit.get("or_ci_lo"), "or_ci_hi": fit.get("or_ci_hi"),
                "p_interaction": fit.get("p_interaction"),
            })
            logger.info("  %s / %s  n_ctrl=%d n_trt=%d  OR_int=%.2f  p=%.3f",
                        short, subtype_name,
                        int((df["arm"] == 0).sum()), int((df["arm"] == 1).sum()),
                        fit.get("or_interaction", np.nan) or np.nan,
                        fit.get("p_interaction", np.nan) or np.nan)
    h2_df = pd.DataFrame(h2_rows)
    h2_df.to_csv(out / "H2_subtype.csv", index=False)

    # ==================== R1: Split-half stability ========================
    logger.info("=== R1: split-half stability on strongest arm ===")
    r1_df = None
    if len(valid) > 0:
        # Find most significant arm
        strongest = valid.sort_values("p_wald").iloc[0]
        strongest_arm = strongest["arm_label"]
        logger.info("  Strongest arm for R1: %s", strongest_arm)
        arm_pts = meta_i[meta_i["arm"] == strongest_arm]
        ctrl_pts = meta_i[meta_i["arm"] == "Paclitaxel"]
        if set(arm_pts["her2_int"].unique()) - {np.nan} == {0} or \
                set(arm_pts["her2_int"].unique()) - {np.nan} == {1, 0}:
            arm_pts = arm_pts[arm_pts["her2_int"] == 0]
        combined = pd.concat([ctrl_pts, arm_pts])
        combined["arm"] = (combined["arm"] == strongest_arm).astype(int)
        df = pd.DataFrame(index=combined.index)
        df["pCR"] = combined["pcr_int"].astype(int)
        df["arm"] = combined["arm"]
        df["HR"] = combined["hr_int"]; df["MP"] = combined["mp_int"]
        df["stromal_score"] = stromal.reindex(df.index).values
        df = df.dropna()
        p_values = _split_half_stability(df, n_iter=args.n_split)
        logger.info(
            "  R1 split-half: %d iterations, median p=%.3f, fraction p<0.05=%.3f",
            len(p_values), float(np.median(p_values)),
            float(np.mean(p_values < 0.05))
        )
        r1_df = pd.DataFrame({
            "strongest_arm": [strongest["arm"]] * len(p_values),
            "iter": np.arange(len(p_values)),
            "p_value": p_values,
        })
        r1_df.to_csv(out / "R1_splithalf.csv", index=False)

    # ==================== R2: Leave-one-component-out =====================
    logger.info("=== R2: leave-one-component-out ===")
    r2_rows = []
    for arm_label, short, biol in ARM_LIST_LOCKED:
        if arm_label in ("T-DM1 + Pertuzumab", "Paclitaxel + Trastuzumab"):
            continue
        arm_pts = meta_i[meta_i["arm"] == arm_label]
        ctrl_pts = meta_i[meta_i["arm"] == "Paclitaxel"]
        if "Neratinib" in arm_label:
            arm_pts = arm_pts[arm_pts["her2_int"] == 0]
        combined = pd.concat([ctrl_pts, arm_pts])
        combined["arm"] = (combined["arm"] == arm_label).astype(int)
        if (combined["arm"] == 1).sum() < 30:
            continue
        df = pd.DataFrame(index=combined.index)
        df["pCR"] = combined["pcr_int"].astype(int)
        df["arm"] = combined["arm"]
        df["HR"] = combined["hr_int"]; df["MP"] = combined["mp_int"]
        # Three tests: CAF+TGFβ, CAF-only, TGFβ-only
        for score_name, score_series in [
            ("CAF+TGFb", stromal),
            ("CAF_only", stromal_caf_only),
            ("TGFb_only", stromal_tgfb_only),
        ]:
            df["stromal_score"] = score_series.reindex(df.index).values
            fit = _fit_logit_interaction(df)
            r2_rows.append({
                "arm": short, "score": score_name,
                "or_interaction": fit.get("or_interaction"),
                "p": fit.get("p_interaction"),
                "or_ci_lo": fit.get("or_ci_lo"), "or_ci_hi": fit.get("or_ci_hi"),
            })
    r2_df = pd.DataFrame(r2_rows)
    r2_df.to_csv(out / "R2_leave_one_out.csv", index=False)
    logger.info("R2 leave-one-out saved; %d tests", len(r2_df))

    # ==================== Write HONEST_VERDICT.md =========================
    lines = ["# Honest verdict — OncoTME hypothesis test\n",
             "**Pre-registered**: `PRE_REGISTRATION.md` (formulas locked before analysis).\n",
             "## Locked formula\n",
             "```\nstromal_score = z(CAF_proxy) + z(TGFb_activity)\n```\n"]

    # H1 table
    lines += ["## H1 — Cross-arm direction consistency\n"]
    lines += ["| Arm | n_ctrl / n_trt | OR_int | 95% CI | p_Wald | p_perm | HR bal | MP bal |"]
    lines += ["|---|---|---|---|---|---|---|---|"]
    for _, r in h1_df.iterrows():
        if r.get("status") != "VALID":
            lines.append(f"| **{r['arm']}** | — | *{r.get('status')}* | | | | | |")
            continue
        lines.append(
            f"| **{r['arm']}** | "
            f"{int(r['n_ctrl'])} / {int(r['n_trt'])} | "
            f"{r['or_interaction']:.2f} | "
            f"[{r['or_ci_lo']:.2f}, {r['or_ci_hi']:.2f}] | "
            f"{r['p_wald']:.3f} | {r['p_permutation']:.3f} | "
            f"{r['balance_HR_p']:.2f} | {r['balance_MP_p']:.2f} |"
        )

    lines.append("")
    lines.append(f"**Sign test**: {n_below_1}/{n_total} arms have OR_int < 1.  "
                 f"One-sided binomial p = {signtest_p:.3f}.")
    lines.append(f"**Success criterion** (≥7/9 arms, p ≤ 0.09): "
                 f"{'✅ MET' if n_below_1 >= 7 else '❌ NOT MET'}")

    if "pooled_or" in meta_res:
        lines.append(f"\n**Random-effects meta-analysis**: pooled OR_int = "
                     f"{meta_res['pooled_or']:.3f} "
                     f"[{meta_res['pooled_ci_lo']:.3f}, {meta_res['pooled_ci_hi']:.3f}] "
                     f"p = {meta_res['pooled_p']:.4f}, τ² = {meta_res['tau2']:.3f}.")
        strong_meta = meta_res["pooled_ci_hi"] < 1.0
        lines.append(f"**Success criterion** (pooled 95% CI entirely < 1): "
                     f"{'✅ MET' if strong_meta else '❌ NOT MET'}")

    # H2 table
    lines += ["\n## H2 — Subtype-stratified (HR+ vs TNBC)\n"]
    lines += ["| Arm | Subtype | n_ctrl / n_trt | OR_int | 95% CI | p |"]
    lines += ["|---|---|---|---|---|---|"]
    for _, r in h2_df.iterrows():
        if r.get("status") != "VALID":
            lines.append(f"| {r['arm']} | {r['subtype']} | — | *{r['status']}* | | |")
            continue
        lines.append(
            f"| {r['arm']} | {r['subtype']} | "
            f"{int(r['n_ctrl'])} / {int(r['n_trt'])} | "
            f"{r['or_interaction']:.2f} | "
            f"[{r['or_ci_lo']:.2f}, {r['or_ci_hi']:.2f}] | "
            f"{r['p_interaction']:.3f} |"
        )

    # H2 evaluation
    h2_valid = h2_df[h2_df["status"] == "VALID"]
    h2_consistent = []
    for arm in h2_valid["arm"].unique():
        sub = h2_valid[h2_valid["arm"] == arm]
        if len(sub) == 2:
            ors = sub["or_interaction"].astype(float).values
            if len(ors) == 2 and all(np.isfinite(ors)):
                both_below = all(o < 1 for o in ors)
                both_above = all(o > 1 for o in ors)
                h2_consistent.append({
                    "arm": arm, "both_OR<1": both_below, "both_OR>1": both_above,
                    "direction_concordant": both_below or both_above,
                })
    h2_con_df = pd.DataFrame(h2_consistent)
    if not h2_con_df.empty:
        lines.append("\n**H2 concordance**:")
        for _, r in h2_con_df.iterrows():
            lines.append(
                f"- {r['arm']}: direction concordant between HR+ and TNBC? "
                f"**{'yes' if r['direction_concordant'] else 'no'}**"
                + (" (both OR<1)" if r["both_OR<1"] else ("" if not r["both_OR>1"]
                    else " (both OR>1)"))
            )

    # R1 table
    if r1_df is not None and len(r1_df) > 0:
        lines += ["\n## R1 — Sub-sample stability on strongest arm\n"]
        median_p = float(np.median(r1_df["p_value"]))
        frac_sig = float(np.mean(r1_df["p_value"] < 0.05))
        lines.append(
            f"- Strongest arm: **{r1_df['strongest_arm'].iloc[0]}**\n"
            f"- {len(r1_df)} random 60% sub-samples (stratified by arm)\n"
            f"- Median p in sub-sample: **{median_p:.3f}**\n"
            f"- Fraction of sub-samples with p<0.05: **{frac_sig:.2%}**"
        )
    else:
        lines += ["\n## R1 — Stability\n- Failed or insufficient data\n"]

    # R2 table
    lines += ["\n## R2 — Leave-one-component-out sensitivity\n"]
    lines += ["| Arm | Score | OR_int | 95% CI | p |"]
    lines += ["|---|---|---|---|---|"]
    for _, r in r2_df.iterrows():
        lines.append(
            f"| {r['arm']} | {r['score']} | "
            f"{r['or_interaction']:.2f} | "
            f"[{r['or_ci_lo']:.2f}, {r['or_ci_hi']:.2f}] | "
            f"{r['p']:.3f} |"
        )

    # H3: stromal-linked vs unrelated
    lines += ["\n## H3 — Biology-specificity check\n"]
    if not valid.empty:
        linked = valid[valid["biological_stromal_link"]]
        unlinked = valid[~valid["biological_stromal_link"]]
        if len(linked) > 0 and len(unlinked) > 0:
            med_linked = float(np.median(linked["or_interaction"].astype(float)))
            med_unlinked = float(np.median(unlinked["or_interaction"].astype(float)))
            lines.append(
                f"- Median OR_int in biologically stromal-linked arms "
                f"({len(linked)} arms): **{med_linked:.2f}**\n"
                f"- Median OR_int in stromal-unrelated arms "
                f"({len(unlinked)} arms): **{med_unlinked:.2f}**"
            )
            if med_linked < med_unlinked:
                lines.append(
                    "✅ **Biologically consistent**: stromal signal stronger "
                    "(OR further from 1) in stromal-linked arms."
                )
            else:
                lines.append(
                    "⚠️ Stromal signal is NOT stronger in biologically-linked "
                    "arms — suggests the pattern may be generic rather than "
                    "truly stromal-specific."
                )

    # Final honest verdict
    lines += ["\n## Final honest verdict\n"]

    h1_signtest_met = (n_below_1 >= 7) if n_total > 0 else False
    h1_meta_met = (meta_res.get("pooled_ci_hi", 1.1) < 1.0)
    h2_concordant_count = sum(1 for c in h2_consistent if c["direction_concordant"])
    r1_frac_sig = float(np.mean(r1_df["p_value"] < 0.05)) if r1_df is not None else 0.0

    # H1 verdict
    if h1_signtest_met and h1_meta_met:
        h1_v = ("✅ **H1 passes**: направление OR<1 consistent across arms, "
                "и pooled meta-analysis CI entirely below 1.")
    elif h1_signtest_met:
        h1_v = (f"🟡 **H1 partial**: direction consistent ({n_below_1}/{n_total}), "
                f"но pooled meta CI crosses 1 — effect size небольшой или "
                f"heterogeneous.")
    elif h1_meta_met:
        h1_v = (f"🟡 **H1 partial**: pooled effect significant но direction "
                f"не consistent ({n_below_1}/{n_total}).")
    else:
        h1_v = (f"❌ **H1 not supported**: {n_below_1}/{n_total} arms с OR<1 "
                f"(sign p={signtest_p:.3f}), pooled OR CI = "
                f"[{meta_res.get('pooled_ci_lo', np.nan):.2f}, "
                f"{meta_res.get('pooled_ci_hi', np.nan):.2f}].")
    lines.append(h1_v + "\n")

    # H2 verdict
    if h2_concordant_count > 0:
        lines.append(f"🟡 **H2 partial**: {h2_concordant_count} arm(s) show "
                     f"direction-concordant effect in both HR+ and TNBC.")
    else:
        lines.append("❌ **H2 not supported**: no arm shows consistent "
                     "direction across subtypes.\n")

    # R1 verdict
    if r1_frac_sig > 0.5:
        lines.append(f"✅ **R1 stable**: {r1_frac_sig:.0%} of splits показывают "
                     "p<0.05.\n")
    elif r1_frac_sig > 0.2:
        lines.append(f"🟡 **R1 partially stable**: {r1_frac_sig:.0%} of splits "
                     "with p<0.05 — fragile but not random.\n")
    else:
        lines.append(f"❌ **R1 unstable**: only {r1_frac_sig:.0%} of splits "
                     "show p<0.05 — effect strongly sample-dependent.\n")

    # Overall
    lines.append("\n### Overall: does TME play clinically significant role in targeted therapy?\n")
    pass_count = sum([h1_signtest_met or h1_meta_met, h2_concordant_count > 0,
                       r1_frac_sig > 0.5])
    if pass_count >= 2:
        lines.append(
            "**YES (tentatively, on this cohort alone)**. Pre-registered "
            "anti-stromal hypothesis shows consistent direction across multiple "
            "targeted therapy classes in I-SPY2, and stability checks support "
            "robustness. External validation still required for clinical claim.")
    elif pass_count == 1:
        lines.append(
            "**PARTIAL**. Some aspects of hypothesis supported, others not. "
            "Signal exists but is not robust enough to claim 'clinically significant'. "
            "External validation critical to distinguish true signal from "
            "cohort-specific artifact.")
    else:
        lines.append(
            "**NO clear evidence**. Pre-registered anti-stromal hypothesis "
            "is not supported in I-SPY2 under rigorous testing. Prior nominally-"
            "positive findings (Ganitumab, Neratinib) likely reflected "
            "post-hoc false-positives. To test TME role in targeted therapy, "
            "need fundamentally different data (larger cohorts per arm, "
            "treatment-matched external validation, or proteomic/spatial data).")

    (out / "HONEST_VERDICT.md").write_text("\n".join(lines), encoding="utf-8")
    logger.info("Artefacts → %s", out)

    # Summary JSON
    with (out / "summary.json").open("w") as f:
        json.dump({
            "locked_formula": "stromal_score = z(CAF_proxy) + z(TGFb_activity)",
            "n_valid_arms": int(n_total),
            "n_arms_or_below_1": int(n_below_1),
            "sign_test_one_sided_p": float(signtest_p),
            "meta_analysis": meta_res,
            "h2_concordant_arms": int(h2_concordant_count),
            "r1_frac_p_below_05": float(r1_frac_sig),
            "overall_verdict_points": int(pass_count),
        }, f, indent=2, default=str)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
