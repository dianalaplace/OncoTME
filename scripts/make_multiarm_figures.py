"""Publication-grade figures для multi-arm spectrum.

1. Fig A: Forest plot — composite × arm interaction OR (all 9 arms).
2. Fig B: Heatmap — arm × TME signature interaction OR (9 × 23).
3. Fig C: Pembro-specific tertile bars (composite low/med/high × control/pembro ORR).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("figures")

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
    "figure.dpi": 120, "savefig.dpi": 300, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
})


def _save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path); fig.savefig(path.with_suffix(".png")); plt.close(fig)
    logger.info("saved %s", path.name)


def fig_forest_composite(multi: pd.DataFrame, out: Path):
    d = multi.sort_values("composite_or_interaction", ascending=True).copy()
    fig, ax = plt.subplots(figsize=(7.5, 0.45 * len(d) + 1))
    y = np.arange(len(d))
    or_vals = d["composite_or_interaction"].values
    lo = d["composite_or_ci_lo"].values
    hi = d["composite_or_ci_hi"].values
    colors = ["#c0392b" if p < 0.05 else "#7f8c8d" for p in d["composite_p_interaction"]]

    ax.hlines(y, lo, hi, color=colors, linewidth=2, alpha=0.85)
    ax.plot(or_vals, y, "o", color="black", markersize=6, markerfacecolor="white", mew=1.5)
    for i, c in enumerate(colors):
        ax.plot(or_vals[i], i, "o", color=c, markersize=6)
    ax.axvline(1.0, color="grey", linewidth=0.8, linestyle="--")

    labels = []
    for _, r in d.iterrows():
        p = r["composite_p_interaction"]
        star = " ★" if p < 0.05 else ""
        labels.append(
            f"{r['arm_name']}  [{r['biology']}]\n"
            f"n={int(r['n_ctrl'])}/{int(r['n_trt'])}  p_int={p:.3f}{star}"
        )
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xscale("log")
    ax.set_xlim(0.3, 10)
    ax.set_xlabel("OR_interaction (composite score × arm)   log scale")
    ax.set_title(
        "Pembrolizumab is the only arm whose response is modulated\n"
        "by composite ICI-TME-readiness score (★ = p<0.05)",
        loc="left",
    )
    _save(fig, out / "fig_A_forest_composite.pdf")


def fig_heatmap_arm_x_signature(all_sigs: pd.DataFrame, out: Path):
    pivot_or = all_sigs.pivot(index="signature", columns="arm_name",
                               values="or_interaction")
    pivot_p = all_sigs.pivot(index="signature", columns="arm_name",
                              values="p_interaction")
    # order arms by strongest interaction (min p)
    arm_order = pivot_p.min().sort_values().index.tolist()
    pivot_or = pivot_or[arm_order]
    pivot_p = pivot_p[arm_order]
    # order signatures by max |log OR| cross arms
    sig_rank = pivot_or.apply(lambda r: (np.log(r).abs()).max(), axis=1)
    sig_order = sig_rank.sort_values(ascending=False).index.tolist()
    pivot_or = pivot_or.loc[sig_order]
    pivot_p = pivot_p.loc[sig_order]

    # color: log OR, diverging
    data = np.log(pivot_or.values)
    vmax = float(np.nanmax(np.abs(data))) * 0.9
    fig, ax = plt.subplots(figsize=(0.8 * len(arm_order) + 2,
                                      0.3 * len(sig_order) + 1))
    im = sns.heatmap(
        pivot_or.apply(np.log), annot=pivot_or.map(
            lambda x: f"{x:.1f}" if pd.notna(x) else ""),
        fmt="", cmap="RdBu_r", center=0, vmin=-vmax, vmax=vmax,
        cbar_kws={"label": "log(OR_interaction)"},
        linewidths=0.3, linecolor="white",
        annot_kws={"fontsize": 7},
        ax=ax,
    )
    # overlay stars for p<0.05
    for i, sig in enumerate(pivot_or.index):
        for j, arm in enumerate(pivot_or.columns):
            p = pivot_p.loc[sig, arm]
            if pd.notna(p) and p < 0.05:
                ax.text(j + 0.5, i + 0.85, "★", ha="center", va="top",
                        fontsize=8, color="black")
    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    ax.set_title("Arm-specific TME interaction — each targeted class has its own "
                 "biology\n(★ = p<0.05, numbers = OR_int; blue=↓benefit, red=↑benefit)",
                 loc="left", fontsize=10)
    _save(fig, out / "fig_B_heatmap_arm_x_sig.pdf")


def fig_pembro_tertile_bars(tertile_df: pd.DataFrame, out: Path):
    # rebuild wide
    d = tertile_df.copy()
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    tert_order = ["low", "med", "high"]
    tert_labels = ["Low\n(TME-cold)", "Medium", "High\n(TME-hot)"]
    x = np.arange(len(tert_order))
    w = 0.38
    ctrl_orr = []; trt_orr = []
    n_ctrl = []; n_trt = []; arr = []
    # Support both column naming conventions (trt/pembro):
    trt_orr_col = "orr_pembro" if "orr_pembro" in d.columns else "orr_trt"
    n_trt_col = "n_pembro" if "n_pembro" in d.columns else "n_trt"
    tert_col = "tertile"
    # composite_tertile_effects.csv uses 'medium'; normalise to 'med'
    d[tert_col] = d[tert_col].replace({"medium": "med"})
    for t in tert_order:
        row = d[d[tert_col] == t].iloc[0] if (d[tert_col] == t).any() else None
        if row is None:
            ctrl_orr.append(np.nan); trt_orr.append(np.nan)
            n_ctrl.append(0); n_trt.append(0); arr.append(np.nan)
        else:
            ctrl_orr.append(row["orr_ctrl"]); trt_orr.append(row[trt_orr_col])
            n_ctrl.append(int(row["n_ctrl"])); n_trt.append(int(row[n_trt_col]))
            arr.append(row["arr"])
    bars_c = ax.bar(x - w / 2, ctrl_orr, w, color="#95a5a6", label="Paclitaxel",
                     edgecolor="black", linewidth=0.5)
    bars_t = ax.bar(x + w / 2, trt_orr, w, color="#2e86ab", label="Paclitaxel + Pembro",
                     edgecolor="black", linewidth=0.5)
    for i, (c, t, ar) in enumerate(zip(ctrl_orr, trt_orr, arr)):
        ax.annotate(f"n={n_ctrl[i]}", xy=(i - w / 2, c), xytext=(0, 3),
                     textcoords="offset points", ha="center", fontsize=8)
        ax.annotate(f"n={n_trt[i]}", xy=(i + w / 2, t), xytext=(0, 3),
                     textcoords="offset points", ha="center", fontsize=8)
        # ARR label
        if not np.isnan(ar):
            y_top = max(c, t) + 0.05
            color = "#2e7ab6" if ar > 0 else "#c0392b"
            ax.annotate(
                f"ARR {ar:+.1%}", xy=(i, y_top), ha="center", va="bottom",
                fontsize=9, color=color, weight="bold",
            )
    ax.set_xticks(x)
    ax.set_xticklabels(tert_labels)
    ax.set_ylim(0, 0.90)
    ax.set_ylabel("pCR rate")
    ax.set_xlabel("Composite ICI-TME-readiness score tertile")
    ax.legend(loc="upper left", frameon=False)
    ax.set_title("Pembrolizumab benefit is restricted to composite-high patients",
                 loc="left")
    _save(fig, out / "fig_C_pembro_tertile_bars.pdf")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--multiarm-csv",
                    default=str(_REPO / "results" / "ispy2_multiarm" / "multiarm_composite.csv"))
    ap.add_argument("--sigs-csv",
                    default=str(_REPO / "results" / "ispy2_multiarm" / "signatures_all_arms.csv"))
    ap.add_argument("--tertile-csv",
                    default=str(_REPO / "results" / "ispy2_composite" /
                                 "composite_tertile_effects.csv"))
    ap.add_argument("--out-dir",
                    default=str(_REPO / "results" / "ispy2_figures"))
    args = ap.parse_args()
    out = Path(args.out_dir)

    multi = pd.read_csv(args.multiarm_csv)
    all_sigs = pd.read_csv(args.sigs_csv)
    tertile = pd.read_csv(args.tertile_csv)

    fig_forest_composite(multi, out)
    fig_heatmap_arm_x_signature(all_sigs, out)
    fig_pembro_tertile_bars(tertile, out)
    logger.info("All figures → %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
