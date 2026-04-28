"""Figures for targeted-therapy-centric analysis.

1. Fig A: Forest plot of composite × arm interaction OR across 6 classes.
2. Fig B: Tertile ARR per class (small multiples).
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("fig")

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.titlesize": 11, "axes.labelsize": 10,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
    "figure.dpi": 120, "savefig.dpi": 300, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
})


def _save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path); fig.savefig(path.with_suffix(".png")); plt.close(fig)
    logger.info("saved %s", path.name)


def fig_forest(res: pd.DataFrame, out: Path):
    d = res.sort_values("or_interaction", ascending=True).copy()
    fig, ax = plt.subplots(figsize=(7, 0.6 * len(d) + 1))
    y = np.arange(len(d))
    or_vals = d["or_interaction"].values
    lo = d["or_ci_lo"].values
    hi = d["or_ci_hi"].values
    colors = []
    for _, r in d.iterrows():
        q = r.get("q_bh_class", np.nan)
        p = r.get("p_interaction", np.nan)
        if pd.notna(q) and q < 0.10:
            colors.append("#c0392b")  # BH-sig
        elif pd.notna(p) and p < 0.05:
            colors.append("#d98c00")  # nominal
        else:
            colors.append("#7f8c8d")

    ax.hlines(y, lo, hi, color=colors, linewidth=2.2, alpha=0.85)
    for i, c in enumerate(colors):
        ax.plot(or_vals[i], i, "o", color=c, markersize=7, markeredgecolor="black",
                markeredgewidth=0.6)
    ax.axvline(1.0, color="grey", linewidth=0.8, linestyle="--")
    labels = []
    for _, r in d.iterrows():
        q = r.get("q_bh_class", np.nan)
        p = r.get("p_interaction", np.nan)
        if pd.notna(q) and q < 0.10:
            tag = " ★★"
        elif pd.notna(p) and p < 0.05:
            tag = " ★"
        else:
            tag = ""
        labels.append(
            f"{r['class_name']}{tag}\n"
            f"n={int(r['n_ctrl'])}/{int(r['n_trt'])}  "
            f"p={p:.3f} q_BH={q:.3f}"
        )
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xscale("log")
    ax.set_xlim(0.5, 6)
    ax.set_xlabel("OR_interaction  (composite × arm)   log scale")
    ax.set_title(
        "Pre-specified TME composites × targeted therapy classes  (I-SPY2, N=986)\n"
        "★★ BH q<0.10,  ★ nominal p<0.05,  composite formulas biology-driven",
        loc="left",
    )
    _save(fig, out / "fig_targeted_forest.pdf")


def fig_tertile_panels(all_tert: pd.DataFrame, res: pd.DataFrame, out: Path):
    classes = res.sort_values("p_interaction")["class_name"].tolist()
    n = len(classes)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axs = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.4 * nrows),
                             squeeze=False)
    for idx, cls in enumerate(classes):
        ax = axs[idx // ncols, idx % ncols]
        sub = all_tert[all_tert["class_name"] == cls]
        if sub.empty:
            ax.axis("off"); continue
        sub = sub.sort_values(
            "tertile",
            key=lambda s: s.map({"low": 0, "med": 1, "high": 2}),
        )
        x = np.arange(len(sub))
        w = 0.38
        ctrl = sub["orr_ctrl"].values
        trt = sub["orr_trt"].values
        ax.bar(x - w/2, ctrl, w, color="#95a5a6",
               label="Paclitaxel", edgecolor="black", linewidth=0.4)
        ax.bar(x + w/2, trt, w, color="#2e86ab",
               label="+ targeted", edgecolor="black", linewidth=0.4)
        for i, (c, t, ar) in enumerate(zip(ctrl, trt, sub["arr"].values)):
            y_top = max(c, t) + 0.03
            color = "#2e7ab6" if ar > 0 else "#c0392b"
            ax.annotate(f"ARR\n{ar:+.0%}", xy=(i, y_top), ha="center", va="bottom",
                         fontsize=8, color=color, weight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(sub["tertile"].tolist())
        ax.set_ylim(0, 0.85)
        ax.set_ylabel("pCR rate")
        r = res[res["class_name"] == cls].iloc[0]
        title_q = r.get("q_bh_class", np.nan)
        title_p = r.get("p_interaction", np.nan)
        marker = "★★" if (pd.notna(title_q) and title_q < 0.10) else \
                 ("★" if (pd.notna(title_p) and title_p < 0.05) else "")
        ax.set_title(f"{cls}  {marker}\np_int={title_p:.3f}, q_BH={title_q:.3f}",
                      loc="left")
        if idx == 0:
            ax.legend(loc="upper left", frameon=False, fontsize=8)
    # hide empty subplots
    for j in range(n, nrows * ncols):
        axs[j // ncols, j % ncols].axis("off")
    fig.suptitle("TME composite tertile × targeted therapy response (ARR shown)",
                  y=1.01)
    _save(fig, out / "fig_targeted_tertile_panels.pdf")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir",
                    default=str(_REPO / "results" / "targeted_composites"))
    args = ap.parse_args()
    d = Path(args.in_dir)
    res = pd.read_csv(d / "targeted_composites_results.csv")
    all_tert = pd.read_csv(d / "all_tertile_effects.csv")

    fig_forest(res, d)
    fig_tertile_panels(all_tert, res, d)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
