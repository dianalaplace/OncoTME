"""Generate publication figures for the clinical document.

1. Forest plot of OR_int across 7 arms (from H1_cross_arm.csv)
2. Tertile bars for Pembro and Ganitumab (from stromal_tertiles.csv)
3. Patho-mechanism schema (text-based diagram)

Output PNG files at high DPI for embedding into docx.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def fig_forest(out: Path):
    df = pd.read_csv(_REPO / "results/honest_test/H1_cross_arm.csv")
    df = df[df["status"] == "VALID"].copy()
    df["short"] = df["arm"]
    rename = {
        "Pembro": "Пембролизумаб (анти-PD-1)",
        "Ganitumab": "Ганитумаб (анти-IGF-1R)",
        "Neratinib": "Нератиниб (pan-HER TKI)",
        "AKT": "MK-2206 (AKT-ингибитор)",
        "Ganetespib": "Ганетеспиб (HSP90)",
        "PARPi": "ABT-888 + карбоплатин",
        "AMG386": "AMG-386 (анти-ангиопоэтин)",
    }
    df["arm_ru"] = df["arm"].map(rename)
    df = df.sort_values("or_interaction", ascending=False).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    y = np.arange(len(df))
    or_vals = df["or_interaction"].values
    lo = df["or_ci_lo"].values
    hi = df["or_ci_hi"].values
    p_vals = df["p_wald"].values

    for i, (or_v, lo_v, hi_v, p_v) in enumerate(zip(or_vals, lo, hi, p_vals)):
        if p_v < 0.05:
            color = "#c0392b"
        elif p_v < 0.10:
            color = "#d68910"
        else:
            color = "#7f8c8d"
        ax.hlines(y[i], lo_v, hi_v, color=color, linewidth=2.5, alpha=0.85)
        ax.plot(or_v, y[i], "s", color=color, markersize=10,
                markeredgecolor="black", markeredgewidth=0.6)

    ax.axvline(1.0, color="black", linewidth=1.0, linestyle="--", alpha=0.7)

    pooled_or = 0.621
    pooled_lo, pooled_hi = 0.486, 0.795
    ax.hlines(-0.8, pooled_lo, pooled_hi, color="#1F3864", linewidth=3.0)
    ax.plot(pooled_or, -0.8, "D", color="#1F3864", markersize=12,
            markeredgecolor="black", markeredgewidth=0.8)

    labels = []
    for _, r in df.iterrows():
        n_total = int(r["n_ctrl"]) + int(r["n_trt"])
        labels.append(f"{r['arm_ru']}\n(n = {n_total}, p = {r['p_wald']:.3f})")
    labels.append("Пул (мета-анализ)\nOR 0,62 (95% ДИ 0,49 до 0,80), p < 0,001")

    ticks = list(y) + [-0.8]
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels)

    ax.set_xscale("log")
    ax.set_xlim(0.20, 2.5)
    xticks = [0.25, 0.5, 1.0, 2.0]
    ax.set_xticks(xticks)
    ax.set_xticklabels([str(x) for x in xticks])
    ax.set_xlabel("Отношение шансов взаимодействия (стромальный индекс × терапия), log-шкала")
    ax.set_title("Эффект стромального индекса по классам таргетной терапии (I-SPY2, n = 986)\n"
                 "красный: p < 0,05; оранжевый: p < 0,10; серый: p >= 0,10",
                 loc="left", fontsize=11)

    ax.text(2.45, len(df) - 0.5, "OR > 1: эффект сильнее\nпри высоком CAF",
            fontsize=8, ha="right", color="#888888", style="italic")
    ax.text(0.21, len(df) - 0.5, "OR < 1: эффект слабее\nпри высоком CAF",
            fontsize=8, ha="left", color="#888888", style="italic")

    ax.set_ylim(-1.6, len(df) - 0.4)
    plt.tight_layout()
    fig.savefig(out / "fig1_forest.png", dpi=300, bbox_inches="tight")
    fig.savefig(out / "fig1_forest.pdf")
    plt.close(fig)
    print("Saved fig1_forest")


def fig_tertile_bars(out: Path):
    df = pd.read_csv(_REPO / "results/honest_test/stromal_tertiles.csv")
    arms_to_plot = [
        ("Pembrolizumab", "Пембролизумаб + паклитаксел"),
        ("Ganitumab", "Ганитумаб + паклитаксел"),
    ]
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, (arm_key, arm_label) in zip(axs, arms_to_plot):
        sub = df[df["arm"] == arm_key].copy()
        sub["tertile_order"] = sub["tertile"].map({"Низкий": 0, "Средний": 1, "Высокий": 2})
        sub = sub.sort_values("tertile_order")
        x = np.arange(len(sub))
        w = 0.38
        bars_c = ax.bar(x - w/2, sub["pcr_ctrl_pct"].values, w,
                        color="#a5a5a5", edgecolor="black", linewidth=0.5,
                        label="Только химиотерапия")
        bars_t = ax.bar(x + w/2, sub["pcr_trt_pct"].values, w,
                        color="#2e86ab", edgecolor="black", linewidth=0.5,
                        label="Химиотерапия + таргетная")
        for i, row in sub.reset_index(drop=True).iterrows():
            ax.text(i - w/2, row["pcr_ctrl_pct"] + 1.5,
                    f"{row['pcr_ctrl_pct']:.0f}%",
                    ha="center", fontsize=9)
            ax.text(i + w/2, row["pcr_trt_pct"] + 1.5,
                    f"{row['pcr_trt_pct']:.0f}%",
                    ha="center", fontsize=9)
            arr = row["arr_pp"]
            color = "#1F8A1F" if arr > 0 else "#c0392b"
            y_top = max(row["pcr_ctrl_pct"], row["pcr_trt_pct"]) + 8
            ax.text(i, y_top, f"ARR\n{arr:+.0f} п.п.",
                    ha="center", va="bottom", fontsize=9.5,
                    color=color, weight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([t for t in sub["tertile"]])
        ax.set_ylabel("Частота pCR, %")
        ax.set_title(arm_label, loc="left")
        ax.set_ylim(0, 80)
        if ax is axs[0]:
            ax.legend(loc="upper right", frameon=False, fontsize=9)
    fig.suptitle("Распределение клинической пользы по тертилям стромального индекса",
                 fontsize=12, y=1.02)
    plt.tight_layout()
    fig.savefig(out / "fig2_tertile_bars.png", dpi=300, bbox_inches="tight")
    fig.savefig(out / "fig2_tertile_bars.pdf")
    plt.close(fig)
    print("Saved fig2_tertile_bars")


def fig_subtype(out: Path):
    df = pd.read_csv(_REPO / "results/honest_test/H2_subtype.csv")
    df = df[df["status"] == "VALID"].copy()
    rename = {"Pembro": "Пембролизумаб", "Ganitumab": "Ганитумаб", "Neratinib": "Нератиниб"}
    df["arm_ru"] = df["arm"].map(rename)
    df["subtype_ru"] = df["subtype"].map({"HR+": "HR+/HER2-", "TNBC": "ТНРМЖ"})

    fig, ax = plt.subplots(figsize=(8, 4.5))
    y_positions = []
    y = 0
    for arm in df["arm_ru"].unique():
        for st in ["HR+/HER2-", "ТНРМЖ"]:
            sub = df[(df["arm_ru"] == arm) & (df["subtype_ru"] == st)]
            if sub.empty:
                continue
            r = sub.iloc[0]
            color = "#c0392b" if r["p_interaction"] < 0.05 else "#7f8c8d"
            ax.hlines(y, r["or_ci_lo"], r["or_ci_hi"], color=color, linewidth=2.2)
            ax.plot(r["or_interaction"], y, "s", color=color, markersize=9,
                    markeredgecolor="black", markeredgewidth=0.5)
            y_positions.append((y, f"{arm}, {st} (n = {int(r['n_ctrl']) + int(r['n_trt'])}, p = {r['p_interaction']:.3f})"))
            y += 1
        y += 0.5
    ax.axvline(1.0, color="black", linewidth=1.0, linestyle="--", alpha=0.6)
    ax.set_yticks([yp[0] for yp in y_positions])
    ax.set_yticklabels([yp[1] for yp in y_positions])
    ax.set_xscale("log")
    ax.set_xlim(0.08, 8)
    ax.set_xlabel("OR взаимодействия (log-шкала)")
    ax.set_title("Подтипоспецифичность стромального биомаркера",
                 loc="left", fontsize=11)
    plt.tight_layout()
    fig.savefig(out / "fig3_subtype.png", dpi=300, bbox_inches="tight")
    fig.savefig(out / "fig3_subtype.pdf")
    plt.close(fig)
    print("Saved fig3_subtype")


if __name__ == "__main__":
    out = _REPO / "results" / "doc_figures"
    out.mkdir(parents=True, exist_ok=True)
    fig_forest(out)
    fig_tertile_bars(out)
    fig_subtype(out)
    print(f"\nAll figures in {out}")
