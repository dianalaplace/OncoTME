"""Генерирует publication-grade фигуры из артефактов run_discovery_nmf.py.

Создаёт:
- ``fig_nmf_stability.pdf`` — cophenetic + explained variance по k
- ``fig_factor_basis_heatmap.pdf`` — H matrix (factor × signature)
- ``fig_factor_loadings_boxplot.pdf`` — W distributions
- ``fig_tertile_km_{os,dfs}.pdf`` — KM кривые по тертилям risk score
- ``fig_dca_{os,dfs}.pdf`` — Decision Curve Analysis
- ``fig_risk_coefs_{os,dfs}.pdf`` — forest plot Cox коэффициентов

Usage::

    python scripts/make_discovery_figures.py --in results/discovery_nmf/tcga_brca/
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
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"))
    plt.close(fig)
    logger.info("saved %s", path.name)


def fig_nmf_stability(in_dir: Path, out_dir: Path) -> None:
    p = in_dir / "nmf_stability_by_k.csv"
    if not p.exists():
        return
    df = pd.read_csv(p)
    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax2 = ax1.twinx()
    ax1.plot(df["k"], df["cophenetic"], "o-", color="#2e7ab6", label="Cophenetic")
    ax1.plot(df["k"], df["stability"], "s--", color="#66a6db", label="Mean W stability")
    ax2.plot(df["k"], df["explained_variance"], "^:", color="#c85a5a",
             label="Explained variance")
    ax1.set_xlabel("Number of NMF factors (k)")
    ax1.set_ylabel("Stability")
    ax2.set_ylabel("Explained variance", color="#c85a5a")
    ax1.set_ylim(0, 1.05)
    ax2.set_ylim(0.85, 1.0)
    ax1.axhline(0.90, color="grey", linestyle=":", linewidth=0.8)
    ax1.text(df["k"].max() - 0.3, 0.905, "Brunet threshold", fontsize=8, color="grey")
    ax1.legend(loc="lower left")
    ax2.legend(loc="lower right")
    ax1.set_title("NMF factor stability across k")
    _save(fig, out_dir / "fig_nmf_stability.pdf")


def fig_factor_basis_heatmap(in_dir: Path, out_dir: Path) -> None:
    p = in_dir / "factor_basis_H.csv"
    if not p.exists():
        return
    H = pd.read_csv(p, index_col=0)
    # нормируем по колонкам (сигнатурам) — показывает, какой фактор доминирует в каждой
    Hn = H.div(H.sum(axis=0).replace(0, 1.0), axis=1)
    # сокращаем имена сигнатур для удобства
    Hn.columns = [c.replace("sig__", "") for c in Hn.columns]
    fig, ax = plt.subplots(figsize=(max(6, Hn.shape[1] * 0.35), 0.6 + Hn.shape[0] * 0.5))
    sns.heatmap(
        Hn, cmap="rocket_r", annot=False,
        cbar_kws={"label": "Column-normalised loading"}, ax=ax,
        linewidths=0.3, linecolor="white",
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("TME latent factor composition (H)")
    plt.setp(ax.get_xticklabels(), rotation=55, ha="right")
    _save(fig, out_dir / "fig_factor_basis_heatmap.pdf")


def fig_factor_loadings_boxplot(in_dir: Path, out_dir: Path) -> None:
    p = in_dir / "factor_loadings_W.csv"
    if not p.exists():
        return
    W = pd.read_csv(p, index_col=0)
    long = W.melt(var_name="Factor", value_name="Loading")
    fig, ax = plt.subplots(figsize=(max(6, W.shape[1] * 0.9), 4))
    sns.boxplot(data=long, x="Factor", y="Loading", ax=ax,
                color="#9bc4e2", fliersize=2, linewidth=1.2)
    ax.set_title("Per-patient loadings by factor (TCGA-BRCA)")
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
    _save(fig, out_dir / "fig_factor_loadings_boxplot.pdf")


def _fig_km_for_endpoint(in_dir: Path, out_dir: Path, endpoint: str) -> None:
    p = in_dir / f"tertile_km_{endpoint}.csv"
    pair = in_dir / f"tertile_pairwise_{endpoint}.csv"
    if not p.exists():
        return
    df = pd.read_csv(p)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    palette = {"low": "#3a8d4d", "medium": "#d9a24b", "high": "#c0392b"}
    for level, sub in df.groupby("tertile"):
        sub = sub.sort_values(sub.columns[0])
        time_col = sub.columns[0]
        surv_col = [c for c in sub.columns if c not in {time_col, "tertile"}][0]
        ax.step(sub[time_col], sub[surv_col], where="post",
                color=palette.get(level, "grey"),
                label=f"{level.capitalize()} risk")
    ax.set_xlabel("Time (days)" if endpoint == "os" else f"Time ({endpoint.upper()})")
    ax.set_ylabel(f"{endpoint.upper()} probability")
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower left")

    # annotation with log-rank p-value если доступно
    summary_json = in_dir / "discovery_summary.json"
    if summary_json.exists():
        import json
        s = json.loads(summary_json.read_text())
        if endpoint in s:
            p_lr = s[endpoint].get("tertile_logrank_p", None)
            c = s[endpoint].get("c_index", None)
            ci = s[endpoint].get("c_index_ci", [None, None])
            txt = []
            if p_lr is not None:
                txt.append(f"log-rank p = {p_lr:.2e}")
            if c is not None:
                txt.append(f"C-index = {c:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]")
            ax.text(0.98, 0.95, "\n".join(txt), transform=ax.transAxes,
                    ha="right", va="top", fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="grey"))
    ax.set_title(f"TME-risk-score tertile Kaplan–Meier ({endpoint.upper()})")
    _save(fig, out_dir / f"fig_tertile_km_{endpoint}.pdf")


def _fig_dca_for_endpoint(in_dir: Path, out_dir: Path, endpoint: str) -> None:
    p = in_dir / f"dca_{endpoint}.csv"
    if not p.exists():
        return
    dca = pd.read_csv(p)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(dca["threshold"], dca["nb_model"], "o-", color="#2e7ab6",
            label="TME risk score")
    ax.plot(dca["threshold"], dca["nb_treat_all"], "--", color="#c0392b",
            label="Treat all")
    ax.plot(dca["threshold"], dca["nb_treat_none"], ":", color="grey",
            label="Treat none")
    ax.set_xlabel("Threshold probability")
    ax.set_ylabel("Net benefit")
    ax.set_title(f"Decision Curve Analysis ({endpoint.upper()})")
    ax.legend()
    ax.axhline(0, color="grey", linewidth=0.5)
    _save(fig, out_dir / f"fig_dca_{endpoint}.pdf")


def _fig_risk_coefs(in_dir: Path, out_dir: Path, endpoint: str) -> None:
    p = in_dir / f"risk_coefs_{endpoint}.csv"
    if not p.exists():
        return
    df = pd.read_csv(p, index_col=0).sort_values("coefficient")
    fig, ax = plt.subplots(figsize=(6, 0.4 + 0.35 * len(df)))
    colors = ["#c0392b" if v > 0 else "#2e7ab6" for v in df["coefficient"]]
    ax.barh(df.index, df["coefficient"], color=colors, alpha=0.85, edgecolor="black",
            linewidth=0.5)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Cox log-hazard coefficient")
    ax.set_title(f"Factor contributions to risk score ({endpoint.upper()})")
    _save(fig, out_dir / f"fig_risk_coefs_{endpoint}.pdf")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_dir", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    in_dir = Path(args.in_dir)
    out_dir = Path(args.out) if args.out else in_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    fig_nmf_stability(in_dir, out_dir)
    fig_factor_basis_heatmap(in_dir, out_dir)
    fig_factor_loadings_boxplot(in_dir, out_dir)
    for ep in ["os", "dfs"]:
        _fig_km_for_endpoint(in_dir, out_dir, ep)
        _fig_dca_for_endpoint(in_dir, out_dir, ep)
        _fig_risk_coefs(in_dir, out_dir, ep)

    logger.info("All figures → %s", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
