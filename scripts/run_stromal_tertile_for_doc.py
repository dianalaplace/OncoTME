"""Compute tertile pCR/ARR per arm using pre-registered stromal_score.

stromal_score = z(CAF_proxy) + z(TGFb_activity)

Output: results/honest_test/stromal_tertiles.csv
        results/honest_test/H1_for_forest.csv (clean version)
"""

from __future__ import annotations

import gzip
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))

import numpy as np
import pandas as pd

from oncotme.features.signatures import load_gmt, ssgsea_scores


ARMS = [
    ("Pembrolizumab", ["Paclitaxel + Pembrolizumab"], False),
    ("Ganitumab", ["Paclitaxel + Ganitumab"], False),
    ("Neratinib (HER2-)", ["Paclitaxel + Neratinib"], True),
    ("MK-2206", ["Paclitaxel + MK-2206"], False),
    ("Ganetespib", ["Paclitaxel + Ganetespib"], False),
    ("ABT-888 + carboplatin", ["Paclitaxel + ABT 888 + Carboplatin"], False),
    ("AMG-386", ["Paclitaxel + AMG 386", "Paclitaxel + AMG-386"], False),
]


def parse_meta(path):
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


def load_gene(path):
    df = pd.read_csv(path, sep="\t", low_memory=False, index_col=0)
    df = df.apply(pd.to_numeric, errors="coerce")
    df.index = df.index.astype(str).str.upper()
    return df.loc[~df.index.duplicated(keep="first")].dropna(how="all")


def zscore(s):
    std = s.std()
    return (s - s.mean()) / (std if std else 1.0)


def main():
    out_dir = _REPO / "results" / "honest_test"
    out_dir.mkdir(parents=True, exist_ok=True)
    gene = load_gene(_REPO / "data/raw/geo/GSE194040_gene_level.txt.gz")
    meta = pd.concat([
        parse_meta(_REPO / "data/raw/geo/GSE194040-GPL20078_series_matrix.txt.gz"),
        parse_meta(_REPO / "data/raw/geo/GSE194040-GPL30493_series_matrix.txt.gz"),
    ])
    meta["resid"] = meta["patient id"].astype(str)
    meta["pcr_int"] = pd.to_numeric(meta["pcr"], errors="coerce")
    meta["her2_int"] = pd.to_numeric(meta["her2"], errors="coerce")
    meta["hr_int"] = pd.to_numeric(meta["hr"], errors="coerce")
    meta = meta.dropna(subset=["pcr_int"])
    meta = meta[meta["resid"].isin(gene.columns.astype(str))]
    print(f"Loaded {len(meta)} samples")

    gmt = load_gmt(_REPO / "signatures" / "tme_signatures.gmt")
    scores = ssgsea_scores(gene[meta["resid"].tolist()], gene_sets=gmt)
    stromal = zscore(scores["CAF_proxy"]) + zscore(scores["TGFb_activity"])
    stromal.name = "stromal_score"

    meta_i = meta.set_index("resid")
    rows = []
    for short, labels, restrict_her2neg in ARMS:
        arm = meta_i[meta_i["arm"].isin(labels)]
        if restrict_her2neg:
            arm = arm[arm["her2_int"] == 0]
        ctrl = meta_i[(meta_i["arm"] == "Paclitaxel") & (meta_i["her2_int"] == 0)]
        df = pd.concat([ctrl, arm])
        df["arm_bin"] = (df["arm"] != "Paclitaxel").astype(int)
        df["stromal"] = stromal.reindex(df.index).values
        df["pCR"] = df["pcr_int"].astype(int)
        df = df.dropna(subset=["stromal", "pCR"])

        # tertiles based on combined (control+treatment) distribution to ensure
        # each tertile has both control and treatment samples
        try:
            df["tertile"] = pd.qcut(df["stromal"], q=3,
                                     labels=["Низкий", "Средний", "Высокий"],
                                     duplicates="drop")
        except Exception:
            continue

        for t in ["Низкий", "Средний", "Высокий"]:
            sub = df[df["tertile"] == t]
            c = sub[sub["arm_bin"] == 0]
            tt = sub[sub["arm_bin"] == 1]
            if len(c) < 3 or len(tt) < 3:
                continue
            orr_c = float(c["pCR"].mean())
            orr_t = float(tt["pCR"].mean())
            arr = orr_t - orr_c
            # Haldane-Anscombe OR with CI
            a = tt["pCR"].sum() + 0.5
            b = (len(tt) - tt["pCR"].sum()) + 0.5
            cc = c["pCR"].sum() + 0.5
            e = (len(c) - c["pCR"].sum()) + 0.5
            or_val = (a/b) / (cc/e)
            se = float(np.sqrt(1/a + 1/b + 1/cc + 1/e))
            rows.append({
                "arm": short, "tertile": t,
                "n_ctrl": int(len(c)), "n_trt": int(len(tt)),
                "pcr_ctrl_pct": round(orr_c * 100, 1),
                "pcr_trt_pct": round(orr_t * 100, 1),
                "arr_pp": round(arr * 100, 1),
                "or": round(or_val, 2),
                "or_ci_lo": round(float(np.exp(np.log(or_val) - 1.96 * se)), 2),
                "or_ci_hi": round(float(np.exp(np.log(or_val) + 1.96 * se)), 2),
            })
    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "stromal_tertiles.csv", index=False)
    print(out.to_string(index=False))
    print(f"\nSaved {len(out)} rows to stromal_tertiles.csv")


if __name__ == "__main__":
    main()
