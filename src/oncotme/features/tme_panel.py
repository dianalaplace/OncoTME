"""Сборка единой TME-feature-frame.

Вход: CohortBundle + config/signatures.yaml.
Выход: DataFrame sample_id × {TME_signatures ∪ baselines ∪ individual_genes ∪ clinical}.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd
import yaml

from ..cohorts.base import CohortBundle
from .signatures import load_gmt, fallback_zscore


def build_tme_features(
    cohort: CohortBundle,
    signatures_config_path: str | Path = "config/signatures.yaml",
    use_ssgsea: bool = True,
) -> pd.DataFrame:
    """Собирает единый feature-frame для одной когорты.

    Колонки группируются по модулям, имена не меняются — чтобы потом
    ``models/blocks.py`` мог делать block-ablation через префиксы.

    Схема имён колонок:
      - ``sig__<name>`` — сигнатурные ssGSEA скоры
      - ``gene__<symbol>`` — прямая экспрессия индивидуального гена
      - ``baseline__<name>`` — эталонные сигнатуры (TIS, CYT, ESTIMATE)
      - ``clin__<name>`` — клинические ковариаты
    """
    cfg = _load_yaml(signatures_config_path)
    gmt = load_gmt(cfg["gmt_file"])

    modules: Dict[str, List[str]] = cfg.get("modules", {})
    module_signatures = {
        sig_name: gmt[sig_name]
        for mod_sigs in modules.values()
        for sig_name in mod_sigs
        if sig_name in gmt
    }

    # ssGSEA или fallback
    if use_ssgsea:
        # TODO: вернуть к ssGSEA, когда signatures.ssgsea_scores будет портирован
        sig_scores = fallback_zscore(cohort.expr, module_signatures)
    else:
        sig_scores = fallback_zscore(cohort.expr, module_signatures)
    sig_scores.columns = [f"sig__{c}" for c in sig_scores.columns]

    # Индивидуальные гены
    individual_genes = _flatten_individual_genes(cfg.get("individual_genes", {}))
    present = [g for g in individual_genes if g in cohort.expr.index]
    gene_frame = cohort.expr.loc[present].T
    gene_frame.columns = [f"gene__{g}" for g in gene_frame.columns]

    # Baselines — пока заглушка; реализуется в features/baselines.py
    baseline_frame = _compute_baselines_stub(cohort, cfg.get("baselines", {}))

    # Клинические
    clin_cols = [c for c in cfg.get("clinical_covariates", []) if c in cohort.clinical.columns]
    clin_frame = cohort.clinical[clin_cols].copy()
    clin_frame.columns = [f"clin__{c}" for c in clin_frame.columns]

    features = pd.concat(
        [sig_scores, gene_frame, baseline_frame, clin_frame],
        axis=1,
    )
    features.index.name = "sample_id"
    return features


def _load_yaml(path: str | Path) -> dict:
    with Path(path).open() as f:
        return yaml.safe_load(f)


def _flatten_individual_genes(section: dict) -> List[str]:
    genes: List[str] = []
    for _group, glist in section.items():
        genes.extend(glist)
    return list(dict.fromkeys(genes))  # уникальные, порядок сохранён


def _compute_baselines_stub(_cohort: CohortBundle, _baselines_cfg: dict) -> pd.DataFrame:
    """Заглушка. Реальная реализация в features/baselines.py (TIS, CYT, ESTIMATE)."""
    return pd.DataFrame()
