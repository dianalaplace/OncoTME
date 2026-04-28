"""Feature-block ablation.

Блоки фичей определяются префиксами в имени колонок (см. features/tme_panel.py):
  - ``sig__``   → сигнатурные (immune / stromal / APM — подмодули из config)
  - ``gene__``  → индивидуальные гены
  - ``baseline__`` → эталонные сигнатуры (TIS, CYT, ESTIMATE)
  - ``clin__``  → клинические

Функции:
- ``select_block(X, block)`` → подматрица только нужного блока
- ``ablation_grid(X, blocks_to_test)`` → список конфигураций (clinical, immune, …, full)
"""

from __future__ import annotations

from typing import Dict, List

import pandas as pd


BLOCK_PREFIXES: Dict[str, List[str]] = {
    "clinical": ["clin__"],
    "immune": ["sig__B_cell", "sig__CD4_T_cell", "sig__CD8_T_cell", "sig__NK_cell",
               "sig__Treg", "sig__Dendritic", "sig__M2_macrophage", "sig__MDSC_proxy",
               "sig__TLS_signature", "sig__IFN_gamma"],
    "stromal": ["sig__CAF_proxy", "sig__TGFb", "sig__angiogenesis", "sig__hypoxia",
                "sig__CXCL1_MDSC_axis", "sig__M2_polarization"],
    "apm": ["sig__HLA_I_score", "sig__HLA_II_score", "sig__APM_score",
            "sig__checkpoint_score", "gene__HLA-", "gene__B2M", "gene__TAP",
            "gene__PSMB", "gene__CIITA", "gene__NLRC5"],
    "baseline": ["baseline__"],
}


def select_block(X: pd.DataFrame, block: str) -> pd.DataFrame:
    """Подматрица по префиксам блока."""
    prefixes = BLOCK_PREFIXES[block]
    cols = [c for c in X.columns if any(c.startswith(p) for p in prefixes)]
    return X[cols]


def ablation_grid(X: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Возвращает словарь ``{config_name: X_subset}`` для сравнения в экспериментах E2."""
    immune = select_block(X, "immune")
    stromal = select_block(X, "stromal")
    apm = select_block(X, "apm")
    clinical = select_block(X, "clinical")

    return {
        "clinical_only": clinical,
        "immune_only": immune,
        "stromal_only": stromal,
        "apm_only": apm,
        "tme_no_apm": pd.concat([immune, stromal], axis=1),
        "tme_full": pd.concat([immune, stromal, apm], axis=1),
        "tme_plus_clinical": pd.concat([immune, stromal, apm, clinical], axis=1),
        "full": X,
    }
