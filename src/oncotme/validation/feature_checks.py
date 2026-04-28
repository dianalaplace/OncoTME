"""Проверки на уровне feature-frame (после ssGSEA / сборки TME-панели).

Ловим:
- NaN после скоринга (например, если сигнатура не пересеклась с экспрессией)
- константные признаки (ничего не дают модели, но часто индикатор бага)
- биологически невозможные корреляции (CD8 vs IFN_γ должны быть +)
- сильно коррелированные пары (|r|>0.98) — потенциальный data-leakage через
  дублирующийся ген/сигнатуру
- распределение скоров (z-score сигнатуры должен иметь mean≈0, std≈1)
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from .checks import (
    _fail,
    _pass,
    _warn,
    assert_expected_correlation_sign,
    assert_no_inf,
    assert_no_nan,
)
from .result import CheckResult


# Биологически ожидаемые знаки корреляций между сигнатурами.
# Мягкое правило: знак обязан совпадать, |r| — warn-порог.
EXPECTED_CORRELATIONS = [
    # (sig1, sig2, sign, min_abs_r)
    ("CD8_T_cell", "IFN_gamma", +1, 0.25),
    ("CD8_T_cell", "checkpoint_score", +1, 0.20),
    ("HLA_I_score", "APM_score", +1, 0.30),
    ("HLA_I_score", "IFN_gamma", +1, 0.20),
    ("TLS_signature", "B_cell", +1, 0.25),
    ("CAF_proxy", "hypoxia", +1, 0.15),
    # TGFβ-high exclusion часто анти-коррелирует с CD8
    ("TGFb", "CD8_T_cell", -1, 0.10),
]


def check_feature_frame(features: pd.DataFrame) -> List[CheckResult]:
    """Общие проверки feature-frame (выход tme_panel.build_tme_features)."""
    results: List[CheckResult] = []

    # 1. не пустой и уникальный индекс
    if features.empty:
        return [_fail("features.nonempty", "Empty feature frame.")]
    results.append(_pass("features.nonempty", f"shape={features.shape}", shape=features.shape))

    # 2. нет NaN-колонок целиком (сигнатура, которая не посчиталась)
    all_nan_cols = features.columns[features.isna().all()].tolist()
    if all_nan_cols:
        results.append(
            _fail(
                "features.no_all_nan_columns",
                f"{len(all_nan_cols)} fully-NaN columns: {all_nan_cols[:10]}",
                columns=all_nan_cols,
            )
        )
    else:
        results.append(_pass("features.no_all_nan_columns", "No fully-NaN columns."))

    # 3. NaN в целом — сигнатура не должна; индивидуальные гены могут
    sig_cols = [c for c in features.columns if c.startswith("sig__")]
    if sig_cols:
        sig_nan = features[sig_cols].isna().sum().sum()
        sig_total = features[sig_cols].size
        if sig_nan > 0:
            results.append(
                _fail(
                    "features.signatures_no_nan",
                    f"{sig_nan}/{sig_total} NaN in signature scores.",
                    n_nan=int(sig_nan),
                )
            )
        else:
            results.append(_pass("features.signatures_no_nan", "Signatures clean."))

    # 4. Inf
    numeric = features.select_dtypes(include=[np.number])
    if not numeric.empty:
        results.append(assert_no_inf(numeric, "features.no_inf"))

    # 5. константные колонки (std == 0)
    stds = numeric.std(skipna=True)
    const_cols = stds[stds == 0].index.tolist()
    if const_cols:
        results.append(
            _warn(
                "features.no_constant_columns",
                f"{len(const_cols)} constant columns: {const_cols[:5]}",
                columns=const_cols,
            )
        )
    else:
        results.append(_pass("features.no_constant_columns", "No constant columns."))

    # 6. quasi-duplicate columns (|r|>0.98, не считая диагональ)
    if numeric.shape[1] >= 2 and numeric.shape[0] >= 10:
        # на больших матрицах ограничиваем до 300 колонок для производительности
        sub = numeric if numeric.shape[1] <= 300 else numeric.sample(300, axis=1, random_state=0)
        corr = sub.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
        near_dup = upper.stack().loc[lambda s: s > 0.98]
        if not near_dup.empty:
            pairs = near_dup.head(5).index.tolist()
            results.append(
                _warn(
                    "features.no_near_duplicate_columns",
                    f"{len(near_dup)} pairs with |r|>0.98. Examples: {pairs}",
                    n_pairs=int(len(near_dup)),
                )
            )
        else:
            results.append(_pass("features.no_near_duplicate_columns", "No near-duplicate pairs."))

    return results


def check_signature_scores(features: pd.DataFrame) -> List[CheckResult]:
    """Биологическая sanity-check: ожидаемые направления корреляций.

    Используется z-score fallback или ssGSEA — в обоих случаях ранжирование
    направлений должно сохраняться, если данные настоящие.
    """
    results: List[CheckResult] = []

    # вытаскиваем чистые имена сигнатур (без префикса sig__)
    sig_cols: Dict[str, str] = {
        c.replace("sig__", ""): c for c in features.columns if c.startswith("sig__")
    }

    if len(sig_cols) < 2:
        return [
            _warn(
                "features.bio_sanity",
                f"Only {len(sig_cols)} signatures present — biological sanity skipped.",
            )
        ]

    for s1, s2, sign, min_r in EXPECTED_CORRELATIONS:
        if s1 not in sig_cols or s2 not in sig_cols:
            continue
        name = f"features.corr_sign.{s1}_vs_{s2}"
        results.append(
            assert_expected_correlation_sign(
                features[sig_cols[s1]],
                features[sig_cols[s2]],
                expected_sign=sign,
                name=name,
                min_abs_corr=min_r,
            )
        )

    # Масштаб сигнатур: различаем z-score output (mean≈0, std≈1) от raw ssGSEA ES (большие числа)
    # FAIL только когда сигнатуры в радикально разных масштабах между собой —
    # это признак смешения z-score и raw ES или кривого concat.
    if sig_cols:
        sig_matrix = features[list(sig_cols.values())]
        col_std = sig_matrix.std()
        col_mean_abs = sig_matrix.mean().abs()
        if (col_std <= 0).any():
            results.append(
                _fail(
                    "features.signatures_scale_consistent",
                    "Some signature columns are constant.",
                )
            )
        else:
            # Оцениваем "режим" по медианным std:
            median_std = col_std.median()
            std_range = col_std.max() / col_std.min()
            if std_range > 1000:
                results.append(
                    _fail(
                        "features.signatures_scale_consistent",
                        f"Signature scales span {std_range:.0f}× — likely mixing z-score and raw ES.",
                        std_ratio=float(std_range),
                    )
                )
            elif median_std < 5:
                # z-score-like: ожидаем центрированность
                high_mean = col_mean_abs[col_mean_abs > 0.5]
                if len(high_mean) > 0:
                    results.append(
                        _warn(
                            "features.signatures_scale_consistent",
                            f"z-score-like output but {len(high_mean)} signatures with "
                            f"|mean|>0.5. Check fallback_zscore.",
                            max_abs_mean=float(col_mean_abs.max()),
                        )
                    )
                else:
                    results.append(
                        _pass(
                            "features.signatures_scale_consistent",
                            f"z-score output centered (median_std={median_std:.2f}).",
                        )
                    )
            else:
                # raw ssGSEA ES: большие числа — ок; проверяем только consistent масштаб
                results.append(
                    _pass(
                        "features.signatures_scale_consistent",
                        f"Raw ssGSEA ES output (median_std={median_std:.1f}, range {std_range:.1f}×).",
                        mode="raw_ssgsea",
                    )
                )

    return results
