"""
Efference copy (Фаза F): ожидаемое наблюдение до шага среды vs факт.

Пороги корреляции по умолчанию (stationary vs vision) задаются env — см. план acceptance.
"""
from __future__ import annotations

import os
from typing import Any

import numpy as np


def efference_corr_proprio_default() -> float:
    try:
        return float(os.environ.get("RKK_EFFERENCE_CORR_PROPRIO", "0.9"))
    except ValueError:
        return 0.9


def efference_corr_vision_default() -> float:
    try:
        return float(os.environ.get("RKK_EFFERENCE_CORR_VISION", "0.7"))
    except ValueError:
        return 0.7


def expected_obs_after_do(
    graph: Any,
    base: dict[str, float],
    variable: str,
    value: float,
) -> dict[str, float]:
    """Один шаг WM после мысленного do — предсказание следующего снимка узлов."""
    if graph is None:
        return dict(base)
    try:
        return graph.propagate_from(dict(base), variable, float(value))
    except Exception:
        return dict(base)


def _corr_component(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2 or b.size < 2:
        return 1.0
    sa = float(np.std(a))
    sb = float(np.std(b))
    if sa < 1e-9 or sb < 1e-9:
        return 1.0 if float(np.abs(a.mean() - b.mean())) < 1e-6 else 0.0
    return float(np.corrcoef(a, b)[0, 1])


def split_obs_vectors(
    expected: dict[str, float],
    actual: dict[str, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Разделить плоские векторы на vision-like и stationary (proprio) по модальности."""
    from engine.precision_channels import modality_of_node

    v_e, v_a, p_e, p_a = [], [], [], []
    keys = sorted(set(expected.keys()) & set(actual.keys()))
    for k in keys:
        try:
            xe = float(expected[k])
            xa = float(actual[k])
        except (TypeError, ValueError):
            continue
        if modality_of_node(k) == "vision":
            v_e.append(xe)
            v_a.append(xa)
        elif modality_of_node(k) in ("proprio", "motor_intent"):
            p_e.append(xe)
            p_a.append(xa)
    return (
        np.asarray(v_e, dtype=float),
        np.asarray(v_a, dtype=float),
        np.asarray(p_e, dtype=float),
        np.asarray(p_a, dtype=float),
    )


def efference_correlation_report(
    expected: dict[str, float],
    actual: dict[str, float],
) -> dict[str, float]:
    ve, va, pe, pa = split_obs_vectors(expected, actual)
    out = {
        "corr_proprio": _corr_component(pe, pa),
        "corr_vision": _corr_component(ve, va),
        "thresh_proprio": efference_corr_proprio_default(),
        "thresh_vision": efference_corr_vision_default(),
    }
    out["proprio_ok"] = float(out["corr_proprio"]) >= out["thresh_proprio"]
    out["vision_ok"] = (
        ve.size < 2 or float(out["corr_vision"]) >= out["thresh_vision"]
    )
    out["ok"] = bool(out["proprio_ok"] and out["vision_ok"])
    return out
