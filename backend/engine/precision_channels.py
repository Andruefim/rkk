"""
Групповые precision для модальностей (Фаза B): веса PE и калибровка.

Единая точка истины для флага и маршрутизации модальностей — ``precision_groups``;
этот модуль держит только формулы PE по узлам и обёртки env для скаляров.

RKK_PRECISION_GROUPS=1 — weighted_squared_error в агрегаторах PE.
RKK_PRECISION_CALIB_WINDOW — окно acceptance-калибровки (см. precision_groups).
"""
from __future__ import annotations

import os
from typing import Iterable

from engine.precision_groups import (
    get_precision_state,
    modality_group_for_var,
    precision_calib_window as _precision_calib_window_pg,
    precision_groups_enabled,
)

_PRECISION_SCALAR_FIELDS = frozenset(
    {"proprio", "vestibular", "motor_intent", "sandbox", "other"}
)


def precision_calib_window() -> int:
    """Backward-compat wrapper; delegates to ``precision_groups``."""
    return max(5, int(_precision_calib_window_pg()))


def modality_of_node(node_id: str) -> str:
    """Alias для канонического ``modality_group_for_var`` (efference, PE-агрегаторы)."""
    return modality_group_for_var(node_id)


def default_precision_vector() -> dict[str, float]:
    """Базовые precision (диагональ γ) по группам; ключи согласованы с ``modality_group_for_var``."""
    out = {
        "vision": 1.0,
        "proprio": 4.0,
        "motor_intent": 2.0,
        "sandbox": 1.5,
        "vestibular": 2.0,
        "other": 1.0,
    }
    for k in list(out.keys()):
        env_key = f"RKK_PRECISION_{k.upper()}"
        try:
            if os.environ.get(env_key) is not None:
                out[k] = float(os.environ.get(env_key, str(out[k])))
        except ValueError:
            pass
    # Legacy env RKK_PRECISION_DEFAULT → applies to fallback bucket "other"
    try:
        if os.environ.get("RKK_PRECISION_DEFAULT") is not None:
            out["other"] = float(os.environ.get("RKK_PRECISION_DEFAULT", str(out["other"])))
    except ValueError:
        pass
    return out


def weighted_squared_error_sum(
    nodes: Iterable[str],
    errors: dict[str, float],
    *,
    precisions: dict[str, float] | None = None,
) -> float:
    """Sum_gamma precision[group] * err² для узлов, присутствующих в errors."""
    prec = precisions or default_precision_vector()
    total = 0.0
    for nid in nodes:
        if nid not in errors:
            continue
        g = modality_group_for_var(str(nid))
        pg = float(prec.get(g, prec.get("other", 1.0)))
        e = float(errors[nid])
        total += pg * (e * e)
    return total


def precision_down_scale(group: str, factor: float = 0.5) -> None:
    """
    Снижение π по группе. ``vision`` → ``PrecisionGroupState.decay_vision``;
    остальные группы — мягкое масштабирование поля в глобальном состоянии.
    """
    st = get_precision_state()
    g = str(group or "").lower()
    f = float(factor)
    if g == "vision":
        st.decay_vision(f)
        return
    if g in _PRECISION_SCALAR_FIELDS:
        cur = float(getattr(st, g))
        setattr(st, g, float(max(0.05, min(4.0, cur * f))))
