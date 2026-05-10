"""
Групповые precision для модальностей (Фаза B): веса PE и калибровка.

RKK_PRECISION_GROUPS=1 — использовать weighted_squared_error в агрегаторах PE.
RKK_PRECISION_CALIB_WINDOW — окно шагов для acceptance-критерия калибровки (default 50).

Онлайн-дообучение precision оставлено заготовкой (precision_* скаляры в словаре).
"""
from __future__ import annotations

import os
from typing import Iterable

import numpy as np

def precision_groups_enabled() -> bool:
    return os.environ.get("RKK_PRECISION_GROUPS", "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def precision_calib_window() -> int:
    try:
        return max(5, int(os.environ.get("RKK_PRECISION_CALIB_WINDOW", "50")))
    except ValueError:
        return 50


def default_precision_vector() -> dict[str, float]:
    """Базовые precision (диагональ γ) по группам; можно переопределить из env."""
    out = {
        "vision": 1.0,
        "proprio": 4.0,
        "motor_intent": 2.0,
        "sandbox": 1.5,
        "default": 1.0,
    }
    for k in list(out.keys()):
        env_key = f"RKK_PRECISION_{k.upper()}"
        if k == "default":
            env_key = "RKK_PRECISION_DEFAULT"
        try:
            if os.environ.get(env_key) is not None:
                out[k] = float(os.environ.get(env_key, str(out[k])))
        except ValueError:
            pass
    return out


def modality_of_node(node_id: str) -> str:
    s = str(node_id)
    if any(s.startswith(p) for p in ("slot_", "visual_", "vl_", "vision_")):
        return "vision"
    if s.startswith("intent_") or s.startswith("phys_intent_"):
        return "motor_intent"
    if any(
        s.startswith(p)
        for p in (
            "cube",
            "ball_",
            "lever_",
            "target_dist",
            "floor_friction",
            "stack_",
            "stability_score",
        )
    ):
        return "sandbox"
    if any(
        s.startswith(p)
        for p in (
            "com_",
            "torso_",
            "foot_",
            "posture_",
            "spine_",
            "neck_",
            "vestibular_",
            "lhip",
            "rhip",
            "lknee",
            "rknee",
            "lankle",
            "rankle",
            "lshoulder",
            "rshoulder",
            "lelbow",
            "relbow",
            "gait_",
            "support_",
            "motor_drive",
        )
    ):
        return "proprio"
    return "default"


def weighted_squared_error_sum(
    nodes: Iterable[str],
    errors: dict[str, float],
    *,
    precisions: dict[str, float] | None = None,
) -> float:
    """
    Sum_gamma precision[group] * err² для узлов, присутствующих в errors.
    """
    prec = precisions or default_precision_vector()
    total = 0.0
    for nid in nodes:
        if nid not in errors:
            continue
        g = modality_of_node(nid)
        pg = float(prec.get(g, prec.get("default", 1.0)))
        e = float(errors[nid])
        total += pg * (e * e)
    return total


def precision_down_scale(group: str, factor: float = 0.5) -> None:
    """Заглушка для будущего состояния precision; сейчас no-op (stateless)."""
    del group, factor
