"""
Этап F — лёгкий «символьный» слой: проверка предсказаний GNN на простые физические ограничения.

Не полноценный символьный движок: список предикатов над состоянием в физических единицах
(после _denorm среды humanoid). Нарушение → не доверять предсказанию, усилить исследование.
"""
from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

import numpy as np


def symbolic_verifier_enabled() -> bool:
    return os.environ.get("RKK_SYMBOLIC_VERIFY", "1").strip().lower() not in (
        "0",
        "false",
        "off",
        "no",
    )


def resolve_denorm_env(env: Any) -> Any | None:
    """Среда с методом _denorm(key, norm_val) → физическое значение."""
    if env is None:
        return None
    if callable(getattr(env, "_denorm", None)):
        return env
    b = getattr(env, "base_env", None)
    if b is not None and callable(getattr(b, "_denorm", None)):
        return b
    return None


def normalized_to_physical_dict(s_norm: dict[str, float], env: Any) -> dict[str, float]:
    """
    Ключи как в графе (в т.ч. phys_* в hybrid) → физические имена и значения для проверок.
    """
    den_env = resolve_denorm_env(env)
    out: dict[str, float] = {}
    if den_env is None:
        for k, v in s_norm.items():
            if str(k).startswith("slot_"):
                continue
            try:
                out[str(k)] = float(v)
            except (TypeError, ValueError):
                pass
        return out

    for k, v in s_norm.items():
        ks = str(k)
        if ks.startswith("slot_"):
            continue
        key = ks[5:] if ks.startswith("phys_") else ks
        try:
            fv = float(v)
        except (TypeError, ValueError):
            continue
        try:
            out[key] = float(den_env._denorm(key, fv))
        except Exception:
            out[key] = fv
    return out


def _has(s: dict[str, float], k: str) -> bool:
    if k not in s:
        return False
    try:
        fv = float(s[k])
    except (TypeError, ValueError):
        return False
    return fv == fv and np.isfinite(fv)


def _c_com_z_not_under_floor(s: dict[str, float]) -> bool:
    if not _has(s, "com_z"):
        return True
    return float(s["com_z"]) > -0.05


def _c_cube0_x_in_scene(s: dict[str, float]) -> bool:
    if not _has(s, "cube0_x"):
        return True
    return abs(float(s["cube0_x"])) < 3.5


def _c_cube0_y_in_scene(s: dict[str, float]) -> bool:
    if not _has(s, "cube0_y"):
        return True
    return abs(float(s["cube0_y"])) < 3.5


def _c_joint_lknee(s: dict[str, float]) -> bool:
    if not _has(s, "lknee"):
        return True
    v = float(s["lknee"])
    return v >= -np.pi - 0.25 and v <= np.pi + 0.25


def _c_joint_rknee(s: dict[str, float]) -> bool:
    if not _has(s, "rknee"):
        return True
    v = float(s["rknee"])
    return v >= -np.pi - 0.25 and v <= np.pi + 0.25


def _c_joint_lshoulder(s: dict[str, float]) -> bool:
    if not _has(s, "lshoulder"):
        return True
    v = float(s["lshoulder"])
    return v >= -np.pi - 0.25 and v <= np.pi + 0.25


def _c_joint_rshoulder(s: dict[str, float]) -> bool:
    if not _has(s, "rshoulder"):
        return True
    v = float(s["rshoulder"])
    return v >= -np.pi - 0.25 and v <= np.pi + 0.25


# Документ 2 style «символьный слой» — расширяемый список; отсутствующие ключи = OK.
PHYSICS_CONSTRAINTS: list[Callable[[dict[str, float]], bool]] = [
    _c_com_z_not_under_floor,
    _c_cube0_x_in_scene,
    _c_cube0_y_in_scene,
    _c_joint_lknee,
    _c_joint_rknee,
    _c_joint_lshoulder,
    _c_joint_rshoulder,
]


def verify_physical_state(s_phys: dict[str, float]) -> tuple[bool, list[str]]:
    failed: list[str] = []
    for i, fn in enumerate(PHYSICS_CONSTRAINTS):
        try:
            ok = bool(fn(s_phys))
        except Exception:
            ok = False
        if not ok:
            failed.append(getattr(fn, "__name__", f"c{i}"))
    return (len(failed) == 0, failed)


def verify_normalized_prediction(s_norm: dict[str, float], env: Any) -> tuple[bool, list[str]]:
    """Полный путь: нормализованное предсказание графа → физика → ограничения."""
    if not symbolic_verifier_enabled():
        return True, []
    s_phys = normalized_to_physical_dict(s_norm, env)
    return verify_physical_state(s_phys)


def downrank_factor_for_violation() -> float:
    try:
        return float(os.environ.get("RKK_SYMBOLIC_EIG_DOWNRANK", "0.12"))
    except ValueError:
        return 0.12


def exploration_blend_from_uncertainty() -> tuple[float, float]:
    """При плохом предсказании: expected_ig ← a*ig + b*uncertainty."""
    try:
        a = float(os.environ.get("RKK_SYMBOLIC_EXPL_A", "0.28"))
        b = float(os.environ.get("RKK_SYMBOLIC_EXPL_B", "0.62"))
    except ValueError:
        a, b = 0.28, 0.62
    return max(0.0, min(1.0, a)), max(0.0, min(1.0, b))
