"""
Исполнитель моторики из уже выученной матрицы GNN: intent_* × W → цели суставов.

Не задаёт поведение — только применяет то, что граф уже открыл.

Вкл.: RKK_CAUSAL_MOTOR_EXECUTOR=1

Структурный хардкод допустим: список имён суставов из URDF (constants), не сценарии.
"""
from __future__ import annotations

import os

import numpy as np

from engine.features.humanoid.constants import ARM_VARS, HEAD_VARS, LEG_VARS, SPINE_VARS


def causal_motor_executor_enabled() -> bool:
    return os.environ.get("RKK_CAUSAL_MOTOR_EXECUTOR", "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


# Кинематика тела из констант URDF — не сценарии «ходьбы»
STRUCTURAL_JOINT_KEYS: frozenset[str] = frozenset(
    list(LEG_VARS) + list(ARM_VARS) + list(SPINE_VARS) + list(HEAD_VARS)
)


def _env_float(key: str, default: float) -> float:
    try:
        return float(os.environ.get(key, str(default)))
    except ValueError:
        return default


class CausalMotorExecutor:
    """Читает W_masked из causal core и строит цели суставов из intent_* и весов."""

    def execute(self, graph, intents: dict[str, float]) -> dict[str, float]:
        core = getattr(graph, "_core", None)
        if core is None:
            return {}
        min_alpha = _env_float("RKK_CAUSAL_EXECUTOR_MIN_ALPHA", 0.25)
        try:
            am = float(getattr(graph, "alpha_mean", 1.0))
        except Exception:
            am = 0.0
        if am < min_alpha:
            return {}
        try:
            W = core.W_masked().detach().float().cpu().numpy()
        except Exception:
            return {}

        n = int(W.shape[0])
        if W.ndim != 2 or W.shape[1] != n:
            return {}

        node_ids = list(getattr(graph, "_node_ids", []))
        if len(node_ids) != n:
            return {}

        thr = _env_float("RKK_CAUSAL_EXECUTOR_THRESH", 0.08)
        gain = _env_float("RKK_CAUSAL_EXECUTOR_GAIN", 2.5)

        nid2i = {name: i for i, name in enumerate(node_ids)}
        result: dict[str, float] = {}

        for intent_name, intent_val in intents.items():
            sin = str(intent_name)
            if not sin.startswith("intent_"):
                continue
            ii = nid2i.get(sin)
            if ii is None:
                continue
            try:
                delta_scale = float(intent_val) - 0.5
            except (TypeError, ValueError):
                continue

            for jj in range(n):
                jname = node_ids[jj]
                if jname not in STRUCTURAL_JOINT_KEYS:
                    continue
                w = float(W[ii, jj])
                if abs(w) < thr:
                    continue
                d = gain * w * delta_scale
                prev = float(result.get(jname, 0.5))
                result[jname] = float(np.clip(prev + d, 0.05, 0.95))

        return result
