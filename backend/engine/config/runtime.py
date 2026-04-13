"""
Часто читаемые параметры окружения (единая точка, без pydantic-settings).
Обновляется при создании; для динамики по-прежнему os.environ в горячих путях.
"""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class RKKRuntimeConfig:
    device: str
    cpg_loop_hz: float
    agent_loop_hz: float
    l3_loop_hz: float

    @classmethod
    def from_env(cls) -> "RKKRuntimeConfig":
        def _f(key: str, default: str) -> float:
            try:
                return float(os.environ.get(key, default))
            except ValueError:
                return float(default)

        return cls(
            device=(os.environ.get("RKK_DEVICE") or "cuda").strip(),
            cpg_loop_hz=max(0.0, min(_f("RKK_CPG_LOOP_HZ", "0"), 240.0)),
            agent_loop_hz=max(0.0, min(_f("RKK_AGENT_LOOP_HZ", "0"), 60.0)),
            l3_loop_hz=max(0.0, min(_f("RKK_L3_LOOP_HZ", "0"), 30.0)),
        )
