"""
Honest locomotion mastery gates and rolling eval (Sprint 5.1 / 8.0).
"""
from __future__ import annotations

import os
from collections import deque
from dataclasses import dataclass, field
from typing import Any


def _ef(key: str, default: float) -> float:
    try:
        return float(os.environ.get(key, str(default)))
    except ValueError:
        return default


def _ei(key: str, default: int) -> int:
    try:
        return max(1, int(os.environ.get(key, str(default))))
    except ValueError:
        return default


def is_locomotion_mastered(metrics: dict[str, Any]) -> bool:
    """
    All criteria must pass — no step_count-only mastery.
    """
    ticks = max(1, int(metrics.get("ticks_in_step3", metrics.get("window_ticks", 1000))))
    disp = float(metrics.get("com_x_displacement", 0.0))
    disp_per_1k = abs(disp) * 1000.0 / float(ticks)

    coupling = float(
        metrics.get(
            "coupling_motor",
            metrics.get("intent_gait_coupling", metrics.get("coupling_final", 0.88)),
        )
    )
    return (
        float(metrics.get("com_x_vel_ema", 0.0)) > _ef("RKK_LOC_MASTER_COM_X_VEL", 0.008)
        and float(metrics.get("pe_fwd_ema", 0.0)) > _ef("RKK_LOC_MASTER_PE_FWD", -0.4)
        and disp_per_1k > _ef("RKK_LOC_MASTER_DISP_PER_1K", 0.3)
        and coupling <= _ef("RKK_LOC_MASTER_COUPLING_MAX", 0.80)
        and float(metrics.get("fall_rate", 1.0)) <= _ef("RKK_LOC_MASTER_FALL_RATE", 0.0)
    )


@dataclass
class EvalResult:
    passed: bool
    scores: dict[str, float] = field(default_factory=dict)
    gate: str = "locomotion_mastery"
    failures: list[str] = field(default_factory=list)


class LocomotionEval:
    """Rolling window evaluator (~60s at ~2.5 Hz ≈ 150 ticks)."""

    PASS_CRITERIA = {
        "com_x_vel_ema_min": 0.008,
        "pe_fwd_ema_min": -0.4,
        "support_asymmetry_min": 0.10,
        "fall_rate_max": 0.0,
        "displacement_per_60s_min": 0.3,
    }

    def __init__(self, window: int | None = None) -> None:
        self._window = window or _ei("RKK_LOCOMOTION_EVAL_WINDOW", 150)
        self._buf: deque[dict[str, float]] = deque(maxlen=self._window)
        self._tick0_com_x: float | None = None
        self._last_result: EvalResult | None = None

    def record_tick(self, metrics: dict[str, Any]) -> None:
        row = {
            "com_x_vel_ema": float(metrics.get("com_x_vel_ema", 0.0)),
            "pe_fwd_ema": float(metrics.get("pe_fwd_ema", 0.0)),
            "fall_rate": float(metrics.get("fall_rate", 0.0)),
            "support_asymmetry": float(metrics.get("support_asymmetry", 0.0)),
            "com_x": float(metrics.get("com_x", metrics.get("com_x_displacement", 0.0))),
        }
        if self._tick0_com_x is None and "com_x_raw" in metrics:
            self._tick0_com_x = float(metrics["com_x_raw"])
        self._buf.append(row)

    def evaluate(self, metrics_window: list[dict] | None = None) -> EvalResult:
        rows = list(metrics_window) if metrics_window is not None else list(self._buf)
        if len(rows) < max(8, self._window // 4):
            self._last_result = EvalResult(
                passed=False,
                scores={},
                failures=["insufficient_window"],
            )
            return self._last_result

        vel = float(sum(r.get("com_x_vel_ema", 0.0) for r in rows) / len(rows))
        pe = float(sum(r.get("pe_fwd_ema", 0.0) for r in rows) / len(rows))
        fall = float(max(r.get("fall_rate", 0.0) for r in rows))
        asym = float(max(r.get("support_asymmetry", 0.0) for r in rows))
        disp = 0.0
        if len(rows) >= 2:
            disp = abs(float(rows[-1].get("com_x", 0.0)) - float(rows[0].get("com_x", 0.0)))
        # Scale displacement to ~60s equivalent
        disp_60s = disp * (150.0 / max(1, len(rows)))

        scores = {
            "com_x_vel_ema": vel,
            "pe_fwd_ema": pe,
            "support_asymmetry": asym,
            "fall_rate": fall,
            "displacement_per_60s": disp_60s,
        }
        failures: list[str] = []
        if vel < self.PASS_CRITERIA["com_x_vel_ema_min"]:
            failures.append("com_x_vel_ema")
        if pe < self.PASS_CRITERIA["pe_fwd_ema_min"]:
            failures.append("pe_fwd_ema")
        if asym < self.PASS_CRITERIA["support_asymmetry_min"]:
            failures.append("support_asymmetry")
        if fall > self.PASS_CRITERIA["fall_rate_max"]:
            failures.append("fall_rate")
        if disp_60s < self.PASS_CRITERIA["displacement_per_60s_min"]:
            failures.append("displacement_per_60s")

        self._last_result = EvalResult(
            passed=len(failures) == 0,
            scores=scores,
            failures=failures,
        )
        return self._last_result

    def snapshot(self) -> dict[str, Any]:
        r = self._last_result or EvalResult(passed=False)
        return {
            "window_ticks": len(self._buf),
            "window_60s_pass": bool(r.passed),
            "displacement_per_60s": round(float(r.scores.get("displacement_per_60s", 0.0)), 4),
            "scores": {k: round(float(v), 4) for k, v in r.scores.items()},
            "failures": list(r.failures),
        }
