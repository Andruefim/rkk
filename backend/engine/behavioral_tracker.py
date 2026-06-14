"""Rolling behavioral metrics for curriculum gates and honest HUD."""
from __future__ import annotations

import os
from collections import deque
from typing import Any


def _env_int(key: str, default: int) -> int:
    try:
        return max(1, int(os.environ.get(key, str(default))))
    except ValueError:
        return default


def _env_float(key: str, default: float) -> float:
    try:
        return float(os.environ.get(key, str(default)))
    except ValueError:
        return default


class BehavioralTracker:
    def __init__(self, window: int | None = None):
        self._window = window or _env_int("RKK_BEHAVIORAL_WINDOW", 200)
        self._posture: deque[float] = deque(maxlen=self._window)
        self._com_x: deque[float] = deque(maxlen=self._window)
        self._com_x_vel: deque[float] = deque(maxlen=self._window)
        self._fallen: deque[int] = deque(maxlen=self._window)
        self._loco_reward: deque[float] = deque(maxlen=self._window)
        self._recovery_learned: deque[int] = deque(maxlen=64)
        self._step3_substate: str = "3a_learning"
        self._step3_enter_tick: int = -1
        self._prev_com_x: float | None = None
        self._ticks_in_step3: int = 0

    def note_step3_entry(self, tick: int) -> None:
        if self._step3_enter_tick < 0:
            self._step3_enter_tick = int(tick)

    def record_tick(
        self,
        *,
        tick: int,
        obs: dict[str, float],
        fallen: bool,
        locomotion_reward: float = 0.0,
        recovery_learned_success: bool | None = None,
        in_step3: bool = False,
    ) -> None:
        ps = float(obs.get("posture_stability", obs.get("phys_posture_stability", 0.5)))
        forward_raw = obs.get("com_forward_raw_m")
        if forward_raw is not None:
            cx = float(forward_raw)
        else:
            cx_raw = obs.get("com_x_raw_m")
            if cx_raw is not None:
                cx = float(cx_raw)
            else:
                cx = float(obs.get("com_y", obs.get("phys_com_y", obs.get("com_x", obs.get("phys_com_x", 0.5)))))
        vel = 0.0
        if self._prev_com_x is not None:
            vel = cx - self._prev_com_x
        self._prev_com_x = cx

        self._posture.append(ps)
        self._com_x.append(cx)
        self._com_x_vel.append(vel)
        self._fallen.append(1 if fallen else 0)
        self._loco_reward.append(float(locomotion_reward))

        if recovery_learned_success is not None:
            self._recovery_learned.append(1 if recovery_learned_success else 0)

        if in_step3:
            self._ticks_in_step3 += 1
            if self._step3_enter_tick < 0:
                self._step3_enter_tick = int(tick)
            self._update_step3_substate()

    def _update_step3_substate(self) -> None:
        if self._step3_substate == "3b_locomotion_mastered":
            return
        min_ticks = _env_int("RKK_STEP3_MASTER_MIN_TICKS", 800)
        if self._ticks_in_step3 < min_ticks:
            return
        snap = self.snapshot()
        snap["ticks_in_step3"] = self._ticks_in_step3
        try:
            from engine.locomotion_mastery import is_locomotion_mastered

            if is_locomotion_mastered(snap):
                self._step3_substate = "3b_locomotion_mastered"
        except ImportError:
            if (
                snap["com_x_vel_ema"] >= _env_float("RKK_STEP3_COM_X_VEL_MIN", 0.002)
                and snap["fall_rate"] < _env_float("RKK_STEP3_FALL_RATE_MAX", 0.05)
                and snap["upright_rate"] > _env_float("RKK_STEP3_UPRIGHT_MIN", 0.90)
            ):
                self._step3_substate = "3b_locomotion_mastered"

    def snapshot(self) -> dict[str, Any]:
        n = max(1, len(self._posture))
        fall_rate = float(sum(self._fallen)) / n if self._fallen else 0.0
        upright = sum(1 for p in self._posture if p > 0.75)
        upright_rate = upright / max(1, len(self._posture))

        com_x_vel_ema = 0.0
        if len(self._com_x) >= 8:
            w = min(40, len(self._com_x) - 1)
            com_x_vel_ema = (float(self._com_x[-1]) - float(self._com_x[-1 - w])) / w
        elif self._com_x_vel:
            alpha = 0.08
            ema = float(self._com_x_vel[0])
            for v in list(self._com_x_vel)[1:]:
                ema = (1 - alpha) * ema + alpha * float(v)
            com_x_vel_ema = ema

        loco_ema = float(sum(self._loco_reward) / max(1, len(self._loco_reward)))
        displacement = 0.0
        if len(self._com_x) >= 2:
            displacement = float(self._com_x[-1]) - float(self._com_x[0])

        rec_n = len(self._recovery_learned)
        rec_rate = float(sum(self._recovery_learned)) / rec_n if rec_n else 0.0

        # Composite behavioral score
        walk = min(1.0, max(0.0, abs(com_x_vel_ema) / 0.0008))
        stable = upright_rate
        safe = 1.0 - min(1.0, fall_rate * 5.0)
        score = 0.45 * walk + 0.35 * stable + 0.20 * safe

        return {
            "behavioral_score": round(score, 4),
            "com_x_vel_ema": round(com_x_vel_ema, 5),
            "com_x_displacement": round(displacement, 5),
            "fall_rate": round(fall_rate, 4),
            "upright_rate": round(upright_rate, 4),
            "locomotion_reward_ema": round(loco_ema, 5),
            "recovery_success_rate": round(rec_rate, 4),
            "step_3_substate": self._step3_substate,
            "ticks_in_step3": self._ticks_in_step3,
        }
