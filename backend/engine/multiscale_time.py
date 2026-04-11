"""
multiscale_time.py — Level 3-I: Multi-scale Time.

Проблема: все системы сейчас работают на одном tick-рейте без явного
разделения по временным масштабам. Биомеханика (CPG, 50ms) смешана
с планированием (GNN, 500ms) и рефлексией (LLM, 5s+).

RSSM и GNN предсказывают следующий тик — но «следующий тик» для CPG
это 50ms, а для LLM это 5 секунд. Без разделения модели путаются.

Решение: явная 4-уровневая временная иерархия:
  τ0 REFLEX   50ms   — CPG + ProprioStream (joint-level)
  τ1 MOTOR   500ms   — MotorCortex + SkillLibrary (movement primitives)
  τ2 COGNIT    5s    — CausalGNN + RSSM + EIG (causal reasoning)
  τ3 REFLECT  60s+   — LLM L2/L3 + Curriculum + Constitutional (language)

Каждый уровень:
  - Имеет свой буфер переходов (собственный шаг обучения)
  - Получает агрегированные сигналы снизу (абстракты)
  - Отправляет намерения вниз (targets)
  - Работает только на «своих» тиках

TemporalHierarchy:
  - .tick(tick, obs) → маршрутизирует вызовы по уровням
  - .should_run(level, tick) → bool
  - .get_level_context(level) → dict агрегатов для этого уровня
  - .set_level_intent(level, intents) → передаёт намерения вниз

TimescaleBuffer:
  - Хранит rolling history на нужном timescale
  - Агрегирует: mean, ema, delta (скорость изменения)

RKK_TIMESCALE_T0=3      — рефлекс каждые N тиков (≈50ms)
RKK_TIMESCALE_T1=30     — моторика каждые N тиков (≈500ms)
RKK_TIMESCALE_T2=300    — когниция каждые N тиков (≈5s)
RKK_TIMESCALE_T3=3600   — рефлексия каждые N тиков (≈60s)
RKK_TIMESCALE_ENABLED=1
"""
from __future__ import annotations

import os
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np


def timescale_enabled() -> bool:
    return os.environ.get("RKK_TIMESCALE_ENABLED", "1").strip().lower() not in (
        "0", "false", "no", "off"
    )


def _env_int(key: str, default: int) -> int:
    try:
        return max(1, int(os.environ.get(key, str(default))))
    except ValueError:
        return default


# ── Temporal levels ───────────────────────────────────────────────────────────
LEVEL_REFLEX   = 0   # τ0: CPG + ProprioStream
LEVEL_MOTOR    = 1   # τ1: MotorCortex + SkillLibrary
LEVEL_COGNIT   = 2   # τ2: CausalGNN + RSSM + EIG
LEVEL_REFLECT  = 3   # τ3: LLM + Curriculum + Constitutional

LEVEL_NAMES = {
    LEVEL_REFLEX:  "reflex",
    LEVEL_MOTOR:   "motor",
    LEVEL_COGNIT:  "cognit",
    LEVEL_REFLECT: "reflect",
}

# Key variables owned by each level
LEVEL_VARS: dict[int, list[str]] = {
    LEVEL_REFLEX: [
        "proprio_balance", "proprio_gait_phase",
        "proprio_left_leg_load", "proprio_right_leg_load",
        "proprio_arm_counterbalance", "proprio_anomaly",
        "foot_contact_l", "foot_contact_r",
        "gait_phase_l", "gait_phase_r",
        "com_z", "torso_pitch",
    ],
    LEVEL_MOTOR: [
        "posture_stability", "support_bias", "com_x",
        "cpg_weight", "mc_walk_drive", "mc_balance_signal",
        "intent_stride", "intent_torso_forward",
        "intent_support_left", "intent_support_right",
    ],
    LEVEL_COGNIT: [
        "intent_gait_coupling", "intent_arm_counterbalance",
        "intent_stop_recover", "posture_stability_ema",
        "own_walk_confidence", "own_stand_confidence",
        "proprio_empowerment", "causal_eig",
    ],
    LEVEL_REFLECT: [
        "curriculum_stage_idx", "llm_posture_score",
        "llm_gait_quality", "constitution_multiplier",
        "fall_rate_ema", "empowerment_ema",
    ],
}


# ── Timescale buffer ──────────────────────────────────────────────────────────
class TimescaleBuffer:
    """
    Rolling buffer для одного временного уровня.
    Накапливает obs samples на своём tick-рейте,
    выдаёт агрегаты (mean, ema, delta) для уровня выше.
    """

    def __init__(self, level: int, maxlen: int = 64):
        self.level = level
        self.name = LEVEL_NAMES[level]
        self._vars = LEVEL_VARS[level]
        self._buf: deque[dict[str, float]] = deque(maxlen=maxlen)
        self._ema: dict[str, float] = {}
        self._ema_alpha = 0.15 if level < 2 else 0.08
        self._last_obs: dict[str, float] = {}

    def push(self, obs: dict[str, float]) -> None:
        """Push one observation. Extract only this level's vars."""
        snap = {}
        for v in self._vars:
            val = obs.get(v, obs.get(f"phys_{v}", None))
            if val is not None:
                snap[v] = float(val)
        if snap:
            self._buf.append(snap)
            self._last_obs = snap
            # Update EMA
            for k, v in snap.items():
                if k not in self._ema:
                    self._ema[k] = v
                self._ema[k] = (1 - self._ema_alpha) * self._ema[k] + self._ema_alpha * v

    def get_aggregates(self) -> dict[str, float]:
        """
        Returns dict with:
          {var}: current value
          {var}_ema: exponential moving average
          {var}_delta: rate of change (current - oldest in buffer)
        """
        if not self._buf:
            return {}
        result = {}

        # Current value
        for k, v in self._last_obs.items():
            result[k] = v

        # EMA
        for k, v in self._ema.items():
            result[f"{k}_ema"] = v

        # Delta over buffer
        if len(self._buf) >= 2:
            oldest = self._buf[0]
            newest = self._buf[-1]
            for k in newest:
                if k in oldest:
                    result[f"{k}_delta"] = newest[k] - oldest[k]

        # Level-specific aggregates
        if self.level == LEVEL_REFLEX:
            result.update(self._reflex_aggregates())
        elif self.level == LEVEL_MOTOR:
            result.update(self._motor_aggregates())

        return result

    def _reflex_aggregates(self) -> dict[str, float]:
        """Fast biomechanical aggregates for motor level."""
        if not self._buf:
            return {}
        recent = list(self._buf)[-10:]

        # Gait periodicity (variance of gait phase)
        phases_l = [s.get("gait_phase_l", 0.5) for s in recent]
        phases_r = [s.get("gait_phase_r", 0.5) for s in recent]
        gait_variance = float(np.var(phases_l) + np.var(phases_r))

        # Fall risk score
        com_z_vals = [s.get("com_z", 0.7) for s in recent]
        posture_vals = [s.get("proprio_balance", 0.5) for s in recent]
        fall_risk = float(np.clip(
            (1.0 - np.mean(com_z_vals)) * 0.5 + (1.0 - np.mean(posture_vals)) * 0.5,
            0.0, 1.0,
        ))

        return {
            "reflex_gait_variance": float(np.clip(gait_variance * 10.0, 0.0, 1.0)),
            "reflex_fall_risk": fall_risk,
            "reflex_contact_quality": float(np.mean([
                s.get("foot_contact_l", 0.5) + s.get("foot_contact_r", 0.5)
                for s in recent
            ])) / 2.0,
        }

    def _motor_aggregates(self) -> dict[str, float]:
        """Movement quality aggregates for cognitive level."""
        if not self._buf:
            return {}
        recent = list(self._buf)[-20:]

        postures = [s.get("posture_stability", 0.5) for s in recent]
        strides = [s.get("intent_stride", 0.5) for s in recent]

        return {
            "motor_posture_mean": float(np.mean(postures)),
            "motor_posture_std": float(np.std(postures)),
            "motor_stride_mean": float(np.mean(strides)),
            "motor_walk_quality": float(np.clip(
                np.mean(postures) - np.std(postures) * 0.5,
                0.0, 1.0,
            )),
        }

    def snapshot(self) -> dict[str, Any]:
        return {
            "level": self.level,
            "name": self.name,
            "buf_len": len(self._buf),
            "last_obs_keys": list(self._last_obs.keys()),
            "ema_keys": list(self._ema.keys()),
        }


# ── Temporal dispatcher ───────────────────────────────────────────────────────
class TemporalDispatcher:
    """
    Управляет расписанием вызовов по уровням.
    Проверяет: должен ли данный level выполняться на этом tick.
    """

    def __init__(self):
        self._periods = {
            LEVEL_REFLEX:  _env_int("RKK_TIMESCALE_T0", 3),
            LEVEL_MOTOR:   _env_int("RKK_TIMESCALE_T1", 30),
            LEVEL_COGNIT:  _env_int("RKK_TIMESCALE_T2", 300),
            LEVEL_REFLECT: _env_int("RKK_TIMESCALE_T3", 3600),
        }
        self._last_run: dict[int, int] = {l: -9999 for l in self._periods}
        self._run_counts: dict[int, int] = {l: 0 for l in self._periods}

    def should_run(self, level: int, tick: int) -> bool:
        period = self._periods.get(level, 1)
        return (tick - self._last_run.get(level, -9999)) >= period

    def mark_ran(self, level: int, tick: int) -> None:
        self._last_run[level] = tick
        self._run_counts[level] = self._run_counts.get(level, 0) + 1

    def ticks_since(self, level: int, tick: int) -> int:
        return tick - self._last_run.get(level, -9999)

    def snapshot(self) -> dict[str, Any]:
        return {
            "periods": self._periods,
            "run_counts": self._run_counts,
            "last_run": self._last_run,
        }


# ── Intent propagator ─────────────────────────────────────────────────────────
class IntentPropagator:
    """
    Передаёт намерения сверху вниз по иерархии.

    LLM (τ3) → setting intent_stride=0.60 для Motor level
    Motor level (τ1) → adjusting CPG weight для Reflex level

    Намерения: soft targets, применяются с decay.
    Нижний уровень всегда может overrule через SurvivalVeto.
    """

    def __init__(self):
        self._intents: dict[int, dict[str, float]] = {l: {} for l in range(4)}
        self._decay: float = 0.95  # намерения затухают без обновления

    def set_intent(self, from_level: int, var: str, value: float, strength: float = 1.0) -> None:
        """Level from_level устанавливает намерение для уровня ниже."""
        target_level = max(0, from_level - 1)
        self._intents[target_level][var] = float(np.clip(value, 0.0, 1.0))

    def get_intents(self, level: int) -> dict[str, float]:
        """Получить активные намерения для данного уровня."""
        return dict(self._intents.get(level, {}))

    def decay_tick(self) -> None:
        """Применить decay ко всем намерениям (вызывать раз в N тиков)."""
        for level in self._intents:
            for var in list(self._intents[level].keys()):
                self._intents[level][var] *= self._decay
                if self._intents[level][var] < 0.01:
                    del self._intents[level][var]

    def snapshot(self) -> dict[str, Any]:
        return {
            f"level_{l}_{LEVEL_NAMES.get(l, str(l))}": dict(intents)
            for l, intents in self._intents.items()
            if intents
        }


# ── Multi-scale Time Controller ───────────────────────────────────────────────
class MultiscaleTimeController:
    """
    Главный контроллер временной иерархии.

    Интегрируется в simulation.py:
      self._timescale = MultiscaleTimeController()

    В начале каждого тика:
      ctx = self._timescale.tick(tick, obs)
      # ctx содержит агрегаты для каждого уровня

    Проверить нужно ли запускать подсистему:
      if self._timescale.should_run(LEVEL_MOTOR, tick):
          self._apply_motor_cortex(...)
          self._timescale.mark_ran(LEVEL_MOTOR, tick)

    При обновлении намерений от LLM:
      self._timescale.set_intent(LEVEL_REFLECT, "intent_stride", 0.62)
    """

    def __init__(self):
        self.dispatcher = TemporalDispatcher()
        self.propagator = IntentPropagator()
        self.buffers = {
            LEVEL_REFLEX:  TimescaleBuffer(LEVEL_REFLEX, maxlen=64),
            LEVEL_MOTOR:   TimescaleBuffer(LEVEL_MOTOR, maxlen=32),
            LEVEL_COGNIT:  TimescaleBuffer(LEVEL_COGNIT, maxlen=16),
            LEVEL_REFLECT: TimescaleBuffer(LEVEL_REFLECT, maxlen=8),
        }
        self._tick_count: int = 0
        self._context_cache: dict[int, dict[str, float]] = {}

    def tick(self, tick: int, obs: dict[str, float]) -> dict[int, dict[str, float]]:
        """
        Update all level buffers with current obs.
        Returns dict of level → aggregates for this tick.
        """
        if not timescale_enabled():
            return {}

        self._tick_count += 1

        # Push obs to each level's buffer on its schedule
        for level, buf in self.buffers.items():
            if self.dispatcher.should_run(level, tick):
                buf.push(obs)

        # Decay intents every 60 ticks
        if tick % 60 == 0:
            self.propagator.decay_tick()

        # Build context cache
        self._context_cache = {
            level: buf.get_aggregates()
            for level, buf in self.buffers.items()
        }
        return self._context_cache

    def should_run(self, level: int, tick: int) -> bool:
        return self.dispatcher.should_run(level, tick)

    def mark_ran(self, level: int, tick: int) -> None:
        self.dispatcher.mark_ran(level, tick)

    def get_level_context(self, level: int) -> dict[str, float]:
        """Get aggregated context for a specific level."""
        return self._context_cache.get(level, self.buffers[level].get_aggregates())

    def set_intent(self, from_level: int, var: str, value: float) -> None:
        self.propagator.set_intent(from_level, var, value)

    def get_intents(self, level: int) -> dict[str, float]:
        return self.propagator.get_intents(level)

    def build_llm_temporal_context(self) -> str:
        """
        Строит форматированный временной контекст для LLM.
        LLM видит агрегаты всех уровней — от рефлекса до когниции.
        """
        lines = ["TEMPORAL CONTEXT (multi-scale):"]

        for level in [LEVEL_REFLEX, LEVEL_MOTOR, LEVEL_COGNIT]:
            ctx = self.get_level_context(level)
            name = LEVEL_NAMES[level]
            period = self.dispatcher._periods[level]
            runs = self.dispatcher._run_counts.get(level, 0)
            lines.append(f"\n[τ{level} {name.upper()} every ~{period} ticks, {runs} updates]")
            # Show key vars only
            key_items = {k: v for k, v in ctx.items() if not k.endswith("_delta")}
            for k, v in sorted(key_items.items())[:8]:
                lines.append(f"  {k}={v:.3f}")

        intents = self.propagator.snapshot()
        if intents:
            lines.append("\nACTIVE INTENTS (from higher levels):")
            for level_key, intent_dict in intents.items():
                if intent_dict:
                    lines.append(f"  [{level_key}] {intent_dict}")

        return "\n".join(lines)

    def snapshot(self) -> dict[str, Any]:
        return {
            "enabled": timescale_enabled(),
            "tick_count": self._tick_count,
            "dispatcher": self.dispatcher.snapshot(),
            "propagator": self.propagator.snapshot(),
            "buffers": {
                LEVEL_NAMES[l]: buf.snapshot()
                for l, buf in self.buffers.items()
            },
        }
