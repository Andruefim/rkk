"""
value_layer.py — Value Layer (РКК v5, Проблема 10) + fixed_root mode.

Добавлено:
  - HomeostaticBounds.fixed_root_mode: bool — пропуск balance-checks
  - HomeostaticBounds.for_fixed_root() — статический метод с мягкими параметрами
    для fixed_root mode: короткий warmup, высокие энтропийные пределы,
    нет ограничений на com_z/fallen.

В fixed_root mode check_action пропускает:
  - PHI_TOO_LOW из-за h_slow (arms не взрывают state)
  - Entropy spike (куб может переместиться резко — это нормально)
  - REPEATED_FAIL снижен до 5 повторений вместо 3

Это устраняет основной alignment deadlock: System1 штрафовалась за
каждое движение рукой, потому что куб смещался и вызывал entropy spike.
"""
from __future__ import annotations

import os
import torch
import numpy as np
from dataclasses import dataclass, field
from enum import Enum

from engine.graph_constants import is_read_only_macro_var


class BlockReason(Enum):
    OK               = "ok"
    VAR_OUT_OF_RANGE = "var_out_of_range"
    PHI_TOO_LOW      = "phi_too_low"
    ENTROPY_SPIKE    = "entropy_spike"
    AUTONOMY_HARM    = "autonomy_harm"
    REPEATED_FAIL    = "repeated_fail"
    READ_ONLY_MACRO  = "read_only_macro"


@dataclass
class CheckResult:
    allowed:         bool
    reason:          BlockReason
    predicted_state: dict[str, float]
    predicted_phi:   float
    penalty:         float = 0.0
    message:         str   = ""
    imagination_steps: int = 0

    @property
    def blocked(self) -> bool:
        return not self.allowed


@dataclass
class HomeostaticBounds:
    var_min:         float = 0.05
    var_max:         float = 0.95
    phi_min:         float = 0.005
    h_slow_max:      float = 18.0
    env_entropy_max_delta: float = 1.2
    s1_penalty:      float = -0.2

    warmup_ticks:    int   = 800
    blend_ticks:     int   = 400
    phi_min_steady:  float = 0.03
    env_entropy_max_delta_steady: float = 0.65
    h_slow_max_steady: float = 14.0
    predict_band_edge_steady: float = 0.015

    # ── fixed_root mode ──────────────────────────────────────────────────────
    # Когда True: пропускаем entropy_spike и phi checks связанные с балансом.
    # Агент исследует arms→cubes без постоянных блокировок.
    fixed_root_mode: bool = False

    @staticmethod
    def for_fixed_root() -> "HomeostaticBounds":
        """
        Параметры Value Layer для fixed_root режима.

        Ключевые изменения:
          - warmup_ticks=300: нет смысла долго ждать, баланс не нужен
          - blend_ticks=200: быстрый переход к рабочим параметрам
          - env_entropy_max_delta=2.0: куб может резко переместиться
          - env_entropy_max_delta_steady=1.2: в рабочем режиме тоже мягко
          - h_slow_max=20.0: руки двигаются активнее → больший temporal state
          - predict_band_edge_steady=0.005: узкий коридор почти не ограничивает
          - fixed_root_mode=True: сигнал для check_action
        """
        return HomeostaticBounds(
            var_min=0.05,
            var_max=0.95,
            phi_min=0.01,
            h_slow_max=20.0,
            env_entropy_max_delta=2.0,
            s1_penalty=-0.15,           # мягче штраф — не надо пугать S1

            warmup_ticks=300,
            blend_ticks=200,
            phi_min_steady=0.02,
            env_entropy_max_delta_steady=1.2,
            h_slow_max_steady=16.0,
            predict_band_edge_steady=0.005,

            fixed_root_mode=True,
        )


@dataclass
class EffectiveVLState:
    phi_min:                      float
    env_entropy_max_delta:        float
    h_slow_max:                   float
    predict_lo:                   float
    predict_hi:                   float
    entropy_spike_phi_low:        float
    entropy_spike_autonomy_harm:  float


@dataclass
class TeacherVLOverlay:
    expires_at_tick: int
    phi_min_delta: float = 0.0
    env_entropy_max_delta: float = 0.0
    h_slow_max_delta: float = 0.0
    predict_lo_delta: float = 0.0
    predict_hi_delta: float = 0.0
    entropy_spike_phi_low_delta: float = 0.0
    entropy_spike_autonomy_delta: float = 0.0


def merge_teacher_vl(
    eff: EffectiveVLState,
    overlay: TeacherVLOverlay | None,
    engine_tick: int,
) -> EffectiveVLState:
    if overlay is None or engine_tick > overlay.expires_at_tick:
        return eff
    pm = float(np.clip(eff.phi_min + overlay.phi_min_delta, 0.005, 0.42))
    em = float(np.clip(eff.env_entropy_max_delta + overlay.env_entropy_max_delta, 0.08, 1.55))
    hm = float(np.clip(eff.h_slow_max + overlay.h_slow_max_delta, 4.0, 24.0))
    pl = float(np.clip(eff.predict_lo + overlay.predict_lo_delta, 0.0, 0.48))
    ph = float(np.clip(eff.predict_hi + overlay.predict_hi_delta, 0.52, 1.0))
    if pl >= ph - 0.02:
        mid = 0.5 * (pl + ph)
        pl, ph = mid - 0.02, mid + 0.02
    espl = float(np.clip(eff.entropy_spike_phi_low + overlay.entropy_spike_phi_low_delta, 0.05, 0.92))
    esph = float(np.clip(eff.entropy_spike_autonomy_harm + overlay.entropy_spike_autonomy_delta, 0.05, 0.92))
    return EffectiveVLState(
        phi_min=pm,
        env_entropy_max_delta=em,
        h_slow_max=hm,
        predict_lo=pl,
        predict_hi=ph,
        entropy_spike_phi_low=espl,
        entropy_spike_autonomy_harm=esph,
    )


def _vl_lerp(a: float, b: float, alpha: float) -> float:
    return a + (b - a) * alpha


def _warmup_alpha(tick: int, warmup: int, blend: int) -> float:
    if warmup <= 0 and blend <= 0:
        return 1.0
    if tick < warmup:
        return 0.0
    if blend <= 0:
        return 1.0
    if tick >= warmup + blend:
        return 1.0
    return (tick - warmup) / blend


def _loco_warmup_relax(engine_tick: int) -> float:
    """
    1.0 early locomotion warmup, 0.0 after warmup.
    Shorter warmup so the agent can explore motor intents earlier.
    """
    try:
        warm = int(os.environ.get("RKK_LOCO_VL_WARMUP_TICKS", "1500"))
    except ValueError:
        warm = 1500
    try:
        blend = int(os.environ.get("RKK_LOCO_VL_WARMUP_BLEND", "800"))
    except ValueError:
        blend = 800
    return float(np.clip(1.0 - _warmup_alpha(engine_tick, warm, blend), 0.0, 1.0))


def effective_vl_state(bounds: HomeostaticBounds, engine_tick: int) -> EffectiveVLState:
    a = _warmup_alpha(engine_tick, bounds.warmup_ticks, bounds.blend_ticks)
    lo = _vl_lerp(0.0, bounds.predict_band_edge_steady, a)
    hi = _vl_lerp(1.0, 1.0 - bounds.predict_band_edge_steady, a)
    return EffectiveVLState(
        phi_min=_vl_lerp(bounds.phi_min, bounds.phi_min_steady, a),
        env_entropy_max_delta=_vl_lerp(
            bounds.env_entropy_max_delta,
            bounds.env_entropy_max_delta_steady,
            a,
        ),
        h_slow_max=_vl_lerp(bounds.h_slow_max, bounds.h_slow_max_steady, a),
        predict_lo=lo,
        predict_hi=hi,
        entropy_spike_phi_low=_vl_lerp(0.40, 0.22, a),
        entropy_spike_autonomy_harm=_vl_lerp(0.48, 0.35, a),
    )


class ValueLayer:
    """
    Value Layer с поддержкой fixed_root mode.

    В fixed_root mode:
      - check §3 (entropy_spike) пропускается — куб может резко сместиться
      - check §5 (h_slow temporal overload) пропускается — arms активны
      - check §6 (phi + entropy) пропускается — нет риска падения
      - check §8 (repeated_fail) порог поднят до 5 (против 3)
    Остаются:
      - §1 var_min/var_max (физический диапазон переменной)
      - §7 autonomy_harm для других агентов (мульти-агент, если есть)
    """

    def __init__(self, bounds: HomeostaticBounds | None = None):
        self.bounds = bounds or HomeostaticBounds()
        self._teacher_vl_overlay: TeacherVLOverlay | None = None
        self._blocked_history: list[tuple[str, float]] = []
        self._max_history = 20
        self.total_checked  = 0
        self.total_blocked  = 0
        self.block_reasons: dict[str, int] = {r.value: 0 for r in BlockReason}
        self.imagination_checks = 0
        self.imagination_blocks = 0

    def set_teacher_vl_overlay(self, overlay: TeacherVLOverlay | None) -> None:
        self._teacher_vl_overlay = overlay

    @staticmethod
    def _clip_state_val(v: float) -> float:
        return float(np.clip(np.nan_to_num(v, nan=0.5, posinf=1.0, neginf=0.0), 0.0, 1.0))

    def _graph_constraints(
        self,
        predicted_state: dict[str, float],
        prev_nodes: dict[str, float],
        eff: EffectiveVLState,
        slot_action: bool,
        raw_prev_entropy: bool,
        skip_entropy: bool = False,
    ) -> tuple[BlockReason, str] | None:
        if slot_action:
            return None
        # predict_band check
        for var_name, pred_val in predicted_state.items():
            pv = self._clip_state_val(pred_val)
            if not np.isfinite(pv):
                return BlockReason.VAR_OUT_OF_RANGE, f"predicted {var_name} non-finite"
            if pv < eff.predict_lo or pv > eff.predict_hi:
                return (
                    BlockReason.VAR_OUT_OF_RANGE,
                    f"predicted {var_name}={pv:.3f} outside [{eff.predict_lo:.3f}, {eff.predict_hi:.3f}]",
                )
        # entropy spike check — пропускаем в fixed_root
        if skip_entropy:
            return None
        pred_for_entropy = {k: self._clip_state_val(v) for k, v in predicted_state.items()}
        if raw_prev_entropy:
            std_prev = float(np.std(list(prev_nodes.values())))
        else:
            std_prev = float(np.std([self._clip_state_val(prev_nodes.get(k, 0.0)) for k in pred_for_entropy]))
        std_new = float(np.std(list(pred_for_entropy.values())))
        d = std_new - std_prev
        if d > eff.env_entropy_max_delta:
            return (
                BlockReason.ENTROPY_SPIKE,
                f"entropy spike +{d:.3f} > {eff.env_entropy_max_delta:.3f}",
            )
        return None

    def check_action(
        self,
        variable:         str,
        value:            float,
        current_nodes:    dict[str, float],
        graph,
        temporal,
        current_phi:      float,
        other_agents_phi: list[float] | None = None,
        engine_tick:      int = 0,
        imagination_horizon: int = 0,
    ) -> CheckResult:
        self.total_checked += 1
        if imagination_horizon > 0:
            self.imagination_checks += 1

        eff = effective_vl_state(self.bounds, engine_tick)
        eff = merge_teacher_vl(eff, self._teacher_vl_overlay, engine_tick)
        fixed_root = self.bounds.fixed_root_mode

        # §1 Переменная в диапазоне
        if value < self.bounds.var_min or value > self.bounds.var_max:
            return self._block(
                BlockReason.VAR_OUT_OF_RANGE, current_nodes, current_phi,
                f"value {value:.2f} out of [{self.bounds.var_min}, {self.bounds.var_max}]",
                variable, value,
            )

        if is_read_only_macro_var(variable):
            return self._block(
                BlockReason.READ_ONLY_MACRO,
                current_nodes,
                current_phi,
                "concept_* — read-only macro (aggregate); do() disabled",
                variable,
                value,
            )

        # slot_*, self_* и intent_* — «внутренние» оси; не жмём узкий predict_band как у физики.
        slot_action = (
            variable.startswith("slot_")
            or variable.startswith("self_")
            or variable.startswith("intent_")
            or variable.startswith("motor_")
            or variable.startswith("phys_intent_")
            or variable.startswith("phys_self_")
        )
        is_intent_action = variable.startswith("intent_") or variable.startswith("phys_intent_")
        loco_relax = 0.0
        relaxed_phi_min = eff.phi_min
        relaxed_h_slow_max = eff.h_slow_max
        relaxed_entropy_phi_low = eff.entropy_spike_phi_low
        repeat_threshold = 5 if fixed_root else 3
        if is_intent_action:
            def _n(k: str, default: float = 0.5) -> float:
                if k in current_nodes:
                    return float(current_nodes[k])
                pk = f"phys_{k}"
                if pk in current_nodes:
                    return float(current_nodes[pk])
                return float(default)

            posture = _n("posture_stability", 0.5)
            support_bias = _n("support_bias", 0.5)
            drive_l = _n("motor_drive_l", 0.5)
            drive_r = _n("motor_drive_r", 0.5)
            gait_l = _n("gait_phase_l", 0.5)
            gait_r = _n("gait_phase_r", 0.5)
            intent_key = variable[5:] if variable.startswith("phys_") else variable
            shift = abs(float(value) - 0.5)
            loco_relax = _loco_warmup_relax(engine_tick)
            if intent_key == "intent_stop_recover":
                loco_relax = max(loco_relax, 0.65)
            if posture < 0.52:
                loco_relax = max(loco_relax, float(np.clip((0.52 - posture) / 0.22, 0.0, 1.0)))
            relaxed_phi_min = eff.phi_min * (1.0 - 0.55 * loco_relax)
            relaxed_h_slow_max = eff.h_slow_max * (1.0 + 0.85 * loco_relax)
            relaxed_entropy_phi_low = min(0.95, eff.entropy_spike_phi_low + 0.20 * loco_relax)
            repeat_threshold = max(repeat_threshold, 3 + int(round(3.0 * loco_relax)))
            # Overspeed intent: already high motor drive + aggressive new intent command.
            if shift > (0.40 + 0.10 * loco_relax) and max(drive_l, drive_r) > (0.82 + 0.06 * loco_relax):
                return self._block(
                    BlockReason.PHI_TOO_LOW,
                    current_nodes,
                    current_phi,
                    f"motor overspeed: drive={max(drive_l, drive_r):.2f}, shift={shift:.2f}",
                    variable,
                    value,
                )
            # Destabilizing support switch while posture is low.
            if intent_key in ("intent_support_left", "intent_support_right"):
                wants_left = intent_key.endswith("left") and value > 0.6
                wants_right = intent_key.endswith("right") and value > 0.6
                unstable = posture < (0.36 - 0.06 * loco_relax)
                opposite_bias = (
                    (wants_left and support_bias > (0.62 + 0.08 * loco_relax))
                    or (wants_right and support_bias < (0.38 - 0.08 * loco_relax))
                )
                if unstable and opposite_bias:
                    return self._block(
                        BlockReason.ENTROPY_SPIKE,
                        current_nodes,
                        current_phi,
                        f"destabilizing support shift: posture={posture:.2f}, bias={support_bias:.2f}",
                        variable,
                        value,
                    )
            # Repeated unstable gait mode: aggressive stride while gait already desynchronized.
            if intent_key == "intent_stride":
                gait_desync = abs(gait_l - gait_r)
                if shift > (0.34 + 0.08 * loco_relax) and posture < (0.42 - 0.06 * loco_relax) and gait_desync > (0.38 + 0.08 * loco_relax):
                    return self._block(
                        BlockReason.REPEATED_FAIL,
                        current_nodes,
                        current_phi,
                        f"unstable gait mode: posture={posture:.2f}, desync={gait_desync:.2f}",
                        variable,
                        value,
                    )

        # §2–4 Виртуальный do() + imagination rollout
        S = dict(current_nodes)
        S1 = graph.propagate_from(S, variable, value)
        br_msg = self._graph_constraints(
            S1, S, eff, slot_action, raw_prev_entropy=True,
            skip_entropy=fixed_root,  # в fixed_root пропускаем entropy_spike
        )
        if br_msg:
            reason, msg = br_msg
            if imagination_horizon > 0:
                self.imagination_blocks += 1
            return self._block(reason, S1, current_phi, f"{msg} [t=1]", variable, value)

        S = S1
        t = 1
        for _ in range(imagination_horizon):
            t += 1
            S_next = graph.rollout_step_free(S)
            br_msg = self._graph_constraints(
                S_next, S, eff, slot_action, raw_prev_entropy=False,
                skip_entropy=fixed_root,
            )
            if br_msg:
                reason, msg = br_msg
                self.imagination_blocks += 1
                return self._block(
                    reason, S_next, current_phi, f"{msg} [t={t}]", variable, value,
                    imagination_steps=t,
                )
            S = S_next

        predicted_state = S
        imagination_evals = 1 + max(0, imagination_horizon)

        # entropy_delta для §6–7 (только если не fixed_root)
        entropy_delta = 0.0
        if not slot_action and not fixed_root:
            pred_for_entropy = {k: self._clip_state_val(v) for k, v in S1.items()}
            current_entropy   = float(np.std(list(current_nodes.values())))
            predicted_entropy = float(np.std(list(pred_for_entropy.values())))
            entropy_delta     = predicted_entropy - current_entropy

        # §5 h_slow temporal overload — пропускаем в fixed_root
        if not fixed_root:
            h_slow_norm = temporal.h_slow.norm().item() if temporal is not None else 0.0
            if h_slow_norm > relaxed_h_slow_max:
                return self._block(
                    BlockReason.PHI_TOO_LOW, predicted_state, current_phi,
                    f"h_slow norm={h_slow_norm:.2f} > {relaxed_h_slow_max:.2f} (temporal overload)",
                    variable, value, imagination_steps=imagination_evals,
                )

        # §6 Φ near minimum + high entropy — пропускаем в fixed_root
        if not fixed_root:
            if current_phi < relaxed_phi_min:
                if abs(value - 0.5) > 0.35 and entropy_delta > relaxed_entropy_phi_low:
                    return self._block(
                        BlockReason.PHI_TOO_LOW, predicted_state, current_phi,
                        f"Φ={current_phi:.3f} < {relaxed_phi_min:.3f} and high-entropy action",
                        variable, value, imagination_steps=imagination_evals,
                    )

        # §7 ΔAutonomy ≥ 0 для других агентов
        if other_agents_phi and not fixed_root:
            for phi_other in other_agents_phi:
                if phi_other < eff.phi_min and entropy_delta > eff.entropy_spike_autonomy_harm:
                    return self._block(
                        BlockReason.AUTONOMY_HARM, predicted_state, current_phi,
                        f"protects agent with Φ={phi_other:.3f}",
                        variable, value, imagination_steps=imagination_evals,
                    )

        # §8 Repeated fail — intent warmup gets additional slack.
        recent_fails = [
            (v, vl) for v, vl in self._blocked_history[-5:]
            if v == variable and abs(vl - value) < 0.05
        ]
        if len(recent_fails) >= repeat_threshold:
            return self._block(
                BlockReason.REPEATED_FAIL, predicted_state, current_phi,
                f"action do({variable}={value:.2f}) blocked {repeat_threshold}+ times recently",
                variable, value, imagination_steps=imagination_evals,
            )

        return CheckResult(
            allowed=True,
            reason=BlockReason.OK,
            predicted_state=predicted_state,
            predicted_phi=current_phi,
            penalty=0.0,
            message="ok",
            imagination_steps=imagination_evals,
        )

    def _block(
        self,
        reason:          BlockReason,
        predicted_state: dict[str, float],
        predicted_phi:   float,
        message:         str,
        variable:        str,
        value:           float,
        imagination_steps: int = 0,
    ) -> CheckResult:
        self.total_blocked += 1
        self.block_reasons[reason.value] = self.block_reasons.get(reason.value, 0) + 1
        self._blocked_history.append((variable, value))
        if len(self._blocked_history) > self._max_history:
            self._blocked_history.pop(0)
        return CheckResult(
            allowed=False,
            reason=reason,
            predicted_state=predicted_state,
            predicted_phi=predicted_phi,
            penalty=self.bounds.s1_penalty,
            message=message,
            imagination_steps=imagination_steps,
        )

    @property
    def block_rate(self) -> float:
        if self.total_checked == 0:
            return 0.0
        return self.total_blocked / self.total_checked

    def snapshot(self, engine_tick: int = 0) -> dict:
        eff  = effective_vl_state(self.bounds, engine_tick)
        eff_m = merge_teacher_vl(eff, self._teacher_vl_overlay, engine_tick)
        ov   = self._teacher_vl_overlay
        teacher_active = ov is not None and engine_tick <= ov.expires_at_tick
        teacher_ttl    = max(0, ov.expires_at_tick - engine_tick) if ov else 0
        w, b = self.bounds.warmup_ticks, self.bounds.blend_ticks
        a    = _warmup_alpha(engine_tick, w, b)
        if w <= 0:
            phase = "steady"
        elif engine_tick < w:
            phase = "warmup"
        elif b > 0 and engine_tick < w + b:
            phase = "blend"
        else:
            phase = "steady"
        return {
            "total_checked":       self.total_checked,
            "total_blocked":       self.total_blocked,
            "block_rate":          round(self.block_rate, 3),
            "block_reasons":       dict(self.block_reasons),
            "phi_min":             round(eff_m.phi_min, 4),
            "var_range":           [self.bounds.var_min, self.bounds.var_max],
            "vl_phase":            phase,
            "vl_strictness":       round(a, 3),
            "warmup_end_tick":     w,
            "blend_end_tick":      w + b,
            "env_entropy_limit":   round(eff_m.env_entropy_max_delta, 3),
            "predict_band":        [round(eff_m.predict_lo, 3), round(eff_m.predict_hi, 3)],
            "imagination_checks":  self.imagination_checks,
            "imagination_blocks":  self.imagination_blocks,
            "teacher_vl_active":   teacher_active,
            "teacher_vl_ttl_ticks":teacher_ttl,
            "fixed_root_mode":     self.bounds.fixed_root_mode,
        }