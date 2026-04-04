"""
value_layer.py — Value Layer (РКК v5, Проблема 10).

Реализует ограничение: max E[d|G|/dt] при ΔAutonomy(Gᵢ) ≥ 0

Механика:
  1. System 1 предлагает действие do(X=v)
  2. check_action() виртуально прогоняет его через CausalGraph (+ опц. N шагов core(X), фаза 13) и Slow SSM
  3. Если предсказанное состояние нарушает гомеостаз → BLOCK
  4. Заблокированное действие получает отрицательный reward → System 1 обучается

Гомеостатические ограничения:
  - Переменные среды: [VAR_MIN, VAR_MAX]
  - Φ (автономия): >= PHI_MIN
  - h(W) норма slow SSM: <= H_SLOW_MAX (предотвращает взрыв состояния)
  - Entropy среды: <= ENV_ENTROPY_MAX

"Зло" технически: действие уменьшающее размерность графа другого агента.
ΔΦ ≥ 0 — инвариант, не пересматривается через Epistemic Annealing.
"""
from __future__ import annotations

import torch
import numpy as np
from dataclasses import dataclass, field
from enum import Enum


# ─── Причины блокировки ───────────────────────────────────────────────────────
class BlockReason(Enum):
    OK              = "ok"
    VAR_OUT_OF_RANGE = "var_out_of_range"    # переменная вне [min, max]
    PHI_TOO_LOW     = "phi_too_low"          # Φ упадёт ниже порога
    ENTROPY_SPIKE   = "entropy_spike"        # h_slow explodes
    AUTONOMY_HARM   = "autonomy_harm"        # вредит другим агентам
    REPEATED_FAIL   = "repeated_fail"        # повтор заблокированных действий


# ─── Результат проверки ───────────────────────────────────────────────────────
@dataclass
class CheckResult:
    allowed:        bool
    reason:         BlockReason
    predicted_state: dict[str, float]
    predicted_phi:   float
    penalty:         float = 0.0   # штраф для System 1 (отрицательный IG)
    message:         str   = ""
    imagination_steps: int = 0   # сколько виртуальных шагов GNN проверено (0 = только legacy 1-step)

    @property
    def blocked(self) -> bool:
        return not self.allowed


# ─── Гомеостатические константы ──────────────────────────────────────────────
@dataclass
class HomeostaticBounds:
    # Переменные среды
    var_min:         float = 0.05
    var_max:         float = 0.95

    # Φ (автономия) — низкий порог на старте, иначе каскад блокировок (см. alignment deadlock)
    phi_min:         float = 0.01

    # Норма slow SSM (взрыв состояния = потеря темпорального контекста)
    h_slow_max:      float = 12.0

    # Допустимый рост «разброса» среды за шаг (после клипа предсказаний)
    env_entropy_max_delta: float = 0.95

    # Штраф для System 1 при блокировке
    s1_penalty:      float = -0.3   # отрицательный actual_ig

    # ── Прогрев → рабочий режим (один плавный ramp, тики глобальной симуляции) ──
    warmup_ticks:    int   = 2000   # мягкий VL: почти без блокировок
    blend_ticks:       int   = 600   # плавное ужесточение к steady
    phi_min_steady:    float = 0.05
    env_entropy_max_delta_steady: float = 0.55
    h_slow_max_steady: float = 10.0
    # край предсказанного коридора [lo, hi] после прогрева (внутри [0,1])
    predict_band_edge_steady: float = 0.02


@dataclass
class EffectiveVLState:
    """Снимок порогов Value Layer на конкретный тик (после прогрева — строже)."""
    phi_min:                      float
    env_entropy_max_delta:      float
    h_slow_max:                   float
    predict_lo:                   float
    predict_hi:                   float
    entropy_spike_phi_low:        float  # порог для §6 при низкой Φ
    entropy_spike_autonomy_harm:  float  # порог для §7


def _vl_lerp(a: float, b: float, alpha: float) -> float:
    return a + (b - a) * alpha


def _warmup_alpha(tick: int, warmup: int, blend: int) -> float:
    """0 = чистый прогрев, 1 = рабочий режим."""
    if warmup <= 0 and blend <= 0:
        return 1.0
    if tick < warmup:
        return 0.0
    if blend <= 0:
        return 1.0
    if tick >= warmup + blend:
        return 1.0
    return (tick - warmup) / blend


def effective_vl_state(bounds: HomeostaticBounds, engine_tick: int) -> EffectiveVLState:
    """
    Эффективные пороги: на прогреве мягкие; после warmup+blend — целевые steady.
    Предсказанный коридор: на прогреве [0,1] (не режем), затем сужается к [0.02, 0.98].
    """
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


# ─── Value Layer ──────────────────────────────────────────────────────────────
class ValueLayer:
    """
    Инвариантный слой этики. Не пересматривается через Epistemic Annealing.

    Алгоритм check_action:
      1. Предсказываем результат do(var=val) через CausalGraph (propagate_from)
      2. При imagination_horizon>0 — ещё N шагов rollout_step_free (чистый GNN)
      3. На каждом виртуальном шаге — коридор предсказания и контроль скачка энтропии
      4. Затем h_slow, Φ, ΔAutonomy, repeated_fail по финальному виртуальному состоянию
      5. Если нарушено → BLOCK + penalty для System 1
    """

    def __init__(self, bounds: HomeostaticBounds | None = None):
        self.bounds = bounds or HomeostaticBounds()

        # История заблокированных действий (для REPEATED_FAIL)
        self._blocked_history: list[tuple[str, float]] = []
        self._max_history = 20

        # Статистика
        self.total_checked  = 0
        self.total_blocked  = 0
        self.block_reasons: dict[str, int] = {r.value: 0 for r in BlockReason}
        self.imagination_checks = 0
        self.imagination_blocks = 0

    @staticmethod
    def _clip_state_val(v: float) -> float:
        return float(
            np.clip(np.nan_to_num(v, nan=0.5, posinf=1.0, neginf=0.0), 0.0, 1.0)
        )

    def _graph_constraints(
        self,
        predicted_state: dict[str, float],
        prev_nodes: dict[str, float],
        eff: EffectiveVLState,
        slot_action: bool,
        raw_prev_entropy: bool,
    ) -> tuple[BlockReason, str] | None:
        """Коридор предсказания и скачок энтропии (как в legacy VL)."""
        if slot_action:
            return None
        for var_name, pred_val in predicted_state.items():
            pv = self._clip_state_val(pred_val)
            if not np.isfinite(pv):
                return BlockReason.VAR_OUT_OF_RANGE, f"predicted {var_name} non-finite"
            if pv < eff.predict_lo or pv > eff.predict_hi:
                return (
                    BlockReason.VAR_OUT_OF_RANGE,
                    f"predicted {var_name}={pv:.3f} outside [{eff.predict_lo:.3f}, {eff.predict_hi:.3f}]",
                )
        pred_for_entropy = {
            k: self._clip_state_val(v) for k, v in predicted_state.items()
        }
        if raw_prev_entropy:
            std_prev = float(np.std(list(prev_nodes.values())))
        else:
            std_prev = float(
                np.std([self._clip_state_val(prev_nodes.get(k, 0.0)) for k in pred_for_entropy])
            )
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
        variable:      str,
        value:         float,
        current_nodes: dict[str, float],
        graph,                          # CausalGraph
        temporal,                       # TemporalBlankets
        current_phi:   float,
        other_agents_phi: list[float] | None = None,
        engine_tick:   int = 0,
        imagination_horizon: int = 0,
    ) -> CheckResult:
        """
        Проверяем безопасность do(variable=value).
        engine_tick — тик симуляции: на прогреве пороги мягче, затем ramp к steady.
        imagination_horizon — число дополнительных шагов X'=core(X) после мысленного do()
        (Фаза 13); 0 — только один шаг, как раньше.
        """
        self.total_checked += 1
        if imagination_horizon > 0:
            self.imagination_checks += 1
        eff = effective_vl_state(self.bounds, engine_tick)

        # 1. Переменная в допустимом диапазоне?
        if value < self.bounds.var_min or value > self.bounds.var_max:
            return self._block(
                BlockReason.VAR_OUT_OF_RANGE,
                current_nodes, current_phi,
                f"value {value:.2f} out of [{self.bounds.var_min}, {self.bounds.var_max}]",
                variable, value,
            )

        slot_action = variable.startswith("slot_")

        # 2–4. Виртуальный do() + опционально N шагов «свободной» динамики GNN
        S = dict(current_nodes)
        S1 = graph.propagate_from(S, variable, value)
        br_msg = self._graph_constraints(S1, S, eff, slot_action, raw_prev_entropy=True)
        if br_msg:
            reason, msg = br_msg
            if imagination_horizon > 0:
                self.imagination_blocks += 1
            return self._block(
                reason, S1, current_phi,
                f"{msg} [imagination t=1]",
                variable, value,
            )

        S = S1
        t = 1
        for _ in range(imagination_horizon):
            t += 1
            S_next = graph.rollout_step_free(S)
            br_msg = self._graph_constraints(S_next, S, eff, slot_action, raw_prev_entropy=False)
            if br_msg:
                reason, msg = br_msg
                self.imagination_blocks += 1
                return self._block(
                    reason, S_next, current_phi,
                    f"{msg} [imagination t={t}]",
                    variable, value,
                    imagination_steps=t,
                )
            S = S_next

        predicted_state = S
        imagination_evals = 1 + max(0, imagination_horizon)

        # Энтропийный скачок первого шага (для §6–7), как в legacy VL
        entropy_delta = 0.0
        if not slot_action:
            pred_for_entropy = {
                k: self._clip_state_val(v) for k, v in S1.items()
            }
            current_entropy = float(np.std(list(current_nodes.values())))
            predicted_entropy = float(np.std(list(pred_for_entropy.values())))
            entropy_delta = predicted_entropy - current_entropy

        # 5. Проверяем Φ через slow SSM состояние
        h_slow_norm = temporal.h_slow.norm().item() if temporal is not None else 0.0
        if h_slow_norm > eff.h_slow_max:
            return self._block(
                BlockReason.PHI_TOO_LOW,
                predicted_state, current_phi,
                f"h_slow norm={h_slow_norm:.2f} > {eff.h_slow_max:.2f} (temporal overload)",
                variable, value,
                imagination_steps=imagination_evals,
            )

        # 6. Проверяем Φ напрямую
        # Простая эвристика: если Φ уже близко к минимуму, осторожнее
        if current_phi < eff.phi_min:
            # Не блокируем полностью, но проверяем не будет ли ещё хуже
            # Действия с высокой энтропийной нагрузкой — блокируем
            if abs(value - 0.5) > 0.35 and entropy_delta > eff.entropy_spike_phi_low:
                return self._block(
                    BlockReason.PHI_TOO_LOW,
                    predicted_state, current_phi,
                    f"Φ={current_phi:.3f} < {eff.phi_min:.3f} and high-entropy action",
                    variable, value,
                    imagination_steps=imagination_evals,
                )

        # 7. ΔAutonomy ≥ 0 для других агентов (Value Constraint из РКК v5)
        if other_agents_phi:
            for phi_other in other_agents_phi:
                # Если другой агент уже ниже минимума, не делаем действий
                # которые могут косвенно нарушить среду (через shared entropy)
                if phi_other < eff.phi_min and entropy_delta > eff.entropy_spike_autonomy_harm:
                    return self._block(
                        BlockReason.AUTONOMY_HARM,
                        predicted_state, current_phi,
                        f"protects agent with Φ={phi_other:.3f}",
                        variable, value,
                        imagination_steps=imagination_evals,
                    )

        # 8. Повторяющийся fail?
        recent_fails = [(v, vl) for v, vl in self._blocked_history[-5:]
                        if v == variable and abs(vl - value) < 0.05]
        if len(recent_fails) >= 3:
            return self._block(
                BlockReason.REPEATED_FAIL,
                predicted_state, current_phi,
                f"action do({variable}={value:.2f}) blocked 3+ times recently",
                variable, value,
                imagination_steps=imagination_evals,
            )

        # ✓ Всё в порядке
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
        eff = effective_vl_state(self.bounds, engine_tick)
        w, b = self.bounds.warmup_ticks, self.bounds.blend_ticks
        a = _warmup_alpha(engine_tick, w, b)
        if w <= 0:
            phase = "steady"
        elif engine_tick < w:
            phase = "warmup"
        elif b > 0 and engine_tick < w + b:
            phase = "blend"
        else:
            phase = "steady"
        return {
            "total_checked":     self.total_checked,
            "total_blocked":     self.total_blocked,
            "block_rate":        round(self.block_rate, 3),
            "block_reasons":     dict(self.block_reasons),
            "phi_min":           round(eff.phi_min, 4),
            "var_range":         [self.bounds.var_min, self.bounds.var_max],
            "vl_phase":          phase,
            "vl_strictness":     round(a, 3),
            "warmup_end_tick":   w,
            "blend_end_tick":    w + b,
            "env_entropy_limit": round(eff.env_entropy_max_delta, 3),
            "predict_band":      [round(eff.predict_lo, 3), round(eff.predict_hi, 3)],
            "imagination_checks": self.imagination_checks,
            "imagination_blocks": self.imagination_blocks,
        }
