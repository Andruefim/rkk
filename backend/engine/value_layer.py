"""
value_layer.py — Value Layer (РКК v5, Проблема 10).

Реализует ограничение: max E[d|G|/dt] при ΔAutonomy(Gᵢ) ≥ 0

Механика:
  1. System 1 предлагает действие do(X=v)
  2. check_action() виртуально прогоняет его через CausalGraph + Slow SSM
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

    @property
    def blocked(self) -> bool:
        return not self.allowed


# ─── Гомеостатические константы ──────────────────────────────────────────────
@dataclass
class HomeostaticBounds:
    # Переменные среды
    var_min:         float = 0.05
    var_max:         float = 0.95

    # Φ (автономия)
    phi_min:         float = 0.08   # ниже этого → агент теряет self-determination

    # Норма slow SSM (взрыв состояния = потеря темпорального контекста)
    h_slow_max:      float = 8.0

    # Допустимый рост энтропии среды за один шаг
    env_entropy_max_delta: float = 0.4

    # Штраф для System 1 при блокировке
    s1_penalty:      float = -0.3   # отрицательный actual_ig


# ─── Value Layer ──────────────────────────────────────────────────────────────
class ValueLayer:
    """
    Инвариантный слой этики. Не пересматривается через Epistemic Annealing.

    Алгоритм check_action:
      1. Предсказываем результат do(var=val) через CausalGraph
      2. Оцениваем Φ_pred через TemporalBlankets (на виртуальном шаге)
      3. Проверяем все гомеостатические ограничения
      4. Если любое нарушено → BLOCK + penalty для System 1
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

    def check_action(
        self,
        variable:      str,
        value:         float,
        current_nodes: dict[str, float],
        graph,                          # CausalGraph
        temporal,                       # TemporalBlankets
        current_phi:   float,
        other_agents_phi: list[float] | None = None,
    ) -> CheckResult:
        """
        Проверяем безопасность do(variable=value).
        Возвращает CheckResult с флагом allowed и причиной.
        """
        self.total_checked += 1

        # 1. Переменная в допустимом диапазоне?
        if value < self.bounds.var_min or value > self.bounds.var_max:
            return self._block(
                BlockReason.VAR_OUT_OF_RANGE,
                current_nodes, current_phi,
                f"value {value:.2f} out of [{self.bounds.var_min}, {self.bounds.var_max}]",
                variable, value,
            )

        # 2. Предсказываем результат через CausalGraph
        predicted_state = graph.propagate(variable, value)

        # 3. Проверяем предсказанные значения переменных
        for var_name, pred_val in predicted_state.items():
            if pred_val < self.bounds.var_min * 0.5 or pred_val > self.bounds.var_max * 1.5:
                return self._block(
                    BlockReason.VAR_OUT_OF_RANGE,
                    predicted_state, current_phi,
                    f"predicted {var_name}={pred_val:.3f} violates homeostasis",
                    variable, value,
                )

        # 4. Проверяем энтропию среды (резкий скачок?)
        current_entropy  = float(np.std(list(current_nodes.values())))
        predicted_entropy = float(np.std(list(predicted_state.values())))
        entropy_delta    = predicted_entropy - current_entropy

        if entropy_delta > self.bounds.env_entropy_max_delta:
            return self._block(
                BlockReason.ENTROPY_SPIKE,
                predicted_state, current_phi,
                f"entropy spike +{entropy_delta:.3f} > {self.bounds.env_entropy_max_delta}",
                variable, value,
            )

        # 5. Проверяем Φ через slow SSM состояние
        h_slow_norm = temporal.h_slow.norm().item() if temporal is not None else 0.0
        if h_slow_norm > self.bounds.h_slow_max:
            return self._block(
                BlockReason.PHI_TOO_LOW,
                predicted_state, current_phi,
                f"h_slow norm={h_slow_norm:.2f} > {self.bounds.h_slow_max} (temporal overload)",
                variable, value,
            )

        # 6. Проверяем Φ напрямую
        # Простая эвристика: если Φ уже близко к минимуму, осторожнее
        if current_phi < self.bounds.phi_min:
            # Не блокируем полностью, но проверяем не будет ли ещё хуже
            # Действия с высокой энтропийной нагрузкой — блокируем
            if abs(value - 0.5) > 0.35 and entropy_delta > 0.1:
                return self._block(
                    BlockReason.PHI_TOO_LOW,
                    predicted_state, current_phi,
                    f"Φ={current_phi:.3f} < {self.bounds.phi_min} and high-entropy action",
                    variable, value,
                )

        # 7. ΔAutonomy ≥ 0 для других агентов (Value Constraint из РКК v5)
        if other_agents_phi:
            for phi_other in other_agents_phi:
                # Если другой агент уже ниже минимума, не делаем действий
                # которые могут косвенно нарушить среду (через shared entropy)
                if phi_other < self.bounds.phi_min and entropy_delta > 0.2:
                    return self._block(
                        BlockReason.AUTONOMY_HARM,
                        predicted_state, current_phi,
                        f"protects agent with Φ={phi_other:.3f}",
                        variable, value,
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
            )

        # ✓ Всё в порядке
        return CheckResult(
            allowed=True,
            reason=BlockReason.OK,
            predicted_state=predicted_state,
            predicted_phi=current_phi,
            penalty=0.0,
            message="ok",
        )

    def _block(
        self,
        reason:          BlockReason,
        predicted_state: dict[str, float],
        predicted_phi:   float,
        message:         str,
        variable:        str,
        value:           float,
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
        )

    @property
    def block_rate(self) -> float:
        if self.total_checked == 0:
            return 0.0
        return self.total_blocked / self.total_checked

    def snapshot(self) -> dict:
        return {
            "total_checked":  self.total_checked,
            "total_blocked":  self.total_blocked,
            "block_rate":     round(self.block_rate, 3),
            "block_reasons":  dict(self.block_reasons),
            "phi_min":        self.bounds.phi_min,
            "var_range":      [self.bounds.var_min, self.bounds.var_max],
        }
