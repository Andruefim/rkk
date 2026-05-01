"""
intrinsic_objective.py — Unified Intrinsic Objective.

Одна цель: максимизировать I(agent; world) — взаимную информацию
между внутренней каузальной моделью и реальностью.

Операционально:
  R_intrinsic(t) = Δcompression(t) + λ·Δprediction_surprise(t)

Внешние hand-crafted награды и RewardCoordinator удалены из симуляции.

Остаётся ОДНО:
  ✓ "Насколько лучше агент понимает мир после этого действия?"

Локомоция возникает инструментально: ходить → больше новых состояний
→ больше информации → выше compression gain. Не потому что "так надо".

Архитектура:
  CausalSurprise   — Δ prediction error как интринсивная метрика
  GoalImagination  — агент генерирует свою следующую цель через GNN rollout
  VariableDiscovery — проактивное открытие новых переменных через EIG
  IntrinsicObjective — единый интерфейс, заменяет все внешние награды

Принцип нейрогенеза-через-цель:
  Если CausalSurprise растёт для области X, но модель не улучшается →
  создать новый узел в GNN для этой области (NeurogenesisEngine.scan_and_grow).

RKK_INTRINSIC_ENABLED=1
RKK_INTRINSIC_LAMBDA=0.4        — вес prediction surprise vs compression
RKK_INTRINSIC_GOAL_HORIZON=12   — шагов imagination для goal generation
RKK_INTRINSIC_DISCOVERY_EIG=0.3 — порог EIG для создания нового узла
"""
from __future__ import annotations

import json
import os
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# #region agent log
_DBG_LOG_F7_IVO = Path(__file__).resolve().parents[2] / "debug-f7a777.log"


def _dbg_ivo(hypothesis_id: str, location: str, message: str, data: dict | None = None) -> None:
    try:
        with _DBG_LOG_F7_IVO.open("a", encoding="utf-8") as _df:
            _df.write(
                json.dumps(
                    {
                        "sessionId": "f7a777",
                        "hypothesisId": hypothesis_id,
                        "location": location,
                        "message": message,
                        "data": data or {},
                        "timestamp": int(time.time() * 1000),
                        "runId": "pre-fix",
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    except Exception:
        pass


# #endregion


# ─── Config ───────────────────────────────────────────────────────────────────
def intrinsic_enabled() -> bool:
    return os.environ.get("RKK_INTRINSIC_ENABLED", "1").strip().lower() not in (
        "0", "false", "no", "off"
    )

def _ef(key: str, default: float) -> float:
    try: return float(os.environ.get(key, str(default)))
    except ValueError: return default

def _ei(key: str, default: int) -> int:
    try: return max(1, int(os.environ.get(key, str(default))))
    except ValueError: return default

# ─── CausalSurprise ───────────────────────────────────────────────────────────
class CausalSurprise:
    """
    Измеряет интринсивную ценность интервенции через два сигнала:

    1. Δcompression: насколько уменьшился MDL графа после do().
       Это прямая мера "граф стал точнее объяснять мир".

    2. Δprediction_error: насколько удивил нас результат do().
       Высокое удивление = в этой области модель плохая → исследуй.

    Итоговая интринсивная награда:
       R = sign(Δcompression) * |Δcompression| + λ * surprise_bonus

    Ключевое: surprise_bonus — это не штраф за ошибку.
    Это БОНУС за нахождение в неизведанной области.
    """

    def __init__(self):
        self._lambda = _ef("RKK_INTRINSIC_LAMBDA", 0.4)

        # Скользящие статистики для нормализации
        self._compression_history: deque[float] = deque(maxlen=200)
        self._surprise_history: deque[float] = deque(maxlen=200)
        self._reward_history: deque[float] = deque(maxlen=500)

        # EMA baseline для адаптивного порога
        self._compression_ema: float = 0.0
        self._surprise_ema: float = 0.0
        self._ema_alpha: float = 0.05

        # Счётчики
        self.total_computations: int = 0
        self.total_discoveries: int = 0  # тиков где compression > 0

    def compute(
        self,
        compression_delta: float,
        prediction_error: float,
        graph_mdl_before: float,
        graph_mdl_after: float,
        n_interventions: int,
    ) -> float:
        """
        Основной расчёт интринсивной награды.

        compression_delta: mdl_before - mdl_after (>0 = граф улучшился)
        prediction_error: |predicted - observed| (>0 = модель удивлена)
        """
        if not intrinsic_enabled():
            return 0.0

        self.total_computations += 1

        # --- Сигнал 1: Compression ---
        # Нормализуем по MDL размеру чтобы сравнивать ранние и поздние тики
        mdl_scale = max(graph_mdl_before, 1.0)
        comp_norm = float(compression_delta) / mdl_scale

        # EMA обновление
        self._compression_ema = (
            (1 - self._ema_alpha) * self._compression_ema
            + self._ema_alpha * comp_norm
        )
        self._compression_history.append(comp_norm)

        # --- Сигнал 2: Prediction Surprise ---
        # Нормализуем относительно исторической средней
        surprise_baseline = float(np.mean(self._surprise_history)) if self._surprise_history else 0.1
        surprise_norm = float(prediction_error) / max(surprise_baseline, 0.01)
        # Ограничиваем: не хотим чтобы огромные ошибки доминировали
        surprise_bonus = float(np.clip(np.log1p(surprise_norm) * 0.5, 0.0, 2.0))

        self._surprise_ema = (
            (1 - self._ema_alpha) * self._surprise_ema
            + self._ema_alpha * float(prediction_error)
        )
        self._surprise_history.append(float(prediction_error))

        # --- Объединяем ---
        # Compression gain — основной сигнал (может быть отрицательным если граф деградировал)
        # Surprise — всегда положительный бонус (исследование важно)
        reward = comp_norm + self._lambda * surprise_bonus

        # Ранний бонус: первые 500 интервенций мир совсем незнаком
        if n_interventions < 500:
            exploration_bonus = 0.3 * (1.0 - n_interventions / 500.0)
            reward += exploration_bonus

        if comp_norm > 0:
            self.total_discoveries += 1

        self._reward_history.append(reward)
        return float(reward)

    def surprise_is_high(self, threshold_sigma: float = 2.0) -> bool:
        """Текущее удивление аномально высокое? → агент в неизведанной области."""
        if len(self._surprise_history) < 20:
            return False
        arr = np.array(self._surprise_history)
        mu, sigma = float(arr.mean()), float(arr.std())
        if sigma < 1e-8:
            return False
        return float(self._surprise_ema) > mu + threshold_sigma * sigma

    def compression_is_stagnant(self, window: int = 50) -> bool:
        """Compression не растёт уже window тиков? → нужна другая стратегия."""
        if len(self._compression_history) < window:
            return False
        recent = list(self._compression_history)[-window:]
        return float(np.mean(recent)) < 1e-6

    def recent_mean(self, window: int = 32) -> float:
        if not self._reward_history:
            return 0.0
        return float(np.mean(list(self._reward_history)[-window:]))

    def snapshot(self) -> dict[str, Any]:
        return {
            "total_computations": self.total_computations,
            "total_discoveries": self.total_discoveries,
            "discovery_rate": round(
                self.total_discoveries / max(self.total_computations, 1), 4
            ),
            "compression_ema": round(self._compression_ema, 6),
            "surprise_ema": round(self._surprise_ema, 5),
            "recent_reward_mean": round(self.recent_mean(), 5),
            "surprise_is_high": self.surprise_is_high(),
            "compression_stagnant": self.compression_is_stagnant(),
        }


# ─── GoalImagination ─────────────────────────────────────────────────────────
class GoalImagination:
    """
    Агент генерирует СВОЮ СЛЕДУЮЩУЮ ЦЕЛЬ через GNN rollout.

    Алгоритм:
      1. Из текущего состояния сэмплируем K случайных интервенций
      2. Для каждой запускаем imagination rollout на H шагов
      3. Выбираем интервенцию с максимальным ожидаемым Δcompression
      4. Эта интервенция становится "целью" на следующие T тиков

    НЕТ захардкоженной цели. Никаких "дойди до target_dist".
    Цель = "сделай то, что больше всего улучшит понимание мира".

    Это фундаментально другое от goal_planning.py:
      goal_planning: "уменьши target_dist" (человеческая цель)
      GoalImagination: "максимизируй compression_gain" (интринсивная цель)
    """

    def __init__(self, device: torch.device):
        self.device = device
        self._horizon = _ei("RKK_INTRINSIC_GOAL_HORIZON", 12)
        self._k_candidates = _ei("RKK_INTRINSIC_GOAL_CANDIDATES", 32)

        # Текущая активная цель
        self._current_goal: dict[str, Any] | None = None
        self._goal_age: int = 0
        self._goal_ttl: int = 20  # тиков пока цель активна

        # История целей для анализа
        self._goal_history: deque[dict[str, Any]] = deque(maxlen=50)

        # Статистика качества imagination
        self._imagination_accuracy: deque[float] = deque(maxlen=100)

        self.total_goals_generated: int = 0

    def generate_goal(
        self,
        graph,
        agent_env,
        n_interventions: int,
        causal_surprise: CausalSurprise,
    ) -> dict[str, Any] | None:
        """
        Генерирует следующую цель через GNN imagination rollout.
        Возвращает {"variable": str, "value": float, "expected_gain": float}.
        """
        if not intrinsic_enabled():
            return None

        core = graph._core
        if core is None:
            return None

        node_ids = list(graph._node_ids)
        d = len(node_ids)
        if d == 0:
            return None

        # Исключаем read-only и self_* из кандидатов
        from engine.graph_constants import is_read_only_macro_var
        candidates_vars = [
            nid for nid in node_ids
            if not is_read_only_macro_var(nid)
            and not nid.startswith("concept_")
            and not nid.startswith("proprio_")
        ]
        if not candidates_vars:
            return None

        # Сэмплируем K интервенций
        k = min(self._k_candidates, len(candidates_vars) * 3)
        rng = np.random.default_rng()

        # Адаптивный диапазон: если стагнация — исследуем агрессивнее
        if causal_surprise.compression_is_stagnant():
            lo, hi = 0.1, 0.9
        elif causal_surprise.surprise_is_high():
            lo, hi = 0.3, 0.7  # осторожнее когда много сюрпризов
        else:
            lo, hi = 0.2, 0.8

        sampled = [
            (rng.choice(candidates_vars), float(rng.uniform(lo, hi)))
            for _ in range(k)
        ]

        # Текущее состояние
        current_state = dict(graph.nodes)
        current_mdl = float(graph.mdl_size)

        best_var, best_val, best_expected = None, 0.5, -np.inf

        with torch.no_grad():
            for var, val in sampled:
                try:
                    # GNN rollout
                    state = graph.propagate_from(current_state, var, val)

                    # Многошаговый rollout
                    for _ in range(min(3, self._horizon)):
                        state = graph.rollout_step_free(state)

                    # Оцениваем ожидаемый Δcompression через proxy:
                    # "насколько состояние после отличается от текущего?"
                    delta = float(np.mean([
                        abs(float(state.get(nid, 0.5)) - float(current_state.get(nid, 0.5)))
                        for nid in node_ids
                    ]))

                    # Неопределённость по этому узлу (чем выше — тем ценнее)
                    if core is not None:
                        A = core.alpha_trust_matrix()
                        idx = node_ids.index(var) if var in node_ids else -1
                        if idx >= 0:
                            uncertainty = float(1.0 - A[idx].mean().item())
                        else:
                            uncertainty = 0.5
                    else:
                        uncertainty = 0.5

                    # Ожидаемый выигрыш = delta * uncertainty
                    # (большое изменение в неизведанной области = ценно)
                    expected_gain = delta * (1.0 + uncertainty)

                    if expected_gain > best_expected:
                        best_expected = expected_gain
                        best_var = var
                        best_val = val

                except Exception:
                    continue

        if best_var is None:
            return None

        goal = {
            "variable": best_var,
            "value": best_val,
            "expected_gain": float(best_expected),
            "generated_at": n_interventions,
            "horizon": self._horizon,
        }

        self._current_goal = goal
        self._goal_age = 0
        self._goal_history.append(dict(goal))
        self.total_goals_generated += 1
        return goal

    def tick_goal(self, actual_compression: float) -> dict[str, Any] | None:
        """
        Обновляет возраст цели. Возвращает текущую цель или None если истекла.
        Записывает точность imagination (предсказали gain, получили actual).
        """
        if self._current_goal is None:
            return None

        self._goal_age += 1

        # Записываем точность imagination
        if self._goal_age == 1:
            expected = self._current_goal.get("expected_gain", 0.0)
            if expected > 0:
                accuracy = min(1.0, actual_compression / max(expected, 1e-8))
                self._imagination_accuracy.append(accuracy)

        if self._goal_age >= self._goal_ttl:
            old_goal = self._current_goal
            self._current_goal = None
            self._goal_age = 0
            return None

        return self._current_goal

    def imagination_accuracy(self) -> float:
        if not self._imagination_accuracy:
            return 0.5
        return float(np.mean(self._imagination_accuracy))

    def snapshot(self) -> dict[str, Any]:
        return {
            "current_goal": self._current_goal,
            "goal_age": self._goal_age,
            "goal_ttl": self._goal_ttl,
            "total_goals_generated": self.total_goals_generated,
            "imagination_accuracy": round(self.imagination_accuracy(), 4),
            "recent_goals": list(self._goal_history)[-5:],
        }


# ─── VariableDiscovery ────────────────────────────────────────────────────────
class VariableDiscovery:
    """
    Проактивное открытие новых переменных через EIG.

    Принцип:
      Если агент систематически ошибается в предсказании определённой
      области состояний, и GNN не может улучшиться (высокий surprise
      при низком compression gain) → там скрытая переменная.

    Алгоритм:
      1. Отслеживаем prediction error по каждому узлу отдельно
      2. Если узел X имеет высокую и СТАБИЛЬНУЮ ошибку →
         GNN не справляется, нужен медиирующий узел
      3. Вызываем NeurogenesisEngine.scan_and_grow() с целевой парой

    Отличие от rsi_structural.py:
      rsi_structural: реактивный (срабатывает при high stress в W.grad)
      VariableDiscovery: проактивный (срабатывает при стабильной ошибке)
    """

    def __init__(self):
        self._eig_threshold = _ef("RKK_INTRINSIC_DISCOVERY_EIG", 0.3)
        # per-node prediction error history
        self._node_errors: dict[str, deque[float]] = {}
        self._window = 30
        self._last_discovery_tick: int = -9999
        self._discovery_cooldown: int = 500
        self.total_discoveries: int = 0
        self._discovery_log: deque[dict[str, Any]] = deque(maxlen=20)

    def update_node_errors(
        self,
        predicted: dict[str, float],
        observed: dict[str, float],
    ) -> None:
        """Записываем ошибку по каждому узлу."""
        for nid in observed:
            if nid not in self._node_errors:
                self._node_errors[nid] = deque(maxlen=self._window)
            pred_val = float(predicted.get(nid, 0.5))
            obs_val = float(observed[nid])
            self._node_errors[nid].append(abs(pred_val - obs_val))

    def find_high_error_nodes(
        self,
        top_k: int = 3,
        min_error: float = 0.15,
    ) -> list[tuple[str, float]]:
        """
        Находим узлы с хронически высокой ошибкой.
        Это кандидаты для нейрогенеза.
        """
        candidates: list[tuple[str, float]] = []
        for nid, errors in self._node_errors.items():
            if len(errors) < self._window // 2:
                continue
            mean_err = float(np.mean(errors))
            # Стабильность ошибки (низкий std = стабильно плохо, не случайный всплеск)
            std_err = float(np.std(errors))
            stability = 1.0 / (1.0 + std_err)
            score = mean_err * stability
            if mean_err >= min_error:
                candidates.append((nid, score))

        candidates.sort(key=lambda x: -x[1])
        return candidates[:top_k]

    def maybe_trigger_neurogenesis(
        self,
        graph,
        agent,
        tick: int,
        causal_surprise: CausalSurprise,
    ) -> dict[str, Any] | None:
        """
        Проверяет нужен ли нейрогенез и запускает его.
        Возвращает событие если новый узел был создан.
        """
        if not intrinsic_enabled():
            return None
        if (tick - self._last_discovery_tick) < self._discovery_cooldown:
            return None
        if not causal_surprise.compression_is_stagnant():
            return None  # Граф ещё учится — нейрогенез преждевременен

        high_error = self.find_high_error_nodes()
        if len(high_error) < 2:
            return None

        # Пытаемся создать медиирующий узел между двумя проблемными
        from engine.rsi_structural import NeurogenesisEngine
        neuro = getattr(agent, "_neuro_engine", NeurogenesisEngine(
            min_interventions=200,
            error_threshold=self._eig_threshold,
        ))

        result = neuro.scan_and_grow(agent, tick)
        if result is None:
            return None

        self._last_discovery_tick = tick
        self.total_discoveries += 1

        event = {
            "tick": tick,
            "new_node": result.get("new_node"),
            "triggered_by": [nid for nid, _ in high_error[:2]],
            "error_scores": [round(s, 4) for _, s in high_error[:2]],
        }
        self._discovery_log.append(event)
        return event

    def snapshot(self) -> dict[str, Any]:
        high_err = self.find_high_error_nodes()
        return {
            "total_discoveries": self.total_discoveries,
            "nodes_tracked": len(self._node_errors),
            "high_error_nodes": [(n, round(s, 4)) for n, s in high_err],
            "recent_discoveries": list(self._discovery_log)[-3:],
        }


# ─── AffectiveDrive ───────────────────────────────────────────────────────────
class AffectiveDrive:
    """
    Phase 1 Embodied Cognition: modulates intrinsic reward via interoception.

    Pain (intero_stress) multiplicatively suppresses EIG:
        R = EIG * sigmoid(-w_stress * intero_stress)
    This reduces curiosity value during pain but does not block it entirely.

    Energy maintenance bonus encourages homeostasis:
        R += w_energy * (intero_energy - 0.5)
    Positive when energy > 0.5 (rewarded), negative when depleted (punished).
    """

    def __init__(self):
        self._w_stress = _ef("RKK_AFFECTIVE_W_STRESS", 4.0)
        self._w_energy = _ef("RKK_AFFECTIVE_W_ENERGY", 0.15)

    def modulate(self, eig_reward: float, intero_energy: float, intero_stress: float) -> float:
        """Apply affective gating to the raw intrinsic reward."""
        # Pain suppression: high stress reduces exploration value
        pain_gate = float(torch.sigmoid(
            torch.tensor(-self._w_stress * intero_stress)
        ).item())
        # Energy bonus: maintaining energy above 0.5 is rewarded
        energy_bonus = self._w_energy * (intero_energy - 0.5)
        return eig_reward * pain_gate + energy_bonus

    def snapshot(self) -> dict[str, Any]:
        return {
            "w_stress": self._w_stress,
            "w_energy": self._w_energy,
        }


# ─── IntrinsicObjective ───────────────────────────────────────────────────────
class IntrinsicObjective:
    """
    Единый интерфейс интринсивной цели. Заменяет все внешние rewards.

    Использование в simulation_main.py:
      self._intrinsic = IntrinsicObjective(device)

    После каждого agent.step():
      r = self._intrinsic.step(
          agent=self.agent,
          result=result,
          tick=self.tick,
          locomotion_ctrl=self._locomotion_controller,
      )

    """

    def __init__(self, device: torch.device):
        self.device = device
        self.causal_surprise = CausalSurprise()
        self.goal_imagination = GoalImagination(device)
        self.variable_discovery = VariableDiscovery()
        self.affective_drive = AffectiveDrive()

        # Когда генерируем следующую цель
        self._goal_generate_every = _ei("RKK_INTRINSIC_GOAL_EVERY", 25)
        self._last_goal_tick: int = -9999

        # История наград
        self._reward_history: deque[float] = deque(maxlen=500)
        self.total_steps: int = 0

    def step(
        self,
        agent,
        result: dict[str, Any],
        tick: int,
        locomotion_ctrl=None,
        motor_cortex=None,
    ) -> float:
        """
        Главный вызов: вычисляет интринсивную награду и применяет её.
        Возвращает скалярную награду.
        """
        if not intrinsic_enabled():
            return 0.0

        self.total_steps += 1

        # --- Извлекаем сигналы ---
        compression_delta = float(result.get("compression_delta", 0.0))
        prediction_error = float(result.get("prediction_error", 0.0))
        graph = agent.graph
        mdl_before = float(graph.mdl_size) + abs(compression_delta)
        mdl_after = float(graph.mdl_size)

        # --- Обновляем per-node ошибки ---
        cf_pred = result.get("cf_predicted") or {}
        cf_obs = result.get("cf_observed") or {}
        if cf_pred and cf_obs:
            self.variable_discovery.update_node_errors(cf_pred, cf_obs)

        # --- УРОВЕНЬ 2: AMP Reward (Adversarial Motion Prior) ---
        amp_reward = 0.0
        try:
            if getattr(agent, "_motion_discriminator", None) is None:
                from engine.mocap_loader import MotionDiscriminator
                agent._motion_discriminator = MotionDiscriminator(state_dim=8).to(self.device)
            
            # Собираем вектор состояния для AMP: 8 ключевых переменных походки
            obs = dict(agent.env.observe())
            keys = ["lhip", "rhip", "lknee", "rknee", "lankle", "rankle", "com_z", "posture_stability"]
            state_vec = []
            for k in keys:
                val = obs.get(k, obs.get(f"phys_{k}", 0.5))
                state_vec.append(float(val))
            
            state_t = torch.tensor(state_vec, dtype=torch.float32, device=self.device)
            amp_reward = agent._motion_discriminator.compute_amp_reward(state_t)
        except Exception as e:
            pass

        # --- Вычисляем интринсивную награду ---
        r = self.causal_surprise.compute(
            compression_delta=compression_delta,
            prediction_error=prediction_error,
            graph_mdl_before=mdl_before,
            graph_mdl_after=mdl_after,
            n_interventions=int(agent._total_interventions),
        )

        # Интегрируем AMP бонус (Уровень 2)
        try:
            lambda_amp = float(os.environ.get("RKK_INTRINSIC_AMP_LAMBDA", "0.3"))
        except:
            lambda_amp = 0.3
        
        # Если это самое начало, полагаемся в основном на AMP, иначе миксуем
        if agent._total_interventions < 2000:
            # Behavioral cloning bias early on
            r = r * 0.5 + amp_reward * 1.5
        else:
            r = r + lambda_amp * amp_reward

        # --- AffectiveDrive: interoceptive modulation (Phase 1) ---
        try:
            obs = dict(agent.env.observe())
            intero_energy = float(obs.get("intero_energy", 0.5))
            intero_stress = float(obs.get("intero_stress", 0.0))
            r = self.affective_drive.modulate(r, intero_energy, intero_stress)
        except Exception:
            pass

        # --- Обновляем активную цель ---
        active_goal = self.goal_imagination.tick_goal(compression_delta)

        # --- Генерируем новую цель если нужно ---
        if (tick - self._last_goal_tick) >= self._goal_generate_every:
            if active_goal is None or self.causal_surprise.compression_is_stagnant():
                try:
                    new_goal = self.goal_imagination.generate_goal(
                        graph=graph,
                        agent_env=agent.env,
                        n_interventions=int(agent._total_interventions),
                        causal_surprise=self.causal_surprise,
                    )
                    if new_goal is not None:
                        self._last_goal_tick = tick
                except Exception:
                    pass

        # --- Проактивный нейрогенез ---
        try:
            neuro_event = self.variable_discovery.maybe_trigger_neurogenesis(
                graph=graph,
                agent=agent,
                tick=tick,
                causal_surprise=self.causal_surprise,
            )
        except Exception:
            neuro_event = None

        # --- Variable Discovery: auto-discover new variable groups ---
        try:
            from engine.variable_bootstrap import get_variable_registry
            registry = get_variable_registry()
            if registry.is_bootstrap:
                # Update pressure from high-error nodes
                high_err = self.variable_discovery.find_high_error_nodes(top_k=5)
                registry.update_pressure(
                    high_error_nodes=high_err,
                    compression_stagnant=self.causal_surprise.compression_is_stagnant(),
                    tick=tick,
                )
                # Auto-discover if pressure threshold met
                new_vars = registry.auto_discover(tick)
                if new_vars:
                    # Add new variables to the agent's graph
                    for var in new_vars:
                        if var not in graph.nodes:
                            graph.set_node(var, 0.5)
                    # Rebind agent environment if needed
                    try:
                        env_obs = dict(agent.env.observe())
                        for var in new_vars:
                            if var in env_obs and var in graph.nodes:
                                graph.nodes[var] = env_obs[var]
                    except Exception:
                        pass
        except ImportError:
            pass
        except Exception:
            pass

        self._apply_intrinsic_reward(r, locomotion_ctrl, motor_cortex, agent)

        self._reward_history.append(r)
        return r

    def _apply_intrinsic_reward(
        self,
        r: float,
        locomotion_ctrl,
        motor_cortex,
        agent,
    ) -> None:
        """
        Полная замена: ТОЛЬКО интринсивная награда идёт в все learners.
        Никаких posture, symmetry, forward_bonus.
        """
        if locomotion_ctrl is not None:
            locomotion_ctrl._reward_history.append(r)
            train_fn = getattr(locomotion_ctrl, "train_cpg_from_intrinsic_history", None)
            if callable(train_fn):
                train_fn()

        if motor_cortex is not None:
            obs = {}
            try:
                obs = dict(agent.env.observe())
            except Exception:
                pass
            posture = float(obs.get("posture_stability", obs.get("phys_posture_stability", 0.5)))
            foot_l = float(obs.get("foot_contact_l", obs.get("phys_foot_contact_l", 0.5)))
            foot_r = float(obs.get("foot_contact_r", obs.get("phys_foot_contact_r", 0.5)))
            cpg_tgt = {}
            if locomotion_ctrl is not None:
                cpg_tgt = dict(getattr(locomotion_ctrl, "_last_command", {}))
            motor_cortex.push_and_train(
                nodes=dict(agent.graph.nodes),
                cpg_targets=cpg_tgt,
                reward=r,
                posture=posture,
                foot_l=foot_l,
                foot_r=foot_r,
            )

    def get_current_goal(self) -> dict[str, Any] | None:
        return self.goal_imagination._current_goal

    def recent_reward(self, window: int = 32) -> float:
        if not self._reward_history:
            return 0.0
        return float(np.mean(list(self._reward_history)[-window:]))

    def snapshot(self) -> dict[str, Any]:
        return {
            "enabled": intrinsic_enabled(),
            "total_steps": self.total_steps,
            "recent_reward": round(self.recent_reward(), 6),
            "causal_surprise": self.causal_surprise.snapshot(),
            "goal_imagination": self.goal_imagination.snapshot(),
            "variable_discovery": self.variable_discovery.snapshot(),
            "affective_drive": self.affective_drive.snapshot(),
        }


# ─── Integration patch ────────────────────────────────────────────────────────
def apply_intrinsic_patch(sim) -> bool:
    """
    Патч Simulation: заменяет внешние rewards на IntrinsicObjective.

    Применяет:
    1. IntrinsicObjective создаётся и сохраняется в sim._intrinsic
    2. Хук после agent.step() в _run_agent_or_skill_step

    Вызов: apply_intrinsic_patch(sim)  # после создания Simulation
    """
    device = getattr(sim, "device", torch.device("cpu"))
    intrinsic = IntrinsicObjective(device)
    sim._intrinsic = intrinsic
    print("[Intrinsic] IntrinsicObjective applied")

    # Патч _run_agent_or_skill_step
    original = sim._run_agent_or_skill_step

    def patched_run(engine_tick: int) -> dict:
        result = original(engine_tick)

        # После каждого шага — вычисляем интринсивную награду
        if not result.get("blocked") and not result.get("skipped"):
            try:
                # #region agent log
                _t_i0 = time.perf_counter()
                # #endregion
                r = intrinsic.step(
                    agent=sim.agent,
                    result=result,
                    tick=engine_tick,
                    locomotion_ctrl=getattr(sim, "_locomotion_controller", None),
                    motor_cortex=getattr(sim, "_motor_cortex", None),
                )
                # #region agent log
                _dbg_ivo(
                    "H5",
                    "intrinsic.patched_run",
                    "intrinsic_step",
                    {"ms": (time.perf_counter() - _t_i0) * 1000, "engine_tick": engine_tick},
                )
                # #endregion
                result["intrinsic_reward"] = round(r, 6)

                # Если нейрогенез произошёл — сообщаем в events
                disc = intrinsic.variable_discovery._discovery_log
                if disc and disc[-1].get("tick") == engine_tick:
                    ev = disc[-1]
                    sim._add_event(
                        f"🧬 IntrinsicDiscovery: {ev.get('new_node', '?')} "
                        f"← [{', '.join(ev.get('triggered_by', [])[:2])}]",
                        "#ff44cc", "phase"
                    )

                # Если новая цель сгенерирована — логируем
                goal = intrinsic.get_current_goal()
                if goal and goal.get("generated_at") == sim.agent._total_interventions:
                    sim._add_event(
                        f"🎯 IntGoal: do({goal['variable']}="
                        f"{goal['value']:.2f}) E[gain]={goal['expected_gain']:.4f}",
                        "#44ffaa", "discovery"
                    )

            except Exception as e:
                result["intrinsic_reward"] = 0.0

        return result

    sim._run_agent_or_skill_step = patched_run

    return True