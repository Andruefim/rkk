"""
learned_motor_primitives.py — Замена SkillLibrary на самообучающиеся примитивы.

Проблема (из лога):
  Путь intent_support_left → foot_contact_l реализован как
  захардкоженная последовательность в SkillLibrary:
    step_forward_L = [doIntent_support_right, doIntent_push_left, ...]

  Агент НЕ ОТКРЫЛ эту последовательность — человек ему её написал.
  Поэтому у него нет медиирующего узла — он просто выполняет скрипт.

Решение:
  1. MotorPatternDetector — обнаруживает повторяющиеся
     каузальные последовательности через GNN transitions
  2. LearnedMotorProgram — нейронный примитив (GRU → joint_targets)
     который учится из observerd successful transitions
  3. MotorPrimitiveLibrary — заменяет SkillLibrary.execute()
     на динамически создаваемые и обучаемые примитивы

Ключевое: программы создаются и называются агентом.
  НЕ: step_forward_L (человек назвал)
  ДА: primitive_A3 → потом LLM называет "это похоже на перенос веса"

RKK_MOTOR_PRIMITIVES_ENABLED=1
RKK_MOTOR_PRIM_MIN_REPEATS=5     — сколько раз паттерн повторился → примитив
RKK_MOTOR_PRIM_HIDDEN=64
RKK_MOTOR_PRIM_LR=1e-3
RKK_MOTOR_MAX_PRIMITIVES=32
"""
from __future__ import annotations

import os
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _prim_enabled() -> bool:
    return os.environ.get("RKK_MOTOR_PRIMITIVES_ENABLED", "1").strip().lower() not in (
        "0", "false", "no", "off"
    )

def _ef(key: str, d: float) -> float:
    try: return float(os.environ.get(key, str(d)))
    except ValueError: return d

def _ei(key: str, d: int) -> int:
    try: return max(1, int(os.environ.get(key, str(d))))
    except ValueError: return d


# ─── Transition Record ────────────────────────────────────────────────────────

@dataclass
class MotorTransition:
    """Одна успешная интервенция с контекстом."""
    tick: int
    intent_var: str          # что изменили
    intent_val: float        # до какого значения
    state_before: dict[str, float]  # snapshot graph nodes
    state_after: dict[str, float]
    compression_gain: float  # насколько улучшилась модель
    intrinsic_reward: float


# ─── Pattern Detector ─────────────────────────────────────────────────────────

class MotorPatternDetector:
    """
    Обнаруживает повторяющиеся последовательности интервенций.

    Принцип:
      Если агент несколько раз делает intent_A → intent_B → intent_C
      с похожим исходным состоянием и это даёт compression gain →
      эта последовательность становится кандидатом на примитив.

    Не просто "частые последовательности" — только те что
    коррелируют с intrinsic_reward > threshold.
    """

    def __init__(self, n_joints: int = 12):
        self._min_repeats = _ei("RKK_MOTOR_PRIM_MIN_REPEATS", 5)
        self._window_len = 4  # длина паттерна (интервенций)
        self._recent: deque[MotorTransition] = deque(maxlen=500)
        # sequence_hash → (count, [transitions])
        self._sequence_counts: dict[str, list[MotorTransition]] = {}
        self._detected_patterns: list[list[MotorTransition]] = []
        self.total_detected: int = 0

    def record(self, transition: MotorTransition) -> None:
        self._recent.append(transition)
        self._check_patterns()

    def _check_patterns(self) -> None:
        if len(self._recent) < self._window_len:
            return

        # Последние window_len транзиций
        window = list(self._recent)[-self._window_len:]

        # Только если все в окне имеют положительный reward
        if not all(t.intrinsic_reward > 0 for t in window):
            return

        # Хэш паттерна = последовательность intent vars
        seq_key = "|".join(t.intent_var for t in window)

        if seq_key not in self._sequence_counts:
            self._sequence_counts[seq_key] = []
        self._sequence_counts[seq_key].append(window[0])

        count = len(self._sequence_counts[seq_key])
        if count == self._min_repeats:
            self._detected_patterns.append(window)
            self.total_detected += 1

    def pop_new_patterns(self) -> list[list[MotorTransition]]:
        """Забирает вновь обнаруженные паттерны."""
        patterns = self._detected_patterns[:]
        self._detected_patterns.clear()
        return patterns

    def snapshot(self) -> dict[str, Any]:
        top = sorted(
            self._sequence_counts.items(),
            key=lambda x: len(x[1]),
            reverse=True
        )[:5]
        return {
            "total_detected": self.total_detected,
            "tracked_sequences": len(self._sequence_counts),
            "top_sequences": [(k, len(v)) for k, v in top],
        }


# ─── Learned Motor Program ────────────────────────────────────────────────────

class LearnedMotorProgram(nn.Module):
    """
    Нейронный моторный примитив.

    Вход: контекст (состояние графа) — d_state
    Выход: последовательность joint targets — [T, n_joints]

    Архитектура: context_encoder → GRU → joint_head
    Обучается из успешных MotorTransition через MSE на joint_targets.

    НЕ ЗАХАРДКОЖЕН: агент сам открывает какие суставы двигать.
    """

    def __init__(
        self,
        prim_id: str,
        d_state: int,
        n_joints: int = 12,
        hidden: int = 64,
        seq_len: int = 8,
    ):
        super().__init__()
        self.prim_id = prim_id
        self.d_state = d_state
        self.n_joints = n_joints
        self.seq_len = seq_len

        # Context encoder: graph state → hidden
        self.ctx_encoder = nn.Sequential(
            nn.Linear(d_state, hidden),
            nn.LayerNorm(hidden),
            nn.Tanh(),
        )

        # GRU генерирует последовательность
        self.gru = nn.GRU(
            input_size=n_joints,
            hidden_size=hidden,
            num_layers=1,
            batch_first=True,
        )

        # Выход: joint targets в [-1, 1]
        self.joint_head = nn.Sequential(
            nn.Linear(hidden, n_joints),
            nn.Tanh(),
        )

        # Метаданные
        self.n_uses: int = 0
        self.n_train_steps: int = 0
        self.total_reward: float = 0.0
        self.mean_compression_gain: float = 0.0
        self._reward_history: deque[float] = deque(maxlen=50)
        self._name_from_llm: str | None = None  # LLM назовёт позже

        # Оптимайзер
        lr = _ef("RKK_MOTOR_PRIM_LR", 1e-3)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, ctx_vec: torch.Tensor) -> torch.Tensor:
        """
        ctx_vec: [d_state] — текущее состояние графа
        Returns: [seq_len, n_joints] — joint targets
        """
        h = self.ctx_encoder(ctx_vec).unsqueeze(0)  # [1, hidden]

        # Начинаем с нулей
        x = torch.zeros(1, self.seq_len, self.n_joints, device=ctx_vec.device)
        out, _ = self.gru(x, h.unsqueeze(0))  # [1, seq_len, hidden]
        joints = self.joint_head(out.squeeze(0))  # [seq_len, n_joints]
        return joints

    def train_on_transition(
        self,
        ctx_vec: torch.Tensor,
        observed_joints: torch.Tensor,  # [seq_len, n_joints]
        reward: float,
    ) -> float:
        """Один шаг обучения. Возвращает loss."""
        self.train()
        self.optimizer.zero_grad()

        predicted = self.forward(ctx_vec)
        loss = F.mse_loss(predicted, observed_joints)

        # Масштабируем по reward (учимся больше от успешных переходов)
        reward_scale = max(0.1, min(2.0, 1.0 + reward))
        loss = loss * reward_scale

        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        self.optimizer.step()

        self.n_train_steps += 1
        self._reward_history.append(reward)
        return float(loss.item())

    def execute(self, ctx_vec: torch.Tensor) -> list[list[float]]:
        """Возвращает последовательность joint targets как список."""
        self.eval()
        self.n_uses += 1
        with torch.no_grad():
            joints = self.forward(ctx_vec)  # [seq_len, n_joints]
        return joints.cpu().tolist()

    def mean_reward(self) -> float:
        if not self._reward_history:
            return 0.0
        return float(np.mean(self._reward_history))

    def snapshot(self) -> dict[str, Any]:
        return {
            "id": self.prim_id,
            "name": self._name_from_llm or self.prim_id,
            "n_uses": self.n_uses,
            "n_train_steps": self.n_train_steps,
            "mean_reward": round(self.mean_reward(), 4),
            "mean_compression_gain": round(self.mean_compression_gain, 4),
        }


# ─── Motor Primitive Library ──────────────────────────────────────────────────

class MotorPrimitiveLibrary:
    """
    Заменяет SkillLibrary.

    Разница:
      SkillLibrary: {'step_forward_L': [('doIntent_support_right', ...), ...]}
      MotorPrimitiveLibrary: {
          'primitive_A3': LearnedMotorProgram(GRU, trained=True),
          ...
      }

    Новые примитивы создаются автоматически из MotorPatternDetector.
    Называются автоматически (primitive_A0, primitive_A1...).
    LLM может дать им имена через name_primitive().
    """

    def __init__(self, d_state: int, n_joints: int, device: torch.device):
        self.d_state = d_state
        self.n_joints = n_joints
        self.device = device
        self._max = _ei("RKK_MOTOR_MAX_PRIMITIVES", 32)
        self._hidden = _ei("RKK_MOTOR_PRIM_HIDDEN", 64)

        self._primitives: dict[str, LearnedMotorProgram] = {}
        self._pattern_detector = MotorPatternDetector(n_joints)
        self._counter: int = 0

        # Текущий активный примитив (если выполняется)
        self._active_prim: str | None = None
        self._active_step: int = 0
        self._active_targets: list[list[float]] | None = None

        self.total_primitives_created: int = 0
        print(f"[MotorPrimLib] Initialized (d_state={d_state}, n_joints={n_joints})")

    def record_transition(self, transition: MotorTransition) -> None:
        """Записываем интервенцию. Детектор ищет паттерны."""
        if not _prim_enabled():
            return

        self._pattern_detector.record(transition)

        # Тренируем активные примитивы на этом переходе
        for prim in self._primitives.values():
            if prim.n_uses > 0:
                self._train_on_transition(prim, transition)

        # Создаём новые примитивы из обнаруженных паттернов
        new_patterns = self._pattern_detector.pop_new_patterns()
        for pattern in new_patterns:
            self._create_primitive_from_pattern(pattern)

    def _train_on_transition(
        self,
        prim: LearnedMotorProgram,
        transition: MotorTransition,
    ) -> None:
        """Обучаем примитив на одном переходе."""
        try:
            ctx_vec = self._state_to_vec(transition.state_before)
            # Создаём приближённые joint targets из diff состояний
            joints = self._extract_joint_targets(
                transition.state_before,
                transition.state_after,
            )
            if joints is None:
                return

            joints_t = torch.tensor(
                joints, dtype=torch.float32, device=self.device
            ).unsqueeze(0).expand(prim.seq_len, -1)  # [seq_len, n_joints]

            prim.train_on_transition(ctx_vec, joints_t, transition.intrinsic_reward)
            prim.total_reward += transition.intrinsic_reward
            prim.mean_compression_gain = (
                0.9 * prim.mean_compression_gain
                + 0.1 * transition.compression_gain
            )
        except Exception:
            pass

    def _create_primitive_from_pattern(
        self,
        pattern: list[MotorTransition],
    ) -> str | None:
        """Создаёт новый LearnedMotorProgram из обнаруженного паттерна."""
        if len(self._primitives) >= self._max:
            self._prune_worst()

        prim_id = f"primitive_{chr(65 + self._counter // 26)}{self._counter % 26}"
        self._counter += 1

        prim = LearnedMotorProgram(
            prim_id=prim_id,
            d_state=self.d_state,
            n_joints=self.n_joints,
            hidden=self._hidden,
            seq_len=len(pattern),
        ).to(self.device)

        # Начальное обучение на паттерне
        for t in pattern:
            self._train_on_transition(prim, t)

        self._primitives[prim_id] = prim
        self.total_primitives_created += 1
        print(f"[MotorPrimLib] Created primitive '{prim_id}' "
              f"from pattern len={len(pattern)}")
        return prim_id

    def _prune_worst(self) -> None:
        """Удаляем наименее эффективный примитив."""
        if not self._primitives:
            return
        worst = min(self._primitives.items(), key=lambda x: x[1].mean_reward())
        print(f"[MotorPrimLib] Pruning '{worst[0]}' (mean_reward={worst[1].mean_reward():.4f})")
        del self._primitives[worst[0]]

    def select_best(self, ctx_vec: torch.Tensor) -> str | None:
        """
        Выбирает лучший примитив для текущего контекста.
        Пока: выбираем по максимальному mean_reward.
        TODO: научиться выбирать по ctx_vec similarity.
        """
        if not self._primitives:
            return None
        best = max(
            self._primitives.items(),
            key=lambda x: x[1].mean_reward(),
        )
        return best[0] if best[1].mean_reward() > 0 else None

    def execute_step(
        self,
        ctx_vec: torch.Tensor,
        prim_id: str | None = None,
    ) -> list[float] | None:
        """
        Выполняет один шаг выбранного примитива.
        Возвращает joint targets для текущего шага.
        """
        if not _prim_enabled() or not self._primitives:
            return None

        # Если не указан — выбираем лучший
        if prim_id is None:
            prim_id = self.select_best(ctx_vec)
        if prim_id is None:
            return None

        prim = self._primitives.get(prim_id)
        if prim is None:
            return None

        # Новый примитив или следующий шаг текущего
        if self._active_prim != prim_id or self._active_targets is None:
            self._active_targets = prim.execute(ctx_vec)
            self._active_prim = prim_id
            self._active_step = 0

        if self._active_step >= len(self._active_targets):
            self._active_targets = None
            self._active_prim = None
            return None

        targets = self._active_targets[self._active_step]
        self._active_step += 1
        return targets

    def name_primitive(self, prim_id: str, name: str) -> None:
        """LLM называет примитив понятным именем."""
        prim = self._primitives.get(prim_id)
        if prim:
            prim._name_from_llm = name
            print(f"[MotorPrimLib] '{prim_id}' named '{name}' by LLM")

    def _state_to_vec(self, state: dict[str, float]) -> torch.Tensor:
        vals = list(state.values())[:self.d_state]
        if len(vals) < self.d_state:
            vals += [0.5] * (self.d_state - len(vals))
        return torch.tensor(vals, dtype=torch.float32, device=self.device)

    def _extract_joint_targets(
        self,
        state_before: dict[str, float],
        state_after: dict[str, float],
    ) -> list[float] | None:
        """Извлекает joint positions из diff состояний."""
        joint_keys = [k for k in state_after if "joint" in k or "q_" in k or "motor" in k]
        if not joint_keys:
            return None
        targets = [float(state_after.get(k, 0.0)) for k in joint_keys[:self.n_joints]]
        while len(targets) < self.n_joints:
            targets.append(0.0)
        return targets[:self.n_joints]

    def snapshot(self) -> dict[str, Any]:
        return {
            "total_created": self.total_primitives_created,
            "active_count": len(self._primitives),
            "active_primitive": self._active_prim,
            "pattern_detector": self._pattern_detector.snapshot(),
            "primitives": [p.snapshot() for p in self._primitives.values()],
        }


# ─── Patch ────────────────────────────────────────────────────────────────────

def apply_motor_primitives_patch(sim) -> bool:
    """
    Патч: заменяет SkillLibrary на MotorPrimitiveLibrary.

    - SkillLibrary.execute() → motor_prim_lib.execute_step()
    - После каждого agent.step() → record_transition()
    - LLM annotations → name_primitive()

    Вызов: apply_motor_primitives_patch(sim)
    """
    if not _prim_enabled():
        print("[MotorPrimitives] Disabled (RKK_MOTOR_PRIMITIVES_ENABLED=0)")
        return False

    device = getattr(sim, "device", torch.device("cpu"))
    graph = sim.agent.graph
    d_state = len(graph._node_ids)
    n_joints = 12  # типичное для PyBullet humanoid

    lib = MotorPrimitiveLibrary(d_state=d_state, n_joints=n_joints, device=device)
    sim._motor_prim_lib = lib

    # Патч _run_agent_or_skill_step для записи транзиций
    original = sim._run_agent_or_skill_step

    def patched(engine_tick: int) -> dict:
        state_before = dict(sim.agent.graph.nodes)
        result = original(engine_tick)
        state_after = dict(sim.agent.graph.nodes)

        transition = MotorTransition(
            tick=engine_tick,
            intent_var=str(result.get("action_var", "")),
            intent_val=float(result.get("action_val", 0.5)),
            state_before=state_before,
            state_after=state_after,
            compression_gain=float(result.get("compression_delta", 0.0)),
            intrinsic_reward=float(result.get("intrinsic_reward", 0.0)),
        )
        lib.record_transition(transition)

        # Сообщаем о новых примитивах
        new_count = lib.total_primitives_created
        if hasattr(patched, "_last_count") and new_count > patched._last_count:
            latest = list(lib._primitives.values())[-1]
            sim._add_event(
                f"⚡ MotorPrim: discovered '{latest.prim_id}'",
                "#aaffaa", "skill"
            )
        patched._last_count = new_count

        return result

    patched._last_count = 0
    sim._run_agent_or_skill_step = patched

    # Переключаем SkillLibrary.execute на MotorPrimLib если навык выполняется
    skill_lib = getattr(sim, "_skill_library", None)
    if skill_lib is not None:
        original_execute = skill_lib.execute

        def patched_execute(skill_name: str, agent_env) -> bool:
            # Если это hardcoded навык — пробуем learned вместо
            ctx = lib._state_to_vec(dict(sim.agent.graph.nodes))
            targets = lib.execute_step(ctx)
            if targets is not None:
                # Применяем learned joint targets вместо hardcoded
                try:
                    agent_env.set_joint_targets(targets)
                    return True
                except Exception:
                    pass
            # Fallback на hardcoded
            return original_execute(skill_name, agent_env)

        skill_lib.execute = patched_execute
        print("[MotorPrimitives] Hooked into SkillLibrary.execute")

    print(f"[MotorPrimitives] Applied. d_state={d_state}, n_joints={n_joints}")
    return True
