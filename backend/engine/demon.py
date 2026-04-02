"""
demon_v2.py — Adversarial Demon v2.

Улучшения:
  1. Targeted атака на одного агента пока Byzantine не вылечит его
     (проверяем через deviance threshold)
  2. PPO-lite policy: reward = prediction_error * (1 - action_complexity)
  3. Режим "Siege": когда Demon добивается успеха — усиливает атаку
  4. Anti-Byzantine: Demon пытается взломать consensus,
     внося скоординированный шум в несколько агентов
  5. Термодинамический штраф L0 (как раньше) — не даём тривиальный шум

Demon теперь настоящий adversarial agent с памятью и стратегией.
"""
from __future__ import annotations

import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
from dataclasses import dataclass, field


# ─── Режимы атаки ─────────────────────────────────────────────────────────────
class AttackMode:
    PROBE      = "probe"       # исследуем агентов (малая сила)
    TARGETED   = "targeted"    # концентрируемся на одном слабом
    SIEGE      = "siege"       # продолжаем атаковать успешного
    ANTI_BYZ   = "anti_byz"   # пытаемся взломать консенсус


@dataclass
class DemonMemory:
    """Demon запоминает что работает против каждого агента."""
    agent_id:        int
    successful_attacks: int = 0
    failed_attacks:     int = 0
    best_phi_drop:      float = 0.0    # максимальное снижение Φ
    last_attack_tick:   int = 0
    vulnerable_edges:   list[str] = field(default_factory=list)


# ─── PPO-lite буфер ───────────────────────────────────────────────────────────
@dataclass
class DemonExperience:
    state:  list[float]   # [phi_target, alpha_target, h_W, demon_energy, phi_others_mean]
    action: list[float]   # [target_idx_norm, strength, variable_norm]
    reward: float         # prediction_error * (1 - complexity) - phi_recovery_penalty


class DemonPolicyNet(nn.Module):
    """
    Нейросетевая политика Демона.
    Input:  [phi_0, phi_1, phi_2, alpha_0, alpha_1, alpha_2,
             h_W_0, h_W_1, h_W_2, demon_energy, byz_round_norm, tick_norm]
    Output: [target_logits(3), strength, variable_idx_norm]
    """
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(12, 64), nn.GELU(),
            nn.Linear(64, 32), nn.GELU(),
        )
        self.target_head   = nn.Linear(32, 3)    # logits для выбора цели
        self.strength_head = nn.Sequential(nn.Linear(32, 1), nn.Sigmoid())
        self.var_head      = nn.Sequential(nn.Linear(32, 1), nn.Sigmoid())

    def forward(self, x: torch.Tensor):
        h = self.backbone(x)
        return (
            self.target_head(h),
            self.strength_head(h).squeeze(-1),
            self.var_head(h).squeeze(-1),
        )


class AdversarialDemon:
    """
    Adversarial Demon v2 — стратегический, с памятью и PPO-lite.
    """

    ENERGY_MAX       = 1.0
    ENERGY_REGEN     = 0.003
    SIEGE_THRESHOLD  = 2          # N успешных атак → siege mode
    ANTI_BYZ_PERIOD  = 200        # каждые N тиков пробуем anti-byzantine

    def __init__(self, n_agents: int, device: torch.device):
        self.n_agents = n_agents
        self.device   = device

        self.energy    = self.ENERGY_MAX
        self.cooldown  = 0
        self.tick      = 0
        self.mode      = AttackMode.PROBE
        self.target    = 0

        self.memory: list[DemonMemory] = [
            DemonMemory(agent_id=i) for i in range(n_agents)
        ]

        self.policy = DemonPolicyNet().to(device)
        self.optim  = torch.optim.Adam(self.policy.parameters(), lr=5e-4)

        self._buffer: deque[DemonExperience] = deque(maxlen=256)
        self._last_action: dict | None = None
        self._last_phi_before: list[float] = [0.0] * n_agents
        self._last_action_complexity = 0.0

        # История успехов для статистики
        self.attack_history: deque[dict] = deque(maxlen=50)

    # ── Один шаг Демона ───────────────────────────────────────────────────────
    def step(self, agent_snapshots: list[dict], env_entropy: float) -> dict | None:
        self.tick += 1

        if self.cooldown > 0:
            self.cooldown -= 1
            self.energy = min(self.ENERGY_MAX, self.energy + self.ENERGY_REGEN)

            # Обучаемся пока ждём cooldown
            if len(self._buffer) >= 16 and self.tick % 10 == 0:
                self._train_step()
            return None

        if self.energy < 0.12:
            return None

        # Запоминаем Φ до атаки
        self._last_phi_before = [s.get("phi", 0.1) for s in agent_snapshots]

        # Выбираем режим атаки
        self._update_mode()

        # Строим state для политики
        state = self._build_state(agent_snapshots)
        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            target_logits, strength, var_norm = self.policy(state_t)

        # Выбираем цель
        if self.mode == AttackMode.TARGETED or self.mode == AttackMode.SIEGE:
            # Принудительно атакуем запомненную цель
            target_idx = self.target
        elif self.mode == AttackMode.ANTI_BYZ:
            # Атакуем агента с наименьшим Φ (самый уязвимый для консенсуса)
            phis = [s.get("phi", 0.1) for s in agent_snapshots]
            target_idx = int(np.argmin(phis))
        else:
            # Политика выбирает
            probs = F.softmax(target_logits.squeeze(0), dim=-1)
            target_idx = int(torch.multinomial(probs, 1).item())

        strength_val = float(strength.item()) * 0.5 + 0.1   # [0.1, 0.6]
        var_norm_val = float(var_norm.item())

        # Термодинамический штраф L0
        intervention = torch.tensor([strength_val, var_norm_val], device=self.device)
        l0_norm      = (intervention.abs() > 0.05).float().sum().item()
        complexity   = 0.3 * l0_norm
        energy_cost  = 0.12 + complexity * 0.08

        self._last_action_complexity = complexity

        if self.energy < energy_cost:
            return None

        self.energy   -= energy_cost
        self.cooldown  = self._cooldown_for_mode()
        self.target    = target_idx

        action = {
            "target_agent":      target_idx,
            "strength":          strength_val,
            "var_idx_raw":       var_norm_val,
            "action_complexity": complexity,
            "energy_spent":      energy_cost,
            "mode":              self.mode,
        }
        self._last_action = action
        return action

    def _cooldown_for_mode(self) -> int:
        base = {
            AttackMode.PROBE:    60,
            AttackMode.TARGETED: 45,
            AttackMode.SIEGE:    30,    # siege = быстрые повторные атаки
            AttackMode.ANTI_BYZ: 80,
        }
        return base.get(self.mode, 60) + np.random.randint(0, 20)

    def _update_mode(self):
        """Обновляем режим на основе памяти."""
        # Проверяем anti-byzantine
        if self.tick % self.ANTI_BYZ_PERIOD == 0:
            self.mode = AttackMode.ANTI_BYZ
            return

        # Ищем агента в siege
        for mem in self.memory:
            if mem.successful_attacks >= self.SIEGE_THRESHOLD:
                self.mode = AttackMode.SIEGE
                self.target = mem.agent_id
                return

        # Атакуем самого уязвимого (при ничьей — случайный выбор, не всегда id=0)
        key_fn = lambda m: m.best_phi_drop - m.failed_attacks * 0.1
        mv      = min(key_fn(m) for m in self.memory)
        weakest = random.choice([m for m in self.memory if key_fn(m) == mv])
        self.mode   = AttackMode.TARGETED
        self.target = weakest.agent_id

    def _build_state(self, snapshots: list[dict]) -> list[float]:
        phis    = [s.get("phi", 0.1) for s in snapshots]
        alphas  = [s.get("alpha_mean", 0.05) for s in snapshots]
        h_Ws    = [min(s.get("h_W", 0.0), 5.0) / 5.0 for s in snapshots]
        return [
            *phis, *alphas, *h_Ws,
            self.energy,
            min(self.tick / 1000.0, 1.0),
            0.0,   # byz_round_norm (заглушка)
        ]

    # ── Обучение после атаки ──────────────────────────────────────────────────
    def learn(self, prediction_error: float, action_complexity: float, agent_snapshots: list[dict] | None = None):
        """Записываем опыт и обучаем политику."""
        if self._last_action is None:
            return

        tid = self._last_action["target_agent"]

        # Вычисляем reward
        phi_after  = (agent_snapshots or [{}])[tid].get("phi", 0.1) if agent_snapshots else 0.1
        phi_before = self._last_phi_before[tid]
        phi_drop   = max(0, phi_before - phi_after)

        # Demon reward: высокая ошибка предсказания + снижение Φ у цели
        reward = prediction_error * 2.0 + phi_drop * 5.0 - action_complexity * 0.3

        # Обновляем память
        mem = self.memory[tid]
        if phi_drop > 0.01:
            mem.successful_attacks += 1
            mem.best_phi_drop = max(mem.best_phi_drop, phi_drop)
        else:
            mem.failed_attacks += 1
            # Сбрасываем siege если много промахов
            if mem.failed_attacks > 5:
                mem.successful_attacks = max(0, mem.successful_attacks - 2)

        mem.last_attack_tick = self.tick

        # Запись в буфер
        state = self._build_state(agent_snapshots or [{} for _ in range(self.n_agents)])
        self._buffer.append(DemonExperience(
            state=state,
            action=[
                self._last_action["target_agent"] / max(self.n_agents - 1, 1),
                self._last_action["strength"],
                self._last_action["var_idx_raw"],
            ],
            reward=reward,
        ))

        # Сохраняем в историю
        self.attack_history.append({
            "tick":      self.tick,
            "target":    tid,
            "mode":      self.mode,
            "phi_drop":  round(phi_drop, 4),
            "reward":    round(reward, 4),
        })

        self._last_action = None

        # Обучаемся каждые 16 опытов
        if len(self._buffer) >= 32 and len(self._buffer) % 16 == 0:
            self._train_step()

    def _train_step(self):
        """PPO-lite: обновляем политику через REINFORCE с baseline."""
        batch = list(self._buffer)[-32:]
        if not batch:
            return

        states  = torch.tensor([e.state  for e in batch], dtype=torch.float32, device=self.device)
        rewards = torch.tensor([e.reward for e in batch], dtype=torch.float32, device=self.device)

        # Baseline
        baseline = rewards.mean()
        advantages = rewards - baseline

        self.optim.zero_grad()
        target_logits, strength, var_norm = self.policy(states)

        # Policy gradient на target selection
        log_probs = F.log_softmax(target_logits, dim=-1)
        actions_t = torch.tensor(
            [int(e.action[0] * max(self.n_agents - 1, 1)) for e in batch],
            device=self.device
        )
        selected_log_probs = log_probs.gather(1, actions_t.unsqueeze(1)).squeeze(1)

        loss = -(selected_log_probs * advantages.detach()).mean()

        # Entropy bonus: поощряем разнообразие атак
        entropy = -(F.softmax(target_logits, dim=-1) * log_probs).sum(dim=-1).mean()
        loss = loss - 0.01 * entropy

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.optim.step()

    # ── Stats ─────────────────────────────────────────────────────────────────
    @property
    def snapshot(self) -> dict:
        total_attacks  = sum(m.successful_attacks + m.failed_attacks for m in self.memory)
        total_success  = sum(m.successful_attacks for m in self.memory)
        success_rate   = total_success / max(total_attacks, 1)
        recent_rewards = [e["reward"] for e in self.attack_history][-10:]

        return {
            "energy":                self.energy,
            "cooldown":              self.cooldown,
            "last_target":           self.target,
            "last_action_complexity": round(self._last_action_complexity, 3),
            "mode":                  self.mode,
            "success_rate":          round(success_rate, 3),
            "mean_recent_reward":    round(float(np.mean(recent_rewards)) if recent_rewards else 0, 3),
            "memory": [
                {
                    "agent":    m.agent_id,
                    "success":  m.successful_attacks,
                    "fail":     m.failed_attacks,
                    "phi_drop": round(m.best_phi_drop, 4),
                }
                for m in self.memory
            ],
        }