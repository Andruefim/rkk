"""
motor_cortex.py — Специализированные моторные субграфы (Phase D: Motor Cortex RSI).

Архитектура «пирамиды управления»:
  L4 (High): Causal GNN (goals, objects, self_*)
  L3 (Mid):  MotorCortexLibrary (learned motor programs)
  L2 (Low):  CPG (rhythmic oscillators — scaffold, fades out)
  L1 (Phys): PyBullet joints

Каждый MotorProgram — маленький MLP (d_in≈8, hidden=32, d_out=6):
  WalkProgram:    (intent_stride, posture, foot_l, foot_r, com_x, com_z, gait_l, gait_r) → leg joints
  BalanceProgram: (torso_roll, torso_pitch, support_bias, com_z, com_x) → corrective joints
  RecoveryProgram:(com_z, posture, foot_l, foot_r) → recovery joint sequence

CPG Annealing:
  cpg_weight ∈ [0, 1]. Final joint = cpg_weight * CPG_out + (1-cpg_weight) * cortex_out
  Снижается плавно по мере роста posture_stability и foot_contact.
  Минимум RKK_CPG_MIN_WEIGHT (дефолт 0.08) — CPG остаётся слабым scaffold.

RSI Integration:
  RSIController вызывает cortex.maybe_spawn_program() при детектировании плато.
  Каждый новый program добавляет абстрактные узлы в основной GNN:
    walk_drive_l, walk_drive_r, balance_signal, recovery_signal
  Это отвязывает высокоуровневый GNN от нейронного шума суставов.
"""
from __future__ import annotations

import os
import math
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Helpers ────────────────────────────────────────────────────────────────────
def _env_float(key: str, default: float) -> float:
    try:
        return float(os.environ.get(key, str(default)))
    except ValueError:
        return default


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.environ.get(key, str(default)))
    except ValueError:
        return default


def _env_bool(key: str, default: bool = False) -> bool:
    v = os.environ.get(key, "1" if default else "0").strip().lower()
    return v in ("1", "true", "yes", "on")


# ── Tiny MLP for motor programs ─────────────────────────────────────────────
class MotorMLP(nn.Module):
    def __init__(self, d_in: int, d_out: int, hidden: int = 32, device: torch.device | None = None):
        super().__init__()
        dev = device or torch.device("cpu")
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, d_out),
            nn.Sigmoid(),  # output in [0,1] — normalized joint space
        ).to(dev)
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.3)
                nn.init.zeros_(m.bias)
        self.to(dev)
        self.device = dev

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def predict(self, x_np: list[float]) -> list[float]:
        with torch.no_grad():
            t = torch.tensor(x_np, dtype=torch.float32, device=self.device).unsqueeze(0)
            return self.net(t).squeeze(0).cpu().tolist()


# ── Motor Program ─────────────────────────────────────────────────────────────
@dataclass
class MotorProgram:
    """Один специализированный моторный модуль."""
    name: str
    input_keys: list[str]   # ключи из graph.nodes (или obs)
    output_keys: list[str]  # имена суставов
    net: MotorMLP
    optim: torch.optim.Optimizer
    device: torch.device

    # Статистика
    uses: int = 0
    train_steps: int = 0
    mean_reward: float = 0.0
    active: bool = True

    # Буфер опыта (state, cpg_teacher_output)
    _buffer: deque = field(default_factory=lambda: deque(maxlen=256))
    _reward_history: deque = field(default_factory=lambda: deque(maxlen=64))

    def infer(self, nodes: dict[str, float]) -> dict[str, float]:
        """Inference: graph nodes → normalized joint targets."""
        x = [float(nodes.get(k, 0.5)) for k in self.input_keys]
        y = self.net.predict(x)
        return {k: float(np.clip(v, 0.05, 0.95)) for k, v in zip(self.output_keys, y)}

    def push_experience(
        self,
        nodes: dict[str, float],
        cpg_targets: dict[str, float],
        reward: float,
    ) -> None:
        """Push (state, teacher_target, reward) into buffer."""
        x = [float(nodes.get(k, 0.5)) for k in self.input_keys]
        y = [float(cpg_targets.get(k, 0.5)) for k in self.output_keys]
        self._buffer.append((x, y, float(reward)))
        self._reward_history.append(float(reward))

    def train_step(self, batch_size: int = 32) -> float | None:
        """Imitation + reward-weighted training."""
        if len(self._buffer) < max(8, batch_size // 2):
            return None
        batch = list(self._buffer)
        if len(batch) > batch_size:
            idx = np.random.choice(len(batch), batch_size, replace=False)
            batch = [batch[i] for i in idx]

        xs = torch.tensor([b[0] for b in batch], dtype=torch.float32, device=self.device)
        ys = torch.tensor([b[1] for b in batch], dtype=torch.float32, device=self.device)
        ws = torch.tensor(
            [max(0.01, b[2] + 1.0) for b in batch],
            dtype=torch.float32, device=self.device,
        )
        ws = ws / (ws.sum() + 1e-8)

        self.optim.zero_grad()
        pred = self.net(xs)
        loss = (F.mse_loss(pred, ys, reduction="none").mean(dim=1) * ws).sum()
        loss.backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)
        self.optim.step()
        self.train_steps += 1
        return float(loss.item())

    @property
    def performance(self) -> float:
        if not self._reward_history:
            return 0.0
        return float(np.mean(list(self._reward_history)[-32:]))

    def snapshot(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "uses": self.uses,
            "train_steps": self.train_steps,
            "mean_reward": round(self.mean_reward, 4),
            "performance": round(self.performance, 4),
            "active": self.active,
            "buffer_size": len(self._buffer),
        }


# ── Motor Cortex Library ──────────────────────────────────────────────────────
class MotorCortexLibrary:
    """
    Библиотека специализированных моторных программ + CPG annealing.

    Главный цикл в simulation._maybe_apply_cpg_locomotion:
      1. CPG генерирует leg_targets
      2. cortex.infer() генерирует cortex_targets
      3. final = lerp(cpg_targets, cortex_targets, 1 - cpg_weight)
      4. cortex.push_experience() → обучение подражанием CPG
      5. cortex.anneal() → cpg_weight снижается по мере роста качества

    Новые программы создаются через maybe_spawn_program() при RSI-плато.
    """

    # Описания встроенных программ
    _PROGRAM_SPECS = {
        "walk": {
            "input_keys": [
                "intent_stride", "intent_support_left", "intent_support_right",
                "intent_torso_forward", "posture_stability", "foot_contact_l",
                "foot_contact_r", "gait_phase_l",
            ],
            "output_keys": ["lhip", "rhip", "lknee", "rknee", "lankle", "rankle"],
            "hidden": 48,
        },
        "balance": {
            "input_keys": [
                "com_z", "torso_roll", "torso_pitch", "support_bias",
                "posture_stability", "foot_contact_l", "foot_contact_r", "com_x",
            ],
            "output_keys": ["lhip", "rhip", "lankle", "rankle", "spine_pitch", "spine_yaw"],
            "hidden": 32,
        },
        "recovery": {
            "input_keys": [
                "com_z", "posture_stability", "foot_contact_l", "foot_contact_r",
                "intent_stop_recover", "torso_roll", "torso_pitch", "intent_stride",
            ],
            "output_keys": ["lhip", "rhip", "lknee", "rknee", "lshoulder", "rshoulder"],
            "hidden": 32,
        },
    }

    def __init__(self, device: torch.device):
        self.device = device
        self.programs: dict[str, MotorProgram] = {}
        self._lock = threading.Lock()

        # CPG annealing state
        self.cpg_weight: float = 1.0          # starts at full CPG
        self._annealing_enabled = False        # activated after N ticks
        self._posture_ema: float = 0.0
        self._contact_ema: float = 0.0
        self._quality_ema: float = 0.0        # combined quality signal
        self._annealing_ticks: int = 0
        self._train_every: int = 8             # train every N uses
        self._total_uses: int = 0

        # Abstract node names injected into main GNN
        self.abstract_nodes: dict[str, float] = {}

        # Performance tracking for RSI
        self._perf_history: deque[float] = deque(maxlen=200)
        self._spawn_history: list[str] = []

        print(f"[MotorCortex] Initialized on {device}")

    # ── Program management ─────────────────────────────────────────────────────
    def ensure_program(self, name: str) -> MotorProgram:
        """Создаём программу если не существует."""
        with self._lock:
            if name in self.programs:
                return self.programs[name]
            spec = self._PROGRAM_SPECS.get(name)
            if spec is None:
                raise ValueError(f"Unknown motor program: {name!r}")
            d_in = len(spec["input_keys"])
            d_out = len(spec["output_keys"])
            hidden = spec.get("hidden", 32)
            net = MotorMLP(d_in, d_out, hidden, self.device)
            optim = torch.optim.Adam(net.parameters(), lr=2e-4, weight_decay=1e-5)
            prog = MotorProgram(
                name=name,
                input_keys=list(spec["input_keys"]),
                output_keys=list(spec["output_keys"]),
                net=net,
                optim=optim,
                device=self.device,
            )
            self.programs[name] = prog
            self._spawn_history.append(name)
            print(f"[MotorCortex] Spawned program: {name} (d_in={d_in}, d_out={d_out}, h={hidden})")
            return prog

    def maybe_spawn_program(self, name: str, reason: str = "") -> bool:
        """RSI hook: создаём программу при плато."""
        if name in self.programs:
            return False
        if name not in self._PROGRAM_SPECS:
            return False
        self.ensure_program(name)
        print(f"[MotorCortex] RSI spawned '{name}': {reason}")
        return True

    # ── Inference ─────────────────────────────────────────────────────────────
    def infer(
        self,
        nodes: dict[str, float],
        active_programs: list[str] | None = None,
    ) -> dict[str, float]:
        """
        Все активные программы генерируют цели; конфликты разрешаются взвешенным средним.
        """
        progs = active_programs or list(self.programs.keys())
        targets: dict[str, list[float]] = {}

        with self._lock:
            for name in progs:
                prog = self.programs.get(name)
                if prog is None or not prog.active:
                    continue
                try:
                    out = prog.infer(nodes)
                    for k, v in out.items():
                        targets.setdefault(k, []).append(v)
                except Exception as e:
                    print(f"[MotorCortex] infer error ({name}): {e}")

        # Simple average where multiple programs share an output
        return {k: float(np.mean(vs)) for k, vs in targets.items()}

    # ── Training ───────────────────────────────────────────────────────────────
    def push_and_train(
        self,
        nodes: dict[str, float],
        cpg_targets: dict[str, float],
        reward: float,
        posture: float,
        foot_l: float,
        foot_r: float,
    ) -> dict[str, float | None]:
        """
        Push experience to all programs; train periodically.
        Returns dict of train losses.
        """
        self._total_uses += 1
        losses: dict[str, float | None] = {}

        with self._lock:
            for name, prog in self.programs.items():
                if not prog.active:
                    continue
                prog.uses += 1
                prog.push_experience(nodes, cpg_targets, reward)
                if prog.uses % self._train_every == 0:
                    loss = prog.train_step()
                    losses[name] = loss
                    if loss is not None:
                        prog.mean_reward = 0.95 * prog.mean_reward + 0.05 * reward

        # Update abstract nodes
        self._update_abstract_nodes(nodes, posture, foot_l, foot_r)

        return losses

    # ── CPG Annealing ─────────────────────────────────────────────────────────
    def enable_annealing(self) -> None:
        """Вызывается из simulation при достижении достаточной стабильности."""
        if not self._annealing_enabled:
            self._annealing_enabled = True
            print(f"[MotorCortex] CPG annealing ENABLED (cpg_weight={self.cpg_weight:.3f})")

    def anneal_step(
        self,
        posture: float,
        foot_l: float,
        foot_r: float,
        fallen: bool,
        tick: int,
    ) -> float:
        """
        Один шаг анилинга. Возвращает текущий cpg_weight.

        Логика:
        - quality = posture * 0.5 + min(foot_l, foot_r) * 0.5
        - EMA quality обновляется каждый тик
        - cpg_weight снижается только если quality_ema > threshold И не упал
        - При падении — частичное восстановление cpg_weight
        """
        if fallen:
            # Partial CPG restoration on fall
            restore = _env_float("RKK_CPG_FALL_RESTORE", 0.35)
            self.cpg_weight = min(1.0, self.cpg_weight + restore)
            self._quality_ema *= 0.5  # reset quality estimate
            return self.cpg_weight

        ema_alpha = _env_float("RKK_CPG_ANNEAL_EMA", 0.02)
        quality = float(np.clip(posture * 0.5 + min(foot_l, foot_r) * 0.5, 0.0, 1.0))
        self._quality_ema = (1.0 - ema_alpha) * self._quality_ema + ema_alpha * quality
        self._posture_ema = (1.0 - ema_alpha) * self._posture_ema + ema_alpha * posture
        self._contact_ema = (1.0 - ema_alpha) * self._contact_ema + ema_alpha * min(foot_l, foot_r)

        if not self._annealing_enabled:
            return self.cpg_weight

        threshold = _env_float("RKK_CPG_ANNEAL_THRESHOLD", 0.58)
        rate = _env_float("RKK_CPG_ANNEAL_RATE", 5e-5)
        min_w = _env_float("RKK_CPG_MIN_WEIGHT", 0.08)

        if self._quality_ema > threshold and len(self.programs) > 0:
            # Check that cortex programs have enough training
            min_steps = _env_int("RKK_CPG_ANNEAL_MIN_TRAIN", 200)
            ready = any(p.train_steps >= min_steps for p in self.programs.values())
            if ready:
                self.cpg_weight = max(min_w, self.cpg_weight - rate)

        self._perf_history.append(quality)
        self._annealing_ticks += 1
        return self.cpg_weight

    def blend_targets(
        self,
        cpg_targets: dict[str, float],
        cortex_targets: dict[str, float],
    ) -> dict[str, float]:
        """
        Смешиваем CPG и cortex с текущим cpg_weight.
        cpg_weight=1.0 → только CPG; cpg_weight=0.0 → только cortex.
        """
        w = self.cpg_weight
        result: dict[str, float] = {}
        all_keys = set(cpg_targets) | set(cortex_targets)
        for k in all_keys:
            cv = cpg_targets.get(k, 0.5)
            xv = cortex_targets.get(k, cv)  # fallback to CPG if cortex doesn't have key
            result[k] = float(np.clip(w * cv + (1.0 - w) * xv, 0.05, 0.95))
        return result

    # ── Abstract nodes ─────────────────────────────────────────────────────────
    def _update_abstract_nodes(
        self,
        nodes: dict[str, float],
        posture: float,
        foot_l: float,
        foot_r: float,
    ) -> None:
        """
        Обновляем абстрактные узлы — сигналы для главного GNN.
        Эти узлы позволяют высокоуровневому графу планировать
        без прямого управления суставами.
        """
        self.abstract_nodes = {
            "mc_walk_drive": float(np.clip(self._quality_ema, 0.05, 0.95)),
            "mc_balance_signal": float(np.clip(posture, 0.05, 0.95)),
            "mc_cpg_weight": float(np.clip(self.cpg_weight, 0.05, 0.95)),
            "mc_cortex_ready": float(
                np.clip(
                    sum(p.train_steps for p in self.programs.values()) / max(1, 100.0 * len(self.programs)),
                    0.05, 0.95,
                )
            ),
        }

    def inject_abstract_nodes_into_graph(self, graph) -> int:
        """
        Добавляем mc_* абстрактные узлы в главный GNN.
        Вызывается один раз при создании первой программы.
        """
        added = 0
        for node_name, val in self.abstract_nodes.items():
            if node_name in graph.nodes:
                graph.nodes[node_name] = float(val)
            else:
                try:
                    graph.set_node(node_name, float(val))
                    added += 1
                except Exception as e:
                    print(f"[MotorCortex] inject node error: {e}")
        return added

    def sync_abstract_nodes_to_graph(self, graph) -> None:
        """Синхронизируем значения mc_* в граф каждый тик."""
        for node_name, val in self.abstract_nodes.items():
            if node_name in graph.nodes:
                graph.nodes[node_name] = float(val)

    # ── RSI integration ────────────────────────────────────────────────────────
    def rsi_check_and_spawn(
        self,
        tick: int,
        posture_mean: float,
        loco_reward_mean: float,
        fallen_rate: float,
    ) -> list[str]:
        """
        RSI: проверяем нужно ли создать новые программы.
        Возвращает список имён новых программ.
        """
        spawned: list[str] = []
        min_ticks = _env_int("RKK_MC_RSI_MIN_TICKS", 1500)
        if tick < min_ticks:
            return spawned

        # Phase 1: ensure walk program exists
        if "walk" not in self.programs and posture_mean > 0.30:
            if self.maybe_spawn_program("walk", f"posture={posture_mean:.3f}"):
                spawned.append("walk")

        # Phase 2: balance program when walk exists but instability persists
        if (
            "walk" in self.programs
            and "balance" not in self.programs
            and posture_mean > 0.45
            and fallen_rate > 0.05
        ):
            if self.maybe_spawn_program("balance", f"fallen_rate={fallen_rate:.3f}"):
                spawned.append("balance")

        # Phase 3: recovery program when balance exists
        if (
            "balance" in self.programs
            and "recovery" not in self.programs
            and fallen_rate > 0.02
        ):
            if self.maybe_spawn_program("recovery", f"loco_r={loco_reward_mean:.3f}"):
                spawned.append("recovery")

        # Enable annealing once walk program has enough training
        if not self._annealing_enabled and "walk" in self.programs:
            walk_prog = self.programs["walk"]
            annealing_start = _env_int("RKK_CPG_ANNEAL_START_STEPS", 300)
            if walk_prog.train_steps >= annealing_start:
                self.enable_annealing()

        return spawned

    # ── Snapshot ───────────────────────────────────────────────────────────────
    def snapshot(self) -> dict[str, Any]:
        return {
            "cpg_weight": round(self.cpg_weight, 4),
            "annealing_enabled": self._annealing_enabled,
            "quality_ema": round(self._quality_ema, 4),
            "posture_ema": round(self._posture_ema, 4),
            "total_uses": self._total_uses,
            "annealing_ticks": self._annealing_ticks,
            "n_programs": len(self.programs),
            "spawn_history": list(self._spawn_history),
            "abstract_nodes": {k: round(v, 4) for k, v in self.abstract_nodes.items()},
            "programs": {name: prog.snapshot() for name, prog in self.programs.items()},
        }
