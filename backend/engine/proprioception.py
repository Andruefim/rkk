"""
proprioception.py — Level 3-G: Proprioception Stream (mini-GNN).

Выделенный быстрый поток для суставов тела, работающий в 50ms цикле.
Когнитивный GNN (d=60+) получает только агрегаты, а не raw joints.

Архитектура:
  ProprioStream (d=12 joints, hidden=24):
    - Быстрый GRU/MLP на 12 суставах ног+рук
    - Выдаёт абстракты: proprio_balance, proprio_gait_phase,
      proprio_left_leg_load, proprio_right_leg_load,
      proprio_arm_momentum_l, proprio_arm_momentum_r
    - Детектирует аномалии: joint_anomaly_score
    - Вычисляет Empowerment (число доступных будущих состояний)

  ProprioAggregator:
    - Собирает агрегаты из ProprioStream
    - Инжектирует их как узлы proprio_* в главный GNN
    - Обновляет каждые RKK_PROPRIO_EVERY=3 тика (быстро)

  EmpowermentEstimator:
    - Monte Carlo оценка числа достижимых состояний за K шагов
    - Reward за нахождение в «богатом» состоянии
    - «Стоять на двух ногах посередине = больше вариантов»

RKK_PROPRIO_ENABLED=1       — включить (default)
RKK_PROPRIO_EVERY=3         — тиков между обновлениями
RKK_PROPRIO_HIDDEN=24       — hidden size mini-GNN
RKK_EMPOWERMENT_K=4         — шаги для empowerment оценки
RKK_EMPOWERMENT_SAMPLES=8   — MC samples
RKK_PROPRIO_ABSTRACT_N=6    — число абстрактных узлов
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


def proprio_enabled() -> bool:
    return os.environ.get("RKK_PROPRIO_ENABLED", "1").strip().lower() not in (
        "0", "false", "no", "off"
    )


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


# ── Joint layout ──────────────────────────────────────────────────────────────
# 12 joints для proprioception stream
PROPRIO_JOINTS = [
    "lhip", "lknee", "lankle",
    "rhip", "rknee", "rankle",
    "lshoulder", "lelbow",
    "rshoulder", "relbow",
    "spine_pitch", "spine_yaw",
]

# Абстрактные узлы выходящие в главный GNN
PROPRIO_ABSTRACTS = [
    "proprio_balance",           # общий баланс тела [0,1]
    "proprio_gait_phase",        # фаза походки [0,1]
    "proprio_left_leg_load",     # нагрузка левой ноги [0,1]
    "proprio_right_leg_load",    # нагрузка правой ноги [0,1]
    "proprio_arm_counterbalance",# компенсация руками [0,1]
    "proprio_anomaly",           # аномалия сустава [0,1] → сигнал опасности
]

N_PROPRIO_IN = len(PROPRIO_JOINTS)   # 12
N_PROPRIO_OUT = len(PROPRIO_ABSTRACTS)  # 6


# ── Mini-GNN for proprioception ───────────────────────────────────────────────
class ProprioNet(nn.Module):
    """
    Лёгкая рекуррентная сеть для 12 суставов → 6 абстрактов.

    Architecture:
      joint_enc:  (12,) → (hidden,)   — encode joint state
      gru:        (hidden, h) → h     — track dynamics
      abstract:   (hidden,) → (6,)    — decode abstracts
      anomaly:    (hidden,) → (1,)    — anomaly score
    """

    def __init__(self, hidden: int = 24, device: torch.device | None = None):
        super().__init__()
        dev = device or torch.device("cpu")
        self.hidden = hidden

        self.joint_enc = nn.Sequential(
            nn.Linear(N_PROPRIO_IN, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.gru = nn.GRUCell(hidden, hidden)
        self.abstract_dec = nn.Sequential(
            nn.Linear(hidden, N_PROPRIO_OUT),
            nn.Sigmoid(),  # [0, 1]
        )
        self.anomaly_dec = nn.Sequential(
            nn.Linear(hidden, 1),
            nn.Sigmoid(),
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.4)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        self.to(dev)
        self.device = dev
        self._h = torch.zeros(1, hidden, device=dev)

    def reset_hidden(self) -> None:
        self._h = torch.zeros(1, self.hidden, device=self.device)

    def forward(
        self, joints: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        joints: (1, 12) normalized joint values
        Returns: abstracts (1, 6), anomaly (1, 1)
        """
        z = self.joint_enc(joints)          # (1, hidden)
        h = self.gru(z, self._h)            # (1, hidden)
        self._h = h.detach()
        abstracts = self.abstract_dec(h)    # (1, 6)
        anomaly = self.anomaly_dec(h)       # (1, 1)
        return abstracts, anomaly

    @torch.no_grad()
    def infer(self, joint_values: list[float]) -> dict[str, float]:
        """Fast inference from normalized joint values."""
        self.eval()
        x = torch.tensor(joint_values, dtype=torch.float32, device=self.device).unsqueeze(0)
        abstracts, anomaly = self(x)
        result = {}
        for i, name in enumerate(PROPRIO_ABSTRACTS):
            result[name] = float(abstracts[0, i].item())
        result["proprio_anomaly"] = float(anomaly[0, 0].item())
        return result


# ── Empowerment estimator ──────────────────────────────────────────────────────
class EmpowermentEstimator:
    """
    Monte Carlo empowerment: сколько различных будущих состояний
    достижимо за K шагов из текущего состояния.

    Empowerment ≈ I(A; S_t+K | S_t) = log(|reachable states|)

    Реализация: семплируем M случайных action sequences,
    прогоняем через RSSM/GNN, считаем стандартное отклонение.
    Высокое std = много вариантов = высокий empowerment.

    Это поощряет агента:
    - Стоять прямо (много возможных движений)
    - Не залезать в углы (мало вариантов движений)
    - Удерживать равновесие (возможен шаг в любую сторону)
    """

    def __init__(self, device: torch.device):
        self.device = device
        self._k = _env_int("RKK_EMPOWERMENT_K", 4)
        self._m = _env_int("RKK_EMPOWERMENT_SAMPLES", 8)
        self._history: deque[float] = deque(maxlen=100)
        self._mean_emp: float = 0.5

    @torch.no_grad()
    def estimate(
        self,
        current_state: dict[str, float],
        graph,              # CausalGraph
        node_ids: list[str],
    ) -> float:
        """
        Estimate empowerment for current state.
        Returns normalized empowerment score [0, 1].
        """
        if graph._core is None or len(node_ids) == 0:
            return 0.5

        d = len(node_ids)
        x0 = torch.tensor(
            [[float(current_state.get(n, 0.0)) for n in node_ids]],
            dtype=torch.float32, device=self.device,
        )

        # Sample M random action sequences and rollout K steps
        reached: list[torch.Tensor] = []
        for _ in range(self._m):
            x = x0.clone()
            for step in range(self._k):
                # Random action: pick a random joint/intent and random value
                a = torch.zeros(1, d, device=self.device)
                idx = np.random.randint(0, d)
                val = float(np.random.uniform(0.2, 0.8))
                a[0, idx] = val
                try:
                    from engine.wm_neural_ode import integrate_world_model_step
                    x = integrate_world_model_step(graph._core, x, a)
                except Exception:
                    break
            reached.append(x.squeeze(0))

        if len(reached) < 2:
            return 0.5

        # Measure diversity of reached states (std across samples)
        reached_stack = torch.stack(reached, dim=0)  # (M, d)
        std = reached_stack.std(dim=0).mean().item()  # mean std across dims

        # Normalize: empowerment = clip(std * scale, 0, 1)
        emp = float(np.clip(std * 8.0, 0.0, 1.0))
        self._history.append(emp)

        # EMA
        alpha = 0.05
        self._mean_emp = (1.0 - alpha) * self._mean_emp + alpha * emp
        return emp

    def get_reward(self, current_emp: float) -> float:
        """
        Reward for empowerment: positive when above historical mean.
        Encourages staying in states with many future options.
        """
        baseline = float(np.mean(list(self._history)[-20:])) if len(self._history) >= 20 else 0.5
        return float(np.clip(current_emp - baseline, -0.5, 0.5))

    def snapshot(self) -> dict[str, Any]:
        return {
            "mean_empowerment": round(self._mean_emp, 4),
            "k_steps": self._k,
            "m_samples": self._m,
            "history_len": len(self._history),
        }


# ── Proprioception Stream Controller ──────────────────────────────────────────
class ProprioceptionStream:
    """
    Полный контроллер проприоцепции.

    Работает на 50ms (≈20Hz) цикле независимо от когнитивного GNN.
    Инжектирует abstracts как узлы proprio_* в главный граф.

    Интеграция в simulation.py:
      self._proprio = ProprioceptionStream(device)

    В _maybe_apply_cpg_locomotion() после joints applied:
      self._proprio.update(obs, graph, tick)
    """

    def __init__(self, device: torch.device):
        self.device = device
        hidden = _env_int("RKK_PROPRIO_HIDDEN", 24)
        self.net = ProprioNet(hidden=hidden, device=device)
        self.optim = torch.optim.Adam(self.net.parameters(), lr=5e-4)
        self.emp = EmpowermentEstimator(device=device)

        self._every = _env_int("RKK_PROPRIO_EVERY", 3)
        self._last_tick: int = -999
        self._abstracts: dict[str, float] = {k: 0.5 for k in PROPRIO_ABSTRACTS}
        self._current_anomaly: float = 0.0
        self._current_empowerment: float = 0.5
        self._abstract_injected: bool = False

        # Training buffer: (joint_t, joint_t+1) pairs for predictive training
        self._train_buf: deque[tuple[list[float], list[float]]] = deque(maxlen=128)
        self._prev_joints: list[float] | None = None
        self.train_steps: int = 0
        self._loss_history: deque[float] = deque(maxlen=50)

    def _extract_joints(self, obs: dict[str, float]) -> list[float]:
        """Extract 12 normalized joint values from observation."""
        return [
            float(obs.get(j, obs.get(f"phys_{j}", 0.5)))
            for j in PROPRIO_JOINTS
        ]

    def update(
        self,
        tick: int,
        obs: dict[str, float],
        graph,
        agent=None,
    ) -> dict[str, float]:
        """
        Main update step. Returns current abstracts.
        """
        if not proprio_enabled():
            return self._abstracts

        if (tick - self._last_tick) < self._every:
            return self._abstracts

        self._last_tick = tick
        joints = self._extract_joints(obs)

        # Training: predict next joint state
        if self._prev_joints is not None:
            self._train_buf.append((list(self._prev_joints), list(joints)))
            if len(self._train_buf) >= 16 and tick % 16 == 0:
                self._train_step()
        self._prev_joints = list(joints)

        # Inference
        abstracts_dict = self.net.infer(joints)
        self._abstracts = abstracts_dict
        self._current_anomaly = float(abstracts_dict.get("proprio_anomaly", 0.0))

        # Compute empowerment (less frequently)
        if tick % 30 == 0 and graph is not None:
            node_ids = list(graph._node_ids)
            state = dict(graph.nodes)
            self._current_empowerment = self.emp.estimate(state, graph, node_ids)

        # Inject abstracts into main GNN
        if graph is not None:
            self._inject_into_graph(graph, tick)

        return self._abstracts

    def _inject_into_graph(self, graph, tick: int) -> None:
        """Inject/update proprio_* abstract nodes in main GNN."""
        # Add nodes on first call
        if not self._abstract_injected:
            for name in PROPRIO_ABSTRACTS:
                if name not in graph.nodes:
                    try:
                        graph.set_node(name, 0.5)
                    except Exception:
                        pass
                    # Add edges: proprio_balance → posture_stability etc.
                    self._inject_structural_edges(graph)
            self._abstract_injected = True

        # Update values every tick
        for name, val in self._abstracts.items():
            if name in graph.nodes:
                graph.nodes[name] = float(np.clip(val, 0.05, 0.95))

        # Add empowerment node
        emp_key = "proprio_empowerment"
        if emp_key not in graph.nodes:
            try:
                graph.set_node(emp_key, 0.5)
            except Exception:
                pass
        if emp_key in graph.nodes:
            graph.nodes[emp_key] = float(np.clip(self._current_empowerment, 0.05, 0.95))

    def _inject_structural_edges(self, graph) -> None:
        """Inject causal edges between proprio abstracts and body variables."""
        edges = [
            # Proprio balance → key body vars
            ("proprio_balance", "posture_stability", 0.40),
            ("proprio_gait_phase", "intent_stride", 0.30),
            ("proprio_left_leg_load", "foot_contact_l", 0.35),
            ("proprio_right_leg_load", "foot_contact_r", 0.35),
            ("proprio_arm_counterbalance", "support_bias", 0.25),
            # Anomaly → recovery signal
            ("proprio_anomaly", "intent_stop_recover", 0.28),
        ]
        for fr, to, w in edges:
            if fr in graph.nodes and to in graph.nodes:
                try:
                    graph.set_edge(fr, to, w, alpha=0.06)
                except Exception:
                    pass

    def _train_step(self) -> float | None:
        """Train ProprioNet to predict next joint state (predictive coding)."""
        if len(self._train_buf) < 16:
            return None

        batch = list(self._train_buf)[-32:]
        xs = torch.tensor([b[0] for b in batch], dtype=torch.float32, device=self.device)
        ys = torch.tensor([b[1] for b in batch], dtype=torch.float32, device=self.device)

        self.net.train()
        self.optim.zero_grad()

        # Encode xs, decode next state prediction
        z = self.net.joint_enc(xs)
        # Simple next-state prediction head (reuse abstract_dec projected)
        # Use anomaly logic: high anomaly = high prediction error
        abstracts, anomaly = self.net(xs[:1])  # dummy forward to train GRU
        # Direct predictive loss on encoded features
        z_next = self.net.joint_enc(ys)
        loss = F.mse_loss(z, z_next.detach())
        loss.backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)
        self.optim.step()

        v = float(loss.item())
        self._loss_history.append(v)
        self.train_steps += 1
        return v

    @property
    def anomaly_score(self) -> float:
        return self._current_anomaly

    @property
    def empowerment(self) -> float:
        return self._current_empowerment

    def get_empowerment_reward(self) -> float:
        return self.emp.get_reward(self._current_empowerment)

    def snapshot(self) -> dict[str, Any]:
        return {
            "enabled": proprio_enabled(),
            "abstracts": {k: round(v, 4) for k, v in self._abstracts.items()},
            "anomaly": round(self._current_anomaly, 4),
            "empowerment": round(self._current_empowerment, 4),
            "train_steps": self.train_steps,
            "mean_loss": round(float(np.mean(list(self._loss_history))), 5) if self._loss_history else 0.0,
            "empowerment_estimator": self.emp.snapshot(),
        }
