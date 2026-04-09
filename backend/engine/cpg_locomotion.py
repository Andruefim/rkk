"""
cpg_locomotion.py — Phase A: Locomotion (CPG поверх агента).

Central Pattern Generator: фазовые осцилляторы для ритма ног; высокоуровневые
сигналы из узлов GNN модулируют амплитуду. Отдельный суррогатный reward
(com_z + продвижение com_x, штраф за падение) крутит параметры CPG.

Включается симуляцией при RKK_LOCOMOTION_CPG=1, humanoid, не fixed_root.

Целостная походка: фаза бёдер + отставание CoM; сила связки задаётся каузальным узлом
intent_gait_coupling (в среде по умолчанию 0.88).
"""
from __future__ import annotations

import os
import math
import torch
import torch.nn as nn
import numpy as np

# Ожидаемый микро-сдвиг нормализованного com_x за шаг при ходьбе; усиление штрафа при отставании массы.
_COM_VEL_EXPECT = 0.022
_COM_LAG_GAIN = 42.0


class CPGNetwork(nn.Module):
    """
    Несколько связанных фазовых осцилляторов (упрощённая фазовая модель).
    Выход: нормализованные цели суставов [0, 1].
    """

    def __init__(self, n_oscillators: int = 4, device: torch.device | None = None):
        super().__init__()
        self.n = n_oscillators
        dev = device or torch.device("cpu")

        self.amplitude = nn.Parameter(torch.ones(n_oscillators, device=dev) * 0.5)
        self.frequency = nn.Parameter(torch.ones(n_oscillators, device=dev) * 1.0)
        self.phase_bias = nn.Parameter(torch.zeros(n_oscillators, n_oscillators, device=dev))

        self.register_buffer("_phase", torch.zeros(n_oscillators, device=dev))

        self.to(dev)

    @torch.no_grad()
    def step(
        self,
        dt: float = 0.05,
        external_command: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Один шаг CPG → (n,) в [0, 1].
        external_command: (>=n,) на том же device, что модуль.
        """
        freq = torch.sigmoid(self.frequency) * 2.0 + 0.3
        amp = torch.sigmoid(self.amplitude) * 0.6

        if external_command is not None:
            ec = external_command[: self.n]
            amp = amp * (0.5 + 0.5 * torch.sigmoid(ec))

        two_pi = 2.0 * math.pi
        p = self._phase + two_pi * freq * float(dt)
        self._phase.copy_(torch.remainder(p, two_pi))

        diff = self._phase.unsqueeze(1) - self._phase.unsqueeze(0) - self.phase_bias
        coupling = torch.sin(diff).sum(dim=1) * 0.1
        p2 = self._phase + coupling * float(dt)
        self._phase.copy_(torch.remainder(p2, two_pi))

        out = amp * torch.sin(self._phase)
        return (out + 1.0) / 2.0

    def get_gait_pattern(self, n_steps: int = 100, dt: float = 0.02) -> np.ndarray:
        hist: list[np.ndarray] = []
        for _ in range(n_steps):
            out = self.step(dt=dt)
            hist.append(out.detach().cpu().numpy())
        return np.stack(hist, axis=0)


class LocomotionController:
    """
    CPG/MotorPolicy → цели суставов и моторный латент для humanoid.
    Управляется motor intents из causal graph и структурным reward.

    Каждый осциллятор управляет ОДНИМ суставом (6 осцилляторов = lhip, rhip, lknee, rknee, lankle, rankle).
    Бёдра и колени работают в анти-фазе: при разгибании бедра колено выпрямляется (опорная фаза),
    при сгибании бедра колено сгибается (фаза переноса).
    """

    def __init__(self, device: torch.device):
        self.device = device
        self.cpg = CPGNetwork(n_oscillators=6, device=device)

        with torch.no_grad():
            self.cpg.phase_bias[0, 1] = math.pi
            self.cpg.phase_bias[1, 0] = -math.pi
            self.cpg.phase_bias[2, 3] = math.pi
            self.cpg.phase_bias[3, 2] = -math.pi
            self.cpg.phase_bias[0, 2] = math.pi * 0.25
            self.cpg.phase_bias[1, 3] = math.pi * 0.25
            self.cpg.frequency.data[:] = 0.3
            self.cpg.amplitude.data[:] = -1.5

        self.optim = torch.optim.Adam(self.cpg.parameters(), lr=3e-4)
        self._step_count = 0
        self._last_com_x: float = 0.5
        self._last_com_z: float = 0.5
        self._reward_history: list[float] = []
        self._last_command: dict[str, float] = {}
        self._last_motor_state: dict[str, float] = {}
        self._last_cpg_sync: dict[str, float] = {}
        self._com_x_prev_step: float | None = None

    @staticmethod
    def _node(agent_nodes: dict[str, float], key: str) -> float:
        v = agent_nodes.get(key)
        if v is None:
            v = agent_nodes.get(f"phys_{key}")
        return float(v if v is not None else 0.5)

    def upper_body_cpg_sync(self) -> dict[str, float]:
        """Фаза CPG + инерция CoM для согласованного корпуса с ногами (после get_joint_targets)."""
        return dict(self._last_cpg_sync)

    def get_joint_targets(self, agent_nodes: dict[str, float], *, dt: float = 0.05) -> dict[str, float]:
        """
        Исправленная версия: сильный forward lean при stride > 0.5
        """
        stride = float(self._node(agent_nodes, "intent_stride") - 0.5)
        sup_l = float(self._node(agent_nodes, "intent_support_left") - 0.5)
        sup_r = float(self._node(agent_nodes, "intent_support_right") - 0.5)
        recover = float(self._node(agent_nodes, "intent_stop_recover") - 0.5)
        energy = float(np.clip(self._node(agent_nodes, "self_energy"), 0.0, 1.0))

        # === КЛЮЧЕВОЕ ИСПРАВЛЕНИЕ ===
        # Сильный наклон торса вперёд при ходьбе
        torso_forward = 0.5 + 0.52 * stride + 0.15 * energy
        torso_forward = np.clip(torso_forward, 0.38, 0.96)

        # Сохраняем для upper_body
        self._last_motor_state["intent_torso_forward"] = float(torso_forward)

        # CPG-команды для ног
        cmd = torch.zeros(6, dtype=torch.float32, device=self.device)
        cmd[0] =  0.19 * stride - 0.08 * sup_r - 0.05 * recover   # lhip
        cmd[1] = -0.19 * stride - 0.08 * sup_l - 0.05 * recover   # rhip
        cmd[2] =  0.14 * sup_l + 0.10 * recover                    # lknee
        cmd[3] =  0.14 * sup_r + 0.10 * recover                    # rknee
        cmd[4] =  0.08 * sup_l - 0.04 * stride + 0.05 * recover    # lankle
        cmd[5] =  0.08 * sup_r + 0.04 * stride + 0.05 * recover    # rankle

        cpg_out = self.cpg.step(dt=dt, external_command=cmd * (0.7 + 0.3 * energy))

        # Формируем цели суставов
        targets: dict[str, float] = {
            "lhip":   float(np.clip(0.50 + 0.17 * (float(cpg_out[0].item())*2 - 1), 0.05, 0.95)),
            "rhip":   float(np.clip(0.50 + 0.17 * (float(cpg_out[1].item())*2 - 1), 0.05, 0.95)),
            "lknee":  float(np.clip(0.50 + 0.14 * (float(cpg_out[2].item())*2 - 1), 0.05, 0.95)),
            "rknee":  float(np.clip(0.50 + 0.14 * (float(cpg_out[3].item())*2 - 1), 0.05, 0.95)),
            "lankle": float(np.clip(0.50 + 0.09 * (float(cpg_out[4].item())*2 - 1), 0.05, 0.95)),
            "rankle": float(np.clip(0.50 + 0.09 * (float(cpg_out[5].item())*2 - 1), 0.05, 0.95)),
        }

        return targets

    def learn_from_reward(
        self,
        com_z: float,
        com_x: float,
        fallen: bool,
        *,
        motor_obs: dict[str, float] | None = None,
    ) -> None:
        """
        Phased reward: standing first, then forward progress.
        Phase 1 (com_z < 0.35): max reward for increasing com_z (get upright).
        Phase 2 (upright): stability + symmetry + gentle forward.
        """
        dx = float(com_x) - self._last_com_x
        dz = float(com_z) - self._last_com_z
        rz = float(np.clip(com_z, 0.0, 1.0))
        motor_obs = motor_obs or {}
        posture = float(np.clip(motor_obs.get("posture_stability", 0.5), 0.0, 1.0))
        contact_l = float(np.clip(motor_obs.get("foot_contact_l", 0.5), 0.0, 1.0))
        contact_r = float(np.clip(motor_obs.get("foot_contact_r", 0.5), 0.0, 1.0))
        bias = float(np.clip(motor_obs.get("support_bias", 0.5), 0.0, 1.0))
        gait_l = float(np.clip(motor_obs.get("gait_phase_l", 0.5), 0.0, 1.0))
        gait_r = float(np.clip(motor_obs.get("gait_phase_r", 0.5), 0.0, 1.0))
        symmetry = 1.0 - min(1.0, abs(gait_l - gait_r) + abs(contact_l - contact_r) + abs(bias - 0.5) * 1.4)

        upright = rz > 0.35

        if not upright:
            reward = (
                rz * 4.0
                + dz * 8.0
                + posture * 1.5
                + min(contact_l, contact_r) * 1.0
                - (5.0 if fallen else 0.0)
            )
        else:
            # Walking: do not punish asymmetric support_bias as harshly; encourage CoM_x forward (ZMP/CoM ahead).
            stride_n = abs(float(self._last_motor_state.get("intent_stride", 0.5)) - 0.5) * 2.0
            torso_n = abs(float(self._last_motor_state.get("intent_torso_forward", 0.5)) - 0.5) * 2.0
            walk_drive = float(np.clip(0.5 * stride_n + 0.5 * torso_n, 0.0, 1.0))
            bias_pen = abs(bias - 0.5) * (1.0 - 0.55 * walk_drive)
            cx = float(com_x)
            forward_bonus = 2.0 * max(0.0, cx - 0.46)
            back_penalty = 1.6 * max(0.0, 0.41 - cx)
            coherence = 0.0
            if self._last_cpg_sync:
                cl = float(self._last_cpg_sync.get("com_lag", 0.0))
                sn = float(self._last_cpg_sync.get("stride_n", 0.0))
                gs = float(self._last_cpg_sync.get("gscale", 1.0))
                coherence = 0.35 * (1.0 - cl) * min(1.0, sn * 2.0) * gs
            reward = (
                rz * 3.0
                + posture * 3.0
                + symmetry * 2.0
                + min(contact_l, contact_r) * 1.5
                + dx * 1.85
                + forward_bonus
                - back_penalty
                + coherence
                - bias_pen * 1.0
                - (5.0 if fallen else 0.0)
            )

        self._last_com_x = float(com_x)
        self._last_com_z = float(com_z)
        self._reward_history.append(reward)

        try:
            win = int(os.environ.get("RKK_CPG_REWARD_WINDOW", "24"))
        except ValueError:
            win = 24
        win = max(4, min(win, 256))

        if len(self._reward_history) < win:
            return

        r_mean = float(np.mean(self._reward_history[-win:]))
        self.optim.zero_grad()
        dev = next(self.cpg.parameters()).device
        scale = torch.tensor(r_mean, device=dev, dtype=torch.float32)
        loss = -scale * (
            0.35 * self.cpg.amplitude.mean()
            + 0.15 * self.cpg.frequency.mean()
            - 0.15 * torch.abs(self.cpg.phase_bias).mean()
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.cpg.parameters(), 0.3)
        self.optim.step()

    def snapshot(self) -> dict:
        with torch.no_grad():
            amp_m = float(torch.sigmoid(self.cpg.amplitude).mean().item())
            fr_m = float((torch.sigmoid(self.cpg.frequency) * 3.0 + 0.5).mean().item())
        rh = self._reward_history[-32:] if self._reward_history else []
        lag = float(self._last_cpg_sync.get("com_lag", 0.0)) if self._last_cpg_sync else 0.0
        return {
            "cpg_steps": self._step_count,
            "amplitude_mean": round(amp_m, 4),
            "frequency_mean_hz": round(fr_m, 3),
            "reward_recent_mean": round(float(np.mean(rh)), 4) if rh else 0.0,
            "last_command_size": len(self._last_command),
            "last_intent_stride": round(float(self._last_motor_state.get("intent_stride", 0.5)), 4) if self._last_motor_state else 0.5,
            "cpg_com_lag": round(lag, 4),
        }
