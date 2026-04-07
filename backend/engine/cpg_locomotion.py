"""
cpg_locomotion.py — Phase A: Locomotion (CPG поверх агента).

Central Pattern Generator: фазовые осцилляторы для ритма ног; высокоуровневые
сигналы из узлов GNN модулируют амплитуду. Отдельный суррогатный reward
(com_z + продвижение com_x, штраф за падение) крутит параметры CPG.

Включается симуляцией при RKK_LOCOMOTION_CPG=1, humanoid, не fixed_root.
"""
from __future__ import annotations

import os
import math
import torch
import torch.nn as nn
import numpy as np


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
        freq = torch.sigmoid(self.frequency) * 3.0 + 0.5
        amp = torch.sigmoid(self.amplitude) * 0.8 + 0.1

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
    """

    CPG_TO_JOINTS: dict[int, list[str]] = {
        0: ["lhip", "lknee"],
        1: ["rhip", "rknee"],
        2: ["lankle"],
        3: ["rankle"],
    }

    def __init__(self, device: torch.device):
        self.device = device
        self.cpg = CPGNetwork(n_oscillators=4, device=device)
        self.optim = torch.optim.Adam(self.cpg.parameters(), lr=1e-3)
        self._step_count = 0
        self._last_com_x: float = 0.5
        self._reward_history: list[float] = []
        self._last_command: dict[str, float] = {}
        self._last_motor_state: dict[str, float] = {}

    @staticmethod
    def _node(agent_nodes: dict[str, float], key: str) -> float:
        v = agent_nodes.get(key)
        if v is None:
            v = agent_nodes.get(f"phys_{key}")
        return float(v if v is not None else 0.5)

    def get_joint_targets(self, agent_nodes: dict[str, float], *, dt: float = 0.05) -> dict[str, float]:
        """agent_nodes: нормализованные узлы графа; intent_* и self_* модулируют ноги."""
        intents = {
            "intent_stride": self._node(agent_nodes, "intent_stride"),
            "intent_support_left": self._node(agent_nodes, "intent_support_left"),
            "intent_support_right": self._node(agent_nodes, "intent_support_right"),
            "intent_torso_forward": self._node(agent_nodes, "intent_torso_forward"),
            "intent_arm_counterbalance": self._node(agent_nodes, "intent_arm_counterbalance"),
            "intent_stop_recover": self._node(agent_nodes, "intent_stop_recover"),
        }
        self._last_motor_state = dict(intents)
        lh = self._node(agent_nodes, "lhip") - 0.5
        rh = self._node(agent_nodes, "rhip") - 0.5
        la = self._node(agent_nodes, "lankle") - 0.5
        ra = self._node(agent_nodes, "rankle") - 0.5
        larm = self._node(agent_nodes, "self_intention_larm") - 0.5
        rarm = self._node(agent_nodes, "self_intention_rarm") - 0.5
        eng = float(np.clip(self._node(agent_nodes, "self_energy"), 0.0, 1.0))

        stride = float(intents["intent_stride"] - 0.5)
        sup_l = float(intents["intent_support_left"] - 0.5)
        sup_r = float(intents["intent_support_right"] - 0.5)
        torso = float(intents["intent_torso_forward"] - 0.5)
        arms = float(intents["intent_arm_counterbalance"] - 0.5)
        recover = float(intents["intent_stop_recover"] - 0.5)

        cmd = torch.tensor([lh, rh, la, ra], dtype=torch.float32, device=self.device)
        cmd[0] = cmd[0] + 0.12 * float(larm) + 0.10 * stride - 0.05 * sup_r + 0.03 * torso
        cmd[1] = cmd[1] + 0.12 * float(rarm) - 0.10 * stride - 0.05 * sup_l + 0.03 * torso
        cmd[2] = cmd[2] + 0.08 * sup_l - 0.04 * stride + 0.03 * recover
        cmd[3] = cmd[3] + 0.08 * sup_r + 0.04 * stride + 0.03 * recover
        cmd = cmd * float(0.68 + 0.58 * eng)

        cpg_out = self.cpg.step(dt=dt, external_command=cmd)
        targets: dict[str, float] = {}
        for osc_idx, joints in self.CPG_TO_JOINTS.items():
            v = float(cpg_out[osc_idx].item())
            if osc_idx == 0:
                v = float(np.clip(v + 0.12 * stride - 0.05 * sup_r, 0.0, 1.0))
            elif osc_idx == 1:
                v = float(np.clip(v - 0.12 * stride - 0.05 * sup_l, 0.0, 1.0))
            elif osc_idx == 2:
                v = float(np.clip(v + 0.06 * recover + 0.04 * sup_l, 0.0, 1.0))
            else:
                v = float(np.clip(v + 0.06 * recover + 0.04 * sup_r, 0.0, 1.0))
            for joint in joints:
                targets[joint] = v
        self._last_command = dict(targets)
        self._step_count += 1
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
        Структурный reward: устойчивость корпуса + опора + симметрия + forward progress.
        """
        dx = float(com_x) - self._last_com_x
        rz = float(np.clip(com_z, 0.0, 1.0))
        motor_obs = motor_obs or {}
        posture = float(np.clip(motor_obs.get("posture_stability", 0.5), 0.0, 1.0))
        contact_l = float(np.clip(motor_obs.get("foot_contact_l", 0.5), 0.0, 1.0))
        contact_r = float(np.clip(motor_obs.get("foot_contact_r", 0.5), 0.0, 1.0))
        bias = float(np.clip(motor_obs.get("support_bias", 0.5), 0.0, 1.0))
        gait_l = float(np.clip(motor_obs.get("gait_phase_l", 0.5), 0.0, 1.0))
        gait_r = float(np.clip(motor_obs.get("gait_phase_r", 0.5), 0.0, 1.0))
        symmetry = 1.0 - min(1.0, abs(gait_l - gait_r) + abs(contact_l - contact_r) + abs(bias - 0.5) * 1.4)
        reward = (
            rz * 1.6
            + dx * 2.8
            + posture * 2.2
            + symmetry * 1.4
            + min(contact_l, contact_r) * 0.8
            - (3.0 if fallen else 0.0)
        )
        self._last_com_x = float(com_x)
        self._reward_history.append(reward)

        try:
            win = int(os.environ.get("RKK_CPG_REWARD_WINDOW", "16"))
        except ValueError:
            win = 16
        win = max(4, min(win, 256))

        if len(self._reward_history) < win:
            return

        r_mean = float(np.mean(self._reward_history[-win:]))
        self.optim.zero_grad()
        dev = next(self.cpg.parameters()).device
        scale = torch.tensor(r_mean, device=dev, dtype=torch.float32)
        # Положительный r_mean → усилить устойчивый моторный ритм, а не только forward progress.
        loss = -scale * (
            0.45 * self.cpg.amplitude.mean()
            + 0.20 * self.cpg.frequency.mean()
            - 0.10 * torch.abs(self.cpg.phase_bias).mean()
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.cpg.parameters(), 0.5)
        self.optim.step()

    def snapshot(self) -> dict:
        with torch.no_grad():
            amp_m = float(torch.sigmoid(self.cpg.amplitude).mean().item())
            fr_m = float((torch.sigmoid(self.cpg.frequency) * 3.0 + 0.5).mean().item())
        rh = self._reward_history[-32:] if self._reward_history else []
        return {
            "cpg_steps": self._step_count,
            "amplitude_mean": round(amp_m, 4),
            "frequency_mean_hz": round(fr_m, 3),
            "reward_recent_mean": round(float(np.mean(rh)), 4) if rh else 0.0,
            "last_command_size": len(self._last_command),
            "last_intent_stride": round(float(self._last_motor_state.get("intent_stride", 0.5)), 4) if self._last_motor_state else 0.5,
        }
