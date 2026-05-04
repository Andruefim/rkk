"""
cpg_locomotion.py — Phase A: Locomotion (CPG поверх агента) + Phase D: CPG Annealing.

Swing (бедро/колено):
  - Фаза осцилляторов CPG [0],[1] для лев/прав бедра; swing_factor = max(0, sin(phi))
  - В swing добавляется подъём бедра и сгиб колена (RKK_CPG_SWING_HIP_LIFT / KNEE_FLEX),
    чтобы не компенсировать только наклоном торса назад.
  - Компенсация торса от com_lag ослаблена (RKK_CPG_COM_LAG_PITCH, дефолт 0.08) — см. environment_humanoid.

ИЗМЕНЕНИЯ (Motor Cortex / Phase D):
  - get_joint_targets: усилен forward lean, добавлен com_lag компенсатор
  - upper_body_cpg_sync: возвращает com_lag для наклона торса вперёд
  - Обучение CPG: train_cpg_from_intrinsic_history() (engine.intristic_objective)

Проблема «заваливается назад»: при stride>0 торс должен быть наклонён вперёд.
Основные фиксы:
  1. intent_torso_forward масштабирован на 1.6× (был 1.45)
  2. com_lag penalty + recovery теперь тянет CoM вперёд активнее
  3. Добавлен _step_phase для определения stance/swing per leg

Central Pattern Generator: фазовые осцилляторы для ритма ног; высокоуровневые
сигналы из узлов GNN модулируют амплитуду.
"""
from __future__ import annotations

import os
import math
import torch
import torch.nn as nn
import numpy as np

_COM_VEL_EXPECT = 0.022
_COM_LAG_GAIN = 42.0

# Swing phase: подъём бедра + сгиб колена (вместо компенсации только торсом назад).
# Нормализованные дельты к целям 0..1; усилить через RKK_CPG_SWING_HIP_LIFT (напр. 0.10–0.12).
def _env_cpg_swing_float(key: str, default: str) -> float:
    try:
        return float(os.environ.get(key, default))
    except ValueError:
        return float(default)


class CPGNetwork(nn.Module):
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

    FIX «заваливается назад»:
      - stride_n > 0 → torso_forward = 0.5 + 0.62*stride (было 0.52)
      - com_lag > threshold → дополнительный наклон вперёд (pitch_add > 0)
      - gscale: масштаб coupling синхронизации
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

        # MOTOR_CORTEX: cpg_weight is set externally by MotorCortexLibrary
        self.cpg_weight: float = 1.0  # used for diagnostics only here

    @staticmethod
    def _node(agent_nodes: dict[str, float], key: str) -> float:
        v = agent_nodes.get(key)
        if v is None:
            v = agent_nodes.get(f"phys_{key}")
        return float(v if v is not None else 0.5)

    def upper_body_cpg_sync(self) -> dict[str, float]:
        """Фаза CPG + инерция CoM для согласованного корпуса с ногами."""
        return dict(self._last_cpg_sync)

    def get_joint_targets(self, agent_nodes: dict[str, float], *, dt: float = 0.05) -> dict[str, float]:
        """
        FIXED: усиленный forward lean при stride > 0.
        
        Ключевые изменения vs оригинала:
        - torso_forward scale: 0.52 → 0.62 (больше наклон вперёд)
        - com_lag penalty активно тянет CoM вперёд при отставании
        - CPG sync: pitch_add при com_lag теперь положительный (наклон вперёд)
        """
        stride = float(self._node(agent_nodes, "intent_stride") - 0.5)
        sup_l = float(self._node(agent_nodes, "intent_support_left") - 0.5)
        sup_r = float(self._node(agent_nodes, "intent_support_right") - 0.5)
        recover = float(self._node(agent_nodes, "intent_stop_recover") - 0.5)
        energy = float(np.clip(self._node(agent_nodes, "self_energy"), 0.0, 1.0))
        com_x = float(self._node(agent_nodes, "com_x"))
        com_z = float(self._node(agent_nodes, "com_z"))

        # === FORWARD LEAN FIX ===
        # При stride > 0 агент должен лидировать корпусом — иначе падает назад
        # com_x: нормализованная позиция ≈ 0.5 = center. >0.5 = forward of stance foot.
        # Мы хотим com_x чуть впереди опорной стопы при ходьбе.
        com_lag = float(np.clip(0.48 - com_x, 0.0, 0.15))  # >0 если CoM отстаёт
        stride_n = max(0.0, stride)  # только positive stride (forward)
        
        # Forward lean: base + stride contribution + com_lag recovery
        torso_forward = (
            0.5
            + 0.62 * stride_n          # главный forward lean при ходьбе (УСИЛЕН с 0.52)
            + 0.40 * com_lag           # активное восстановление если CoM позади
            + 0.15 * energy            # энергичные движения = больше наклон
        )
        torso_forward = float(np.clip(torso_forward, 0.38, 0.96))
        self._last_motor_state["intent_torso_forward"] = float(torso_forward)

        from engine.features.humanoid.constants import UPPER_BODY_INTENT_VARS

        for _uk in UPPER_BODY_INTENT_VARS:
            self._last_motor_state[str(_uk)] = float(self._node(agent_nodes, str(_uk)))

        intent_agg = 0.0
        try:
            for key in agent_nodes:
                sk = str(key)
                if not sk.startswith("intent_"):
                    continue
                if sk == "intent_gait_coupling":
                    continue
                intent_agg += abs(float(self._node(agent_nodes, sk)) - 0.5)
        except Exception:
            intent_agg = 0.0
        drive_damp = float(np.clip(1.0 - 0.06 * intent_agg, 0.72, 1.0))

        # CPG команды для ног
        cmd = torch.zeros(6, dtype=torch.float32, device=self.device)
        cmd[0] =  0.19 * stride - 0.08 * sup_r - 0.05 * recover   # lhip
        cmd[1] = -0.19 * stride - 0.08 * sup_l - 0.05 * recover   # rhip
        cmd[2] =  0.14 * sup_l + 0.10 * recover                    # lknee
        cmd[3] =  0.14 * sup_r + 0.10 * recover                    # rknee
        cmd[4] =  0.08 * sup_l - 0.04 * stride + 0.05 * recover    # lankle
        cmd[5] =  0.08 * sup_r + 0.04 * stride + 0.05 * recover    # rankle

        cpg_out = self.cpg.step(
            dt=dt,
            external_command=cmd * (0.7 + 0.3 * energy) * drive_damp,
        )

        # Торс синхронизация с CPG: pitch_add > 0 = наклон вперёд
        gscale = float(np.clip(stride_n * 1.8, 0.0, 1.0))
        s = float(torch.sin(self.cpg._phase[0]).item())
        c_m = float(torch.cos(self.cpg._phase[2]).item())
        
        # com_lag → небольшой pitch вперёд; коэффициент снижен — основную работу делают hip/knee в swing
        try:
            _lag_pitch = float(os.environ.get("RKK_CPG_COM_LAG_PITCH", "0.08"))
        except ValueError:
            _lag_pitch = 0.08
        _lag_pitch = float(np.clip(_lag_pitch, 0.0, 0.35))
        pitch_add = (
            -0.055 * s * gscale
            + _lag_pitch * com_lag * stride_n
        )
        yaw_add = 0.05 * c_m * gscale
        lsh_add = -0.065 * s * gscale
        rsh_add =  0.065 * s * gscale

        phi_l = float(self.cpg._phase[0].item())
        phi_r = float(self.cpg._phase[1].item())

        def _swing_factor(phi: float) -> float:
            # 0 в «низе» синуса, пик около π/2 для положительного подъёма бедра в swing
            return float(max(0.0, math.sin(phi)))

        swing_l = _swing_factor(phi_l)
        swing_r = _swing_factor(phi_r)
        hip_lift = _env_cpg_swing_float("RKK_CPG_SWING_HIP_LIFT", "0.08")
        knee_flex = _env_cpg_swing_float("RKK_CPG_SWING_KNEE_FLEX", "0.10")
        walk_gate = float(np.clip(stride_n * (0.35 + 0.65 * gscale), 0.0, 1.0))

        self._last_cpg_sync = {
            "sin": s, "cos_mid": c_m,
            "stride_n": stride_n,
            "com_lag": com_lag,
            "gscale": gscale,
            "pitch_add": float(pitch_add),
            "swing_l": swing_l,
            "swing_r": swing_r,
            "phi_l": phi_l,
            "phi_r": phi_r,
        }

        knee_base = 0.45
        targets: dict[str, float] = {
            "lhip":   float(np.clip(0.50 + 0.17 * (float(cpg_out[0].item())*2 - 1), 0.05, 0.95)),
            "rhip":   float(np.clip(0.50 + 0.17 * (float(cpg_out[1].item())*2 - 1), 0.05, 0.95)),
            "lknee":  float(np.clip(knee_base + 0.14 * (float(cpg_out[2].item())*2 - 1), 0.05, 0.95)),
            "rknee":  float(np.clip(knee_base + 0.14 * (float(cpg_out[3].item())*2 - 1), 0.05, 0.95)),
            "lankle": float(np.clip(0.50 + 0.09 * (float(cpg_out[4].item())*2 - 1), 0.05, 0.95)),
            "rankle": float(np.clip(0.50 + 0.09 * (float(cpg_out[5].item())*2 - 1), 0.05, 0.95)),
        }

        # Swing phase: hip lift + knee flex (минус = сгибание колена в нормализованных целях)
        targets["lhip"] = float(
            np.clip(targets["lhip"] + hip_lift * swing_l * walk_gate, 0.05, 0.95)
        )
        targets["lknee"] = float(
            np.clip(targets["lknee"] - knee_flex * swing_l * walk_gate, 0.05, 0.95)
        )
        targets["rhip"] = float(
            np.clip(targets["rhip"] + hip_lift * swing_r * walk_gate, 0.05, 0.95)
        )
        targets["rknee"] = float(
            np.clip(targets["rknee"] - knee_flex * swing_r * walk_gate, 0.05, 0.95)
        )

        self._step_count += 1
        self._last_command = dict(targets)
        return targets

    def train_cpg_from_intrinsic_history(self) -> None:
        """
        Обучение CPG только из _reward_history (заполняется IntrinsicObjective).
        Вызывается после agent.step; см. engine.intristic_objective.
        """
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
        sync = self._last_cpg_sync or {}
        return {
            "cpg_steps": self._step_count,
            "amplitude_mean": round(amp_m, 4),
            "frequency_mean_hz": round(fr_m, 3),
            "reward_recent_mean": round(float(np.mean(rh)), 4) if rh else 0.0,
            "last_command_size": len(self._last_command),
            "last_intent_stride": round(float(self._last_motor_state.get("intent_stride", 0.5)), 4) if self._last_motor_state else 0.5,
            "cpg_com_lag": round(lag, 4),
            "cpg_weight": round(self.cpg_weight, 4),  # MOTOR_CORTEX
            "swing_l": round(float(sync.get("swing_l", 0.0)), 4),
            "swing_r": round(float(sync.get("swing_r", 0.0)), 4),
            "pitch_add": round(float(sync.get("pitch_add", 0.0)), 5),
        }