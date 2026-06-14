"""
cpg_locomotion.py — Phase A: Locomotion (CPG поверх агента) + Phase D: CPG Annealing.

Swing (бедро/колено):
  - Фаза осцилляторов CPG [0],[1] для лев/прав бедра; swing_factor = max(0, sin(phi))
  - В swing добавляется подъём бедра и сгиб колена (RKK_CPG_SWING_HIP_LIFT / KNEE_FLEX),
    чтобы не компенсировать только наклоном торса назад.
  - walk_gate учитывает intent_stop_recover и низкий com_z (при stride≈0.5 иначе gate=0).
  - Двусторонний tuck бёдер/коленей: RKK_CPG_RECOVERY_HIP_TUCK, RKK_CPG_RECOVERY_KNEE_FLEX_EXTRA.
  - Связка «наклон вперёд в интентах» → бёдра: RKK_CPG_TORSO_HIP_COEFF, RKK_CPG_TORSO_KNEE_COEFF (масштаб по low_z/recover/torso).
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
        self.register_buffer("epsilon_amp", torch.zeros(n_oscillators, device=dev))
        self.register_buffer("epsilon_freq", torch.zeros(n_oscillators, device=dev))
        self.register_buffer("epsilon_phase", torch.zeros(n_oscillators, n_oscillators, device=dev))

        self.to(dev)

    @torch.no_grad()
    def resample_perturbations(self, sigma_amp: float = 0.05, sigma_freq: float = 0.05, sigma_phase: float = 0.05) -> None:
        self.epsilon_amp.normal_(0.0, sigma_amp)
        self.epsilon_freq.normal_(0.0, sigma_freq)
        self.epsilon_phase.normal_(0.0, sigma_phase)

    @torch.no_grad()
    def clear_perturbations(self) -> None:
        self.epsilon_amp.zero_()
        self.epsilon_freq.zero_()
        self.epsilon_phase.zero_()

    @torch.no_grad()
    def step(
        self,
        dt: float = 0.05,
        external_command: torch.Tensor | None = None,
        *,
        coupling_gain: float = 0.1,
    ) -> torch.Tensor:
        freq = torch.sigmoid(self.frequency + self.epsilon_freq) * 2.0 + 0.3
        amp = torch.sigmoid(self.amplitude + self.epsilon_amp) * 0.6

        if external_command is not None:
            ec = external_command[: self.n]
            amp = amp * (0.5 + 0.5 * torch.sigmoid(ec))

        two_pi = 2.0 * math.pi
        p = self._phase + two_pi * freq * float(dt)
        self._phase.copy_(torch.remainder(p, two_pi))

        diff = self._phase.unsqueeze(1) - self._phase.unsqueeze(0) - (self.phase_bias + self.epsilon_phase)
        coupling = torch.sin(diff).sum(dim=1) * float(max(0.02, coupling_gain))
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
            self.cpg.amplitude.data[:] = 0.0  # Increased from -1.5 so it can actually move

        self.optim = torch.optim.Adam(self.cpg.parameters(), lr=1e-2)
        self._step_count = 0
        self._last_com_x: float = 0.5
        self._last_com_x_vel: float = 0.0
        self._last_com_z: float = 0.5
        self._reward_history: list[float] = []
        self._last_command: dict[str, float] = {}
        self._last_motor_state: dict[str, float] = {}
        self._last_cpg_sync: dict[str, float] = {}
        self._com_x_prev_step: float | None = None
        self._reward_baseline: float = 0.5
        self.cpg.resample_perturbations()

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
        stride_raw = float(self._node(agent_nodes, "intent_stride"))
        sup_l_val = float(self._node(agent_nodes, "intent_support_left"))
        sup_r_val = float(self._node(agent_nodes, "intent_support_right"))

        phi_l_pre = float(self.cpg._phase[0].item())
        phi_r_pre = float(self.cpg._phase[1].item())

        def _swing_factor(phi: float) -> float:
            return float(max(0.0, math.sin(phi)))

        swing_l_pre = _swing_factor(phi_l_pre)
        swing_r_pre = _swing_factor(phi_r_pre)
        if stride_raw >= 0.54:
            try:
                from engine.locomote_gait import alternating_support_from_swings

                alt_l, alt_r = alternating_support_from_swings(swing_l_pre, swing_r_pre)
                blend = float(os.environ.get("RKK_CPG_SUPPORT_BLEND", "0.85"))
                sup_l_val = float(
                    np.clip((1.0 - blend) * sup_l_val + blend * alt_l, 0.22, 0.82)
                )
                sup_r_val = float(
                    np.clip((1.0 - blend) * sup_r_val + blend * alt_r, 0.22, 0.82)
                )
                self._last_motor_state["intent_support_left"] = sup_l_val
                self._last_motor_state["intent_support_right"] = sup_r_val
            except Exception:
                pass

        sup_l = sup_l_val - 0.5
        sup_r = sup_r_val - 0.5
        recover = float(self._node(agent_nodes, "intent_stop_recover") - 0.5)
        energy = float(np.clip(self._node(agent_nodes, "self_energy"), 0.0, 1.0))
        com_x = float(self._node(agent_nodes, "com_x"))
        com_y = float(self._node(agent_nodes, "com_y"))
        com_z = float(self._node(agent_nodes, "com_z"))
        fwd_pos = com_y if abs(com_y - 0.5) >= abs(com_x - 0.5) else com_x
        self._last_com_x_vel = fwd_pos - float(getattr(self, "_last_com_x", fwd_pos))
        self._last_com_x = fwd_pos
        self._last_com_z = com_z

        # === FORWARD LEAN FIX ===
        # При stride > 0 агент должен лидировать корпусом — иначе падает назад
        # com_x: нормализованная позиция ≈ 0.5 = center. >0.5 = forward of stance foot.
        # Мы хотим com_x чуть впереди опорной стопы при ходьбе.
        com_lag = float(np.clip(0.48 - com_x, 0.0, 0.15))  # >0 если CoM отстаёт
        stride_n = max(0.0, stride)  # только positive stride (forward)
        recover_raw = float(self._node(agent_nodes, "intent_stop_recover"))
        recover_n = float(np.clip((recover_raw - 0.5) * 2.0, 0.0, 1.0))
        low_z = float(0.0)
        if com_z < 0.52:
            low_z = float(np.clip((0.52 - com_z) / max(0.22, 1e-6), 0.0, 1.0))
        
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

        self._last_motor_state["intent_stride"] = float(np.clip(stride_raw, 0.05, 0.95))
        coupling_raw = float(self._node(agent_nodes, "intent_gait_coupling"))
        self._last_motor_state["intent_gait_coupling"] = float(
            np.clip(coupling_raw, 0.05, 0.95)
        )

        # Epistemic exploration: boost stride slightly when WM uncertainty is high
        try:
            ep_bonus = float(os.environ.get("RKK_CPG_EPISTEMIC_STRIDE", "0.04"))
            meta_pe = float(agent_nodes.get("meta_prediction_error", 0.0) or 0.0)
            if meta_pe <= 0.0:
                meta_pe = float(agent_nodes.get("self_attention", 0.5)) * 0.2
            if ep_bonus > 0 and meta_pe > 0.12 and stride_raw < 0.68:
                boosted = float(np.clip(stride_raw + ep_bonus * min(meta_pe, 0.35), 0.05, 0.95))
                self._last_motor_state["intent_stride"] = boosted
                stride = boosted - 0.5
                stride_n = max(0.0, stride)
        except Exception:
            pass

        posture = float(self._node(agent_nodes, "posture_stability"))
        static_penalty_applied = False
        try:
            static_on = os.environ.get("RKK_CPG_STATIC_PENALTY", "1").strip().lower() not in (
                "0",
                "false",
                "no",
                "off",
            )
            vel_thr = float(os.environ.get("RKK_CPG_STATIC_VEL_THRESH", "0.006"))
            ps_thr = float(os.environ.get("RKK_STABLE_LOCOMOTE_PS", "0.90"))
            if static_on and stride_raw >= 0.58 and posture >= ps_thr - 0.02:
                vel = abs(float(self._last_com_x_vel))
                if vel < vel_thr:
                    nudge = float(os.environ.get("RKK_CPG_STATIC_STRIDE_NUDGE", "0.07"))
                    boosted = float(np.clip(stride_raw + nudge, 0.05, 0.95))
                    self._last_motor_state["intent_stride"] = boosted
                    stride_raw = boosted
                    stride = boosted - 0.5
                    stride_n = max(0.0, stride)
                    static_penalty_applied = True
        except Exception:
            pass

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

        if static_penalty_applied:
            try:
                cmd_boost = float(os.environ.get("RKK_CPG_STATIC_CMD_BOOST", "0.18"))
                cmd = cmd * (1.0 + cmd_boost)
            except Exception:
                pass

        if stride_raw >= 0.54:
            with torch.no_grad():
                self.cpg.phase_bias[0, 1] = math.pi
                self.cpg.phase_bias[1, 0] = -math.pi
        coupling_gain = 0.1
        try:
            c_raw = float(self._node(agent_nodes, "intent_gait_coupling"))
            coupling_gain = float(np.clip((c_raw - 0.5) * 0.35 + 0.06, 0.05, 0.14))
        except Exception:
            coupling_gain = 0.1

        cpg_out = self.cpg.step(
            dt=dt,
            external_command=cmd * (0.7 + 0.3 * energy) * drive_damp,
            coupling_gain=coupling_gain,
        )

        # Торс синхронизация с CPG: pitch_add > 0 = наклон вперёд
        gscale = float(
            np.clip(max(stride_n, 0.42 * recover_n + 0.48 * low_z) * 1.8, 0.0, 1.0)
        )
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

        locomote_walk = stride_raw >= 0.58 and posture >= float(
            os.environ.get("RKK_STABLE_LOCOMOTE_PS", "0.88")
        ) - 0.02
        macro_loc = float(self._node(agent_nodes, "executive_macro_hint")) >= 0.95
        pe_fwd = float(agent_nodes.get("hai_pe_fwd_ema", 0.0) or 0.0)
        if locomote_walk or macro_loc:
            try:
                pe_thr = float(os.environ.get("RKK_CPG_FORWARD_PITCH_PE_THR", "-0.3"))
            except ValueError:
                pe_thr = -0.3
            if pe_fwd < pe_thr or abs(self._last_com_x_vel) < float(
                os.environ.get("RKK_CPG_STATIC_VEL_THRESH", "0.006")
            ):
                offset = float(os.environ.get("RKK_CPG_FORWARD_PITCH_OFFSET", "0.07"))
                pitch_add = float(pitch_add) + offset
                if pitch_add < 0.0:
                    pitch_add = offset * 0.85
                torso_forward = float(
                    np.clip(
                        max(torso_forward, float(self._node(agent_nodes, "intent_torso_forward"))),
                        0.52,
                        0.96,
                    )
                )
                self._last_motor_state["intent_torso_forward"] = float(torso_forward)

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
        walk_blend = float(
            np.clip(max(stride_n, 0.5 * recover_n + 0.45 * low_z), 0.0, 1.0)
        )
        walk_gate = float(np.clip(walk_blend * (0.35 + 0.65 * gscale), 0.0, 1.0))
        try:
            suppress = float(os.environ.get("RKK_CPG_RECOVERY_WALK_SUPPRESS", "0.85"))
        except ValueError:
            suppress = 0.85
        suppress = float(np.clip(suppress, 0.0, 1.0))
        rec_gate = float(np.clip(max(recover_n, low_z * 0.9), 0.0, 1.0))
        if rec_gate > 0.35:
            walk_gate *= float(1.0 - suppress * rec_gate)

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

        targets: dict[str, float] = {
            "lhip":   float(np.clip(0.50 + 0.18 * (float(cpg_out[0].item())*2 - 1), 0.05, 0.95)),
            "rhip":   float(np.clip(0.50 + 0.18 * (float(cpg_out[1].item())*2 - 1), 0.05, 0.95)),
            "lknee":  float(np.clip(0.50 + 0.15 * (float(cpg_out[2].item())*2 - 1), 0.05, 0.95)),
            "rknee":  float(np.clip(0.50 + 0.15 * (float(cpg_out[3].item())*2 - 1), 0.05, 0.95)),
            "lankle": float(np.clip(0.50 + 0.10 * (float(cpg_out[4].item())*2 - 1), 0.05, 0.95)),
            "rankle": float(np.clip(0.50 + 0.10 * (float(cpg_out[5].item())*2 - 1), 0.05, 0.95)),
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

        if static_penalty_applied:
            try:
                push = float(os.environ.get("RKK_CPG_ANKLE_PUSHOFF", "0.05"))
                targets["lankle"] = float(
                    np.clip(targets["lankle"] + push * max(swing_r, 0.35), 0.05, 0.95)
                )
                targets["rankle"] = float(
                    np.clip(targets["rankle"] + push * max(swing_l, 0.35), 0.05, 0.95)
                )
                hip_push = float(os.environ.get("RKK_CPG_HIP_PUSHOFF", "0.03"))
                targets["rhip"] = float(np.clip(targets["rhip"] + hip_push * stride_n, 0.05, 0.95))
            except Exception:
                pass

        # Вставание со спины: оба бедра к корпусу (stride≈0.5 давал walk_gate≈0 — только торс).
        torso_f = float(self._node(agent_nodes, "intent_torso_forward"))
        tuck_gate = float(
            np.clip(
                0.5 * recover_n
                + 0.48 * low_z
                + 0.42 * max(0.0, torso_f - 0.50),
                0.0,
                1.0,
            )
        )
        hip_tuck = _env_cpg_swing_float("RKK_CPG_RECOVERY_HIP_TUCK", "0.04")
        knee_tuck_x = _env_cpg_swing_float("RKK_CPG_RECOVERY_KNEE_FLEX_EXTRA", "0.02")
        targets["lhip"] = float(np.clip(targets["lhip"] + hip_tuck * tuck_gate, 0.05, 0.95))
        targets["rhip"] = float(np.clip(targets["rhip"] + hip_tuck * tuck_gate, 0.05, 0.95))
        targets["lknee"] = float(np.clip(targets["lknee"] - knee_tuck_x * tuck_gate, 0.05, 0.95))
        targets["rknee"] = float(np.clip(targets["rknee"] - knee_tuck_x * tuck_gate, 0.05, 0.95))

        # Наклон «вперёд» в интентах → прижать бёдра (не только spine_pitch).
        try:
            coeff_th = float(os.environ.get("RKK_CPG_TORSO_HIP_COEFF", "0.17"))
        except ValueError:
            coeff_th = 0.17
        try:
            coeff_tk = float(os.environ.get("RKK_CPG_TORSO_KNEE_COEFF", "0.10"))
        except ValueError:
            coeff_tk = 0.10
        coeff_th = float(np.clip(coeff_th, 0.0, 0.45))
        coeff_tk = float(np.clip(coeff_tk, 0.0, 0.35))
        raw_torso = float(self._node(agent_nodes, "intent_torso_forward"))
        torso_excess = float(max(0.0, raw_torso - 0.48))
        torso_hip_scale = float(
            np.clip(
                0.52 * low_z + 0.48 * recover_n + 0.38 * max(0.0, raw_torso - 0.52),
                0.0,
                1.0,
            )
        )
        hip_from_torso = coeff_th * torso_excess * torso_hip_scale
        knee_from_torso = coeff_tk * torso_excess * torso_hip_scale
        targets["lhip"] = float(np.clip(targets["lhip"] + hip_from_torso, 0.05, 0.95))
        targets["rhip"] = float(np.clip(targets["rhip"] + hip_from_torso, 0.05, 0.95))
        targets["lknee"] = float(np.clip(targets["lknee"] - knee_from_torso, 0.05, 0.95))
        targets["rknee"] = float(np.clip(targets["rknee"] - knee_from_torso, 0.05, 0.95))

        try:
            from engine.genome.priors import (
                genome_walk_cpg_boost,
                genome_walk_enabled,
                genome_walk_joint_amp_scale,
            )

            if genome_walk_enabled():
                stride_raw = float(self._node(agent_nodes, "intent_stride"))
                posture = float(self._node(agent_nodes, "posture_stability"))
                if stride_raw > 0.53 and posture > 0.50:
                    boost = genome_walk_cpg_boost()
                    amp = genome_walk_joint_amp_scale()
                    scale = float(boost * amp / max(1.0, boost))
                    for leg in ("lhip", "rhip", "lknee", "rknee", "lankle", "rankle"):
                        targets[leg] = float(
                            np.clip(0.5 + (targets[leg] - 0.5) * scale, 0.05, 0.95)
                        )
        except Exception:
            pass

        self._step_count += 1
        self._last_command = dict(targets)
        return targets

    def train_cpg_from_intrinsic_history(self) -> None:
        """
        Обучение CPG только из _reward_history (заполняется IntrinsicObjective).
        Вызывается после agent.step; см. engine.intristic_objective.
        """
        if not self._reward_history:
            return

        # Keep history bounded to avoid memory leak
        if len(self._reward_history) > 128:
            self._reward_history = self._reward_history[-128:]

        r_total = self._reward_history[-1]
        com_x_vel = float(getattr(self, "_last_com_x_vel", 0.0))
        try:
            fwd_scale = float(os.environ.get("RKK_CPG_FWD_BONUS", "0.4"))
        except ValueError:
            fwd_scale = 0.4
        fwd = fwd_scale * float(np.clip(com_x_vel - 0.01, 0.0, 1.0))
        r_total = r_total + fwd

        baseline = getattr(self, "_reward_baseline", 0.5)
        diff = r_total - baseline
        self._reward_baseline = 0.98 * baseline + 0.02 * r_total

        self.optim.zero_grad()
        loss = - diff * (
            (self.cpg.epsilon_amp * self.cpg.amplitude).sum() +
            (self.cpg.epsilon_freq * self.cpg.frequency).sum() +
            (self.cpg.epsilon_phase * self.cpg.phase_bias).sum()
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.cpg.parameters(), 0.3)
        self.optim.step()

        # Resample perturbations for the next step exploration
        self.cpg.resample_perturbations()

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
            "com_x_vel": round(float(getattr(self, "_last_com_x_vel", 0.0)), 5),
            "last_command_size": len(self._last_command),
            "last_intent_stride": round(float(self._last_motor_state.get("intent_stride", 0.5)), 4) if self._last_motor_state else 0.5,
            "cpg_com_lag": round(lag, 4),
            "cpg_weight": round(self.cpg_weight, 4),  # MOTOR_CORTEX
            "swing_l": round(float(sync.get("swing_l", 0.0)), 4),
            "swing_r": round(float(sync.get("swing_r", 0.0)), 4),
            "pitch_add": round(float(sync.get("pitch_add", 0.0)), 5),
        }