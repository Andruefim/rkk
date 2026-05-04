"""Публичная среда гуманоида (observe / intervene / CPG hooks)."""
from __future__ import annotations

import os

import numpy as np
import torch

from engine.features.humanoid.constants import (
    ARM_VARS,
    FALLEN_Z,
    FIXED_BASE_VARS,
    HEAD_VARS,
    HUMANOID_URDF_STAND_EULER,
    INTERO_VARS,
    LEG_VARS,
    MOTOR_INTENT_DEFAULTS,
    MOTOR_INTENT_VARS,
    SELF_VARS,
    SPINE_VARS,
    STAND_Z,
    VAR_NAMES,
    _RANGES,
)
from engine.features.humanoid.deps import PYBULLET_AVAILABLE
from engine.features.humanoid.fallback import _FallbackHumanoid
from engine.features.humanoid.pybullet_humanoid import _PyBulletHumanoid

# ─── EnvironmentHumanoid ─────────────────────────────────────────────────────
class EnvironmentHumanoid:
    """
    Среда гуманоида для Singleton AGI.

    fixed_root=True: база зафиксирована, variable_ids = FIXED_BASE_VARS (вкл. self_*).
    set_fixed_root(bool): переключение в runtime.
    """

    PRESET = "humanoid"

    def __init__(
        self,
        device: torch.device | None = None,
        steps_per_do: int = 10,
        fixed_root: bool = False,
    ):
        self.device = device or torch.device("cpu")
        try:
            spd = int(os.environ.get("RKK_STEPS_PER_DO", str(steps_per_do)))
        except ValueError:
            spd = steps_per_do
        self.steps_per_do = max(1, min(int(spd), 128))
        self.preset       = self.PRESET
        self.n_interventions = 0
        self._fixed_root  = fixed_root

        if PYBULLET_AVAILABLE:
            self._sim     = _PyBulletHumanoid(fixed_root=fixed_root)
            self._backend = "pybullet"
        else:
            self._sim     = _FallbackHumanoid(fixed_root=fixed_root)
            self._backend = "fallback"

        mode_label = "fixed_root" if fixed_root else "full"
        var_count  = len(FIXED_BASE_VARS if fixed_root else VAR_NAMES)
        print(f"[HumanoidEnv] backend={self._backend}, mode={mode_label}, vars={var_count}")
        # When True, CPG owns leg joints; _apply_motor_intents only sets upper body.
        self.cpg_owns_legs: bool = False
        # Самомодель: значения держим в среде, observe() мержит с физикой; intervene(self_*) не трогает суставы.
        self._self_state: dict[str, float] = {k: 0.5 for k in SELF_VARS}
        self._motor_state: dict[str, float] = {
            k: float(MOTOR_INTENT_DEFAULTS.get(k, 0.5)) for k in MOTOR_INTENT_VARS
        }
        # ── Interoception (Phase 1: Embodied Cognition) ──
        self._intero_state: dict[str, float] = {
            "intero_energy": 1.0,
            "intero_stress": 0.0,
        }
        self._intero_control_lost: bool = False
        self._prev_raw_obs: dict[str, float] | None = None

    # ── Fixed root switch ─────────────────────────────────────────────────────
    @property
    def fixed_root(self) -> bool:
        return self._fixed_root

    def set_fixed_root(self, enabled: bool) -> None:
        """
        Переключить fixed_root mode без пересоздания среды.
        Simulation.enable/disable_fixed_root() вызывает это и затем
        rebind_variables() на агенте.
        """
        if enabled == self._fixed_root:
            return
        self._fixed_root = enabled
        if isinstance(self._sim, _PyBulletHumanoid):
            if enabled:
                self._sim.enable_fixed_root()
            else:
                self._sim.disable_fixed_root()
        else:
            self._sim.fixed_root = enabled

    # ── Нормализация ──────────────────────────────────────────────────────────
    def _norm(self, key: str, val: float) -> float:
        lo, hi = _RANGES.get(key, (-1.0, 1.0))
        return float(np.clip((val - lo) / (hi - lo), 0.05, 0.95))

    def _denorm(self, key: str, val: float) -> float:
        lo, hi = _RANGES.get(key, (-1.0, 1.0))
        return float(val * (hi - lo) + lo)

    # ── Observe ───────────────────────────────────────────────────────────────
    def observe(self) -> dict[str, float]:
        raw = self._sim.get_state()
        active = set(FIXED_BASE_VARS if self._fixed_root else VAR_NAMES)
        out = {k: self._norm(k, v) for k, v in raw.items() if k in active}
        for mk in MOTOR_INTENT_VARS:
            if mk in active:
                out[mk] = float(np.clip(self._motor_state.get(mk, 0.5), 0.05, 0.95))
        out.update(self._derived_motor_observables(raw))
        for sk in SELF_VARS:
            if sk in active:
                out[sk] = float(np.clip(self._self_state.get(sk, 0.5), 0.05, 0.95))
        for ik in INTERO_VARS:
            if ik in active:
                out[ik] = float(np.clip(self._intero_state.get(ik, 0.5), 0.05, 0.95))
        return out

    @property
    def variables(self) -> dict[str, float]:
        return self.observe()

    @property
    def variable_ids(self) -> list[str]:
        return list(FIXED_BASE_VARS if self._fixed_root else VAR_NAMES)

    def _derived_motor_observables(self, raw: dict[str, float]) -> dict[str, float]:
        """Compute causal motor variables from current humanoid pose."""
        com_x = float(raw.get("com_x", 0.0))
        com_z = float(raw.get("com_z", STAND_Z))
        torso_roll = float(raw.get("torso_roll", 0.0))
        torso_pitch = float(raw.get("torso_pitch", 0.0))
        lhip = float(raw.get("lhip", 0.5))
        rhip = float(raw.get("rhip", 0.5))
        lknee = float(raw.get("lknee", 0.5))
        rknee = float(raw.get("rknee", 0.5))
        lankle = float(raw.get("lankle", 0.5))
        rankle = float(raw.get("rankle", 0.5))
        lsh = float(raw.get("lshoulder", 0.0))
        rsh = float(raw.get("rshoulder", 0.0))
        lel = float(raw.get("lelbow", 0.0))
        rel = float(raw.get("relbow", 0.0))
        sp_pitch = float(raw.get("spine_pitch", 0.0))
        sp_yaw = float(raw.get("spine_yaw", 0.0))
        lf = float(raw.get("lfoot_z", 0.05))
        rf = float(raw.get("rfoot_z", 0.05))
        support_l = float(np.clip(1.0 - lf / max(STAND_Z * 0.18, 1e-6), 0.0, 1.0))
        support_r = float(np.clip(1.0 - rf / max(STAND_Z * 0.18, 1e-6), 0.0, 1.0))
        gait_l = float(np.clip(0.5 + 0.5 * np.sin(3.2 * (lhip - 0.5) - 1.7 * (lknee - 0.5)), 0.0, 1.0))
        gait_r = float(np.clip(0.5 + 0.5 * np.sin(3.2 * (rhip - 0.5) - 1.7 * (rknee - 0.5)), 0.0, 1.0))
        # Lateral + forward CoM; extra term when a foot is loaded nudges the bias channel toward forefoot ZMP.
        load = max(support_l, support_r)
        fwd_zmp = 0.07 * load * float(np.clip((com_x - 0.40) / 0.38, 0.0, 1.0))
        support_bias = float(np.clip(0.5 + 0.45 * ((support_l - support_r) + 0.65 * com_x) + fwd_zmp, 0.0, 1.0))
        motor_drive_l = float(np.clip(np.mean([abs(lhip - 0.5), abs(lknee - 0.5), abs(lankle - 0.5)]) * 1.8, 0.0, 1.0))
        motor_drive_r = float(np.clip(np.mean([abs(rhip - 0.5), abs(rknee - 0.5), abs(rankle - 0.5)]) * 1.8, 0.0, 1.0))
        roll_from_upright = torso_roll - HUMANOID_URDF_STAND_EULER[0]
        lsh_neutral = 0.0
        rsh_neutral = 0.0
        joint_deviations = [
            abs(lsh - lsh_neutral) * 0.6, abs(rsh - rsh_neutral) * 0.6,
            abs(lel) * 0.8, abs(rel) * 0.8,
            abs(sp_pitch) * 2.0, abs(sp_yaw) * 2.0,
        ]
        joint_penalty = float(np.clip(np.mean(joint_deviations), 0.0, 1.0))
        posture_stability = float(np.clip(
            1.0
            - (abs(roll_from_upright) + abs(torso_pitch)) * 0.6
            - abs(com_z - STAND_Z) / max(STAND_Z, 0.01) * 0.4
            - joint_penalty * 0.3,
            0.0, 1.0,
        ))
        return {
            "gait_phase_l": gait_l,
            "gait_phase_r": gait_r,
            "foot_contact_l": support_l,
            "foot_contact_r": support_r,
            "support_bias": support_bias,
            "motor_drive_l": motor_drive_l,
            "motor_drive_r": motor_drive_r,
            "posture_stability": posture_stability,
        }

    def gt_edges(self) -> list[tuple[str, str, float]]:
        """Removed GT edges constraint to rely solely on self-supervised discovery."""
        return []

    # ── Interoception update ──────────────────────────────────────────────────
    # Constants for energy / stress dynamics
    _ENERGY_DRAIN_PER_INTENT = 0.004   # per unit of intent deviation from current pose
    _ENERGY_DRAIN_PER_JOINT  = 0.002   # per unit of joint deviation from neutral
    _ENERGY_RECOVERY         = 0.006   # passive recovery per physics step
    _ENERGY_LOSS_THRESHOLD   = 0.05    # below this: control lost
    _ENERGY_RECOVER_THRESHOLD = 0.20   # above this: control regained (hysteresis)
    _STRESS_LIMIT_ZONE       = 0.10    # norm-distance from [0.05, 0.95] edges
    _STRESS_IMPACT_GAIN      = 2.5     # multiplier for rapid state changes
    _STRESS_DECAY            = 0.015   # passive decay per step

    def _update_interoception(self) -> None:
        """
        Update intero_energy and intero_stress from current physics.

        Energy drains when motor intents deviate from current joint pose
        and when joints deviate from neutral (active movement costs energy).
        Energy passively recovers each step.

        Stress grows when joints approach anatomical limits and when
        rapid state changes (impacts) occur. Stress decays passively.

        When energy < threshold, agent loses motor control (hysteresis).
        """
        raw = self._sim.get_state()
        st = self._intero_state

        # ── Energy drain ──────────────────────────────────────────────────
        intent_cost = 0.0
        for mk in MOTOR_INTENT_VARS:
            intent_cost += abs(self._motor_state.get(mk, 0.5) - 0.5)

        joint_cost = 0.0
        all_joints = ARM_VARS + SPINE_VARS + HEAD_VARS
        if not self._fixed_root:
            all_joints = LEG_VARS + all_joints
        for jv in all_joints:
            if jv in raw:
                neutral = self._JOINT_NEUTRAL.get(jv, 0.5)
                joint_cost += abs(self._norm(jv, raw[jv]) - neutral)

        drain = (
            intent_cost * self._ENERGY_DRAIN_PER_INTENT
            + joint_cost * self._ENERGY_DRAIN_PER_JOINT
        )
        new_energy = st["intero_energy"] - drain + self._ENERGY_RECOVERY
        st["intero_energy"] = float(np.clip(new_energy, 0.0, 1.0))

        # ── Loss of control (hysteresis) ──────────────────────────────────
        if st["intero_energy"] <= self._ENERGY_LOSS_THRESHOLD:
            if not self._intero_control_lost:
                self._intero_control_lost = True
        elif st["intero_energy"] >= self._ENERGY_RECOVER_THRESHOLD:
            if self._intero_control_lost:
                self._intero_control_lost = False

        # ── Stress from joint limits ──────────────────────────────────────
        stress_delta = 0.0
        limit_lo = 0.05 + self._STRESS_LIMIT_ZONE
        limit_hi = 0.95 - self._STRESS_LIMIT_ZONE
        for jv in (LEG_VARS + ARM_VARS + SPINE_VARS + HEAD_VARS):
            if jv not in raw:
                continue
            nv = self._norm(jv, raw[jv])
            if nv < limit_lo:
                stress_delta += (limit_lo - nv) * 0.6
            elif nv > limit_hi:
                stress_delta += (nv - limit_hi) * 0.6

        # ── Stress from impacts (rapid state changes) ─────────────────────
        if self._prev_raw_obs is not None:
            for iv in ("com_z", "torso_roll", "torso_pitch"):
                prev_v = self._prev_raw_obs.get(iv)
                curr_v = raw.get(iv)
                if prev_v is not None and curr_v is not None:
                    delta = abs(float(curr_v) - float(prev_v))
                    if delta > 0.04:
                        stress_delta += delta * self._STRESS_IMPACT_GAIN

        self._prev_raw_obs = dict(raw)

        new_stress = st["intero_stress"] + stress_delta - self._STRESS_DECAY
        st["intero_stress"] = float(np.clip(new_stress, 0.0, 1.0))

    # ── do() ─────────────────────────────────────────────────────────────────
    _JOINT_NEUTRAL: dict[str, float] = {
        "lshoulder": 0.50, "rshoulder": 0.50,
        "lelbow": 0.50, "relbow": 0.50,
        "spine_pitch": 0.50, "spine_yaw": 0.50,
        "neck_pitch": 0.50, "neck_yaw": 0.50,
        "lhip": 0.50, "rhip": 0.50,
        "lknee": 0.45, "rknee": 0.45,
        "lankle": 0.50, "rankle": 0.50,
    }
    _JOINT_COMFORT_RANGE: dict[str, float] = {
        "lshoulder": 0.10, "rshoulder": 0.10,
        "lelbow": 0.08, "relbow": 0.08,
        "spine_pitch": 0.05, "spine_yaw": 0.05,
        "neck_pitch": 0.06, "neck_yaw": 0.06,
        "lhip": 0.12, "rhip": 0.12,
        "lknee": 0.12, "rknee": 0.12,
        "lankle": 0.10, "rankle": 0.10,
    }

    @classmethod
    def _comfort_zone(cls, var: str) -> tuple[float, float]:
        # Убраны жесткие лимиты _JOINT_COMFORT_RANGE для открытого обучения моторике.
        return 0.05, 0.95

    def intervene(self, variable: str, value: float, *, count_intervention: bool = True) -> dict[str, float]:
        if count_intervention:
            self.n_interventions += 1

        if variable in INTERO_VARS:
            # Intero vars are read-only sensors; ignore external writes.
            self._sim.step(self.steps_per_do)
            self._update_interoception()
            return self.observe()

        if variable in SELF_VARS:
            self._self_state[variable] = float(np.clip(value, 0.05, 0.95))
            self._sim.step(self.steps_per_do)
            self._update_interoception()
            return self.observe()

        if variable in MOTOR_INTENT_VARS:
            self._motor_state[variable] = float(np.clip(value, 0.05, 0.95))
            if not self._intero_control_lost:
                self._apply_motor_intents()
            self._sim.step(self.steps_per_do)
            self._update_interoception()
            return self.observe()

        if self._fixed_root:
            controllable = ARM_VARS + SPINE_VARS + HEAD_VARS
        else:
            controllable = LEG_VARS + ARM_VARS + SPINE_VARS + HEAD_VARS

        if variable in controllable and not self._intero_control_lost:
            lo, hi = self._comfort_zone(variable)
            clamped = float(np.clip(value, lo, hi))
            self._sim.set_joint(variable, clamped)

        self._sim.step(self.steps_per_do)
        self._update_interoception()
        return self.observe()

    def intervene_burst(
        self,
        pairs: list[tuple[str, float]],
        *,
        count_intervention: bool = False,
    ) -> dict[str, float]:
        """
        Несколько do() за один физический settle-step: важно для согласованных intent/joint.
        Порядок: self_* и motor intent в _motor_state → один _apply_motor_intents → явные set_joint.
        """
        if not pairs:
            return self.observe()
        if count_intervention:
            self.n_interventions += 1
        if self._fixed_root:
            controllable = ARM_VARS + SPINE_VARS + HEAD_VARS
        else:
            controllable = LEG_VARS + ARM_VARS + SPINE_VARS + HEAD_VARS
        touched_intent = False
        joints_after: list[tuple[str, float]] = []
        for variable, value in pairs:
            v = float(np.clip(value, 0.05, 0.95))
            if variable in INTERO_VARS:
                continue  # read-only sensors
            elif variable in SELF_VARS:
                self._self_state[variable] = v
            elif variable in MOTOR_INTENT_VARS:
                self._motor_state[variable] = v
                touched_intent = True
            elif variable in controllable:
                lo, hi = self._comfort_zone(variable)
                joints_after.append((variable, float(np.clip(v, lo, hi))))
        if not self._intero_control_lost:
            if touched_intent:
                self._apply_motor_intents()
            for variable, v in joints_after:
                self._sim.set_joint(variable, v)
        self._sim.step(self.steps_per_do)
        self._update_interoception()
        return self.observe()

    def _apply_upper_body_from_intents(self, *, cpg_sync: dict[str, float] | None = None) -> None:
        """
        Торс и плечи из intent_* — используется при CPG на ногах и в apply_cpg_leg_targets.
        cpg_sync: опционально подмешивает pitch_add от ритма ног.
        """
        ms = self._motor_state
        torso = float(ms.get("intent_torso_forward", 0.5))
        arm_cb = float(ms.get("intent_arm_counterbalance", 0.5))
        rr = float(ms.get("intent_reach_right", 0.5))
        rl = float(ms.get("intent_reach_left", 0.5))
        grasp = float(ms.get("intent_grasp", 0.5))
        look = float(ms.get("intent_look_at", 0.5))
        lean = float(ms.get("intent_lean_forward", 0.5))
        wave = float(ms.get("intent_wave", 0.5))

        pitch = float(np.clip(0.5 + (torso - 0.5) * 0.55 + (lean - 0.5) * 0.28, 0.05, 0.95))
        if cpg_sync:
            pitch = float(
                np.clip(pitch + 0.35 * float(cpg_sync.get("pitch_add", 0.0)), 0.05, 0.95)
            )
        self._sim.set_joint("spine_pitch", pitch)

        l_sh = float(
            np.clip(
                0.5
                - (arm_cb - 0.5) * 0.65
                + (rl - 0.5) * 0.88
                - (wave - 0.5) * 0.12,
                0.05,
                0.95,
            )
        )
        r_sh = float(
            np.clip(
                0.5
                + (arm_cb - 0.5) * 0.65
                + (rr - 0.5) * 0.88
                + (wave - 0.5) * 0.12,
                0.05,
                0.95,
            )
        )
        self._sim.set_joint("lshoulder", l_sh)
        self._sim.set_joint("rshoulder", r_sh)

        g = float(np.clip((grasp - 0.5) * 1.1, -0.35, 0.35))
        self._sim.set_joint("lelbow", float(np.clip(0.5 - g, 0.05, 0.95)))
        self._sim.set_joint("relbow", float(np.clip(0.5 - g, 0.05, 0.95)))

        self._sim.set_joint(
            "neck_yaw",
            float(np.clip(0.5 + (look - 0.5) * 0.75 + (rr - 0.5) * 0.15 - (rl - 0.5) * 0.15, 0.05, 0.95)),
        )
        self._sim.set_joint(
            "neck_pitch",
            float(
                np.clip(
                    0.5
                    + abs(look - 0.5) * 0.22
                    + (lean - 0.5) * 0.2
                    - 0.08 * max(0.0, rr - 0.5)
                    - 0.08 * max(0.0, rl - 0.5),
                    0.05,
                    0.95,
                )
            ),
        )

    def _apply_motor_intents(self) -> None:
        """
        Отображение intent_* на цели суставов (observe всё ещё из фактической физики).
        При cpg_owns_legs ноги задаёт CPG — здесь только верх тела.
        """
        if self._intero_control_lost:
            return
        ms = self._motor_state
        sup_l = float(ms.get("intent_support_left", 0.5))
        sup_r = float(ms.get("intent_support_right", 0.5))
        recover = float(ms.get("intent_stop_recover", 0.5))
        stride = float(ms.get("intent_stride", 0.5))
        torso = float(ms.get("intent_torso_forward", 0.5))
        arm_cb = float(ms.get("intent_arm_counterbalance", 0.5))

        if self._fixed_root:
            self._sim.set_joint(
                "spine_pitch",
                float(np.clip(0.5 + (torso - 0.5) * 0.5, 0.05, 0.95)),
            )
            self._sim.set_joint(
                "lshoulder",
                float(np.clip(0.5 - (arm_cb - 0.5) * 0.6, 0.05, 0.95)),
            )
            self._sim.set_joint(
                "rshoulder",
                float(np.clip(0.5 + (arm_cb - 0.5) * 0.6, 0.05, 0.95)),
            )
            return

        if self.cpg_owns_legs:
            self._apply_upper_body_from_intents(cpg_sync=None)
            return

        knee = float(
            np.clip(
                0.45 - (sup_l + sup_r - 1.0) * 0.20 + (recover - 0.5) * 0.45,
                0.05,
                0.95,
            )
        )
        self._sim.set_joint("lknee", knee)
        self._sim.set_joint("rknee", knee)

        hip_base = float(np.clip(0.5 - (sup_l + sup_r - 1.0) * 0.15, 0.05, 0.95))
        self._sim.set_joint(
            "lhip",
            float(np.clip(hip_base - (stride - 0.5) * 0.3, 0.05, 0.95)),
        )
        self._sim.set_joint(
            "rhip",
            float(np.clip(hip_base + (stride - 0.5) * 0.3, 0.05, 0.95)),
        )

        ankle = float(np.clip(0.5 + (sup_l + sup_r - 1.0) * 0.12, 0.05, 0.95))
        self._sim.set_joint("lankle", ankle)
        self._sim.set_joint("rankle", ankle)

        self._sim.set_joint(
            "spine_pitch",
            float(np.clip(0.5 + (torso - 0.5) * 0.55, 0.05, 0.95)),
        )
        self._sim.set_joint(
            "lshoulder",
            float(np.clip(0.5 - (arm_cb - 0.5) * 0.65, 0.05, 0.95)),
        )
        self._sim.set_joint(
            "rshoulder",
            float(np.clip(0.5 + (arm_cb - 0.5) * 0.65, 0.05, 0.95)),
        )

    def apply_motor_intent_residuals(self, residuals: dict[str, float]) -> None:
        """
        Add small deltas to motor intents without an extra PyBullet step (L0 reflex).
        Used by hierarchical active inference: PE → nudge stance / gait coupling only.
        """
        if self._intero_control_lost:
            return
        for k, delta in (residuals or {}).items():
            if k not in MOTOR_INTENT_VARS:
                continue
            prev = float(self._motor_state.get(k, 0.5))
            self._motor_state[k] = float(
                np.clip(prev + float(delta), 0.05, 0.95)
            )

    def set_joint_targets(self, targets: dict[str, float]) -> None:
        """
        Execute raw joint targets (for MotorPrimitiveLibrary).
        Respects fixed root bounds, leg ownership, and interoceptive control.
        """
        if self._intero_control_lost:
            return
        for name, val in targets.items():
            if self._fixed_root and name in LEG_VARS:
                continue
            if getattr(self, "cpg_owns_legs", False) and name in LEG_VARS:
                continue
            lo, hi = self._comfort_zone(name)
            clamped = float(np.clip(val, lo, hi))
            self._sim.set_joint(name, clamped)


    def apply_cpg_leg_targets(
        self,
        targets: dict[str, float],
        *,
        cpg_sync: dict[str, float] | None = None,
    ) -> None:
        """
        Phase A locomotion: низкоуровневые цели на ноги без увеличения n_interventions.
        Also marks cpg_owns_legs=True so _apply_motor_intents doesn't fight CPG.
        cpg_sync: фаза CPG + отставание CoM — согласует корпус с ритмом ног.
        """
        if self._fixed_root:
            return
        self.cpg_owns_legs = True
        try:
            n_sub = int(os.environ.get("RKK_CPG_PHYS_SUBSTEPS", "0"))
        except ValueError:
            n_sub = 0
        if n_sub <= 0:
            n_sub = max(1, self.steps_per_do // 2)
        n_sub = min(max(n_sub, 1), 32)
        if not self._intero_control_lost:
            self._apply_upper_body_from_intents(cpg_sync=cpg_sync)
            for name, val in targets.items():
                if name in LEG_VARS:
                    self._sim.set_joint(name, float(np.clip(val, 0.05, 0.95)))
            for name, val in targets.items():
                if name in (ARM_VARS + SPINE_VARS + HEAD_VARS):
                    self._sim.set_joint(name, float(np.clip(val, 0.05, 0.95)))
        self._sim.step(n_sub)
        self._update_interoception()

    def update_self_feedback(
        self,
        variable: str,
        intended_norm: float,
        observed: dict[str, float],
        predicted: dict[str, float] | None = None,
        prediction_error_phys: float = 0.0,
    ) -> None:
        """
        Петля самомодели: «хотел / сделал / получил» и расхождение модели с миром → коррекция self_*.
        При активной цели (self_goal_active): наблюдаемый target_dist подстраивает self_goal_target_dist.
        """
        if variable in SELF_VARS:
            return
        try:
            lr = float(os.environ.get("RKK_SELF_FEEDBACK_LR", "0.18"))
        except ValueError:
            lr = 0.18
        lr = max(0.0, min(0.5, lr))
        st = self._self_state
        pred = predicted or {}

        if variable in ("lshoulder", "lelbow"):
            actual = float(observed.get(variable, intended_norm))
            gap = actual - intended_norm
            st["self_intention_larm"] = float(
                np.clip(st["self_intention_larm"] + lr * gap, 0.05, 0.95)
            )
        elif variable in ("rshoulder", "relbow"):
            actual = float(observed.get(variable, intended_norm))
            gap = actual - intended_norm
            st["self_intention_rarm"] = float(
                np.clip(st["self_intention_rarm"] + lr * gap, 0.05, 0.95)
            )

        if variable in SPINE_VARS or variable in HEAD_VARS:
            actual = float(observed.get(variable, intended_norm))
            gap = actual - intended_norm
            st["self_attention"] = float(
                np.clip(st["self_attention"] + 0.1 * lr * gap, 0.05, 0.95)
            )

        # Self-goal refinement: фактический target_dist → подстроить порог цели.
        if "target_dist" in observed and float(st.get("self_goal_active", 0.0)) > 0.5:
            actual_td = float(observed["target_dist"])
            gap_td = actual_td - float(st["self_goal_target_dist"])
            st["self_goal_target_dist"] = float(
                np.clip(st["self_goal_target_dist"] + 0.08 * gap_td, 0.05, 0.95)
            )

        pe = float(np.clip(prediction_error_phys, 0.0, 1.0))
        st["self_energy"] = float(np.clip(st["self_energy"] - 0.07 * pe, 0.05, 0.95))
        st["self_attention"] = float(np.clip(st["self_attention"] + 0.055 * pe, 0.05, 0.95))

        for arm, ski in (("lshoulder", "self_intention_larm"), ("rshoulder", "self_intention_rarm")):
            if arm not in observed or arm not in pred:
                continue
            pgap = float(observed[arm]) - float(pred.get(arm, observed[arm]))
            if abs(pgap) > 0.08:
                st[ski] = float(np.clip(st[ski] + 0.12 * lr * pgap, 0.05, 0.95))

    # ── Discovery rate ────────────────────────────────────────────────────────
    def discovery_rate(self, agent_edges: list[dict]) -> float:
        gt = self.gt_edges()
        gt_set = {(e["from_"], e["to"]) for e in gt}
        hits = sum(1 for e in agent_edges if (e.get("from_"), e.get("to")) in gt_set)
        return hits / len(gt_set) if gt_set else 0.0

    def gt_edges(self) -> list[dict]:
        """Removed GT edges to allow true open-ended discovery metrics."""
        return []

    # ── Упал? ─────────────────────────────────────────────────────────────────
    def is_fallen(self) -> bool:
        # В fixed_root mode робот никогда не падает
        if self._fixed_root:
            return False
        obs = self.observe()
        return obs.get("com_z", 0.5) < self._norm("com_z", FALLEN_Z)

    def reset_stance(self) -> None:
        self._sim.reset_stance()
        for k in SELF_VARS:
            self._self_state[k] = 0.5
        for k in MOTOR_INTENT_VARS:
            self._motor_state[k] = float(MOTOR_INTENT_DEFAULTS.get(k, 0.5))
        self._intero_state = {"intero_energy": 1.0, "intero_stress": 0.0}
        self._intero_control_lost = False
        self._prev_raw_obs = None

    # ── Camera / Skeleton ─────────────────────────────────────────────────────
    def get_frame_base64(self, view: str | None = None, **kwargs) -> str | None:
        return self._sim.get_frame_base64(view, **kwargs)

    def get_joint_positions_world(self) -> list[dict]:
        return self._sim.get_all_link_positions()

    def get_cube_positions(self) -> list[dict]:
        return self._sim.get_cube_positions()

    def get_target(self) -> dict:
        return {"x": 0.0, "y": 0.0, "z": STAND_Z}

    def get_full_scene(self) -> dict:
        scene = {
            "skeleton":   self.get_joint_positions_world(),
            "ankleQuats": self._sim.get_ankle_quaternions_three_js(),
            "cubes":      self.get_cube_positions(),
            "target":     self.get_target(),
            "fallen":     self.is_fallen(),
            "com_z":      self.observe().get("com_z", 0.5) if not self._fixed_root else 0.75,
            "fixed_root": self._fixed_root,
        }
        fn = getattr(self._sim, "get_sandbox_scene_extras", None)
        if callable(fn):
            scene.update(fn())
        return scene
