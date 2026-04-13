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

    # ── do() ─────────────────────────────────────────────────────────────────
    _JOINT_NEUTRAL: dict[str, float] = {
        "lshoulder": 0.50, "rshoulder": 0.50,
        "lelbow": 0.50, "relbow": 0.50,
        "spine_pitch": 0.50, "spine_yaw": 0.50,
        "neck_pitch": 0.50, "neck_yaw": 0.50,
        "lhip": 0.50, "rhip": 0.50,
        "lknee": 0.50, "rknee": 0.50,
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
        n = cls._JOINT_NEUTRAL.get(var, 0.5)
        r = cls._JOINT_COMFORT_RANGE.get(var, 0.45)
        return (max(0.05, n - r), min(0.95, n + r))

    def intervene(self, variable: str, value: float, *, count_intervention: bool = True) -> dict[str, float]:
        if count_intervention:
            self.n_interventions += 1

        if variable in SELF_VARS:
            self._self_state[variable] = float(np.clip(value, 0.05, 0.95))
            self._sim.step(self.steps_per_do)
            return self.observe()

        if variable in MOTOR_INTENT_VARS:
            self._motor_state[variable] = float(np.clip(value, 0.05, 0.95))
            self._apply_motor_intents()
            self._sim.step(self.steps_per_do)
            return self.observe()

        if self._fixed_root:
            controllable = ARM_VARS + SPINE_VARS + HEAD_VARS
        else:
            controllable = LEG_VARS + ARM_VARS + SPINE_VARS + HEAD_VARS

        if variable in controllable:
            lo, hi = self._comfort_zone(variable)
            clamped = float(np.clip(value, lo, hi))
            self._sim.set_joint(variable, clamped)

        self._sim.step(self.steps_per_do)
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
            if variable in SELF_VARS:
                self._self_state[variable] = v
            elif variable in MOTOR_INTENT_VARS:
                self._motor_state[variable] = v
                touched_intent = True
            elif variable in controllable:
                lo, hi = self._comfort_zone(variable)
                joints_after.append((variable, float(np.clip(v, lo, hi))))
        if touched_intent:
            self._apply_motor_intents()
        for variable, v in joints_after:
            self._sim.set_joint(variable, v)
        self._sim.step(self.steps_per_do)
        return self.observe()

    def _apply_upper_body_from_intents(self, *, cpg_sync: dict[str, float] | None = None) -> None:
        """
        Позвоночник + руки из тех же intent_*, что и весь моторный слой (граф / скиллы).
        Вызывается и при полном intervene, и перед CPG на ногах — руки не «застывают» между тиками.

        cpg_sync (опц.): фаза осцилляторов CPG + оценка отставания CoM — качает торс и плечи в фазе с ногами,
        чтобы масса тела не оставалась «мертвой» пока шагают только ноги.
        """
        intents = self._motor_state
        stride = float(intents.get("intent_stride", 0.5) - 0.5)
        sup_l = float(intents.get("intent_support_left", 0.5) - 0.5)
        sup_r = float(intents.get("intent_support_right", 0.5) - 0.5)
        torso = float(intents.get("intent_torso_forward", 0.5) - 0.5)
        torso = torso * 1.45
        arms = float(intents.get("intent_arm_counterbalance", 0.5) - 0.5)
        recover = float(intents.get("intent_stop_recover", 0.5) - 0.5)

        pitch_add = 0.0
        yaw_add = 0.0
        lsh_add = 0.0
        rsh_add = 0.0
        if cpg_sync:
            s = float(cpg_sync.get("sin", 0.0))
            c_m = float(cpg_sync.get("cos_mid", 0.0))
            sn = float(np.clip(cpg_sync.get("stride_n", 0.0), 0.0, 1.0))
            lag = float(np.clip(cpg_sync.get("com_lag", 0.0), 0.0, 1.0))
            
            # com_lag → небольшой наклон вперёд; вес синхронизирован с cpg_locomotion (бедро/колено в swing)
            try:
                _lag_pitch_u = float(os.environ.get("RKK_CPG_COM_LAG_PITCH", "0.08"))
            except ValueError:
                _lag_pitch_u = 0.08
            _lag_pitch_u = float(np.clip(_lag_pitch_u, 0.0, 0.35))
            pitch_add = (-0.055 * s * sn) + (_lag_pitch_u * lag * sn)
            
            yaw_add = 0.05 * c_m * sn
            lsh_add = -0.065 * s * sn
            rsh_add = 0.065 * s * sn
            gs = float(np.clip(cpg_sync.get("gscale", 1.0), 0.0, 1.0))
            pitch_add *= gs
            yaw_add *= gs
            lsh_add *= gs
            rsh_add *= gs

        def clip01(v: float) -> float:
            return float(np.clip(v, 0.05, 0.95))

        if self._fixed_root:
            self._sim.set_joint(
                "spine_pitch",
                # ИСПРАВЛЕНО: Меняем минус на плюс перед stride
                clip01(0.50 + 0.10 * torso + 0.10 * recover + 0.05 * arms + 0.15 * max(0.0, stride) + pitch_add),
            )
            self._sim.set_joint("spine_yaw", clip01(0.5 + 0.06 * (sup_l - sup_r) + yaw_add))
            self._sim.set_joint("lshoulder", clip01(0.50 + 0.05 * arms + 0.02 * recover + lsh_add))
            self._sim.set_joint("rshoulder", clip01(0.50 - 0.05 * arms + 0.02 * recover + rsh_add))
            self._sim.set_joint("lelbow", clip01(0.50 + 0.06 * arms))
            self._sim.set_joint("relbow", clip01(0.50 - 0.06 * arms))
            return

        self._sim.set_joint(
            "spine_pitch",
            # ИСПРАВЛЕНО: Меняем минус на плюс перед stride (с 0.08 на 0.12 для лучшего наката массы)
            clip01(0.50 + 0.10 * torso + 0.10 * recover + 0.05 * arms + 0.12 * max(0.0, stride) + pitch_add),
        )
        self._sim.set_joint("spine_yaw", clip01(0.5 + 0.06 * (sup_l - sup_r) + yaw_add))
        self._sim.set_joint(
            "lshoulder",
            clip01(0.50 + 0.04 * arms + 0.01 * stride + 0.02 * recover + lsh_add),
        )
        self._sim.set_joint(
            "rshoulder",
            clip01(0.50 - 0.04 * arms - 0.01 * stride + 0.02 * recover + rsh_add),
        )
        self._sim.set_joint("lelbow", clip01(0.50 + 0.05 * arms))
        self._sim.set_joint("relbow", clip01(0.50 - 0.05 * arms))

    def _apply_motor_intents(self) -> None:
        """
        Legs from intents + upper body from _apply_upper_body_from_intents.
        When cpg_owns_legs is True, skip leg joints (CPG controls them directly).
        """
        intents = self._motor_state
        stride = float(intents.get("intent_stride", 0.5) - 0.5)
        sup_l = float(intents.get("intent_support_left", 0.5) - 0.5)
        sup_r = float(intents.get("intent_support_right", 0.5) - 0.5)
        torso = float(intents.get("intent_torso_forward", 0.5) - 0.5)
        recover = float(intents.get("intent_stop_recover", 0.5) - 0.5)

        def clip01(v: float) -> float:
            return float(np.clip(v, 0.05, 0.95))

        if self._fixed_root:
            self._apply_upper_body_from_intents()
            return

        if not self.cpg_owns_legs:
            self._sim.set_joint("lhip", clip01(0.50 + 0.14 * stride - 0.08 * sup_r + 0.05 * torso - 0.06 * recover))
            self._sim.set_joint("rhip", clip01(0.50 - 0.14 * stride - 0.08 * sup_l + 0.05 * torso - 0.06 * recover))
            self._sim.set_joint("lknee", clip01(0.50 + 0.12 * sup_l + 0.10 * recover))
            self._sim.set_joint("rknee", clip01(0.50 + 0.12 * sup_r + 0.10 * recover))
            self._sim.set_joint("lankle", clip01(0.50 + 0.08 * sup_l - 0.03 * stride - 0.04 * recover))
            self._sim.set_joint("rankle", clip01(0.50 + 0.08 * sup_r + 0.03 * stride - 0.04 * recover))
        self._apply_upper_body_from_intents()

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
        self._apply_upper_body_from_intents(cpg_sync=cpg_sync)
        for name, val in targets.items():
            if name in LEG_VARS:
                self._sim.set_joint(name, float(np.clip(val, 0.05, 0.95)))
        self._sim.step(n_sub)

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
        if self._fixed_root:
            # В fixed_root: arm→cube + самомодель (намерение → тело → мир)
            return [
                {"from_": "self_intention_larm", "to": "lshoulder", "weight": 0.55},
                {"from_": "self_intention_larm", "to": "cube0_x", "weight": 0.35},
                {"from_": "self_intention_rarm", "to": "rshoulder", "weight": 0.55},
                {"from_": "self_intention_rarm", "to": "cube1_x", "weight": 0.35},
                {"from_": "self_attention", "to": "neck_yaw", "weight": 0.35},
                {"from_": "self_energy", "to": "lshoulder", "weight": 0.25},
                {"from_": "lshoulder", "to": "cube0_x", "weight": 0.6},
                {"from_": "lshoulder", "to": "cube0_y", "weight": 0.4},
                {"from_": "rshoulder", "to": "cube1_x", "weight": 0.6},
                {"from_": "rshoulder", "to": "cube1_y", "weight": 0.4},
                {"from_": "intent_stride", "to": "lhip", "weight": 0.55},
                {"from_": "intent_stride", "to": "rhip", "weight": 0.55},
                {"from_": "intent_stride", "to": "intent_torso_forward", "weight": 0.42},
                {"from_": "intent_stride", "to": "spine_pitch", "weight": 0.38},
                {"from_": "intent_arm_counterbalance", "to": "lshoulder", "weight": 0.45},
                {"from_": "intent_arm_counterbalance", "to": "rshoulder", "weight": 0.45},
                {"from_": "intent_torso_forward", "to": "spine_pitch", "weight": 0.52},
                {"from_": "intent_stride", "to": "intent_gait_coupling", "weight": 0.40},
                {"from_": "self_energy", "to": "intent_gait_coupling", "weight": 0.26},
                {"from_": "lelbow",    "to": "cube0_z", "weight": 0.3},
                {"from_": "relbow",    "to": "cube1_z", "weight": 0.3},
                {"from_": "lshoulder", "to": "lelbow",  "weight": 0.5},
                {"from_": "rshoulder", "to": "relbow",  "weight": 0.5},
                {"from_": "spine_yaw", "to": "cube2_x", "weight": 0.3},
            ]
        # Полный режим
        edges = []
        for v in LEG_VARS:
            edges.append({"from_": v, "to": "com_z",  "weight": 0.5})
        edges.append({"from_": "lhip",       "to": "com_x",    "weight": 0.7})
        edges.append({"from_": "rhip",        "to": "com_x",    "weight": 0.7})
        edges.append({"from_": "lknee",       "to": "lfoot_z",  "weight": 0.8})
        edges.append({"from_": "rknee",       "to": "rfoot_z",  "weight": 0.8})
        edges.append({"from_": "lshoulder",   "to": "cube0_x",  "weight": 0.6})
        edges.append({"from_": "rshoulder",   "to": "cube1_x",  "weight": 0.6})
        edges.append({"from_": "com_z",       "to": "torso_roll","weight": -0.4})
        edges.extend([
            {"from_": "intent_stride", "to": "lhip", "weight": 0.7},
            {"from_": "intent_stride", "to": "rhip", "weight": 0.7},
            {"from_": "intent_stride", "to": "intent_torso_forward", "weight": 0.45},
            {"from_": "intent_stride", "to": "spine_pitch", "weight": 0.38},
            {"from_": "intent_stride", "to": "com_x", "weight": 0.28},
            {"from_": "intent_support_left", "to": "lknee", "weight": 0.45},
            {"from_": "intent_support_right", "to": "rknee", "weight": 0.45},
            {"from_": "intent_torso_forward", "to": "spine_pitch", "weight": 0.58},
            {"from_": "intent_torso_forward", "to": "com_x", "weight": 0.32},
            {"from_": "intent_stride", "to": "intent_gait_coupling", "weight": 0.42},
            {"from_": "self_energy", "to": "intent_gait_coupling", "weight": 0.28},
            {"from_": "intent_stop_recover", "to": "intent_gait_coupling", "weight": -0.35},
            {"from_": "posture_stability", "to": "intent_gait_coupling", "weight": 0.22},
            {"from_": "intent_stop_recover", "to": "com_z", "weight": 0.35},
            {"from_": "intent_arm_counterbalance", "to": "lshoulder", "weight": 0.4},
            {"from_": "intent_arm_counterbalance", "to": "rshoulder", "weight": 0.4},
            {"from_": "lhip", "to": "gait_phase_l", "weight": 0.35},
            {"from_": "rhip", "to": "gait_phase_r", "weight": 0.35},
            {"from_": "lknee", "to": "foot_contact_l", "weight": 0.45},
            {"from_": "rknee", "to": "foot_contact_r", "weight": 0.45},
            {"from_": "foot_contact_l", "to": "support_bias", "weight": 0.25},
            {"from_": "foot_contact_r", "to": "support_bias", "weight": 0.25},
            {"from_": "support_bias", "to": "torso_roll", "weight": -0.2},
            {"from_": "self_intention_larm", "to": "lshoulder", "weight": 0.5},
            {"from_": "self_intention_rarm", "to": "rshoulder", "weight": 0.5},
            {"from_": "self_attention", "to": "neck_yaw", "weight": 0.3},
        ])
        return edges

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
