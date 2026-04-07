"""
environment_humanoid.py — Humanoid Robot Sandbox (Фаза 11 + fixed_root).

Добавлен fixed_root mode:
  - PyBullet JOINT_FIXED constraint фиксирует base в воздухе
  - variable_ids → FIXED_BASE_VARS (+ self_* самомодель)
  - is_fallen() всегда False → ValueLayer не блокирует
  - GNN: arms→cubes + self_*→суставы/кубы (self задаёт агент, не PyBullet)
  - EnvironmentHumanoid.set_fixed_root(bool) — переключение в runtime

FIXED_BASE_VARS = ARM + SPINE + HEAD + CUBE_VARS + SANDBOX_VARS + SELF_VARS (самомодель)
"""
from __future__ import annotations

import os
import threading
import time
import numpy as np
import torch
import base64
from io import BytesIO
from pathlib import Path

try:
    import pybullet as pb
    import pybullet_data as pbd
    PYBULLET_AVAILABLE = True
except ImportError:
    PYBULLET_AVAILABLE = False
    print("[HumanoidEnv] pybullet not installed, using fallback")

try:
    from PIL import Image as PILImage
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


# ─── Переменные ───────────────────────────────────────────────────────────────
TORSO_VARS  = ["com_x", "com_y", "com_z", "torso_roll", "torso_pitch"]
SPINE_VARS  = ["spine_yaw", "spine_pitch"]
HEAD_VARS   = ["neck_yaw", "neck_pitch"]
LEG_VARS    = ["lhip", "lknee", "lankle", "rhip", "rknee", "rankle"]
ARM_VARS    = ["lshoulder", "lelbow", "rshoulder", "relbow"]
FOOT_VARS   = ["lfoot_z", "rfoot_z"]
CUBE_VARS   = [f"cube{i}_{d}" for i in range(3) for d in ["x","y","z"]]
# Песочница: мяч, рычаг (проксимити), расстояние до зоны доставки — обогащают каузальный граф.
# Фаза 2 (часть 2): floor_friction / stack_height / stability_score — наблюдаемые эффекты скрытых cube_temp, spring_k.
SANDBOX_VARS = [
    "ball_x", "ball_y", "ball_z",
    "lever_pin", "target_dist",
    "floor_friction",
    "stack_height",
    "stability_score",
]
# Motor-intent layer: high-level causal controls that can be planned/intervened on
# without directly addressing raw joints.
MOTOR_INTENT_VARS: tuple[str, ...] = (
    "intent_stride",
    "intent_support_left",
    "intent_support_right",
    "intent_torso_forward",
    "intent_arm_counterbalance",
    "intent_stop_recover",
)
# Derived motor observables that should be visible to the graph and hybrid view.
MOTOR_OBSERVABLE_VARS: tuple[str, ...] = (
    "gait_phase_l",
    "gait_phase_r",
    "foot_contact_l",
    "foot_contact_r",
    "support_bias",
    "motor_drive_l",
    "motor_drive_r",
    "posture_stability",
)
# Этап Г — самомодель; Этап E — self_goal_* + imagination-планирование к target_dist (см. goal_planning.py).
SELF_VARS: tuple[str, ...] = (
    "self_intention_larm",
    "self_intention_rarm",
    "self_energy",
    "self_attention",
    # Этап E: цель (нормализованный порог target_dist + включение планировщика)
    "self_goal_target_dist",
    "self_goal_active",
)
VAR_NAMES   = (
    TORSO_VARS + SPINE_VARS + HEAD_VARS + LEG_VARS + ARM_VARS + FOOT_VARS
    + CUBE_VARS + SANDBOX_VARS + list(MOTOR_INTENT_VARS) + list(MOTOR_OBSERVABLE_VARS) + list(SELF_VARS)
)

# Фаза 1: URDF-топология → frozen_edges (meta: alpha_trust для отчёта; в W — clamp + mask L1/grad).
# В графе нет узлов «spine/chest» — только spine_*, neck_*.
URDF_FROZEN_EDGES: dict[tuple[str, str], dict[str, float]] = {
    ("lhip", "lknee"): {"alpha_trust": 1.0},
    ("lknee", "lankle"): {"alpha_trust": 1.0},
    ("rhip", "rknee"): {"alpha_trust": 1.0},
    ("rknee", "rankle"): {"alpha_trust": 1.0},
    ("lshoulder", "lelbow"): {"alpha_trust": 1.0},
    ("rshoulder", "relbow"): {"alpha_trust": 1.0},
    ("spine_yaw", "spine_pitch"): {"alpha_trust": 1.0},
    ("spine_pitch", "neck_yaw"): {"alpha_trust": 1.0},
    ("neck_yaw", "neck_pitch"): {"alpha_trust": 1.0},
}
HUMANOID_KINEMATIC_EDGE_PRIORS: tuple[tuple[str, str], ...] = tuple(URDF_FROZEN_EDGES.keys())

# LocalReflex (Фаза 1): только соседи по кинематике — отдельные цепи d=len(chain).
KINEMATIC_CHAINS: tuple[tuple[str, ...], ...] = (
    ("lhip", "lknee", "lankle"),
    ("rhip", "rknee", "rankle"),
    ("lshoulder", "lelbow"),
    ("rshoulder", "relbow"),
    ("spine_yaw", "spine_pitch", "neck_yaw", "neck_pitch"),
)

# Fixed-base mode: таз зафиксирован в воздухе, баланс исключён.
# Агент учит arms/spine/neck → кубы + сигналы песочницы (мяч, рычаг, цель) + self_*.
FIXED_BASE_VARS: list[str] = (
    ARM_VARS
    + SPINE_VARS
    + HEAD_VARS
    + CUBE_VARS
    + SANDBOX_VARS
    + list(MOTOR_INTENT_VARS)
    + list(MOTOR_OBSERVABLE_VARS)
    + list(SELF_VARS)
)

# Нормализация диапазонов
_RANGES = {}
for v in TORSO_VARS[:2]:  _RANGES[v] = (-1.5, 1.5)
_RANGES["com_z"]          = (0.0,  1.5)
_RANGES["torso_roll"]     = (-1.2, 1.2)
_RANGES["torso_pitch"]    = (-1.2, 1.2)
for v in SPINE_VARS:       _RANGES[v] = (-1.2, 1.2)
for v in HEAD_VARS:        _RANGES[v] = (-1.2, 1.2)
for v in LEG_VARS:         _RANGES[v] = (-1.5, 1.5)
for v in ARM_VARS:         _RANGES[v] = (-2.0, 2.0)
for v in FOOT_VARS:        _RANGES[v] = (-0.1, 0.5)
for v in CUBE_VARS:
    if v.endswith("_z"):   _RANGES[v] = (-0.1, 1.0)
    else:                  _RANGES[v] = (-2.0, 2.0)
for v in ("ball_x", "ball_y"):
    _RANGES[v] = (-2.5, 2.5)
_RANGES["ball_z"]       = (0.0, 2.0)
_RANGES["lever_pin"]    = (0.0, 1.0)
_RANGES["target_dist"]  = (0.0, 4.0)
_RANGES["floor_friction"] = (0.1, 1.0)
_RANGES["stack_height"] = (0.0, 0.8)
_RANGES["stability_score"] = (0.0, 1.0)
for _mv in MOTOR_INTENT_VARS + MOTOR_OBSERVABLE_VARS:
    _RANGES[_mv] = (0.0, 1.0)
for _sv in SELF_VARS:
    _RANGES[_sv] = (0.0, 1.0)

# Масштаб URDF относительно прежнего «эталона» 0.36 (0.18 = ровно в 2× меньше по линейным размерам).
HUMANOID_URDF_LEGACY_SCALE = 0.36
HUMANOID_URDF_GLOBAL_SCALING = 0.18
_HSZ = HUMANOID_URDF_GLOBAL_SCALING / HUMANOID_URDF_LEGACY_SCALE

FALLEN_Z     = 0.25 * _HSZ
STAND_Z      = 0.85 * _HSZ
HUMANOID_URDF_STAND_EULER = (np.pi / 2, 0.0, 0.0)
HUMANOID_URDF_SPAWN_Z = 1.15 * _HSZ


def _np_quat_from_axis_angle(axis: np.ndarray, angle: float) -> list[float]:
    ax = np.asarray(axis, dtype=float).reshape(3)
    n = float(np.linalg.norm(ax))
    if n < 1e-9:
        return [0.0, 0.0, 0.0, 1.0]
    ax = ax / n
    half = 0.5 * float(angle)
    s = np.sin(half)
    return [float(ax[0] * s), float(ax[1] * s), float(ax[2] * s), float(np.cos(half))]


def _np_quat_mul(q1: list[float], q2: list[float]) -> list[float]:
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return [
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    ]


_PB_VEC_TO_THREE = np.array(
    [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]], dtype=float
)


def _rotmat_to_xyzw(R: np.ndarray) -> tuple[float, float, float, float]:
    m = np.asarray(R, dtype=float).reshape(3, 3)
    tr = float(np.trace(m))
    if tr > 0.0:
        s = 0.5 / np.sqrt(tr + 1.0)
        w = 0.25 / s
        x = (m[2, 1] - m[1, 2]) * s
        y = (m[0, 2] - m[2, 0]) * s
        z = (m[1, 0] - m[0, 1]) * s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    n = float(np.sqrt(x * x + y * y + z * z + w * w))
    if n < 1e-12:
        return (0.0, 0.0, 0.0, 1.0)
    return (x / n, y / n, z / n, w / n)


def _forward_kinematics_skeleton(
    cx: float, cy: float, cz: float, joints: dict[str, float]
) -> list[dict]:
    sk = float(HUMANOID_URDF_GLOBAL_SCALING) / float(HUMANOID_URDF_LEGACY_SCALE)
    j = joints
    pelvis_z = cz - 0.12 * sk
    neck_z = cz + 0.17 * sk
    spine_z = 0.5 * (neck_z + pelvis_z)
    head_z = cz + 0.29 * sk
    sh_z = neck_z - 0.02 * sk
    hip_z = pelvis_z - 0.08 * sk

    head = [cx, cy, head_z]
    neck = [cx, cy, neck_z]
    spine = [cx, cy, spine_z]
    pelvis = [cx, cy, pelvis_z]
    lshld = [cx - 0.26 * sk, cy, sh_z]
    rshld = [cx + 0.26 * sk, cy, sh_z]
    lelbow = [
        cx - 0.26 * sk - np.sin(j.get("lshoulder", 0)) * 0.28 * sk,
        cy,
        sh_z - np.cos(j.get("lshoulder", 0)) * 0.28 * sk,
    ]
    relbow = [
        cx + 0.26 * sk + np.sin(j.get("rshoulder", 0)) * 0.28 * sk,
        cy,
        sh_z - np.cos(j.get("rshoulder", 0)) * 0.28 * sk,
    ]
    lhand = [
        lelbow[0] - np.sin(j.get("lelbow", 0)) * 0.22 * sk,
        cy,
        lelbow[2] - np.cos(j.get("lelbow", 0)) * 0.22 * sk,
    ]
    rhand = [
        relbow[0] + np.sin(j.get("relbow", 0)) * 0.22 * sk,
        cy,
        relbow[2] - np.cos(j.get("relbow", 0)) * 0.22 * sk,
    ]
    hip_half = 0.15 * sk
    lhip_p = [cx - hip_half, cy, hip_z]
    rhip_p = [cx + hip_half, cy, hip_z]
    lknee_p = [
        cx - hip_half + np.sin(j.get("lhip", 0)) * 0.35 * sk,
        cy,
        hip_z - np.cos(j.get("lhip", 0)) * 0.35 * sk,
    ]
    rknee_p = [
        cx + hip_half + np.sin(j.get("rhip", 0)) * 0.35 * sk,
        cy,
        hip_z - np.cos(j.get("rhip", 0)) * 0.35 * sk,
    ]
    lfoot = [
        lknee_p[0] + np.sin(j.get("lknee", 0)) * 0.30 * sk,
        cy,
        lknee_p[2] - np.cos(j.get("lknee", 0)) * 0.30 * sk,
    ]
    rfoot = [
        rknee_p[0] + np.sin(j.get("rknee", 0)) * 0.30 * sk,
        cy,
        rknee_p[2] - np.cos(j.get("rknee", 0)) * 0.30 * sk,
    ]
    sc = float(HUMANOID_URDF_GLOBAL_SCALING)
    lsole = [lfoot[0] - 0.05 * sc, lfoot[1], lfoot[2] - 0.12 * sc]
    rsole = [rfoot[0] + 0.05 * sc, rfoot[1], rfoot[2] - 0.12 * sc]
    pts = [
        head, neck, spine, pelvis,
        lshld, rshld, lelbow, relbow, lhand, rhand,
        lhip_p, rhip_p, lknee_p, rknee_p, lfoot, rfoot,
        lsole, rsole,
    ]
    return [{"x": float(p[0]), "y": float(p[1]), "z": float(p[2])} for p in pts]


# ─── Фаза 2 (часть 2): скрытые переменные песочницы ────────────────────────────
class InstrumentalSandbox:
    """
    Скрытые: _cube_temp (нагрев от рычага), _spring_k_real (пружина cube0–cube1).
    Наблюдаемые через get_state: floor_friction, stack_height, stability_score.
    """

    def _init_instrumental(self) -> None:
        self._cube_temp: float = 0.0
        self._floor_friction_base: float = 0.5
        self._spring_k: float = 0.0
        self._spring_k_real: float = float(np.random.uniform(0.1, 0.8))
        self._stack_height: float = 0.0
        self._stability_score: float = 0.0
        self._ankle_friction_scale: float = 1.0

    def _reset_instrumental_hidden(self) -> None:
        self._cube_temp = 0.0
        self._stack_height = 0.0
        self._stability_score = 0.0
        self._spring_k_real = float(np.random.uniform(0.1, 0.8))
        self._ankle_friction_scale = 1.0

    def _compute_floor_friction_effect(self) -> float:
        return float(np.clip(1.0 - self._cube_temp * 0.65, 0.15, 1.0))

    def _update_stack_metrics_pybullet(self) -> None:
        if not getattr(self, "cube_ids", None):
            return
        cid = self.client
        positions = [
            pb.getBasePositionAndOrientation(c, physicsClientId=cid)[0]
            for c in self.cube_ids
        ]
        zs = [p[2] for p in positions]
        self._stack_height = float(max(zs) - min(zs))
        xs = np.array([p[0] for p in positions], dtype=float)
        ys = np.array([p[1] for p in positions], dtype=float)
        spread = float(np.std(xs) + np.std(ys))
        self._stability_score = float(np.clip(1.0 - spread / 1.5, 0.0, 1.0))

    def _update_stack_metrics_fallback(self) -> None:
        if not hasattr(self, "cubes") or len(self.cubes) == 0:
            return
        zs = [float(self.cubes[i][2]) for i in range(len(self.cubes))]
        self._stack_height = float(max(zs) - min(zs))
        xs = np.array([float(self.cubes[i][0]) for i in range(len(self.cubes))], dtype=float)
        ys = np.array([float(self.cubes[i][1]) for i in range(len(self.cubes))], dtype=float)
        spread = float(np.std(xs) + np.std(ys))
        self._stability_score = float(np.clip(1.0 - spread / 1.5, 0.0, 1.0))

    def _instrumental_spring_pybullet(self) -> None:
        if len(getattr(self, "cube_ids", [])) < 2:
            return
        cid = self.client
        p0, _ = pb.getBasePositionAndOrientation(self.cube_ids[0], physicsClientId=cid)
        p1, _ = pb.getBasePositionAndOrientation(self.cube_ids[1], physicsClientId=cid)
        delta = np.array(p1[:3], dtype=float) - np.array(p0[:3], dtype=float)
        dist = float(np.linalg.norm(delta))
        rest = 0.6
        if dist > 1e-4:
            force = -self._spring_k_real * (dist - rest) * (delta / dist)
            pb.applyExternalForce(
                self.cube_ids[1], -1,
                force.tolist(), list(p1),
                pb.WORLD_FRAME,
                physicsClientId=cid,
            )

    def _instrumental_spring_fallback(self) -> None:
        if not hasattr(self, "cubes") or len(self.cubes) < 2:
            return
        d = np.array(self.cubes[1], dtype=float) - np.array(self.cubes[0], dtype=float)
        dist = float(np.linalg.norm(d))
        rest = 0.6
        if dist > 1e-4:
            self.cubes[1] = self.cubes[1] + (
                -self._spring_k_real * (dist - rest) * (d / dist) * 0.002
            )

    def _tick_hidden_state(self) -> None:
        lp = float(self._compute_lever_pin())
        tau_heat, tau_cool = 0.04, 0.008
        if lp > 0.5:
            self._cube_temp = min(1.0, self._cube_temp + tau_heat * (lp - 0.3))
        else:
            self._cube_temp = max(0.0, self._cube_temp - tau_cool)
        if getattr(self, "client", None) is not None:
            self._instrumental_spring_pybullet()
            self._update_stack_metrics_pybullet()
        else:
            self._instrumental_spring_fallback()
            self._update_stack_metrics_fallback()

    def _apply_friction_to_ankle_joints(self) -> None:
        friction = self._compute_floor_friction_effect()
        if getattr(self, "client", None) is None:
            self._ankle_friction_scale = float(friction)
            return
        rid, cid = self.robot_id, self.client
        for var in ("lankle", "rankle"):
            if var not in self.joint_by_var:
                continue
            jid = self.joint_by_var[var]
            if jid >= len(self._joint_types):
                continue
            if self._joint_types[jid] == pb.JOINT_SPHERICAL:
                continue
            pb.setJointMotorControl2(
                rid, jid,
                controlMode=pb.VELOCITY_CONTROL,
                targetVelocity=0,
                force=50.0 * friction,
                physicsClientId=cid,
            )


# ─── Fallback гуманоид ────────────────────────────────────────────────────────
class _FallbackHumanoid(InstrumentalSandbox):
    def __init__(self, fixed_root: bool = False):
        self.fixed_root = fixed_root
        self.joints = {v: 0.0 for v in LEG_VARS + ARM_VARS + SPINE_VARS + HEAD_VARS}
        self.com    = np.array([0.0, 0.0, STAND_Z])
        self.torso_euler = np.zeros(3)
        self.cubes  = np.array([
            [ 0.8,  0.3, 0.15],
            [-0.5,  0.6, 0.15],
            [ 0.2, -0.7, 0.25],
        ])
        self.ball = np.array([0.5, -0.35, 0.12], dtype=np.float64)
        self._lever_center = np.array([-1.0, 0.5, 0.1], dtype=np.float64)
        self._target_pad = np.array([1.85, -0.75, 0.02], dtype=np.float64)
        self._vel   = np.zeros(3)
        self._dt    = 0.02
        self._init_instrumental()

    def _compute_lever_pin(self) -> float:
        pts = [self.ball] + [self.cubes[i] for i in range(len(self.cubes))]
        lc = self._lever_center[:2]
        d = min(float(np.linalg.norm(p[:2] - lc[:2])) for p in pts)
        return float(np.clip(1.0 - d / 0.35, 0.0, 1.0))

    def step(self, n: int = 10):
        if self.fixed_root:
            # В fixed_root mode нет гравитации на базу
            self.com[:2] += np.random.normal(0, 0.0005, 2)
            self._tick_hidden_state()
            self._apply_friction_to_ankle_joints()
            return
        for _ in range(n):
            balance = abs(self.torso_euler[0]) + abs(self.torso_euler[1])
            self.com[2] += (-0.02 * balance + 0.001 * (np.random.rand()-0.5)) * self._dt
            self.com[2]  = np.clip(self.com[2], 0.0, 1.5)
            self.com[:2] += np.random.normal(0, 0.002, 2)
            self.torso_euler[:2] += np.random.normal(0, 0.003, 2)
            self.torso_euler[:2] = np.clip(self.torso_euler[:2], -1.2, 1.2)
        self._tick_hidden_state()
        self._apply_friction_to_ankle_joints()

    def set_joint(self, name: str, val: float):
        if name in self.joints:
            lo, hi = _RANGES.get(name, (-2.0, 2.0))
            self.joints[name] = float(np.clip(val * (hi - lo) + lo, lo, hi))
            if not self.fixed_root:
                if "hip" in name:
                    idx = 0 if name.startswith("l") else 1
                    self.com[idx] += val * 0.02
                if "knee" in name:
                    self.com[2] = max(0.1, self.com[2] + val * 0.01)
            # Куб-взаимодействие через плечо
            if "shoulder" in name:
                cube_idx = 0 if name.startswith("l") else 1
                self.cubes[cube_idx][0] += (val - 0.5) * 0.04

    def get_foot_z(self) -> tuple[float, float]:
        k_l = self.joints.get("lknee", 0)
        k_r = self.joints.get("rknee", 0)
        f = float(getattr(self, "_ankle_friction_scale", 1.0))
        return (
            max(0, 0.1 - k_l * 0.05) * f,
            max(0, 0.1 - k_r * 0.05) * f,
        )

    def get_all_link_positions(self) -> list[dict]:
        cx, cy, cz = float(self.com[0]), float(self.com[1]), float(self.com[2])
        return _forward_kinematics_skeleton(cx, cy, cz, self.joints)

    def get_state(self) -> dict:
        lf_z, rf_z = self.get_foot_z()
        s = {}
        s["com_x"]        = float(self.com[0])
        s["com_y"]        = float(self.com[1])
        s["com_z"]        = float(self.com[2])
        s["torso_roll"]   = float(self.torso_euler[0])
        s["torso_pitch"]  = float(self.torso_euler[1])
        for v in LEG_VARS + ARM_VARS + SPINE_VARS + HEAD_VARS:
            s[v] = float(self.joints.get(v, 0.0))
        s["lfoot_z"]  = float(lf_z)
        s["rfoot_z"]  = float(rf_z)
        for i, cube in enumerate(self.cubes):
            s[f"cube{i}_x"] = float(cube[0])
            s[f"cube{i}_y"] = float(cube[1])
            s[f"cube{i}_z"] = float(cube[2])
        s["ball_x"] = float(self.ball[0])
        s["ball_y"] = float(self.ball[1])
        s["ball_z"] = float(self.ball[2])
        pts = [self.ball] + [self.cubes[i] for i in range(len(self.cubes))]
        d_lv = min(float(np.linalg.norm(p[:2] - self._lever_center[:2])) for p in pts)
        s["lever_pin"] = float(np.clip(1.0 - d_lv / 0.35, 0.0, 1.0))
        d_tg = min(
            float(np.linalg.norm(self.cubes[i][:2] - self._target_pad[:2]))
            for i in range(len(self.cubes))
        )
        s["target_dist"] = float(d_tg)
        s["floor_friction"] = self._compute_floor_friction_effect()
        s["stack_height"] = float(self._stack_height)
        s["stability_score"] = float(self._stability_score)
        return s

    def get_cube_positions(self) -> list[dict]:
        return [{"x":float(c[0]),"y":float(c[1]),"z":float(c[2])} for c in self.cubes]

    def get_sandbox_scene_extras(self) -> dict:
        pts = [self.ball] + [self.cubes[i].copy() for i in range(len(self.cubes))]
        d_lv = min(float(np.linalg.norm(p[:2] - self._lever_center[:2])) for p in pts)
        lp = float(np.clip(1.0 - d_lv / 0.35, 0.0, 1.0))
        return {
            "ball": {"x": float(self.ball[0]), "y": float(self.ball[1]), "z": float(self.ball[2])},
            "lever": {
                "x": float(self._lever_center[0]),
                "y": float(self._lever_center[1]),
                "z": float(self._lever_center[2]),
                "pressed": lp > 0.75,
                "pin": round(lp, 3),
            },
            "delivery_target": {
                "x": float(self._target_pad[0]),
                "y": float(self._target_pad[1]),
                "z": float(self._target_pad[2]),
            },
        }

    def get_frame_base64(self, view="side", **kwargs) -> str | None:
        return None

    def get_ankle_quaternions_three_js(self) -> list[dict[str, float]]:
        return [
            {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
            {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
        ]

    def reset_stance(self) -> None:
        self.joints = {v: 0.0 for v in LEG_VARS + ARM_VARS + SPINE_VARS + HEAD_VARS}
        self.com = np.array([0.0, 0.0, STAND_Z], dtype=np.float64)
        self.torso_euler = np.zeros(3)
        self._vel = np.zeros(3)
        self.ball = np.array([0.5, -0.35, 0.12], dtype=np.float64)
        self._reset_instrumental_hidden()

    # fixed_root controls для fallback — только флаг
    def enable_fixed_root(self) -> None:
        self.fixed_root = True

    def disable_fixed_root(self) -> None:
        self.fixed_root = False


# ─── PyBullet гуманоид ────────────────────────────────────────────────────────
class _PyBulletHumanoid(InstrumentalSandbox):

    _JOINT_MAP = {
        "leftHip":     "lhip",   "left_hip":    "lhip",   "LeftHip":     "lhip",
        "leftKnee":    "lknee",  "left_knee":   "lknee",  "LeftKnee":    "lknee",
        "leftAnkle":   "lankle", "left_ankle":  "lankle",
        "rightHip":    "rhip",   "right_hip":   "rhip",   "RightHip":    "rhip",
        "rightKnee":   "rknee",  "right_knee":  "rknee",  "RightKnee":   "rknee",
        "rightAnkle":  "rankle", "right_ankle": "rankle",
        "leftShoulder":"lshoulder", "left_shoulder":"lshoulder",
        "rightShoulder":"rshoulder","right_shoulder":"rshoulder",
        "leftElbow":   "lelbow", "left_elbow":  "lelbow",
        "rightElbow":  "relbow", "right_elbow": "relbow",
    }

    def __init__(self, fixed_root: bool = False):
        # ── PyBullet init ────────────────────────────────────────────────────
        self._physics_lock = threading.RLock()
        self._bg_running = False
        self._bg_thread: threading.Thread | None = None
        self._bg_hz = 0.0

        self.client = pb.connect(pb.DIRECT)
        pb.setGravity(0, 0, -9.81, physicsClientId=self.client)
        pb.setAdditionalSearchPath(pbd.getDataPath(), physicsClientId=self.client)
        pb.setTimeStep(1/240., physicsClientId=self.client)
        pb.setPhysicsEngineParameter(numSolverIterations=50, physicsClientId=self.client)

        self.floor_id = pb.loadURDF("plane.urdf", physicsClientId=self.client)
        self._build_ramp()

        self.cube_ids = []
        cube_configs = [
            {"pos": [1.0,  0.3, 0.15], "size": 0.12, "mass": 2.0,  "color": [1.0, 0.4, 0.1, 1]},
            {"pos": [-0.6, 0.8, 0.12], "size": 0.10, "mass": 0.5,  "color": [0.2, 0.7, 1.0, 1]},
            {"pos": [0.3, -0.9, 0.20], "size": 0.16, "mass": 6.0,  "color": [0.25, 0.85, 0.35, 1]},
        ]
        for cfg in cube_configs:
            hs = cfg["size"] / 2
            col = pb.createCollisionShape(pb.GEOM_BOX, halfExtents=[hs,hs,hs], physicsClientId=self.client)
            vis = pb.createVisualShape(pb.GEOM_BOX, halfExtents=[hs,hs,hs],
                                        rgbaColor=cfg["color"], physicsClientId=self.client)
            b = pb.createMultiBody(
                baseMass=cfg["mass"],
                baseCollisionShapeIndex=col,
                baseVisualShapeIndex=vis,
                basePosition=cfg["pos"],
                physicsClientId=self.client
            )
            pb.changeDynamics(b, -1, lateralFriction=0.5, physicsClientId=self.client)
            self.cube_ids.append(b)

        # Песочница: рычаг (зона проксимити), цель доставки, лёгкий мяч
        self.lever_center = np.array([-1.05, 0.55, 0.10], dtype=np.float64)
        self.target_pad = np.array([1.85, -0.72, 0.02], dtype=np.float64)
        self.lever_trigger_r = 0.26
        self._ball_start = [0.55, -0.45, 0.11]
        br = 0.09
        col_ball = pb.createCollisionShape(pb.GEOM_SPHERE, radius=br, physicsClientId=self.client)
        vis_ball = pb.createVisualShape(
            pb.GEOM_SPHERE, radius=br,
            rgbaColor=[0.95, 0.92, 0.35, 1.0], physicsClientId=self.client,
        )
        self.ball_id = pb.createMultiBody(
            baseMass=0.14,
            baseCollisionShapeIndex=col_ball,
            baseVisualShapeIndex=vis_ball,
            basePosition=self._ball_start,
            physicsClientId=self.client,
        )
        pb.changeDynamics(
            self.ball_id, -1, restitution=0.55, lateralFriction=0.35,
            rollingFriction=0.02, physicsClientId=self.client,
        )
        self._build_low_incline()
        self._build_lever_pedestal()
        self._build_target_marker()

        self.robot_id = self._load_humanoid()
        self.n_joints = pb.getNumJoints(self.robot_id, physicsClientId=self.client)

        self.joint_map: dict[int, str] = {}
        self.joint_by_var: dict[str, int] = {}
        for i in range(self.n_joints):
            info = pb.getJointInfo(self.robot_id, i, physicsClientId=self.client)
            jname = info[1].decode("utf-8")
            for key, varname in self._JOINT_MAP.items():
                if key.lower() in jname.lower():
                    self.joint_map[i] = varname
                    self.joint_by_var[varname] = i
                    break

        self.link_names: list[str] = []
        self._joint_types: list[int] = []
        for i in range(self.n_joints):
            info = pb.getJointInfo(self.robot_id, i, physicsClientId=self.client)
            self.link_names.append(info[12].decode("utf-8"))
            self._joint_types.append(int(info[2]))

        self._neck_euler = np.zeros(3, dtype=float)
        self._spine_euler = np.zeros(3, dtype=float)
        for i in range(self.n_joints):
            info = pb.getJointInfo(self.robot_id, i, physicsClientId=self.client)
            jname = info[1].decode("utf-8").lower()
            if jname == "neck":
                self.joint_by_var["neck_yaw"] = i
                self.joint_by_var["neck_pitch"] = i
            elif jname == "spine":
                self.joint_by_var["spine_yaw"] = i
                self.joint_by_var["spine_pitch"] = i

        # ── Fixed root constraint (None = not applied) ───────────────────────
        self._root_constraint: int | None = None

        self.reset_stance()

        # Применяем fixed_root ПОСЛЕ reset_stance
        if fixed_root:
            self.enable_fixed_root()

        self._maybe_start_physics_bg()
        self._init_instrumental()

    def _maybe_start_physics_bg(self) -> None:
        """
        Опционально: непрерывный stepSimulation в фоне (RKK_PHYSICS_BG_HZ, напр. 120).
        Тогда step(n) на главном потоке не вызывается — физика идёт между тиками агента.
        Все вызовы PyBullet сериализуются через _physics_lock (в т.ч. камера, reset).
        """
        try:
            hz = float(os.environ.get("RKK_PHYSICS_BG_HZ", "0"))
        except ValueError:
            hz = 0.0
        if hz <= 0 or hz > 480:
            return
        self._bg_hz = hz
        self._bg_running = True
        dt = 1.0 / hz
        cid = self.client

        def _loop() -> None:
            while self._bg_running:
                t0 = time.perf_counter()
                with self._physics_lock:
                    pb.stepSimulation(physicsClientId=cid)
                    self._tick_hidden_state()
                    self._apply_friction_to_ankle_joints()
                elapsed = time.perf_counter() - t0
                slp = dt - elapsed
                if slp > 0:
                    time.sleep(slp)

        self._bg_thread = threading.Thread(
            target=_loop, daemon=True, name="RKK-PyBullet-bg-physics"
        )
        self._bg_thread.start()
        print(f"[HumanoidEnv] Background physics ~{hz:.0f} Hz (RKK_PHYSICS_BG_HZ); main step() is no-op")

    def _stop_physics_bg(self) -> None:
        self._bg_running = False
        th = self._bg_thread
        if th is not None and th.is_alive():
            th.join(timeout=1.0)
        self._bg_thread = None
        self._bg_hz = 0.0

    # ── Fixed root constraint ─────────────────────────────────────────────────
    def enable_fixed_root(self) -> None:
        """
        Фиксируем базу робота в мировых координатах через JOINT_FIXED constraint.
        Вызывается ПОСЛЕ reset_stance() чтобы зафиксировать стабильную позу.
        """
        with self._physics_lock:
            if self._root_constraint is not None:
                return  # уже зафиксирован

            pos, orn = pb.getBasePositionAndOrientation(
                self.robot_id, physicsClientId=self.client
            )
            self._root_constraint = pb.createConstraint(
                self.robot_id, -1,           # parent: robot base link
                -1, -1,                       # child: world frame
                pb.JOINT_FIXED,
                [0, 0, 0],                    # joint axis (unused for fixed)
                [0, 0, 0],                    # parent frame position (local)
                list(pos),                    # child frame position (world)
                parentFrameOrientation=[0, 0, 0, 1],
                childFrameOrientation=list(orn),
                physicsClientId=self.client,
            )
            # Снижаем максимальную силу constraint чтобы не было артефактов
            pb.changeConstraint(
                self._root_constraint,
                maxForce=5000.0,
                physicsClientId=self.client,
            )
            print(f"[HumanoidEnv] Fixed root constraint #{self._root_constraint} at z={pos[2]:.3f}")

    def disable_fixed_root(self) -> None:
        """Снимаем фиксацию базы — робот снова свободно движется."""
        with self._physics_lock:
            if self._root_constraint is None:
                return
            try:
                pb.removeConstraint(self._root_constraint, physicsClientId=self.client)
            except Exception as e:
                print(f"[HumanoidEnv] removeConstraint error: {e}")
            self._root_constraint = None
            print("[HumanoidEnv] Fixed root constraint removed")

    @property
    def fixed_root(self) -> bool:
        return self._root_constraint is not None

    # ─────────────────────────────────────────────────────────────────────────
    def _build_ramp(self):
        import math
        angle = math.radians(15)
        half  = [1.0, 0.5, 0.03]
        col = pb.createCollisionShape(pb.GEOM_BOX, halfExtents=half, physicsClientId=self.client)
        vis = pb.createVisualShape(pb.GEOM_BOX, halfExtents=half,
                                    rgbaColor=[0.4, 0.35, 0.3, 1.0], physicsClientId=self.client)
        orn = pb.getQuaternionFromEuler([angle, 0, 0])
        pb.createMultiBody(
            baseMass=0, baseCollisionShapeIndex=col, baseVisualShapeIndex=vis,
            basePosition=[2.0, 0, 0.3*math.sin(angle) + 0.03],
            baseOrientation=orn, physicsClientId=self.client
        )

    def _build_low_incline(self) -> None:
        """Второй пологий наклон — другое направление, больше контактов / скольжения."""
        import math
        ang = math.radians(10)
        half = [0.55, 0.4, 0.028]
        col = pb.createCollisionShape(pb.GEOM_BOX, halfExtents=half, physicsClientId=self.client)
        vis = pb.createVisualShape(
            pb.GEOM_BOX, halfExtents=half,
            rgbaColor=[0.32, 0.38, 0.42, 1.0], physicsClientId=self.client,
        )
        orn = pb.getQuaternionFromEuler([0, 0, ang])
        pb.createMultiBody(
            baseMass=0, baseCollisionShapeIndex=col, baseVisualShapeIndex=vis,
            basePosition=[-1.35, -0.95, 0.22 * math.sin(ang) + 0.03],
            baseOrientation=orn, physicsClientId=self.client,
        )

    def _build_lever_pedestal(self) -> None:
        """Визуальный «рычаг» / кнопка — каузальность через lever_pin (проксимити объектов)."""
        half = [0.07, 0.07, 0.055]
        col = pb.createCollisionShape(pb.GEOM_BOX, halfExtents=half, physicsClientId=self.client)
        vis = pb.createVisualShape(
            pb.GEOM_BOX, halfExtents=half,
            rgbaColor=[0.88, 0.62, 0.15, 1.0], physicsClientId=self.client,
        )
        lc = self.lever_center
        pb.createMultiBody(
            baseMass=0, baseCollisionShapeIndex=col, baseVisualShapeIndex=vis,
            basePosition=[float(lc[0]), float(lc[1]), float(lc[2])],
            physicsClientId=self.client,
        )

    def _build_target_marker(self) -> None:
        """Плоская мишень — зона доставки куба (target_dist в observe)."""
        half = [0.22, 0.22, 0.012]
        col = pb.createCollisionShape(pb.GEOM_BOX, halfExtents=half, physicsClientId=self.client)
        vis = pb.createVisualShape(
            pb.GEOM_BOX, halfExtents=half,
            rgbaColor=[0.15, 0.85, 0.55, 0.85], physicsClientId=self.client,
        )
        tp = self.target_pad
        pb.createMultiBody(
            baseMass=0, baseCollisionShapeIndex=col, baseVisualShapeIndex=vis,
            basePosition=[float(tp[0]), float(tp[1]), float(tp[2])],
            physicsClientId=self.client,
        )

    def _load_humanoid(self) -> int:
        local = Path(__file__).resolve().parent / "data" / "humanoid" / "humanoid.urdf"
        candidates: list[str] = []
        if local.is_file():
            candidates.append(str(local))
        candidates.extend(["humanoid/humanoid.urdf", "humanoid.urdf", "atlas/atlas.urdf"])

        self._reset_base_pos = [0.0, 0.0, HUMANOID_URDF_SPAWN_Z]
        self._reset_base_orn = pb.getQuaternionFromEuler(HUMANOID_URDF_STAND_EULER)

        for rel in candidates:
            try:
                rel_l = rel.replace("\\", "/").lower()
                gs = 1.0 if "atlas" in rel_l else HUMANOID_URDF_GLOBAL_SCALING
                if "atlas" in rel_l:
                    pos = [0.0, 0.0, 1.0]
                    orn = pb.getQuaternionFromEuler([0.0, 0.0, 0.0])
                else:
                    pos = [0.0, 0.0, HUMANOID_URDF_SPAWN_Z]
                    orn = pb.getQuaternionFromEuler(HUMANOID_URDF_STAND_EULER)
                robot = pb.loadURDF(
                    rel,
                    basePosition=pos,
                    baseOrientation=orn,
                    flags=pb.URDF_USE_SELF_COLLISION,
                    globalScaling=gs,
                    physicsClientId=self.client,
                )
                self._reset_base_pos = list(pos)
                self._reset_base_orn = orn
                print(f"[HumanoidEnv] Loaded: {rel}" + (f" (globalScaling={gs})" if gs != 1.0 else ""))
                return robot
            except Exception as e:
                print(f"[HumanoidEnv] Could not load {rel}: {e}")

        return self._build_custom_humanoid()

    def _build_custom_humanoid(self) -> int:
        masses     = [8.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 3.0, 3.0, 1.5, 1.5, 0.5, 0.5]
        col_shapes, vis_shapes, positions, orientations = [], [], [], []
        inertial_pos, inertial_orn, parents, jtypes, jaxes = [], [], [], [], []

        segments = [
            ([0.1,0.15,0.12], [0.7,0.7,0.7,1], [0,0,0],      0),
            ([0.05,0.05,0.05],[0.9,0.8,0.7,1], [0,0,0.22],   0),
            ([0.04,0.04,0.14],[0.7,0.7,0.7,1], [-0.18,0,0.1],0),
            ([0.04,0.04,0.14],[0.7,0.7,0.7,1], [0.18,0,0.1], 0),
            ([0.03,0.03,0.12],[0.8,0.8,0.8,1], [-0.18,0,-0.14],2),
            ([0.03,0.03,0.12],[0.8,0.8,0.8,1], [0.18,0,-0.14],3),
            ([0.06,0.06,0.15],[0.7,0.7,0.7,1], [-0.10,0,-0.15],0),
            ([0.06,0.06,0.15],[0.7,0.7,0.7,1], [0.10,0,-0.15], 0),
            ([0.05,0.05,0.14],[0.8,0.8,0.8,1], [0,0,-0.15],   6),
            ([0.05,0.05,0.14],[0.8,0.8,0.8,1], [0,0,-0.15],   7),
            ([0.09,0.04,0.035],[0.6,0.6,0.6,1], [0,0,-0.14],  8),
            ([0.09,0.04,0.035],[0.6,0.6,0.6,1], [0,0,-0.14],  9),
        ]
        for i, (he, col_c, pos, par) in enumerate(segments):
            col = pb.createCollisionShape(pb.GEOM_BOX, halfExtents=he, physicsClientId=self.client)
            vis = pb.createVisualShape(pb.GEOM_BOX, halfExtents=he, rgbaColor=col_c, physicsClientId=self.client)
            col_shapes.append(col); vis_shapes.append(vis)
            positions.append(pos); orientations.append([0,0,0,1])
            inertial_pos.append([0,0,0]); inertial_orn.append([0,0,0,1])
            parents.append(par)
            jtypes.append(pb.JOINT_REVOLUTE)
            jaxes.append([1,0,0] if i not in [1,2,3] else [0,1,0])

        base_col = pb.createCollisionShape(pb.GEOM_BOX, halfExtents=[0.1,0.08,0.12], physicsClientId=self.client)
        base_vis = pb.createVisualShape(pb.GEOM_BOX, halfExtents=[0.1,0.08,0.12],
                                         rgbaColor=[0.5,0.5,0.6,1.0], physicsClientId=self.client)
        robot = pb.createMultiBody(
            baseMass=masses[0],
            baseCollisionShapeIndex=base_col,
            baseVisualShapeIndex=base_vis,
            basePosition=[0, 0, 1.0],
            linkMasses=masses[1:len(segments)+1],
            linkCollisionShapeIndices=col_shapes,
            linkVisualShapeIndices=vis_shapes,
            linkPositions=positions,
            linkOrientations=orientations,
            linkInertialFramePositions=inertial_pos,
            linkInertialFrameOrientations=inertial_orn,
            linkParentIndices=parents,
            linkJointTypes=jtypes,
            linkJointAxis=jaxes,
            physicsClientId=self.client,
        )
        custom_map = {6:"lhip",7:"rhip",8:"lknee",9:"rknee",2:"lshoulder",3:"rshoulder",4:"lelbow",5:"relbow"}
        self.joint_map   = custom_map
        self.joint_by_var = {v:k for k,v in custom_map.items()}
        self._reset_base_pos = [0.0, 0.0, 1.0]
        self._reset_base_orn = pb.getQuaternionFromEuler([0.0, 0.0, 0.0])
        return robot

    def step(self, n: int = 10):
        if self._bg_hz > 0:
            return
        with self._physics_lock:
            for _ in range(n):
                pb.stepSimulation(physicsClientId=self.client)
                self._tick_hidden_state()
                self._apply_friction_to_ankle_joints()

    def _motor_relax_velocity(self) -> None:
        rid, cid = self.robot_id, self.client
        quat_id = [0.0, 0.0, 0.0, 1.0]
        motor_m = getattr(pb, "setJointMotorControlMultiDof", None)
        for i in range(self.n_joints):
            jt = self._joint_types[i]
            if jt == pb.JOINT_FIXED:
                continue
            if jt == pb.JOINT_SPHERICAL and callable(motor_m):
                motor_m(
                    rid, i, pb.POSITION_CONTROL,
                    targetPosition=quat_id,
                    positionGain=1.0, velocityGain=0.35,
                    maxVelocity=8.0, force=[220.0, 220.0, 220.0],
                    physicsClientId=cid,
                )
            else:
                pb.setJointMotorControl2(
                    rid, i, controlMode=pb.POSITION_CONTROL,
                    targetPosition=0.0,
                    positionGain=0.55, velocityGain=0.12,
                    force=100.0, physicsClientId=cid,
                )

    def _motor_stabilize_neutral_pose(self) -> None:
        rid, cid = self.robot_id, self.client
        quat_id = [0.0, 0.0, 0.0, 1.0]
        motor_m = getattr(pb, "setJointMotorControlMultiDof", None)
        for i in range(self.n_joints):
            jt = self._joint_types[i]
            if jt == pb.JOINT_FIXED:
                continue
            if jt == pb.JOINT_SPHERICAL and callable(motor_m):
                motor_m(
                    rid, i, pb.POSITION_CONTROL,
                    targetPosition=quat_id,
                    positionGain=1.0, velocityGain=0.35,
                    maxVelocity=8.0, force=[220.0, 220.0, 220.0],
                    physicsClientId=cid,
                )
            else:
                pb.setJointMotorControl2(
                    rid, i, controlMode=pb.POSITION_CONTROL,
                    targetPosition=0.0,
                    positionGain=0.55, velocityGain=0.12,
                    force=100.0, physicsClientId=cid,
                )

    def _snap_base_spine_vertical(self) -> None:
        if not {"root", "neck"}.issubset(set(self.link_names)):
            return
        rid, cid = self.robot_id, self.client
        i_root = self.link_names.index("root")
        i_neck = self.link_names.index("neck")
        st_r = pb.getLinkState(rid, i_root, computeForwardKinematics=1, physicsClientId=cid)
        st_n = pb.getLinkState(rid, i_neck, computeForwardKinematics=1, physicsClientId=cid)
        pr = np.array(st_r[4][:3], dtype=float)
        pn = np.array(st_n[4][:3], dtype=float)
        spine = pn - pr
        ln = float(np.linalg.norm(spine))
        if ln < 1e-6:
            return
        spine = spine / ln
        ez = np.array([0.0, 0.0, 1.0], dtype=float)
        c = float(np.clip(np.dot(spine, ez), -1.0, 1.0))
        if c > 0.998:
            return
        axis = np.cross(spine, ez)
        an = float(np.linalg.norm(axis))
        if an < 1e-8:
            return
        axis = axis / an
        ang = float(np.arccos(c))
        dq = _np_quat_from_axis_angle(axis, ang)
        pos, orn = pb.getBasePositionAndOrientation(rid, physicsClientId=cid)
        new_orn = _np_quat_mul(dq, list(orn))
        pb.resetBasePositionAndOrientation(rid, pos, new_orn, physicsClientId=cid)

    def reset_stance(self) -> None:
        """
        Сброс позы. Если fixed_root был активен — временно снимаем constraint,
        сбрасываем, и заново применяем.
        """
        with self._physics_lock:
            self._reset_stance_locked()

    def _reset_stance_locked(self) -> None:
        had_fixed = self._root_constraint is not None
        if had_fixed:
            if self._root_constraint is not None:
                try:
                    pb.removeConstraint(self._root_constraint, physicsClientId=self.client)
                except Exception as e:
                    print(f"[HumanoidEnv] removeConstraint error: {e}")
                self._root_constraint = None

        self._neck_euler[:] = 0.0
        self._spine_euler[:] = 0.0
        rid = self.robot_id
        cid = self.client
        pb.resetBasePositionAndOrientation(
            rid, self._reset_base_pos, self._reset_base_orn,
            physicsClientId=cid,
        )
        pb.resetBaseVelocity(rid, [0, 0, 0], [0, 0, 0], physicsClientId=cid)

        dof_reset = getattr(pb, "resetJointStateMultiDof", None)
        quat_identity = [0.0, 0.0, 0.0, 1.0]
        omega_zero = [0.0, 0.0, 0.0]

        for i in range(self.n_joints):
            jt = self._joint_types[i]
            if jt == pb.JOINT_FIXED:
                continue
            if jt == pb.JOINT_SPHERICAL and callable(dof_reset):
                dof_reset(rid, i, quat_identity, omega_zero, physicsClientId=cid)
            else:
                pb.resetJointState(rid, i, targetValue=0.0, targetVelocity=0.0, physicsClientId=cid)

        self._motor_stabilize_neutral_pose()
        for _ in range(260):
            pb.stepSimulation(physicsClientId=cid)

        self._motor_relax_velocity()
        for _ in range(80):
            pb.stepSimulation(physicsClientId=cid)

        pb.resetBaseVelocity(rid, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], physicsClientId=cid)

        self._snap_base_spine_vertical()
        pb.resetBaseVelocity(rid, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], physicsClientId=cid)
        self._motor_relax_velocity()
        for _ in range(72):
            pb.stepSimulation(physicsClientId=cid)
        pb.resetBaseVelocity(rid, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], physicsClientId=cid)

        # Восстанавливаем fixed_root если был (уже под lock — RLock)
        if had_fixed:
            pos, orn = pb.getBasePositionAndOrientation(
                self.robot_id, physicsClientId=cid
            )
            self._root_constraint = pb.createConstraint(
                self.robot_id, -1, -1, -1,
                pb.JOINT_FIXED,
                [0, 0, 0], [0, 0, 0],
                list(pos),
                parentFrameOrientation=[0, 0, 0, 1],
                childFrameOrientation=list(orn),
                physicsClientId=cid,
            )
            pb.changeConstraint(
                self._root_constraint, maxForce=5000.0, physicsClientId=cid,
            )

        if getattr(self, "ball_id", None) is not None:
            pb.resetBasePositionAndOrientation(
                self.ball_id,
                self._ball_start,
                [0, 0, 0, 1],
                physicsClientId=cid,
            )
            pb.resetBaseVelocity(
                self.ball_id, [0, 0, 0], [0, 0, 0], physicsClientId=cid,
            )

        self._reset_instrumental_hidden()

    def set_joint(self, var_name: str, target_pos: float):
        with self._physics_lock:
            if var_name not in self.joint_by_var:
                return
            jid = self.joint_by_var[var_name]
            lo, hi = _RANGES.get(var_name, (-2.0, 2.0))
            real_pos = float(np.clip(target_pos * (hi - lo) + lo, lo, hi))
            rid, cid = self.robot_id, self.client
            jt = self._joint_types[jid]
            motor_m = getattr(pb, "setJointMotorControlMultiDof", None)

            if var_name in ("neck_yaw", "neck_pitch"):
                if not callable(motor_m) or jt != pb.JOINT_SPHERICAL:
                    return
                if var_name == "neck_yaw":
                    self._neck_euler[2] = 0.55 * real_pos
                else:
                    self._neck_euler[0] = 0.45 * real_pos
                ex, ey, ez = float(self._neck_euler[0]), float(self._neck_euler[1]), float(self._neck_euler[2])
                q = pb.getQuaternionFromEuler((ex, ey, ez))
                motor_m(rid, jid, pb.POSITION_CONTROL, targetPosition=list(q),
                        positionGain=0.62, velocityGain=0.18, maxVelocity=4.0,
                        force=[110.0, 110.0, 110.0], physicsClientId=cid)
                return

            if var_name in ("spine_yaw", "spine_pitch"):
                if not callable(motor_m) or jt != pb.JOINT_SPHERICAL:
                    return
                if var_name == "spine_yaw":
                    self._spine_euler[2] = 0.50 * real_pos
                else:
                    self._spine_euler[0] = 0.40 * real_pos
                ex, ey, ez = float(self._spine_euler[0]), float(self._spine_euler[1]), float(self._spine_euler[2])
                q = pb.getQuaternionFromEuler((ex, ey, ez))
                motor_m(rid, jid, pb.POSITION_CONTROL, targetPosition=list(q),
                        positionGain=0.70, velocityGain=0.20, maxVelocity=3.5,
                        force=[180.0, 180.0, 180.0], physicsClientId=cid)
                return

            if jt == pb.JOINT_SPHERICAL and callable(motor_m):
                if var_name == "lshoulder":
                    q = pb.getQuaternionFromEuler((0.32 * real_pos, 0.42 * real_pos, 0.28 * real_pos))
                elif var_name == "rshoulder":
                    q = pb.getQuaternionFromEuler((0.32 * real_pos, -0.42 * real_pos, -0.28 * real_pos))
                elif var_name == "lhip":
                    q = pb.getQuaternionFromEuler((0.1 * real_pos, 0.42 * real_pos, 0.05 * real_pos))
                elif var_name == "rhip":
                    q = pb.getQuaternionFromEuler((0.1 * real_pos, -0.42 * real_pos, -0.05 * real_pos))
                elif var_name == "lankle":
                    q = pb.getQuaternionFromEuler((-0.22 * real_pos, 0.1 * real_pos, 0.0))
                elif var_name == "rankle":
                    q = pb.getQuaternionFromEuler((-0.22 * real_pos, -0.1 * real_pos, 0.0))
                else:
                    q = [0.0, 0.0, 0.0, 1.0]
                motor_m(rid, jid, pb.POSITION_CONTROL, targetPosition=list(q),
                        positionGain=0.52, velocityGain=0.15, maxVelocity=5.5,
                        force=[165.0, 165.0, 165.0], physicsClientId=cid)
            else:
                pb.setJointMotorControl2(
                    rid, jid, controlMode=pb.POSITION_CONTROL,
                    targetPosition=real_pos,
                    positionGain=0.5, velocityGain=0.1, force=80.0, physicsClientId=cid,
                )

    def get_com(self) -> tuple[np.ndarray, np.ndarray]:
        pos, orn = pb.getBasePositionAndOrientation(self.robot_id, physicsClientId=self.client)
        euler    = pb.getEulerFromQuaternion(orn)
        return np.array(pos), np.array(euler)

    def get_joint_angle(self, var_name: str) -> float:
        if var_name == "neck_yaw":   return float(self._neck_euler[2])
        if var_name == "neck_pitch": return float(self._neck_euler[0])
        if var_name == "spine_yaw":  return float(self._spine_euler[2])
        if var_name == "spine_pitch":return float(self._spine_euler[0])
        if var_name not in self.joint_by_var:
            return 0.0
        jid = self.joint_by_var[var_name]
        rid, cid = self.robot_id, self.client
        jt = self._joint_types[jid]
        gmd = getattr(pb, "getJointStateMultiDof", None)
        if jt == pb.JOINT_SPHERICAL and callable(gmd):
            q = gmd(rid, jid, physicsClientId=cid)[0]
            if isinstance(q, (list, tuple)) and len(q) >= 4:
                x, y, z, w = float(q[0]), float(q[1]), float(q[2]), float(q[3])
                imag_norm = float(np.sqrt(x*x + y*y + z*z))
                w_cl = float(np.clip(abs(w), 0.0, 1.0))
                ang = 2.0 * float(np.arctan2(imag_norm, w_cl))
                if imag_norm < 1e-8:
                    return 0.0
                ax, ay, az = abs(x), abs(y), abs(z)
                if ax >= ay and ax >= az:
                    sgn = 1.0 if x >= 0.0 else -1.0
                elif ay >= az:
                    sgn = 1.0 if y >= 0.0 else -1.0
                else:
                    sgn = 1.0 if z >= 0.0 else -1.0
                return float(np.clip(sgn * ang, -2.0, 2.0))
        st = pb.getJointState(rid, jid, physicsClientId=cid)
        return float(st[0])

    def get_foot_heights(self) -> tuple[float, float]:
        lz = rz = 0.05
        n_links = pb.getNumJoints(self.robot_id, physicsClientId=self.client)
        zs = []
        for i in range(n_links):
            try:
                st = pb.getLinkState(self.robot_id, i, physicsClientId=self.client)
                zs.append((i, st[4][2]))
            except Exception:
                pass
        zs.sort(key=lambda x: x[1])
        if len(zs) >= 2:
            lz = max(0.0, zs[0][1])
            rz = max(0.0, zs[1][1])
        return float(lz), float(rz)

    def get_cube_state(self) -> list[np.ndarray]:
        positions = []
        for cid in self.cube_ids:
            pos, _ = pb.getBasePositionAndOrientation(cid, physicsClientId=self.client)
            positions.append(np.array(pos))
        return positions

    def _sandbox_dynamic_positions(self) -> list[np.ndarray]:
        out: list[np.ndarray] = []
        if getattr(self, "ball_id", None) is not None:
            pos, _ = pb.getBasePositionAndOrientation(self.ball_id, physicsClientId=self.client)
            out.append(np.array(pos, dtype=np.float64))
        for cid in self.cube_ids:
            pos, _ = pb.getBasePositionAndOrientation(cid, physicsClientId=self.client)
            out.append(np.array(pos, dtype=np.float64))
        return out

    def _compute_lever_pin(self) -> float:
        pts = self._sandbox_dynamic_positions()
        if not pts:
            return 0.0
        lc = self.lever_center[:2]
        d = min(float(np.linalg.norm(p[:2] - lc)) for p in pts)
        return float(np.clip(1.0 - d / max(self.lever_trigger_r * 1.55, 1e-6), 0.0, 1.0))

    def _compute_target_dist(self) -> float:
        if not self.cube_ids:
            return 2.5
        tg = self.target_pad[:2]
        best = 1e9
        for cid in self.cube_ids:
            pos, _ = pb.getBasePositionAndOrientation(cid, physicsClientId=self.client)
            best = min(best, float(np.linalg.norm(np.array(pos[:2]) - tg)))
        return float(best)

    def get_state(self) -> dict:
        with self._physics_lock:
            com, euler = self.get_com()
            lf, rf = self.get_foot_heights()
            s = {}
            s["com_x"]       = float(com[0])
            s["com_y"]       = float(com[1])
            s["com_z"]       = float(com[2])
            s["torso_roll"]  = float(euler[0])
            s["torso_pitch"] = float(euler[1])
            for v in SPINE_VARS + HEAD_VARS:
                s[v] = self.get_joint_angle(v)
            for v in LEG_VARS + ARM_VARS:
                s[v] = self.get_joint_angle(v)
            s["lfoot_z"] = lf
            s["rfoot_z"] = rf
            for i, cp in enumerate(self.get_cube_state()):
                s[f"cube{i}_x"] = float(cp[0])
                s[f"cube{i}_y"] = float(cp[1])
                s[f"cube{i}_z"] = float(cp[2])
            if getattr(self, "ball_id", None) is not None:
                bp, _ = pb.getBasePositionAndOrientation(self.ball_id, physicsClientId=self.client)
                s["ball_x"] = float(bp[0])
                s["ball_y"] = float(bp[1])
                s["ball_z"] = float(bp[2])
            else:
                s["ball_x"] = s["ball_y"] = 0.0
                s["ball_z"] = 0.12
            s["lever_pin"] = self._compute_lever_pin()
            s["target_dist"] = self._compute_target_dist()
            s["floor_friction"] = self._compute_floor_friction_effect()
            s["stack_height"] = float(self._stack_height)
            s["stability_score"] = float(self._stability_score)
            return s

    def _named_link_world_positions(self) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        for i in range(self.n_joints):
            name = self.link_names[i]
            st = pb.getLinkState(
                self.robot_id, i, computeForwardKinematics=1, physicsClientId=self.client,
            )
            out[name] = np.array(st[4][:3], dtype=float)
        return out

    def _sole_center_world(self, ankle_link: str) -> np.ndarray | None:
        if ankle_link not in self.link_names:
            return None
        i = self.link_names.index(ankle_link)
        st = pb.getLinkState(self.robot_id, i, computeForwardKinematics=1, physicsClientId=self.client)
        lw = np.array(st[4][:3], dtype=float)
        R = np.array(pb.getMatrixFromQuaternion(st[5]), dtype=float).reshape(3, 3)
        local = np.array([0.15, -0.09, -0.16], dtype=float) * float(HUMANOID_URDF_GLOBAL_SCALING)
        return lw + R @ local

    def get_ankle_quaternions_three_js(self) -> list[dict[str, float]]:
        with self._physics_lock:
            out: list[dict[str, float]] = []
            rid, cid = self.robot_id, self.client
            for name in ("left_ankle", "right_ankle"):
                if name not in self.link_names:
                    out.append({"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0})
                    continue
                i = self.link_names.index(name)
                st = pb.getLinkState(rid, i, computeForwardKinematics=1, physicsClientId=cid)
                R = np.array(pb.getMatrixFromQuaternion(st[5]), dtype=float).reshape(3, 3)
                R3 = _PB_VEC_TO_THREE @ R @ _PB_VEC_TO_THREE.T
                x, y, z, w = _rotmat_to_xyzw(R3)
                out.append({"x": float(x), "y": float(y), "z": float(z), "w": float(w)})
            return out

    def _skeleton_from_urdf_links(self) -> list[dict] | None:
        need = {
            "neck", "spine", "chest", "root",
            "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
            "left_wrist", "right_wrist",
            "left_hip", "right_hip", "left_knee", "right_knee",
            "left_ankle", "right_ankle",
        }
        if not need.issubset(set(self.link_names)):
            return None
        p = self._named_link_world_positions()

        def vec(name: str) -> np.ndarray:
            return p[name].copy()

        neck_v = vec("neck")
        chest_v = vec("chest")
        up = neck_v - chest_v
        ln = float(np.linalg.norm(up))
        if ln < 1e-5:
            up = np.array([0.0, 0.0, 1.0], dtype=float)
        else:
            up = up / ln
        sk = float(HUMANOID_URDF_GLOBAL_SCALING) / float(HUMANOID_URDF_LEGACY_SCALE)
        head_v = neck_v + 0.26 * sk * up

        order = [
            head_v, neck_v, vec("spine"), vec("root"),
            vec("left_shoulder"), vec("right_shoulder"),
            vec("left_elbow"), vec("right_elbow"),
            vec("left_wrist"), vec("right_wrist"),
            vec("left_hip"), vec("right_hip"),
            vec("left_knee"), vec("right_knee"),
            vec("left_ankle"), vec("right_ankle"),
        ]
        ls = self._sole_center_world("left_ankle")
        rs = self._sole_center_world("right_ankle")
        if ls is not None and rs is not None:
            order.append(ls)
            order.append(rs)
        return [{"x": float(v[0]), "y": float(v[1]), "z": float(v[2])} for v in order]

    def get_all_link_positions(self) -> list[dict]:
        with self._physics_lock:
            urdf_pts = self._skeleton_from_urdf_links()
            if urdf_pts is not None:
                return urdf_pts
            pos, _ = pb.getBasePositionAndOrientation(self.robot_id, physicsClientId=self.client)
            cx, cy, cz = float(pos[0]), float(pos[1]), float(pos[2])
            j = {v: self.get_joint_angle(v) for v in LEG_VARS + ARM_VARS + SPINE_VARS + HEAD_VARS}
            return _forward_kinematics_skeleton(cx, cy, cz, j)

    def get_cube_positions(self) -> list[dict]:
        with self._physics_lock:
            result = []
            for cid in self.cube_ids:
                pos, _ = pb.getBasePositionAndOrientation(cid, physicsClientId=self.client)
                result.append({"x": float(pos[0]), "y": float(pos[1]), "z": float(pos[2])})
            return result

    def get_sandbox_scene_extras(self) -> dict:
        with self._physics_lock:
            ball = {"x": 0.0, "y": 0.0, "z": 0.12}
            if getattr(self, "ball_id", None) is not None:
                p, _ = pb.getBasePositionAndOrientation(self.ball_id, physicsClientId=self.client)
                ball = {"x": float(p[0]), "y": float(p[1]), "z": float(p[2])}
            lp = self._compute_lever_pin()
            return {
                "ball": ball,
                "lever": {
                    "x": float(self.lever_center[0]),
                    "y": float(self.lever_center[1]),
                    "z": float(self.lever_center[2]),
                    "pressed": lp > 0.75,
                    "pin": round(lp, 3),
                },
                "delivery_target": {
                    "x": float(self.target_pad[0]),
                    "y": float(self.target_pad[1]),
                    "z": float(self.target_pad[2]),
                },
            }

    def _link_world_pos(self, link_name: str) -> np.ndarray | None:
        if link_name not in self.link_names:
            return None
        i = self.link_names.index(link_name)
        st = pb.getLinkState(self.robot_id, i, computeForwardKinematics=1, physicsClientId=self.client)
        return np.array(st[4][:3], dtype=float)

    def _ego_camera_rt(self) -> tuple[list[float], list[float], list[float]] | None:
        names = set(self.link_names)
        if "neck" not in names or "chest" not in names:
            return None
        pos_n = self._link_world_pos("neck")
        pos_c = self._link_world_pos("chest")
        if pos_n is None or pos_c is None:
            return None
        up = pos_n - pos_c
        ln = float(np.linalg.norm(up))
        if ln < 1e-6:
            up = np.array([0.0, 0.0, 1.0], dtype=float)
        else:
            up = up / ln

        fwd: np.ndarray | None = None
        if {"left_shoulder", "right_shoulder"}.issubset(names):
            pl = self._link_world_pos("left_shoulder")
            pr = self._link_world_pos("right_shoulder")
            if pl is not None and pr is not None:
                right_body = pr - pl
                rn = float(np.linalg.norm(right_body))
                if rn >= 1e-5:
                    right_body = right_body / rn
                    fwd = np.cross(up, right_body)
                    fn = float(np.linalg.norm(fwd))
                    if fn >= 1e-5:
                        fwd = fwd / fn
                    else:
                        fwd = None
        if fwd is None:
            i_neck = self.link_names.index("neck")
            st_n = pb.getLinkState(self.robot_id, i_neck, computeForwardKinematics=1, physicsClientId=self.client)
            R = np.array(pb.getMatrixFromQuaternion(st_n[5]), dtype=float).reshape(3, 3)
            local_fwd = np.array([1.0, 0.0, 0.0], dtype=float)
            cand = R @ local_fwd
            cand = cand - float(np.dot(cand, up)) * up
            fn = float(np.linalg.norm(cand))
            if fn < 1e-5:
                return None
            fwd = cand / fn

        sk = float(HUMANOID_URDF_GLOBAL_SCALING) / float(HUMANOID_URDF_LEGACY_SCALE)
        head_anchor = pos_n + 0.26 * sk * up
        eye = head_anchor + 0.04 * sk * up + 0.06 * sk * fwd
        target = eye + 2.4 * sk * fwd
        return eye.tolist(), target.tolist(), up.tolist()

    def get_frame_base64(
        self,
        view: str | None = None,
        width: int = 480,
        height: int = 360,
        jpeg_quality: int = 85,
    ) -> str | None:
        if not PIL_AVAILABLE:
            return None
        try:
            with self._physics_lock:
                eg = self._ego_camera_rt()
                if eg is None:
                    vm = pb.computeViewMatrix(
                        [2.2, -2.2, 1.6], [0, 0, 0.75], [0, 0, 1],
                        physicsClientId=self.client,
                    )
                    ego_cam = False
                else:
                    eye, tgt, cup = eg
                    vm = pb.computeViewMatrix(eye, tgt, cup, physicsClientId=self.client)
                    ego_cam = True
                pm = pb.computeProjectionMatrixFOV(
                    fov=60, aspect=width/height, nearVal=0.1, farVal=15.0,
                    physicsClientId=self.client)
                need = width * height * 4
                rgba = None
                hwgl = getattr(pb, "ER_BULLET_HARDWARE_OPENGL", None)
                for renderer in (hwgl, pb.ER_TINY_RENDERER):
                    if renderer is None:
                        continue
                    try:
                        _, _, rgba_try, _, _ = pb.getCameraImage(
                            width, height, vm, pm,
                            renderer=renderer,
                            physicsClientId=self.client,
                        )
                        pix_try = np.asarray(rgba_try, dtype=np.uint8).reshape(-1)
                        if pix_try.size >= need:
                            rgba = rgba_try
                            break
                    except Exception:
                        continue
                if rgba is None:
                    raise ValueError("getCameraImage: all renderers failed")
                pix = np.asarray(rgba, dtype=np.uint8).reshape(-1)
                if pix.size < need:
                    raise ValueError(f"camera pixels {pix.size} < expected {need}")
                rgb = pix[:need].reshape((height, width, 4))[:, :, :3]
                if ego_cam:
                    rgb = np.ascontiguousarray(rgb[:, ::-1, :])
                img = PILImage.fromarray(rgb)
                buf = BytesIO()
                q = int(np.clip(jpeg_quality, 40, 95))
                img.save(buf, format="JPEG", quality=q, optimize=True)
                return base64.b64encode(buf.getvalue()).decode()
        except Exception as e:
            print(f"[HumanoidEnv] Camera error: {e}")
            return None

    def __del__(self):
        try:
            self._stop_physics_bg()
            with self._physics_lock:
                pb.disconnect(self.client)
        except Exception:
            pass


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
        steps_per_do: int = 24,
        fixed_root: bool = False,
    ):
        self.device = device or torch.device("cpu")
        try:
            spd = int(os.environ.get("RKK_STEPS_PER_DO", str(steps_per_do)))
        except ValueError:
            spd = steps_per_do
        # Больше шагов PyBullet на один do() — инерция успевает проявиться до снимка для GNN/VL.
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
        # Самомодель: значения держим в среде, observe() мержит с физикой; intervene(self_*) не трогает суставы.
        self._self_state: dict[str, float] = {k: 0.5 for k in SELF_VARS}
        self._motor_state: dict[str, float] = {k: 0.5 for k in MOTOR_INTENT_VARS}

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
        lf = float(raw.get("lfoot_z", 0.05))
        rf = float(raw.get("rfoot_z", 0.05))
        support_l = float(np.clip(1.0 - lf / max(STAND_Z * 0.18, 1e-6), 0.0, 1.0))
        support_r = float(np.clip(1.0 - rf / max(STAND_Z * 0.18, 1e-6), 0.0, 1.0))
        gait_l = float(np.clip(0.5 + 0.5 * np.sin(3.2 * (lhip - 0.5) - 1.7 * (lknee - 0.5)), 0.0, 1.0))
        gait_r = float(np.clip(0.5 + 0.5 * np.sin(3.2 * (rhip - 0.5) - 1.7 * (rknee - 0.5)), 0.0, 1.0))
        support_bias = float(np.clip(0.5 + 0.45 * ((support_l - support_r) + 0.6 * com_x), 0.0, 1.0))
        motor_drive_l = float(np.clip(np.mean([abs(lhip - 0.5), abs(lknee - 0.5), abs(lankle - 0.5)]) * 1.8, 0.0, 1.0))
        motor_drive_r = float(np.clip(np.mean([abs(rhip - 0.5), abs(rknee - 0.5), abs(rankle - 0.5)]) * 1.8, 0.0, 1.0))
        posture_stability = float(np.clip(1.0 - (abs(torso_roll) + abs(torso_pitch)) * 0.35 - abs(com_z - STAND_Z) * 0.45, 0.0, 1.0))
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

        # В fixed_root mode управляем только руками и головой (ноги зафиксированы)
        if self._fixed_root:
            controllable = ARM_VARS + SPINE_VARS + HEAD_VARS
        else:
            controllable = LEG_VARS + ARM_VARS + SPINE_VARS + HEAD_VARS

        if variable in controllable:
            self._sim.set_joint(variable, value)

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
                joints_after.append((variable, v))
        if touched_intent:
            self._apply_motor_intents()
        for variable, v in joints_after:
            self._sim.set_joint(variable, v)
        self._sim.step(self.steps_per_do)
        return self.observe()

    def _apply_upper_body_from_intents(self) -> None:
        """
        Позвоночник + руки из тех же intent_*, что и весь моторный слой (граф / скиллы).
        Вызывается и при полном intervene, и перед CPG на ногах — руки не «застывают» между тиками.
        """
        intents = self._motor_state
        stride = float(intents.get("intent_stride", 0.5) - 0.5)
        sup_l = float(intents.get("intent_support_left", 0.5) - 0.5)
        sup_r = float(intents.get("intent_support_right", 0.5) - 0.5)
        torso = float(intents.get("intent_torso_forward", 0.5) - 0.5)
        arms = float(intents.get("intent_arm_counterbalance", 0.5) - 0.5)
        recover = float(intents.get("intent_stop_recover", 0.5) - 0.5)

        def clip01(v: float) -> float:
            return float(np.clip(v, 0.05, 0.95))

        if self._fixed_root:
            self._sim.set_joint(
                "spine_pitch",
                clip01(0.5 + 0.18 * torso + 0.12 * recover + 0.06 * arms),
            )
            self._sim.set_joint("spine_yaw", clip01(0.5 + 0.06 * (sup_l - sup_r)))
            self._sim.set_joint("lshoulder", clip01(0.5 + 0.16 * arms + 0.09 * recover))
            self._sim.set_joint("rshoulder", clip01(0.5 - 0.16 * arms + 0.09 * recover))
            self._sim.set_joint("lelbow", clip01(0.5 + 0.14 * arms))
            self._sim.set_joint("relbow", clip01(0.5 - 0.14 * arms))
            return

        self._sim.set_joint(
            "spine_pitch",
            clip01(0.5 + 0.14 * torso + 0.16 * recover + 0.07 * arms),
        )
        self._sim.set_joint("spine_yaw", clip01(0.5 + 0.06 * (sup_l - sup_r)))
        self._sim.set_joint(
            "lshoulder",
            clip01(0.5 + 0.17 * arms + 0.03 * stride + 0.05 * recover),
        )
        self._sim.set_joint(
            "rshoulder",
            clip01(0.5 - 0.17 * arms - 0.03 * stride + 0.05 * recover),
        )
        self._sim.set_joint("lelbow", clip01(0.5 + 0.14 * arms))
        self._sim.set_joint("relbow", clip01(0.5 - 0.14 * arms))

    def _apply_motor_intents(self) -> None:
        """Ноги из интентов + верх из _apply_upper_body_from_intents (каузально от intent_*)."""
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

        self._sim.set_joint("lhip", clip01(0.5 + 0.10 * stride - 0.06 * sup_r + 0.03 * torso - 0.04 * recover))
        self._sim.set_joint("rhip", clip01(0.5 - 0.10 * stride - 0.06 * sup_l + 0.03 * torso - 0.04 * recover))
        self._sim.set_joint("lknee", clip01(0.5 + 0.12 * sup_l + 0.10 * recover))
        self._sim.set_joint("rknee", clip01(0.5 + 0.12 * sup_r + 0.10 * recover))
        self._sim.set_joint("lankle", clip01(0.5 + 0.08 * sup_l - 0.02 * stride - 0.04 * recover))
        self._sim.set_joint("rankle", clip01(0.5 + 0.08 * sup_r + 0.02 * stride - 0.04 * recover))
        self._apply_upper_body_from_intents()

    def apply_cpg_leg_targets(self, targets: dict[str, float]) -> None:
        """
        Phase A locomotion: низкоуровневые цели на ноги без увеличения n_interventions.
        """
        if self._fixed_root:
            return
        try:
            n_sub = int(os.environ.get("RKK_CPG_PHYS_SUBSTEPS", "0"))
        except ValueError:
            n_sub = 0
        if n_sub <= 0:
            n_sub = max(1, self.steps_per_do // 2)
        n_sub = min(max(n_sub, 1), 32)
        self._apply_upper_body_from_intents()
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
                {"from_": "intent_arm_counterbalance", "to": "lshoulder", "weight": 0.45},
                {"from_": "intent_arm_counterbalance", "to": "rshoulder", "weight": 0.45},
                {"from_": "intent_torso_forward", "to": "spine_pitch", "weight": 0.45},
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
            {"from_": "intent_support_left", "to": "lknee", "weight": 0.45},
            {"from_": "intent_support_right", "to": "rknee", "weight": 0.45},
            {"from_": "intent_torso_forward", "to": "spine_pitch", "weight": 0.5},
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
            self._motor_state[k] = 0.5

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


# ─── Seeds ────────────────────────────────────────────────────────────────────
def humanoid_hardcoded_seeds() -> list[dict]:
    """Биомеханические text priors для полного режима (суставы → COM, стопы)."""
    return [
        {"from_": "lhip",       "to": "com_x",    "weight": 0.22, "alpha": 0.05},
        {"from_": "rhip",       "to": "com_x",    "weight": 0.22, "alpha": 0.05},
        {"from_": "lknee",      "to": "lfoot_z",  "weight": 0.25, "alpha": 0.05},
        {"from_": "rknee",      "to": "rfoot_z",  "weight": 0.25, "alpha": 0.05},
        {"from_": "lhip",       "to": "com_z",    "weight": 0.18, "alpha": 0.05},
        {"from_": "rhip",       "to": "com_z",    "weight": 0.18, "alpha": 0.05},
        {"from_": "lshoulder",  "to": "cube0_x",  "weight": 0.20, "alpha": 0.05},
        {"from_": "rshoulder",  "to": "cube1_x",  "weight": 0.20, "alpha": 0.05},
        {"from_": "com_z",      "to": "torso_roll","weight":-0.15, "alpha": 0.05},
        {"from_": "self_intention_larm", "to": "lshoulder", "weight": 0.18, "alpha": 0.05},
        {"from_": "self_intention_rarm", "to": "rshoulder", "weight": 0.18, "alpha": 0.05},
        {"from_": "self_attention", "to": "neck_yaw", "weight": 0.12, "alpha": 0.04},
        {"from_": "self_energy", "to": "lshoulder", "weight": 0.10, "alpha": 0.04},
    ]


def fixed_root_seeds() -> list[dict]:
    """
    Text priors для fixed_root mode.
    Фокус: arms → cubes (3 куба × 3 оси), цепочка плечо→локоть→куб,
    позвоночник → cube2.

    Веса слабые (0.18–0.28) — как обычно для seeds.
    """
    return [
        # Левая рука → куб 0
        {"from_": "lshoulder", "to": "cube0_x", "weight": 0.24, "alpha": 0.05},
        {"from_": "lshoulder", "to": "cube0_y", "weight": 0.18, "alpha": 0.05},
        {"from_": "lelbow",    "to": "cube0_x", "weight": 0.20, "alpha": 0.05},
        {"from_": "lelbow",    "to": "cube0_z", "weight": 0.18, "alpha": 0.05},
        # Правая рука → куб 1
        {"from_": "rshoulder", "to": "cube1_x", "weight": 0.24, "alpha": 0.05},
        {"from_": "rshoulder", "to": "cube1_y", "weight": 0.18, "alpha": 0.05},
        {"from_": "relbow",    "to": "cube1_x", "weight": 0.20, "alpha": 0.05},
        {"from_": "relbow",    "to": "cube1_z", "weight": 0.18, "alpha": 0.05},
        # Кинематическая цепочка плечо → локоть
        {"from_": "lshoulder", "to": "lelbow",  "weight": 0.28, "alpha": 0.05},
        {"from_": "rshoulder", "to": "relbow",  "weight": 0.28, "alpha": 0.05},
        # Позвоночник → куб 2 (вращение торса)
        {"from_": "spine_yaw",   "to": "cube2_x", "weight": 0.20, "alpha": 0.05},
        {"from_": "spine_pitch",  "to": "cube2_z", "weight": 0.16, "alpha": 0.05},
        # Шея → куб (при взгляде / наклоне)
        {"from_": "neck_yaw",   "to": "cube0_x", "weight": 0.12, "alpha": 0.04},
        # Самомодель (слабые семена)
        {"from_": "self_intention_larm", "to": "lshoulder", "weight": 0.22, "alpha": 0.05},
        {"from_": "self_intention_larm", "to": "cube0_x", "weight": 0.16, "alpha": 0.04},
        {"from_": "self_intention_rarm", "to": "rshoulder", "weight": 0.22, "alpha": 0.05},
        {"from_": "self_attention", "to": "neck_yaw", "weight": 0.14, "alpha": 0.04},
        {"from_": "self_energy", "to": "lshoulder", "weight": 0.12, "alpha": 0.04},
    ]


def merge_humanoid_golden_with_llm_edges(llm_edges: list[dict]) -> list[dict]:
    golden = humanoid_hardcoded_seeds()
    seen = {(e["from_"], e["to"]) for e in golden}
    out = list(golden)
    for e in llm_edges:
        from_ = e.get("from_") or e.get("from")
        to = e.get("to")
        if not from_ or not to:
            continue
        from_, to = str(from_).strip(), str(to).strip()
        key = (from_, to)
        if key in seen:
            continue
        seen.add(key)
        row = {"from_": from_, "to": to, "weight": float(e.get("weight", 0.25)), "alpha": float(e.get("alpha", 0.05))}
        out.append(row)
    return out