"""
environment_humanoid.py — Humanoid Robot Sandbox (Фаза 11 + fixed_root).

Добавлен fixed_root mode:
  - PyBullet JOINT_FIXED constraint фиксирует base в воздухе
  - variable_ids → FIXED_BASE_VARS (17 переменных вместо 26)
  - is_fallen() всегда False → ValueLayer не блокирует
  - GNN учит только arms→cubes, spine/neck pose
  - EnvironmentHumanoid.set_fixed_root(bool) — переключение в runtime

FIXED_BASE_VARS = ARM + SPINE + HEAD + CUBE_VARS + SANDBOX_VARS (фикс. база + песочница)
"""
from __future__ import annotations

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
SANDBOX_VARS = ["ball_x", "ball_y", "ball_z", "lever_pin", "target_dist"]
VAR_NAMES   = (
    TORSO_VARS + SPINE_VARS + HEAD_VARS + LEG_VARS + ARM_VARS + FOOT_VARS
    + CUBE_VARS + SANDBOX_VARS
)

# Fixed-base mode: таз зафиксирован в воздухе, баланс исключён.
# Агент учит arms/spine/neck → кубы + сигналы песочницы (мяч, рычаг, цель).
FIXED_BASE_VARS: list[str] = ARM_VARS + SPINE_VARS + HEAD_VARS + CUBE_VARS + SANDBOX_VARS

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

FALLEN_Z     = 0.25
STAND_Z      = 0.85

HUMANOID_URDF_GLOBAL_SCALING = 0.36
HUMANOID_URDF_STAND_EULER = (np.pi / 2, 0.0, 0.0)
HUMANOID_URDF_SPAWN_Z = 1.15


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
    j = joints
    pelvis_z = cz - 0.12
    neck_z = cz + 0.17
    spine_z = 0.5 * (neck_z + pelvis_z)
    head_z = cz + 0.29
    sh_z = neck_z - 0.02
    hip_z = pelvis_z - 0.08

    head = [cx, cy, head_z]
    neck = [cx, cy, neck_z]
    spine = [cx, cy, spine_z]
    pelvis = [cx, cy, pelvis_z]
    lshld = [cx - 0.26, cy, sh_z]
    rshld = [cx + 0.26, cy, sh_z]
    lelbow = [
        cx - 0.26 - np.sin(j.get("lshoulder", 0)) * 0.28,
        cy,
        sh_z - np.cos(j.get("lshoulder", 0)) * 0.28,
    ]
    relbow = [
        cx + 0.26 + np.sin(j.get("rshoulder", 0)) * 0.28,
        cy,
        sh_z - np.cos(j.get("rshoulder", 0)) * 0.28,
    ]
    lhand = [
        lelbow[0] - np.sin(j.get("lelbow", 0)) * 0.22,
        cy,
        lelbow[2] - np.cos(j.get("lelbow", 0)) * 0.22,
    ]
    rhand = [
        relbow[0] + np.sin(j.get("relbow", 0)) * 0.22,
        cy,
        relbow[2] - np.cos(j.get("relbow", 0)) * 0.22,
    ]
    hip_half = 0.15
    lhip_p = [cx - hip_half, cy, hip_z]
    rhip_p = [cx + hip_half, cy, hip_z]
    lknee_p = [
        cx - hip_half + np.sin(j.get("lhip", 0)) * 0.35,
        cy,
        hip_z - np.cos(j.get("lhip", 0)) * 0.35,
    ]
    rknee_p = [
        cx + hip_half + np.sin(j.get("rhip", 0)) * 0.35,
        cy,
        hip_z - np.cos(j.get("rhip", 0)) * 0.35,
    ]
    lfoot = [
        lknee_p[0] + np.sin(j.get("lknee", 0)) * 0.30,
        cy,
        lknee_p[2] - np.cos(j.get("lknee", 0)) * 0.30,
    ]
    rfoot = [
        rknee_p[0] + np.sin(j.get("rknee", 0)) * 0.30,
        cy,
        rknee_p[2] - np.cos(j.get("rknee", 0)) * 0.30,
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


# ─── Fallback гуманоид ────────────────────────────────────────────────────────
class _FallbackHumanoid:
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

    def step(self, n: int = 10):
        if self.fixed_root:
            # В fixed_root mode нет гравитации на базу
            self.com[:2] += np.random.normal(0, 0.0005, 2)
            return
        for _ in range(n):
            balance = abs(self.torso_euler[0]) + abs(self.torso_euler[1])
            self.com[2] += (-0.02 * balance + 0.001 * (np.random.rand()-0.5)) * self._dt
            self.com[2]  = np.clip(self.com[2], 0.0, 1.5)
            self.com[:2] += np.random.normal(0, 0.002, 2)
            self.torso_euler[:2] += np.random.normal(0, 0.003, 2)
            self.torso_euler[:2] = np.clip(self.torso_euler[:2], -1.2, 1.2)

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
        return (max(0, 0.1 - k_l * 0.05), max(0, 0.1 - k_r * 0.05))

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

    # fixed_root controls для fallback — только флаг
    def enable_fixed_root(self) -> None:
        self.fixed_root = True

    def disable_fixed_root(self) -> None:
        self.fixed_root = False


# ─── PyBullet гуманоид ────────────────────────────────────────────────────────
class _PyBulletHumanoid:

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

    # ── Fixed root constraint ─────────────────────────────────────────────────
    def enable_fixed_root(self) -> None:
        """
        Фиксируем базу робота в мировых координатах через JOINT_FIXED constraint.
        Вызывается ПОСЛЕ reset_stance() чтобы зафиксировать стабильную позу.
        """
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
        for _ in range(n):
            pb.stepSimulation(physicsClientId=self.client)

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
        had_fixed = self._root_constraint is not None
        if had_fixed:
            self.disable_fixed_root()

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

        # Восстанавливаем fixed_root если был
        if had_fixed:
            self.enable_fixed_root()

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

    def set_joint(self, var_name: str, target_pos: float):
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
        head_v = neck_v + 0.26 * up

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
        urdf_pts = self._skeleton_from_urdf_links()
        if urdf_pts is not None:
            return urdf_pts
        pos, _ = pb.getBasePositionAndOrientation(self.robot_id, physicsClientId=self.client)
        cx, cy, cz = float(pos[0]), float(pos[1]), float(pos[2])
        j = {v: self.get_joint_angle(v) for v in LEG_VARS + ARM_VARS + SPINE_VARS + HEAD_VARS}
        return _forward_kinematics_skeleton(cx, cy, cz, j)

    def get_cube_positions(self) -> list[dict]:
        result = []
        for cid in self.cube_ids:
            pos, _ = pb.getBasePositionAndOrientation(cid, physicsClientId=self.client)
            result.append({"x": float(pos[0]), "y": float(pos[1]), "z": float(pos[2])})
        return result

    def get_sandbox_scene_extras(self) -> dict:
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

        head_anchor = pos_n + 0.26 * up
        eye = head_anchor + 0.04 * up + 0.06 * fwd
        target = eye + 2.4 * fwd
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
            pb.disconnect(self.client)
        except Exception:
            pass


# ─── EnvironmentHumanoid ─────────────────────────────────────────────────────
class EnvironmentHumanoid:
    """
    Среда гуманоида для Singleton AGI.

    fixed_root=True: база зафиксирована, variable_ids = FIXED_BASE_VARS (17 vars).
    set_fixed_root(bool): переключение в runtime.
    """

    PRESET = "humanoid"

    def __init__(
        self,
        device: torch.device | None = None,
        steps_per_do: int = 12,
        fixed_root: bool = False,
    ):
        self.device       = device or torch.device("cpu")
        self.steps_per_do = steps_per_do
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
        return {k: self._norm(k, v) for k, v in raw.items() if k in active}

    @property
    def variables(self) -> dict[str, float]:
        return self.observe()

    @property
    def variable_ids(self) -> list[str]:
        return list(FIXED_BASE_VARS if self._fixed_root else VAR_NAMES)

    # ── do() ─────────────────────────────────────────────────────────────────
    def intervene(self, variable: str, value: float) -> dict[str, float]:
        self.n_interventions += 1

        # В fixed_root mode управляем только руками и головой (ноги зафиксированы)
        if self._fixed_root:
            controllable = ARM_VARS + SPINE_VARS + HEAD_VARS
        else:
            controllable = LEG_VARS + ARM_VARS + SPINE_VARS + HEAD_VARS

        if variable in controllable:
            self._sim.set_joint(variable, value)

        self._sim.step(self.steps_per_do)
        return self.observe()

    # ── Discovery rate ────────────────────────────────────────────────────────
    def discovery_rate(self, agent_edges: list[dict]) -> float:
        gt = self.gt_edges()
        gt_set = {(e["from_"], e["to"]) for e in gt}
        hits = sum(1 for e in agent_edges if (e.get("from_"), e.get("to")) in gt_set)
        return hits / len(gt_set) if gt_set else 0.0

    def gt_edges(self) -> list[dict]:
        if self._fixed_root:
            # В fixed_root: только arm→cube каузальность
            return [
                {"from_": "lshoulder", "to": "cube0_x", "weight": 0.6},
                {"from_": "lshoulder", "to": "cube0_y", "weight": 0.4},
                {"from_": "rshoulder", "to": "cube1_x", "weight": 0.6},
                {"from_": "rshoulder", "to": "cube1_y", "weight": 0.4},
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