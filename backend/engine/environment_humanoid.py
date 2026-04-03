"""
environment_humanoid.py — Humanoid Robot Sandbox (Фаза 11).

Среда: Полноценный гуманоид (humanoid.urdf) в комнате с интерактивными объектами.
Задача: AGI открывает законы биомеханики через интервенции.

Физический мир:
  - Гуманоид (humanoid.urdf / atlas.urdf из pybullet_data)
  - Пол (plane.urdf)
  - 3 куба разного размера и массы (интерактивные)
  - Рампа (наклонная плоскость, нагружает гравитацию)

Редуцированные переменные (~22):
  Торс:  com_x, com_y, com_z, roll, pitch                    (5)
  Ноги:  lhip, lknee, lankle, rhip, rknee, rankle            (6)
  Руки:  lshoulder, lelbow, rshoulder, relbow                 (4)
  Стопы: lfoot_z, rfoot_z                                     (2)
  Кубы:  cube0_x,y,z  cube1_x,y,z  cube2_x,y,z              (9)
  Итого: 26 переменных

GT каузальная структура (скрытая):
  lhip, rknee  → com_x/y      (локомоция)
  lknee, rknee → lfoot_z/rfoot_z (положение стоп)
  gravity_implicit → com_z    (баланс)
  lshoulder/rshoulder → cube_x/y/z (взаимодействие)

Value Layer интеграция:
  com_z < FALLEN_THRESHOLD → BlockReason.FALLEN + штраф

do() оператор:
  do(lhip=0.6)  → устанавливаем позицию (PD контроль)
  do(lknee=0.4) → сгибаем колено

Камера:
  diag / side / front / top + ego (от шеи, вперёд по горизонтали)
  get_frame_base64(view=...) → JPEG base64
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
LEG_VARS    = ["lhip", "lknee", "lankle", "rhip", "rknee", "rankle"]
ARM_VARS    = ["lshoulder", "lelbow", "rshoulder", "relbow"]
FOOT_VARS   = ["lfoot_z", "rfoot_z"]
CUBE_VARS   = [f"cube{i}_{d}" for i in range(3) for d in ["x","y","z"]]
VAR_NAMES   = TORSO_VARS + LEG_VARS + ARM_VARS + FOOT_VARS + CUBE_VARS

# Нормализация диапазонов
_RANGES = {}
for v in TORSO_VARS[:2]:  _RANGES[v] = (-1.5, 1.5)   # COM xy
_RANGES["com_z"]          = (0.0,  1.5)                # высота
_RANGES["torso_roll"]     = (-1.2, 1.2)
_RANGES["torso_pitch"]    = (-1.2, 1.2)
for v in LEG_VARS:         _RANGES[v] = (-1.5, 1.5)   # angles rad
for v in ARM_VARS:         _RANGES[v] = (-2.0, 2.0)
for v in FOOT_VARS:        _RANGES[v] = (-0.1, 0.5)
for v in CUBE_VARS:
    if v.endswith("_z"):   _RANGES[v] = (-0.1, 1.0)
    else:                  _RANGES[v] = (-2.0, 2.0)

FALLEN_Z     = 0.25   # ниже этой высоты = упал
STAND_Z      = 0.85   # нормальная высота стоя

# humanoid.urdf из PyBullet: звенья вытянуты ~5m по цепочке — для UI/fallback (~1.8m) уменьшаем модель.
HUMANOID_URDF_GLOBAL_SCALING = 0.36

# URDF: туловище вдоль локальной Y; мир PyBullet Z-up. Без поворота робот «лежит» вдоль пола.
# +90° вокруг X: ось роста → Z. Высота базы подобрана под plane + globalScaling.
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
    """Hamilton product, порядок как в PyBullet [x,y,z,w]."""
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return [
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    ]


def _forward_kinematics_skeleton(
    cx: float, cy: float, cz: float, joints: dict[str, float]
) -> list[dict]:
    """
    15 точек в том же порядке, что rkk-humanoid.jsx:
    0 голова, 1 шея (хаб рук), 2 таз/root (хаб ног), 3–8 руки, 9–14 ноги.
    Отдельного узла «грудь» нет — плечи у шеи. Оси: x,y горизонталь, z вверх.
    """
    j = joints
    pelvis_z = cz - 0.12
    neck_z = cz + 0.17
    head_z = cz + 0.29
    sh_z = neck_z - 0.02  # плечи у шейного узла
    hip_z = pelvis_z - 0.08

    head = [cx, cy, head_z]
    neck = [cx, cy, neck_z]
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
    lhip_p = [cx - 0.12, cy, hip_z]
    rhip_p = [cx + 0.12, cy, hip_z]
    lknee_p = [
        cx - 0.12 + np.sin(j.get("lhip", 0)) * 0.35,
        cy,
        hip_z - np.cos(j.get("lhip", 0)) * 0.35,
    ]
    rknee_p = [
        cx + 0.12 + np.sin(j.get("rhip", 0)) * 0.35,
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
    pts = [
        head, neck, pelvis,
        lshld, rshld, lelbow, relbow, lhand, rhand,
        lhip_p, rhip_p, lknee_p, rknee_p, lfoot, rfoot,
    ]
    return [{"x": float(p[0]), "y": float(p[1]), "z": float(p[2])} for p in pts]


# ─── Fallback гуманоид (аналитический) ───────────────────────────────────────
class _FallbackHumanoid:
    """Простая кинематическая модель без физики."""

    def __init__(self):
        self.joints = {v: 0.0 for v in LEG_VARS + ARM_VARS}
        self.com    = np.array([0.0, 0.0, STAND_Z])
        self.torso_euler = np.zeros(3)
        self.cubes  = np.array([
            [ 0.8,  0.3, 0.15],
            [-0.5,  0.6, 0.15],
            [ 0.2, -0.7, 0.25],
        ])
        self._vel   = np.zeros(3)
        self._dt    = 0.02

    def step(self, n: int = 10):
        for _ in range(n):
            # Гравитация и лёгкая нестабильность
            balance = abs(self.torso_euler[0]) + abs(self.torso_euler[1])
            self.com[2] += (-0.02 * balance + 0.001 * (np.random.rand()-0.5)) * self._dt
            self.com[2]  = np.clip(self.com[2], 0.0, 1.5)
            self.com[:2] += np.random.normal(0, 0.002, 2)
            self.torso_euler[:2] += np.random.normal(0, 0.003, 2)
            self.torso_euler[:2] = np.clip(self.torso_euler[:2], -1.2, 1.2)

    def set_joint(self, name: str, val: float):
        if name in self.joints:
            self.joints[name] = np.clip(float(val), -2.0, 2.0)
            # Кинематическое влияние на COM
            if "hip" in name:
                idx = 0 if name.startswith("l") else 1
                self.com[idx] += val * 0.02
            if "knee" in name:
                self.com[2] = max(0.1, self.com[2] + val * 0.01)

    def get_foot_z(self) -> tuple[float, float]:
        k_l = self.joints.get("lknee", 0)
        k_r = self.joints.get("rknee", 0)
        return (max(0, 0.1 - k_l * 0.05), max(0, 0.1 - k_r * 0.05))

    def get_all_link_positions(self) -> list[dict]:
        """Скелетон для Three.js: 15 точек, порядок как на фронте (без узла грудь)."""
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
        for v in LEG_VARS + ARM_VARS:
            s[v] = float(self.joints.get(v, 0.0))
        s["lfoot_z"]  = float(lf_z)
        s["rfoot_z"]  = float(rf_z)
        for i, cube in enumerate(self.cubes):
            s[f"cube{i}_x"] = float(cube[0])
            s[f"cube{i}_y"] = float(cube[1])
            s[f"cube{i}_z"] = float(cube[2])
        return s

    def get_cube_positions(self) -> list[dict]:
        return [{"x":float(c[0]),"y":float(c[1]),"z":float(c[2])} for c in self.cubes]

    def get_frame_base64(self, view="side") -> str | None:
        return None

    def reset_stance(self) -> None:
        self.joints = {v: 0.0 for v in LEG_VARS + ARM_VARS}
        self.com = np.array([0.0, 0.0, STAND_Z], dtype=np.float64)
        self.torso_euler = np.zeros(3)
        self._vel = np.zeros(3)


# ─── PyBullet гуманоид ────────────────────────────────────────────────────────
class _PyBulletHumanoid:

    # Joint name → наш variable name (частичное совпадение)
    _JOINT_MAP = {
        "leftHip":     "lhip",
        "left_hip":    "lhip",
        "LeftHip":     "lhip",
        "leftKnee":    "lknee",
        "left_knee":   "lknee",
        "LeftKnee":    "lknee",
        "leftAnkle":   "lankle",
        "left_ankle":  "lankle",
        "rightHip":    "rhip",
        "right_hip":   "rhip",
        "RightHip":    "rhip",
        "rightKnee":   "rknee",
        "right_knee":  "rknee",
        "RightKnee":   "rknee",
        "rightAnkle":  "rankle",
        "right_ankle": "rankle",
        "leftShoulder":"lshoulder",
        "left_shoulder":"lshoulder",
        "rightShoulder":"rshoulder",
        "right_shoulder":"rshoulder",
        "leftElbow":   "lelbow",
        "left_elbow":  "lelbow",
        "rightElbow":  "relbow",
        "right_elbow": "relbow",
    }

    def __init__(self):
        self.client = pb.connect(pb.DIRECT)
        pb.setGravity(0, 0, -9.81, physicsClientId=self.client)
        pb.setAdditionalSearchPath(pbd.getDataPath(), physicsClientId=self.client)
        pb.setTimeStep(1/240., physicsClientId=self.client)
        pb.setPhysicsEngineParameter(numSolverIterations=50, physicsClientId=self.client)

        # Пол
        self.floor_id = pb.loadURDF("plane.urdf", physicsClientId=self.client)

        # Рампа (наклонная плоскость)
        self._build_ramp()

        # Кубы разного размера и массы
        self.cube_ids = []
        cube_configs = [
            {"pos": [1.0,  0.3, 0.15], "size": 0.12, "mass": 2.0,  "color": [1.0, 0.4, 0.1, 1]},
            {"pos": [-0.6, 0.8, 0.12], "size": 0.10, "mass": 0.5,  "color": [0.2, 0.7, 1.0, 1]},
            {"pos": [0.3, -0.9, 0.20], "size": 0.16, "mass": 5.0,  "color": [0.3, 1.0, 0.4, 1]},
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

        # Гуманоид
        self.robot_id = self._load_humanoid()
        self.n_joints = pb.getNumJoints(self.robot_id, physicsClientId=self.client)

        # Строим маппинг joint_index → variable_name
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

        # Имена звеньев + типы суставов (для reset_stance: spherical → MultiDof)
        self.link_names: list[str] = []
        self._joint_types: list[int] = []
        for i in range(self.n_joints):
            info = pb.getJointInfo(self.robot_id, i, physicsClientId=self.client)
            self.link_names.append(info[12].decode("utf-8"))
            self._joint_types.append(int(info[2]))

        # Без «прогрева» на VELOCITY для всех суставов: для spherical это ломает осанку.
        # Сразу нейтральная стойка (как при резете после падения).
        self.reset_stance()

    def _build_ramp(self):
        """Наклонная плоскость 2×1 под 15°."""
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

    def _load_humanoid(self) -> int:
        # Локальная копия humanoid.urdf: у сферических суставов добавлен <axis> — иначе urdfdom
        # пишет b3Warning «no axis element… defaulting to (1,0,0)».
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

        # Крайний fallback: строим вручную из multi-body
        return self._build_custom_humanoid()

    def _build_custom_humanoid(self) -> int:
        """Строим гуманоида из 15 сочленений если URDF недоступен."""
        masses     = [8.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 3.0, 3.0, 1.5, 1.5, 0.5, 0.5]
        col_shapes, vis_shapes, positions, orientations = [], [], [], []
        inertial_pos, inertial_orn, parents, jtypes, jaxes = [], [], [], [], []

        segments = [
            # (halfExtents, color,           pos_rel_parent,  parent)
            ([0.1,0.15,0.12], [0.7,0.7,0.7,1], [0,0,0],      0),   # 0 torso
            ([0.05,0.05,0.05],[0.9,0.8,0.7,1], [0,0,0.22],   0),   # 1 head
            ([0.04,0.04,0.14],[0.7,0.7,0.7,1], [-0.18,0,0.1],0),   # 2 lup-arm
            ([0.04,0.04,0.14],[0.7,0.7,0.7,1], [0.18,0,0.1], 0),   # 3 rup-arm
            ([0.03,0.03,0.12],[0.8,0.8,0.8,1], [-0.18,0,-0.14],2), # 4 lforearm
            ([0.03,0.03,0.12],[0.8,0.8,0.8,1], [0.18,0,-0.14],3),  # 5 rforearm
            ([0.06,0.06,0.15],[0.7,0.7,0.7,1], [-0.08,0,-0.15],0), # 6 lthigh
            ([0.06,0.06,0.15],[0.7,0.7,0.7,1], [0.08,0,-0.15], 0), # 7 rthigh
            ([0.05,0.05,0.14],[0.8,0.8,0.8,1], [0,0,-0.15],   6),  # 8 lshin
            ([0.05,0.05,0.14],[0.8,0.8,0.8,1], [0,0,-0.15],   7),  # 9 rshin
            ([0.07,0.03,0.03],[0.6,0.6,0.6,1], [0,0,-0.14],   8),  # 10 lfoot
            ([0.07,0.03,0.03],[0.6,0.6,0.6,1], [0,0,-0.14],   9),  # 11 rfoot
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
        # Маппинг вручную для кастомного тела
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
        """
        Режим удержания между шагами: те же цели, что в _motor_stabilize_neutral_pose.
        Слабый VELOCITY на spherical через setJointMotorControl2 давал завал назад (~30–40°);
        оставляем MultiDof POSITION к нейтрали с теми же gain/force, что после reset.
        """
        rid, cid = self.robot_id, self.client
        quat_id = [0.0, 0.0, 0.0, 1.0]
        motor_m = getattr(pb, "setJointMotorControlMultiDof", None)
        for i in range(self.n_joints):
            jt = self._joint_types[i]
            if jt == pb.JOINT_FIXED:
                continue
            if jt == pb.JOINT_SPHERICAL and callable(motor_m):
                motor_m(
                    rid, i,
                    pb.POSITION_CONTROL,
                    targetPosition=quat_id,
                    positionGain=1.0,
                    velocityGain=0.35,
                    maxVelocity=8.0,
                    force=[220.0, 220.0, 220.0],
                    physicsClientId=cid,
                )
            else:
                pb.setJointMotorControl2(
                    rid, i,
                    controlMode=pb.POSITION_CONTROL,
                    targetPosition=0.0,
                    positionGain=0.55,
                    velocityGain=0.12,
                    force=100.0,
                    physicsClientId=cid,
                )

    def _motor_stabilize_neutral_pose(self) -> None:
        """
        После kinematic reset сферы под гравитацией снова «схлопываются», если
        оставить force=0.1. Кратко держим нейтраль: spherical → identity quat,
        revolute → 0 с нормальным PD.
        """
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
                    positionGain=1.0,
                    velocityGain=0.35,
                    maxVelocity=8.0,
                    force=[220.0, 220.0, 220.0],
                    physicsClientId=cid,
                )
            else:
                pb.setJointMotorControl2(
                    rid, i,
                    controlMode=pb.POSITION_CONTROL,
                    targetPosition=0.0,
                    positionGain=0.55, velocityGain=0.12,
                    force=100.0,
                    physicsClientId=cid,
                )

    def _snap_base_spine_vertical(self) -> None:
        """
        Нейтральные суставы + гравитация дают ~25–35° наклон таз→шея от мирового Z.
        Поворачиваем базу вокруг мира так, чтобы вектор root→neck совпал с +Z,
        затем короткий settle (стопы остаются над полом при типичном спавне).
        """
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
        Вернуть робота в спавн-позу. Сферические суставы — MultiDof + краткий
        POSITION_CONTROL, иначе гравитация снова ломает осанку при слабом VELOCITY.
        """
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
                dof_reset(
                    rid, i, quat_identity, omega_zero, physicsClientId=cid,
                )
            else:
                pb.resetJointState(
                    rid, i,
                    targetValue=0.0, targetVelocity=0.0,
                    physicsClientId=cid,
                )

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

    def set_joint(self, var_name: str, target_pos: float):
        """PD control на сустав."""
        if var_name not in self.joint_by_var:
            return
        jid = self.joint_by_var[var_name]
        lo, hi = _RANGES.get(var_name, (-2.0, 2.0))
        real_pos = target_pos * (hi - lo) + lo
        pb.setJointMotorControl2(
            self.robot_id, jid,
            controlMode=pb.POSITION_CONTROL,
            targetPosition=float(np.clip(real_pos, lo, hi)),
            positionGain=0.5, velocityGain=0.1,
            force=80.0,
            physicsClientId=self.client
        )

    def get_com(self) -> tuple[np.ndarray, np.ndarray]:
        """Центр масс тела + ориентация торса."""
        pos, orn = pb.getBasePositionAndOrientation(self.robot_id, physicsClientId=self.client)
        euler    = pb.getEulerFromQuaternion(orn)
        return np.array(pos), np.array(euler)

    def get_joint_angle(self, var_name: str) -> float:
        if var_name not in self.joint_by_var:
            return 0.0
        jid = self.joint_by_var[var_name]
        st  = pb.getJointState(self.robot_id, jid, physicsClientId=self.client)
        return float(st[0])

    def get_foot_heights(self) -> tuple[float, float]:
        """Высота стоп через link state."""
        lz = rz = 0.05
        n_links = pb.getNumJoints(self.robot_id, physicsClientId=self.client)
        # Ищем самые нижние links по Z
        zs = []
        for i in range(n_links):
            try:
                st = pb.getLinkState(self.robot_id, i, physicsClientId=self.client)
                zs.append((i, st[4][2]))   # world position z
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

    def get_state(self) -> dict:
        com, euler = self.get_com()
        lf, rf = self.get_foot_heights()
        s = {}
        s["com_x"]       = float(com[0])
        s["com_y"]       = float(com[1])
        s["com_z"]       = float(com[2])
        s["torso_roll"]  = float(euler[0])
        s["torso_pitch"] = float(euler[1])
        for v in LEG_VARS + ARM_VARS:
            s[v] = self.get_joint_angle(v)
        s["lfoot_z"] = lf
        s["rfoot_z"] = rf
        for i, cp in enumerate(self.get_cube_state()):
            s[f"cube{i}_x"] = float(cp[0])
            s[f"cube{i}_y"] = float(cp[1])
            s[f"cube{i}_z"] = float(cp[2])
        return s

    def _named_link_world_positions(self) -> dict[str, np.ndarray]:
        """Мировые позиции звеньев (child link каждого joint)."""
        out: dict[str, np.ndarray] = {}
        for i in range(self.n_joints):
            name = self.link_names[i]
            st = pb.getLinkState(
                self.robot_id, i,
                computeForwardKinematics=1,
                physicsClientId=self.client,
            )
            out[name] = np.array(st[4][:3], dtype=float)
        return out

    def _skeleton_from_urdf_links(self) -> list[dict] | None:
        """
        Позиции звеньев из PyBullet (мир Z вверх). База URDF повёрнута так,
        что цепочка идёт вдоль Z — см. HUMANOID_URDF_STAND_EULER при loadURDF.
        """
        need = {
            "neck", "chest", "root",
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

        # chest только для направления головы; в скелет не попадает (руки от шеи)
        order = [
            head_v,
            neck_v,
            vec("root"),
            vec("left_shoulder"),
            vec("right_shoulder"),
            vec("left_elbow"),
            vec("right_elbow"),
            vec("left_wrist"),
            vec("right_wrist"),
            vec("left_hip"),
            vec("right_hip"),
            vec("left_knee"),
            vec("right_knee"),
            vec("left_ankle"),
            vec("right_ankle"),
        ]
        return [{"x": float(v[0]), "y": float(v[1]), "z": float(v[2])} for v in order]

    def get_all_link_positions(self) -> list[dict]:
        """
        15 точек (как rkk-humanoid.jsx): голова, шея, таз; руки от шеи, ноги от таза.
        Для URDF humanoid — getLinkState; иначе FK от базы (кастомный multi-body).
        """
        urdf_pts = self._skeleton_from_urdf_links()
        if urdf_pts is not None:
            return urdf_pts
        pos, _ = pb.getBasePositionAndOrientation(self.robot_id, physicsClientId=self.client)
        cx, cy, cz = float(pos[0]), float(pos[1]), float(pos[2])
        j = {v: self.get_joint_angle(v) for v in LEG_VARS + ARM_VARS}
        return _forward_kinematics_skeleton(cx, cy, cz, j)

    def get_cube_positions(self) -> list[dict]:
        result = []
        for cid in self.cube_ids:
            pos, _ = pb.getBasePositionAndOrientation(cid, physicsClientId=self.client)
            result.append({"x": float(pos[0]), "y": float(pos[1]), "z": float(pos[2])})
        return result

    _CAMERA_CONFIGS = {
        "side":  {"eye": [0, -3.5, 1.2], "target": [0, 0, 0.7], "up": [0, 0, 1]},
        "front": {"eye": [3.5, 0,  1.2], "target": [0, 0, 0.7], "up": [0, 0, 1]},
        "top":   {"eye": [0,  0,   4.0], "target": [0, 0, 0.5], "up": [0, 1, 0]},
        "diag":  {"eye": [2.5, -2.5, 2.0], "target": [0, 0, 0.7], "up": [0, 0, 1]},
    }

    def _ego_camera_rt(self) -> tuple[list[float], list[float], list[float]] | None:
        """Глаза у шеи: вперёд по горизонтали, up вдоль груди→шея."""
        names = set(self.link_names)
        if "neck" not in names or "chest" not in names:
            return None
        i_neck = self.link_names.index("neck")
        i_chest = self.link_names.index("chest")
        st_n = pb.getLinkState(
            self.robot_id, i_neck, computeForwardKinematics=1, physicsClientId=self.client
        )
        st_c = pb.getLinkState(
            self.robot_id, i_chest, computeForwardKinematics=1, physicsClientId=self.client
        )
        pos_n = np.array(st_n[0][:3], dtype=float)
        pos_c = np.array(st_c[0][:3], dtype=float)
        up = pos_n - pos_c
        ln = float(np.linalg.norm(up))
        if ln < 1e-6:
            up = np.array([0.0, 0.0, 1.0], dtype=float)
        else:
            up = up / ln
        # Горизонтальный «взгляд»: направление в плоскости, перпендикулярной up
        for pref in (
            np.array([0.0, 1.0, 0.0], dtype=float),
            np.array([1.0, 0.0, 0.0], dtype=float),
        ):
            fwd = pref - float(np.dot(pref, up)) * up
            fn = float(np.linalg.norm(fwd))
            if fn >= 1e-5:
                fwd = fwd / fn
                break
        else:
            return None
        # Слегка впереди и выше центра шеи
        eye = pos_n + 0.10 * up + 0.05 * fwd
        target = eye + 2.2 * fwd
        return eye.tolist(), target.tolist(), up.tolist()

    def get_frame_base64(self, view: str = "diag",
                         width: int = 480, height: int = 360) -> str | None:
        if not PIL_AVAILABLE:
            return None
        try:
            vkey = (view or "diag").lower()
            if vkey in ("ego", "first_person", "fp"):
                eg = self._ego_camera_rt()
                if eg is not None:
                    eye, tgt, cup = eg
                    vm = pb.computeViewMatrix(eye, tgt, cup, physicsClientId=self.client)
                else:
                    vkey = "diag"
                    eg = None
            else:
                eg = None

            if eg is None:
                cfg = self._CAMERA_CONFIGS.get(vkey, self._CAMERA_CONFIGS["diag"])
                vm = pb.computeViewMatrix(cfg["eye"], cfg["target"], cfg["up"],
                                         physicsClientId=self.client)
            pm   = pb.computeProjectionMatrixFOV(
                fov=60, aspect=width/height, nearVal=0.1, farVal=15.0,
                physicsClientId=self.client)
            _, _, rgba, _, _ = pb.getCameraImage(
                width, height, vm, pm,
                renderer=pb.ER_TINY_RENDERER,
                physicsClientId=self.client,
            )
            # PyBullet может вернуть tuple/list/1D-массив — не индексировать как (H,W,4)
            pix = np.asarray(rgba, dtype=np.uint8).reshape(-1)
            need = width * height * 4
            if pix.size < need:
                raise ValueError(f"camera pixels {pix.size} < expected {need}")
            rgb = pix[:need].reshape((height, width, 4))[:, :, :3]
            img = PILImage.fromarray(rgb)
            buf = BytesIO()
            img.save(buf, format="JPEG", quality=85)
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

    Совместима с Environment / EnvironmentRobot:
      observe()         → dict[str, float] (нормализованные)
      intervene()       → dict[str, float]
      discovery_rate()  → float
      gt_edges()        → list[dict]
      get_frame_base64() → str | None
    """

    PRESET = "humanoid"

    def __init__(self, device: torch.device | None = None, steps_per_do: int = 12):
        self.device       = device or torch.device("cpu")
        self.steps_per_do = steps_per_do
        self.preset       = self.PRESET
        self.n_interventions = 0

        if PYBULLET_AVAILABLE:
            self._sim     = _PyBulletHumanoid()
            self._backend = "pybullet"
        else:
            self._sim     = _FallbackHumanoid()
            self._backend = "fallback"

        print(f"[HumanoidEnv] backend={self._backend}, vars={len(VAR_NAMES)}")

    # ── Нормализация ───────────────────────────────────────────────────────────
    def _norm(self, key: str, val: float) -> float:
        lo, hi = _RANGES.get(key, (-1.0, 1.0))
        return float(np.clip((val - lo) / (hi - lo), 0.05, 0.95))

    def _denorm(self, key: str, val: float) -> float:
        lo, hi = _RANGES.get(key, (-1.0, 1.0))
        return float(val * (hi - lo) + lo)

    # ── Observe ────────────────────────────────────────────────────────────────
    def observe(self) -> dict[str, float]:
        raw = self._sim.get_state()
        return {k: self._norm(k, v) for k, v in raw.items() if k in VAR_NAMES}

    @property
    def variables(self) -> dict[str, float]:
        return self.observe()

    @property
    def variable_ids(self) -> list[str]:
        return list(VAR_NAMES)

    # ── do() ──────────────────────────────────────────────────────────────────
    def intervene(self, variable: str, value: float) -> dict[str, float]:
        self.n_interventions += 1
        if variable in LEG_VARS + ARM_VARS:
            self._sim.set_joint(variable, value)
        self._sim.step(self.steps_per_do)
        return self.observe()

    # ── Discovery rate ─────────────────────────────────────────────────────────
    def discovery_rate(self, agent_edges: list[dict]) -> float:
        gt = self.gt_edges()
        gt_set = {(e["from_"], e["to"]) for e in gt}
        hits = sum(1 for e in agent_edges if (e.get("from_"), e.get("to")) in gt_set)
        return hits / len(gt_set) if gt_set else 0.0

    def gt_edges(self) -> list[dict]:
        edges = []
        # Интеграция: скорость сустава → позиция
        for v in LEG_VARS:
            edges.append({"from_": v, "to": "com_z",  "weight": 0.5})
        edges.append({"from_": "lhip",  "to": "com_x",  "weight": 0.7})
        edges.append({"from_": "rhip",  "to": "com_x",  "weight": 0.7})
        edges.append({"from_": "lknee", "to": "lfoot_z","weight": 0.8})
        edges.append({"from_": "rknee", "to": "rfoot_z","weight": 0.8})
        # Кинематика
        edges.append({"from_": "lshoulder", "to": "cube0_x", "weight": 0.6})
        edges.append({"from_": "rshoulder", "to": "cube1_x", "weight": 0.6})
        edges.append({"from_": "com_z",     "to": "torso_roll", "weight": -0.4})
        return edges

    # ── Упал ли? ───────────────────────────────────────────────────────────────
    def is_fallen(self) -> bool:
        obs = self.observe()
        return obs.get("com_z", 0.5) < self._norm("com_z", FALLEN_Z)

    def reset_stance(self) -> None:
        self._sim.reset_stance()

    # ── Camera / Skeleton ──────────────────────────────────────────────────────
    def get_frame_base64(self, view: str = "diag") -> str | None:
        return self._sim.get_frame_base64(view)

    def get_joint_positions_world(self) -> list[dict]:
        return self._sim.get_all_link_positions()

    def get_cube_positions(self) -> list[dict]:
        return self._sim.get_cube_positions()

    def get_target(self) -> dict:
        # Для гуманоида "цель" — это стоять прямо (целевая позиция COM)
        return {"x": 0.0, "y": 0.0, "z": STAND_Z}

    def get_full_scene(self) -> dict:
        """Всё для Three.js за один вызов."""
        return {
            "skeleton": self.get_joint_positions_world(),
            "cubes":    self.get_cube_positions(),
            "target":   self.get_target(),
            "fallen":   self.is_fallen(),
            "com_z":    self.observe().get("com_z", 0.5),
        }


# ─── Seeds ────────────────────────────────────────────────────────────────────
def humanoid_hardcoded_seeds() -> list[dict]:
    """Биомеханические text priors: суставы → COM, стопы."""
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