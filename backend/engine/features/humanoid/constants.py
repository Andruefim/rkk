"""Имена переменных, диапазоны нормализации, константы URDF для humanoid."""
from __future__ import annotations

from pathlib import Path

import numpy as np

# ─── Переменные ───────────────────────────────────────────────────────────────
TORSO_VARS = ["com_x", "com_y", "com_z", "torso_roll", "torso_pitch"]
SPINE_VARS = ["spine_yaw", "spine_pitch"]
HEAD_VARS = ["neck_yaw", "neck_pitch"]
# Один сенсорный канал (3 скаляра): единичный вектор g в системе координат головы (вестибулярный ориентир).
VESTIBULAR_VARS: tuple[str, ...] = ("vestibular_gx", "vestibular_gy", "vestibular_gz")
LEG_VARS = ["lhip", "lknee", "lankle", "rhip", "rknee", "rankle"]
ARM_VARS = ["lshoulder", "lelbow", "lwrist", "rshoulder", "relbow", "rwrist"]
FOOT_VARS = ["lfoot_z", "rfoot_z"]
CUBE_VARS = [f"cube{i}_{d}" for i in range(3) for d in ["x", "y", "z"]]
SANDBOX_VARS = [
    "ball_x",
    "ball_y",
    "ball_z",
    "lever_pin",
    "target_dist",
    "floor_friction",
    "stack_height",
    "stability_score",
]
MOTOR_INTENT_VARS: tuple[str, ...] = (
    "intent_stride",
    "intent_support_left",
    "intent_support_right",
    "intent_torso_forward",
    "intent_gait_coupling",
    "intent_arm_counterbalance",
    "intent_stop_recover",
)
MOTOR_INTENT_DEFAULTS: dict[str, float] = {
    "intent_gait_coupling": 0.88,
}
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
SELF_VARS: tuple[str, ...] = (
    "self_intention_larm",
    "self_intention_rarm",
    "self_energy",
    "self_attention",
    "self_goal_target_dist",
    "self_goal_active",
)
VAR_NAMES = (
    TORSO_VARS
    + SPINE_VARS
    + HEAD_VARS
    + list(VESTIBULAR_VARS)
    + LEG_VARS
    + ARM_VARS
    + FOOT_VARS
    + CUBE_VARS
    + SANDBOX_VARS
    + list(MOTOR_INTENT_VARS)
    + list(MOTOR_OBSERVABLE_VARS)
    + list(SELF_VARS)
)

URDF_FROZEN_EDGES: dict[tuple[str, str], dict[str, float]] = {
    ("lhip", "lknee"): {"alpha_trust": 1.0},
    ("lknee", "lankle"): {"alpha_trust": 1.0},
    ("rhip", "rknee"): {"alpha_trust": 1.0},
    ("rknee", "rankle"): {"alpha_trust": 1.0},
    ("lshoulder", "lelbow"): {"alpha_trust": 1.0},
    ("lelbow", "lwrist"): {"alpha_trust": 1.0},
    ("rshoulder", "relbow"): {"alpha_trust": 1.0},
    ("relbow", "rwrist"): {"alpha_trust": 1.0},
    ("spine_yaw", "spine_pitch"): {"alpha_trust": 1.0},
    ("spine_pitch", "neck_yaw"): {"alpha_trust": 1.0},
    ("neck_yaw", "neck_pitch"): {"alpha_trust": 1.0},
}
HUMANOID_KINEMATIC_EDGE_PRIORS: tuple[tuple[str, str], ...] = tuple(URDF_FROZEN_EDGES.keys())

KINEMATIC_CHAINS: tuple[tuple[str, ...], ...] = (
    ("lhip", "lknee", "lankle"),
    ("rhip", "rknee", "rankle"),
    ("lshoulder", "lelbow", "lwrist"),
    ("rshoulder", "relbow", "rwrist"),
    ("spine_yaw", "spine_pitch", "neck_yaw", "neck_pitch"),
)

FIXED_BASE_VARS: list[str] = (
    ARM_VARS
    + SPINE_VARS
    + HEAD_VARS
    + list(VESTIBULAR_VARS)
    + CUBE_VARS
    + SANDBOX_VARS
    + list(MOTOR_INTENT_VARS)
    + list(MOTOR_OBSERVABLE_VARS)
    + list(SELF_VARS)
)

_RANGES: dict[str, tuple[float, float]] = {}
for v in TORSO_VARS[:2]:
    _RANGES[v] = (-1.5, 1.5)
_RANGES["com_z"] = (0.0, 1.8)
_RANGES["torso_roll"] = (-1.2, 1.2)
_RANGES["torso_pitch"] = (-1.2, 1.2)
for v in SPINE_VARS:
    _RANGES[v] = (-1.2, 1.2)
for v in HEAD_VARS:
    _RANGES[v] = (-1.2, 1.2)
for v in VESTIBULAR_VARS:
    _RANGES[v] = (-1.0, 1.0)
for v in LEG_VARS:
    if "knee" in v:
        _RANGES[v] = (-3.14, 0.1)
    else:
        _RANGES[v] = (-1.5, 1.5)
for v in ARM_VARS:
    _RANGES[v] = (-2.0, 2.0)
for v in ("lwrist", "rwrist"):
    _RANGES[v] = (-0.8, 0.8)
for v in FOOT_VARS:
    _RANGES[v] = (-0.1, 0.5)
for v in CUBE_VARS:
    if v.endswith("_z"):
        _RANGES[v] = (-0.1, 1.0)
    else:
        _RANGES[v] = (-2.0, 2.0)
for v in ("ball_x", "ball_y"):
    _RANGES[v] = (-2.5, 2.5)
_RANGES["ball_z"] = (0.0, 2.0)
_RANGES["lever_pin"] = (0.0, 1.0)
_RANGES["target_dist"] = (0.0, 4.0)
_RANGES["floor_friction"] = (0.1, 1.0)
_RANGES["stack_height"] = (0.0, 0.8)
_RANGES["stability_score"] = (0.0, 1.0)
for _mv in MOTOR_INTENT_VARS + MOTOR_OBSERVABLE_VARS:
    _RANGES[_mv] = (0.0, 1.0)
for _sv in SELF_VARS:
    _RANGES[_sv] = (0.0, 1.0)

HUMANOID_URDF_LEGACY_SCALE = 0.45
# URDF linear dims in-file were scaled down ~3.25×; keep same world size as pre-rescale × 0.225.
HUMANOID_URDF_GLOBAL_SCALING = 0.225 * 3.25
_HSZ = HUMANOID_URDF_GLOBAL_SCALING / HUMANOID_URDF_LEGACY_SCALE

FALLEN_Z = 0.30
STAND_Z = 0.80
HUMANOID_URDF_STAND_EULER = (np.pi / 2, 0.0, 0.0)
HUMANOID_URDF_SPAWN_Z = 1.10
PENTHOUSE_SPAWN_X = 1.85
PENTHOUSE_SPAWN_Y = -1.15

# Путь к URDF относительно корня пакета `engine/` (features/humanoid/constants.py → .. ×3)
HUMANOID_URDF_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "humanoid" / "humanoid.urdf"
