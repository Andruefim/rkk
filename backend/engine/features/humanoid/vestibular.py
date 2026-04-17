"""Направление силы тяжести в локальной системе головы (вестибулярный ориентир)."""
from __future__ import annotations

import math

import numpy as np

# Мир: +Z вверх, ускорение свободного падения (0, 0, -g) → единичный вектор «вниз»
GRAVITY_DIR_WORLD = np.array([0.0, 0.0, -1.0], dtype=np.float64)


def _quat_to_rot_local_to_world(x: float, y: float, z: float, w: float) -> np.ndarray:
    """Ориентация звена: локальный вектор v_local → v_world = R @ v_local (как в PyBullet)."""
    x, y, z, w = float(x), float(y), float(z), float(w)
    n = math.sqrt(x * x + y * y + z * z + w * w)
    if n > 1e-12:
        x, y, z, w = x / n, y / n, z / n, w / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def gravity_dir_in_link_frame(quat_xyzw: tuple[float, float, float, float]) -> tuple[float, float, float]:
    """Единичный вектор направления гравитации, выраженный в системе координат звена (головы)."""
    R = _quat_to_rot_local_to_world(*quat_xyzw)
    gl = R.T @ GRAVITY_DIR_WORLD
    ln = float(np.linalg.norm(gl))
    if ln > 1e-9:
        gl = gl / ln
    return float(gl[0]), float(gl[1]), float(gl[2])


def _euler_xyz_to_quat(roll: float, pitch: float, yaw: float) -> tuple[float, float, float, float]:
    """Та же схема, что PyBullet getQuaternionFromEuler([roll, pitch, yaw])."""
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return x, y, z, w


def gravity_dir_fallback_torso_then_neck(
    torso_roll: float,
    torso_pitch: float,
    torso_yaw: float,
    neck_yaw: float,
    neck_pitch: float,
) -> tuple[float, float, float]:
    """
    Заглушка без PyBullet: грубая ориентация головы как торс, затем шея (yaw/pitch).
    Достаточно для согласованного масштаба с сырыми torso_* / neck_* в fallback-среде.
    """
    qx, qy, qz, qw = _euler_xyz_to_quat(
        float(torso_roll), float(torso_pitch), float(torso_yaw)
    )
    R0 = _quat_to_rot_local_to_world(qx, qy, qz, qw)
    nx, ny, nz, nw = _euler_xyz_to_quat(0.0, float(neck_pitch), float(neck_yaw))
    Rn = _quat_to_rot_local_to_world(nx, ny, nz, nw)
    R = R0 @ Rn
    gl = R.T @ GRAVITY_DIR_WORLD
    ln = float(np.linalg.norm(gl))
    if ln > 1e-9:
        gl = gl / ln
    return float(gl[0]), float(gl[1]), float(gl[2])
