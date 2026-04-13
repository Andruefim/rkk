"""Кватернионы, FK-скелет для fallback, базис PB→Three.js."""
from __future__ import annotations

import numpy as np

from engine.features.humanoid.constants import (
    HUMANOID_URDF_GLOBAL_SCALING,
    HUMANOID_URDF_LEGACY_SCALE,
)


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
        head,
        neck,
        spine,
        pelvis,
        lshld,
        rshld,
        lelbow,
        relbow,
        lhand,
        rhand,
        lhip_p,
        rhip_p,
        lknee_p,
        rknee_p,
        lfoot,
        rfoot,
        lsole,
        rsole,
    ]
    return [{"x": float(p[0]), "y": float(p[1]), "z": float(p[2])} for p in pts]
