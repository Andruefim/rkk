"""Fallback humanoid без PyBullet (детерминированная заглушка)."""
from __future__ import annotations

import os

import numpy as np

from engine.features.humanoid.constants import (
    ARM_VARS,
    HEAD_VARS,
    LEG_VARS,
    PENTHOUSE_SPAWN_X,
    PENTHOUSE_SPAWN_Y,
    SPINE_VARS,
    STAND_Z,
    _RANGES,
)
from engine.features.humanoid.kinematics import _forward_kinematics_skeleton
from engine.features.humanoid.sandbox import InstrumentalSandbox
from engine.features.humanoid.vestibular import gravity_dir_fallback_torso_then_neck


class _FallbackHumanoid(InstrumentalSandbox):
    def __init__(self, fixed_root: bool = False):
        self.fixed_root = fixed_root
        self.joints = {v: 0.0 for v in LEG_VARS + ARM_VARS + SPINE_VARS + HEAD_VARS}
        self.com = np.array([0.0, 0.0, STAND_Z])
        self.torso_euler = np.zeros(3)
        self.cubes = np.array(
            [
                [-3.8, 2.6, 0.16],
                [3.4, -3.2, 0.14],
                [-1.8, -3.5, 0.24],
            ]
        )
        self.ball = np.array([1.8, 1.4, 0.14], dtype=np.float64)
        self.ball_id = 9000
        self._object_registry.append({
            "ref": "ball",
            "id": "ball",
            "body_id": int(self.ball_id),
            "semantic": "ball",
            "movable": True,
            "mass": 0.14,
            "x": float(self.ball[0]),
            "y": float(self.ball[1]),
            "z": float(self.ball[2]),
            "source": "sandbox_spawn",
        })
        self._lever_center = np.array([-4.2, 2.5, 0.1], dtype=np.float64)
        self._target_pad = np.array([3.6, -3.9, 0.02], dtype=np.float64)
        self._vel = np.zeros(3)
        self._dt = 0.02
        self._manip_chair = np.array([0.0, 0.0, 0.4], dtype=np.float64)
        self._manip_chair_start = np.array([0.0, 0.0, 0.4], dtype=np.float64)
        self._object_registry: list[dict] = []
        self._physics_step_count = 0
        if os.environ.get("RKK_MANIP_CHAIR", "0").strip().lower() in ("1", "true", "yes", "on"):
            spawn = np.array([float(PENTHOUSE_SPAWN_X), float(PENTHOUSE_SPAWN_Y)], dtype=float)
            center = np.array([0.0, 0.0], dtype=float)
            delta = center - spawn
            n = float(np.linalg.norm(delta))
            fwd = delta / n if n > 1e-6 else np.array([1.0, 0.0])
            pos = spawn + fwd * 1.05
            self._manip_chair = np.array([pos[0], pos[1], 0.4], dtype=np.float64)
            self._manip_chair_start = self._manip_chair.copy()
            self._object_registry.append({
                "ref": "manip_chair_front",
                "id": "manip_chair_front",
                "body_id": 9001,
                "semantic": "chair",
                "movable": True,
                "mass": 5.5,
                "x": float(pos[0]),
                "y": float(pos[1]),
                "z": 0.4,
                "source": "manip_chair",
            })
        self._init_instrumental()

    def _compute_lever_pin(self) -> float:
        pts = [self.ball] + [self.cubes[i] for i in range(len(self.cubes))]
        lc = self._lever_center[:2]
        d = min(float(np.linalg.norm(p[:2] - lc[:2])) for p in pts)
        return float(np.clip(1.0 - d / 0.35, 0.0, 1.0))

    def step(self, n: int = 10):
        self._physics_step_count += int(n)
        if self.fixed_root:
            self.com[:2] += np.random.normal(0, 0.0005, 2)
            self._tick_hidden_state()
            self._apply_friction_to_ankle_joints()
            return
        for _ in range(n):
            balance = abs(self.torso_euler[0]) + abs(self.torso_euler[1])
            self.com[2] += (-0.02 * balance + 0.001 * (np.random.rand() - 0.5)) * self._dt
            self.com[2] = np.clip(self.com[2], 0.0, 1.5)
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
        s["com_x"] = float(self.com[0])
        s["com_y"] = float(self.com[1])
        s["com_z"] = float(self.com[2])
        s["torso_roll"] = float(self.torso_euler[0])
        s["torso_pitch"] = float(self.torso_euler[1])
        for v in LEG_VARS + ARM_VARS + SPINE_VARS + HEAD_VARS:
            s[v] = float(self.joints.get(v, 0.0))
        tr, tp, ty = (
            float(self.torso_euler[0]),
            float(self.torso_euler[1]),
            float(self.torso_euler[2]),
        )
        gx, gy, gz = gravity_dir_fallback_torso_then_neck(
            tr,
            tp,
            ty,
            float(self.joints.get("neck_yaw", 0.0)),
            float(self.joints.get("neck_pitch", 0.0)),
        )
        s["vestibular_gx"], s["vestibular_gy"], s["vestibular_gz"] = gx, gy, gz
        s["lfoot_z"] = float(lf_z)
        s["rfoot_z"] = float(rf_z)
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
        return [{"x": float(c[0]), "y": float(c[1]), "z": float(c[2])} for c in self.cubes]

    def get_sandbox_scene_extras(self) -> dict:
        pts = [self.ball] + [self.cubes[i].copy() for i in range(len(self.cubes))]
        d_lv = min(float(np.linalg.norm(p[:2] - self._lever_center[:2])) for p in pts)
        lp = float(np.clip(1.0 - d_lv / 0.35, 0.0, 1.0))
        registry = []
        for row in getattr(self, "_object_registry", []) or []:
            entry = dict(row)
            if str(entry.get("ref")) == "manip_chair_front":
                entry["x"] = float(self._manip_chair[0])
                entry["y"] = float(self._manip_chair[1])
                entry["z"] = float(self._manip_chair[2])
            registry.append(entry)
        ball = {
            "x": float(self.ball[0]),
            "y": float(self.ball[1]),
            "z": float(self.ball[2]),
        }
        if getattr(self, "ball_id", None) is not None:
            ball["body_id"] = int(self.ball_id)
        return {
            "ball": ball,
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
            "static_geometry": [],
            "props": [],
            "registry": registry,
        }

    def get_physics_object_positions(self) -> dict:
        return self.get_sandbox_scene_extras()

    def get_manipulation_target_pose(self, ref: str) -> dict | None:
        if str(ref) != "manip_chair_front":
            return None
        return {
            "ref": "manip_chair_front",
            "body_id": 9001,
            "x": float(self._manip_chair[0]),
            "y": float(self._manip_chair[1]),
            "z": float(self._manip_chair[2]),
        }

    def apply_manipulation_push(
        self,
        body_id: int,
        direction_xy: tuple[float, float],
        force_n: float | None = None,
    ) -> dict:
        if int(body_id) != 9001:
            return {"applied": False, "reason": "unknown_body"}
        dx, dy = float(direction_xy[0]), float(direction_xy[1])
        n = float(np.hypot(dx, dy))
        if n < 1e-6:
            return {"applied": False, "reason": "zero_direction"}
        scale = 0.004 * float(force_n or 38.0)
        self._manip_chair[0] += (dx / n) * scale
        self._manip_chair[1] += (dy / n) * scale
        for row in getattr(self, "_object_registry", []) or []:
            if row.get("body_id") == 9001:
                row["x"] = float(self._manip_chair[0])
                row["y"] = float(self._manip_chair[1])
        return {"applied": True, "body_id": 9001, "force_n": float(force_n or 38.0)}

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
        self.ball = np.array([1.8, 1.4, 0.14], dtype=np.float64)
        self._manip_chair = self._manip_chair_start.copy()
        for row in getattr(self, "_object_registry", []) or []:
            if row.get("body_id") == 9001:
                row["x"] = float(self._manip_chair[0])
                row["y"] = float(self._manip_chair[1])
                row["z"] = float(self._manip_chair[2])
        self._reset_instrumental_hidden()

    def enable_fixed_root(self) -> None:
        self.fixed_root = True

    def disable_fixed_root(self) -> None:
        self.fixed_root = False

    def apply_variant_physics(
        self,
        *,
        mass_scale: float = 1.30,
        friction_scale: float = 0.70,
        com_offset_z: float = 0.02,
    ) -> None:
        self._variant_mass_scale = float(mass_scale)
        self._variant_friction_scale = float(friction_scale)
        self._variant_com_offset_z = float(com_offset_z)

    def get_dynamics_params(self) -> dict[str, float]:
        """Fallback stub — aligns keys with PyBullet ``get_dynamics_params``."""
        msc = float(getattr(self, "_variant_mass_scale", 1.0))
        fsc = float(getattr(self, "_variant_friction_scale", 1.0))
        return {
            "schema_version": 1.0,
            "backend": 0.0,
            "fixed_root": 1.0 if self.fixed_root else 0.0,
            "gravity_x": 0.0,
            "gravity_y": 0.0,
            "gravity_z": -9.81,
            "timestep": float(self._dt),
            "base_mass": 70.0 * msc,
            "base_lateral_friction": 0.8 * fsc,
            "floor_lateral_friction": 1.0 * fsc,
            "urdf_global_scale": 1.0,
            "variant_com_offset_z": float(getattr(self, "_variant_com_offset_z", 0.0)),
        }
