"""Скрытые переменные песочницы (cube_temp, spring, stack metrics)."""
from __future__ import annotations

import numpy as np

from engine.features.humanoid.deps import pb


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
                self.cube_ids[1],
                -1,
                force.tolist(),
                list(p1),
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
                rid,
                jid,
                controlMode=pb.VELOCITY_CONTROL,
                targetVelocity=0,
                force=50.0 * friction,
                physicsClientId=cid,
            )
