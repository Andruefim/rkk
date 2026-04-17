"""PyBullet humanoid + scene (URDF, physics bg, camera)."""
from __future__ import annotations

import base64
import os
import threading
import time
from io import BytesIO
from pathlib import Path

import numpy as np
import torch

from engine.features.humanoid.constants import (
    ARM_VARS,
    HEAD_VARS,
    HUMANOID_URDF_GLOBAL_SCALING,
    HUMANOID_URDF_LEGACY_SCALE,
    HUMANOID_URDF_PATH,
    HUMANOID_URDF_SPAWN_Z,
    HUMANOID_URDF_STAND_EULER,
    LEG_VARS,
    PENTHOUSE_SPAWN_X,
    PENTHOUSE_SPAWN_Y,
    SPINE_VARS,
    STAND_Z,
    _RANGES,
)
from engine.features.humanoid.deps import PIL_AVAILABLE, PILImage, pb, pbd
from engine.features.humanoid.kinematics import (
    _forward_kinematics_skeleton,
    _np_quat_from_axis_angle,
    _np_quat_mul,
    _PB_VEC_TO_THREE,
    _rotmat_to_xyzw,
)
from engine.features.humanoid.sandbox import InstrumentalSandbox

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
        pb.setPhysicsEngineParameter(numSolverIterations=80, physicsClientId=self.client)

        self.floor_id = pb.loadURDF("plane.urdf", physicsClientId=self.client)
        # Экспорт для Three.js (tx,ty,tz = y-up): penthouse + рычаг/мишень
        self._static_scene_export: list[dict] = []

        self.cube_ids = []
        cube_configs = [
            {"pos": [-3.8, 2.6, 0.16], "size": 0.15, "mass": 2.0, "color": [1.0, 0.4, 0.1, 1]},
            {"pos": [3.4, -3.2, 0.14], "size": 0.125, "mass": 0.5, "color": [0.2, 0.7, 1.0, 1]},
            {"pos": [-1.8, -3.5, 0.24], "size": 0.20, "mass": 6.0, "color": [0.25, 0.85, 0.35, 1]},
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
        self.lever_center = np.array([-4.2, 2.5, 0.10], dtype=np.float64)
        self.target_pad = np.array([3.6, -3.9, 0.02], dtype=np.float64)
        self.lever_trigger_r = 0.26
        self._ball_start = [1.8, 1.4, 0.1375]
        br = 0.1125
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
        self._build_lever_pedestal()
        self._build_target_marker()
        self.prop_ids: list[int] = []
        self._prop_starts: list[list[float]] = []
        self._prop_meta: list[dict] = []
        self._build_rich_environment()

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

        for i in range(self.n_joints):
            jt = self._joint_types[i]
            if jt != pb.JOINT_FIXED:
                pb.changeDynamics(self.robot_id, i,
                    jointDamping=0.5, linearDamping=0.04, angularDamping=0.04,
                    physicsClientId=self.client)

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
    def _static_box_three(
        self,
        tx: float,
        ty: float,
        tz: float,
        hx: float,
        hy: float,
        hz: float,
        rgb: tuple[float, float, float],
        rx: float = 0.0,
        ry: float = 0.0,
        rz: float = 0.0,
        *,
        style: str = "default",
    ) -> None:
        """Статика: центр в координатах Three.js (tx,ty,tz), полуразмеры (hx,hy,hz), y-up."""
        half = [hx, hz, hy]
        pos_pb = [tx, tz, ty]
        orn = pb.getQuaternionFromEuler([rx, ry, rz])
        cid = self.client
        col = pb.createCollisionShape(pb.GEOM_BOX, halfExtents=half, physicsClientId=cid)
        vis = pb.createVisualShape(
            pb.GEOM_BOX,
            halfExtents=half,
            rgbaColor=[rgb[0], rgb[1], rgb[2], 1.0],
            physicsClientId=cid,
        )
        pb.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=col,
            baseVisualShapeIndex=vis,
            basePosition=pos_pb,
            baseOrientation=orn,
            physicsClientId=cid,
        )
        self._static_scene_export.append({
            "kind": "box",
            "tx": float(tx),
            "ty": float(ty),
            "tz": float(tz),
            "hx": float(hx),
            "hy": float(hy),
            "hz": float(hz),
            "r": rgb[0],
            "g": rgb[1],
            "b": rgb[2],
            "rx": float(rx),
            "ry": float(ry),
            "rz": float(rz),
            "style": style,
        })

    def _static_cylinder_pb(
        self,
        pos_pb: list[float],
        radius: float,
        height: float,
        rgb: tuple[float, float, float],
        *,
        style: str = "default",
    ) -> None:
        """Статический цилиндр (ось Z в PyBullet = высота в Three)."""
        cid = self.client
        col = pb.createCollisionShape(
            pb.GEOM_CYLINDER, radius=radius, height=height, physicsClientId=cid
        )
        vis = pb.createVisualShape(
            pb.GEOM_CYLINDER,
            radius=radius,
            length=height,
            rgbaColor=[rgb[0], rgb[1], rgb[2], 1.0],
            physicsClientId=cid,
        )
        pb.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=col,
            baseVisualShapeIndex=vis,
            basePosition=pos_pb,
            physicsClientId=cid,
        )
        self._static_scene_export.append({
            "kind": "cylinder",
            "tx": float(pos_pb[0]),
            "ty": float(pos_pb[2]),
            "tz": float(pos_pb[1]),
            "radius": float(radius),
            "height": float(height),
            "r": rgb[0],
            "g": rgb[1],
            "b": rgb[2],
            "rx": 0.0,
            "ry": 0.0,
            "rz": 0.0,
            "style": style,
        })

    def _static_sphere_pb(
        self,
        pos_pb: list[float],
        radius: float,
        rgb: tuple[float, float, float],
        *,
        style: str = "default",
    ) -> None:
        cid = self.client
        col = pb.createCollisionShape(pb.GEOM_SPHERE, radius=radius, physicsClientId=cid)
        vis = pb.createVisualShape(
            pb.GEOM_SPHERE,
            radius=radius,
            rgbaColor=[rgb[0], rgb[1], rgb[2], 1.0],
            physicsClientId=cid,
        )
        pb.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=col,
            baseVisualShapeIndex=vis,
            basePosition=pos_pb,
            physicsClientId=cid,
        )
        self._static_scene_export.append({
            "kind": "sphere",
            "tx": float(pos_pb[0]),
            "ty": float(pos_pb[2]),
            "tz": float(pos_pb[1]),
            "radius": float(radius),
            "r": rgb[0],
            "g": rgb[1],
            "b": rgb[2],
            "style": style,
        })

    def _export_torus_visual(
        self,
        tx: float,
        ty: float,
        tz: float,
        radius: float,
        tube: float,
        rgb: tuple[float, float, float],
        rx: float,
        ry: float,
        rz: float,
    ) -> None:
        """Только визуал для Three.js; коллизии см. _ribbon_collision_ring."""
        self._static_scene_export.append({
            "kind": "torus",
            "tx": float(tx),
            "ty": float(ty),
            "tz": float(tz),
            "radius": float(radius),
            "tube": float(tube),
            "r": rgb[0],
            "g": rgb[1],
            "b": rgb[2],
            "rx": float(rx),
            "ry": float(ry),
            "rz": float(rz),
            "style": "chrome",
        })

    def _ribbon_collision_ring(self, cx: float, cy: float, z: float, major_r: float) -> None:
        import math

        cid = self.client
        n = 16
        for i in range(n):
            ang = (i / n) * math.tau
            mx = cx + math.cos(ang) * major_r
            my = cy + math.sin(ang) * major_r
            col = pb.createCollisionShape(
                pb.GEOM_BOX, halfExtents=[0.22, 0.09, 0.12], physicsClientId=cid
            )
            pb.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=col,
                baseVisualShapeIndex=-1,
                basePosition=[mx, my, z],
                baseOrientation=pb.getQuaternionFromEuler([0.0, 0.0, ang]),
                physicsClientId=cid,
            )

    def _add_cafe_set_pb(self, px: float, pz: float, scale: float) -> None:
        import math

        s = float(scale)
        self._static_cylinder_pb(
            [px, pz, 0.78 * s], 0.38 * s, 0.025 * s, (0.95, 0.97, 1.0), style="glass"
        )
        self._static_cylinder_pb(
            [px, pz, 0.38 * s], 0.06 * s, 0.76 * s, (0.9, 0.92, 0.94), style="chrome"
        )
        for c in range(4):
            ang = (c / 4) * math.tau + 0.4
            sx = px + math.cos(ang) * 0.55 * s
            sz = pz + math.sin(ang) * 0.55 * s
            self._static_box_three(
                sx,
                0.42 * s,
                sz,
                0.07 * s,
                0.03 * s,
                0.07 * s,
                (0.12, 0.14, 0.17),
                style="seat",
            )
            self._static_cylinder_pb(
                [sx, sz, 0.6 * s], 0.015 * s, 0.38 * s, (0.88, 0.9, 0.92), style="chrome"
            )

    def _build_central_tree(self) -> None:
        self._static_cylinder_pb([0.0, 0.0, 0.11], 1.42, 0.22, (0.94, 0.96, 0.98), style="planter")
        self._static_cylinder_pb([0.0, 0.0, 0.55], 0.16, 1.1, (0.24, 0.16, 0.1), style="wood")
        self._static_sphere_pb([0.0, 0.0, 1.55], 1.15, (0.15, 0.48, 0.22), style="plant")

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
        self._static_scene_export.append({
            "kind": "box",
            "tx": float(lc[0]), "ty": float(lc[2]), "tz": float(lc[1]),
            "hx": 0.07, "hy": 0.055, "hz": 0.07,
            "r": 0.88, "g": 0.62, "b": 0.15,
            "rx": 0.0, "ry": 0.0, "rz": 0.0,
        })

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
        self._static_scene_export.append({
            "kind": "box",
            "tx": float(tp[0]), "ty": float(tp[2]), "tz": float(tp[1]),
            "hx": 0.22, "hy": 0.012, "hz": 0.22,
            "r": 0.15, "g": 0.85, "b": 0.55,
            "rx": 0.0, "ry": 0.0, "rz": 0.0,
        })

    def _build_rich_environment(self) -> None:
        """
        Penthouse lounge (как Three.js): центральное дерево, кафе-столы, хром-ленты,
        киоск/барьер по периметру. Координаты Three.js: tx=x_pb, ty=z_pb, tz=y_pb.
        """
        import math

        cid = self.client

        self._build_central_tree()
        cafe_sets = [
            (-5.0, 3.0, 1.0),
            (4.0, -4.0, 0.95),
            (-3.0, -5.0, 1.0),
            (6.0, 2.0, 0.9),
        ]
        for px, pz, sc in cafe_sets:
            self._add_cafe_set_pb(px, pz, sc)

        for k in range(3):
            major = 1.2 + 0.3 * k
            tx = -7.0 + k * 3.0
            ty = 0.85
            tz = -6.0
            self._ribbon_collision_ring(tx, tz, ty, major)
            self._export_torus_visual(
                tx,
                ty,
                tz,
                major,
                0.08,
                (0.9, 0.92, 0.95),
                math.pi / 2.3,
                0.0,
                float(k) * 0.9,
            )

        # Низкий круглый «киоск» (как на референсе справа)
        self._static_cylinder_pb([7.2, -2.5, 0.45], 0.55, 0.9, (0.96, 0.98, 1.0), style="planter")
        self._static_box_three(7.2, 0.98, -2.5, 0.42, 0.32, 0.02, (0.4, 0.85, 0.95), style="default")

        # Изогнутые лавки (белые блоки)
        self._static_box_three(-5.5, 0.22, -4.5, 1.1, 0.11, 0.35, (0.96, 0.97, 0.99), style="default")
        self._static_box_three(5.5, 0.22, 4.0, 1.0, 0.11, 0.38, (0.96, 0.97, 0.99), style="default")

        prop_cfgs = [
            {"pos": [-4.2, 4.0, 0.09], "half": 0.07, "mass": 0.35, "color": [0.9, 0.35, 0.25, 1.0]},
            {"pos": [4.5, -0.5, 0.08], "half": 0.065, "mass": 0.28, "color": [0.35, 0.55, 0.95, 1.0]},
            {"pos": [-2.4, -4.5, 0.075], "half": 0.06, "mass": 0.22, "color": [0.45, 0.85, 0.4, 1.0]},
            {"pos": [3.0, 4.2, 0.085], "half": 0.07, "mass": 0.4, "color": [0.95, 0.75, 0.2, 1.0]},
            {"pos": [-1.0, -2.8, 0.07], "half": 0.065, "mass": 0.3, "color": [0.75, 0.45, 0.85, 1.0]},
            {"pos": [0.5, 5.0, 0.08], "half": 0.06, "mass": 0.25, "color": [0.3, 0.85, 0.75, 1.0]},
        ]
        for cfg in prop_cfgs:
            hs = float(cfg["half"])
            colp = pb.createCollisionShape(pb.GEOM_BOX, halfExtents=[hs, hs, hs], physicsClientId=cid)
            visp = pb.createVisualShape(
                pb.GEOM_BOX, halfExtents=[hs, hs, hs],
                rgbaColor=cfg["color"], physicsClientId=cid,
            )
            pos = cfg["pos"]
            bid = pb.createMultiBody(
                baseMass=float(cfg["mass"]),
                baseCollisionShapeIndex=colp,
                baseVisualShapeIndex=visp,
                basePosition=pos,
                physicsClientId=cid,
            )
            pb.changeDynamics(bid, -1, lateralFriction=0.55, physicsClientId=cid)
            self.prop_ids.append(bid)
            self._prop_starts.append([float(pos[0]), float(pos[1]), float(pos[2])])
            self._prop_meta.append({
                "half": hs,
                "r": float(cfg["color"][0]),
                "g": float(cfg["color"][1]),
                "b": float(cfg["color"][2]),
            })

    def _load_humanoid(self) -> int:
        local = HUMANOID_URDF_PATH
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
                    pos = [float(PENTHOUSE_SPAWN_X), float(PENTHOUSE_SPAWN_Y), float(HUMANOID_URDF_SPAWN_Z)]
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
                    positionGain=1.2, velocityGain=0.40,
                    maxVelocity=6.0, force=[12000.0, 12000.0, 12000.0],
                    physicsClientId=cid,
                )
            else:
                pb.setJointMotorControl2(
                    rid, i, controlMode=pb.POSITION_CONTROL,
                    targetPosition=0.0,
                    positionGain=0.90, velocityGain=0.25,
                    force=8000.0, physicsClientId=cid,
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

        for pid, p0 in zip(getattr(self, "prop_ids", []) or [], getattr(self, "_prop_starts", []) or []):
            pb.resetBasePositionAndOrientation(
                pid, p0, [0, 0, 0, 1], physicsClientId=cid,
            )
            pb.resetBaseVelocity(pid, [0, 0, 0], [0, 0, 0], physicsClientId=cid)

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
                    # БЫЛО: self._neck_euler[0] = 0.45 * real_pos
                    self._neck_euler[1] = 0.45 * real_pos  # ИСПРАВЛЕНО: Индекс 1 (Ось Y - Pitch)
                ex, ey, ez = float(self._neck_euler[0]), float(self._neck_euler[1]), float(self._neck_euler[2])
                q = pb.getQuaternionFromEuler((ex, ey, ez))
                motor_m(rid, jid, pb.POSITION_CONTROL, targetPosition=list(q),
                        positionGain=0.62, velocityGain=0.18, maxVelocity=4.0,
                        force=[4000.0, 4000.0, 4000.0], physicsClientId=cid)
                return

            if var_name in ("spine_yaw", "spine_pitch"):
                if not callable(motor_m) or jt != pb.JOINT_SPHERICAL:
                    return
                if var_name == "spine_yaw":
                    self._spine_euler[2] = 0.65 * real_pos
                else:
                    self._spine_euler[1] = 0.85 * real_pos  # Увеличена гибкость для подъема торса
                ex, ey, ez = float(self._spine_euler[0]), float(self._spine_euler[1]), float(self._spine_euler[2])
                q = pb.getQuaternionFromEuler((ex, ey, ez))
                motor_m(rid, jid, pb.POSITION_CONTROL, targetPosition=list(q),
                        positionGain=0.85, velocityGain=0.25, maxVelocity=3.5,
                        force=[10000.0, 10000.0, 10000.0], physicsClientId=cid)
                return

            is_leg = var_name in ("lhip", "rhip", "lknee", "rknee", "lankle", "rankle")
            if jt == pb.JOINT_SPHERICAL and callable(motor_m):
                if var_name == "lshoulder":
                    q = pb.getQuaternionFromEuler((0.32 * real_pos, 0.42 * real_pos, 0.28 * real_pos))
                elif var_name == "rshoulder":
                    q = pb.getQuaternionFromEuler((0.32 * real_pos, -0.42 * real_pos, -0.28 * real_pos))
                elif var_name == "lhip":
                    q = pb.getQuaternionFromEuler((0.4 * real_pos, 0.85 * real_pos, 0.1 * real_pos))
                elif var_name == "rhip":
                    q = pb.getQuaternionFromEuler((0.4 * real_pos, -0.85 * real_pos, -0.1 * real_pos))
                elif var_name == "lankle":
                    q = pb.getQuaternionFromEuler((-0.22 * real_pos, 0.1 * real_pos, 0.0))
                elif var_name == "rankle":
                    q = pb.getQuaternionFromEuler((-0.22 * real_pos, -0.1 * real_pos, 0.0))
                else:
                    q = [0.0, 0.0, 0.0, 1.0]
                if is_leg:
                    motor_m(rid, jid, pb.POSITION_CONTROL, targetPosition=list(q),
                            positionGain=0.85, velocityGain=0.25, maxVelocity=4.0,
                            force=[12000.0, 12000.0, 12000.0], physicsClientId=cid)
                else:
                    motor_m(rid, jid, pb.POSITION_CONTROL, targetPosition=list(q),
                            positionGain=0.52, velocityGain=0.15, maxVelocity=5.5,
                            force=[3000.0, 3000.0, 3000.0], physicsClientId=cid)
            else:
                if is_leg:
                    pb.setJointMotorControl2(
                        rid, jid, controlMode=pb.POSITION_CONTROL,
                        targetPosition=real_pos,
                        positionGain=0.80, velocityGain=0.20, force=8000.0, physicsClientId=cid,
                    )
                else:
                    pb.setJointMotorControl2(
                        rid, jid, controlMode=pb.POSITION_CONTROL,
                        targetPosition=real_pos,
                        positionGain=0.5, velocityGain=0.1, force=2000.0, physicsClientId=cid,
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
        for pid in getattr(self, "prop_ids", []) or []:
            pos, _ = pb.getBasePositionAndOrientation(pid, physicsClientId=self.client)
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
            props_out: list[dict] = []
            for i, pid in enumerate(getattr(self, "prop_ids", []) or []):
                p, _ = pb.getBasePositionAndOrientation(pid, physicsClientId=self.client)
                meta = (self._prop_meta[i] if i < len(self._prop_meta) else {"half": 0.07, "r": 0.5, "g": 0.5, "b": 0.5})
                hs = float(meta.get("half", 0.07))
                props_out.append({
                    "x": float(p[0]), "y": float(p[1]), "z": float(p[2]),
                    "hx": hs, "hy": hs, "hz": hs,
                    "r": float(meta.get("r", 0.5)),
                    "g": float(meta.get("g", 0.5)),
                    "b": float(meta.get("b", 0.5)),
                })
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
                "static_geometry": [{**row} for row in getattr(self, "_static_scene_export", [])],
                "props": props_out,
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
