"""
environment_robot.py — PyBullet Robot Sandbox (Фаза 11).

Среда: 3-сочленённый манипулятор-рука (упрощённый KUKA) в пространстве.
Задача: AGI обнаруживает законы кинематики через do()-интервенции.

Ground-truth каузальная структура (скрытая от агента):
  j0_vel → j0_pos  (интеграция)
  j1_vel → j1_pos
  j2_vel → j2_pos
  j0_pos, j1_pos, j2_pos → ee_x, ee_y, ee_z  (прямая кинематика)
  ee_x, ee_y, ee_z + target_x, target_y, target_z → contact

Переменные (12):
  j0_pos, j0_vel,  j1_pos, j1_vel,  j2_pos, j2_vel  (суставы)
  ee_x, ee_y, ee_z                                   (конец руки)
  target_x, target_y, target_z                       (цель)

do() оператор:
  do(j0_vel = 0.8)  → устанавливаем скорость сустава 0, шагаем физику N раз
  do(j1_vel = 0.3)  → поворачиваем локоть

Камера:
  pb.getCameraImage() → RGB-кадр для UI
  Endpoint: GET /camera/frame → base64 PNG

Обнаружение структуры (discovery_rate):
  Проверяем есть ли рёбра jN_vel → jN_pos (кинематика суставов)
  Проверяем есть ли рёбра jN_pos → ee_x/y/z (прямая кинематика)
"""
from __future__ import annotations

import numpy as np
import torch
import base64
from io import BytesIO

try:
    import pybullet as pb
    import pybullet_data
    from PIL import Image as PILImage
    PYBULLET_AVAILABLE = True
    PIL_AVAILABLE = True
except ImportError:
    PYBULLET_AVAILABLE = False
    PIL_AVAILABLE = False
    try:
        from PIL import Image as PILImage
        PIL_AVAILABLE = True
    except ImportError:
        PIL_AVAILABLE = False

# ─── Константы ────────────────────────────────────────────────────────────────
N_JOINTS   = 3
POS_RANGE  = (-np.pi, np.pi)      # угол сустава
VEL_RANGE  = (-2.0, 2.0)          # угловая скорость
EE_RANGE   = (-0.8, 0.8)          # координата конца руки (метры)
TARGET_RANGE = (-0.5, 0.5)        # диапазон цели

# Имена переменных
VAR_NAMES = []
for i in range(N_JOINTS):
    VAR_NAMES += [f"j{i}_pos", f"j{i}_vel"]
VAR_NAMES += ["ee_x", "ee_y", "ee_z", "target_x", "target_y", "target_z"]
# = 12 переменных


class _FallbackRobot:
    """Аналитическая кинематика когда PyBullet недоступен."""

    L = [0.3, 0.25, 0.2]   # длины звеньев

    def __init__(self):
        self.joints_pos = np.zeros(N_JOINTS)
        self.joints_vel = np.zeros(N_JOINTS)
        self.target     = np.array([0.3, 0.2, 0.0])
        self._dt        = 0.05

    def step(self, n: int = 8):
        for _ in range(n):
            self.joints_pos = np.clip(
                self.joints_pos + self.joints_vel * self._dt,
                POS_RANGE[0], POS_RANGE[1]
            )
            self.joints_vel *= 0.95   # демпфирование

    def set_velocity(self, joint_idx: int, vel: float):
        if 0 <= joint_idx < N_JOINTS:
            self.joints_vel[joint_idx] = np.clip(vel, VEL_RANGE[0], VEL_RANGE[1])

    def get_ee_position(self) -> np.ndarray:
        """Планарная прямая кинематика."""
        x = y = z = 0.0
        angle = 0.0
        for i in range(N_JOINTS):
            angle += self.joints_pos[i]
            x += self.L[i] * np.cos(angle)
            y += self.L[i] * np.sin(angle)
        z = 0.15 * self.joints_pos[0]   # небольшая высота от первого сустава
        return np.array([x, y, z])

    def get_state(self) -> dict:
        ee = self.get_ee_position()
        state = {}
        for i in range(N_JOINTS):
            state[f"j{i}_pos"] = float(self.joints_pos[i])
            state[f"j{i}_vel"] = float(self.joints_vel[i])
        state["ee_x"]     = float(ee[0])
        state["ee_y"]     = float(ee[1])
        state["ee_z"]     = float(ee[2])
        state["target_x"] = float(self.target[0])
        state["target_y"] = float(self.target[1])
        state["target_z"] = float(self.target[2])
        return state

    def randomize_target(self):
        self.target = np.random.uniform(TARGET_RANGE[0], TARGET_RANGE[1], 3)

    def get_frame_base64(self) -> str | None:
        return None


class _PyBulletRobot:
    """PyBullet манипулятор с реальной физикой."""

    # Длины звеньев для кастомного робота
    LINK_LENGTHS = [0.30, 0.25, 0.20]

    def __init__(self):
        self.client = pb.connect(pb.DIRECT)
        pb.setGravity(0, 0, -9.81, physicsClientId=self.client)
        pb.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client)

        # Пол
        pb.loadURDF("plane.urdf", physicsClientId=self.client)

        # Строим простой 3-звенный манипулятор программно
        self.body_id = self._build_robot()
        self.n_joints = pb.getNumJoints(self.body_id, physicsClientId=self.client)
        # Берём первые N_JOINTS суставов (вращательных)
        self.joint_ids = list(range(min(N_JOINTS, self.n_joints)))

        # Цель (визуальная сфера)
        self.target = np.array([0.3, 0.2, 0.3])
        target_col = pb.createCollisionShape(pb.GEOM_SPHERE, radius=0.03, physicsClientId=self.client)
        target_vis = pb.createVisualShape(pb.GEOM_SPHERE, radius=0.03,
                                          rgbaColor=[1, 0.3, 0.1, 0.8], physicsClientId=self.client)
        self.target_id = pb.createMultiBody(
            baseMass=0, baseCollisionShapeIndex=target_col,
            baseVisualShapeIndex=target_vis,
            basePosition=self.target.tolist(),
            physicsClientId=self.client
        )

        pb.setTimeStep(1/240., physicsClientId=self.client)

    def _build_robot(self) -> int:
        """Строим 3-звенный манипулятор из геометрических примитивов."""
        # Используем KUKA если доступен, иначе строим вручную
        try:
            robot = pb.loadURDF(
                pybullet_data.getDataPath() + "/kuka_iiwa/model.urdf",
                basePosition=[0, 0, 0],
                physicsClientId=self.client
            )
            return robot
        except Exception:
            pass

        # Fallback: простой манипулятор из box + revolute joints
        half = 0.02
        link_masses     = [1.0] * N_JOINTS
        link_col_shapes = []
        link_vis_shapes = []
        link_positions  = []
        link_orientations = []
        link_inertial_pos = []
        link_inertial_orn = []
        link_parent_idx   = []
        link_joint_types  = []
        link_joint_axes   = []

        for i in range(N_JOINTS):
            length = self.LINK_LENGTHS[i]
            col = pb.createCollisionShape(
                pb.GEOM_BOX, halfExtents=[half, half, length/2],
                physicsClientId=self.client
            )
            vis = pb.createVisualShape(
                pb.GEOM_BOX, halfExtents=[half, half, length/2],
                rgbaColor=[0.2+i*0.25, 0.6, 0.9, 1.0],
                physicsClientId=self.client
            )
            link_col_shapes.append(col)
            link_vis_shapes.append(vis)
            link_positions.append([0, 0, length] if i > 0 else [0, 0, 0.15])
            link_orientations.append([0, 0, 0, 1])
            link_inertial_pos.append([0, 0, 0])
            link_inertial_orn.append([0, 0, 0, 1])
            link_parent_idx.append(i)
            link_joint_types.append(pb.JOINT_REVOLUTE)
            link_joint_axes.append([0, 1, 0])

        base_col = pb.createCollisionShape(pb.GEOM_BOX, halfExtents=[0.05, 0.05, 0.1],
                                            physicsClientId=self.client)
        base_vis = pb.createVisualShape(pb.GEOM_BOX, halfExtents=[0.05, 0.05, 0.1],
                                         rgbaColor=[0.3, 0.3, 0.4, 1.0],
                                         physicsClientId=self.client)
        robot = pb.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=base_col,
            baseVisualShapeIndex=base_vis,
            basePosition=[0, 0, 0.1],
            linkMasses=link_masses,
            linkCollisionShapeIndices=link_col_shapes,
            linkVisualShapeIndices=link_vis_shapes,
            linkPositions=link_positions,
            linkOrientations=link_orientations,
            linkInertialFramePositions=link_inertial_pos,
            linkInertialFrameOrientations=link_inertial_orn,
            linkParentIndices=link_parent_idx,
            linkJointTypes=link_joint_types,
            linkJointAxis=link_joint_axes,
            physicsClientId=self.client
        )
        return robot

    def step(self, n: int = 8):
        for _ in range(n):
            pb.stepSimulation(physicsClientId=self.client)

    def set_velocity(self, joint_idx: int, vel: float):
        if joint_idx < len(self.joint_ids):
            jid = self.joint_ids[joint_idx]
            pb.setJointMotorControl2(
                self.body_id, jid,
                controlMode=pb.VELOCITY_CONTROL,
                targetVelocity=float(np.clip(vel, VEL_RANGE[0], VEL_RANGE[1])),
                force=50.0,
                physicsClientId=self.client
            )

    def get_joint_state(self, joint_idx: int) -> tuple[float, float]:
        jid = self.joint_ids[min(joint_idx, len(self.joint_ids)-1)]
        state = pb.getJointState(self.body_id, jid, physicsClientId=self.client)
        return float(state[0]), float(state[1])   # pos, vel

    def get_ee_position(self) -> np.ndarray:
        """Позиция последнего звена."""
        last_j = self.joint_ids[-1]
        link_state = pb.getLinkState(self.body_id, last_j, physicsClientId=self.client)
        return np.array(link_state[0])   # world position

    def get_state(self) -> dict:
        state = {}
        for i in range(N_JOINTS):
            pos, vel = self.get_joint_state(i)
            state[f"j{i}_pos"] = pos
            state[f"j{i}_vel"] = vel
        ee = self.get_ee_position()
        state["ee_x"]     = float(ee[0])
        state["ee_y"]     = float(ee[1])
        state["ee_z"]     = float(ee[2])
        state["target_x"] = float(self.target[0])
        state["target_y"] = float(self.target[1])
        state["target_z"] = float(self.target[2])
        return state

    def randomize_target(self):
        self.target = np.random.uniform(TARGET_RANGE[0], TARGET_RANGE[1], 3)
        self.target[2] = abs(self.target[2]) + 0.1   # чуть выше пола
        pb.resetBasePositionAndOrientation(
            self.target_id, self.target.tolist(), [0, 0, 0, 1],
            physicsClientId=self.client
        )

    def get_frame_base64(self, width: int = 320, height: int = 240) -> str | None:
        """Рендерим кадр и возвращаем base64 PNG."""
        try:
            view = pb.computeViewMatrix(
                cameraEyePosition    = [0.8, -0.8, 0.8],
                cameraTargetPosition = [0.0, 0.0, 0.3],
                cameraUpVector       = [0, 0, 1],
                physicsClientId=self.client
            )
            proj = pb.computeProjectionMatrixFOV(
                fov=60, aspect=width/height,
                nearVal=0.1, farVal=5.0,
                physicsClientId=self.client
            )
            _, _, rgba, _, _ = pb.getCameraImage(
                width, height, view, proj,
                renderer=pb.ER_TINY_RENDERER,
                physicsClientId=self.client
            )
            if PIL_AVAILABLE:
                img  = PILImage.fromarray(rgba[:, :, :3].astype(np.uint8))
                buf  = BytesIO()
                img.save(buf, format="PNG")
                return base64.b64encode(buf.getvalue()).decode()
        except Exception:
            pass
        return None

    def get_joint_positions_world(self) -> list[dict]:
        """Мировые позиции суставов для Three.js визуализации."""
        result = [{"x": 0.0, "y": 0.0, "z": 0.1}]   # база
        for jid in self.joint_ids:
            try:
                state = pb.getLinkState(self.body_id, jid, physicsClientId=self.client)
                p = state[0]
                result.append({"x": float(p[0]), "y": float(p[1]), "z": float(p[2])})
            except Exception:
                result.append({"x": 0.0, "y": 0.0, "z": 0.0})
        return result

    def __del__(self):
        try:
            pb.disconnect(self.client)
        except Exception:
            pass


# ─── EnvironmentRobot ─────────────────────────────────────────────────────────
class EnvironmentRobot:
    """
    Среда робота-манипулятора для Singleton AGI.

    Интерфейс совместим с Environment и EnvironmentPyBullet:
      observe()        → dict[str, float]  (нормализованные значения)
      intervene()      → dict[str, float]
      discovery_rate() → float
      gt_edges()       → list[dict]

    do() оператор:
      do(j0_vel = 0.8) → устанавливаем скорость сустава 0
      do(j1_vel = 0.3) → поворачиваем локоть
    """

    PRESET = "robot"

    def __init__(self, device: torch.device | None = None, steps_per_do: int = 12):
        self.device       = device or torch.device("cpu")
        self.steps_per_do = steps_per_do
        self.preset       = self.PRESET
        self.n_interventions = 0
        self._target_randomize_every = 50

        if PYBULLET_AVAILABLE:
            self._robot   = _PyBulletRobot()
            self._backend = "pybullet"
        else:
            self._robot   = _FallbackRobot()
            self._backend = "fallback"

        print(f"[RobotEnv] backend={self._backend}, joints={N_JOINTS}, vars={len(VAR_NAMES)}")

    # ── Нормализация ───────────────────────────────────────────────────────────
    def _normalize_val(self, key: str, val: float) -> float:
        if key.endswith("_pos"):
            lo, hi = POS_RANGE
        elif key.endswith("_vel"):
            lo, hi = VEL_RANGE
        elif key.startswith("target_") or key.startswith("ee_"):
            lo, hi = EE_RANGE
        else:
            lo, hi = -1.0, 1.0
        norm = (val - lo) / (hi - lo)
        return float(np.clip(norm, 0.05, 0.95))

    def _denormalize_vel(self, norm: float) -> float:
        lo, hi = VEL_RANGE
        return float(norm * (hi - lo) + lo)

    # ── Observe ────────────────────────────────────────────────────────────────
    def observe(self) -> dict[str, float]:
        raw = self._robot.get_state()
        return {k: self._normalize_val(k, v) for k, v in raw.items()}

    @property
    def variables(self) -> dict[str, float]:
        return self.observe()

    @property
    def variable_ids(self) -> list[str]:
        return list(VAR_NAMES)

    # ── do() ──────────────────────────────────────────────────────────────────
    def intervene(self, variable: str, value: float) -> dict[str, float]:
        self.n_interventions += 1

        # Периодически рандомизируем цель → агент исследует разные положения
        if self.n_interventions % self._target_randomize_every == 0:
            self._robot.randomize_target()

        # Парсим переменную
        if "_vel" in variable and variable.startswith("j"):
            try:
                joint_idx = int(variable[1])   # j0_vel → 0
                real_vel  = self._denormalize_vel(value)
                self._robot.set_velocity(joint_idx, real_vel)
            except (ValueError, IndexError):
                pass

        # Шагаем физику
        self._robot.step(self.steps_per_do)

        return self.observe()

    # ── Discovery rate ─────────────────────────────────────────────────────────
    def discovery_rate(self, agent_edges: list[dict]) -> float:
        """
        GT каузальная структура:
          jN_vel → jN_pos   (интеграция)
          jN_pos → ee_x/y/z (кинематика)
        """
        gt = set()
        for i in range(N_JOINTS):
            gt.add((f"j{i}_vel", f"j{i}_pos"))
            for coord in ["ee_x", "ee_y", "ee_z"]:
                gt.add((f"j{i}_pos", coord))

        hits = sum(
            1 for e in agent_edges
            if (e.get("from_"), e.get("to")) in gt
        )
        return hits / len(gt) if gt else 0.0

    def gt_edges(self) -> list[dict]:
        edges = []
        for i in range(N_JOINTS):
            edges.append({"from_": f"j{i}_vel", "to": f"j{i}_pos", "weight": 0.9})
            for coord in ["ee_x", "ee_y", "ee_z"]:
                edges.append({"from_": f"j{i}_pos", "to": coord, "weight": 0.6})
        return edges

    # ── Camera ────────────────────────────────────────────────────────────────
    def get_frame_base64(self) -> str | None:
        return self._robot.get_frame_base64()

    def get_joint_positions_world(self) -> list[dict]:
        """Для Three.js визуализации скелетона."""
        if hasattr(self._robot, "get_joint_positions_world"):
            return self._robot.get_joint_positions_world()
        # Fallback: аналитическая FK
        robot = self._robot
        positions = [{"x": 0.0, "y": 0.0, "z": 0.1}]
        angle = 0.0
        x = y = z = 0.0
        for i in range(N_JOINTS):
            angle += robot.joints_pos[i]
            x += robot.L[i] * np.cos(angle)
            y += robot.L[i] * np.sin(angle)
            positions.append({"x": float(x), "y": float(y), "z": float(z + 0.1)})
        return positions

    def get_target(self) -> dict:
        t = self._robot.target
        return {"x": float(t[0]), "y": float(t[1]), "z": float(t[2])}


# ─── Hardcoded seeds для робота ───────────────────────────────────────────────
def robot_hardcoded_seeds() -> list[dict]:
    """Text priors: скорость → позиция сустава, позиция → конец руки."""
    seeds = []
    for i in range(N_JOINTS):
        seeds.append({"from_": f"j{i}_vel", "to": f"j{i}_pos", "weight": 0.25, "alpha": 0.05})
        seeds.append({"from_": f"j{i}_pos", "to": "ee_z", "weight": 0.18, "alpha": 0.05})
    seeds.append({"from_": "j0_pos", "to": "ee_x", "weight": 0.22, "alpha": 0.05})
    seeds.append({"from_": "j1_pos", "to": "ee_y", "weight": 0.20, "alpha": 0.05})
    return seeds
