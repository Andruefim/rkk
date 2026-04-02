"""
environment_pybullet.py — PyBullet 3D физическая среда.

Архитектура:
  - Сцена: N объектов (кубы/сферы) в 3D пространстве
  - State: object-centric [x, y, z, vx, vy, vz] per object → граф узлов
  - do() operator: применяем импульс к объекту (интервенционная жёсткость)
  - Ground truth: гравитация + столкновения → каузальный граф

Переменные (например 3 объекта):
  obj0_x, obj0_y, obj0_z, obj0_vx, obj0_vy, obj0_vz
  obj1_x, obj1_y, obj1_z, ...

Causal structure (скрытая):
  vx → x (интеграция)
  vy → y
  vz → z
  gravity → vy (−g при отсутствии опоры)
  collision(obj0, obj1) → vx, vy, vz (оба)

do() примеры:
  do(obj0_vx = 2.0)  → применяем импульс к obj0 по X
  do(obj1_vy = 3.0)  → подбрасываем obj1 вверх

Требования: pip install pybullet
"""
from __future__ import annotations

import time
import numpy as np
import torch
from dataclasses import dataclass
from typing import Callable

try:
    import pybullet as pb
    import pybullet_data
    PYBULLET_AVAILABLE = True
except ImportError:
    PYBULLET_AVAILABLE = False
    print("[PyBullet] pybullet not installed, using physics fallback")


# ─── Fallback симулятор (без PyBullet) ───────────────────────────────────────
class _FallbackPhysics:
    """Простая 2D физика: gravity + упругие столкновения."""

    G     = 9.81
    FLOOR = 0.0
    RESTITUTION = 0.7

    def __init__(self, n_objects: int):
        self.n = n_objects
        # state: [x, y, z, vx, vy, vz] per object
        self.state = np.zeros((n_objects, 6))
        # Инициализируем случайные позиции
        for i in range(n_objects):
            self.state[i] = [
                (i - n_objects / 2) * 0.5,   # x
                0.5 + i * 0.3,                # y (выше пола)
                0.0,                          # z
                np.random.uniform(-0.2, 0.2), # vx
                0.0,                          # vy
                0.0,                          # vz
            ]

    def step(self, dt: float = 0.016):
        """Один физический шаг."""
        for i in range(self.n):
            x, y, z, vx, vy, vz = self.state[i]

            # Гравитация
            vy -= self.G * dt

            # Интеграция позиции
            x += vx * dt
            y += vy * dt
            z += vz * dt

            # Отражение от пола
            if y < self.FLOOR:
                y  = self.FLOOR
                vy = -vy * self.RESTITUTION
                vx *= 0.95   # трение

            # Ограничение области
            if abs(x) > 3.0:
                vx = -vx * self.RESTITUTION
                x  = np.sign(x) * 3.0

            self.state[i] = [x, y, z, vx, vy, vz]

        # Столкновения между объектами (простые сферы r=0.2)
        for i in range(self.n):
            for j in range(i + 1, self.n):
                dx = self.state[j, 0] - self.state[i, 0]
                dy = self.state[j, 1] - self.state[i, 1]
                dist = np.sqrt(dx**2 + dy**2) + 1e-6
                if dist < 0.4:
                    # Упругое столкновение
                    nx, ny = dx / dist, dy / dist
                    dvx = self.state[j, 3] - self.state[i, 3]
                    dvy = self.state[j, 4] - self.state[i, 4]
                    dot = dvx * nx + dvy * ny
                    if dot < 0:
                        impulse = dot * self.RESTITUTION
                        self.state[i, 3] += impulse * nx
                        self.state[i, 4] += impulse * ny
                        self.state[j, 3] -= impulse * nx
                        self.state[j, 4] -= impulse * ny

    def apply_impulse(self, obj_id: int, vx: float = 0, vy: float = 0, vz: float = 0):
        """Применяем импульс к объекту."""
        if 0 <= obj_id < self.n:
            self.state[obj_id, 3] += vx
            self.state[obj_id, 4] += vy
            self.state[obj_id, 5] += vz

    def set_velocity(self, obj_id: int, vx: float = 0, vy: float = 0, vz: float = 0):
        """Устанавливаем скорость объекта (жёсткая интервенция)."""
        if 0 <= obj_id < self.n:
            self.state[obj_id, 3] = vx
            self.state[obj_id, 4] = vy
            self.state[obj_id, 5] = vz

    def get_state(self) -> np.ndarray:
        return self.state.copy()


# ─── PyBullet физика ──────────────────────────────────────────────────────────
class _PyBulletPhysics:
    """PyBullet wrapper."""

    def __init__(self, n_objects: int, gui: bool = False):
        self.n = n_objects
        self.gui = gui

        mode = pb.GUI if gui else pb.DIRECT
        self.client = pb.connect(mode)
        pb.setGravity(0, -9.81, 0, physicsClientId=self.client)
        pb.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client)

        # Пол
        self.plane_id = pb.loadURDF(
            "plane.urdf", physicsClientId=self.client
        )

        # Объекты (кубы разных цветов)
        self.body_ids = []
        for i in range(n_objects):
            half_size = 0.15
            col = pb.createCollisionShape(pb.GEOM_BOX, halfExtents=[half_size]*3, physicsClientId=self.client)
            vis = pb.createVisualShape(pb.GEOM_BOX, halfExtents=[half_size]*3,
                                       rgbaColor=[i/n_objects, 1-i/n_objects, 0.5, 1],
                                       physicsClientId=self.client)
            start_pos = [(i - n_objects/2) * 0.5, 0.5 + i * 0.3, 0]
            body = pb.createMultiBody(
                baseMass=1.0,
                baseCollisionShapeIndex=col,
                baseVisualShapeIndex=vis,
                basePosition=start_pos,
                physicsClientId=self.client,
            )
            self.body_ids.append(body)

        # Шаг симуляции 60 Hz
        pb.setTimeStep(1/60.0, physicsClientId=self.client)

    def step(self, n_substeps: int = 4):
        for _ in range(n_substeps):
            pb.stepSimulation(physicsClientId=self.client)

    def get_state(self) -> np.ndarray:
        state = np.zeros((self.n, 6))
        for i, body in enumerate(self.body_ids):
            pos, _ = pb.getBasePositionAndOrientation(body, physicsClientId=self.client)
            vel, _ = pb.getBaseVelocity(body, physicsClientId=self.client)
            state[i] = [pos[0], pos[1], pos[2], vel[0], vel[1], vel[2]]
        return state

    def apply_impulse(self, obj_id: int, vx: float = 0, vy: float = 0, vz: float = 0):
        if 0 <= obj_id < self.n:
            pb.applyExternalForce(
                self.body_ids[obj_id], -1,
                [vx * 50, vy * 50, vz * 50],   # force = impulse * mass_factor
                [0, 0, 0], pb.LINK_FRAME,
                physicsClientId=self.client,
            )

    def set_velocity(self, obj_id: int, vx: float, vy: float, vz: float):
        pb.resetBaseVelocity(
            self.body_ids[obj_id], [vx, vy, vz], [0, 0, 0],
            physicsClientId=self.client,
        )

    def __del__(self):
        try:
            pb.disconnect(self.client)
        except Exception:
            pass


# ─── EnvironmentPyBullet ─────────────────────────────────────────────────────
class EnvironmentPyBullet:
    """
    Физическая среда для РКК агента.

    Совместима с интерфейсом Environment:
      observe()       → dict[str, float]
      intervene()     → dict[str, float]
      discovery_rate() → float

    Переменные: objN_x, objN_y, objN_z, objN_vx, objN_vy, objN_vz
    Нормализованы в [0, 1] для совместимости с Value Layer.

    do() оператор:
      do(obj0_vx = 0.8)  → устанавливаем vx объекта 0 в 0.8 (нормализованное)
      Физика отвечает реальным результатом → нельзя галлюцинировать.
    """

    PRESET = "pybullet"

    # Нормализация: реальные физические диапазоны → [0, 1]
    POS_RANGE  = (-3.0, 3.0)    # координаты
    VEL_RANGE  = (-5.0, 5.0)    # скорости

    def __init__(
        self,
        n_objects:    int = 3,
        device:       torch.device | None = None,
        use_pybullet: bool = True,
        steps_per_do: int = 10,   # физических шагов после каждого do()
    ):
        self.n_objects    = n_objects
        self.device       = device or torch.device("cpu")
        self.steps_per_do = steps_per_do
        self.preset       = self.PRESET
        self.n_interventions = 0

        # Инициализируем физику
        if use_pybullet and PYBULLET_AVAILABLE:
            self._physics = _PyBulletPhysics(n_objects, gui=False)
            self._backend = "pybullet"
        else:
            self._physics = _FallbackPhysics(n_objects)
            self._backend = "fallback"

        print(f"[PhysicsEnv] {n_objects} objects, backend={self._backend}")

        # Генерируем имена переменных
        self._var_names = []
        for i in range(n_objects):
            for dim in ["x", "y", "z", "vx", "vy", "vz"]:
                self._var_names.append(f"obj{i}_{dim}")

        # Начальное состояние
        self._raw_state = self._physics.get_state()

    # ── Нормализация ───────────────────────────────────────────────────────────
    def _normalize(self, raw: np.ndarray) -> np.ndarray:
        """Нормализуем реальные значения в [0.05, 0.95]."""
        result = np.zeros_like(raw)
        for i in range(self.n_objects):
            for j, (lo, hi) in enumerate([
                self.POS_RANGE, self.POS_RANGE, self.POS_RANGE,
                self.VEL_RANGE, self.VEL_RANGE, self.VEL_RANGE,
            ]):
                val = raw[i, j]
                norm = (val - lo) / (hi - lo)
                result[i, j] = float(np.clip(norm, 0.05, 0.95))
        return result

    def _denormalize_velocity(self, norm_val: float, dim: str) -> float:
        """Денормализуем для do() оператора."""
        lo, hi = self.VEL_RANGE
        return float(norm_val * (hi - lo) + lo)

    # ── Observe ────────────────────────────────────────────────────────────────
    def observe(self) -> dict[str, float]:
        """Нормализованное наблюдение текущего состояния."""
        raw  = self._physics.get_state()
        norm = self._normalize(raw)
        obs  = {}
        for i in range(self.n_objects):
            dims = ["x", "y", "z", "vx", "vy", "vz"]
            for j, dim in enumerate(dims):
                obs[f"obj{i}_{dim}"] = float(norm[i, j])
        return obs

    # ── do() operator ─────────────────────────────────────────────────────────
    def intervene(self, variable: str, value: float) -> dict[str, float]:
        """
        Жёсткая интервенция через физику.
        variable: "obj0_vx", "obj1_vy", etc.
        value: нормализованное значение [0, 1]
        """
        self.n_interventions += 1

        # Парсим переменную
        parts = variable.split("_")
        if len(parts) < 2 or not parts[0].startswith("obj"):
            return self.observe()

        try:
            obj_id = int(parts[0][3:])
            dim    = "_".join(parts[1:])
        except ValueError:
            return self.observe()

        if obj_id >= self.n_objects:
            return self.observe()

        # Денормализуем и применяем
        if dim in ("vx", "vy", "vz"):
            real_val = self._denormalize_velocity(value, dim)
            dim_map = {"vx": 0, "vy": 1, "vz": 2}
            kwargs = {dim: real_val}
            self._physics.set_velocity(obj_id, **{
                "vx": real_val if dim == "vx" else 0,
                "vy": real_val if dim == "vy" else 0,
                "vz": real_val if dim == "vz" else 0,
            })
        # Позиции не трогаем напрямую (это было бы нефизично)
        # Вместо этого применяем импульс

        # Прокручиваем физику
        for _ in range(self.steps_per_do):
            self._physics.step()

        return self.observe()

    # ── Discovery rate ─────────────────────────────────────────────────────────
    def discovery_rate(self, agent_edges: list[dict]) -> float:
        """
        Оцениваем насколько граф агента соответствует физике.

        Ground truth каузальные паттерны:
          vx → x  (скорость интегрирует позицию)
          vy → y
          proximity → collision (через vx, vy)

        Проверяем есть ли у агента эти направленные рёбра.
        """
        gt_edges = set()
        for i in range(self.n_objects):
            gt_edges.add((f"obj{i}_vx", f"obj{i}_x"))
            gt_edges.add((f"obj{i}_vy", f"obj{i}_y"))
            gt_edges.add((f"obj{i}_vz", f"obj{i}_z"))

        # Кросс-объектные (столкновения)
        for i in range(self.n_objects):
            for j in range(self.n_objects):
                if i != j:
                    gt_edges.add((f"obj{i}_x", f"obj{j}_vx"))   # позиция→скорость другого
                    gt_edges.add((f"obj{i}_y", f"obj{j}_vy"))

        if not gt_edges:
            return 0.0

        hits = 0
        for e in agent_edges:
            key = (e.get("from_"), e.get("to"))
            if key in gt_edges:
                hits += 1

        return hits / len(gt_edges)

    @property
    def variable_ids(self) -> list[str]:
        return list(self._var_names)

    @property
    def variables(self) -> dict[str, float]:
        return self.observe()

    def gt_edges(self) -> list[dict]:
        """Ground truth рёбра для bootstrap priors."""
        edges = []
        for i in range(self.n_objects):
            edges.append({"from_": f"obj{i}_vx", "to": f"obj{i}_x", "weight": 0.9})
            edges.append({"from_": f"obj{i}_vy", "to": f"obj{i}_y", "weight": 0.9})
        return edges


# ─── Hardcoded seeds для PyBullet среды ─────────────────────────────────────
def pybullet_hardcoded_seeds(n_objects: int = 3) -> list[dict]:
    """Text priors для физической среды: скорость → позиция."""
    seeds = []
    for i in range(n_objects):
        for vdim, pdim in [("vx","x"), ("vy","y"), ("vz","z")]:
            seeds.append({
                "from_": f"obj{i}_{vdim}",
                "to":    f"obj{i}_{pdim}",
                "weight": 0.25,
                "alpha": 0.05,
            })
    return seeds
