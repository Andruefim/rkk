"""
visual_grounding.py — Level 1-B: Visual Body Grounding.

Связывает визуальный cortex (SlotAttention) с физическим телом (PyBullet joints).

Проблема: slot_3 и "lknee" — это разные пространства.
  - SlotAttention видит пиксели → создаёт абстрактные slot_0..K
  - GNN видит joint values → lhip, lknee, lankle...
  - Без моста они не связаны — агент не может «видеть что его нога упала»

Решение:
  1. Project joints to camera: PyBullet joint world positions →
     (view_matrix, proj_matrix) → нормализованные 2D coords [0,1]×[0,1]
  2. Overlap with attn_masks: для каждого slot_k взять attention mask (48×48),
     найти center-of-mass маски, сравнить с joint 2D positions
  3. Score: overlap_score[k][joint] = max(1 - dist / threshold, 0)
  4. Inject: set_edge(f"slot_{k}", best_joint, score * weight) в GNN

Дополнительно:
  - Строим slot→body_part map (словарь для UI: slot_0 → "lknee")
  - Обновляем VLM lexicon автоматически из grounding (slot label = joint name)
  - Фиксируем temporal stability: если slot k стабильно указывает на joint j
    за N frames → "confirmed grounding" → более сильное ребро

RKK_VISUAL_GROUNDING=1          — включить grounding (default)
RKK_GROUNDING_EVERY=30          — тиков между обновлениями
RKK_GROUNDING_DIST_THRESH=0.22  — max normalized distance для match
RKK_GROUNDING_MIN_SCORE=0.25    — min overlap score для inject
RKK_GROUNDING_EDGE_W=0.35       — вес ребра при confirmed grounding
RKK_GROUNDING_CONFIRM_N=8       — frames для подтверждения
"""
from __future__ import annotations

import os
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

try:
    import pybullet as pb
    _PB_AVAILABLE = True
except ImportError:
    _PB_AVAILABLE = False


def _env_float(key: str, default: float) -> float:
    try:
        return float(os.environ.get(key, str(default)))
    except ValueError:
        return default


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.environ.get(key, str(default)))
    except ValueError:
        return default


def grounding_enabled() -> bool:
    return os.environ.get("RKK_VISUAL_GROUNDING", "1").strip().lower() not in ("0", "false", "no", "off")


# ── Joint descriptors ─────────────────────────────────────────────────────────
# Приоритетный список суставов для grounding (наиболее визуально заметные)
GROUNDING_JOINTS = [
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "neck", "spine", "chest", "root",
    "left_wrist", "right_wrist",
]

# Маппинг PyBullet link names → humanoid variable names
LINK_TO_VAR = {
    "left_hip": "lhip", "right_hip": "rhip",
    "left_knee": "lknee", "right_knee": "rknee",
    "left_ankle": "lankle", "right_ankle": "rankle",
    "left_shoulder": "lshoulder", "right_shoulder": "rshoulder",
    "left_elbow": "lelbow", "right_elbow": "relbow",
    "neck": "neck_yaw", "spine": "spine_yaw",
    "chest": "spine_pitch", "root": "com_z",
    "left_wrist": "lelbow",
    "right_wrist": "relbow",
}

# Цвета для UI (slot colors)
SLOT_COLORS_HEX = [
    "#ff5050", "#50c8ff", "#50ff64", "#ffc850",
    "#c850ff", "#ff8c50", "#50ffdc", "#b4b4ff",
]


# ── Joint projection ──────────────────────────────────────────────────────────
def project_joints_to_camera(
    physics_client: int,
    robot_id: int,
    link_names: list[str],
    view_matrix: list[float],
    proj_matrix: list[float],
    image_width: int = 288,
    image_height: int = 216,
) -> dict[str, tuple[float, float]]:
    """
    Проецируем world-space позиции суставов в нормализованные 2D coords [0,1].

    Returns: dict[link_name → (u, v)] где (0,0)=top-left, (1,1)=bottom-right.
    """
    if not _PB_AVAILABLE:
        return {}

    projections: dict[str, tuple[float, float]] = {}

    # View и proj матрицы как np arrays 4x4 (column-major из PyBullet → transpose)
    V = np.array(view_matrix, dtype=np.float64).reshape(4, 4).T
    P = np.array(proj_matrix, dtype=np.float64).reshape(4, 4).T

    def world_to_ndc(pos_world: np.ndarray) -> tuple[float, float] | None:
        """Converts 3D world position to normalized [0,1] image coords."""
        p4 = np.array([pos_world[0], pos_world[1], pos_world[2], 1.0], dtype=np.float64)
        # View space
        v_space = V @ p4
        # Clip space
        c_space = P @ v_space
        if abs(c_space[3]) < 1e-8:
            return None
        # NDC: [-1,1]
        ndc = c_space[:3] / c_space[3]
        # Depth check (in front of camera)
        if ndc[2] < -1.0 or ndc[2] > 1.0:
            return None
        # Convert to image coords [0,1] (note: y is flipped in image space)
        u = float((ndc[0] + 1.0) * 0.5)
        v = float((1.0 - ndc[1]) * 0.5)  # flip Y
        return (u, v)

    n_joints = pb.getNumJoints(robot_id, physicsClientId=physics_client)
    for i in range(n_joints):
        try:
            info = pb.getJointInfo(robot_id, i, physicsClientId=physics_client)
            link_name = info[12].decode("utf-8")
            if link_name not in GROUNDING_JOINTS:
                continue
            state = pb.getLinkState(
                robot_id, i,
                computeForwardKinematics=1,
                physicsClientId=physics_client,
            )
            world_pos = np.array(state[4][:3], dtype=np.float64)
            uv = world_to_ndc(world_pos)
            if uv is not None:
                projections[link_name] = uv
        except Exception:
            continue

    return projections


def get_camera_matrices_from_pybullet_humanoid(
    physics_client: int,
    robot_id: int,
    link_names: list[str],
    image_width: int = 288,
    image_height: int = 216,
) -> tuple[list[float] | None, list[float] | None]:
    """
    Получаем view и projection матрицы, которые использует humanoid camera.
    Пытаемся восстановить их из ego camera (neck/chest), иначе default side view.
    """
    if not _PB_AVAILABLE:
        return None, None

    try:
        # Try to find neck and chest for ego camera
        neck_pos = None
        chest_pos = None
        n_joints = pb.getNumJoints(robot_id, physicsClientId=physics_client)
        for i in range(n_joints):
            info = pb.getJointInfo(robot_id, i, physicsClientId=physics_client)
            ln = info[12].decode("utf-8")
            st = pb.getLinkState(robot_id, i, computeForwardKinematics=1, physicsClientId=physics_client)
            wp = np.array(st[4][:3], dtype=np.float64)
            if ln == "neck":
                neck_pos = wp
            elif ln == "chest":
                chest_pos = wp

        if neck_pos is not None and chest_pos is not None:
            up = neck_pos - chest_pos
            ln = float(np.linalg.norm(up))
            if ln > 1e-6:
                up = up / ln
            fwd = np.array([1.0, 0.0, 0.0], dtype=np.float64)
            fwd = fwd - float(np.dot(fwd, up)) * up
            fn = float(np.linalg.norm(fwd))
            if fn > 1e-5:
                fwd = fwd / fn
            else:
                fwd = np.array([1.0, 0.0, 0.0], dtype=np.float64)
            eye = neck_pos + 0.06 * up + 0.04 * fwd
            target = eye + 2.4 * fwd
            vm = pb.computeViewMatrix(
                eye.tolist(), target.tolist(), up.tolist(),
                physicsClientId=physics_client,
            )
        else:
            # Fallback: side view
            base_pos, _ = pb.getBasePositionAndOrientation(robot_id, physicsClientId=physics_client)
            vm = pb.computeViewMatrix(
                [base_pos[0] + 2.2, base_pos[1] - 2.2, base_pos[2] + 1.6],
                [base_pos[0], base_pos[1], base_pos[2] + 0.75],
                [0, 0, 1],
                physicsClientId=physics_client,
            )

        pm = pb.computeProjectionMatrixFOV(
            fov=60, aspect=image_width / image_height,
            nearVal=0.1, farVal=15.0,
            physicsClientId=physics_client,
        )
        return list(vm), list(pm)
    except Exception:
        return None, None


# ── Overlap computation ───────────────────────────────────────────────────────
def compute_slot_joint_overlap(
    attn_masks: torch.Tensor,          # (K, H', W') — от cortex.encode()
    joint_projections: dict[str, tuple[float, float]],  # link_name → (u, v)
    dist_threshold: float = 0.22,
) -> dict[int, dict[str, float]]:
    """
    Вычисляем overlap между slot attention masks и проецированными суставами.

    Для каждого slot k и joint j:
      1. Находим center-of-mass attention mask k (нормализованные coords)
      2. Вычисляем расстояние до joint j projection
      3. overlap_score = max(0, 1 - dist / threshold)

    Returns: {slot_k: {link_name: score}}
    """
    K, H, W = attn_masks.shape
    overlap: dict[int, dict[str, float]] = {k: {} for k in range(K)}

    # Precompute center of mass for each slot mask
    slot_centers: list[tuple[float, float] | None] = []
    masks_np = attn_masks.float().cpu().numpy()

    for k in range(K):
        mask = masks_np[k]  # (H', W')
        total = float(mask.sum())
        if total < 1e-6:
            slot_centers.append(None)
            continue
        # Grid coords
        ys = np.linspace(0.0, 1.0, H)
        xs = np.linspace(0.0, 1.0, W)
        grid_x, grid_y = np.meshgrid(xs, ys)
        cx = float((mask * grid_x).sum() / total)
        cy = float((mask * grid_y).sum() / total)
        slot_centers.append((cx, cy))

    # Compute overlap scores
    for k, center in enumerate(slot_centers):
        if center is None:
            continue
        cx, cy = center
        for link_name, (jx, jy) in joint_projections.items():
            # Clamp joint projection to [0,1]
            jx = float(np.clip(jx, 0.0, 1.0))
            jy = float(np.clip(jy, 0.0, 1.0))
            dist = float(np.sqrt((cx - jx) ** 2 + (cy - jy) ** 2))
            score = max(0.0, 1.0 - dist / max(dist_threshold, 1e-6))
            if score > 0.01:
                overlap[k][link_name] = score

    return overlap


# ── Grounding state ───────────────────────────────────────────────────────────
@dataclass
class GroundingEntry:
    """Трекинг grounding для одного (slot, joint) pair."""
    slot_k: int
    link_name: str
    var_name: str          # humanoid variable name
    scores: deque = field(default_factory=lambda: deque(maxlen=16))
    confirmed: bool = False
    edge_injected: bool = False

    @property
    def mean_score(self) -> float:
        if not self.scores:
            return 0.0
        return float(np.mean(self.scores))


class VisualGroundingController:
    """
    Контроллер визуального grounding.

    Интегрируется в simulation._run_single_agent_timestep_inner через:
      1. _maybe_run_visual_grounding() вызывается каждые N тиков
      2. Обновляет slot_to_joint_map и VLM lexicon
      3. Инжектирует causal edges slot_k → joint_var в GNN

    Требует:
      - visual_env: EnvironmentVisual с _last_attn не None
      - base_env: EnvironmentHumanoid с _PyBulletHumanoid
      - agent.graph: CausalGraph
    """

    def __init__(self):
        self._last_tick: int = -999_999
        self._grounding: dict[tuple[int, str], GroundingEntry] = {}  # (slot_k, link) → entry
        self.slot_to_joint: dict[int, str] = {}     # slot_k → best var_name
        self.slot_to_link: dict[int, str] = {}      # slot_k → best link_name
        self.slot_to_score: dict[int, float] = {}   # slot_k → best score
        self.total_edges_injected: int = 0
        self.total_updates: int = 0
        self._confirm_n = _env_int("RKK_GROUNDING_CONFIRM_N", 8)

    def should_run(self, tick: int) -> bool:
        if not grounding_enabled():
            return False
        every = _env_int("RKK_GROUNDING_EVERY", 30)
        return (tick - self._last_tick) >= every

    def update(
        self,
        tick: int,
        visual_env,          # EnvironmentVisual
        agent_graph,         # CausalGraph
        physics_client: int | None = None,
        robot_id: int | None = None,
        link_names: list[str] | None = None,
        image_width: int = 288,
        image_height: int = 216,
    ) -> dict[str, Any]:
        """
        Полный цикл grounding update.
        Returns: summary dict with edges injected, confirmed slots.
        """
        self._last_tick = tick
        self.total_updates += 1
        result: dict[str, Any] = {
            "tick": tick,
            "edges_injected": 0,
            "confirmed_slots": 0,
            "slot_to_joint": {},
            "ok": False,
        }

        if not grounding_enabled():
            return result

        # Get latest attention masks from visual cortex
        attn_masks = getattr(visual_env, "_last_attn", None)
        if attn_masks is None:
            return result

        # Get joint projections from PyBullet
        joint_projections: dict[str, tuple[float, float]] = {}

        if physics_client is not None and robot_id is not None:
            ln = link_names or []
            vm, pm = get_camera_matrices_from_pybullet_humanoid(
                physics_client, robot_id, ln, image_width, image_height
            )
            if vm is not None and pm is not None:
                joint_projections = project_joints_to_camera(
                    physics_client, robot_id, ln, vm, pm, image_width, image_height
                )

        # Fallback: use approximate skeleton from agent graph nodes
        if not joint_projections:
            joint_projections = self._estimate_from_graph(agent_graph)

        if not joint_projections:
            return result

        # Compute overlaps
        dist_thresh = _env_float("RKK_GROUNDING_DIST_THRESH", 0.22)
        min_score = _env_float("RKK_GROUNDING_MIN_SCORE", 0.25)
        overlap = compute_slot_joint_overlap(attn_masks, joint_projections, dist_thresh)

        # Update grounding entries
        n_slots = attn_masks.shape[0]
        for k in range(n_slots):
            slot_scores = overlap.get(k, {})
            if not slot_scores:
                continue
            for link_name, score in slot_scores.items():
                key = (k, link_name)
                if key not in self._grounding:
                    var_name = LINK_TO_VAR.get(link_name, link_name)
                    self._grounding[key] = GroundingEntry(
                        slot_k=k, link_name=link_name, var_name=var_name
                    )
                entry = self._grounding[key]
                entry.scores.append(score)
                # Check confirmation
                if len(entry.scores) >= self._confirm_n and entry.mean_score > min_score:
                    entry.confirmed = True

        # Find best joint for each slot
        new_slot_to_joint: dict[int, str] = {}
        new_slot_to_link: dict[int, str] = {}
        new_slot_to_score: dict[int, float] = {}

        for k in range(n_slots):
            best_score = 0.0
            best_link = ""
            best_var = ""
            for (sk, link_name), entry in self._grounding.items():
                if sk != k:
                    continue
                s = entry.mean_score
                if s > best_score:
                    best_score = s
                    best_link = link_name
                    best_var = entry.var_name
            if best_score >= min_score:
                new_slot_to_joint[k] = best_var
                new_slot_to_link[k] = best_link
                new_slot_to_score[k] = best_score

        self.slot_to_joint = new_slot_to_joint
        self.slot_to_link = new_slot_to_link
        self.slot_to_score = new_slot_to_score

        # Inject confirmed grounding edges into GNN
        edge_w = _env_float("RKK_GROUNDING_EDGE_W", 0.35)
        edges_injected = 0
        confirmed_count = 0

        for (sk, link_name), entry in self._grounding.items():
            if not entry.confirmed:
                continue
            confirmed_count += 1
            slot_node = f"slot_{sk}"
            var_node = entry.var_name

            # Check both nodes exist in graph
            if slot_node not in agent_graph.nodes:
                continue
            if var_node not in agent_graph.nodes:
                # Try phys_ prefix for hybrid mode
                phys_var = f"phys_{var_node}"
                if phys_var not in agent_graph.nodes:
                    continue
                var_node = phys_var

            # Inject edge with strength proportional to mean_score
            w = float(np.clip(entry.mean_score * edge_w, 0.08, edge_w))
            try:
                agent_graph.set_edge(slot_node, var_node, w, alpha=0.07)
                if not entry.edge_injected:
                    entry.edge_injected = True
                    edges_injected += 1
                    self.total_edges_injected += 1
            except Exception:
                pass

        # Update VLM lexicon with grounding info (Phase M: simulation._maybe_run_visual_grounding
        # вызывает _phase_m_sync_from_vision() сразу после этого update).
        if hasattr(visual_env, "_slot_lexicon") and new_slot_to_joint:
            for k, var_name in new_slot_to_joint.items():
                slot_id = f"slot_{k}"
                existing = visual_env._slot_lexicon.get(slot_id) or {}
                # Only update if no existing label or new score is much better
                existing_conf = float(existing.get("confidence", 0.0))
                new_conf = new_slot_to_score.get(k, 0.0)
                if new_conf > existing_conf + 0.1:
                    visual_env._slot_lexicon[slot_id] = {
                        "label": _var_to_body_label(var_name),
                        "likely_phys": [var_name],
                        "confidence": round(new_conf, 3),
                        "source": "grounding",
                    }

        result["edges_injected"] = edges_injected
        result["confirmed_slots"] = confirmed_count
        result["slot_to_joint"] = {f"slot_{k}": v for k, v in new_slot_to_joint.items()}
        result["ok"] = True
        return result

    def _estimate_from_graph(
        self, graph
    ) -> dict[str, tuple[float, float]]:
        """
        Fallback: грубая оценка joint projections из graph.nodes.
        Использует нормализованные значения суставов как proxy для 2D позиции.
        Это очень приближённо, но позволяет работать без PyBullet.
        """
        nodes = graph.nodes

        def n(k: str) -> float:
            return float(nodes.get(k, nodes.get(f"phys_{k}", 0.5)))

        com_z = n("com_z")
        com_x = n("com_x")

        # Map joints to approximate image coords based on body geometry
        # These are rough heuristics for a standing humanoid seen from the front
        projections: dict[str, tuple[float, float]] = {}

        # Normalize com_z: 0.3=fallen, 1.0=standing → image y: 0.1=top, 0.9=bottom
        body_top = float(np.clip(1.0 - com_z * 0.65, 0.05, 0.85))
        body_cx = float(np.clip(0.5 - (com_x - 0.5) * 0.4, 0.1, 0.9))

        projections["neck"]           = (body_cx, body_top + 0.05)
        projections["chest"]          = (body_cx, body_top + 0.15)
        projections["spine"]          = (body_cx, body_top + 0.22)
        projections["root"]           = (body_cx, body_top + 0.30)
        projections["left_hip"]       = (body_cx - 0.08, body_top + 0.32)
        projections["right_hip"]      = (body_cx + 0.08, body_top + 0.32)
        projections["left_knee"]      = (body_cx - 0.08, body_top + 0.50)
        projections["right_knee"]     = (body_cx + 0.08, body_top + 0.50)
        projections["left_ankle"]     = (body_cx - 0.08, body_top + 0.68)
        projections["right_ankle"]    = (body_cx + 0.08, body_top + 0.68)
        projections["left_shoulder"]  = (body_cx - 0.14, body_top + 0.10)
        projections["right_shoulder"] = (body_cx + 0.14, body_top + 0.10)
        projections["left_elbow"]     = (body_cx - 0.18, body_top + 0.22)
        projections["right_elbow"]    = (body_cx + 0.18, body_top + 0.22)

        return projections

    def snapshot(self) -> dict[str, Any]:
        confirmed = sum(1 for e in self._grounding.values() if e.confirmed)
        return {
            "enabled": grounding_enabled(),
            "total_updates": self.total_updates,
            "total_edges_injected": self.total_edges_injected,
            "n_grounding_pairs": len(self._grounding),
            "n_confirmed": confirmed,
            "slot_to_joint": {f"slot_{k}": v for k, v in self.slot_to_joint.items()},
            "slot_to_score": {f"slot_{k}": round(v, 3) for k, v in self.slot_to_score.items()},
            "last_tick": self._last_tick,
        }


def _var_to_body_label(var_name: str) -> str:
    """Convert humanoid variable name to human-readable body part label."""
    mapping = {
        "lhip": "left hip", "rhip": "right hip",
        "lknee": "left knee", "rknee": "right knee",
        "lankle": "left ankle", "rankle": "right ankle",
        "lshoulder": "left shoulder", "rshoulder": "right shoulder",
        "lelbow": "left elbow", "relbow": "right elbow",
        "spine_yaw": "spine", "spine_pitch": "torso",
        "neck_yaw": "neck", "com_z": "body center",
    }
    # Handle phys_ prefix
    key = var_name[5:] if var_name.startswith("phys_") else var_name
    return mapping.get(key, key.replace("_", " "))


# ── Integration helpers ───────────────────────────────────────────────────────
def get_pybullet_state_from_humanoid_env(base_env) -> tuple[int | None, int | None, list[str]]:
    """
    Extract (physics_client, robot_id, link_names) from EnvironmentHumanoid.
    Handles both _PyBulletHumanoid and fallback.
    """
    if base_env is None:
        return None, None, []

    # Try direct attribute access (works for EnvironmentHumanoid wrapping _PyBulletHumanoid)
    sim = getattr(base_env, "_sim", None)
    if sim is None:
        return None, None, []

    client = getattr(sim, "client", None)
    robot_id = getattr(sim, "robot_id", None)
    link_names = getattr(sim, "link_names", [])

    if client is None or robot_id is None:
        return None, None, []

    return int(client), int(robot_id), list(link_names)
