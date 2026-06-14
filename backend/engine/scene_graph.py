"""
Scene object affordance layer — GNN node updates for world objects (Sprint 7.0).
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np

ARM_REACH_M = 0.85


def scene_graph_enabled() -> bool:
    return os.environ.get("RKK_SCENE_GRAPH", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


@dataclass
class SceneObject:
    obj_id: str
    obj_x: float = 0.0
    obj_y: float = 0.0
    obj_z: float = 0.0
    obj_mass: float = 1.0
    obj_distance_to_agent: float = 10.0
    obj_in_reach: float = 0.0
    obj_type: str = "box"


@dataclass
class SceneGraphObserver:
    objects: dict[str, SceneObject] = field(default_factory=dict)
    last_affordances: list[str] = field(default_factory=list)

    def observe_scene(self, sim: Any) -> dict[str, SceneObject]:
        if not scene_graph_enabled():
            return {}
        self.objects.clear()
        agent_xy = self._agent_xy(sim)
        extras = self._scene_extras(sim)
        if not extras:
            return {}

        specs: list[tuple[str, dict, str]] = []
        ball = extras.get("ball")
        if isinstance(ball, dict):
            specs.append(("ball", ball, "sphere"))
        tgt = extras.get("delivery_target")
        if isinstance(tgt, dict):
            specs.append(("target", tgt, "target"))
        props = extras.get("props") or []
        if isinstance(props, list):
            for i, p in enumerate(props[:6]):
                if isinstance(p, dict):
                    specs.append((f"prop_{i}", p, str(p.get("type", "box"))))

        for oid, pos, otype in specs:
            ox = float(pos.get("x", pos.get("hx", 0.0)))
            oy = float(pos.get("y", pos.get("hy", 0.0)))
            oz = float(pos.get("z", pos.get("hz", 0.5)))
            dist = float(math.hypot(ox - agent_xy[0], oy - agent_xy[1]))
            in_reach = 1.0 if dist < ARM_REACH_M else 0.0
            self.objects[oid] = SceneObject(
                obj_id=oid,
                obj_x=ox,
                obj_y=oy,
                obj_z=oz,
                obj_mass=float(pos.get("mass", 1.0)),
                obj_distance_to_agent=dist,
                obj_in_reach=in_reach,
                obj_type=otype,
            )
        return dict(self.objects)

    def update_gnn(self, sim: Any) -> dict[str, float]:
        """Write scene scalars into graph.nodes (prefixed scene_* keys)."""
        objs = self.observe_scene(sim)
        if not objs:
            return {}
        nodes = sim.agent.graph.nodes
        patch: dict[str, float] = {}
        nearest = min(objs.values(), key=lambda o: o.obj_distance_to_agent)
        patch["scene_nearest_dist"] = float(np.clip(nearest.obj_distance_to_agent / 3.0, 0.0, 1.0))
        patch["scene_nearest_in_reach"] = float(nearest.obj_in_reach)
        patch["scene_object_count"] = float(np.clip(len(objs) / 6.0, 0.0, 1.0))
        patch["scene_has_target"] = 1.0 if any(o.obj_type == "target" for o in objs.values()) else 0.0

        for oid, obj in objs.items():
            prefix = f"scene_{oid}"
            patch[f"{prefix}_dist"] = float(np.clip(obj.obj_distance_to_agent / 3.0, 0.0, 1.0))
            patch[f"{prefix}_in_reach"] = float(obj.obj_in_reach)
            patch[f"{prefix}_x"] = float(np.clip(0.5 + obj.obj_x * 0.1, 0.0, 1.0))
            patch[f"{prefix}_y"] = float(np.clip(0.5 + obj.obj_y * 0.1, 0.0, 1.0))

        for k, v in patch.items():
            nodes[k] = float(v)

        self.last_affordances = []
        if patch.get("scene_has_target", 0.0) > 0.5:
            self.last_affordances.append("ApproachObject")
        if patch.get("scene_nearest_in_reach", 0.0) > 0.5:
            self.last_affordances.append("ReachAndGrasp")
        return patch

    def snapshot(self) -> dict[str, Any]:
        in_reach = sum(1 for o in self.objects.values() if o.obj_in_reach > 0.5)
        return {
            "objects_tracked": len(self.objects),
            "objects_in_reach": in_reach,
            "affordances_available": list(self.last_affordances),
            "objects": [
                {
                    "id": o.obj_id,
                    "type": o.obj_type,
                    "dist": round(o.obj_distance_to_agent, 3),
                    "in_reach": bool(o.obj_in_reach > 0.5),
                }
                for o in self.objects.values()
            ],
        }

    @staticmethod
    def _agent_xy(sim: Any) -> tuple[float, float]:
        try:
            obs = dict(sim.agent.env.observe())
            cx = obs.get("com_forward_raw_m", obs.get("com_x_raw_m"))
            cy = obs.get("com_lateral_raw_m", obs.get("com_y_raw_m"))
            if cx is not None and cy is not None:
                return float(cx), float(cy)
        except Exception:
            pass
        return 0.0, 0.0

    @staticmethod
    def _scene_extras(sim: Any) -> dict:
        fn = getattr(sim, "get_sandbox_scene_extras", None)
        if callable(fn):
            try:
                return dict(fn() or {})
            except Exception:
                pass
        base = getattr(sim.agent.env, "base_env", None) or sim.agent.env
        sim_pb = getattr(base, "_sim", None)
        if sim_pb is not None and hasattr(sim_pb, "get_physics_object_positions"):
            try:
                return dict(sim_pb.get_physics_object_positions())
            except Exception:
                pass
        return {}
