"""
Фаза 2 (часть 1): иерархический каузальный граф L0–L4.

L1: агрегаты по кинематическим цепочкам (mean/std за окно) → виртуальные узлы l1_* в основном GNN (L2).
L3/L4: заготовки goal + привязка concept→goal (без полной интеграции планировщика в этой части).

Имена цепей совпадают с диаграммой; состав суставов — из engine.environment_humanoid.KINEMATIC_CHAINS.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from engine.causal_graph import CausalGraph
from engine.environment_humanoid import (
    KINEMATIC_CHAINS as _HUMANOID_KINEMATIC_CHAINS,
    MOTOR_INTENT_VARS,
    MOTOR_OBSERVABLE_VARS,
)

# Имена цепей → те же суставы, что в URDF LocalReflex
KINEMATIC_CHAINS: dict[str, list[str]] = {
    "left_leg": list(_HUMANOID_KINEMATIC_CHAINS[0]),
    "right_leg": list(_HUMANOID_KINEMATIC_CHAINS[1]),
    "left_arm": list(_HUMANOID_KINEMATIC_CHAINS[2]),
    "right_arm": list(_HUMANOID_KINEMATIC_CHAINS[3]),
    "spine": list(_HUMANOID_KINEMATIC_CHAINS[4]),
}

L1_AGGREGATION_WINDOW = max(1, int(os.environ.get("RKK_L1_AGG_WINDOW", "6")))


def hierarchical_graph_enabled() -> bool:
    return os.environ.get("RKK_HIERARCHICAL_GRAPH", "1").strip().lower() not in (
        "0",
        "false",
        "off",
        "no",
    )


@dataclass
class LocalReflexState:
    """Состояние одной кинематической цепочки на уровне L1."""

    chain_name: str
    joint_ids: list[str]
    values: dict[str, float] = field(default_factory=dict)
    history: list[dict[str, float]] = field(default_factory=list)
    graph: CausalGraph | None = None

    def aggregate(self) -> dict[str, float]:
        """История → виртуальные признаки для L2."""
        if not self.history:
            return {j: float(self.values.get(j, 0.5)) for j in self.joint_ids}
        agg: dict[str, float] = {}
        win = self.history[-L1_AGGREGATION_WINDOW:]
        for j in self.joint_ids:
            vals = [float(h.get(j, 0.5)) for h in win]
            agg[f"l1_{self.chain_name}_{j}_mean"] = float(np.mean(vals))
            agg[f"l1_{self.chain_name}_{j}_std"] = (
                float(np.std(vals, ddof=0)) if len(vals) > 1 else 0.0
            )
        return agg


class HierarchicalGraph:
    """
    Обёртка над CausalGraph (L2): агрегаты L1 инжектятся как узлы l1_*.
    """

    def __init__(self, base_graph: CausalGraph, device: torch.device):
        self.L2 = base_graph
        self.device = device

        self.L1_chains: dict[str, LocalReflexState] = {}
        for chain_name, joints in KINEMATIC_CHAINS.items():
            g = CausalGraph(device)
            for j in joints:
                g.set_node(j, 0.5)
            pairs: list[tuple[str, str]] = []
            for i in range(len(joints) - 1):
                a, b = joints[i], joints[i + 1]
                g.set_edge(a, b, 0.85, 0.97)
                pairs.append((a, b))
            if pairs:
                g.freeze_kinematic_priors(pairs)
            self.L1_chains[chain_name] = LocalReflexState(
                chain_name=chain_name,
                joint_ids=list(joints),
                graph=g,
            )

        motor_keys = list(MOTOR_OBSERVABLE_VARS + MOTOR_INTENT_VARS)
        g = CausalGraph(device)
        for k in motor_keys:
            g.set_node(k, 0.5)
        self.L1_chains["motor"] = LocalReflexState(
            chain_name="motor",
            joint_ids=motor_keys,
            graph=g,
        )

        self._l1_virtual_nodes: dict[str, float] = {}
        self._active_goal: dict[str, Any] | None = None
        self._concept_goals: dict[str, str] = {}
        self._l1_step_count = 0

    def step_l1(self, raw_obs: dict[str, float]) -> dict[str, float]:
        """Один шаг L1: обновить цепочки и при необходимости пересчитать агрегаты для L2."""
        for chain in self.L1_chains.values():
            for j in chain.joint_ids:
                if j in raw_obs:
                    chain.values[j] = float(raw_obs[j])
            chain.history.append(dict(chain.values))
            max_hist = L1_AGGREGATION_WINDOW * 4
            if len(chain.history) > max_hist:
                chain.history = chain.history[-(L1_AGGREGATION_WINDOW * 2) :]

        self._l1_step_count += 1
        if self._l1_step_count % L1_AGGREGATION_WINDOW == 0:
            virtual: dict[str, float] = {}
            for chain in self.L1_chains.values():
                virtual.update(chain.aggregate())
            self._l1_virtual_nodes = virtual
        return dict(self._l1_virtual_nodes)

    def inject_l1_virtual_nodes(self) -> int:
        """Записать виртуальные узлы в L2 (основной граф агента)."""
        to_add: dict[str, float] = {}
        for vnode, vval in self._l1_virtual_nodes.items():
            if vnode not in self.L2.nodes:
                to_add[vnode] = float(vval)
            else:
                self.L2.nodes[vnode] = float(vval)
        if not to_add:
            return 0
        base_ids = list(self.L2._node_ids)
        new_ids = base_ids + list(to_add.keys())
        vals = {nid: float(self.L2.nodes.get(nid, 0.5)) for nid in base_ids}
        vals.update(to_add)
        self.L2.rebind_variables(new_ids, vals, preserve_state=True)
        return len(to_add)

    def set_l3_goal(self, target_var: str, target_value: float, horizon: int = 20) -> None:
        self._active_goal = {
            "var": target_var,
            "value": float(target_value),
            "horizon": int(horizon),
        }
        if "self_goal_active" in self.L2.nodes:
            self.L2.nodes["self_goal_active"] = 0.9
        if "self_goal_target_dist" in self.L2.nodes:
            self.L2.nodes["self_goal_target_dist"] = float(target_value)

    def bind_concept_to_goal(self, concept_id: str, goal_var: str) -> None:
        self._concept_goals[str(concept_id)] = str(goal_var)
        if concept_id in self.L2.nodes and goal_var in self.L2.nodes:
            self.L2.set_edge(concept_id, goal_var, 0.55, 0.07)

    def snapshot(self) -> dict[str, Any]:
        return {
            "l1_chains": list(self.L1_chains.keys()),
            "l1_step": self._l1_step_count,
            "virtual_nodes": len(self._l1_virtual_nodes),
            "l1_agg_window": L1_AGGREGATION_WINDOW,
            "active_goal": self._active_goal,
            "concept_goals": dict(self._concept_goals),
        }
