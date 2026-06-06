"""Мир humanoid, переключатель среды, bounds."""
from __future__ import annotations

import os

import torch

from engine.agent import RKKAgent
from engine.value_layer import HomeostaticBounds


def resolve_torch_device(requested: str | None = None) -> torch.device:
    """
    Выбор устройства для GNN, демона, temporal и CausalVisualCortex.
    Переменная окружения RKK_DEVICE перекрывает аргумент.
    """
    req = (os.environ.get("RKK_DEVICE") or requested or "cuda").strip().lower()
    if req in ("mps", "mps:0"):
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        print("[RKK] RKK_DEVICE=mps, но MPS недоступен → CPU")
        return torch.device("cpu")
    if req == "cpu":
        return torch.device("cpu")
    if req.startswith("cuda"):
        if torch.cuda.is_available():
            return torch.device(req)
        print(
            f"[RKK] Запрошено {req}, но torch.cuda.is_available()=False "
            "(поставьте PyTorch с CUDA или задайте RKK_DEVICE=cpu) → CPU"
        )
        return torch.device("cpu")
    try:
        return torch.device(req)
    except Exception:
        print(f"[RKK] Неизвестное RKK_DEVICE={req!r} → CPU")
        return torch.device("cpu")


HUMANOID_TOPOLOGY_WORLDS = frozenset({"humanoid", "humanoid_variant"})

WORLDS = {
    "humanoid": {"label": "Humanoid", "color": "#cc44ff"},
    "humanoid_variant": {"label": "Humanoid (variant)", "color": "#aa66ff"},
    "cartpole": {"label": "Cartpole (stub)", "color": "#44aaff"},
    "grid_nav": {"label": "Grid nav (stub)", "color": "#44ff88"},
    "symbolic_control": {"label": "Symbolic control (stub)", "color": "#ffaa44"},
}


def is_humanoid_topology(world: str | None) -> bool:
    """Same causal variable_ids and role map (Track B0–B1)."""
    return (world or "") in HUMANOID_TOPOLOGY_WORLDS


def _make_env(world: str, device: torch.device):
    if world == "humanoid_variant":
        from engine.environment_humanoid_variant import EnvironmentHumanoidVariant

        return EnvironmentHumanoidVariant(device=device)
    if world == "humanoid":
        from engine.environment_humanoid import EnvironmentHumanoid

        return EnvironmentHumanoid(device=device)
    if world == "grid_nav":
        from engine.environment_grid_nav import EnvironmentGridNav

        return EnvironmentGridNav(device=device)
    if world == "symbolic_control":
        from engine.environment_symbolic import EnvironmentSymbolic

        return EnvironmentSymbolic(device=device)
    if world == "cartpole":
        from engine.environment_cartpole import EnvironmentCartpole

        return EnvironmentCartpole(device=device)
    raise ValueError(
        f"unsupported world {world!r} "
        "(expected humanoid, humanoid_variant, grid_nav, symbolic_control)"
    )


def default_bounds() -> HomeostaticBounds:
    return HomeostaticBounds(
        var_min=0.05,
        var_max=0.95,
        phi_min=0.01,
        h_slow_max=14.0,
        env_entropy_max_delta=0.96,
        warmup_ticks=3000,
        blend_ticks=800,
        phi_min_steady=0.03,
        env_entropy_max_delta_steady=0.85,
        h_slow_max_steady=12.0,
        predict_band_edge_steady=0.015,
    )


class WorldSwitcher:
    def __init__(self, agent: RKKAgent, device: torch.device):
        self.agent = agent
        self.device = device
        self.history: list[dict] = []

    def switch(self, new_world: str) -> dict:
        old = self.agent.env.preset
        if old == new_world:
            return {"switched": False, "world": new_world}

        new_env = _make_env(new_world, self.device)

        old_nodes = set(self.agent.graph.nodes.keys())
        init_obs = new_env.observe()
        new_vars = new_env.variable_ids
        new_nodes = [v for v in new_vars if v not in old_nodes]

        if len(new_vars) != len(self.agent.graph._node_ids) or set(new_vars) != set(
            self.agent.graph._node_ids
        ):
            self.agent.graph.rebind_variables(list(new_vars), init_obs)
        else:
            for var_id in new_vars:
                self.agent.graph.set_node(var_id, init_obs.get(var_id, 0.5))

        self.agent.env = new_env
        self.agent.graph.set_env_preset(str(getattr(new_env, "preset", new_world)))

        from engine.temporal import TemporalBlankets

        new_d = len(new_vars)
        if self.agent.temporal.d_input != new_d:
            self.agent.temporal = TemporalBlankets(d_input=new_d, device=self.device)

        self.agent.temporal.step(init_obs)
        self.agent.graph.record_observation(init_obs)

        rec = {
            "from_world": old,
            "to_world": new_world,
            "new_nodes": new_nodes,
            "total_nodes": len(self.agent.graph.nodes),
            "gnn_d": self.agent.graph._d,
        }
        self.history.append(rec)
        print(f"[WorldSwitch] {old} -> {new_world} | +{len(new_nodes)} nodes | d={self.agent.graph._d}")
        fn = getattr(self, "_on_switch_hook", None)
        if callable(fn):
            fn(new_world)
        return {"switched": True, **rec}
