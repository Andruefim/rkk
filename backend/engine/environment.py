"""
Environment — ground truth causal world.
Gym-совместимый интерфейс для будущей интеграции с RL.
"""
from __future__ import annotations
import torch
import numpy as np
from dataclasses import dataclass

@dataclass
class GTEdge:
    from_:     str
    to:        str
    weight:    float
    noise_std: float
    nonlinear: bool = False


PRESETS: dict[str, tuple[list[dict], list[dict]]] = {
    "physics": (
        [
            {"id": "Temp",        "value": 0.5},
            {"id": "Pressure",    "value": 0.5},
            {"id": "Volume",      "value": 0.5},
            {"id": "Energy",      "value": 0.3},
            {"id": "StateChange", "value": 0.1},
            {"id": "Entropy",     "value": 0.2},
        ],
        [
            {"from_": "Temp",     "to": "Pressure",    "weight":  0.80, "noise_std": 0.04},
            {"from_": "Temp",     "to": "Energy",      "weight":  0.70, "noise_std": 0.05},
            {"from_": "Pressure", "to": "Volume",      "weight": -0.60, "noise_std": 0.06, "nonlinear": True},
            {"from_": "Energy",   "to": "StateChange", "weight":  0.75, "noise_std": 0.05},
            {"from_": "Volume",   "to": "StateChange", "weight":  0.40, "noise_std": 0.08},
            {"from_": "Energy",   "to": "Entropy",     "weight":  0.55, "noise_std": 0.06},
        ]
    ),
    "chemistry": (
        [
            {"id": "Reactant_A", "value": 0.8},
            {"id": "Reactant_B", "value": 0.6},
            {"id": "Catalyst",   "value": 0.3},
            {"id": "Temp",       "value": 0.5},
            {"id": "Rate",       "value": 0.4},
            {"id": "Product",    "value": 0.2},
        ],
        [
            {"from_": "Reactant_A", "to": "Rate",    "weight": 0.65, "noise_std": 0.05},
            {"from_": "Reactant_B", "to": "Rate",    "weight": 0.55, "noise_std": 0.05},
            {"from_": "Catalyst",   "to": "Rate",    "weight": 0.80, "noise_std": 0.03},
            {"from_": "Temp",       "to": "Rate",    "weight": 0.70, "noise_std": 0.06, "nonlinear": True},
            {"from_": "Rate",       "to": "Product", "weight": 0.90, "noise_std": 0.04},
            {"from_": "Temp",       "to": "Product", "weight": 0.20, "noise_std": 0.08},
        ]
    ),
    "logic": (
        [
            {"id": "Input",     "value": 1.0},
            {"id": "Condition", "value": 0.5},
            {"id": "Branch_A",  "value": 0.0},
            {"id": "Branch_B",  "value": 1.0},
            {"id": "Output",    "value": 0.5},
            {"id": "Error",     "value": 0.0},
        ],
        [
            {"from_": "Input",     "to": "Condition", "weight":  0.90, "noise_std": 0.01},
            {"from_": "Condition", "to": "Branch_A",  "weight":  0.95, "noise_std": 0.01},
            {"from_": "Condition", "to": "Branch_B",  "weight": -0.95, "noise_std": 0.01},
            {"from_": "Branch_A",  "to": "Output",    "weight":  0.80, "noise_std": 0.02},
            {"from_": "Branch_B",  "to": "Output",    "weight":  0.60, "noise_std": 0.02},
            {"from_": "Input",     "to": "Error",     "weight":  0.10, "noise_std": 0.05},
        ]
    ),
}


class Environment:
    def __init__(self, preset: str = "physics", device: torch.device | None = None):
        self.preset  = preset
        self.device  = device or torch.device("cpu")
        self.n_interventions = 0

        nodes_cfg, edges_cfg = PRESETS[preset]
        self.variables: dict[str, float] = {n["id"]: n["value"] for n in nodes_cfg}
        self._gt: list[GTEdge] = [GTEdge(**e) for e in edges_cfg]

    # ── Observe ────────────────────────────────────────────────────────────────
    def observe(self) -> dict[str, float]:
        return dict(self.variables)

    def observe_tensor(self) -> torch.Tensor:
        """Возвращает наблюдение как pinned-memory тензор для быстрой GPU-передачи."""
        vals = list(self.variables.values())
        t    = torch.tensor(vals, dtype=torch.float32).pin_memory()
        return t

    # ── do(variable = value) ───────────────────────────────────────────────────
    def intervene(self, variable: str, value: float) -> dict[str, float]:
        """
        Единственная точка 'интервенционной жёсткости'.
        Возвращает ground-truth ответ — нельзя галлюцинировать.
        """
        if variable not in self.variables:
            return self.observe()

        self.n_interventions += 1
        prev    = dict(self.variables)
        new_val = dict(prev)
        new_val[variable] = value

        visited = {variable}
        queue   = [variable]
        while queue:
            cur = queue.pop(0)
            for e in self._gt:
                if e.from_ != cur:
                    continue
                upstream_delta = new_val[cur] - prev[cur]
                effect = e.weight * upstream_delta
                if e.nonlinear:
                    effect = float(np.tanh(effect * 2))
                noise = np.random.normal(0, e.noise_std)
                new_val[e.to] = float(np.clip(
                    new_val.get(e.to, prev[e.to]) + effect + noise, 0.0, 1.0
                ))
                if e.to not in visited:
                    visited.add(e.to)
                    queue.append(e.to)

        self.variables = new_val
        return self.observe()

    # ── Discovery rate ─────────────────────────────────────────────────────────
    def discovery_rate(self, agent_edges: list[dict]) -> float:
        hits = 0
        for gt in self._gt:
            for ae in agent_edges:
                if (ae["from_"] == gt.from_ and ae["to"] == gt.to
                        and abs(ae["weight"] - gt.weight) < 0.30):
                    hits += 1
                    break
        return hits / len(self._gt) if self._gt else 0.0

    @property
    def variable_ids(self) -> list[str]:
        return list(self.variables.keys())

    def gt_edges(self) -> list[dict]:
        return [{"from_": e.from_, "to": e.to, "weight": e.weight} for e in self._gt]
