"""
causal_graph.py — NOTEARS / GNN как nn.Module.

Фаза 1: URDF-цепочки гуманоида — freeze_kinematic_priors (W≈0.85), нуление grad по этим
  позициям в train_step и clamp после optimizer.step (RKK_FREEZE_URDF).

Фаза 10+: GNN заменяет NOTEARS как ядро каузального графа.

Фаза B (Predictive World Model):
  train_step: смесь интервенций (_int_buffer) и пассивных пар подряд из _obs_buffer
    (X_t, a=0)→X_{t+1} — физика/кубы без явного do. Доля пассива: RKK_WM_PASSIVE_MIX (по умолч. 0.35).
  Интервенции: X_pred = forward_dynamics(X_t, a_t), L_rec = MSE(X_pred, X_{t+1}).
  propagate / imagination: то же f(state, action); rollout_step_free — f(X, 0).
  Neural ODE (опционально): RKK_WM_NEURAL_ODE=1 + torchdiffeq — интеграл
  dY/dτ = forward_dynamics(Y,a) − Y по τ (см. engine.wm_neural_ode).

  USE_GNN = True   → CausalGNNCore  (message passing + action_enc)
  USE_GNN = False  → NOTEARSCore    (forward_dynamics ≈ forward(X+a))

  L_total = L_rec + λ_dag*h(W_free) + λ_l1*|W|₁ (на frozen позициях L1 и DAG не давят; см. dag_constraint_masked)
h(W)    = tr(exp(W∘W)) - d
"""
from __future__ import annotations

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass

from engine.environment_humanoid import HUMANOID_KINEMATIC_EDGE_PRIORS
from engine.graph_constants import is_read_only_macro_var
from engine.wm_neural_ode import integrate_world_model_step

# ── Переключение ядра ─────────────────────────────────────────────────────────
USE_GNN = True   # False → NOTEARS (откат к фазам 1-9)

# Фаза 1: пары для freeze; dict { (f,t): {alpha_trust} } — см. environment_humanoid.URDF_FROZEN_EDGES
URDF_FROZEN_EDGE_LIST: list[tuple[str, str]] = list(HUMANOID_KINEMATIC_EDGE_PRIORS)


# ─── Edge ─────────────────────────────────────────────────────────────────────
@dataclass
class Edge:
    from_:  str
    to:     str
    weight: float
    alpha_trust: float
    intervention_count: int = 1

    def as_dict(self):
        return {
            "from_": self.from_, "to": self.to,
            "weight": round(self.weight, 4),
            "alpha_trust": round(self.alpha_trust, 4),
            "intervention_count": self.intervention_count,
        }


# ─── NOTEARSCore (фазы 1-9, fallback) ────────────────────────────────────────
class NOTEARSCore(nn.Module):
    def __init__(self, d: int, device: torch.device):
        super().__init__()
        self.d      = d
        self.device = device
        self.W      = nn.Parameter(torch.zeros(d, d, device=device))
        mask = 1.0 - torch.eye(d, device=device)
        self.register_buffer("mask", mask)

    def W_masked(self) -> torch.Tensor:
        return self.W * self.mask

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return X @ self.W_masked()

    def dag_constraint(self) -> torch.Tensor:
        W2  = self.W_masked() ** 2
        exp = torch.linalg.matrix_exp(W2)
        return exp.trace() - self.d

    def dag_constraint_masked(self, free_mask: torch.Tensor) -> torch.Tensor:
        """
        DAG-штраф только по «обучаемым» рёбрам: (W∘mask_diag)∘free_mask.
        Замороженные URDF-позиции (free_mask=0) не входят в tr(exp(·)).
        """
        Wm = self.W_masked() * free_mask
        W2 = Wm ** 2
        exp = torch.linalg.matrix_exp(W2)
        return exp.trace() - self.d

    def intervention_loss(self, X_obs, X_int, int_var_idx, int_val):
        a = torch.zeros_like(X_obs)
        a[:, int_var_idx] = int_val
        return F.mse_loss(self.forward_dynamics(X_obs, a), X_int)

    def forward_dynamics(self, X: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """
        X' ≈ forward(X_do), где в координатах с ненулевым a подставляется абсолютное
        целевое значение do (как в старом X_do[:, idx] = val); иначе — пассивный шаг.
        """
        m = (torch.abs(a) > 1e-8).float()
        x_in = X * (1.0 - m) + a * m
        return self.forward(x_in)

    def l1_reg(self) -> torch.Tensor:
        return self.W_masked().abs().sum()

    def alpha_trust_matrix(self) -> torch.Tensor:
        W_abs = self.W_masked().abs()
        return W_abs / W_abs.max() if W_abs.max() > 0 else W_abs


# ─── CausalGraph ──────────────────────────────────────────────────────────────
class CausalGraph:
    LAMBDA_DAG  = 0.5
    LAMBDA_INT  = 2.0
    LAMBDA_L1   = 0.02
    EDGE_THRESH = 0.05
    # Фаза 1: целевой вес замороженных рёбер в W (после каждого optim.step снова clamp).
    FROZEN_EDGE_W = 0.85

    def __init__(self, device: torch.device):
        self.device     = device
        self.nodes:     dict[str, float] = {}
        self._node_ids: list[str]        = []
        self._d         = 0
        self._core      = None
        self._optim     = None
        self._obs_buffer: list[list[float]] = []
        self._int_buffer: list[dict]        = []
        self.BUFFER_SIZE = 64
        self._edge_cache: list[Edge] | None = None
        self._mdl_cache:  float | None      = None
        self.train_losses: list[float]      = []
        # (from, to) — не обновляются градиентом WM; W[i,j] фиксируется на FROZEN_EDGE_W.
        self._frozen_edge_set: set[tuple[str, str]] = set()
        # Макро-узлы concept_N: агрегат по members, метаданные детектора.
        self._concept_meta: dict[str, dict] = {}

    def set_node(self, id_: str, value: float = 0.0) -> None:
        self._invalidate_cache()
        if id_ not in self.nodes:
            self._node_ids.append(id_)
            self._d += 1
        self.nodes[id_] = value
        if self._core is None or self._core.d != self._d:
            self._rebuild_core()

    def rebind_variables(self, ordered_ids: list[str], values: dict[str, float]) -> None:
        """
        Полностью заменить набор узлов (например humanoid → только slot_* в visual mode).
        Сбрасывает NOTEARS/GNN буферы, чтобы размеры строк совпадали с новым d.
        """
        self._invalidate_cache()
        self.nodes = {k: float(values.get(k, 0.5)) for k in ordered_ids}
        self._node_ids = list(ordered_ids)
        self._d = len(ordered_ids)
        keep = set(ordered_ids)
        self._concept_meta = {k: v for k, v in self._concept_meta.items() if k in keep}
        self._obs_buffer.clear()
        self._int_buffer.clear()
        self._core = None
        self._optim = None
        self._rebuild_core()

    def apply_env_observation(self, env_obs: dict[str, float]) -> None:
        """Обновить узлы из observe() среды; пересчитать значения concept_* (среднее по members)."""
        for k, v in env_obs.items():
            if k not in self.nodes:
                continue
            try:
                self.nodes[k] = float(v)
            except (TypeError, ValueError):
                pass
        self.refresh_concept_aggregates()

    def refresh_concept_aggregates(self) -> None:
        for cid, meta in self._concept_meta.items():
            if cid not in self.nodes:
                continue
            mems = meta.get("members") or []
            if not mems:
                continue
            vals = [float(self.nodes.get(m, 0.5)) for m in mems]
            self.nodes[cid] = float(np.clip(float(np.mean(vals)), 0.01, 0.99))

    def snapshot_vec_dict(self) -> dict[str, float]:
        return {nid: float(self.nodes.get(nid, 0.5)) for nid in self._node_ids}

    def materialize_concept_macro(
        self,
        node_id: str,
        member_nodes: list[str],
        *,
        detector_id: str = "",
        pattern: list[str] | None = None,
    ) -> bool:
        """
        Вставить макро-узел в GNN: rebind_variables(d+1), слабые рёбра member→concept.
        Узел не участвует в env.intervene(); значение = mean(members) каждый тик.
        """
        if not str(node_id).startswith("concept_"):
            return False
        if node_id in self.nodes:
            return False
        base = [n for n in self._node_ids if not str(n).startswith("concept_")]
        mems = [m for m in member_nodes if m]
        if not mems or any(m not in base for m in mems):
            return False
        new_ids = base + [node_id]
        vals: dict[str, float] = {k: float(self.nodes.get(k, 0.5)) for k in base}
        mv = [float(self.nodes.get(m, 0.5)) for m in mems]
        vals[node_id] = float(np.clip(float(np.mean(mv)), 0.05, 0.95))
        self.rebind_variables(new_ids, vals)
        try:
            w_m = float(os.environ.get("RKK_CONCEPT_MACRO_EDGE_W", "0.18"))
        except ValueError:
            w_m = 0.18
        try:
            a_m = float(os.environ.get("RKK_CONCEPT_MACRO_EDGE_ALPHA", "0.08"))
        except ValueError:
            a_m = 0.08
        for m in mems:
            self.set_edge(m, node_id, w_m, a_m)
        if mems:
            self.set_edge(node_id, mems[-1], 0.06, 0.05)
        self._concept_meta[node_id] = {
            "members": list(mems),
            "pattern": list(pattern or []),
            "detector_id": detector_id,
        }
        self._sync_frozen_W_into_core()
        return True

    def freeze_kinematic_priors(
        self, frozen: list[tuple[str, str]] | None = None
    ) -> None:
        """Биомеханические рёбра: высокий вес в W, градиент по ним нулится, после step — clamp."""
        pairs = list(frozen) if frozen is not None else list(HUMANOID_KINEMATIC_EDGE_PRIORS)
        self._frozen_edge_set = {(a, b) for a, b in pairs}
        self._sync_frozen_W_into_core()

    def _sync_frozen_W_into_core(self) -> None:
        if self._core is None or not self._frozen_edge_set:
            return
        wval = float(self.FROZEN_EDGE_W)
        with torch.no_grad():
            W = self._core.W
            w = W.detach().clone()
            for f, t in self._frozen_edge_set:
                if f in self._node_ids and t in self._node_ids:
                    i, j = self._node_ids.index(f), self._node_ids.index(t)
                    w[i, j] = wval
            W.copy_(w)
        self._invalidate_cache()

    def _zero_grad_frozen_W(self) -> None:
        if not self._frozen_edge_set or self._core is None:
            return
        W = self._core.W
        if W.grad is None:
            return
        for f, t in self._frozen_edge_set:
            if f in self._node_ids and t in self._node_ids:
                i, j = self._node_ids.index(f), self._node_ids.index(t)
                W.grad[i, j] = 0.0

    def _clamp_frozen_W_after_step(self) -> None:
        self._sync_frozen_W_into_core()

    def _maybe_compile_gnn_core(self) -> None:
        if not USE_GNN or self._core is None:
            return
        v = os.environ.get("RKK_GNN_COMPILE", "0").strip().lower()
        if v not in ("1", "true", "yes", "on"):
            return
        if self.device.type not in ("cuda", "mps"):
            return
        if not hasattr(torch, "compile"):
            return
        try:
            self._core = torch.compile(self._core, mode="reduce-overhead")
            print(f"[CausalGraph] GNN torch.compile (d={self._d}, device={self.device.type})")
        except Exception as e:
            print(f"[CausalGraph] torch.compile skipped: {e}")

    def _rebuild_core(self):
        if USE_GNN:
            from engine.causal_gnn import CausalGNNCore
            old_W = None
            if self._core is not None:
                # GNN resize: мигрируем через resize_to если можем
                if hasattr(self._core, 'resize_to'):
                    new_core  = self._core.resize_to(self._d)
                    self._core = new_core
                    self._maybe_compile_gnn_core()
                    self._optim = torch.optim.Adam(self._core.parameters(), lr=5e-3)
                    self._invalidate_cache()
                    self._sync_frozen_W_into_core()
                    return
                old_W = self._core.W_masked().detach().clone()
            self._core = CausalGNNCore(self._d, self.device)
        else:
            old_W = None
            if self._core is not None and self._core.d < self._d:
                old_W = self._core.W_masked().detach().clone()
            self._core = NOTEARSCore(self._d, self.device)

        if old_W is not None:
            old_d = old_W.shape[0]
            with torch.no_grad():
                W = self._core.W
                w = W.detach().clone()
                w[:old_d, :old_d] = old_W.to(w.device, dtype=w.dtype)
                W.copy_(w)

        if USE_GNN:
            self._maybe_compile_gnn_core()

        self._optim = torch.optim.Adam(self._core.parameters(), lr=5e-3)
        self._invalidate_cache()
        self._sync_frozen_W_into_core()

    def record_observation(self, obs: dict[str, float]) -> None:
        if not self._node_ids:
            return
        vec = [obs.get(nid, 0.0) for nid in self._node_ids]
        self._obs_buffer.append(vec)
        if len(self._obs_buffer) > self.BUFFER_SIZE * 4:
            self._obs_buffer = self._obs_buffer[-self.BUFFER_SIZE * 2:]

    def record_intervention(self, var_name: str, val: float,
                            obs_before: dict, obs_after: dict) -> None:
        if is_read_only_macro_var(var_name):
            return
        if var_name not in self._node_ids:
            return
        self._int_buffer.append({
            "idx":        self._node_ids.index(var_name),
            "val":        val,
            "obs_before": [obs_before.get(n, 0.0) for n in self._node_ids],
            "obs_after":  [obs_after.get(n, 0.0)  for n in self._node_ids],
        })
        if len(self._int_buffer) > self.BUFFER_SIZE:
            self._int_buffer = self._int_buffer[-self.BUFFER_SIZE:]

    def train_step(self) -> dict[str, float] | None:
        """
        World model: батч = интервенции + пассивные переходы из подряд идущих наблюдений.
        Пассив: a=0, L_rec штрафует f(X_t,0) vs X_{t+1} (кубы/физика между тиками).
        """
        if self._core is None:
            return None

        int_b = self._int_buffer
        obs_b = self._obs_buffer
        batch_cap = 32
        try:
            passive_ratio = float(os.environ.get("RKK_WM_PASSIVE_MIX", "0.35"))
        except ValueError:
            passive_ratio = 0.35
        passive_ratio = max(0.0, min(0.75, passive_ratio))

        L = len(obs_b)
        max_pairs = max(0, L - 1)
        n_p = min(int(round(batch_cap * passive_ratio)), max_pairs)
        n_i = min(batch_cap - n_p, len(int_b))

        if n_p + n_i < 4:
            n_i = min(batch_cap, len(int_b))
            n_p = min(max_pairs, batch_cap - n_i)
        if n_p + n_i < 4:
            return None

        rows_X: list[list[float]] = []
        rows_Y: list[list[float]] = []
        rows_a: list[list[float]] = []

        for item in int_b[-n_i:] if n_i > 0 else []:
            rows_X.append(item["obs_before"])
            rows_Y.append(item["obs_after"])
            arow = [0.0] * self._d
            arow[item["idx"]] = item["val"]
            rows_a.append(arow)

        if n_p > 0:
            start_t = L - n_p - 1
            for k in range(n_p):
                rows_X.append(obs_b[start_t + k])
                rows_Y.append(obs_b[start_t + k + 1])
                rows_a.append([0.0] * self._d)

        X_t = torch.tensor(rows_X, dtype=torch.float32, device=self.device)
        X_tp1 = torch.tensor(rows_Y, dtype=torch.float32, device=self.device)
        a_t = torch.tensor(rows_a, dtype=torch.float32, device=self.device)

        self._optim.zero_grad()

        X_pred = integrate_world_model_step(self._core, X_t, a_t)
        l_rec = F.mse_loss(X_pred, X_tp1)

        dag_mask_frozen = os.environ.get("RKK_DAG_MASK_FROZEN", "1").strip().lower() not in (
            "0", "false", "no", "off",
        )
        dag_free = torch.ones(self._d, self._d, device=self.device, dtype=torch.float32)
        for f, t in self._frozen_edge_set:
            if f in self._node_ids and t in self._node_ids:
                i, j = self._node_ids.index(f), self._node_ids.index(t)
                dag_free[i, j] = 0.0
        if (
            dag_mask_frozen
            and self._frozen_edge_set
            and hasattr(self._core, "dag_constraint_masked")
        ):
            h_W = self._core.dag_constraint_masked(dag_free)
        else:
            h_W = self._core.dag_constraint()
        l_dag = self.LAMBDA_DAG * h_W.abs()
        wm = self._core.W_masked()
        l1_mask = torch.ones_like(wm)
        for f, t in self._frozen_edge_set:
            if f in self._node_ids and t in self._node_ids:
                i, j = self._node_ids.index(f), self._node_ids.index(t)
                l1_mask[i, j] = 0.0
        l_l1 = self.LAMBDA_L1 * (wm.abs() * l1_mask).sum()

        l_int = torch.tensor(0.0, device=self.device)

        loss = l_rec + l_dag + l_l1 + l_int
        loss.backward()
        self._zero_grad_frozen_W()
        torch.nn.utils.clip_grad_norm_(self._core.parameters(), max_norm=1.0)
        self._optim.step()
        self._clamp_frozen_W_after_step()

        self._invalidate_cache()
        self.train_losses.append(loss.item())
        if len(self.train_losses) > 100:
            self.train_losses.pop(0)

        return {
            "loss":  round(loss.item(), 5),
            "l_rec": round(l_rec.item(), 5),
            "l_dag": round(l_dag.item(), 5),
            "l_int": round(l_int.item(), 5),
            "l_l1":  round(l_l1.item(), 5),
            "h_W":   round(h_W.item(), 5),
            "batch_int": n_i,
            "batch_passive": n_p,
        }

    def set_edge(self, from_: str, to: str, weight: float, alpha: float) -> None:
        if self._core is None or from_ not in self._node_ids or to not in self._node_ids:
            return
        i, j = self._node_ids.index(from_), self._node_ids.index(to)
        with torch.no_grad():
            W = self._core.W
            w = W.detach().clone()
            old_w = float(w[i, j].item())
            w[i, j] = 0.7 * old_w + 0.3 * float(weight)
            W.copy_(w)
        self._invalidate_cache()

    def remove_edge(self, from_: str, to: str) -> None:
        if self._core is None or from_ not in self._node_ids or to not in self._node_ids:
            return
        i, j = self._node_ids.index(from_), self._node_ids.index(to)
        with torch.no_grad():
            W = self._core.W
            w = W.detach().clone()
            w[i, j] = 0.0
            W.copy_(w)
        self._invalidate_cache()

    @property
    def edges(self) -> list[Edge]:
        if self._edge_cache is not None:
            return self._edge_cache
        if self._core is None:
            return []
        W     = self._core.W_masked().detach()
        alpha = self._core.alpha_trust_matrix().detach()
        result = []
        for i, from_ in enumerate(self._node_ids):
            for j, to in enumerate(self._node_ids):
                w = W[i, j].item()
                if abs(w) >= self.EDGE_THRESH:
                    a_tr = 1.0 if (from_, to) in self._frozen_edge_set else alpha[i, j].item()
                    result.append(Edge(
                        from_=from_, to=to,
                        weight=round(w, 4),
                        alpha_trust=round(float(a_tr), 4),
                        intervention_count=1,
                    ))
        self._edge_cache = result
        return result

    def edge_uncertainty(self, from_: str, to: str) -> float:
        if self._core is None or from_ not in self._node_ids or to not in self._node_ids:
            return 1.0
        i, j  = self._node_ids.index(from_), self._node_ids.index(to)
        if (from_, to) in self._frozen_edge_set:
            return 0.0
        alpha = self._core.alpha_trust_matrix()[i, j].item()
        return 1.0 - alpha

    @property
    def alpha_mean(self) -> float:
        if self._core is None:
            return 0.05
        alpha = self._core.alpha_trust_matrix()
        mask  = alpha > 0.01
        return float(alpha[mask].mean().item()) if mask.sum() > 0 else 0.05

    @property
    def mdl_size(self) -> float:
        if self._mdl_cache is not None:
            return self._mdl_cache
        if self._core is None:
            return 0.0
        W        = self._core.W_masked().detach()
        alpha    = self._core.alpha_trust_matrix().detach()
        alpha_u  = alpha.clone()
        for f, t in self._frozen_edge_set:
            if f in self._node_ids and t in self._node_ids:
                i, j = self._node_ids.index(f), self._node_ids.index(t)
                alpha_u[i, j] = 1.0
        sig_mask = W.abs() >= self.EDGE_THRESH
        if sig_mask.sum() == 0:
            return 0.0
        mdl = (1 + (1 - alpha_u[sig_mask])).sum().item()
        self._mdl_cache = mdl
        return mdl

    def propagate(self, variable: str, value: float) -> dict[str, float]:
        return self.propagate_from(self.nodes, variable, value)

    def path_min_alpha_trust_on_path(self, path_nodes: list[str]) -> float:
        """Минимальный α_trust по ориентированным рёбрам цепочки (для условия macro-concept)."""
        if self._core is None or len(path_nodes) < 2:
            return 0.0
        alpha = self._core.alpha_trust_matrix()
        m = 1.0
        for i in range(len(path_nodes) - 1):
            a, b = path_nodes[i], path_nodes[i + 1]
            if a not in self._node_ids or b not in self._node_ids:
                return 0.0
            ii, jj = self._node_ids.index(a), self._node_ids.index(b)
            av = 1.0 if (a, b) in self._frozen_edge_set else float(alpha[ii, jj].item())
            m = min(m, av)
        return m

    def propagate_from(
        self, base: dict[str, float], variable: str, value: float
    ) -> dict[str, float]:
        """
        Предсказание после do(variable=value) от произвольного снимка узлов
        (Фаза 13: imagination rollout без мутации self.nodes).
        """
        if is_read_only_macro_var(variable):
            return dict(base)
        if self._core is None:
            return dict(base)
        state_vec = torch.tensor(
            [[float(base.get(n, 0.0)) for n in self._node_ids]],
            dtype=torch.float32, device=self.device,
        )
        a_vec = torch.zeros(1, self._d, dtype=torch.float32, device=self.device)
        if variable in self._node_ids:
            a_vec[0, self._node_ids.index(variable)] = float(value)
        with torch.no_grad():
            pred = integrate_world_model_step(self._core, state_vec, a_vec)
        result = {nid: float(pred[0, i].item()) for i, nid in enumerate(self._node_ids)}
        return result

    def rollout_step_free(self, base: dict[str, float]) -> dict[str, float]:
        """
        Один шаг «свободной» динамики GNN: X' = core(X), без интервенции.
        Используется для многошагового imagination после мысленного do().
        """
        if self._core is None:
            return dict(base)
        state_vec = torch.tensor(
            [[float(base.get(n, 0.0)) for n in self._node_ids]],
            dtype=torch.float32, device=self.device,
        )
        z = torch.zeros_like(state_vec)
        with torch.no_grad():
            pred = integrate_world_model_step(self._core, state_vec, z)
        return {nid: float(pred[0, i].item()) for i, nid in enumerate(self._node_ids)}

    def _invalidate_cache(self):
        self._edge_cache = None
        self._mdl_cache  = None

    def to_dict(self) -> dict:
        W_data = h_data = core_type = None
        if self._core is not None:
            W_data    = self._core.W_masked().detach().cpu().tolist()
            h_data    = round(self._core.dag_constraint().item(), 5)
            core_type = "gnn" if USE_GNN else "notears"
        return {
            "nodes":     self.nodes,
            "edges":     [e.as_dict() for e in self.edges],
            "mdl":       self.mdl_size,
            "W":         W_data,
            "h_W":       h_data,
            "d":         self._d,
            "core_type": core_type,
        }

    def clone(self) -> "CausalGraph":
        g = CausalGraph(self.device)
        g.nodes      = dict(self.nodes)
        g._node_ids  = list(self._node_ids)
        g._d         = self._d
        g._frozen_edge_set = set(self._frozen_edge_set)
        g._concept_meta = {k: dict(v) for k, v in self._concept_meta.items()}
        if self._core is not None:
            g._rebuild_core()
            with torch.no_grad():
                g._core.W.copy_(self._core.W)
        return g