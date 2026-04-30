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

Фаза N (LeWM integration):
  Parallel sequence training (teacher forcing по всей траектории):
    - _seq_buffer хранит последовательности длиной T (RKK_WM_SEQ_LEN, default 16)
    - forward_dynamics_seq: один forward pass для B×T тиков вместо цикла
    - SIGReg anti-collapse: isotropic Gaussian regularizer на латентных embeddings
    - Автоматический fallback к legacy single-transition training пока нет sequences
  Config: RKK_WM_SEQ_LEN, RKK_WM_SIGREG_LAMBDA, RKK_WM_SIGREG_PROJ

  L_total = L_rec + λ_sig*SIGReg(Z) + λ_dag*h(W_free) + λ_l1*|W|₁
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
        """
        Оптимизированный DAG constraint через Taylor expansion (до 4-го порядка).
        Снижает вычислительную нагрузку по сравнению с полным matrix_exp.
        """
        M = self.W_masked() ** 2
        
        # Вычисляем степени матрицы для следа
        M2 = torch.matmul(M, M)
        M3 = torch.matmul(M2, M)
        M4 = torch.matmul(M3, M)
        
        # h(W) ≈ tr(M^2)/2! + tr(M^3)/3! + tr(M^4)/4!
        # tr(I) и tr(M) опущены, так как они сокращаются с -d и равны 0 соответственно.
        trace_sum = (M2.trace() / 2.0) + (M3.trace() / 6.0) + (M4.trace() / 24.0)
        return trace_sum

    def dag_constraint_masked(self, free_mask: torch.Tensor) -> torch.Tensor:
        """
        Фаза 1: Оптимизированный DAG-штраф с учетом маски свободных параметров.
        Использует разложение Тейлора для ускорения расчета.
        """
        # Применяем маску к весам перед возведением в квадрат
        Wm = self.W_masked() * free_mask
        M = Wm ** 2
        
        M2 = torch.matmul(M, M)
        M3 = torch.matmul(M2, M)
        M4 = torch.matmul(M3, M)
        
        trace_sum = (M2.trace() / 2.0) + (M3.trace() / 6.0) + (M4.trace() / 24.0)
        return trace_sum

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
    LAMBDA_DAG  = 0.3
    LAMBDA_INT  = 6.0
    LAMBDA_L1   = 0.001
    EDGE_THRESH = 0.03
    # Фаза 1: целевой вес замороженных рёбер в W (после каждого optim.step снова clamp).
    FROZEN_EDGE_W = 0.85

    def __init__(self, device: torch.device):
        self.device     = device
        self.nodes:     dict[str, float] = {}
        self._node_ids: list[str]        = []
        self._d         = 0
        self.MAX_D      = int(os.environ.get("RKK_GNN_MAX_D", "256"))
        self._core      = None
        self._optim     = None
        self._obs_buffer: list[list[float]] = []
        self._int_buffer: list[dict]        = []
        self.BUFFER_SIZE = 128
        self._edge_cache: list[Edge] | None = None
        self._mdl_cache:  float | None      = None
        self.train_losses: list[float]      = []
        self._frozen_edge_set: set[tuple[str, str]] = set()
        # Фаза 3: запрещённые рёбра (например world -> intent) W[i,j] = 0
        self._forbidden_edge_set: set[tuple[str, str]] = set()
        # Макро-узлы concept_N: агрегат по members, метаданные детектора.
        self._concept_meta: dict[str, dict] = {}
        # LeWM: SIGReg anti-collapse regularizer (lazy init after core is built)
        self._sigreg = None
        # LeWM: sequence buffer — consecutive observation runs for parallel training
        self._seq_obs_run: list[list[float]] = []  # current contiguous run
        self._seq_buffer: list[list[list[float]]] = []  # completed sequences
        try:
            self._seq_len = int(os.environ.get("RKK_WM_SEQ_LEN", "16"))
        except ValueError:
            self._seq_len = 16
        self._seq_len = max(4, min(64, self._seq_len))

    def set_node(self, id_: str, value: float = 0.0) -> None:
        self._invalidate_cache()
        if id_ not in self.nodes:
            self._node_ids.append(id_)
            self._d += 1
        self.nodes[id_] = value
        if self._core is None or self._core.d != self._d:
            self._rebuild_core()

    def rebind_variables(
        self,
        ordered_ids: list[str],
        values: dict[str, float],
        *,
        preserve_state: bool = False,
    ) -> None:
        """
        Полностью заменить набор узлов (например humanoid → только slot_* в visual mode).
        По умолчанию сбрасывает NOTEARS/GNN буферы, чтобы размеры строк совпадали с новым d.
        preserve_state=True: переносит пересекающуюся часть весов и ремапит буферы obs/int.
        """
        old_ids = list(self._node_ids)
        old_obs = [list(row) for row in self._obs_buffer] if preserve_state else []
        old_int = [dict(item) for item in self._int_buffer] if preserve_state else []
        old_core = self._core if preserve_state else None

        self._invalidate_cache()
        self.nodes = {k: float(values.get(k, 0.5)) for k in ordered_ids}
        self._node_ids = list(ordered_ids)
        self._d = len(ordered_ids)
        keep = set(ordered_ids)
        self._concept_meta = {k: v for k, v in self._concept_meta.items() if k in keep}
        if preserve_state:
            old_pos = {nid: i for i, nid in enumerate(old_ids)}
            # Переносим пассивные буферы в новый порядок координат.
            remapped_obs: list[list[float]] = []
            for row in old_obs:
                if not isinstance(row, list):
                    continue
                new_row = []
                for nid in self._node_ids:
                    if nid in old_pos and old_pos[nid] < len(row):
                        new_row.append(float(row[old_pos[nid]]))
                    else:
                        new_row.append(float(self.nodes.get(nid, 0.5)))
                remapped_obs.append(new_row)
            self._obs_buffer = remapped_obs[-self.BUFFER_SIZE * 2 :]

            remapped_int: list[dict] = []
            for item in old_int:
                idx = item.get("idx")
                if not isinstance(idx, int) or idx < 0 or idx >= len(old_ids):
                    continue
                var_name = old_ids[idx]
                if var_name not in self._node_ids:
                    continue

                obs_before_raw = item.get("obs_before")
                obs_after_raw = item.get("obs_after")
                if not isinstance(obs_before_raw, list) or not isinstance(obs_after_raw, list):
                    continue

                new_before = []
                new_after = []
                for nid in self._node_ids:
                    if nid in old_pos and old_pos[nid] < len(obs_before_raw):
                        new_before.append(float(obs_before_raw[old_pos[nid]]))
                    else:
                        new_before.append(float(self.nodes.get(nid, 0.5)))
                    if nid in old_pos and old_pos[nid] < len(obs_after_raw):
                        new_after.append(float(obs_after_raw[old_pos[nid]]))
                    else:
                        new_after.append(float(self.nodes.get(nid, 0.5)))

                remapped_int.append({
                    "idx": self._node_ids.index(var_name),
                    "val": float(item.get("val", 0.0)),
                    "obs_before": new_before,
                    "obs_after": new_after,
                })
            self._int_buffer = remapped_int[-self.BUFFER_SIZE :]
        else:
            self._obs_buffer.clear()
            self._int_buffer.clear()

        self._core = old_core if preserve_state else None
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
        base = list(self._node_ids)
        mems = [m for m in member_nodes if m]
        if not mems or any(m not in base for m in mems):
            return False
        new_ids = base + [node_id]
        vals: dict[str, float] = {k: float(self.nodes.get(k, 0.5)) for k in base}
        mv = [float(self.nodes.get(m, 0.5)) for m in mems]
        vals[node_id] = float(np.clip(float(np.mean(mv)), 0.05, 0.95))
        self.rebind_variables(new_ids, vals, preserve_state=True)
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

    def freeze_forbidden_priors(self, forbidden: list[tuple[str, str]]) -> None:
        """Топологическое Я: запрещенные рёбра (вес = 0, градиент = 0)."""
        self._forbidden_edge_set = {(a, b) for a, b in forbidden}
        self._sync_frozen_W_into_core()

    def _sync_frozen_W_into_core(self) -> None:
        if self._core is None:
            return
        if not self._frozen_edge_set and not self._forbidden_edge_set:
            return
        wval = float(self.FROZEN_EDGE_W)
        with torch.no_grad():
            W = self._core.W
            w = W.detach().clone()
            for f, t in self._frozen_edge_set:
                if f in self._node_ids and t in self._node_ids:
                    i, j = self._node_ids.index(f), self._node_ids.index(t)
                    w[i, j] = wval
            for f, t in self._forbidden_edge_set:
                if f in self._node_ids and t in self._node_ids:
                    i, j = self._node_ids.index(f), self._node_ids.index(t)
                    w[i, j] = 0.0
            W.copy_(w)
        self._invalidate_cache()

    def _zero_grad_frozen_W(self) -> None:
        if self._core is None:
            return
        W = self._core.W
        if W.grad is None:
            return
            
        if self._d < self.MAX_D:
            W.grad[self._d:, :] = 0.0
            W.grad[:, self._d:] = 0.0

        if not self._frozen_edge_set and not self._forbidden_edge_set:
            return
            
        for f, t in self._frozen_edge_set:
            if f in self._node_ids and t in self._node_ids:
                i, j = self._node_ids.index(f), self._node_ids.index(t)
                W.grad[i, j] = 0.0
        for f, t in self._forbidden_edge_set:
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
            # dynamic=True: d меняется (neurogenesis, visual rebind) — без этого inductor
            # может зафиксировать старую размерность и падать «expected sequence length …».
            try:
                self._core = torch.compile(
                    self._core, mode="reduce-overhead", dynamic=True
                )
            except TypeError:
                self._core = torch.compile(self._core, mode="reduce-overhead")
            print(f"[CausalGraph] GNN torch.compile (d={self._d}, device={self.device.type})")
        except Exception as e:
            print(f"[CausalGraph] torch.compile skipped: {e}")

    def _unwrap_gnn_core(self):
        """Снять обёртку torch.compile — resize_to только у «сырого» CausalGNNCore."""
        c = self._core
        if c is None:
            return None
        om = getattr(c, "_orig_mod", None)
        if om is not None:
            return om
        return c

    def _coerce_row_to_d(self, row: list[float] | list) -> list[float]:
        """Строки буферов после роста d могут быть короче/длиннее текущего _node_ids."""
        d = self._d
        r = [float(x) for x in row]
        if len(r) == d:
            return r
        if len(r) < d:
            tail = [
                float(self.nodes.get(self._node_ids[i], 0.5))
                for i in range(len(r), d)
            ]
            return r + tail
        return r[:d]

    def _rebuild_core(self):
        if self._core is not None:
            self._invalidate_cache()
            self._sync_frozen_W_into_core()
            return

        if USE_GNN:
            from engine.causal_gnn import CausalGNNCore
            self._core = CausalGNNCore(self.MAX_D, self.device)
            self._maybe_compile_gnn_core()
        else:
            self._core = NOTEARSCore(self.MAX_D, self.device)

        self._optim = torch.optim.Adam(self._core.parameters(), lr=5e-3)
        self._invalidate_cache()
        self._sync_frozen_W_into_core()

    def _pad(self, x: torch.Tensor) -> torch.Tensor:
        pad_len = self.MAX_D - x.shape[-1]
        if pad_len > 0:
            return torch.nn.functional.pad(x, (0, pad_len))
        return x

    def forward_dynamics(self, X: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        if self._core is None:
            return X
        X_pad = self._pad(X)
        a_pad = self._pad(a)
        if hasattr(self._core, "forward_dynamics"):
            pred = self._core.forward_dynamics(X_pad, a_pad)
        else:
            m = (torch.abs(a_pad) > 1e-8).float()
            x_in = X_pad * (1.0 - m) + a_pad * m
            pred = self._core(x_in)
        return pred[..., :self._d]

    def forward_dynamics_seq(self, X_seq: torch.Tensor, A_seq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self._core is None:
            return X_seq, torch.zeros(0, device=self.device)
        X_pad = self._pad(X_seq)
        A_pad = self._pad(A_seq)
        if hasattr(self._core, "forward_dynamics_seq"):
            pred, z = self._core.forward_dynamics_seq(X_pad, A_pad)
        else:
            pred = self._core(X_pad + A_pad)
            z = torch.zeros(X_seq.shape[0] * X_seq.shape[1], 1, device=self.device)
        return pred[..., :self._d], z

    def record_observation(self, obs: dict[str, float]) -> None:
        if not self._node_ids:
            return
        vec = [obs.get(nid, 0.0) for nid in self._node_ids]
        self._obs_buffer.append(vec)
        if len(self._obs_buffer) > self.BUFFER_SIZE * 4:
            self._obs_buffer = self._obs_buffer[-self.BUFFER_SIZE * 2:]

        # LeWM: accumulate contiguous sequences for parallel training
        coerced = self._coerce_row_to_d(vec)
        self._seq_obs_run.append(coerced)
        T = self._seq_len
        if len(self._seq_obs_run) >= T:
            # slice last T observations as one complete sequence
            seq = self._seq_obs_run[-T:]
            self._seq_buffer.append(seq)
            # keep overlap of T//2 for next sequence
            self._seq_obs_run = self._seq_obs_run[-(T // 2):]
            # cap buffer
            max_seqs = max(8, self.BUFFER_SIZE // T)
            if len(self._seq_buffer) > max_seqs * 2:
                self._seq_buffer = self._seq_buffer[-max_seqs:]

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
        World model training step.

        LeWM mode (when sequences available):
          - Sample B sequences of length T from _seq_buffer
          - Parallel forward: forward_dynamics_seq(X_seq, A_seq) → all timesteps at once
          - Teacher forcing: X_pred[t] → X_target[t+1], for all t in parallel
          - SIGReg: anti-collapse on flattened latent embeddings (B*T, D)
          - Loss = L_rec + λ_dag * h(W) + λ_l1 * |W|₁ + λ_sig * SIGReg(Z)

        Legacy mode (fallback when no sequences yet):
          - Batch = interventions + passive single-transitions (original behavior)
        """
        if self._core is None:
            return None

        # ── Try LeWM parallel sequence training first ─────────────────────────
        has_seq = (
            USE_GNN
            and len(self._seq_buffer) >= 2
            and hasattr(self._core, "forward_dynamics_seq")
        )
        if has_seq:
            return self._train_step_seq()

        # ── Legacy single-transition training (fallback) ──────────────────────
        return self._train_step_legacy()

    def _ensure_sigreg(self) -> None:
        """Lazy-init SIGReg on first use (needs device from core)."""
        if self._sigreg is not None:
            return
        from engine.causal_gnn import SIGReg
        try:
            n_proj = int(os.environ.get("RKK_WM_SIGREG_PROJ", "256"))
        except ValueError:
            n_proj = 256
        n_proj = max(32, min(2048, n_proj))
        self._sigreg = SIGReg(knots=17, num_proj=n_proj).to(self.device)

    def _compute_dag_and_l1(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Shared DAG constraint + L1 computation for both training modes."""
        dag_mask_frozen = os.environ.get("RKK_DAG_MASK_FROZEN", "1").strip().lower() not in (
            "0", "false", "no", "off",
        )
        dag_free = torch.ones(self.MAX_D, self.MAX_D, device=self.device, dtype=torch.float32)
        if self._d < self.MAX_D:
            dag_free[self._d:, :] = 0.0
            dag_free[:, self._d:] = 0.0
            
        for f, t in self._frozen_edge_set | self._forbidden_edge_set:
            if f in self._node_ids and t in self._node_ids:
                i, j = self._node_ids.index(f), self._node_ids.index(t)
                dag_free[i, j] = 0.0
                
        if hasattr(self._core, "dag_constraint_masked"):
            h_W = self._core.dag_constraint_masked(dag_free)
        else:
            h_W = self._core.dag_constraint()
            
        l_dag = self.LAMBDA_DAG * h_W.abs()

        wm = self._core.W_masked()
        l1_mask = torch.ones_like(wm)
        if self._d < self.MAX_D:
            l1_mask[self._d:, :] = 0.0
            l1_mask[:, self._d:] = 0.0
            
        for f, t in self._frozen_edge_set | self._forbidden_edge_set:
            if f in self._node_ids and t in self._node_ids:
                i, j = self._node_ids.index(f), self._node_ids.index(t)
                l1_mask[i, j] = 0.0
        l_l1 = self.LAMBDA_L1 * (wm.abs() * l1_mask).sum()

        return h_W, l_dag, l_l1

    def _finish_train_step(self, loss: torch.Tensor) -> None:
        """Shared backward + optimizer step for both training modes."""
        loss.backward()
        self._zero_grad_frozen_W()
        torch.nn.utils.clip_grad_norm_(self._core.parameters(), max_norm=1.0)
        self._optim.step()
        self._clamp_frozen_W_after_step()
        self._invalidate_cache()
        self.train_losses.append(loss.item())
        if len(self.train_losses) > 100:
            self.train_losses.pop(0)

    def _train_step_seq(self) -> dict[str, float] | None:
        """
        LeWM-style parallel sequence training.
        Teacher forcing по всей траектории → все шаги независимы → один forward.
        """
        import random

        self._ensure_sigreg()

        # Read SIGReg weight from env
        try:
            lambda_sig = float(os.environ.get("RKK_WM_SIGREG_LAMBDA", "0.1"))
        except ValueError:
            lambda_sig = 0.1

        # Sample batch of sequences
        buf = self._seq_buffer
        batch_size = min(4, len(buf))
        indices = random.sample(range(len(buf)), batch_size)
        seqs = [buf[i] for i in indices]

        T = len(seqs[0])
        B = batch_size
        d = self._d

        # Coerce all sequences to current dimension _d before tensor creation
        # (fixes crashes when variable discovery adds new nodes dynamically)
        coerced_seqs = []
        for seq in seqs:
            coerced_seqs.append([self._coerce_row_to_d(row) for row in seq])

        # Build tensors: (B, T, d)
        X_seq = torch.tensor(coerced_seqs, dtype=torch.float32, device=self.device)  # (B, T, d)
        # BUGFIX: WM needs to know what the agent did. Inject motor intent variables as actions.
        A_seq = torch.zeros(B, T, d, dtype=torch.float32, device=self.device)
        for i, nid in enumerate(self._node_ids):
            if nid.startswith("intent_"):
                A_seq[:, :, i] = X_seq[:, :, i]

        self._optim.zero_grad()

        # ── Parallel forward (LeWM core) ──────────────────────────────────────
        X_pred, Z_flat = self.forward_dynamics_seq(X_seq, A_seq)
        # X_pred: (B, T-1, d) — predicted next states
        # Z_flat: (B*T, d*hidden) — latent embeddings for SIGReg

        # Teacher forcing target: actual observations at t+1
        X_target = X_seq[:, 1:]  # (B, T-1, d)

        # ── Losses ────────────────────────────────────────────────────────────
        l_rec = F.mse_loss(X_pred, X_target)

        # SIGReg anti-collapse
        l_sig = lambda_sig * self._sigreg(Z_flat)

        # DAG + L1 (shared)
        h_W, l_dag, l_l1 = self._compute_dag_and_l1()

        loss = l_rec + l_sig + l_dag + l_l1

        self._finish_train_step(loss)

        return {
            "loss":    round(loss.item(), 5),
            "l_rec":   round(l_rec.item(), 5),
            "l_sig":   round(l_sig.item(), 5),
            "l_dag":   round(l_dag.item(), 5),
            "l_l1":    round(l_l1.item(), 5),
            "h_W":     round(h_W.item(), 5),
            "mode":    "seq",
            "batch_B": B,
            "seq_T":   T,
        }

    def _train_step_legacy(self) -> dict[str, float] | None:
        """
        Legacy single-transition training (original behavior).
        Used as fallback when sequences haven't accumulated yet.
        """
        int_b = self._int_buffer
        obs_b = self._obs_buffer
        batch_cap = 32
        try:
            passive_ratio = float(os.environ.get("RKK_WM_PASSIVE_MIX", "0.15"))
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
            rows_X.append(self._coerce_row_to_d(item["obs_before"]))
            rows_Y.append(self._coerce_row_to_d(item["obs_after"]))
            arow = [0.0] * self._d
            idx = int(item["idx"])
            if 0 <= idx < self._d:
                arow[idx] = float(item["val"])
            rows_a.append(arow)

        if n_p > 0:
            start_t = L - n_p - 1
            for k in range(n_p):
                rows_X.append(self._coerce_row_to_d(obs_b[start_t + k]))
                rows_Y.append(self._coerce_row_to_d(obs_b[start_t + k + 1]))
                rows_a.append([0.0] * self._d)

        X_t = torch.tensor(rows_X, dtype=torch.float32, device=self.device)
        X_tp1 = torch.tensor(rows_Y, dtype=torch.float32, device=self.device)
        a_t = torch.tensor(rows_a, dtype=torch.float32, device=self.device)

        self._optim.zero_grad()

        X_pred = integrate_world_model_step(self, X_t, a_t)
        l_rec = F.mse_loss(X_pred, X_tp1)

        h_W, l_dag, l_l1 = self._compute_dag_and_l1()

        if n_i > 0:
            int_pred = X_pred[:n_i]
            int_true = X_tp1[:n_i]
            l_int = self.LAMBDA_INT * F.mse_loss(int_pred, int_true)
        else:
            l_int = torch.tensor(0.0, device=self.device)

        loss = l_rec + l_dag + l_l1 + l_int

        self._finish_train_step(loss)

        return {
            "loss":  round(loss.item(), 5),
            "l_rec": round(l_rec.item(), 5),
            "l_dag": round(l_dag.item(), 5),
            "l_int": round(l_int.item(), 5),
            "l_l1":  round(l_l1.item(), 5),
            "h_W":   round(h_W.item(), 5),
            "mode":  "legacy",
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
            pred = integrate_world_model_step(self, state_vec, a_vec)
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
            pred = integrate_world_model_step(self, state_vec, z)
        return {nid: float(pred[0, i].item()) for i, nid in enumerate(self._node_ids)}

    # ── LeWM: Differentiable propagation + CEM planner ────────────────────────

    def propagate_tensor(
        self, var: str, value: float, *, base: dict[str, float] | None = None,
    ) -> torch.Tensor | None:
        """
        Torch-версия propagate: возвращает тензор (d,) с градиентным графом.

        Для differentiable planning / CEM:
          pred = graph.propagate_tensor("lhip", 0.7)
          loss = -pred[com_z_idx]   # maximize CoM height
          loss.backward()           # gradient flows into GNN weights
        """
        if self._core is None or not self._node_ids:
            return None
        src = base or self.nodes
        x = torch.tensor(
            [float(src.get(n, 0.0)) for n in self._node_ids],
            dtype=torch.float32, device=self.device,
        ).unsqueeze(0)  # (1, d)
        a = torch.zeros(1, self._d, dtype=torch.float32, device=self.device)
        if var in self._node_ids:
            a[0, self._node_ids.index(var)] = float(value)
        pred = self.forward_dynamics(x, a)  # (1, d) — differentiable!
        return pred.squeeze(0)  # (d,)

    def propagate_batch(
        self,
        actions: torch.Tensor,
        *,
        base: dict[str, float] | None = None,
        rollout_steps: int = 0,
    ) -> torch.Tensor:
        """
        Batch forward через world model: N кандидатов за один forward pass.

        Args:
            actions: (N, d) — каждая строка = action vector (sparse: do(var)=val)
            base: начальное состояние (если None → self.nodes)
            rollout_steps: число дополнительных free-rollout шагов после do()

        Returns:
            (N, d) — предсказанные состояния после действия + rollout.
        """
        if self._core is None:
            return actions  # fallback
        src = base or self.nodes
        N = actions.shape[0]
        x0 = torch.tensor(
            [float(src.get(n, 0.0)) for n in self._node_ids],
            dtype=torch.float32, device=self.device,
        ).unsqueeze(0).expand(N, -1)  # (N, d)
        with torch.no_grad():
            pred = self.forward_dynamics(x0, actions)  # (N, d)
            # Multi-step rollout: free dynamics (a=0)
            for _ in range(rollout_steps):
                pred = self.forward_dynamics(pred, torch.zeros_like(pred))
        return pred

    def cem_plan(
        self,
        objective_idx: int | list[int],
        *,
        variable_mask: list[str] | None = None,
        n_samples: int = 64,
        n_elite: int = 8,
        n_iters: int = 5,
        rollout_steps: int = 2,
        maximize: bool = True,
    ) -> dict[str, float] | None:
        """
        CEM (Cross-Entropy Method) planner поверх world model.

        Суть (LeWM Appendix B):
          1. Сэмплировать N кандидатов-действий из N(μ, σ)
          2. Прогнать каждый через forward_dynamics (параллельно!)
          3. Оценить objective по целевым узлам
          4. Выбрать top-K elite → обновить μ, σ
          5. Повторить → финальный μ = лучшее действие

        Args:
            objective_idx: индекс(ы) узла для оптимизации (напр. com_z)
            variable_mask: какие переменные можно менять (None = все)
            n_samples: число кандидатов на итерацию
            n_elite: число лучших для обновления распределения
            n_iters: число итераций CEM
            rollout_steps: шаги free-rollout после действия
            maximize: True = max objective, False = min

        Returns:
            dict {var_name: value} — best action found, or None
        """
        if self._core is None or not self._node_ids:
            return None

        d = self._d
        device = self.device

        # Which variables can be acted upon
        if variable_mask is not None:
            act_indices = [
                self._node_ids.index(v) for v in variable_mask
                if v in self._node_ids
            ]
        else:
            act_indices = list(range(d))
        if not act_indices:
            return None
        n_act = len(act_indices)

        # Objective indices
        if isinstance(objective_idx, int):
            obj_idx = [objective_idx]
        else:
            obj_idx = list(objective_idx)

        # CEM distribution: μ=0.5 (normalized action space), σ=0.2
        mu = torch.full((n_act,), 0.5, device=device)
        sigma = torch.full((n_act,), 0.2, device=device)

        best_action = None
        best_score = float('-inf') if maximize else float('inf')

        for _it in range(n_iters):
            # Sample (N, n_act) from N(μ, σ), clamp to [0.05, 0.95]
            noise = torch.randn(n_samples, n_act, device=device)
            samples = (mu.unsqueeze(0) + sigma.unsqueeze(0) * noise).clamp(0.05, 0.95)

            # Build full action tensor (N, d) — sparse
            actions = torch.zeros(n_samples, d, device=device)
            for ki, ai in enumerate(act_indices):
                actions[:, ai] = samples[:, ki]

            # Parallel forward through world model
            pred = self.propagate_batch(
                actions, rollout_steps=rollout_steps
            )  # (N, d)

            # Evaluate objective
            obj_vals = pred[:, obj_idx].sum(dim=-1)  # (N,)

            # Select elites
            if maximize:
                _, elite_idx = obj_vals.topk(n_elite, largest=True)
            else:
                _, elite_idx = obj_vals.topk(n_elite, largest=False)

            elite_samples = samples[elite_idx]  # (K, n_act)
            elite_score = obj_vals[elite_idx[0]].item()

            # Update best
            if maximize and elite_score > best_score:
                best_score = elite_score
                best_action = actions[elite_idx[0]]
            elif not maximize and elite_score < best_score:
                best_score = elite_score
                best_action = actions[elite_idx[0]]

            # Update distribution
            mu = elite_samples.mean(dim=0)
            sigma = elite_samples.std(dim=0).clamp(min=0.02)

        if best_action is None:
            return None

        # Convert to dict
        result = {}
        for ki, ai in enumerate(act_indices):
            val = float(best_action[ai].item())
            if abs(val) > 0.01:  # only non-trivial actions
                result[self._node_ids[ai]] = val
        result["_cem_score"] = best_score

        return result

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