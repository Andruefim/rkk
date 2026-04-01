"""
agent_v4.py — RKKAgent с Value Layer (Шаг А).

Изменения:
  - ValueLayer.check_action() вызывается перед каждым do()
  - Заблокированные действия → penalty для System 1 + лог события
  - LLM/RAG seed interface: inject_text_priors(edges_json)
  - Fallback scorer когда System 1 буфер ещё мал
  - other_agents_phi передаётся из Simulation для ΔΦ≥0 constraint
"""
from __future__ import annotations

import torch
import numpy as np
from collections import deque

from engine.causal_graph import CausalGraph
from engine.environment  import Environment
from engine.system1      import System1
from engine.temporal     import TemporalBlankets
from engine.value_layer  import ValueLayer, HomeostaticBounds, BlockReason

ACTIVATIONS   = ["relu", "gelu", "tanh"]
NOTEARS_EVERY = 8
MAX_FALLBACK_TRIES = 6   # если действие заблокировано — пробуем следующее


class RKKAgent:
    def __init__(
        self,
        agent_id: int,
        name:     str,
        env:      Environment,
        device:   torch.device,
        bounds:   HomeostaticBounds | None = None,
    ):
        self.id         = agent_id
        self.name       = name
        self.env        = env
        self.device     = device
        self.activation = ACTIVATIONS[agent_id % 3]

        self.graph   = CausalGraph(device)
        self.system1 = System1(activation=self.activation, device=device)
        self.temporal = TemporalBlankets(
            d_input=len(env.variable_ids), device=device
        )
        self.value_layer = ValueLayer(bounds)

        self._cg_history: deque[float] = deque(maxlen=20)
        self._total_interventions = 0
        self._total_blocked       = 0
        self._last_do             = "—"
        self._last_blocked_reason = ""
        self._last_result: dict | None = None
        self._peak_discovery_rate: float = 0.0
        self._notears_steps  = 0
        self._last_notears_loss: dict | None = None

        # Φ других агентов (заполняется Simulation-ом перед step())
        self.other_agents_phi: list[float] = []

        self._bootstrap()

    # ── Bootstrap + LLM seed interface ───────────────────────────────────────
    def _bootstrap(self):
        for var_id, val in self.env.variables.items():
            self.graph.set_node(var_id, val)

        obs0 = dict(self.env.variables)
        self.graph.record_observation(obs0)
        self.temporal.step(obs0)

        # Text priors (spurious + partial GT)
        gt = self.env.gt_edges()
        for e in gt[:2]:
            noisy_w = e["weight"] * 0.3 + (np.random.rand() - 0.5) * 0.4
            self.graph.set_edge(e["from_"], e["to"], noisy_w, alpha=0.06)

        var_ids = self.env.variable_ids
        if len(var_ids) >= 4:
            self.graph.set_edge(var_ids[1], var_ids[3],  0.35, alpha=0.05)
            self.graph.set_edge(var_ids[2], var_ids[0], -0.20, alpha=0.04)

    def inject_text_priors(self, edges: list[dict]) -> int:
        """
        LLM/RAG seed interface.

        edges: [{"from_": "Temp", "to": "Pressure", "weight": 0.8}, ...]

        Все рёбра загружаются с alpha=0.05 (низкое доверие).
        Epistemic Annealing + NOTEARS выжгут ошибочные за N интервенций.

        Возвращает количество успешно загруженных рёбер.
        """
        count = 0
        for e in edges:
            from_ = e.get("from_") or e.get("from")
            to    = e.get("to")
            w     = float(e.get("weight", 0.3))

            if not from_ or not to:
                continue
            if from_ not in self.graph.nodes or to not in self.graph.nodes:
                continue

            # Добавляем с минимальным доверием — как "прочитал в книге"
            alpha = float(e.get("alpha", 0.05))
            self.graph.set_edge(from_, to, w * 0.4, alpha=alpha)
            count += 1

        return count

    # ── Helpers ───────────────────────────────────────────────────────────────
    def _get_h_W(self) -> float:
        if self.graph._core is None:
            return 0.0
        return float(self.graph._core.dag_constraint().item())

    def _get_grad_norm(self, i: int, j: int) -> float:
        if self.graph._core is None or self.graph._core.W.grad is None:
            return 0.0
        return float(self.graph._core.W.grad[i, j].abs().item())

    # ── Epistemic scoring ─────────────────────────────────────────────────────
    def score_interventions(self) -> list[dict]:
        var_ids   = self.env.variable_ids
        h_W_norm  = min(abs(self._get_h_W()) / max(self.graph._d, 1), 1.0)
        disc_rate = self.discovery_rate

        features_batch: list[list[float]] = []
        candidates:     list[dict]        = []

        for v_from in var_ids:
            for v_to in var_ids:
                if v_from == v_to:
                    continue

                uncertainty = self.graph.edge_uncertainty(v_from, v_to)
                w_ij = grad_norm = 0.0

                if (self.graph._core is not None
                        and v_from in self.graph._node_ids
                        and v_to   in self.graph._node_ids):
                    ii = self.graph._node_ids.index(v_from)
                    jj = self.graph._node_ids.index(v_to)
                    w_ij      = self.graph._core.W_masked()[ii, jj].item()
                    grad_norm = self._get_grad_norm(ii, jj)

                alpha    = 1.0 - uncertainty
                val_from = self.graph.nodes.get(v_from, 0.5)
                val_to   = self.graph.nodes.get(v_to, 0.5)
                ic = next(
                    (e.intervention_count for e in self.graph.edges
                     if e.from_ == v_from and e.to == v_to), 0
                )

                feat = self.system1.build_features(
                    w_ij=w_ij, alpha_ij=alpha,
                    val_from=val_from, val_to=val_to,
                    uncertainty=uncertainty, h_W_norm=h_W_norm,
                    grad_norm_ij=grad_norm,
                    intervention_count=ic,
                    discovery_rate=disc_rate,
                )
                features_batch.append(feat)
                candidates.append({
                    "variable":    v_from,
                    "target":      v_to,
                    "value":       0.9 if np.random.rand() > 0.5 else 0.1,
                    "uncertainty": uncertainty,
                    "features":    feat,
                    "expected_ig": 0.0,
                })

        if not features_batch:
            return []

        scores = self.system1.score(features_batch)
        for i, cand in enumerate(candidates):
            cand["expected_ig"] = scores[i]

        return sorted(candidates, key=lambda x: -x["expected_ig"])

    # ── Один шаг с Value Layer ────────────────────────────────────────────────
    def step(self) -> dict:
        scores = self.score_interventions()
        if not scores:
            return {"blocked": False, "skipped": True}

        current_phi = self.phi_approx()
        chosen      = None
        check_result = None
        blocked_count = 0

        # Перебираем кандидатов пока не найдём допустимое действие
        for candidate in scores[:MAX_FALLBACK_TRIES]:
            var   = candidate["variable"]
            value = candidate["value"]

            check_result = self.value_layer.check_action(
                variable=var,
                value=value,
                current_nodes=dict(self.graph.nodes),
                graph=self.graph,
                temporal=self.temporal,
                current_phi=current_phi,
                other_agents_phi=self.other_agents_phi,
            )

            if check_result.allowed:
                chosen = candidate
                break
            else:
                # Штрафуем System 1 за предложение опасного действия
                self.system1.push_experience(
                    features=candidate["features"],
                    actual_ig=check_result.penalty,   # отрицательный IG
                )
                blocked_count += 1
                self._total_blocked += 1
                self._last_blocked_reason = check_result.reason.value

        # Все кандидаты заблокированы — возвращаем событие
        if chosen is None:
            return {
                "blocked":       True,
                "blocked_count": blocked_count,
                "reason":        self._last_blocked_reason,
                "variable":      scores[0]["variable"] if scores else "?",
                "value":         scores[0]["value"] if scores else 0.5,
                "updated_edges": [],
                "compression_delta": 0.0,
                "prediction_error":  0.0,
            }

        # ── Выполняем допустимое действие ────────────────────────────────────
        var   = chosen["variable"]
        value = chosen["value"]

        mdl_before = self.graph.mdl_size
        obs_before = dict(self.env.observe())
        predicted  = self.graph.propagate(var, value)
        observed   = self.env.intervene(var, value)

        # Temporal step
        self.temporal.step(observed)

        # NOTEARS буферы
        self.graph.record_observation(obs_before)
        self.graph.record_observation(observed)
        self.graph.record_intervention(var, value, obs_before, observed)

        # NOTEARS train
        notears_result = None
        if self._total_interventions % NOTEARS_EVERY == 0:
            notears_result = self.graph.train_step()
            if notears_result:
                self._notears_steps += 1
                self._last_notears_loss = notears_result

        # Обновляем узлы
        for node_id, obs_val in observed.items():
            self.graph.nodes[node_id] = obs_val

        mdl_after         = self.graph.mdl_size
        compression_delta = mdl_before - mdl_after
        self._cg_history.append(compression_delta)

        # Обучаем System 1 реальным исходом
        pred_val  = predicted.get(list(observed.keys())[-1], 0.5)
        obs_last  = list(observed.values())[-1]
        actual_ig = float(abs(pred_val - obs_last))

        self.system1.push_experience(
            features=chosen["features"],
            actual_ig=actual_ig,
        )

        # SSM train
        u_next = torch.tensor(
            [observed.get(n, 0.0) for n in self.env.variable_ids],
            dtype=torch.float32, device=self.device
        )
        self.temporal.train_step(u_next)

        self._total_interventions += 1
        self._last_do = f"do({var}={value:.2f})"
        self._last_blocked_reason = ""

        cur_dr = self.discovery_rate
        if cur_dr > self._peak_discovery_rate:
            self._peak_discovery_rate = cur_dr

        self._last_result = {
            "blocked":           False,
            "blocked_count":     blocked_count,
            "variable":          var,
            "value":             value,
            "compression_delta": compression_delta,
            "updated_edges":     [f"{e.from_}→{e.to}" for e in self.graph.edges[:4]],
            "pruned_edges":      [],
            "prediction_error":  float(np.mean([
                abs(predicted.get(k, 0) - v) for k, v in observed.items()
            ])),
            "notears":           notears_result,
        }
        return self._last_result

    # ── Demon ─────────────────────────────────────────────────────────────────
    def demon_disrupt(self) -> str:
        if self.graph._core is None:
            return "no core"
        with torch.no_grad():
            W   = self.graph._core.W
            sig = (W.abs() > 0.05).nonzero(as_tuple=False)
            if len(sig) == 0:
                return "no significant edges"
            idx  = sig[np.random.randint(len(sig))]
            i, j = idx[0].item(), idx[1].item()
            noise = (np.random.rand() - 0.5) * 0.3
            W[i, j] += noise
            fn = self.graph._node_ids[i] if i < len(self.graph._node_ids) else f"v{i}"
            tn = self.graph._node_ids[j] if j < len(self.graph._node_ids) else f"v{j}"
        self.graph._invalidate_cache()
        return f"W[{fn}→{tn}] +{noise:.3f}"

    # ── Properties ────────────────────────────────────────────────────────────
    @property
    def compression_gain(self) -> float:
        if not self._cg_history:
            return 0.0
        return float(np.mean(list(self._cg_history)))

    @property
    def discovery_rate(self) -> float:
        return self.env.discovery_rate([
            {"from_": e.from_, "to": e.to, "weight": e.weight}
            for e in self.graph.edges
        ])

    @property
    def peak_discovery_rate(self) -> float:
        return self._peak_discovery_rate

    def phi_approx(self) -> float:
        return self.temporal.phi_approx()

    def record_phi(self, _: float):
        pass  # temporal управляет историей сам

    # ── Snapshot ──────────────────────────────────────────────────────────────
    def snapshot(self) -> dict:
        cur_dr = self.discovery_rate
        if cur_dr > self._peak_discovery_rate:
            self._peak_discovery_rate = cur_dr

        h_W     = self._get_h_W()
        tb_info = self.temporal.slow_state_summary()
        s1_info = {
            "buffer_size": len(self.system1.buffer),
            "mean_loss":   round(self.system1.mean_loss, 6),
        }
        vl_info = self.value_layer.snapshot()

        notears_info = None
        if self._last_notears_loss:
            notears_info = {
                "steps":  self._notears_steps,
                "loss":   self._last_notears_loss.get("loss", 0),
                "h_W":    round(h_W, 4),
                "l_int":  self._last_notears_loss.get("l_int", 0),
            }

        return {
            "id":                    self.id,
            "name":                  self.name,
            "env_type":              self.env.preset,
            "activation":            self.activation,
            "graph_mdl":             round(self.graph.mdl_size, 3),
            "compression_gain":      round(self.compression_gain, 4),
            "alpha_mean":            round(self.graph.alpha_mean, 3),
            "phi":                   round(self.phi_approx(), 4),
            "node_count":            len(self.graph.nodes),
            "edge_count":            len(self.graph.edges),
            "total_interventions":   self._total_interventions,
            "total_blocked":         self._total_blocked,
            "last_do":               self._last_do,
            "last_blocked_reason":   self._last_blocked_reason,
            "discovery_rate":        round(cur_dr, 3),
            "peak_discovery_rate":   round(self._peak_discovery_rate, 3),
            "h_W":                   round(h_W, 4),
            "notears":               notears_info,
            "temporal":              tb_info,
            "system1":               s1_info,
            "value_layer":           vl_info,
            "edges": [e.as_dict() for e in self.graph.edges],
        }