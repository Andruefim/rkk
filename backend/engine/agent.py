"""
agent_v4.py — RKKAgent с Value Layer (Шаг А).

Изменения:
  - ValueLayer.check_action() вызывается перед каждым do()
  - Заблокированные действия → penalty для System 1 + лог события
  - LLM/RAG seed interface: inject_text_priors(edges_json)
  - Fallback scorer когда System 1 буфер ещё мал
  - other_agents_phi передаётся из Simulation для ΔΦ≥0 constraint

Этап B (гипотезо-ориентированное исследование):
  score_interventions() — аппроксимация информационного выигрыша: чувствительность по узлам
  плюс суррогат снижения суммарной epistemic mass по рёбрам при предсказанном obs (не полный
  байесовский H(W)−E[H(W|obs)]). RKK_EIG_ENTROPY_TERM, RKK_EIG_POSTERIOR_ETA.
  Переключатель: RKK_HYPOTHESIS_EIG=1 (по умолчанию) | 0 | system1 | off | false
  В snapshot: h_W_edge_entropy — сумма бинарных энтропий по α_trust рёбер (диагностика неопределённости W).
  RKK_SCORE_ASYNC=1: score_interventions в фоновом daemon-потоке (тик не ждёт; возможна гонка с train_step — не рекомендуется).
  По умолчанию RKK_SCORE_ASYNC=0 — синхронный пересчёт в главном потоке (стабильно, без общего lock на граф).

Этап Г (самомодель): self_* + update_self_feedback() в humanoid — коррекция намерений по исходу do()
  и по промаху GNN (RKK_SELF_FEEDBACK_LR).

Этап E (целевое планирование): при self_goal_active и наличии target_dist в графе — поиск действия
  через imagination (propagate_from + rollout_step_free), см. engine.goal_planning; RKK_GOAL_PLANNING=0 отключает.

Этап F (символьный верификатор): проверка предсказания propagate на PHYSICS_CONSTRAINTS (engine.symbolic_verifier);
  нарушение → не prepend goal-plan, смешивание expected_ig с uncertainty на следующем шаге; RKK_SYMBOLIC_VERIFY=0 отключает.

Этап G (RSI lite): плато discovery_rate → агент усиливает L1, удваивает BUFFER_SIZE графа (до капа), +1 imagination;
  engine.rsi_lite, RKK_RSI_LITE=0 отключает; RKK_RSI_PLATEAU_TICKS, RKK_RSI_MIN_INTERVENTIONS.
"""
from __future__ import annotations

import os
import threading
from typing import Any
import torch
import numpy as np
from collections import deque

from engine.causal_graph import CausalGraph
from engine.graph_constants import is_read_only_macro_var
from engine.environment  import Environment
from engine.system1      import System1
from engine.temporal     import TemporalBlankets
from engine.value_layer  import ValueLayer, HomeostaticBounds, BlockReason
from engine.phase3_teacher import TeacherIGRule
from engine.environment_humanoid import SELF_VARS
from engine.goal_planning import (
    goal_planning_globally_disabled,
    parse_plan_value_levels,
    plan_beam_k,
    plan_depth,
    plan_max_branch,
    planning_graph_motor_vars,
)
from engine.symbolic_verifier import (
    downrank_factor_for_violation,
    exploration_blend_from_uncertainty,
    symbolic_verifier_enabled,
    verify_normalized_prediction,
)
from engine.wm_neural_ode import integrate_world_model_step
from engine.rsi_lite import (
    rsi_buffer_cap,
    rsi_imagination_cap,
    rsi_improvement_eps,
    rsi_l1_max,
    rsi_l1_scale,
    rsi_lite_enabled,
    rsi_min_interventions,
    rsi_plateau_interventions,
)
from engine.local_reflex import local_reflex_train_enabled, train_chains_parallel

ACTIVATIONS   = ["relu", "gelu", "tanh"]
NOTEARS_EVERY = 8
MAX_FALLBACK_TRIES = 5  # больше кандидатов, чтобы пройти Value Layer в начале обучения
# Вес slot_* в actual_ig для System 1; основной сигнал — не-визуальные узлы (RKK_VISUAL_IG_WEIGHT=0 → только физика).
VISUAL_IG_WEIGHT = float(os.environ.get("RKK_VISUAL_IG_WEIGHT", "0.1"))
_SELF_VAR_SET = frozenset(SELF_VARS)
# RKK_LOCOMOTION_CPG=1: CPG ведёт ноги; EIG не выбирает прямые do() по этим узлам.
_LOCOMOTION_CPG_LEG_EIG_BLOCK = frozenset(
    {"lhip", "lknee", "lankle", "rhip", "rknee", "rankle"}
)


def _is_motor_intent_var(name: str) -> bool:
    return str(name).startswith("intent_") or str(name).startswith("phys_intent_")


def _hypothesis_eig_from_env() -> bool:
    """Этап B: байесовский выбор эксперимента (EIG) вместо только System 1."""
    v = os.environ.get("RKK_HYPOTHESIS_EIG", "1").strip().lower()
    return v not in ("0", "false", "off", "system1", "no", "s1")


def _eig_chunk_size() -> int:
    try:
        return max(1, int(os.environ.get("RKK_EIG_BATCH", "256")))
    except ValueError:
        return 256


def _score_cache_every() -> int:
    """Пересчёт score_interventions не чаще чем раз в N тиков движка (RKK_SCORE_CACHE_EVERY; 1 = каждый тик)."""
    try:
        return max(1, int(os.environ.get("RKK_SCORE_CACHE_EVERY", "1")))
    except ValueError:
        return 1


def _score_async_enabled() -> bool:
    """Фоновый поток для score_interventions; по умолчанию выкл. (лок на весь WM давал рывки UI)."""
    v = os.environ.get("RKK_SCORE_ASYNC", "0").strip().lower()
    return v in ("1", "true", "yes", "on")


def _imagination_horizon_from_env() -> int:
    """Фаза 13: RKK_IMAGINATION_STEPS — число шагов core(X) после мысленного do(); 0 = как раньше."""
    raw = os.environ.get("RKK_IMAGINATION_STEPS", "2")
    try:
        h = int(raw)
    except ValueError:
        h = 0
    return max(0, h)


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
        self._imagination_horizon = _imagination_horizon_from_env()

        self._cg_history: deque[float] = deque(maxlen=20)
        self._total_interventions = 0
        self._total_blocked       = 0
        self._last_do             = "—"
        self._last_blocked_reason = ""
        self._last_result: dict | None = None
        self._symbolic_prediction_bad = False
        self._peak_discovery_rate: float = 0.0
        self._rsi_ref_discovery: float = 0.0
        self._rsi_plateau_count: int = 0
        self._rsi_adjustment_count: int = 0
        self._notears_steps  = 0
        self._last_notears_loss: dict | None = None
        self._local_reflex_cores: dict[tuple[str, ...], Any] = {}
        self._last_local_reflex_train: dict | None = None

        # Φ других агентов (заполняется Simulation-ом перед step())
        self.other_agents_phi: list[float] = []
        self._last_engine_tick = 0
        self._score_cache: list[dict] = []
        self._score_cache_tick: int = -9_999_999
        self._score_thread: threading.Thread | None = None
        self._score_result: list[dict] = []
        self._score_lock = threading.Lock()

        # Фаза 3: LLM-учитель (IG-бонус затухает с числом интервенций)
        self._teacher_rules: list[TeacherIGRule] = []
        self._teacher_weight: float = 0.0

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

        # Фаза 1: заморозка URDF-цепочек в L1 (humanoid VAR_NAMES).
        fr = os.environ.get("RKK_FREEZE_URDF", "1").strip().lower()
        if fr not in ("0", "false", "no", "off") and "lhip" in self.env.variable_ids:
            self.graph.freeze_kinematic_priors()

    def inject_text_priors(self, edges: list[dict]) -> dict:
        """
        LLM/RAG seed interface.

        edges: [{"from_": "Temp", "to": "Pressure", "weight": 0.8}, ...]

        Все рёбра загружаются с alpha=0.05 (низкое доверие).
        Epistemic Annealing + NOTEARS выжгут ошибочные за N интервенций.

        Узлы from_/to должны совпадать с id переменных окружения (env.variable_ids).

        Возвращает {"injected": n, "skipped": [причины...]}.
        """
        count   = 0
        skipped: list[str] = []
        valid   = set(self.graph.nodes.keys())

        for e in edges:
            from_ = e.get("from_") or e.get("from")
            to    = e.get("to")
            w     = float(e.get("weight", 0.3))

            if not from_ or not to:
                skipped.append(f"нет from_/to: {e!r}")
                continue
            if is_read_only_macro_var(from_) or is_read_only_macro_var(to):
                skipped.append(f"read-only macro: {from_!r}→{to!r}")
                continue
            if from_ not in self.graph.nodes:
                skipped.append(f"неизвестный узел «{from_}» (доступны: {sorted(valid)})")
                continue
            if to not in self.graph.nodes:
                skipped.append(f"неизвестный узел «{to}» (доступны: {sorted(valid)})")
                continue

            alpha = float(e.get("alpha", 0.05))
            # Слабые семена по умолчанию (0.2–0.3 экв.) — не «пугают» граф и VL
            w_scaled = min(0.3, max(0.08, float(w) * 0.28))
            self.graph.set_edge(from_, to, w_scaled, alpha=alpha)
            count += 1

        return {"injected": count, "skipped": skipped, "node_ids": sorted(valid)}

    # ── Helpers ───────────────────────────────────────────────────────────────
    def _get_h_W(self) -> float:
        if self.graph._core is None:
            return 0.0
        return float(self.graph._core.dag_constraint().item())

    @staticmethod
    def _marginal_node_uncertainty(unc_m: np.ndarray) -> np.ndarray:
        """
        Маргинальная неопределённость по узлу j: max по всем рёбрам (j→·) и (·→j).
        unc_m[i,j] — epistemic mass на ребре i→j (posterior proxy: 1 − α_trust).
        """
        row_max = unc_m.max(axis=1)
        col_max = unc_m.max(axis=0)
        return np.maximum(row_max, col_max).astype(np.float64, copy=False)

    def _batch_hypothesis_eig(
        self,
        candidates: list[dict],
        X_np: np.ndarray,
        u_node: np.ndarray,
        nid_to_i: dict[str, int],
        unc_m: np.ndarray,
        node_ids: list[str],
        env: Environment,
    ) -> list[float]:
        """
        Суррогат «информативности» действия: (1) чувствительность Σ_j u(j)|ΔX_j|;
        (2) суррогат снижения неопределённости по рёбрам после гипотетического наблюдения
        (масштабирование unc_ij пропорционально |ΔX_i|+|ΔX_j|). Это не точный EIG по H(W).
        """
        core = self.graph._core
        if core is None or not candidates:
            return []
        fd = getattr(core, "forward_dynamics", None)
        if not callable(fd):
            return []

        try:
            lam = float(os.environ.get("RKK_EIG_ENTROPY_TERM", "0.22"))
        except ValueError:
            lam = 0.22
        try:
            eta = float(os.environ.get("RKK_EIG_POSTERIOR_ETA", "0.18"))
        except ValueError:
            eta = 0.18
        lam = max(0.0, lam)
        eta = max(0.0, min(0.95, eta))

        d = int(X_np.shape[0])
        device = self.device
        chunk = _eig_chunk_size()
        eigs: list[float] = []
        x0 = torch.from_numpy(X_np).to(dtype=torch.float32, device=device).unsqueeze(0)
        uu = unc_m.reshape(1, d, d)

        for start in range(0, len(candidates), chunk):
            sub = candidates[start : start + chunk]
            b = len(sub)
            x_batch = x0.expand(b, -1)
            a_batch = torch.zeros(b, d, device=device, dtype=torch.float32)
            for bi, cand in enumerate(sub):
                idx = nid_to_i.get(cand["variable"])
                if idx is not None:
                    a_batch[bi, idx] = float(cand["value"])
            with torch.no_grad():
                pred = integrate_world_model_step(core, x_batch, a_batch)
            delta = (pred - x_batch).abs().cpu().numpy()
            ab = np.abs(delta)
            S = np.clip(ab[:, :, None] + ab[:, None, :], 0.0, 1.0)
            new_u = uu * (1.0 - eta * S)
            new_u = np.maximum(new_u, 0.0)
            reduction = (uu - new_u).sum(axis=(1, 2))
            sens = (delta * u_node.reshape(1, -1)).sum(axis=1)
            total = sens + lam * reduction
            if symbolic_verifier_enabled():
                fac = downrank_factor_for_violation()
                d_nodes = len(node_ids)
                for bi in range(b):
                    pd = {
                        node_ids[j]: float(pred[bi, j].item())
                        for j in range(min(d_nodes, int(pred.shape[1])))
                    }
                    ok, _ = verify_normalized_prediction(pd, env)
                    if not ok:
                        total[bi] *= fac
            eigs.extend(total.tolist())
        return eigs

    def _rollout_imagination_state(
        self, base: dict[str, float], var: str, val: float
    ) -> dict[str, float]:
        """Этап E: один мысленный do + столько же свободных шагов, сколько в VL imagination."""
        s = self.graph.propagate_from(dict(base), var, float(val))
        for _ in range(max(0, self._imagination_horizon)):
            s = self.graph.rollout_step_free(s)
        return s

    def _features_for_intervention_pair(self, v_from: str, v_to: str) -> list[float]:
        """Один вектор признаков System1 для пары (в_from→в_to), как в score_interventions."""
        h_W_norm = min(abs(self._get_h_W()) / max(self.graph._d, 1), 1.0)
        disc_rate = self.discovery_rate
        ic_map: dict[tuple[str, str], int] = {}
        for e in self.graph.edges:
            ic_map[(e.from_, e.to)] = e.intervention_count
        nid_to_i = {n: i for i, n in enumerate(self.graph._node_ids)}
        core = self.graph._core
        ii, jj = nid_to_i.get(v_from), nid_to_i.get(v_to)
        if core is not None and ii is not None and jj is not None:
            with torch.no_grad():
                unc_t = (1.0 - core.alpha_trust_matrix()).detach().float().cpu().numpy()
                W_m = core.W_masked().detach().float().cpu().numpy()
                g_m = None
                if core.W.grad is not None:
                    g_m = core.W.grad.detach().float().abs().cpu().numpy()
            uncertainty = float(unc_t[ii, jj])
            w_ij = float(W_m[ii, jj])
            grad_norm = float(g_m[ii, jj]) if g_m is not None else 0.0
        else:
            uncertainty, w_ij, grad_norm = 1.0, 0.0, 0.0
        alpha = 1.0 - uncertainty
        val_from = self.graph.nodes.get(v_from, 0.5)
        val_to = self.graph.nodes.get(v_to, 0.5)
        ic = ic_map.get((v_from, v_to), 0)
        return self.system1.build_features(
            w_ij=w_ij, alpha_ij=alpha,
            val_from=val_from, val_to=val_to,
            uncertainty=uncertainty, h_W_norm=h_W_norm,
            grad_norm_ij=grad_norm,
            intervention_count=ic,
            discovery_rate=disc_rate,
        )

    def _build_goal_planned_candidate(self, var: str, val: float) -> dict:
        feat = self._features_for_intervention_pair(var, "target_dist")
        return {
            "variable":    var,
            "target":      "target_dist",
            "value":       float(val),
            "uncertainty": 0.35,
            "features":    feat,
            "expected_ig": 1.0,
            "from_goal_plan": True,
        }

    def _maybe_goal_planned_candidate(self) -> dict | None:
        if goal_planning_globally_disabled():
            return None
        if self.graph._core is None:
            return None
        if self.graph.nodes.get("self_goal_active") is None:
            return None
        if float(self.graph.nodes.get("self_goal_active", 0)) <= 0.45:
            return None
        if "target_dist" not in self.graph.nodes:
            return None

        state0 = dict(self.graph.nodes)
        cur_td = float(state0.get("target_dist", 0.5))
        goal_thr = float(state0.get("self_goal_target_dist", 0.42))
        if cur_td <= goal_thr + 0.015:
            return None

        motor = planning_graph_motor_vars(self.env, list(self.graph._node_ids))
        if not motor:
            return None

        levels = parse_plan_value_levels()
        actions = [(v, x) for v in motor for x in levels]
        max_b = plan_max_branch()
        if len(actions) > max_b:
            idx = np.random.choice(len(actions), size=max_b, replace=False)
            actions = [actions[i] for i in idx]

        depth = plan_depth()
        beam_k = plan_beam_k()

        def _td(s: dict[str, float]) -> float:
            return float(s.get("target_dist", cur_td))

        best_td = cur_td
        best_first: tuple[str, float] | None = None

        if depth <= 1:
            for var, val in actions:
                try:
                    sfin = self._rollout_imagination_state(state0, var, val)
                except Exception:
                    continue
                if symbolic_verifier_enabled():
                    ok, _ = verify_normalized_prediction(dict(sfin), self.env)
                    if not ok:
                        continue
                td = _td(sfin)
                if td < best_td - 1e-6:
                    best_td = td
                    best_first = (var, val)
        else:
            scored: list[tuple[float, str, float, dict[str, float]]] = []
            for var, val in actions:
                try:
                    s1 = self._rollout_imagination_state(state0, var, val)
                except Exception:
                    continue
                if symbolic_verifier_enabled():
                    ok, _ = verify_normalized_prediction(dict(s1), self.env)
                    if not ok:
                        continue
                scored.append((_td(s1), var, val, dict(s1)))
            scored.sort(key=lambda t: t[0])
            for _td1, v1, x1, s1 in scored[:beam_k]:
                for v2, x2 in actions:
                    try:
                        sfin = self._rollout_imagination_state(s1, v2, x2)
                    except Exception:
                        continue
                    if symbolic_verifier_enabled():
                        ok, _ = verify_normalized_prediction(dict(sfin), self.env)
                        if not ok:
                            continue
                    td = _td(sfin)
                    if td < best_td - 1e-6:
                        best_td = td
                        best_first = (v1, x1)

        if best_first is None:
            return None
        return self._build_goal_planned_candidate(best_first[0], best_first[1])

    def _is_locomotion_primary_active(self) -> bool:
        """Если CPG управляет ногами, EIG не должен конкурировать за суставы — только intent_* и др."""
        v = os.environ.get("RKK_LOCOMOTION_CPG", "0").strip().lower()
        return v in ("1", "true", "yes", "on")

    # ── Epistemic scoring ─────────────────────────────────────────────────────
    def score_interventions(self) -> list[dict]:
        var_ids   = self.env.variable_ids
        h_W_norm  = min(abs(self._get_h_W()) / max(self.graph._d, 1), 1.0)
        disc_rate = self.discovery_rate

        # Один проход по рёбрам: счётчики интервенций (раньше — O(pairs×|E|) через next() в цикле)
        ic_map: dict[tuple[str, str], int] = {}
        for e in self.graph.edges:
            ic_map[(e.from_, e.to)] = e.intervention_count

        # Имя узла → индекс без O(d) list.index на каждую пару
        nid_to_i = {n: i for i, n in enumerate(self.graph._node_ids)}

        # Один раз W, α и |grad| на CPU — вместо O(d²) вызовов alpha_trust_matrix / W_masked
        core = self.graph._core
        W_m = unc_m = g_m = None
        if core is not None:
            with torch.no_grad():
                W_t = core.W_masked().detach().float()
                A_t = core.alpha_trust_matrix().detach().float()
                W_m = W_t.cpu().numpy()
                unc_m = (1.0 - A_t).cpu().numpy()
            if core.W.grad is not None:
                g_m = core.W.grad.detach().float().abs().cpu().numpy()

        d = len(var_ids)
        if d == 0:
            return []

        # Счётчики интервенций по парам (только известные рёбра — O(|E|))
        ic_mat = np.zeros((d, d), dtype=np.float64)
        v2i = {v: i for i, v in enumerate(var_ids)}
        for (vf, vt), c in ic_map.items():
            i = v2i.get(vf)
            j = v2i.get(vt)
            if i is not None and j is not None and i != j:
                ic_mat[i, j] = float(c)

        ridx = np.zeros(d, dtype=np.int64)
        valid_node = np.zeros(d, dtype=bool)
        for i, v in enumerate(var_ids):
            ji = nid_to_i.get(v)
            if ji is not None:
                ridx[i] = ji
                valid_node[i] = True

        nodes_arr = np.array(
            [float(self.graph.nodes.get(v, 0.5)) for v in var_ids],
            dtype=np.float64,
        )
        mask = ~np.eye(d, dtype=bool)
        fi, fj = np.where(mask)
        n_pairs = len(fi)

        if W_m is not None:
            ii_n = ridx[fi]
            jj_n = ridx[fj]
            ok = valid_node[fi] & valid_node[fj]
            w_ij = np.zeros(n_pairs, dtype=np.float64)
            uncertainty = np.ones(n_pairs, dtype=np.float64)
            grad_norm = np.zeros(n_pairs, dtype=np.float64)
            w_ij[ok] = W_m[ii_n[ok], jj_n[ok]]
            uncertainty[ok] = unc_m[ii_n[ok], jj_n[ok]]
            if g_m is not None:
                grad_norm[ok] = g_m[ii_n[ok], jj_n[ok]]
        else:
            w_ij = np.zeros(n_pairs, dtype=np.float64)
            uncertainty = np.ones(n_pairs, dtype=np.float64)
            grad_norm = np.zeros(n_pairs, dtype=np.float64)

        alpha = 1.0 - uncertainty
        val_from = nodes_arr[fi]
        val_to = nodes_arr[fj]
        ic_v = ic_mat[fi, fj]
        h_clip = float(np.clip(h_W_norm, 0.0, 1.0))
        disc_v = float(np.clip(disc_rate, 0.0, 1.0))

        feats_arr = np.column_stack(
            [
                np.tanh(w_ij),
                np.clip(alpha, 0.0, 1.0),
                np.clip(val_from, 0.0, 1.0),
                np.clip(val_to, 0.0, 1.0),
                np.clip(uncertainty, 0.0, 1.0),
                np.full(n_pairs, h_clip, dtype=np.float64),
                np.tanh(grad_norm),
                np.clip(ic_v / 100.0, 0.0, 1.0),
                np.full(n_pairs, disc_v, dtype=np.float64),
            ]
        )
        features_batch = feats_arr.tolist()

        rng = np.random.default_rng()
        posture_now = float(
            self.graph.nodes.get(
                "posture_stability",
                self.graph.nodes.get("phys_posture_stability", 0.5),
            )
        )
        foot_l_now = float(
            self.graph.nodes.get(
                "foot_contact_l",
                self.graph.nodes.get("phys_foot_contact_l", 0.5),
            )
        )
        foot_r_now = float(
            self.graph.nodes.get(
                "foot_contact_r",
                self.graph.nodes.get("phys_foot_contact_r", 0.5),
            )
        )
        stable_stance = posture_now > 0.70 and min(foot_l_now, foot_r_now) > 0.56
        candidates: list[dict] = []
        for k in range(n_pairs):
            i, j = int(fi[k]), int(fj[k])
            vf, vt = var_ids[i], var_ids[j]
            unc_k = float(uncertainty[k])
            feat_k = features_batch[k]
            if _is_motor_intent_var(vf):
                if stable_stance:
                    lo, hi = 0.30, 0.72
                else:
                    lo, hi = 0.35, 0.68
                if str(vf).endswith("stride"):
                    hi = min(hi, 0.62 if stable_stance else 0.56)
                if str(vf).endswith("stop_recover"):
                    lo, hi = (0.55, 0.80) if not stable_stance else (0.40, 0.65)
                rand_value = float(np.clip(rng.uniform(lo, hi), 0.06, 0.94))
            else:
                rand_value = float(np.clip(rng.uniform(0.15, 0.85), 0.06, 0.94))
            candidates.append({
                "variable":    vf,
                "target":      vt,
                "value":       rand_value,
                "uncertainty": unc_k,
                "features":    feat_k,
                "expected_ig": 0.0,
            })

        if self._is_locomotion_primary_active():
            candidates = [
                c
                for c in candidates
                if c["variable"] not in _LOCOMOTION_CPG_LEG_EIG_BLOCK
            ]
            if posture_now < 0.65:
                candidates = [
                    c
                    for c in candidates
                    if str(c["variable"]).startswith("intent_")
                    or str(c["variable"]).startswith("phys_intent_")
                ]

        if not candidates or not features_batch:
            return []

        use_eig = _hypothesis_eig_from_env() and W_m is not None and unc_m is not None
        if use_eig:
            x_vec = np.array(
                [float(self.graph.nodes.get(n, 0.0)) for n in self.graph._node_ids],
                dtype=np.float64,
            )
            u_node = self._marginal_node_uncertainty(unc_m)
            eigs = self._batch_hypothesis_eig(
                candidates, x_vec, u_node, nid_to_i, unc_m,
                list(self.graph._node_ids), self.env,
            )
            if len(eigs) == len(candidates):
                # Учитываем гипотезу «это ребро неизвестно»: масштаб EIG по unc(v_from→v_to).
                for i, cand in enumerate(candidates):
                    eigs[i] *= 1.0 + float(cand["uncertainty"])
                arr = np.array(eigs, dtype=np.float64)
                lo, hi = float(arr.min()), float(arr.max())
                if hi > lo + 1e-12:
                    normed = (arr - lo) / (hi - lo)
                else:
                    normed = np.full_like(arr, 0.5)
                for i, cand in enumerate(candidates):
                    cand["eig_raw"] = float(eigs[i])
                    cand["expected_ig"] = float(normed[i])
            else:
                use_eig = False

        if not use_eig:
            scores = self.system1.score(features_batch)
            for i, cand in enumerate(candidates):
                cand["expected_ig"] = scores[i]

        if symbolic_verifier_enabled() and self._symbolic_prediction_bad:
            a, b = exploration_blend_from_uncertainty()
            for cand in candidates:
                unc = float(cand.get("uncertainty", 0.5))
                cand["expected_ig"] = a * float(cand["expected_ig"]) + b * unc

        return sorted(candidates, key=lambda x: -x["expected_ig"])

    def _score_async_worker(self) -> None:
        try:
            with torch.no_grad():
                result = self.score_interventions()
            with self._score_lock:
                self._score_result = result
        except Exception as ex:
            print(f"[RKKAgent] score_interventions (async): {ex}")

    def set_teacher_state(self, rules: list[TeacherIGRule], weight: float) -> None:
        """Фаза 3: правила от LLM и текущий teacher_weight (симуляция считает annealing)."""
        self._teacher_rules = list(rules)
        self._teacher_weight = float(max(0.0, min(1.0, weight)))

    def _teacher_ig_bonus(self, variable: str, nodes: dict[str, float]) -> float:
        w = self._teacher_weight
        if w <= 0 or not self._teacher_rules:
            return 0.0
        acc = 0.0
        for r in self._teacher_rules:
            if r.target_var != variable:
                continue
            if r.when_var:
                val = nodes.get(r.when_var)
                if val is None:
                    continue
                if r.when_min is not None and float(val) < r.when_min:
                    continue
                if r.when_max is not None and float(val) > r.when_max:
                    continue
            acc += r.bonus * w
        return min(0.28, acc)

    # ── Один шаг с Value Layer ────────────────────────────────────────────────
    def step(self, engine_tick: int = 0, *, enable_l3: bool = True) -> dict:
        self._last_engine_tick = engine_tick
        try:
            self.graph.apply_env_observation(dict(self.env.observe()))
        except Exception:
            pass
        sce = _score_cache_every()
        if (
            sce > 1
            and self._score_cache
            and (engine_tick - self._score_cache_tick) < sce
        ):
            scores = list(self._score_cache)
        elif _score_async_enabled():
            if self._score_thread is None or not self._score_thread.is_alive():
                self._score_thread = threading.Thread(
                    target=self._score_async_worker,
                    name="rkk_score_interventions",
                    daemon=True,
                )
                self._score_thread.start()
            with self._score_lock:
                have = list(self._score_result) if self._score_result else []
            if have:
                scores = have
            elif self._score_cache:
                scores = list(self._score_cache)
            else:
                with torch.no_grad():
                    scores = self.score_interventions()
                with self._score_lock:
                    self._score_result = list(scores)
            if sce > 1:
                self._score_cache = list(scores)
                self._score_cache_tick = engine_tick
        else:
            with torch.no_grad():
                scores = self.score_interventions()
            if sce > 1:
                self._score_cache = list(scores)
                self._score_cache_tick = engine_tick
        gp = self._maybe_goal_planned_candidate() if enable_l3 else None
        if gp is not None and not (
            symbolic_verifier_enabled() and self._symbolic_prediction_bad
        ):
            scores.insert(0, gp)
        if not scores:
            return {
                "blocked": False, "skipped": True, "prediction_error": 0.0,
                "cf_predicted": {}, "cf_observed": {}, "goal_planned": False,
            }

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
                engine_tick=engine_tick,
                imagination_horizon=(self._imagination_horizon if enable_l3 else 0),
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
                "cf_predicted": {},
                "cf_observed": {},
                "goal_planned": False,
            }

        # ── Выполняем допустимое действие ────────────────────────────────────
        var   = chosen["variable"]
        value = chosen["value"]

        if is_read_only_macro_var(var):
            return {
                "blocked": True,
                "blocked_count": blocked_count + 1,
                "reason": "read_only_macro",
                "variable": var,
                "value": float(value),
                "updated_edges": [],
                "compression_delta": 0.0,
                "prediction_error": 0.0,
                "cf_predicted": {},
                "cf_observed": {},
                "goal_planned": False,
            }

        mdl_before = self.graph.mdl_size
        obs_before_env = dict(self.env.observe())
        self.graph.apply_env_observation(obs_before_env)
        obs_before_full = self.graph.snapshot_vec_dict()
        predicted  = self.graph.propagate(var, value)
        sym_ok, sym_fail = True, []
        if symbolic_verifier_enabled():
            sym_ok, sym_fail = verify_normalized_prediction(dict(predicted), self.env)
            self._symbolic_prediction_bad = not sym_ok
        else:
            self._symbolic_prediction_bad = False
        observed_env = self.env.intervene(var, value)

        # Temporal step (только размерность среды)
        self.temporal.step(observed_env)

        self.graph.apply_env_observation(observed_env)
        observed_full = self.graph.snapshot_vec_dict()

        # NOTEARS / GNN буферы — полный вектор узлов (включая concept_*)
        self.graph.record_observation(obs_before_full)
        self.graph.record_observation(observed_full)
        self.graph.record_intervention(var, value, obs_before_full, observed_full)

        # NOTEARS train
        notears_result = None
        if self._total_interventions % NOTEARS_EVERY == 0:
            notears_result = self.graph.train_step()
            if notears_result:
                self._notears_steps += 1
                self._last_notears_loss = notears_result
            self._maybe_train_local_reflex()

        mdl_after         = self.graph.mdl_size
        compression_delta = mdl_before - mdl_after
        self._cg_history.append(compression_delta)

        # System 1: IG по физике; slot_* и self_* не доминируют метрику (self — прямое задание агентом).
        nids = self.graph._node_ids
        phys_ids = [
            k for k in nids
            if k not in _SELF_VAR_SET and not str(k).startswith("slot_")
        ]
        slot_ids = [k for k in nids if str(k).startswith("slot_")]

        def _mean_abs_err(keys: list) -> float:
            if not keys:
                return 0.0
            return float(np.mean([
                abs(float(predicted.get(k, 0.5)) - float(observed_full.get(k, 0.5)))
                for k in keys
            ]))

        pe_phys = _mean_abs_err(phys_ids)

        # Этап Г: петля «намерение ↔ исход» + ошибка модели → self_* (только среды с методом).
        fn_sf = getattr(self.env, "update_self_feedback", None)
        if callable(fn_sf):
            try:
                fn_sf(
                    variable=var,
                    intended_norm=value,
                    observed=observed_env,
                    predicted=predicted,
                    prediction_error_phys=pe_phys,
                )
            except Exception:
                pass
            obs_self = dict(self.env.observe())
            for sk in _SELF_VAR_SET:
                if sk in self.graph.nodes and sk in obs_self:
                    self.graph.nodes[sk] = float(obs_self[sk])
            self.graph.refresh_concept_aggregates()
        pe_slot = _mean_abs_err(slot_ids)
        w_vis = min(0.45, max(0.0, VISUAL_IG_WEIGHT))
        if slot_ids and phys_ids:
            actual_ig = (1.0 - w_vis) * pe_phys + w_vis * pe_slot
        elif phys_ids:
            actual_ig = pe_phys
        else:
            actual_ig = pe_slot

        t_bonus = self._teacher_ig_bonus(var, dict(self.graph.nodes))
        actual_ig = float(np.clip(actual_ig + t_bonus, 0.0, 1.0))

        self.system1.push_experience(
            features=chosen["features"],
            actual_ig=actual_ig,
        )

        # SSM train
        u_next = torch.tensor(
            [observed_env.get(n, 0.0) for n in self.env.variable_ids],
            dtype=torch.float32, device=self.device
        )
        self.temporal.train_step(u_next)

        self._total_interventions += 1
        self._last_do = f"do({var}={value:.2f})"
        self._last_blocked_reason = ""

        cur_dr = self.discovery_rate
        if cur_dr > self._peak_discovery_rate:
            self._peak_discovery_rate = cur_dr

        rsi_event = self._tick_rsi_lite_discovery(cur_dr)

        _cf_keys = list(self.graph._node_ids)[:48]
        self._last_result = {
            "blocked":           False,
            "blocked_count":     blocked_count,
            "variable":          var,
            "value":             value,
            "compression_delta": compression_delta,
            "updated_edges":     [f"{e.from_}→{e.to}" for e in self.graph.edges[:4]],
            "pruned_edges":      [],
            "prediction_error":  float(np.mean([
                abs(predicted.get(k, 0) - v) for k, v in observed_env.items()
            ])),
            "cf_predicted": {k: float(round(float(predicted.get(k, 0.0)), 4)) for k in _cf_keys},
            "cf_observed":  {k: float(round(float(observed_full.get(k, 0.0)), 4)) for k in _cf_keys},
            "goal_planned":  bool(chosen.get("from_goal_plan")),
            "symbolic_ok": sym_ok,
            "symbolic_violations": sym_fail,
            "rsi_lite": rsi_event,
            "notears":           notears_result,
        }
        return self._last_result

    # ── Demon ─────────────────────────────────────────────────────────────────
    def demon_disrupt(self) -> str:
        if self.graph._core is None:
            return "no core"
        with torch.no_grad():
            W = self.graph._core.W
            sig = (W.abs() > 0.05).nonzero(as_tuple=False)
            if len(sig) == 0:
                return "no significant edges"
            idx = sig[np.random.randint(len(sig))]
            i, j = idx[0].item(), idx[1].item()
            noise = (np.random.rand() - 0.5) * 0.3
            # Нельзя W[i,j] += … — это in-place на view листа с requires_grad.
            w_new = W.detach().clone()
            w_new[i, j] = w_new[i, j] + float(noise)
            W.copy_(w_new)
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

    def _apply_rsi_lite(self) -> dict[str, float | int]:
        g = self.graph
        cur_l1 = float(getattr(g, "LAMBDA_L1", CausalGraph.LAMBDA_L1))
        new_l1 = min(cur_l1 * rsi_l1_scale(), rsi_l1_max())
        g.LAMBDA_L1 = new_l1
        cap_b = rsi_buffer_cap()
        g.BUFFER_SIZE = min(cap_b, int(g.BUFFER_SIZE) * 2)
        cap_i = rsi_imagination_cap()
        self._imagination_horizon = min(cap_i, self._imagination_horizon + 1)
        self._rsi_adjustment_count += 1
        return {
            "LAMBDA_L1": float(new_l1),
            "BUFFER_SIZE": int(g.BUFFER_SIZE),
            "imagination_horizon": int(self._imagination_horizon),
        }

    def _tick_rsi_lite_discovery(self, cur_dr: float) -> dict[str, float | int] | None:
        if not rsi_lite_enabled():
            return None
        if self._total_interventions < rsi_min_interventions():
            return None
        eps = rsi_improvement_eps()
        if cur_dr > self._rsi_ref_discovery + eps:
            self._rsi_ref_discovery = float(cur_dr)
            self._rsi_plateau_count = 0
            return None
        self._rsi_plateau_count += 1
        if self._rsi_plateau_count < rsi_plateau_interventions():
            return None
        self._rsi_plateau_count = 0
        self._rsi_ref_discovery = float(cur_dr)
        return self._apply_rsi_lite()

    def _maybe_train_local_reflex(self) -> None:
        if not local_reflex_train_enabled():
            return
        self._last_local_reflex_train = train_chains_parallel(
            graph=self.graph,
            device=self.graph.device,
            cores=self._local_reflex_cores,
        )

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
        vl_info = dict(self.value_layer.snapshot(self._last_engine_tick))
        vl_info["imagination_horizon"] = self._imagination_horizon

        notears_info = None
        if self._last_notears_loss:
            notears_info = {
                "steps":  self._notears_steps,
                "loss":   self._last_notears_loss.get("loss", 0),
                "h_W":    round(h_W, 4),
                "l_int":  self._last_notears_loss.get("l_int", 0),
            }

        h_W_edge_entropy = None
        core = self.graph._core
        if core is not None:
            with torch.no_grad():
                A = core.alpha_trust_matrix().detach().float().cpu().numpy()
            p = np.clip(A, 1e-7, 1.0 - 1e-7)
            h_W_edge_entropy = float(-(p * np.log(p) + (1.0 - p) * np.log(1.0 - p)).sum())

        snap: dict = {
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
            "teacher": {
                "weight":     round(self._teacher_weight, 4),
                "rules":      len(self._teacher_rules),
            },
            "hypothesis_eig": _hypothesis_eig_from_env(),
            "h_W_edge_entropy": None if h_W_edge_entropy is None else round(h_W_edge_entropy, 4),
            "rsi_lite": {
                "enabled": rsi_lite_enabled(),
                "plateau_count": self._rsi_plateau_count,
                "ref_discovery": round(self._rsi_ref_discovery, 5),
                "adjustments": self._rsi_adjustment_count,
                "LAMBDA_L1": round(float(getattr(self.graph, "LAMBDA_L1", CausalGraph.LAMBDA_L1)), 5),
                "graph_BUFFER_SIZE": int(self.graph.BUFFER_SIZE),
                "imagination_horizon": int(self._imagination_horizon),
            },
            "local_reflex_train": self._last_local_reflex_train,
            "edges": [e.as_dict() for e in self.graph.edges],
        }
        if self.env.preset == "pybullet":
            pos_fn = getattr(self.env, "object_positions_world", None)
            if callable(pos_fn):
                snap["physics_objects"] = pos_fn()
        return snap