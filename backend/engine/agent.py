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

import json
import os
import sys
import threading
import time
import warnings
from pathlib import Path
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
MAX_FALLBACK_TRIES = 5  # больше кандидатов, чтобы пройти Value Layer в начале обучения
# Вес slot_* в actual_ig для System 1; основной сигнал — не-визуальные узлы (RKK_VISUAL_IG_WEIGHT=0 → только физика).
VISUAL_IG_WEIGHT = float(os.environ.get("RKK_VISUAL_IG_WEIGHT", "0.1"))
_SELF_VAR_SET = frozenset(SELF_VARS)
# RKK_LOCOMOTION_CPG=1: CPG ведёт ноги; EIG не выбирает прямые do() по этим узлам.
_LOCOMOTION_CPG_LEG_EIG_BLOCK = frozenset(
    {"lhip", "lknee", "lankle", "rhip", "rknee", "rankle"}
)
# CEM motor vars: intent variables that CEM planner can optimize over.
_CEM_MOTOR_PREFIXES = ("intent_", "phys_intent_")

# #region agent log
_DBG_LOG_F7_AGENT = Path(__file__).resolve().parents[2] / "debug-f7a777.log"


def _dbg_agent(hypothesis_id: str, location: str, message: str, data: dict | None = None) -> None:
    try:
        with _DBG_LOG_F7_AGENT.open("a", encoding="utf-8") as _df:
            _df.write(
                json.dumps(
                    {
                        "sessionId": "f7a777",
                        "hypothesisId": hypothesis_id,
                        "location": location,
                        "message": message,
                        "data": data or {},
                        "timestamp": int(time.time() * 1000),
                        "runId": "pre-fix",
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    except Exception:
        pass


# #endregion


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


def _score_max_candidates() -> int:
    """Cap intervention pairs before EIG batch (0 = unlimited). RKK_SCORE_MAX_CANDIDATES."""
    try:
        v = int(os.environ.get("RKK_SCORE_MAX_CANDIDATES", "512"))
    except ValueError:
        return 512
    return max(0, v)


def _score_cache_every() -> int:
    """Пересчёт score_interventions не чаще чем раз в N тиков движка (RKK_SCORE_CACHE_EVERY; 1 = каждый тик)."""
    try:
        return max(1, int(os.environ.get("RKK_SCORE_CACHE_EVERY", "1")))
    except ValueError:
        return 1


def _notears_every() -> int:
    """Частота graph.train_step(): RKK_NOTEAR_EVERY или legacy NOTEARS_EVERY в env (дефолт 8)."""
    try:
        raw = os.environ.get("RKK_NOTEAR_EVERY") or os.environ.get("NOTEARS_EVERY", "8")
        return max(1, int(raw))
    except ValueError:
        return 8


_score_async_win_warned = False


def _score_async_enabled() -> bool:
    """Фоновый поток для score_interventions; по умолчанию выкл. (лок на весь WM давал рывки UI)."""
    global _score_async_win_warned
    v = os.environ.get("RKK_SCORE_ASYNC", "0").strip().lower()
    if v not in ("1", "true", "yes", "on"):
        return False
    # concurrent score_interventions vs train_step / graph mutation → undefined behavior;
    # on Windows this showed up as native crash (e.g. 0xC0000005) under WS load.
    if sys.platform == "win32":
        if not _score_async_win_warned:
            warnings.warn(
                "RKK_SCORE_ASYNC is ignored on Windows (unsafe concurrent CausalGraph access); "
                "use RKK_SCORE_ASYNC=0.",
                UserWarning,
                stacklevel=2,
            )
            _score_async_win_warned = True
        return False
    return True


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
        fd = getattr(self.graph, "forward_dynamics", None)
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
        if unc_m.ndim != 2 or unc_m.shape[0] < d or unc_m.shape[1] < d:
            return []
        if unc_m.shape[0] != d or unc_m.shape[1] != d:
            unc_m = unc_m[:d, :d]
        if int(u_node.shape[0]) != d:
            return []
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
                if idx is not None and 0 <= int(idx) < d:
                    a_batch[bi, int(idx)] = float(cand["value"])
            with torch.no_grad():
                pred = fd(x_batch, a_batch)
            d_x = int(x_batch.shape[-1])
            if int(pred.shape[-1]) != d_x:
                if int(pred.shape[-1]) > d_x:
                    pred = pred[..., :d_x]
                else:
                    return []
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
            ic = 1 if abs(w_ij) >= self.graph.EDGE_THRESH else 0
        else:
            uncertainty, w_ij, grad_norm = 1.0, 0.0, 0.0
            ic = 0
        alpha = 1.0 - uncertainty
        val_from = self.graph.nodes.get(v_from, 0.5)
        val_to = self.graph.nodes.get(v_to, 0.5)
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

    # ── CEM planning: model-based action selection via world model ─────────────
    def _cem_planning_enabled(self) -> bool:
        v = os.environ.get("RKK_CEM_PLANNING", "1").strip().lower()
        return v in ("1", "true", "yes", "on")

    def _maybe_cem_candidate(self, engine_tick: int) -> dict | None:
        """
        CEM planner: use world model for model-based action selection.

        Replaces the broken feedback loop:
          OLD: score_interventions → numpy propagate → scalar actual_ig → System1
          NEW: CEM samples 64 actions → forward_dynamics batch → pick best com_z

        Only activates for motor intent variables (intent_stride, intent_stop_recover, etc.)
        and only when the world model has trained enough to be informative.
        """
        if not self._cem_planning_enabled():
            return None
        if self.graph._core is None:
            return None
        # Wait for world model to train a bit before trusting CEM
        if self._notears_steps < 20:
            return None
        # Don't CEM every tick — every 4th tick is enough
        try:
            cem_every = int(os.environ.get("RKK_CEM_EVERY", "4"))
        except ValueError:
            cem_every = 4
        if engine_tick % max(1, cem_every) != 0:
            return None

        # Find objective indices: com_z and posture_stability
        nids = self.graph._node_ids
        obj_indices = []
        for target in ("com_z", "phys_com_z", "posture_stability", "phys_posture_stability"):
            if target in nids:
                obj_indices.append(nids.index(target))
        if not obj_indices:
            return None

        # Motor intent variables that CEM can optimize
        motor_vars = [
            v for v in nids
            if any(v.startswith(p) for p in _CEM_MOTOR_PREFIXES)
        ]
        if not motor_vars:
            return None

        # Read CEM hyperparams from env
        try:
            n_samples = int(os.environ.get("RKK_CEM_SAMPLES", "64"))
        except ValueError:
            n_samples = 64
        try:
            n_iters = int(os.environ.get("RKK_CEM_ITERS", "5"))
        except ValueError:
            n_iters = 5
        try:
            rollout = int(os.environ.get("RKK_CEM_ROLLOUT", "2"))
        except ValueError:
            rollout = 2

        try:
            result = self.graph.cem_plan(
                objective_idx=obj_indices,
                variable_mask=motor_vars,
                n_samples=min(128, max(16, n_samples)),
                n_elite=max(4, n_samples // 8),
                n_iters=min(10, max(2, n_iters)),
                rollout_steps=min(4, max(0, rollout)),
                maximize=True,
            )
        except Exception:
            return None

        if result is None:
            return result
        cem_score = result.pop("_cem_score", 0.0)

        # Pick the variable with highest action magnitude as the "chosen" do()
        best_var = None
        best_val = 0.0
        for var, val in result.items():
            if abs(val) > abs(best_val):
                best_var = var
                best_val = val
        if best_var is None:
            return None

        # Build candidate in the same format as score_interventions
        target_name = nids[obj_indices[0]] if obj_indices else best_var
        feat = self._features_for_intervention_pair(best_var, target_name)

        return {
            "variable":     best_var,
            "target":       target_name,
            "value":        float(np.clip(best_val, 0.05, 0.95)),
            "uncertainty":  0.5,
            "features":     feat,
            "expected_ig":  float(np.clip(cem_score, 0.0, 1.0)),
            "from_cem":     True,
            "cem_score":    cem_score,
        }

    def _tier1_edge_cap_from_env(self) -> int:
        try:
            return max(0, int(os.environ.get("RKK_TIER1_EDGE_CAP", "2048")))
        except ValueError:
            return 2048

    def _snapshot_edges_max_from_env(self) -> int:
        try:
            return int(os.environ.get("RKK_SNAPSHOT_EDGES_MAX", "512"))
        except ValueError:
            return 512

    def _sample_significant_edge_pairs(
        self, max_pairs: int, rng: np.random.Generator
    ) -> list[tuple[str, str]]:
        """Pairs (from,to) with |W_ij|≥EDGE_THRESH without building graph.edges.

        Если значимых ячеек много, полный ``mask.nonzero()`` даёт O(|E|) GPU/CPU — при |E|≈30k это дорого.
        Тогда включается случайное сэмплирование индексов (батчи), пока не набран max_pairs.
        """
        if max_pairs <= 0:
            return []
        core = self.graph._core
        if core is None:
            return []
        nids = self.graph._node_ids
        d = len(nids)
        if d <= 1:
            return []
        thresh = float(self.graph.EDGE_THRESH)
        try:
            full_scan_max = int(os.environ.get("RKK_TIER1_FULL_SCAN_MAX_EDGES", "4096"))
        except ValueError:
            full_scan_max = 4096
        full_scan_max = max(256, full_scan_max)

        with torch.no_grad():
            W = core.W_masked()
            mask = W.abs() >= thresh
            n_sig = int(mask.sum().item())

        if n_sig == 0:
            return []

        if n_sig <= full_scan_max:
            ij = mask.nonzero(as_tuple=False)
            n = int(ij.shape[0])
            ij_np = ij.cpu().numpy()
            take = min(max_pairs, n)
            if n > take:
                sel = rng.choice(n, size=take, replace=False)
                ij_np = ij_np[sel]
            out: list[tuple[str, str]] = []
            for row in ij_np:
                i, j = int(row[0]), int(row[1])
                if i < len(nids) and j < len(nids):
                    out.append((nids[i], nids[j]))
            return out

        try:
            probe_factor = int(os.environ.get("RKK_TIER1_PROBE_FACTOR", "64"))
        except ValueError:
            probe_factor = 64
        probe_factor = max(8, probe_factor)
        max_attempts = min(d * d, max_pairs * probe_factor)
        batch_cap = 8192

        device = W.device
        seen: set[tuple[int, int]] = set()
        out_pairs: list[tuple[str, str]] = []
        attempts = 0
        while len(out_pairs) < max_pairs and attempts < max_attempts:
            batch = min(batch_cap, max_attempts - attempts)
            if batch <= 0:
                break
            ii = torch.randint(0, d, (batch,), device=device)
            jj = torch.randint(0, d, (batch,), device=device)
            ok = (ii != jj) & (W[ii, jj].abs() >= thresh)
            hit_idx = ok.nonzero(as_tuple=False).flatten()
            attempts += batch
            for hi in hit_idx.tolist():
                iik = int(ii[hi])
                jjk = int(jj[hi])
                key = (iik, jjk)
                if key in seen:
                    continue
                seen.add(key)
                out_pairs.append((nids[iik], nids[jjk]))
                if len(out_pairs) >= max_pairs:
                    break
        return out_pairs

    def _gt_discovery_rate_fast(self) -> float:
        """O(|GT|) vs env.discovery_rate(agent_edges) which is O(|GT|×|E|) when |E|≈d²."""
        gt_list = getattr(self.env, "_gt", None)
        if not gt_list:
            return 0.0
        core = self.graph._core
        if core is None:
            return 0.0
        nid_to_i = {n: i for i, n in enumerate(self.graph._node_ids)}
        with torch.no_grad():
            W_m = core.W_masked().detach().float().cpu().numpy()
        hits = 0
        for gt in gt_list:
            ii, jj = nid_to_i.get(gt.from_), nid_to_i.get(gt.to)
            if ii is None or jj is None:
                continue
            if abs(float(W_m[ii, jj]) - float(gt.weight)) < 0.30:
                hits += 1
        return hits / len(gt_list)

    def _snapshot_edges_payload(self) -> tuple[int, list[dict]]:
        """edge_count + capped edge list for WS/UI without materializing full graph.edges."""
        rng = np.random.default_rng()
        lim = self._snapshot_edges_max_from_env()
        ec = int(self.graph.edge_count)
        if lim <= 0:
            return ec, []
        n_sample = 0 if ec <= 0 else min(lim, ec)
        pairs = self._sample_significant_edge_pairs(n_sample, rng)
        core = self.graph._core
        if core is None or not pairs:
            return ec, []
        nid_to_i = {n: i for i, n in enumerate(self.graph._node_ids)}
        frozen = self.graph._frozen_edge_set
        out: list[dict] = []
        with torch.no_grad():
            W_m = core.W_masked().detach().float().cpu().numpy()
            A_m = core.alpha_trust_matrix().detach().float().cpu().numpy()
        for fr, to in pairs:
            ii, jj = nid_to_i.get(fr), nid_to_i.get(to)
            if ii is None or jj is None:
                continue
            w_ij = float(W_m[ii, jj])
            a_tr = 1.0 if (fr, to) in frozen else float(A_m[ii, jj])
            out.append(
                {
                    "from_": fr,
                    "to": to,
                    "weight": round(w_ij, 4),
                    "alpha_trust": round(float(a_tr), 4),
                    "intervention_count": 1,
                }
            )
        return ec, out

    def _first_significant_edge_labels(self, n: int) -> list[str]:
        """Up to n edge labels from W (no full graph.edges materialization)."""
        if n <= 0:
            return []
        core = self.graph._core
        if core is None:
            return []
        with torch.no_grad():
            W = core.W_masked()
            mask = W.abs() >= self.graph.EDGE_THRESH
            ij = mask.nonzero(as_tuple=False)
        if ij.numel() == 0:
            return []
        nids = self.graph._node_ids
        out: list[str] = []
        for row in ij[:n]:
            i, j = int(row[0]), int(row[1])
            if i < len(nids) and j < len(nids):
                out.append(f"{nids[i]}→{nids[j]}")
            if len(out) >= n:
                break
        return out

    _EDGE_FIRST_OBJECTIVE_TARGETS = (
        "com_z",
        "phys_com_z",
        "posture_stability",
        "phys_posture_stability",
        "target_dist",
        "foot_contact_l",
        "foot_contact_r",
    )

    def _frontier_sample_k_from_env(self) -> int:
        try:
            return max(0, int(os.environ.get("RKK_FRONTIER_SAMPLE", "128")))
        except ValueError:
            return 128

    def _build_candidates_edge_first(
        self,
        *,
        var_ids: list[str],
        nid_to_i: dict[str, int],
        ic_map: dict[tuple[str, str], int],
        W_m: np.ndarray | None,
        unc_m: np.ndarray | None,
        g_m: np.ndarray | None,
        h_W_norm: float,
        disc_rate: float,
    ) -> list[dict]:
        """
        Edge-first candidate generation: Tier1 = sample of significant W_ij (not full graph.edges).

        Tier 1: random sample of |W|≥EDGE_THRESH (RKK_TIER1_EDGE_CAP, default 2048)
        Tier 2: motor intent × objective targets (fixed small set)
        Tier 3: random frontier pairs (RKK_FRONTIER_SAMPLE), optional if unc_m available
        """
        frontier_k = self._frontier_sample_k_from_env()
        v2i = {v: i for i, v in enumerate(var_ids)}
        d = len(var_ids)
        h_clip = float(np.clip(h_W_norm, 0.0, 1.0))
        disc_clip = float(np.clip(disc_rate, 0.0, 1.0))

        def _get_edge_features(vf: str, vt: str) -> tuple[list[float], float]:
            ii = nid_to_i.get(vf)
            jj = nid_to_i.get(vt)
            if W_m is not None and ii is not None and jj is not None and unc_m is not None:
                unc_k = float(unc_m[ii, jj])
                w_ij = float(W_m[ii, jj])
                grad_norm = float(g_m[ii, jj]) if g_m is not None else 0.0
            else:
                unc_k, w_ij, grad_norm = 1.0, 0.0, 0.0
            alpha = 1.0 - unc_k
            ic = ic_map.get((vf, vt), 0)
            feat = self.system1.build_features(
                w_ij=w_ij,
                alpha_ij=alpha,
                val_from=float(self.graph.nodes.get(vf, 0.5)),
                val_to=float(self.graph.nodes.get(vt, 0.5)),
                uncertainty=unc_k,
                h_W_norm=h_clip,
                grad_norm_ij=grad_norm,
                intervention_count=ic,
                discovery_rate=disc_clip,
            )
            return feat, unc_k

        rng = np.random.default_rng()
        posture_now = float(
            self.graph.nodes.get(
                "posture_stability",
                self.graph.nodes.get("phys_posture_stability", 0.5),
            )
        )
        foot_l = float(
            self.graph.nodes.get(
                "foot_contact_l",
                self.graph.nodes.get("phys_foot_contact_l", 0.5),
            )
        )
        foot_r = float(
            self.graph.nodes.get(
                "foot_contact_r",
                self.graph.nodes.get("phys_foot_contact_r", 0.5),
            )
        )
        stable_stance = posture_now > 0.70 and min(foot_l, foot_r) > 0.56

        try:
            _sparse_min_unc = float(os.environ.get("RKK_SPARSE_EIG_MIN_UNC", "0.15"))
        except ValueError:
            _sparse_min_unc = 0.15
        _sparse_min_unc = max(0.0, min(0.8, _sparse_min_unc))

        def _make_candidate(vf: str, vt: str) -> dict | None:
            if vf == vt:
                return None
            if vf not in v2i or vt not in v2i:
                return None
            feat, unc_k = _get_edge_features(vf, vt)
            is_motor = _is_motor_intent_var(vf)
            if _sparse_min_unc > 0 and unc_k < _sparse_min_unc and not is_motor:
                return None
            if is_motor:
                if stable_stance:
                    lo, hi = 0.30, 0.72
                else:
                    lo, hi = 0.35, 0.68
                if str(vf).endswith("stride"):
                    hi = min(hi, 0.62 if stable_stance else 0.56)
                if str(vf).endswith("stop_recover"):
                    lo, hi = (0.55, 0.80) if not stable_stance else (0.40, 0.65)
                rand_val = float(np.clip(rng.uniform(lo, hi), 0.06, 0.94))
            else:
                rand_val = float(np.clip(rng.uniform(0.15, 0.85), 0.06, 0.94))
            return {
                "variable":    vf,
                "target":      vt,
                "value":       rand_val,
                "uncertainty": unc_k,
                "features":    feat,
                "expected_ig": 0.0,
            }

        seen: set[tuple[str, str]] = set()
        candidates: list[dict] = []

        tier1_cap = self._tier1_edge_cap_from_env()
        for vf, vt in self._sample_significant_edge_pairs(tier1_cap, rng):
            key = (vf, vt)
            if key in seen:
                continue
            seen.add(key)
            if self._is_locomotion_primary_active() and vf in _LOCOMOTION_CPG_LEG_EIG_BLOCK:
                continue
            c = _make_candidate(vf, vt)
            if c is not None:
                candidates.append(c)

        motor_vars = [v for v in var_ids if _is_motor_intent_var(v)]
        for mv in motor_vars:
            for tv in self._EDGE_FIRST_OBJECTIVE_TARGETS:
                key = (mv, tv)
                if key in seen:
                    continue
                seen.add(key)
                c = _make_candidate(mv, tv)
                if c is not None:
                    candidates.append(c)

        if frontier_k > 0 and unc_m is not None and d > 1:
            n_try = min(frontier_k * 4, d * d)
            fi_s = rng.integers(0, d, size=n_try)
            fj_s = rng.integers(0, d, size=n_try)
            added = 0
            for fi_k, fj_k in zip(fi_s, fj_s):
                if added >= frontier_k:
                    break
                vf = var_ids[int(fi_k)]
                vt = var_ids[int(fj_k)]
                if vf == vt:
                    continue
                key = (vf, vt)
                if key in seen:
                    continue
                seen.add(key)
                if self._is_locomotion_primary_active() and vf in _LOCOMOTION_CPG_LEG_EIG_BLOCK:
                    continue
                c = _make_candidate(vf, vt)
                if c is not None:
                    candidates.append(c)
                    added += 1

        return candidates

    # ── Epistemic scoring ─────────────────────────────────────────────────────
    def score_interventions(self) -> list[dict]:
        """
        Векторизованный отбор кандидатов: numpy по всем off-diagonal парам var_ids,
        build_features только для top-cap пар + Tier2 motor×objectives (без обхода graph.edges).
        """
        var_ids = self.env.variable_ids
        d = len(var_ids)
        if d <= 1:
            return []

        h_W_norm = min(abs(self._get_h_W()) / max(self.graph._d, 1), 1.0)
        h_clip = float(np.clip(h_W_norm, 0.0, 1.0))
        disc_rate = self.discovery_rate
        disc_clip = float(np.clip(disc_rate, 0.0, 1.0))

        nid_to_i = {n: i for i, n in enumerate(self.graph._node_ids)}
        v2i = {v: i for i, v in enumerate(var_ids)}

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

        cap = _score_max_candidates()
        tier1_select = cap if cap > 0 else self._tier1_edge_cap_from_env()
        tier1_select = max(1, tier1_select)

        mask = ~np.eye(d, dtype=bool)
        fi, fj = np.where(mask)
        n_pairs = int(fi.shape[0])

        ridx = np.zeros(d, dtype=np.int64)
        valid_node = np.zeros(d, dtype=bool)
        for i, v in enumerate(var_ids):
            ji = nid_to_i.get(v)
            if ji is not None:
                ridx[i] = ji
                valid_node[i] = True

        if unc_m is not None and n_pairs > 0:
            ii_n = ridx[fi]
            jj_n = ridx[fj]
            ok = valid_node[fi] & valid_node[fj]
            unc_pairs = np.ones(n_pairs, dtype=np.float32)
            unc_pairs[ok] = unc_m[ii_n[ok], jj_n[ok]]
        else:
            unc_pairs = np.ones(n_pairs, dtype=np.float32)

        is_motor_var = np.array(
            [_is_motor_intent_var(v) for v in var_ids],
            dtype=bool,
        )
        is_motor_arr = is_motor_var[fi]

        try:
            sparse_min_unc = float(os.environ.get("RKK_SPARSE_EIG_MIN_UNC", "0.15"))
        except ValueError:
            sparse_min_unc = 0.15
        sparse_min_unc = max(0.0, min(0.8, sparse_min_unc))

        valid_pairs = is_motor_arr | (unc_pairs >= sparse_min_unc)

        if self._is_locomotion_primary_active():
            blocked = _LOCOMOTION_CPG_LEG_EIG_BLOCK
            leg_block_var = np.array([v in blocked for v in var_ids], dtype=bool)
            cpg_block = leg_block_var[fi]
            valid_pairs &= ~cpg_block

        rng = np.random.default_rng()
        posture = float(
            self.graph.nodes.get(
                "posture_stability",
                self.graph.nodes.get("phys_posture_stability", 0.5),
            )
        )
        foot_l = float(
            self.graph.nodes.get(
                "foot_contact_l",
                self.graph.nodes.get("phys_foot_contact_l", 0.5),
            )
        )
        foot_r = float(
            self.graph.nodes.get(
                "foot_contact_r",
                self.graph.nodes.get("phys_foot_contact_r", 0.5),
            )
        )
        stable_stance = posture > 0.70 and min(foot_l, foot_r) > 0.56

        nodes_arr = np.array(
            [float(self.graph.nodes.get(v, 0.5)) for v in var_ids],
            dtype=np.float64,
        )

        candidates: list[dict] = []

        valid_idx = np.flatnonzero(valid_pairs)
        if valid_idx.size > 0:
            if valid_idx.size > tier1_select:
                scores_sel = unc_pairs[valid_idx]
                top_k = np.argpartition(scores_sel, -tier1_select)[-tier1_select:]
                selected = valid_idx[top_k]
            else:
                selected = valid_idx

            for k in selected:
                i_v, j_v = int(fi[k]), int(fj[k])
                vf, vt = var_ids[i_v], var_ids[j_v]
                unc_k = float(unc_pairs[k])
                ii, jj = int(ridx[i_v]), int(ridx[j_v])
                if W_m is not None and valid_node[i_v] and valid_node[j_v]:
                    w_ij = float(W_m[ii, jj])
                    grad_n = float(g_m[ii, jj]) if g_m is not None else 0.0
                else:
                    w_ij, grad_n = 0.0, 0.0
                alpha = 1.0 - unc_k
                feat = self.system1.build_features(
                    w_ij=w_ij,
                    alpha_ij=alpha,
                    val_from=float(nodes_arr[i_v]),
                    val_to=float(nodes_arr[j_v]),
                    uncertainty=unc_k,
                    h_W_norm=h_clip,
                    grad_norm_ij=grad_n,
                    intervention_count=0,
                    discovery_rate=disc_clip,
                )
                is_motor = bool(is_motor_arr[k])
                if is_motor:
                    lo, hi = (0.30, 0.72) if stable_stance else (0.35, 0.68)
                    if str(vf).endswith("stride"):
                        hi = min(hi, 0.62 if stable_stance else 0.56)
                    if str(vf).endswith("stop_recover"):
                        lo, hi = (0.55, 0.80) if not stable_stance else (0.40, 0.65)
                    val = float(np.clip(rng.uniform(lo, hi), 0.06, 0.94))
                else:
                    val = float(np.clip(rng.uniform(0.15, 0.85), 0.06, 0.94))
                candidates.append(
                    {
                        "variable": vf,
                        "target": vt,
                        "value": val,
                        "uncertainty": unc_k,
                        "features": feat,
                        "expected_ig": 0.0,
                    }
                )

        seen: set[tuple[str, str]] = {(c["variable"], c["target"]) for c in candidates}
        motor_vars = [v for v in var_ids if _is_motor_intent_var(v)]
        for mv in motor_vars:
            for tv in self._EDGE_FIRST_OBJECTIVE_TARGETS:
                if (mv, tv) in seen:
                    continue
                if mv not in v2i or tv not in v2i:
                    continue
                i_v, j_v = v2i[mv], v2i[tv]
                if not valid_node[i_v] or not valid_node[j_v]:
                    continue
                ii, jj = int(ridx[i_v]), int(ridx[j_v])
                if unc_m is not None:
                    unc_k = float(unc_m[ii, jj])
                else:
                    unc_k = 1.0
                w_ij = float(W_m[ii, jj]) if W_m is not None else 0.0
                grad_n = float(g_m[ii, jj]) if g_m is not None else 0.0
                feat = self.system1.build_features(
                    w_ij=w_ij,
                    alpha_ij=1.0 - unc_k,
                    val_from=float(nodes_arr[i_v]),
                    val_to=float(nodes_arr[j_v]),
                    uncertainty=unc_k,
                    h_W_norm=h_clip,
                    grad_norm_ij=grad_n,
                    intervention_count=0,
                    discovery_rate=disc_clip,
                )
                lo, hi = (0.30, 0.72) if stable_stance else (0.35, 0.68)
                if str(mv).endswith("stride"):
                    hi = min(hi, 0.62 if stable_stance else 0.56)
                if str(mv).endswith("stop_recover"):
                    lo, hi = (0.55, 0.80) if not stable_stance else (0.40, 0.65)
                val = float(np.clip(rng.uniform(lo, hi), 0.06, 0.94))
                candidates.append(
                    {
                        "variable": mv,
                        "target": tv,
                        "value": val,
                        "uncertainty": unc_k,
                        "features": feat,
                        "expected_ig": 0.0,
                    }
                )
                seen.add((mv, tv))

        if self._is_locomotion_primary_active():
            candidates = [
                c
                for c in candidates
                if c["variable"] not in _LOCOMOTION_CPG_LEG_EIG_BLOCK
            ]

        if cap > 0 and len(candidates) > cap:
            candidates.sort(key=lambda c: -float(c.get("uncertainty", 0.0)))
            candidates = candidates[:cap]

        if not candidates:
            return []

        scores = self.system1.score([c["features"] for c in candidates])
        for i, cand in enumerate(candidates):
            cand["expected_ig"] = float(scores[i])

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
        _slow_t = {
            "observe": 0.0,
            "score_interventions": 0.0,
            "train_step": 0.0,
            "cem": 0.0,
            "discovery_rate": 0.0,
        }

        def _report_if_slow_tick() -> None:
            total = sum(_slow_t.values())
            if total <= 1.0:
                return
            extra = ""
            try:
                g = self.graph
                extra = (
                    f" | BUFFER_SIZE={g.BUFFER_SIZE} d={g._d} "
                    f"buffer_fill={len(g._obs_buffer)}"
                )
            except Exception:
                pass
            parts = " | ".join(
                f"{k}={v:.3f}s"
                for k, v in sorted(_slow_t.items(), key=lambda x: -x[1])
            )
            print(
                f"[SLOW TICK {engine_tick}] total={total:.2f}s{extra} | {parts}",
                flush=True,
            )

        self._last_engine_tick = engine_tick
        _t0 = time.perf_counter()
        try:
            self.graph.apply_env_observation(dict(self.env.observe()))
        except Exception:
            pass
        _slow_t["observe"] = time.perf_counter() - _t0
        sce = _score_cache_every()
        _t0_si = time.perf_counter()
        # #region agent log
        _t_score = time.perf_counter()
        _score_mode = "?"
        # #endregion
        if (
            sce > 1
            and self._score_cache
            and (engine_tick - self._score_cache_tick) < sce
        ):
            scores = list(self._score_cache)
            # #region agent log
            _score_mode = "sce_span_cache"
            # #endregion
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
                # #region agent log
                _score_mode = "async_have"
                # #endregion
            elif self._score_cache:
                scores = list(self._score_cache)
                # #region agent log
                _score_mode = "async_stale_cache"
                # #endregion
            else:
                with torch.no_grad():
                    scores = self.score_interventions()
                # #region agent log
                _score_mode = "async_fallback_sync"
                # #endregion
                with self._score_lock:
                    self._score_result = list(scores)
            if sce > 1:
                self._score_cache = list(scores)
                self._score_cache_tick = engine_tick
        else:
            with torch.no_grad():
                scores = self.score_interventions()
            # #region agent log
            _score_mode = "sync"
            # #endregion
            if sce > 1:
                self._score_cache = list(scores)
                self._score_cache_tick = engine_tick
        _slow_t["score_interventions"] = time.perf_counter() - _t0_si
        # #region agent log
        _dbg_agent(
            "H1",
            "RKKAgent.step",
            "scores_resolved",
            {
                "mode": _score_mode,
                "ms": (time.perf_counter() - _t_score) * 1000,
                "n_scores": len(scores),
                "engine_tick": engine_tick,
            },
        )
        # #endregion
        gp = self._maybe_goal_planned_candidate() if enable_l3 else None
        if gp is not None and not (
            symbolic_verifier_enabled() and self._symbolic_prediction_bad
        ):
            scores.insert(0, gp)

        # ── CEM planning: use world model for model-based action selection ────
        _t0_cem = time.perf_counter()
        cem_cand = self._maybe_cem_candidate(engine_tick) if enable_l3 else None
        _slow_t["cem"] = time.perf_counter() - _t0_cem
        if cem_cand is not None:
            scores.insert(0, cem_cand)

        if not scores:
            _report_if_slow_tick()
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
            _report_if_slow_tick()
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
            _report_if_slow_tick()
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
        if self._total_interventions % _notears_every() == 0:
            # #region agent log
            _t_ts = time.perf_counter()
            # #endregion
            notears_result = self.graph.train_step()
            _slow_t["train_step"] = time.perf_counter() - _t_ts
            # #region agent log
            _dbg_agent(
                "H2",
                "RKKAgent.step",
                "graph.train_step",
                {
                    "ms": _slow_t["train_step"] * 1000,
                    "interventions": int(self._total_interventions),
                },
            )
            # #endregion
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

        # SSM train — размерность = temporal.d_input (= |graph._node_ids|), не только env.variable_ids
        u_next = torch.tensor(
            [float(self.graph.nodes.get(n, 0.5)) for n in self.graph._node_ids],
            dtype=torch.float32,
            device=self.device,
        )
        self.temporal.train_step(u_next)

        self._total_interventions += 1
        try:
            _v_do = float(value)
        except (TypeError, ValueError):
            _v_do = 0.5
        self._last_do = f"do({var}={_v_do:.2f})"
        self._last_blocked_reason = ""

        _t0_dr = time.perf_counter()
        cur_dr = self.discovery_rate
        _slow_t["discovery_rate"] = time.perf_counter() - _t0_dr
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
            "updated_edges":     self._first_significant_edge_labels(4),
            "pruned_edges":      [],
            "prediction_error":  float(np.mean([
                abs(predicted.get(k, 0) - v) for k, v in observed_env.items()
            ])),
            "cf_predicted": {k: float(round(float(predicted.get(k, 0.0)), 4)) for k in _cf_keys},
            "cf_observed":  {k: float(round(float(observed_full.get(k, 0.0)), 4)) for k in _cf_keys},
            "goal_planned":  bool(chosen.get("from_goal_plan")),
            "from_cem":      bool(chosen.get("from_cem")),
            "symbolic_ok": sym_ok,
            "symbolic_violations": sym_fail,
            "rsi_lite": rsi_event,
            "notears":           notears_result,
        }
        _report_if_slow_tick()
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
        """
        Blend of GT-based and self-supervised discovery rate.
        As the agent matures, self-supervised metric gets more weight.
        """
        gt_dr = self._gt_discovery_rate_fast()
        ss_dr = self.self_supervised_discovery_rate
        # Blend: GT dominates early (calibration), self-supervised dominates later
        if self._total_interventions < 200:
            return gt_dr
        blend = min(1.0, (self._total_interventions - 200) / 1000.0)
        return (1.0 - blend) * gt_dr + blend * ss_dr

    @property
    def self_supervised_discovery_rate(self) -> float:
        """
        Discovery rate without ground-truth edges.
        Based on CausalSurprise compression discoveries — the fraction of
        interventions that actually improved the causal model.
        """
        # Try to get from IntrinsicObjective (if simulation has it patched in)
        try:
            from engine.intristic_objective import IntrinsicObjective
            # Walk up to find intrinsic objective
            for attr_name in ("_intrinsic",):
                # IntrinsicObjective attaches to simulation, not agent
                # We use the causal_surprise directly if available
                pass
            # Fallback: use graph-level stats
            if self.graph.train_losses:
                recent = self.graph.train_losses[-20:]
                if len(recent) >= 5:
                    # Discovery = loss is still decreasing (model is learning)
                    early = float(np.mean(recent[:len(recent)//2]))
                    late = float(np.mean(recent[len(recent)//2:]))
                    if early > 1e-8:
                        improvement = max(0.0, (early - late) / early)
                        return float(np.clip(improvement * 2.0, 0.0, 1.0))
        except Exception:
            pass
        return 0.5  # neutral default

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
        }
        el_ec, el_list = self._snapshot_edges_payload()
        snap["edge_count"] = el_ec
        snap["edges"] = el_list
        if self.env.preset == "pybullet":
            pos_fn = getattr(self.env, "object_positions_world", None)
            if callable(pos_fn):
                snap["physics_objects"] = pos_fn()
        return snap