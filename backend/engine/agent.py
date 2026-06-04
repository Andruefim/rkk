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
from dataclasses import dataclass
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
from engine.value_layer  import (
    ValueLayer,
    HomeostaticBounds,
    BlockReason,
    efference_predicted_veto,
)
from engine.environment_humanoid import SELF_VARS


@dataclass
class TeacherIGRule:
    """Optional IG bonus when when_var condition holds (legacy hook; rules usually empty)."""

    target_var: str
    when_var: str | None
    when_min: float | None
    when_max: float | None
    bonus: float

from engine.goal_planning import (
    goal_planning_globally_disabled,
    plan_max_branch_effective,
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
from engine.eval_mode import curriculum_context_tags, eval_mode_enabled
from engine.trajectory_contrastive import TrajectoryCollector, trajectory_enabled
from engine.progressive_scope import ProgressiveScope, progressive_scope_enabled

ACTIVATIONS   = ["relu", "gelu", "tanh"]
# RKK_VL_FALLBACK_TRIES (default 3): сколько top-кандидатов проверяет Value Layer за тик.
# Вес slot_* в actual_ig для System 1; основной сигнал — не-визуальные узлы (RKK_VISUAL_IG_WEIGHT=0 → только физика).
VISUAL_IG_WEIGHT = float(os.environ.get("RKK_VISUAL_IG_WEIGHT", "0.1"))
_SELF_VAR_SET = frozenset(SELF_VARS)


def _efference_copy_enabled() -> bool:
    return os.environ.get("RKK_EFFERENCE_COPY", "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _scalar_prediction_error(
    observed_env: dict[str, Any],
    predicted: dict[str, Any],
) -> float:
    """Mean |pred−obs| over env keys, or √(weighted mean precision×err²) when ``RKK_PRECISION_GROUPS``."""
    from engine.precision_channels import (
        default_precision_vector,
        precision_groups_enabled,
        weighted_squared_error_sum,
    )

    err_dict: dict[str, float] = {}
    for k, v in observed_env.items():
        try:
            err_dict[str(k)] = abs(float(predicted.get(k, 0.0)) - float(v))
        except (TypeError, ValueError):
            continue
    if not err_dict:
        return 0.0
    if precision_groups_enabled():
        ws = weighted_squared_error_sum(
            err_dict.keys(), err_dict, precisions=default_precision_vector()
        )
        return float(np.clip(np.sqrt(ws / len(err_dict)), 0.0, 1.5))
    return float(np.mean(list(err_dict.values())))


def _intervention_bootstrap_ticks() -> int:
    try:
        return max(0, int(os.environ.get("RKK_INTERVENTION_BOOTSTRAP_TICKS", "500")))
    except ValueError:
        return 500


def _homeostatic_ig_enabled() -> bool:
    return os.environ.get("RKK_HOMEOSTATIC_IG", "1").strip().lower() in (
        "1", "true", "yes", "on",
    )


def _homeostatic_abs_delta(before: dict, after: dict) -> float | None:
    """Среднее |Δ| по homeostatic-осям среды (нормированные [0,1])."""
    rows = (
        ("posture_stability", "phys_posture_stability"),
        ("com_z", "phys_com_z"),
        ("foot_contact_l", "phys_foot_contact_l"),
        ("foot_contact_r", "phys_foot_contact_r"),
    )
    deltas: list[float] = []
    for p, alt in rows:
        b = before.get(p)
        if b is None:
            b = before.get(alt)
        a = after.get(p)
        if a is None:
            a = after.get(alt)
        if b is None or a is None:
            continue
        try:
            deltas.append(abs(float(a) - float(b)))
        except (TypeError, ValueError):
            continue
    if not deltas:
        return None
    return float(np.clip(np.mean(deltas), 0.0, 1.0))


def _joint_keys_for_ig(nids: list[str]) -> list[str]:
    """Суставы графа для IG при падении (short name или phys_* зеркало)."""
    from engine.features.humanoid.constants import ARM_VARS, HEAD_VARS, LEG_VARS, SPINE_VARS

    allowed = set(LEG_VARS) | set(ARM_VARS) | set(SPINE_VARS) | set(HEAD_VARS)
    out: list[str] = []
    for k in nids:
        sk = str(k)
        if sk in allowed:
            out.append(sk)
        elif sk.startswith("phys_") and sk[5:] in allowed:
            out.append(sk)
    return out


def _ig_fallen_posture_th() -> float:
    try:
        return float(os.environ.get("RKK_IG_FALLEN_POSTURE_TH", "0.25"))
    except ValueError:
        return 0.25


def _ig_fallen_gain() -> float:
    try:
        return max(0.1, float(os.environ.get("RKK_IG_FALLEN_GAIN", "3.0")))
    except ValueError:
        return 3.0


def _ig_joint_fallback_eps() -> float:
    try:
        return max(0.0, float(os.environ.get("RKK_IG_JOINT_FALLBACK_EPS", "1e-4")))
    except ValueError:
        return 1e-4


def _ig_free_joint_coef() -> float:
    """Вес joint PE в свободном режиме (homeostatic + joint), RKK_IG_FREE_JOINT_COEF."""
    try:
        return max(0.0, float(os.environ.get("RKK_IG_FREE_JOINT_COEF", "0.3")))
    except ValueError:
        return 0.3


def _homeostatic_graph_keys(nids: list[str]) -> list[str]:
    """Ключи графа для PE по оси homeostatic (posture/com/feet), по одному на группу."""
    ns = set(nids)
    pairs = (
        ("posture_stability", "phys_posture_stability"),
        ("com_z", "phys_com_z"),
        ("foot_contact_l", "phys_foot_contact_l"),
        ("foot_contact_r", "phys_foot_contact_r"),
    )
    out: list[str] = []
    for a, b in pairs:
        if a in ns:
            out.append(a)
        elif b in ns:
            out.append(b)
    return out


def _env_fixed_root_flag(env: object) -> bool:
    """True если humanoid (или base_env под Visual) в режиме fixed_root."""
    if env is None:
        return False
    for ref in (env, getattr(env, "base_env", None)):
        if ref is None:
            continue
        try:
            fr = getattr(ref, "fixed_root", False)
        except Exception:
            continue
        if isinstance(fr, bool) and fr:
            return True
    return False


def _graph_fixed_root_flag(nodes: object) -> bool:
    """Узлы графа (если есть): self_fixed_root / fixed_root > 0.5."""
    if not isinstance(nodes, dict):
        return False
    for key in ("self_fixed_root", "fixed_root"):
        if key not in nodes:
            continue
        try:
            if float(nodes[key]) > 0.5:
                return True
        except (TypeError, ValueError):
            continue
    return False


def _ig_diag_enabled() -> bool:
    return os.environ.get("RKK_IG_DIAG", "1").strip().lower() in ("1", "true", "yes", "on")


# RKK_LOCOMOTION_CPG=1: CPG ведёт ноги; EIG не выбирает прямые do() по этим узлам.
_LOCOMOTION_CPG_LEG_EIG_BLOCK = frozenset(
    {"lhip", "lknee", "lankle", "rhip", "rknee", "rankle"}
)
# #region agent log
_DBG_LOG_F7_AGENT = Path(__file__).resolve().parents[2] / "debug-f7a777.log"


def _dbg_agent(hypothesis_id: str, location: str, message: str, data: dict | None = None) -> None:
    if os.environ.get("RKK_DBG_AGENT", "0").strip().lower() not in ("1", "true", "yes", "on"):
        return
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
    """Пересчёт score_interventions не чаще чем раз в N тиков движка (RKK_SCORE_CACHE_EVERY)."""
    if "RKK_SCORE_CACHE_EVERY" in os.environ:
        try:
            return max(1, int(os.environ["RKK_SCORE_CACHE_EVERY"]))
        except ValueError:
            return 1
    # Windows: sync score ~1–3s; без env — реже пересчёт, чаще stale-кеш.
    if sys.platform == "win32":
        return 12
    return 4


def _score_stale_mult() -> int:
    """Допустимый возраст кеша score = sce * mult (RKK_SCORE_STALE_MULT)."""
    try:
        return max(1, int(os.environ.get("RKK_SCORE_STALE_MULT", "8")))
    except ValueError:
        return 8


def _score_stale_only() -> bool:
    """Не блокировать тик синхронным score при протухшем кеше (RKK_SCORE_STALE_ONLY)."""
    return os.environ.get("RKK_SCORE_STALE_ONLY", "1").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _notears_every() -> int:
    """Частота graph.train_step(): RKK_NOTEAR_EVERY или legacy NOTEARS_EVERY в env (дефолт 8)."""
    try:
        raw = os.environ.get("RKK_NOTEAR_EVERY") or os.environ.get("NOTEARS_EVERY", "8")
        return max(1, int(raw))
    except ValueError:
        return 8


def _wm_train_due(engine_tick: int, total_interventions: int) -> bool:
    """
    WM train_step cadence. RKK_WM_TRAIN_EVERY>0 → по engine tick; иначе по интервенциям (legacy).
    """
    try:
        te = int(os.environ.get("RKK_WM_TRAIN_EVERY", "0"))
    except ValueError:
        te = 0
    if te > 0:
        return int(engine_tick) > 0 and int(engine_tick) % te == 0
    return int(total_interventions) % _notears_every() == 0


def _max_fallback_tries_from_env() -> int:
    try:
        n = int(os.environ.get("RKK_VL_FALLBACK_TRIES", "3"))
    except ValueError:
        n = 3
    return max(1, min(12, n))


def _vl_fast_fixed_root_intents_enabled() -> bool:
    return os.environ.get("RKK_VL_FAST_FIXED_ROOT", "1").strip().lower() in (
        "1", "true", "yes", "on",
    )


def _vl_fast_fallen_intents_enabled() -> bool:
    return os.environ.get("RKK_VL_FAST_FALLEN", "1").strip().lower() in (
        "1", "true", "yes", "on",
    )


def _vl_fast_intent_enabled() -> bool:
    """Upright walk: skip WM propagate_from_batch when all VL candidates are intents."""
    return os.environ.get("RKK_VL_FAST_INTENT", "1").strip().lower() in (
        "1", "true", "yes", "on",
    )


def _filter_intent_scores(scores: list[dict]) -> list[dict]:
    return [
        c
        for c in scores
        if str(c.get("variable", "")).startswith(("intent_", "phys_intent_"))
    ]


def _nodes_low_posture_for_fast_vl(nodes: dict[str, float]) -> bool:
    if os.environ.get("RKK_VL_FAST_LOW_POSTURE", "1").strip().lower() in (
        "0",
        "false",
        "no",
        "off",
    ):
        return False
    ps = float(
        nodes.get("posture_stability", nodes.get("phys_posture_stability", 0.5))
    )
    cz = float(nodes.get("com_z", nodes.get("phys_com_z", 0.5)))
    try:
        ps_th = float(os.environ.get("RKK_VL_FAST_POSTURE_TH", "0.48"))
    except ValueError:
        ps_th = 0.48
    try:
        cz_th = float(os.environ.get("RKK_VL_FAST_COM_Z_TH", "0.40"))
    except ValueError:
        cz_th = 0.40
    return ps < ps_th or cz < cz_th


def _cheap_vl_s1_intent_batch(
    current_nodes: dict[str, float],
    vl_batch: list[dict],
) -> list[dict] | None:
    """
    Без WM: S1 = текущие узлы с одной подстановкой intent/phys_intent.
    Только если все кандидаты — high-level intents (pelvis fixed).
    """
    if not vl_batch:
        return None
    out: list[dict] = []
    for c in vl_batch:
        var = str(c.get("variable", ""))
        if not (var.startswith("intent_") or var.startswith("phys_intent_")):
            return None
        s1 = dict(current_nodes)
        s1[var] = float(c.get("value", 0.5))
        out.append(s1)
    return out


def _imagination_horizon_from_env() -> int:
    """Фаза 13: RKK_IMAGINATION_STEPS — число шагов core(X) после мысленного do(); 0 = как раньше."""
    raw = os.environ.get("RKK_IMAGINATION_STEPS", "1")
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
        # Φ других агентов (заполняется Simulation-ом перед step())
        self.other_agents_phi: list[float] = []
        self._last_engine_tick = 0
        self._score_cache: list[dict] = []
        self._score_cache_tick: int = -9_999_999
        # Cache for dag_constraint (4× d×d matmuls) — recompute at most once per score_cache window
        self._h_W_cache: float = 0.0
        self._h_W_cache_tick: int = -9_999_999
        self._disc_rate_tick: int = -1
        self._disc_rate_val: float = 0.0

        # Фаза 3: LLM-учитель (IG-бонус затухает с числом интервенций)
        self._teacher_rules: list[TeacherIGRule] = []
        self._teacher_weight: float = 0.0

        # Curriculum: после снятия fixed_root (см. Simulation.disable_fixed_root)
        self._post_fr_explore_until: int = 0
        self._post_fr_vl_relax_until: int = 0
        # SleepConsolidation._end_sleep: реже полный score_interventions (см. _effective_score_cache_every)
        self._post_sleep_score_cache_relax_until: int = 0
        # fixed_root: если кэш scoring снова ставит тот же top-1 что и последний do — ротируем список (см. step)
        self._last_applied_do_key: tuple[str, float] | None = None
        self._repeat_same_top_scores: int = 0

        # System 2: optional slow macro plan injected into agent.step (see engine.system2).
        self._system2_candidate: dict | None = None
        self._s2_planning_context: dict[str, Any] | None = None
        self._s2_wm_cache_cand: dict | None = None
        self._s2_wm_cache_tick: int = -10**9

        # Phase T: Trajectory contrastive learning
        self._traj_collector = TrajectoryCollector()
        # Phase T: Progressive variable scope
        self._prog_scope = ProgressiveScope()

        # Phase 6: causal-surprise replay + genome EMA
        self._replay_buffer: deque[dict] = deque(maxlen=512)
        self._genome_ema_W: torch.Tensor | None = None

        # Phase 5: meta-causal self-model (lazy init in simulation._ensure_phase5)
        self._w_meta: Any | None = None

        self._bootstrap()

    def set_system2_candidate(self, candidate: dict | None) -> None:
        """Simulation sets one scoring row before agent.step (from_system2)."""
        self._system2_candidate = candidate

    def set_s2_planning_context(self, ctx: dict[str, Any] | None) -> None:
        """Контекст System2 для S2-gated WM planner (после system2.tick)."""
        self._s2_planning_context = ctx

    def _observe_env(self) -> dict[str, float]:
        """One PyBullet observe per sim tick when Simulation cache is available."""
        sim = self._resolve_rkk_sim()
        if sim is not None:
            fn = getattr(sim, "_env_observe_cached", None)
            if callable(fn):
                return dict(fn())
        return dict(self.env.observe())

    def _resolve_rkk_sim(self) -> Any | None:
        """Simulation back-ref (not PyBullet _sim)."""
        env = self.env
        sim = getattr(env, "_rkk_sim", None)
        if sim is not None:
            return sim
        base = getattr(env, "base_env", None)
        if base is not None:
            return getattr(base, "_rkk_sim", None)
        return None

    # ── Bootstrap + LLM seed interface ───────────────────────────────────────
    def _bootstrap(self):
        self.graph.set_env_preset(str(getattr(self.env, "preset", "humanoid")))
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

    def _refresh_h_W_cache_if_needed(self) -> None:
        """dag_constraint on d=256 is costly — share cache with score_interventions window."""
        from engine.graph_perf import is_large_graph

        sce = self._effective_score_cache_every(self._last_engine_tick)
        if self._last_engine_tick - self._h_W_cache_tick >= max(1, sce):
            if is_large_graph(self.graph):
                self._h_W_cache = float(getattr(self, "_h_W_cache", 0.5) or 0.5)
            else:
                self._h_W_cache = float(abs(self._get_h_W()))
            self._h_W_cache_tick = self._last_engine_tick

    def _cached_h_W_abs(self) -> float:
        self._refresh_h_W_cache_if_needed()
        return float(self._h_W_cache)

    @staticmethod
    def _marginal_node_uncertainty(unc_m: np.ndarray) -> np.ndarray:
        """
        Маргинальная неопределённость по узлу j: max по всем рёбрам (j→·) и (·→j).
        unc_m[i,j] — epistemic mass на ребре i→j (posterior proxy: 1 − α_trust).
        """
        row_max = unc_m.max(axis=1)
        col_max = unc_m.max(axis=0)
        return np.maximum(row_max, col_max).astype(np.float64, copy=False)

    def _batch_rollout_imagination_states(
        self,
        base: dict[str, float],
        actions: list[tuple[str, float]],
        *,
        row_bases: list[dict[str, float]] | None = None,
    ) -> list[dict[str, float]]:
        """Batched do + free-rollout (goal planning / diagnostics)."""
        if not actions:
            return []
        if row_bases is None:
            states = self.graph.propagate_from_batch(dict(base), actions)
        else:
            states = self.graph.propagate_from_multi_batch(row_bases, actions)
        for _ in range(max(0, self._imagination_horizon)):
            states = self.graph.rollout_step_free_batch(states)
        return states

    def _rollout_imagination_state(
        self, base: dict[str, float], var: str, val: float
    ) -> dict[str, float]:
        """Этап E: один мысленный do + столько же свободных шагов, сколько в VL imagination."""
        out = self._batch_rollout_imagination_states(dict(base), [(var, float(val))])
        return out[0] if out else dict(base)

    def _effective_imagination_horizon(self, enable_l3: bool) -> int:
        if not enable_l3:
            return 0
        if getattr(self.value_layer.bounds, "fixed_root_mode", False):
            return 0
        if _env_fixed_root_flag(self.env) or _graph_fixed_root_flag(self.graph.nodes):
            return 0
        ctx = getattr(self, "_s2_planning_context", None)
        if isinstance(ctx, dict) and ctx.get("fallen_override_active"):
            return 0
        return self._imagination_horizon

    def _s2_wm_cache_every(self) -> int:
        ctx = getattr(self, "_s2_planning_context", None) or {}
        if ctx.get("fallen_override_active"):
            try:
                return max(1, int(os.environ.get("RKK_S2_WM_CACHE_EVERY_FALLEN", "24")))
            except ValueError:
                return 24
        try:
            return max(1, int(os.environ.get("RKK_S2_WM_CACHE_EVERY", "8")))
        except ValueError:
            return 8

    def _enrich_s2_wm_candidate(self, cand: dict, *, macro: str | None = None) -> dict:
        """System1 needs features[] on every WM / schedule candidate."""
        out = dict(cand)
        var = str(out.get("variable", ""))
        if not var:
            return out
        target = str(out.get("target", "posture_stability"))
        if not out.get("features"):
            try:
                out["features"] = self._features_for_intervention_pair(var, target)
            except Exception:
                out["features"] = []
        out.setdefault("from_s2_wm_planner", True)
        out.setdefault("from_system2", True)
        if macro:
            out.setdefault("s2_wm_macro", macro)
        out.setdefault("s2_wm_score", float(out.get("s2_wm_score", 0.0)))
        return out

    def _maybe_s2_wm_candidate(
        self,
        *,
        enable_l3: bool,
        fixed_root: bool,
        engine_tick: int,
        slow_t: dict[str, float],
    ) -> dict | None:
        s2_ctx = getattr(self, "_s2_planning_context", None)
        if not enable_l3 or s2_ctx is None:
            return None
        from engine.system2.wm_planner import plan_s2_wm_candidate, s2_wm_planner_enabled

        if not s2_wm_planner_enabled():
            return None
        if s2_ctx.get("wm_override_schedule_only") and s2_ctx.get(
            "fallen_override_active"
        ):
            sched = s2_ctx.get("recovery_schedule_candidate")
            if isinstance(sched, dict) and sched.get("variable"):
                slow_t["s2_wm_planner"] = 0.0
                return self._enrich_s2_wm_candidate(
                    sched, macro=str(s2_ctx.get("macro", "RECOVER_POSTURE"))
                )
        every = self._s2_wm_cache_every()
        cached = getattr(self, "_s2_wm_cache_cand", None)
        ct = int(getattr(self, "_s2_wm_cache_tick", -10**9))
        if cached is not None and (engine_tick - ct) < every:
            slow_t["s2_wm_planner"] = 0.0
            if cached.get("variable") and not cached.get("features"):
                cached = self._enrich_s2_wm_candidate(
                    cached, macro=str(s2_ctx.get("macro", ""))
                )
            return cached
        t0 = time.perf_counter()
        cand = plan_s2_wm_candidate(
            self,
            planning_context=s2_ctx,
            enable_l3=enable_l3,
            fixed_root=fixed_root,
        )
        slow_t["s2_wm_planner"] = time.perf_counter() - t0
        self._s2_wm_cache_cand = cand
        self._s2_wm_cache_tick = engine_tick
        return cand

    def _goal_planning_suppressed(self) -> bool:
        if goal_planning_globally_disabled() or self.graph._core is None:
            return True
        if getattr(self.value_layer.bounds, "fixed_root_mode", False):
            return True
        if _env_fixed_root_flag(self.env) or _graph_fixed_root_flag(self.graph.nodes):
            return True
        try:
            min_steps = int(os.environ.get("RKK_GOAL_PLAN_MIN_WM_STEPS", "40"))
        except ValueError:
            min_steps = 40
        return self._notears_steps < min_steps

    def _features_for_intervention_pair(self, v_from: str, v_to: str) -> list[float]:
        """Один вектор признаков System1 для пары (в_from→в_to), как в score_interventions."""
        from engine.graph_perf import is_large_graph

        if is_large_graph(self.graph):
            disc_rate = self._discovery_rate_for_tick(self._last_engine_tick)
            val_from = self.graph.nodes.get(v_from, 0.5)
            val_to = self.graph.nodes.get(v_to, 0.5)
            return self.system1.build_features(
                w_ij=0.0,
                alpha_ij=0.5,
                val_from=val_from,
                val_to=val_to,
                uncertainty=0.45,
                h_W_norm=0.5,
                grad_norm_ij=0.0,
                intervention_count=0,
                discovery_rate=disc_rate,
            )
        h_W_norm = min(self._cached_h_W_abs() / max(self.graph._d, 1), 1.0)
        disc_rate = self._discovery_rate_for_tick(self._last_engine_tick)
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
            in_unc = (
                0 <= ii < unc_t.shape[0]
                and 0 <= jj < unc_t.shape[1]
                and 0 <= ii < W_m.shape[0]
                and 0 <= jj < W_m.shape[1]
            )
            if in_unc:
                uncertainty = float(unc_t[ii, jj])
                w_ij = float(W_m[ii, jj])
                if (
                    g_m is not None
                    and 0 <= ii < g_m.shape[0]
                    and 0 <= jj < g_m.shape[1]
                ):
                    grad_norm = float(g_m[ii, jj])
                else:
                    grad_norm = 0.0
                ic = 1 if abs(w_ij) >= self.graph.EDGE_THRESH else 0
            else:
                uncertainty, w_ij, grad_norm = 1.0, 0.0, 0.0
                ic = 0
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

    def _s2_wm_task_active(self) -> bool:
        from engine.system2.wm_planner import s2_wm_planner_enabled, task_from_planning_context

        if not s2_wm_planner_enabled():
            return False
        ctx = getattr(self, "_s2_planning_context", None)
        if not ctx:
            return False
        return task_from_planning_context(ctx, dict(self.graph.nodes)).active

    def _maybe_goal_planned_candidate(self) -> dict | None:
        if self._s2_wm_task_active():
            return None
        if self._goal_planning_suppressed():
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
        fixed_root = (
            getattr(self.value_layer.bounds, "fixed_root_mode", False)
            or _env_fixed_root_flag(self.env)
            or _graph_fixed_root_flag(self.graph.nodes)
        )
        max_b = plan_max_branch_effective(fixed_root=fixed_root)
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
            try:
                states_fin = self._batch_rollout_imagination_states(state0, actions)
            except Exception:
                states_fin = []
            for i, (var, val) in enumerate(actions):
                if i >= len(states_fin):
                    break
                sfin = states_fin[i]
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
            try:
                states1 = self._batch_rollout_imagination_states(state0, actions)
            except Exception:
                states1 = []
            for i, (var, val) in enumerate(actions):
                if i >= len(states1):
                    break
                s1 = states1[i]
                if symbolic_verifier_enabled():
                    ok, _ = verify_normalized_prediction(dict(s1), self.env)
                    if not ok:
                        continue
                scored.append((_td(s1), var, val, dict(s1)))
            scored.sort(key=lambda t: t[0])
            row_bases: list[dict[str, float]] = []
            row_actions: list[tuple[str, float]] = []
            row_meta: list[tuple[str, float]] = []
            for _td1, v1, x1, s1 in scored[:beam_k]:
                for v2, x2 in actions:
                    row_bases.append(s1)
                    row_actions.append((v2, x2))
                    row_meta.append((v1, x1))
            try:
                states2 = self._batch_rollout_imagination_states(
                    state0, row_actions, row_bases=row_bases
                )
            except Exception:
                states2 = []
            for j, sfin in enumerate(states2):
                if j >= len(row_meta):
                    break
                v1, x1 = row_meta[j]
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

    def _tier1_edge_cap_from_env(self) -> int:
        try:
            return max(0, int(os.environ.get("RKK_TIER1_EDGE_CAP", "2048")))
        except ValueError:
            return 2048

    def _snapshot_edges_max_from_env(self) -> int:
        from engine.graph_perf import snapshot_edges_max_for_graph

        return snapshot_edges_max_for_graph(self.graph)

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
        d_graph = len(nids)
        if d_graph <= 1:
            return []
        try:
            full_scan_max = int(os.environ.get("RKK_TIER1_FULL_SCAN_MAX_EDGES", "4096"))
        except ValueError:
            full_scan_max = 4096
        full_scan_max = max(256, full_scan_max)

        thresh = float(self.graph.EDGE_THRESH)
        with torch.no_grad():
            W = core.W_masked()
            mask = W.abs() >= thresh
            n_sig = int(mask.sum().item())

        dW = int(W.shape[0])
        d = min(d_graph, dW, int(W.shape[1]))
        if d <= 1:
            return []

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
            if not (
                0 <= ii < W_m.shape[0]
                and 0 <= jj < W_m.shape[1]
            ):
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
            if not (
                0 <= ii < W_m.shape[0]
                and 0 <= jj < W_m.shape[1]
                and 0 <= ii < A_m.shape[0]
                and 0 <= jj < A_m.shape[1]
            ):
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
            if (
                W_m is not None
                and unc_m is not None
                and ii is not None
                and jj is not None
                and 0 <= ii < W_m.shape[0]
                and 0 <= jj < W_m.shape[1]
                and 0 <= ii < unc_m.shape[0]
                and 0 <= jj < unc_m.shape[1]
            ):
                unc_k = float(unc_m[ii, jj])
                w_ij = float(W_m[ii, jj])
                if (
                    g_m is not None
                    and 0 <= ii < g_m.shape[0]
                    and 0 <= jj < g_m.shape[1]
                ):
                    grad_norm = float(g_m[ii, jj])
                else:
                    grad_norm = 0.0
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
        very_stable = posture_now > 0.85 and min(foot_l, foot_r) > 0.70

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
                if very_stable:
                    lo, hi = 0.35, 0.65
                elif stable_stance:
                    lo, hi = 0.40, 0.60
                else:
                    lo, hi = 0.42, 0.58
                if str(vf).endswith("stride"):
                    hi = min(hi, 0.58 if very_stable else 0.54)
                if str(vf).endswith("stop_recover"):
                    lo, hi = (0.55, 0.72) if not stable_stance else (0.45, 0.60)
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

        bticks = _intervention_bootstrap_ticks()

        # Cache dag_constraint (4× d×d matmuls) — reuse across score cache window
        self._refresh_h_W_cache_if_needed()
        h_W_norm = min(self._h_W_cache / max(self.graph._d, 1), 1.0)
        h_clip = float(np.clip(h_W_norm, 0.0, 1.0))
        disc_rate = self._discovery_rate_for_tick(self._last_engine_tick)
        disc_clip = float(np.clip(disc_rate, 0.0, 1.0))

        nid_to_i = {n: i for i, n in enumerate(self.graph._node_ids)}
        v2i = {v: i for i, v in enumerate(var_ids)}

        core = self.graph._core
        W_m = unc_m = g_m = None
        wm_cap = len(self.graph._node_ids)
        if core is not None:
            with torch.no_grad():
                W_t = core.W_masked().detach().float()
                A_t = core.alpha_trust_matrix().detach().float()
                W_m = W_t.cpu().numpy()
                unc_m = (1.0 - A_t).cpu().numpy()
            if W_m is not None and W_m.ndim == 2:
                wm_cap = min(wm_cap, int(W_m.shape[0]), int(W_m.shape[1]))
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
            if ji is not None and int(ji) < wm_cap:
                ridx[i] = int(ji)
                valid_node[i] = True

        if unc_m is not None and n_pairs > 0:
            ii_n = ridx[fi]
            jj_n = ridx[fj]
            ok = valid_node[fi] & valid_node[fj]
            if unc_m.ndim == 2:
                ok = ok & (ii_n < unc_m.shape[0]) & (jj_n < unc_m.shape[1])
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

        src_controllable = np.array(
            [not is_read_only_macro_var(v) for v in var_ids],
            dtype=bool,
        )
        valid_pairs &= src_controllable[fi]

        # Phase T: progressive scope — restrict interventions to current phase
        if progressive_scope_enabled():
            scope_allowed = self._prog_scope.get_intervention_filter(var_ids)
            scope_mask = np.array(
                [v in scope_allowed for v in var_ids], dtype=bool,
            )
            valid_pairs &= scope_mask[fi]

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
        very_stable = posture > 0.85 and min(foot_l, foot_r) > 0.70

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

            # ── Vectorized feature construction for all selected candidates ──
            sel_fi = fi[selected]
            sel_fj = fj[selected]
            sel_iv = sel_fi.astype(np.intp)
            sel_jv = sel_fj.astype(np.intp)
            sel_ii = ridx[sel_iv]
            sel_jj = ridx[sel_jv]
            sel_unc = unc_pairs[selected].astype(np.float32)
            sel_motor = is_motor_arr[selected]

            # Gather W and grad values vectorized
            n_sel = len(selected)
            w_arr = np.zeros(n_sel, dtype=np.float32)
            g_arr = np.zeros(n_sel, dtype=np.float32)
            ok_wm = valid_node[sel_iv] & valid_node[sel_jv]
            if W_m is not None:
                ok_wm &= (sel_ii < W_m.shape[0]) & (sel_jj < W_m.shape[1])
                w_arr[ok_wm] = W_m[sel_ii[ok_wm], sel_jj[ok_wm]]
                if g_m is not None:
                    ok_gm = ok_wm & (sel_ii < g_m.shape[0]) & (sel_jj < g_m.shape[1])
                    g_arr[ok_gm] = g_m[sel_ii[ok_gm], sel_jj[ok_gm]]

            # Build all features in one vectorized call
            feats_np = self.system1.build_features_batch(
                w_ij=w_arr,
                alpha_ij=1.0 - sel_unc,
                val_from=nodes_arr[sel_iv].astype(np.float32),
                val_to=nodes_arr[sel_jv].astype(np.float32),
                uncertainty=sel_unc,
                h_W_norm=h_clip,
                grad_norm_ij=g_arr,
                intervention_count=np.zeros(n_sel, dtype=np.float32),
                discovery_rate=disc_clip,
            )

            vals_non_motor = rng.uniform(0.15, 0.85, size=n_sel).astype(np.float64)
            if very_stable:
                vals_motor = rng.uniform(0.35, 0.65, size=n_sel).astype(np.float64)
            elif stable_stance:
                vals_motor = rng.uniform(0.40, 0.60, size=n_sel).astype(np.float64)
            else:
                vals_motor = rng.uniform(0.42, 0.58, size=n_sel).astype(np.float64)
            vals = np.where(sel_motor, vals_motor, vals_non_motor)
            vals = np.clip(vals, 0.06, 0.94)

            # Build candidates list
            for idx_k in range(n_sel):
                k = int(selected[idx_k])
                i_v, j_v = int(sel_iv[idx_k]), int(sel_jv[idx_k])
                vf, vt = var_ids[i_v], var_ids[j_v]
                # Per-motor adjustments for stride/stop_recover
                if sel_motor[idx_k]:
                    vf_s = str(vf)
                    if vf_s.endswith("stride"):
                        hi_lim = 0.58 if very_stable else (0.54 if stable_stance else 0.52)
                        vals[idx_k] = min(vals[idx_k], hi_lim)
                    if vf_s.endswith("stop_recover"):
                        lo, hi = (0.55, 0.72) if not stable_stance else (0.45, 0.60)
                        vals[idx_k] = float(np.clip(rng.uniform(lo, hi), 0.06, 0.94))
                candidates.append(
                    {
                        "variable": vf,
                        "target": vt,
                        "value": float(vals[idx_k]),
                        "uncertainty": float(sel_unc[idx_k]),
                        "features": feats_np[idx_k].tolist(),
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
                if unc_m is not None and 0 <= ii < unc_m.shape[0] and 0 <= jj < unc_m.shape[1]:
                    unc_k = float(unc_m[ii, jj])
                else:
                    unc_k = 1.0
                if W_m is not None and 0 <= ii < W_m.shape[0] and 0 <= jj < W_m.shape[1]:
                    w_ij = float(W_m[ii, jj])
                else:
                    w_ij = 0.0
                if (
                    g_m is not None
                    and 0 <= ii < g_m.shape[0]
                    and 0 <= jj < g_m.shape[1]
                ):
                    grad_n = float(g_m[ii, jj])
                else:
                    grad_n = 0.0
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
                if very_stable:
                    lo, hi = 0.35, 0.65
                elif stable_stance:
                    lo, hi = 0.40, 0.60
                else:
                    lo, hi = 0.42, 0.58
                if str(mv).endswith("stride"):
                    hi = min(hi, 0.58 if very_stable else 0.54)
                if str(mv).endswith("stop_recover"):
                    lo, hi = (0.55, 0.72) if not stable_stance else (0.45, 0.60)
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

        # Extract features as numpy array for fast scoring (avoids list→tensor conversion)
        feats_all = np.array([c["features"] for c in candidates], dtype=np.float32)
        scores = self.system1.score_np(feats_all)
        for i, cand in enumerate(candidates):
            cand["expected_ig"] = float(scores[i])

        if bticks > 0 and self._total_interventions < bticks:
            floor = 0.88
            for cand in candidates:
                if _is_motor_intent_var(str(cand.get("variable", ""))):
                    cand["expected_ig"] = max(float(cand["expected_ig"]), floor)

        if symbolic_verifier_enabled() and self._symbolic_prediction_bad:
            a, b = exploration_blend_from_uncertainty()
            for cand in candidates:
                unc = float(cand.get("uncertainty", 0.5))
                cand["expected_ig"] = a * float(cand["expected_ig"]) + b * unc

        try:
            dr_stuck = float(os.environ.get("RKK_SCORE_STUCK_DR_MAX", "0.02"))
        except ValueError:
            dr_stuck = 0.02
        try:
            min_iv_stuck = int(os.environ.get("RKK_SCORE_STUCK_MIN_INTERVENTIONS", "800"))
        except ValueError:
            min_iv_stuck = 800
        try:
            noise_scale = float(os.environ.get("RKK_SCORE_STUCK_NOISE", "0.3"))
        except ValueError:
            noise_scale = 0.3
        if (
            noise_scale > 0.0
            and float(disc_rate) < dr_stuck
            and self._total_interventions >= min_iv_stuck
        ):
            for cand in candidates:
                cand["expected_ig"] = float(cand["expected_ig"]) + float(
                    rng.uniform(0.0, noise_scale)
                )

        return sorted(candidates, key=lambda x: -x["expected_ig"])

    def _effective_score_cache_every(self, engine_tick: int) -> int:
        """
        После сна REM/урок уплотняют W — score_interventions тяжелее; на окне тиков
        держим минимальный интервал не ниже RKK_POST_SLEEP_SCORE_CACHE_EVERY_FLOOR.
        """
        sce = _score_cache_every()
        try:
            until = int(getattr(self, "_post_sleep_score_cache_relax_until", 0) or 0)
        except (TypeError, ValueError):
            until = 0
        if until > 0 and engine_tick <= until:
            try:
                floor = max(
                    1,
                    int(os.environ.get("RKK_POST_SLEEP_SCORE_CACHE_EVERY_FLOOR", "32")),
                )
            except ValueError:
                floor = 32
            return max(sce, floor)
        return sce

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

    def _push_causal_replay(
        self,
        var: str,
        val: float,
        obs_before: dict,
        obs_after: dict,
        *,
        compression_delta: float,
        prediction_error: float,
    ) -> None:
        """Priority replay by causal_surprise × structural_importance."""
        structural = 1.0
        try:
            structural = float(self.graph.alpha_mean)
        except Exception:
            pass
        priority = abs(prediction_error) * (1.0 + abs(compression_delta)) * max(0.1, structural)
        self._replay_buffer.append({
            "var": var,
            "val": val,
            "obs_before": dict(obs_before),
            "obs_after": dict(obs_after),
            "priority": priority,
            "tick": int(self._last_engine_tick),
        })

    def sample_replay_batch(self, k: int = 4) -> list[dict]:
        """Sample top-priority transitions for consolidation replay."""
        if not self._replay_buffer:
            return []
        ranked = sorted(self._replay_buffer, key=lambda x: -x.get("priority", 0.0))
        return ranked[: max(1, k)]

    def _genome_ema_update(self) -> None:
        """Slow EMA of executive W into genome prior (RKK_GENOME_EMA_TAU)."""
        try:
            tau = float(os.environ.get("RKK_GENOME_EMA_TAU", "0.001"))
        except ValueError:
            tau = 0.001
        if tau <= 0 or self.graph._core is None:
            return
        W = self.graph._core.W.detach()
        d = self.graph._d
        if d < 1:
            return
        block = W[:d, :d].clone()
        if self._genome_ema_W is None or self._genome_ema_W.shape != block.shape:
            self._genome_ema_W = block.clone()
            return
        self._genome_ema_W.mul_(1.0 - tau).add_(block, alpha=tau)
        if getattr(self.graph, "_ensemble", None) is not None:
            self.graph._ensemble.sync_from_executive(self._genome_ema_W, idx=0)

    # ── Один шаг с Value Layer ────────────────────────────────────────────────
    def step(self, engine_tick: int = 0, *, enable_l3: bool = True, fallen: bool = False) -> dict:
        _step_t0 = time.perf_counter()
        _fallen_fast = bool(fallen)
        _eval_fast = eval_mode_enabled()
        _perf_fast = _fallen_fast or _eval_fast
        _slow_t = {
            "observe": 0.0,
            "score_interventions": 0.0,
            "value_layer": 0.0,
            "intervene": 0.0,
            "train_step": 0.0,
            "cem": 0.0,
            "discovery_rate": 0.0,
        }

        def _report_if_slow_tick() -> None:
            from engine.tick_profiler import get_tick_profiler

            if get_tick_profiler().enabled():
                return
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
            self.graph.apply_env_observation(
                self._observe_env(), engine_tick=engine_tick
            )
        except Exception:
            pass
        _slow_t["observe"] = time.perf_counter() - _t0
        from engine.system2.wm_planner import s2_wm_gate_strict

        _use_s2_wm_strict = (
            enable_l3 and s2_wm_gate_strict() and self._s2_wm_task_active()
        )
        if _use_s2_wm_strict:
            scores = []
            _score_mode = "s2_wm_strict"
            _slow_t["score_interventions"] = 0.0
            _t_score = time.perf_counter()
        elif _perf_fast and self._score_cache:
            scores = _filter_intent_scores(self._score_cache)
            if not scores:
                s2c = getattr(self, "_system2_candidate", None)
                if s2c is not None:
                    scores = [s2c]
                else:
                    scores = list(self._score_cache)[: _max_fallback_tries_from_env()]
            _score_mode = "fallen_score_cache"
            _slow_t["score_interventions"] = 0.0
            _t_score = time.perf_counter()
        else:
            sce = self._effective_score_cache_every(engine_tick)
            stale_mult = _score_stale_mult()
            cache_age = engine_tick - self._score_cache_tick
            cache_fresh = bool(self._score_cache) and cache_age < sce
            cache_stale_ok = bool(self._score_cache) and cache_age < sce * stale_mult
            due_recompute = not self._score_cache or cache_age >= sce
            _t0_si = time.perf_counter()
            # #region agent log
            _t_score = time.perf_counter()
            _score_mode = "?"
            # #endregion
            _fr_score_only = bool(
                _env_fixed_root_flag(self.env)
                or _graph_fixed_root_flag(self.graph.nodes)
            )
            try:
                from engine.graph_perf import is_large_graph as _is_large_graph

                _large_graph_score_cache = _is_large_graph(self.graph) and bool(
                    self._score_cache
                )
            except ImportError:
                _large_graph_score_cache = False
            if _large_graph_score_cache:
                scores = list(self._score_cache)
                _score_mode = "large_graph_cache_only"
            elif _fr_score_only and self._score_cache:
                scores = list(self._score_cache)
                _score_mode = "fixed_root_cache_only"
                # Не фоновый score_interventions при fixed_root — гонка с train_step вешает ~280.
            elif cache_fresh:
                scores = list(self._score_cache)
                _score_mode = "cache"
            elif cache_stale_ok and _score_stale_only() and due_recompute:
                scores = list(self._score_cache)
                _score_mode = "stale"
            elif due_recompute and (
                not self._score_cache or engine_tick % max(1, sce) == 0
            ):
                cap_ms = 0.0
                try:
                    cap_ms = float(os.environ.get("RKK_SCORE_SYNC_CAP_MS", "400"))
                except ValueError:
                    cap_ms = 400.0
                if (
                    cap_ms > 0
                    and _score_stale_only()
                    and self._score_cache
                ):
                    scores = list(self._score_cache)
                    _score_mode = "sync_cap_stale"
                else:
                    _t_cap = time.perf_counter()
                    with torch.no_grad():
                        scores = self.score_interventions()
                    if cap_ms > 0 and (time.perf_counter() - _t_cap) * 1000.0 > cap_ms:
                        if self._score_cache:
                            scores = list(self._score_cache)
                            _score_mode = "sync_cap_fallback"
                        else:
                            _score_mode = "sync_refresh_slow"
                    else:
                        _score_mode = "sync_refresh"
                if sce > 1:
                    self._score_cache = list(scores)
                    self._score_cache_tick = engine_tick
            elif self._score_cache:
                scores = list(self._score_cache)
                _score_mode = "sync_deferred"
            else:
                with torch.no_grad():
                    scores = self.score_interventions()
                _score_mode = "sync_bootstrap"
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
        s2_ctx = getattr(self, "_s2_planning_context", None)
        s2_task_active = self._s2_wm_task_active()
        fr_for_plan = bool(
            _env_fixed_root_flag(self.env) or _graph_fixed_root_flag(self.graph.nodes)
        )

        wm_cand: dict | None = None
        if enable_l3 and s2_ctx is not None and not (
            _fallen_fast and not s2_ctx.get("fallen_override_active")
        ):
            wm_cand = self._maybe_s2_wm_candidate(
                enable_l3=enable_l3,
                fixed_root=fr_for_plan,
                engine_tick=engine_tick,
                slow_t=_slow_t,
            )

        if s2_wm_gate_strict() and s2_task_active:
            scores = []
            if wm_cand is not None:
                scores.append(wm_cand)
            else:
                s2c = getattr(self, "_system2_candidate", None)
                if s2c is not None:
                    scores.append(s2c)
                self._system2_candidate = None
            _slow_t["cem"] = 0.0
            if scores and (s2_ctx or {}).get("fallen_override_active"):
                k_top = (
                    str(scores[0].get("variable", "")),
                    round(float(scores[0].get("value", 0.5)), 4),
                )
                le = getattr(self, "_last_applied_do_key", None)
                if le is not None and k_top == le:
                    self._repeat_same_top_scores += 1
                else:
                    self._repeat_same_top_scores = 0
                try:
                    rlim = max(
                        4,
                        int(os.environ.get("RKK_S2_WM_STUCK_ROTATE_TICKS", "16")),
                    )
                except ValueError:
                    rlim = 16
                if self._repeat_same_top_scores >= rlim and scores:
                    v0 = scores[0]
                    var0 = str(v0.get("variable", ""))
                    val0 = float(v0.get("value", 0.5))
                    alt = float(np.clip(val0 - 0.12, 0.06, 0.94))
                    if abs(alt - val0) < 0.02:
                        alt = float(np.clip(val0 + 0.12, 0.06, 0.94))
                    scores[0] = {**v0, "value": alt, "s2_wm_stuck_nudge": True}
                    self._repeat_same_top_scores = 0
        else:
            gp = (
                self._maybe_goal_planned_candidate()
                if enable_l3 and not _perf_fast
                else None
            )
            if gp is not None and not (
                symbolic_verifier_enabled() and self._symbolic_prediction_bad
            ):
                scores.insert(0, gp)

            if wm_cand is not None:
                scores.insert(0, wm_cand)
            else:
                s2c = getattr(self, "_system2_candidate", None)
                if s2c is not None:
                    scores.insert(0, s2c)
            self._system2_candidate = None

        post_to = int(getattr(self, "_post_fr_explore_until", 0))
        if post_to > engine_tick and len(scores) >= 10:
            head = scores[:10]
            np.random.default_rng().shuffle(head)
            scores = head + scores[10:]

        # Pelvis fixed: при застое «тот же кандидат #0 == последний do» VL каждый раз одобряет одно и то же.
        # RKK_FIXED_ROOT_SCORE_ROTATE_TICKS: после стольких тиков подряд — сдвинуть #0 в конец очереди.
        if enable_l3:
            try:
                fr_now = _env_fixed_root_flag(self.env) or _graph_fixed_root_flag(
                    self.graph.nodes
                )
            except Exception:
                fr_now = False
            if fr_now and scores:
                k_top = (
                    str(scores[0].get("variable", "")),
                    round(float(scores[0].get("value", 0.5)), 4),
                )
                le = getattr(self, "_last_applied_do_key", None)
                if le is not None and k_top == le:
                    self._repeat_same_top_scores += 1
                else:
                    self._repeat_same_top_scores = 0
                try:
                    rlim = max(
                        4, int(os.environ.get("RKK_FIXED_ROOT_SCORE_ROTATE_TICKS", "24"))
                    )
                except ValueError:
                    rlim = 24
                if self._repeat_same_top_scores >= rlim and len(scores) >= 2:
                    scores = scores[1:] + scores[:1]
                    self._repeat_same_top_scores = 0
                    try:
                        sce_rot = self._effective_score_cache_every(engine_tick)
                    except Exception:
                        sce_rot = _score_cache_every()
                    if sce_rot > 1:
                        self._score_cache = list(scores)
                        self._score_cache_tick = engine_tick
            elif not fr_now:
                self._repeat_same_top_scores = 0

        if not scores:
            _report_if_slow_tick()
            return {
                "blocked": False, "skipped": True, "prediction_error": 0.0,
                "cf_predicted": {}, "cf_observed": {}, "goal_planned": False,
                "from_system2": False,
            }

        _nodes_now = dict(self.graph.nodes)
        _vl_fast = bool(_perf_fast)
        if not _vl_fast and _vl_fast_fallen_intents_enabled():
            _vl_fast = _nodes_low_posture_for_fast_vl(_nodes_now)
        if isinstance(s2_ctx, dict) and s2_ctx.get("fallen_override_active"):
            _vl_fast = True
        if _vl_fast:
            _intent_scores = _filter_intent_scores(scores)
            if _intent_scores:
                scores = _intent_scores
        elif _vl_fast_intent_enabled():
            _intent_scores = _filter_intent_scores(scores)
            if _intent_scores:
                scores = _intent_scores

        current_phi = self.phi_approx()
        chosen      = None
        check_result = None
        blocked_count = 0

        _pfr_vl = 0.0
        if engine_tick < int(getattr(self, "_post_fr_vl_relax_until", 0)):
            try:
                _pfr_vl = float(os.environ.get("RKK_POST_FR_VL_LOCO_BLEND", "0.3"))
            except ValueError:
                _pfr_vl = 0.3
            _pfr_vl = float(np.clip(_pfr_vl, 0.0, 1.0))
        _recovery_vl = False
        s2_ctx_vl = getattr(self, "_s2_planning_context", None)
        if isinstance(s2_ctx_vl, dict) and s2_ctx_vl.get("fallen_override_active"):
            if os.environ.get("RKK_S2_RECOVERY_VL_RELAX", "1").strip().lower() not in (
                "0",
                "false",
                "no",
                "off",
            ):
                _recovery_vl = True
                _pfr_vl = 1.0

        vl_horizon = (
            0
            if (_fallen_fast or _vl_fast)
            else self._effective_imagination_horizon(enable_l3)
        )
        vl_tries = min(_max_fallback_tries_from_env(), len(scores))
        if _fallen_fast or _vl_fast:
            try:
                vl_tries = max(1, min(vl_tries, int(os.environ.get("RKK_VL_FALLBACK_TRIES_FALLEN", "1"))))
            except ValueError:
                vl_tries = 1
        vl_batch = scores[:vl_tries]
        current_nodes = _nodes_now
        _t0_vl = time.perf_counter()
        cheap: list[dict] | None = None
        if vl_horizon == 0 and _vl_fast_fixed_root_intents_enabled():
            _fr_vl = _env_fixed_root_flag(self.env) or _graph_fixed_root_flag(
                self.graph.nodes
            )
            _use_cheap = _fr_vl or (
                _vl_fast and _vl_fast_fallen_intents_enabled()
            ) or _vl_fast_intent_enabled()
            if _use_cheap:
                cheap = _cheap_vl_s1_intent_batch(current_nodes, vl_batch)
        if cheap is not None:
            s1_batch = cheap
        else:
            s1_batch = self.graph.propagate_from_batch(
                current_nodes,
                [(c["variable"], c["value"]) for c in vl_batch],
            )

        # Перебираем кандидатов пока не найдём допустимое действие
        chosen_pre_s1: dict[str, float] | None = None
        for i, candidate in enumerate(vl_batch):
            var   = candidate["variable"]
            value = candidate["value"]
            pre_s1 = s1_batch[i] if i < len(s1_batch) else None

            check_result = self.value_layer.check_action(
                variable=var,
                value=value,
                current_nodes=current_nodes,
                graph=self.graph,
                temporal=self.temporal,
                current_phi=current_phi,
                other_agents_phi=self.other_agents_phi,
                engine_tick=engine_tick,
                imagination_horizon=vl_horizon,
                post_fr_loco_relax=_pfr_vl,
                recovery_override=_recovery_vl,
                precomputed_s1=pre_s1,
            )

            if check_result.allowed:
                chosen = candidate
                chosen_pre_s1 = pre_s1
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
        _slow_t["value_layer"] = time.perf_counter() - _t0_vl

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
                "from_system2": False,
            }

        # ── Выполняем допустимое действие ────────────────────────────────────
        var   = chosen["variable"]
        value = chosen["value"]

        # Global safety cap: motor intents stay near neutral to prevent falling
        if _is_motor_intent_var(var):
            value = float(np.clip(value, 0.35, 0.65))

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
                "from_system2": False,
            }

        if _fallen_fast:
            mdl_before = 0.0
        else:
            mdl_before = self.graph.mdl_size
        obs_before_env = self._observe_env()
        self.graph.apply_env_observation(obs_before_env)
        obs_before_full = self.graph.snapshot_vec_dict()
        if (_fallen_fast or _vl_fast) and chosen_pre_s1 is not None:
            predicted = dict(chosen_pre_s1)
        else:
            predicted = self.graph.propagate(var, value)
        _fixed_root_now = bool(
            getattr(self.value_layer.bounds, "fixed_root_mode", False)
            or _env_fixed_root_flag(self.env)
            or _graph_fixed_root_flag(self.graph.nodes)
        )
        _ev_block, _ev_msg = efference_predicted_veto(
            var, dict(predicted), fixed_root=_fixed_root_now
        )
        if _ev_block:
            _report_if_slow_tick()
            self._total_blocked += 1
            self._last_blocked_reason = "efference_predicted_veto"
            return {
                "blocked": True,
                "blocked_count": blocked_count + 1,
                "reason": "efference_predicted_veto",
                "message": _ev_msg,
                "variable": var,
                "value": float(value),
                "updated_edges": [],
                "compression_delta": 0.0,
                "prediction_error": 0.0,
                "cf_predicted": {},
                "cf_observed": {},
                "goal_planned": False,
                "from_system2": False,
            }
        sym_ok, sym_fail = True, []
        if symbolic_verifier_enabled():
            sym_ok, sym_fail = verify_normalized_prediction(dict(predicted), self.env)
            self._symbolic_prediction_bad = not sym_ok
        else:
            self._symbolic_prediction_bad = False
        _t0_iv = time.perf_counter()
        observed_env = self.env.intervene(var, value)

        # Temporal step (только размерность среды)
        self.temporal.step(observed_env)

        self.graph.apply_env_observation(observed_env)
        observed_full = self.graph.snapshot_vec_dict()

        # NOTEARS / GNN буферы — полный вектор узлов (включая concept_*)
        self.graph.record_observation(obs_before_full)
        self.graph.record_observation(observed_full)
        self.graph.record_intervention(var, value, obs_before_full, observed_full)
        _slow_t["intervene"] = time.perf_counter() - _t0_iv

        # NOTEARS train
        notears_result = None
        sim_ref = self._resolve_rkk_sim()
        wm_warmup = int(getattr(self, "_wm_warmup_until", 0) or 0)
        edge_blocked = bool(getattr(sim_ref, "_edge_growth_blocked", False)) if sim_ref else False
        skip_notears = (
            (wm_warmup > 0 and engine_tick <= wm_warmup)
            or edge_blocked
        )
        if sim_ref is not None:
            cnt = int(self.graph.edge_count)
            prev = int(getattr(sim_ref, "_prev_edge_count", cnt))
            try:
                single_cap = max(10, int(os.environ.get("RKK_MAX_EDGE_DELTA_SINGLE", "120")))
            except ValueError:
                single_cap = 120
            if cnt - prev > single_cap:
                skip_notears = True
                sim_ref._edge_growth_blocked = True
        _train_due = _wm_train_due(engine_tick, self._total_interventions)
        if _fallen_fast:
            try:
                fe = int(os.environ.get("RKK_WM_TRAIN_EVERY_FALLEN", "0"))
            except ValueError:
                fe = 0
            if fe > 0:
                _train_due = int(engine_tick) > 0 and int(engine_tick) % fe == 0
            else:
                _train_due = False
        if _train_due and not skip_notears and not eval_mode_enabled():
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

        if _fallen_fast:
            mdl_after = mdl_before
        else:
            mdl_after = self.graph.mdl_size
        compression_delta = mdl_before - mdl_after
        self._cg_history.append(compression_delta)

        pe_mean = float(
            np.mean([
                abs(float(predicted.get(k, 0.5)) - float(observed_full.get(k, 0.5)))
                for k in self.graph._node_ids[:32]
            ]) if self.graph._node_ids else 0.0
        )
        if not _perf_fast:
            self._push_causal_replay(
                var, value, obs_before_full, observed_full,
                compression_delta=compression_delta,
                prediction_error=pe_mean,
            )
            self.graph.update_ensemble_posterior(
                obs_before_full,
                observed_full,
                var,
                value,
                intervention_index=int(self._total_interventions),
            )
            self._genome_ema_update()

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

        def _joint_ig_value() -> float | None:
            jkeys = _joint_keys_for_ig(list(self.graph._node_ids))
            if not jkeys:
                return None
            return float(
                np.clip(_mean_abs_err(jkeys) * _ig_fallen_gain(), 0.0, 1.0)
            )

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
            try:
                fr_ob = _env_fixed_root_flag(self.env) or _graph_fixed_root_flag(
                    self.graph.nodes
                )
            except Exception:
                fr_ob = False
            if fr_ob and str(var).startswith("intent_"):
                obs_self = dict(observed_env)
                st_src = getattr(self.env, "base_env", self.env)
                st = getattr(st_src, "_self_state", None)
                if isinstance(st, dict):
                    for sk in _SELF_VAR_SET:
                        if sk in st:
                            obs_self[sk] = float(np.clip(float(st[sk]), 0.05, 0.95))
                else:
                    obs_self = dict(self.env.observe())
            else:
                obs_self = dict(self.env.observe())
            for sk in _SELF_VAR_SET:
                if sk in self.graph.nodes and sk in obs_self:
                    self.graph.nodes[sk] = float(obs_self[sk])
            self.graph.refresh_concept_aggregates()
        pe_slot = _mean_abs_err(slot_ids)
        w_vis = min(0.45, max(0.0, VISUAL_IG_WEIGHT))
        ig_home = (
            _homeostatic_abs_delta(obs_before_env, observed_env)
            if _homeostatic_ig_enabled()
            else None
        )

        posture_now = float(
            self.graph.nodes.get(
                "posture_stability",
                self.graph.nodes.get("phys_posture_stability", 1.0),
            )
        )
        is_fallen = posture_now < _ig_fallen_posture_th()

        is_fixed_root = _fixed_root_now

        # Падение и fixed_root: homeostatic (posture/com/foot) часто константы → joint PE×gain.
        use_joint_ig = is_fallen or is_fixed_root
        if use_joint_ig:
            jv = _joint_ig_value()
            if jv is not None:
                actual_ig = jv
            elif ig_home is not None:
                actual_ig = ig_home
            elif slot_ids and phys_ids:
                actual_ig = (1.0 - w_vis) * pe_phys + w_vis * pe_slot
            elif phys_ids:
                actual_ig = pe_phys
            else:
                actual_ig = pe_slot
        elif ig_home is not None:
            # Свободное стояние: |Δobserve| по homeostatic часто ≈0 — добавляем PE по тем же осям + суставам.
            hk = _homeostatic_graph_keys(list(nids))
            jk = _joint_keys_for_ig(list(nids))
            jc = _ig_free_joint_coef()
            homeo_pe = _mean_abs_err(hk) if hk else 0.0
            joint_pe = float(_mean_abs_err(jk) * jc) if jk else 0.0
            if hk or jk:
                actual_ig = float(np.clip(homeo_pe + joint_pe, 0.0, 1.0))
            else:
                actual_ig = float(ig_home)
        elif slot_ids and phys_ids:
            actual_ig = (1.0 - w_vis) * pe_phys + w_vis * pe_slot
        elif phys_ids:
            actual_ig = pe_phys
        else:
            actual_ig = pe_slot

        # Любое состояние без homeostatic сигнала (не падение): запасной joint IG.
        _jf = _ig_joint_fallback_eps()
        if not is_fallen and float(actual_ig) <= _jf:
            jv2 = _joint_ig_value()
            if jv2 is not None and float(jv2) > _jf:
                actual_ig = float(jv2)

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
        if not _perf_fast:
            self.temporal.train_step(u_next)

        self._total_interventions += 1
        self._last_applied_do_key = (str(var), round(float(value), 4))
        try:
            _v_do = float(value)
        except (TypeError, ValueError):
            _v_do = 0.5
        self._last_do = f"do({var}={_v_do:.2f})"
        self._last_blocked_reason = ""

        # Phase T: feed trajectory collector + progressive scope
        _sim = self._resolve_rkk_sim()
        if trajectory_enabled() and not _perf_fast:
            is_env_fallen = False
            try:
                fn_fallen = getattr(self.env, 'is_fallen', None)
                if callable(fn_fallen):
                    is_env_fallen = fn_fallen()
                else:
                    is_env_fallen = is_fallen
            except Exception:
                is_env_fallen = is_fallen
            _cur_tags = {}
            if _sim is not None:
                _cur_tags = curriculum_context_tags(_sim, self)
            completed_seg = self._traj_collector.tick(
                obs=observed_env,
                action=(var, float(value)),
                is_fallen=is_env_fallen,
                node_ids=list(self.graph._node_ids),
                engine_tick=engine_tick,
                curriculum_tags=_cur_tags,
            )
            if completed_seg is not None and not eval_mode_enabled():
                # Feed completed segment to GNN for trajectory-level training
                self.graph._traj_segments.append(completed_seg)
                if len(self.graph._traj_segments) > self.graph._traj_max_segments:
                    self.graph._traj_segments = self.graph._traj_segments[-self.graph._traj_max_segments:]
        if progressive_scope_enabled() and not _perf_fast:
            tq = self._traj_collector.recent_quality() if trajectory_enabled() else None
            self._prog_scope.tick(
                is_fallen=is_fallen,
                posture=posture_now,
                quality=tq,
                is_fixed_root=is_fixed_root,
                sim=_sim,
                agent=self,
            )

        if _ig_diag_enabled() and self._total_interventions % 50 == 0:
            try:
                am = float(self.graph.alpha_mean)
            except Exception:
                am = float("nan")
            print(
                f"[IGDiag] iv={self._total_interventions} "
                f"actual_ig={actual_ig:.4f} "
                f"is_fallen={is_fallen} "
                f"is_fixed_root={is_fixed_root} "
                f"alpha_mean={am:.3f} "
                f"var={var}",
                flush=True,
            )

        _t0_dr = time.perf_counter()
        if _perf_fast:
            try:
                dr_every = max(1, int(os.environ.get("RKK_DISCOVERY_RATE_EVERY_FALLEN", "8")))
            except ValueError:
                dr_every = 8
            if engine_tick % dr_every == 0 or self._disc_rate_tick < 0:
                cur_dr = self._discovery_rate_for_tick(engine_tick)
            else:
                cur_dr = float(self._disc_rate_val)
        else:
            cur_dr = self._discovery_rate_for_tick(engine_tick)
        _slow_t["discovery_rate"] = time.perf_counter() - _t0_dr
        if cur_dr > self._peak_discovery_rate:
            self._peak_discovery_rate = cur_dr

        rsi_event = self._tick_rsi_lite_discovery(cur_dr)

        _cf_keys = list(self.graph._node_ids)[:48]
        _eff = None
        if _efference_copy_enabled():
            try:
                from engine.efference_copy import efference_correlation_report

                _eff = efference_correlation_report(
                    dict(predicted), dict(observed_full)
                )
            except Exception:
                _eff = None
        if (
            _eff is not None
            and not _eff.get("ok", True)
            and os.environ.get("RKK_PRECISION_DOWN_ON_EFFERENCE", "0").strip().lower()
            in ("1", "true", "yes", "on")
        ):
            try:
                from engine.precision_groups import get_precision_state

                get_precision_state().decay_vision(0.85)
            except Exception:
                pass
        self._last_result = {
            "blocked":           False,
            "blocked_count":     blocked_count,
            "variable":          var,
            "value":             value,
            "compression_delta": compression_delta,
            "updated_edges":     self._first_significant_edge_labels(4),
            "pruned_edges":      [],
            "prediction_error":  _scalar_prediction_error(observed_env, predicted),
            "cf_predicted": {k: float(round(float(predicted.get(k, 0.0)), 4)) for k in _cf_keys},
            "cf_observed":  {k: float(round(float(observed_full.get(k, 0.0)), 4)) for k in _cf_keys},
            "goal_planned":  bool(chosen.get("from_goal_plan")),
            "from_cem":      bool(chosen.get("from_cem")),
            "from_system2": bool(chosen.get("from_system2")),
            "from_s2_wm_planner": bool(chosen.get("from_s2_wm_planner")),
            "s2_wm_macro": chosen.get("s2_wm_macro"),
            "s2_wm_score": chosen.get("s2_wm_score"),
            "symbolic_ok": sym_ok,
            "symbolic_violations": sym_fail,
            "rsi_lite": rsi_event,
            "notears":           notears_result,
            "efference": _eff,
        }
        _report_if_slow_tick()
        self._last_step_timings = {
            k: round(float(v) * 1000.0, 2) for k, v in _slow_t.items()
        }
        self._last_step_timings["total_ms"] = round(
            (time.perf_counter() - _step_t0) * 1000.0, 2
        )
        from engine.tick_profiler import get_tick_profiler

        _prof = get_tick_profiler()
        if _prof.enabled():
            _prof.merge_dict_seconds(_slow_t, prefix="agent")
            _prof.record("agent.step_total", self._last_step_timings["total_ms"])
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

    def _discovery_rate_for_tick(self, engine_tick: int) -> float:
        """Один расчёт discovery_rate на engine-тик (step + snapshot + score)."""
        if engine_tick == self._disc_rate_tick:
            return self._disc_rate_val
        gt_dr = self._gt_discovery_rate_fast()
        ss_dr = self.self_supervised_discovery_rate
        if self._total_interventions < 200:
            val = gt_dr
        else:
            blend = min(1.0, (self._total_interventions - 200) / 1000.0)
            val = (1.0 - blend) * gt_dr + blend * ss_dr
        self._disc_rate_tick = engine_tick
        self._disc_rate_val = float(val)
        return self._disc_rate_val

    @property
    def discovery_rate(self) -> float:
        """
        Blend of GT-based and self-supervised discovery rate.
        As the agent matures, self-supervised metric gets more weight.
        """
        return self._discovery_rate_for_tick(self._last_engine_tick)

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

    def phi_approx(self) -> float:
        return self.temporal.phi_approx()

    def record_phi(self, _: float):
        pass  # temporal управляет историей сам

    # ── Snapshot ──────────────────────────────────────────────────────────────
    def snapshot(self) -> dict:
        cur_dr = self._discovery_rate_for_tick(self._last_engine_tick)
        if cur_dr > self._peak_discovery_rate:
            self._peak_discovery_rate = cur_dr

        h_W     = self._cached_h_W_abs()
        tb_info = self.temporal.slow_state_summary()
        phi_raw = self.phi_approx()
        fallen_penalty = 0.0
        try:
            nodes = self.graph.nodes
            ps_ph = float(
                nodes.get(
                    "posture_stability",
                    nodes.get("phys_posture_stability", 0.5),
                )
            )
            cz_ph = float(nodes.get("com_z", nodes.get("phys_com_z", 0.5)))
            fn_f = getattr(self.env, "is_fallen", None)
            env_fallen = bool(fn_f()) if callable(fn_f) else False
            if env_fallen or ps_ph < 0.42 or cz_ph < 0.38:
                fallen_penalty = 0.5
        except Exception:
            pass
        phi_eff = phi_raw * (1.0 - fallen_penalty)
        behavioral_score = None
        sim_ref = self._resolve_rkk_sim()
        if sim_ref is not None:
            fn = getattr(sim_ref, "_behavioral_snapshot_cached", None)
            if callable(fn):
                bs = fn()
                if bs:
                    behavioral_score = bs.get("behavioral_score")
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

        h_W_edge_entropy = getattr(self, "_h_W_edge_entropy_cache", None)
        core = self.graph._core
        try:
            alpha_every = max(1, int(os.environ.get("RKK_SNAPSHOT_ALPHA_EVERY", "40")))
        except ValueError:
            alpha_every = 40
        from engine.graph_perf import is_large_graph

        if core is not None and not is_large_graph(self.graph) and (
            h_W_edge_entropy is None
            or self._last_engine_tick % alpha_every == 0
        ):
            with torch.no_grad():
                A = core.alpha_trust_matrix().detach().float().cpu().numpy()
            p = np.clip(A, 1e-7, 1.0 - 1e-7)
            h_W_edge_entropy = float(-(p * np.log(p) + (1.0 - p) * np.log(1.0 - p)).sum())
            self._h_W_edge_entropy_cache = h_W_edge_entropy

        _lr = getattr(self, "_last_result", None) or {}
        snap: dict = {
            "id":                    self.id,
            "name":                  self.name,
            "env_type":              self.env.preset,
            "activation":            self.activation,
            "graph_mdl":             round(self.graph.mdl_size, 3),
            "compression_gain":      round(self.compression_gain, 4),
            "alpha_mean":            round(self.graph.alpha_mean, 3),
            "post_fr_wm_lr_active": bool(
                float(getattr(self.graph, "_post_fr_wm_lr_mult", 1.0)) > 1.001
            ),
            "post_fr_wm_lr_mult": round(
                float(getattr(self.graph, "_post_fr_wm_lr_mult", 1.0)), 3
            ),
            "phi":                   round(phi_eff, 4),
            "phi_raw":               round(phi_raw, 4),
            "node_count":            len(self.graph.nodes),
            "total_interventions":   self._total_interventions,
            "total_blocked":         self._total_blocked,
            "last_do":               self._last_do,
            "from_system2":          bool(_lr.get("from_system2", False)),
            "last_blocked_reason":   self._last_blocked_reason,
            "discovery_rate":        round(cur_dr, 3),
            "structural_discovery":  round(cur_dr, 3),
            "behavioral_score":      behavioral_score,
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
            "progressive_scope": (
                {
                    **self._prog_scope.snapshot(),
                    "mastery_quality": round(self._prog_scope.mastery_quality, 3),
                }
                if getattr(self, "_prog_scope", None)
                else None
            ),
            "trajectory": {
                "enabled": True,
                "segments": len(self.graph._traj_segments) if hasattr(self.graph, "_traj_segments") else 0,
            },
        }
        if eval_mode_enabled():
            snap["edge_count"] = int(self.graph.edge_count)
            snap["edges"] = []
        else:
            el_ec, el_list = self._snapshot_edges_payload()
            snap["edge_count"] = el_ec
            snap["edges"] = el_list
        if self.env.preset == "pybullet":
            pos_fn = getattr(self.env, "object_positions_world", None)
            if callable(pos_fn):
                snap["physics_objects"] = pos_fn()
        core = self.graph.get_world_model_core()
        if core is not None:
            snap["wm"] = {
                "mechanism_hidden": int(getattr(core, "hidden", 24)),
                "type": "gnn",
            }
        ens = getattr(self.graph, "_ensemble", None)
        if ens is not None:
            snap["graph_ensemble"] = ens.snapshot()
        try:
            snap.update(self.graph.discovery_snapshot_fields())
        except Exception:
            pass
        wmeta = getattr(self, "_w_meta", None)
        if wmeta is not None:
            try:
                snap["w_meta"] = wmeta.snapshot()
                snap["meta_prediction_error"] = wmeta.meta_prediction_error_rolling(500)
            except Exception:
                pass
        snap["replay_buffer_len"] = len(getattr(self, "_replay_buffer", []))
        try:
            role_types = self.graph.role_type_map()
            if role_types:
                snap["meta"] = {"role_types": role_types}
        except Exception:
            pass
        return snap