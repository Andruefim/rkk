from __future__ import annotations

import json
import os
import random
import uuid
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

from engine.goal_planning import resolve_humanoid_base
from engine.system2.learned_student import LearnedMacroStudent
from engine.system2.macros import macro_bundle
from engine.system2.schema import System2Proposal
from engine.system2.student import MacroStudent, choose_macro_from_obs
from engine.system2.teacher import (
    llm_teacher_enabled,
    proposal_from_llm,
    proposal_from_llm_cache_only,
    proposal_from_llm_network_fetch,
    vlm_slots_from_sim,
)
from engine.system2.validate import validate_proposal

# Must match engine.features.humanoid.constants.SELF_VARS (avoid importing humanoid package here).
_SELF_SET = frozenset(
    (
        "self_intention_larm",
        "self_intention_rarm",
        "self_energy",
        "self_attention",
        "self_com_z_target",
        "self_posture_target",
        "self_goal_target_dist",
        "self_goal_active",
    )
)

_MACRO_MEMBER_CANDIDATES: dict[str, tuple[str, ...]] = {
    "RECOVER_POSTURE": (
        "posture_stability",
        "com_z",
        "intent_stop_recover",
        "intent_torso_forward",
        "intent_stride",
    ),
    "LOCOMOTE_DELIVERY": (
        "target_dist",
        "intent_stride",
        "posture_stability",
        "com_z",
        "intent_gait_coupling",
    ),
    "EXPLORE": (
        "com_x",
        "posture_stability",
        "intent_torso_forward",
        "intent_arm_counterbalance",
    ),
    "IDLE": ("posture_stability", "com_z"),
}


def system2_enabled() -> bool:
    return os.environ.get("RKK_SYSTEM2", "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _plan_every_ticks() -> int:
    try:
        return max(8, int(os.environ.get("RKK_SYSTEM2_PLAN_EVERY", "48")))
    except ValueError:
        return 48


def _macro_horizon_ticks() -> int:
    try:
        return max(4, int(os.environ.get("RKK_SYSTEM2_MACRO_TICKS", "36")))
    except ValueError:
        return 36


def _residual_scale() -> float:
    try:
        return float(os.environ.get("RKK_SYSTEM2_RESIDUAL_SCALE", "1.0"))
    except ValueError:
        return 1.0


def _distill_log_path() -> Path:
    raw = os.environ.get("RKK_SYSTEM2_DISTILL_LOG", "logs/system2_distill.jsonl")
    return Path(raw)


def _distill_enabled() -> bool:
    return os.environ.get("RKK_SYSTEM2_DISTILL", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


def _neuro_enabled() -> bool:
    return os.environ.get("RKK_SYSTEM2_NEURO", "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _neuro_streak_need() -> int:
    try:
        return max(3, int(os.environ.get("RKK_SYSTEM2_NEURO_STREAK", "8")))
    except ValueError:
        return 8


def _neuro_cooldown_ticks() -> int:
    try:
        return max(120, int(os.environ.get("RKK_SYSTEM2_NEURO_COOLDOWN", "2400")))
    except ValueError:
        return 2400


def _neuro_max_nodes() -> int:
    try:
        return max(0, int(os.environ.get("RKK_SYSTEM2_NEURO_MAX", "2")))
    except ValueError:
        return 2


def _residual_min_every() -> int:
    try:
        return max(1, int(os.environ.get("RKK_SYSTEM2_RESIDUAL_MIN_EVERY", "6")))
    except ValueError:
        return 6


def _residual_same_cooldown() -> int:
    try:
        return max(0, int(os.environ.get("RKK_SYSTEM2_RESIDUAL_SAME_COOLDOWN", "12")))
    except ValueError:
        return 12


def _residual_redundant_skip() -> bool:
    return os.environ.get("RKK_SYSTEM2_RESIDUAL_REDUNDANT", "1").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _s2_concept_count(graph: Any) -> int:
    ids = getattr(graph, "_node_ids", []) or []
    return sum(1 for k in ids if str(k).startswith("concept_s2_"))


_S2_LLM_EXECUTOR: ThreadPoolExecutor | None = None


def _system2_llm_executor() -> ThreadPoolExecutor:
    global _S2_LLM_EXECUTOR
    if _S2_LLM_EXECUTOR is None:
        _S2_LLM_EXECUTOR = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="rkk_s2_llm",
        )
    return _S2_LLM_EXECUTOR


def _llm_sync_forced() -> bool:
    """RKK_SYSTEM2_LLM_SYNC=1 — блокирующий Ollama в тике (отладка)."""
    return os.environ.get("RKK_SYSTEM2_LLM_SYNC", "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


class System2Controller:
    """
    Медленный контур: раз в N тиков выбирает макрос (LLM или студент),
    выставляет self_goal_* в графе, мягко сдвигает intent через residuals,
    передаёт один приоритетный кандидат в agent.step.
    """

    def __init__(self) -> None:
        self._student = MacroStudent()
        self._learned = LearnedMacroStudent()
        self._active_macro = "IDLE"
        self._macro_until_tick = -1
        self._last_plan_tick = -10**9
        self._last_source = "init"
        self._macro_start_obs: dict[str, float] = {}
        self._last_diag: dict[str, Any] = {}
        self._outcome_ema = 0.5
        self._last_residual_tick = -10**9
        self._prev_residual_macro = "IDLE"
        self._bootstrap_attempted = False
        self._neuro_streak: dict[str, int] = {}
        self._last_neuro_tick = -10**9
        self._online_buf: deque[dict[str, Any]] = deque(maxlen=384)
        self._last_neuro_node: str | None = None
        self._llm_future: Future[System2Proposal | None] | None = None
        self._llm_submit_tick: int = -1

    def _obs_floats(self, obs: dict[str, Any]) -> dict[str, float]:
        out: dict[str, float] = {}
        for k, v in obs.items():
            try:
                out[str(k)] = float(v)
            except (TypeError, ValueError):
                continue
        return out

    def _merge_llm_goal_into_graph(self, graph: Any, proposal: System2Proposal) -> None:
        g = proposal.goal
        nodes = getattr(graph, "nodes", None)
        if not isinstance(nodes, dict):
            return
        if g.com_z_min is not None:
            for key in ("self_com_z_target",):
                if key in nodes:
                    nodes[key] = float(max(0.05, min(0.95, g.com_z_min)))
        if g.posture_stability_min is not None:
            for key in ("self_posture_target",):
                if key in nodes:
                    nodes[key] = float(
                        max(0.05, min(0.95, g.posture_stability_min))
                    )
        if g.target_dist_max is not None and "self_goal_target_dist" in nodes:
            nodes["self_goal_target_dist"] = float(
                max(0.05, min(0.95, g.target_dist_max))
            )

    def _sync_self_graph_to_env(self, base: Any, graph: Any) -> None:
        fn = getattr(base, "apply_self_state_patch", None)
        if not callable(fn):
            return
        nodes = getattr(graph, "nodes", None)
        if not isinstance(nodes, dict):
            return
        patch = {k: float(nodes[k]) for k in _SELF_SET if k in nodes}
        if patch:
            try:
                fn(patch)
            except Exception:
                pass

    def _append_distill(
        self,
        *,
        tick: int,
        macro: str,
        source: str,
        success: bool,
        delta: dict[str, float],
    ) -> None:
        if not _distill_enabled():
            return
        path = _distill_log_path()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            line = json.dumps(
                {
                    "tick": tick,
                    "macro": macro,
                    "source": source,
                    "success": success,
                    "delta": delta,
                },
                ensure_ascii=False,
            )
            with open(path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except OSError:
            pass

    def _maybe_materialize_macro_concept(
        self,
        *,
        sim_tick: int,
        agent: Any,
        macro: str,
        success: bool,
    ) -> str | None:
        if not _neuro_enabled() or not success or macro == "IDLE":
            return None
        need = _neuro_streak_need()
        if self._neuro_streak.get(macro, 0) < need:
            return None
        if sim_tick - self._last_neuro_tick < _neuro_cooldown_ticks():
            return None
        graph = agent.graph
        max_n = _neuro_max_nodes()
        if max_n <= 0 or _s2_concept_count(graph) >= max_n:
            self._neuro_streak[macro] = 0
            return None
        cands = _MACRO_MEMBER_CANDIDATES.get(macro, ("posture_stability", "com_z"))
        mems = [m for m in cands if m in getattr(graph, "nodes", {})]
        if len(mems) < 2:
            self._neuro_streak[macro] = 0
            return None
        suf = uuid.uuid4().hex[:6]
        node_id = f"concept_s2_{macro}_{suf}"
        try:
            ok = bool(
                graph.materialize_concept_macro(
                    node_id,
                    mems,
                    detector_id=f"system2:{macro}",
                    pattern=[macro],
                )
            )
        except Exception:
            ok = False
        if ok:
            self._last_neuro_tick = sim_tick
            self._neuro_streak[macro] = 0
            return node_id
        self._neuro_streak[macro] = max(0, self._neuro_streak.get(macro, 0) - 3)
        return None

    def _apply_planning_step(
        self,
        sim_tick: int,
        agent: Any,
        obs_f: dict[str, float],
        base: Any,
        graph: Any,
        node_keys: frozenset[str],
        sim: Any | None,
        proposal_llm: System2Proposal | None,
    ) -> dict[str, Any]:
        """Один цикл планирования; обновляет `_last_plan_tick` только здесь (после готовности LLM)."""
        if (
            self._learned.enabled()
            and not self._bootstrap_attempted
            and os.environ.get("RKK_SYSTEM2_STUDENT_BOOTSTRAP", "0").strip().lower()
            in ("1", "true", "yes", "on")
        ):
            self._learned.bootstrap_from_log(_distill_log_path())
        self._bootstrap_attempted = True

        p_llm = proposal_llm
        if p_llm is not None:
            p_llm = validate_proposal(p_llm, allowed_intent_keys=node_keys)

        macro_heur = choose_macro_from_obs(obs_f)
        stud_conf = 0.0
        if self._learned.enabled():
            macro, stud_conf = self._learned.predict(obs_f)
            source = "student_learned"
        else:
            macro = macro_heur
            source = "student"

        proposal_effective: System2Proposal | None = None
        if p_llm is not None:
            llm_m = p_llm.normalized_macro()
            try:
                lam = float(os.environ.get("RKK_SYSTEM2_LLM_BLEND", "0.65"))
            except ValueError:
                lam = 0.65
            lam = max(0.0, min(1.0, lam))
            use_llm = False
            if llm_m != "IDLE":
                if self._learned.enabled():
                    p_use = lam + (1.0 - lam) * (1.0 - float(stud_conf))
                    p_use = max(0.0, min(1.0, p_use))
                    use_llm = random.random() < p_use
                else:
                    use_llm = random.random() < lam
            if use_llm:
                macro = llm_m
                source = "llm"
                proposal_effective = p_llm

        bundle = macro_bundle(macro)
        graph_patch: dict[str, float] = dict(bundle.get("graph") or {})
        cand_tpl = bundle.get("candidate")
        residuals: dict[str, float] = dict(bundle.get("residuals") or {})

        if proposal_effective and proposal_effective.intent_deltas:
            for k, dv in proposal_effective.intent_deltas.items():
                if k in node_keys:
                    residuals[k] = residuals.get(k, 0.0) + float(dv)

        nodes = agent.graph.nodes
        for k, v in graph_patch.items():
            if k in nodes:
                nodes[k] = float(max(0.05, min(0.95, float(v))))

        if proposal_effective:
            self._merge_llm_goal_into_graph(agent.graph, proposal_effective)

        self._sync_self_graph_to_env(base, agent.graph)

        residuals_applied = False
        fn = getattr(base, "apply_motor_intent_residuals", None)
        if callable(fn) and residuals and self._should_apply_residuals(sim_tick, macro):
            scale = _residual_scale()
            scaled = {k: float(v) * scale for k, v in residuals.items() if k in node_keys}
            if scaled:
                try:
                    fn(scaled)
                    residuals_applied = True
                    self._last_residual_tick = sim_tick
                    self._prev_residual_macro = macro
                except Exception:
                    pass

        candidate = None
        last_var = ""
        if cand_tpl and isinstance(cand_tpl, dict):
            var = str(cand_tpl.get("variable", ""))
            val = float(cand_tpl.get("value", 0.5))
            target = str(cand_tpl.get("target", "posture_stability"))
            unc = float(cand_tpl.get("uncertainty", 0.45))
            eig = float(cand_tpl.get("expected_ig", 0.75))
            eig *= 0.55 + 0.45 * self._outcome_ema
            last_var = var
            if var.startswith("intent_") or var.startswith("phys_intent_"):
                try:
                    feat = agent._features_for_intervention_pair(var, target)
                except Exception:
                    feat = []
                candidate = {
                    "variable": var,
                    "target": target,
                    "value": float(max(0.06, min(0.94, val))),
                    "uncertainty": unc,
                    "features": feat,
                    "expected_ig": float(max(0.08, min(0.98, eig))),
                    "from_system2": True,
                }

        agent.set_system2_candidate(candidate)
        self._active_macro = macro
        self._last_source = source
        self._macro_until_tick = sim_tick + _macro_horizon_ticks()
        self._macro_start_obs = dict(obs_f)
        self._last_plan_tick = sim_tick

        inflight = self._llm_future is not None and not self._llm_future.done()
        self._last_diag = {
            "enabled": True,
            "macro": macro,
            "source": source,
            "until": self._macro_until_tick,
            "has_candidate": candidate is not None,
            "last_candidate_var": last_var,
            "blocked": False,
            "outcome_ema": round(self._outcome_ema, 4),
            "student_conf": round(float(stud_conf), 4) if self._learned.enabled() else None,
            "residuals_applied": residuals_applied,
            "neuro_streak": dict(self._neuro_streak),
            "online_buf": len(self._online_buf),
            "last_neuro_node": self._last_neuro_node,
            "llm_inflight": bool(inflight),
            "llm_submit_tick": self._llm_submit_tick if inflight else None,
        }
        return self._last_diag

    def _should_apply_residuals(self, sim_tick: int, macro: str) -> bool:
        gap = sim_tick - self._last_residual_tick
        if gap < _residual_min_every():
            return False
        if _residual_redundant_skip() and macro == self._prev_residual_macro:
            if gap < _residual_same_cooldown():
                return False
        return True

    def tick(
        self,
        *,
        sim_tick: int,
        agent: Any,
        obs: dict[str, Any],
        sim: Any | None = None,
    ) -> dict[str, Any]:
        if not system2_enabled():
            self._last_diag = {"enabled": False}
            return self._last_diag

        base = resolve_humanoid_base(agent.env)
        if base is None:
            self._last_diag = {"enabled": True, "skipped": "no_humanoid_base"}
            return self._last_diag

        if getattr(base, "_fixed_root", False):
            off_fr = os.environ.get("RKK_SYSTEM2_FIXED_ROOT", "0").strip().lower()
            if off_fr not in ("1", "true", "yes", "on"):
                self._last_diag = {"enabled": True, "skipped": "fixed_root"}
                return self._last_diag

        obs_f = self._obs_floats(obs)
        graph = agent.graph
        node_keys = frozenset(getattr(graph, "_node_ids", ()) or ())

        ending_macro = self._active_macro
        ending_source = self._last_source
        # Завершение макроса: оценка прогресса для студента + лог дистилляции
        if sim_tick >= self._macro_until_tick and self._macro_start_obs:
            cz0 = float(
                self._macro_start_obs.get(
                    "com_z", self._macro_start_obs.get("phys_com_z", 0.5)
                )
            )
            cz1 = float(obs_f.get("com_z", obs_f.get("phys_com_z", cz0)))
            ps0 = float(
                self._macro_start_obs.get(
                    "posture_stability",
                    self._macro_start_obs.get("phys_posture_stability", 0.5),
                )
            )
            ps1 = float(
                obs_f.get(
                    "posture_stability",
                    obs_f.get("phys_posture_stability", ps0),
                )
            )
            success = (cz1 - cz0) > 0.018 or (ps1 - ps0) > 0.04
            self._student.record_outcome(ending_macro, success, weight=1.0)
            self._outcome_ema = float(
                max(0.0, min(1.0, 0.92 * self._outcome_ema + 0.08 * (1.0 if success else 0.0)))
            )
            if self._learned.enabled():
                self._learned.learn(ending_macro, success, dict(self._macro_start_obs))
            self._append_distill(
                tick=sim_tick,
                macro=ending_macro,
                source=ending_source,
                success=success,
                delta={"d_com_z": round(cz1 - cz0, 5), "d_posture": round(ps1 - ps0, 5)},
            )
            try:
                self._online_buf.append(
                    {
                        "tick": sim_tick,
                        "macro": ending_macro,
                        "source": ending_source,
                        "success": success,
                        "obs0": dict(self._macro_start_obs),
                        "obs1": dict(obs_f),
                    }
                )
            except Exception:
                pass
            if success:
                self._neuro_streak[ending_macro] = self._neuro_streak.get(ending_macro, 0) + 1
            else:
                self._neuro_streak[ending_macro] = 0
            neuro_new = self._maybe_materialize_macro_concept(
                sim_tick=sim_tick,
                agent=agent,
                macro=ending_macro,
                success=success,
            )
            if neuro_new:
                self._last_neuro_node = neuro_new
            self._macro_start_obs = {}

        plan_every = _plan_every_ticks()
        should_plan = (sim_tick - self._last_plan_tick) >= plan_every
        if sim_tick >= self._macro_until_tick:
            should_plan = True

        if self._llm_future is not None and self._llm_future.done():
            drained_llm: System2Proposal | None
            try:
                drained_llm = self._llm_future.result(timeout=0)
            except Exception:
                drained_llm = None
            self._llm_future = None
            if drained_llm is not None:
                drained_llm = validate_proposal(
                    drained_llm, allowed_intent_keys=node_keys
                )
            return self._apply_planning_step(
                sim_tick,
                agent,
                obs_f,
                base,
                graph,
                node_keys,
                sim,
                drained_llm,
            )

        if not should_plan:
            inflight = self._llm_future is not None and not self._llm_future.done()
            self._last_diag = {
                "enabled": True,
                "macro": self._active_macro,
                "until": self._macro_until_tick,
                "idle": True,
                "outcome_ema": round(self._outcome_ema, 4),
                "student_conf": self._last_diag.get("student_conf"),
                "last_source": self._last_source,
                "last_neuro_node": self._last_neuro_node,
                "llm_inflight": bool(inflight),
                "llm_submit_tick": self._llm_submit_tick if inflight else None,
            }
            return self._last_diag

        summary = {
            "tick": sim_tick,
            "com_z": obs_f.get("com_z", obs_f.get("phys_com_z")),
            "posture_stability": obs_f.get(
                "posture_stability", obs_f.get("phys_posture_stability")
            ),
            "target_dist": obs_f.get("target_dist", obs_f.get("phys_target_dist")),
            "foot_l": obs_f.get("foot_contact_l", obs_f.get("phys_foot_contact_l")),
            "foot_r": obs_f.get("foot_contact_r", obs_f.get("phys_foot_contact_r")),
        }

        llm_on = llm_teacher_enabled()
        if llm_on and not _llm_sync_forced():
            cached = proposal_from_llm_cache_only(summary)
            if cached is not None:
                prop = validate_proposal(
                    cached, allowed_intent_keys=node_keys
                )
                return self._apply_planning_step(
                    sim_tick,
                    agent,
                    obs_f,
                    base,
                    graph,
                    node_keys,
                    sim,
                    prop,
                )
            if self._llm_future is not None and not self._llm_future.done():
                self._last_diag = {
                    "enabled": True,
                    "macro": self._active_macro,
                    "until": self._macro_until_tick,
                    "idle": True,
                    "outcome_ema": round(self._outcome_ema, 4),
                    "student_conf": self._last_diag.get("student_conf"),
                    "last_source": self._last_source,
                    "last_neuro_node": self._last_neuro_node,
                    "llm_inflight": True,
                    "llm_submit_tick": self._llm_submit_tick,
                    "llm_submitted": False,
                }
                return self._last_diag
            vlm = vlm_slots_from_sim(sim)
            self._llm_future = _system2_llm_executor().submit(
                proposal_from_llm_network_fetch,
                dict(summary),
                vlm,
            )
            self._llm_submit_tick = sim_tick
            self._last_diag = {
                "enabled": True,
                "macro": self._active_macro,
                "until": self._macro_until_tick,
                "idle": True,
                "outcome_ema": round(self._outcome_ema, 4),
                "student_conf": self._last_diag.get("student_conf"),
                "last_source": self._last_source,
                "last_neuro_node": self._last_neuro_node,
                "llm_inflight": True,
                "llm_submit_tick": self._llm_submit_tick,
                "llm_submitted": True,
            }
            return self._last_diag

        proposal_llm: System2Proposal | None = None
        if llm_on and _llm_sync_forced():
            proposal_llm = proposal_from_llm(summary, sim=sim)
            if proposal_llm is not None:
                proposal_llm = validate_proposal(
                    proposal_llm, allowed_intent_keys=node_keys
                )

        return self._apply_planning_step(
            sim_tick,
            agent,
            obs_f,
            base,
            graph,
            node_keys,
            sim,
            proposal_llm,
        )
