from __future__ import annotations

import bisect
import json
import logging
import os
import random
import uuid
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import Any

from engine.goal_planning import resolve_humanoid_base
from engine.system2.learned_student import LearnedMacroStudent, snapshot_obs_for_distill
from engine.system2.macros import macro_bundle
from engine.system2.schema import (
    EpisodeSuccessSpec,
    System2Proposal,
    merge_episode_success_specs,
)
from engine.system2.success_predicates import (
    build_s2_detector_id,
    curriculum_stage_to_spec,
    episode_success_with_pe_fallback,
    should_attach_curriculum_pe_spec,
)
from engine.system2.student import MacroStudent, choose_macro_from_obs
from engine.system2.teacher import (
    build_compact_recovery_state,
    collect_s2_planning_frames,
    enrich_summary_for_s2_llm,
    llm_plan_worker,
    llm_teacher_enabled,
    proposal_from_llm_cache_only,
    recovery_llm_enabled,
    recovery_steps_from_llm_network_fetch,
    s2_recovery_vision_enabled,
    s2_unified_slots_enabled,
    s2_unified_vlm_enabled,
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


def _pe_distill_extra(
    pe_diag: dict[str, Any], spec: EpisodeSuccessSpec
) -> dict[str, Any]:
    """JSONL fields for Wave-2 PE / homeostasis (optional)."""
    out: dict[str, Any] = {}
    for k in ("pe_total", "max_pe", "homeo_veto", "veto_reason", "wave1", "reason"):
        if k in pe_diag and pe_diag[k] is not None:
            out[k] = pe_diag[k]
    if spec.expected_state:
        out["expected_state"] = {str(k): float(v) for k, v in spec.expected_state.items()}
    if spec.skill_id:
        out["skill_id"] = spec.skill_id
    return out


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


def _s2_build_llm_plan_job(
    *,
    agent: Any,
    graph: Any,
    sim: Any | None,
    obs_f: dict[str, float],
    base: Any,
    summary: dict[str, Any],
    fallen: bool,
    node_keys: frozenset[str],
) -> dict[str, Any]:
    job: dict[str, Any] = {
        "summary": dict(summary),
        "vlm_slots": vlm_slots_from_sim(sim),
    }
    if not s2_unified_vlm_enabled() or sim is None or not getattr(sim, "_visual_mode", False):
        return job
    from engine.slot_lexicon import allowed_target_ids

    job["summary"] = enrich_summary_for_s2_llm(base, obs_f, dict(summary), fallen=fallen)
    imgs, cap = collect_s2_planning_frames(agent, fallen=fallen)
    vis = getattr(sim, "_visual_env", None)
    if not imgs and vis is not None:
        try:
            snap = vis.get_slot_visualization()
        except Exception:
            snap = {}
        fb = snap.get("frame") if isinstance(snap, dict) else None
        if fb:
            imgs = [str(fb)]
            cap = "Cached visual-environment camera frame (same tick as slot masks)."
    if not imgs and not s2_unified_slots_enabled():
        return job

    job["unified_vlm"] = True
    job["images"] = imgs
    job["image_caption"] = cap
    job["include_slots"] = s2_unified_slots_enabled()
    if vis is not None and job["include_slots"]:
        n = int(getattr(vis, "n_slots", 0) or 0)
        job["slot_ids"] = [f"slot_{k}" for k in range(max(0, n))]
        job["variable_ids"] = [str(k) for k in sorted(node_keys)]
        allowed = allowed_target_ids(job["variable_ids"])
        alist = sorted(allowed)
        job["allowed_csv"] = ", ".join(alist[:80]) + (" ..." if len(alist) > 80 else "")
        try:
            snap = vis.get_slot_visualization()
            job["masks_b64"] = list((snap or {}).get("masks") or [])
        except Exception:
            job["masks_b64"] = []
    return job


def _s2_apply_unified_slot_lexicon(
    sim: Any | None,
    *,
    labels: dict[str, dict[str, Any]] | None,
    sim_tick: int,
    frame_b64: str | None,
) -> None:
    if not labels or sim is None:
        return
    vis = getattr(sim, "_visual_env", None)
    if vis is None:
        return
    try:
        vis.set_slot_lexicon(labels, sim_tick, frame_b64)
    except Exception:
        return
    sync = getattr(sim, "_phase_m_sync_from_vision", None)
    if callable(sync):
        try:
            sync()
        except Exception:
            pass


def _s2_recovery_exo_images(agent: Any) -> list[str] | None:
    if not s2_recovery_vision_enabled():
        return None
    fn = getattr(getattr(agent, "env", None), "get_frame_base64", None)
    if not callable(fn):
        return None
    try:
        b = fn("exo")
        return [str(b)] if b else None
    except Exception:
        return None


def _s2_override_enabled() -> bool:
    return os.environ.get("RKK_S2_OVERRIDE", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


def _s2_override_fallen_ticks_need() -> int:
    try:
        return max(1, int(os.environ.get("RKK_S2_OVERRIDE_FALLEN_TICKS", "8")))
    except ValueError:
        return 8


def _s2_override_max_ticks() -> int:
    try:
        return max(8, int(os.environ.get("RKK_S2_OVERRIDE_MAX_TICKS", "420")))
    except ValueError:
        return 420


def _recovery_llm_calls_max() -> int:
    """Максимум вызовов recovery LLM за одну сессию fallen_override (включая первый)."""
    try:
        return max(1, int(os.environ.get("RKK_S2_RECOVERY_LLM_CALLS_MAX", "14")))
    except ValueError:
        return 14


def _recovery_replan_min_interval_ticks() -> int:
    try:
        return max(4, int(os.environ.get("RKK_S2_RECOVERY_REPLAN_MIN_TICKS", "48")))
    except ValueError:
        return 48


def _recovery_replan_after_last_step_ticks() -> int:
    try:
        return max(0, int(os.environ.get("RKK_S2_RECOVERY_REPLAN_AFTER_LAST_STEP_TICKS", "12")))
    except ValueError:
        return 12


def _recovery_replan_empty_schedule_ticks() -> int:
    """Если шаги так и не пришли (или пустой JSON), повторить запрос не раньше этого age сессии."""
    try:
        return max(8, int(os.environ.get("RKK_S2_RECOVERY_REPLAN_EMPTY_SCHEDULE_TICKS", "40")))
    except ValueError:
        return 40


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
        self._macro_episode_spec: EpisodeSuccessSpec = EpisodeSuccessSpec()
        self._recovery_episode_spec: EpisodeSuccessSpec = EpisodeSuccessSpec()
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
        self._llm_slot_lexicon_frame: str | None = None
        self._s2_override_active: bool = False
        self._s2_override_start_tick: int = -1
        self._s2_fallen_streak_override: int = 0
        self._recovery_steps: list[dict[str, Any]] = []
        self._recovery_cumulative: list[int] = []
        self._recovery_llm_future: Future[Any] | None = None
        self._override_start_obs_f: dict[str, float] = {}
        self._recovery_schedule_anchor_tick: int = -1
        self._recovery_llm_dispatch_tick: int = -1
        self._recovery_llm_dispatch_count: int = 0

    def ollama_busy(self) -> bool:
        """True, если к Ollama уходит или ждётся ответ (план S2 или recovery)."""
        plan = self._llm_future is not None and not self._llm_future.done()
        rec = (
            self._recovery_llm_future is not None
            and not self._recovery_llm_future.done()
        )
        return plan or rec

    def _obs_floats(self, obs: dict[str, Any]) -> dict[str, float]:
        out: dict[str, float] = {}
        for k, v in obs.items():
            try:
                out[str(k)] = float(v)
            except (TypeError, ValueError):
                continue
        return out

    def _macro_outcome_deferred(self, sim_tick: int) -> bool:
        """Номинальный горизонт вышел, но исход макроса не считаем — ждём async LLM."""
        if not self._macro_start_obs or self._macro_until_tick < 0:
            return False
        if sim_tick < self._macro_until_tick:
            return False
        fut = self._llm_future
        return fut is not None and not fut.done()

    def _macro_horizon_expired(self, sim_tick: int) -> bool:
        ut = self._macro_until_tick
        if ut < 0 or sim_tick <= ut:
            return False
        if self._macro_outcome_deferred(sim_tick):
            return False
        return not self._macro_start_obs

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
        obs0: dict[str, float] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        if not _distill_enabled():
            return
        path = _distill_log_path()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            row: dict[str, Any] = {
                "tick": tick,
                "macro": macro,
                "source": source,
                "success": success,
                "delta": delta,
            }
            if obs0:
                row["obs0"] = obs0
            if extra:
                for k, v in extra.items():
                    if v is not None:
                        row[k] = v
            line = json.dumps(row, ensure_ascii=False)
            with open(path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except OSError:
            pass

    def _neuro_streak_key(self, macro: str, episode_spec: EpisodeSuccessSpec | None) -> str:
        spec = episode_spec or EpisodeSuccessSpec()
        if spec.expected_state:
            return build_s2_detector_id(macro, spec.skill_id, spec.expected_state)
        return macro

    def _maybe_materialize_macro_concept(
        self,
        *,
        sim_tick: int,
        agent: Any,
        macro: str,
        success: bool,
        episode_spec: EpisodeSuccessSpec | None = None,
    ) -> str | None:
        if not _neuro_enabled() or not success or macro == "IDLE":
            return None
        spec_e = episode_spec or EpisodeSuccessSpec()
        streak_key = self._neuro_streak_key(macro, spec_e)
        need = _neuro_streak_need()
        if self._neuro_streak.get(streak_key, 0) < need:
            return None
        if sim_tick - self._last_neuro_tick < _neuro_cooldown_ticks():
            return None
        graph = agent.graph
        max_n = _neuro_max_nodes()
        if max_n <= 0 or _s2_concept_count(graph) >= max_n:
            self._neuro_streak[streak_key] = 0
            return None
        cands = _MACRO_MEMBER_CANDIDATES.get(macro, ("posture_stability", "com_z"))
        graph_nodes = getattr(graph, "nodes", {}) or {}
        es = spec_e.expected_state
        if es:
            detector_id = build_s2_detector_id(macro, spec_e.skill_id, es)
            extra_members = [k for k in sorted(es.keys()) if k in graph_nodes][:12]
            base_mems = [m for m in cands if m in graph_nodes]
            mems = list(dict.fromkeys(base_mems + extra_members))
            pattern = [macro] + sorted(k for k in es if k in graph_nodes)[:16]
        else:
            detector_id = f"system2:{macro}"
            mems = [m for m in cands if m in graph_nodes]
            pattern = [macro]
        if len(mems) < 2:
            self._neuro_streak[streak_key] = 0
            return None
        suf = uuid.uuid4().hex[:6]
        node_id = f"concept_s2_{macro}_{suf}"
        try:
            ok = bool(
                graph.materialize_concept_macro(
                    node_id,
                    mems,
                    detector_id=detector_id,
                    pattern=pattern,
                )
            )
        except Exception:
            ok = False
        if ok:
            self._last_neuro_tick = sim_tick
            self._neuro_streak[streak_key] = 0
            return node_id
        self._neuro_streak[streak_key] = max(0, self._neuro_streak.get(streak_key, 0) - 3)
        return None

    def _clear_override_session(self) -> None:
        self._s2_override_active = False
        self._s2_override_start_tick = -1
        self._recovery_steps = []
        self._recovery_cumulative = []
        self._recovery_llm_future = None
        self._override_start_obs_f = {}
        self._recovery_episode_spec = EpisodeSuccessSpec()
        self._recovery_schedule_anchor_tick = -1
        self._recovery_llm_dispatch_tick = -1
        self._recovery_llm_dispatch_count = 0

    def _ingest_recovery_steps(
        self, steps: list[dict[str, Any]] | None, *, sim_tick: int
    ) -> None:
        self._recovery_steps = list(steps or [])
        acc = 0
        cums: list[int] = []
        for s in self._recovery_steps:
            acc += int(max(1, s.get("ticks", 1)))
            cums.append(acc)
        self._recovery_cumulative = cums
        self._recovery_schedule_anchor_tick = int(sim_tick)

    def _recovery_extra_residuals(self, sim_tick: int) -> dict[str, float]:
        if not self._recovery_steps or not self._recovery_cumulative:
            return {}
        anchor = int(self._recovery_schedule_anchor_tick)
        if anchor < 0:
            anchor = int(self._s2_override_start_tick)
        rel = max(0, int(sim_tick) - anchor)
        idx = bisect.bisect_right(self._recovery_cumulative, rel)
        idx = min(idx, len(self._recovery_steps) - 1)
        d = self._recovery_steps[idx].get("intent_deltas") or {}
        return dict(d) if isinstance(d, dict) else {}

    def _dispatch_recovery_llm_once(
        self,
        sim_tick: int,
        agent: Any,
        base: Any,
        obs_f: dict[str, float],
        sim: Any | None,
    ) -> None:
        if not recovery_llm_enabled():
            return
        compact = build_compact_recovery_state(base, obs_f)
        exo_imgs = _s2_recovery_exo_images(agent)
        vlm = vlm_slots_from_sim(sim)
        if _llm_sync_forced():
            sr = recovery_steps_from_llm_network_fetch(
                compact,
                vlm,
                images_b64=exo_imgs,
            )
            if isinstance(sr, tuple) and len(sr) == 3:
                st_l, es_r, mx_r = sr
                self._ingest_recovery_steps(st_l, sim_tick=sim_tick)
                mx_f: float | None = None
                if mx_r is not None:
                    try:
                        mx_f = float(max(0.02, min(6.0, float(mx_r))))
                    except (TypeError, ValueError):
                        mx_f = None
                self._recovery_episode_spec = EpisodeSuccessSpec(
                    expected_state=dict(es_r or {}),
                    max_prediction_error=mx_f,
                    skill_id="recovery_llm",
                )
        else:
            self._recovery_llm_future = _system2_llm_executor().submit(
                partial(
                    recovery_steps_from_llm_network_fetch,
                    images_b64=exo_imgs,
                ),
                compact,
                vlm,
            )
        self._recovery_llm_dispatch_tick = int(sim_tick)
        self._recovery_llm_dispatch_count = int(self._recovery_llm_dispatch_count) + 1

    def _maybe_dispatch_recovery_llm_replan(
        self,
        sim_tick: int,
        agent: Any,
        base: Any,
        obs_f: dict[str, float],
        sim: Any | None,
    ) -> None:
        if not recovery_llm_enabled():
            return
        if self._recovery_llm_future is not None:
            return
        if self._recovery_llm_dispatch_count >= _recovery_llm_calls_max():
            return
        dt = int(self._recovery_llm_dispatch_tick)
        if dt >= 0 and (int(sim_tick) - dt) < _recovery_replan_min_interval_ticks():
            return
        rel_session = int(sim_tick) - int(self._s2_override_start_tick)
        anc = int(self._recovery_schedule_anchor_tick)
        if anc < 0:
            anc = int(self._s2_override_start_tick)
        rel_sched = max(0, int(sim_tick) - anc)
        if self._recovery_cumulative:
            last = int(self._recovery_cumulative[-1])
            need = rel_sched >= last + _recovery_replan_after_last_step_ticks()
        else:
            need = rel_session >= _recovery_replan_empty_schedule_ticks()
        if not need:
            return
        self._dispatch_recovery_llm_once(sim_tick, agent, base, obs_f, sim)

    def _override_episode_eval(
        self, obs_f: dict[str, float], *, fallen: bool
    ) -> tuple[bool, dict[str, Any]]:
        if fallen:
            return False, {"fallen": True}
        return episode_success_with_pe_fallback(
            self._override_start_obs_f,
            obs_f,
            self._recovery_episode_spec,
            macro="RECOVER_POSTURE",
        )

    def _record_override_distill_neuro(
        self,
        *,
        sim_tick: int,
        agent: Any,
        obs_f: dict[str, float],
        success: bool,
        source_note: str,
        pe_diag: dict[str, Any] | None = None,
    ) -> None:
        macro = "RECOVER_POSTURE"
        cz0 = float(
            self._override_start_obs_f.get(
                "com_z", self._override_start_obs_f.get("phys_com_z", 0.5)
            )
        )
        cz1 = float(obs_f.get("com_z", obs_f.get("phys_com_z", cz0)))
        ps0 = float(
            self._override_start_obs_f.get(
                "posture_stability",
                self._override_start_obs_f.get("phys_posture_stability", 0.5),
            )
        )
        ps1 = float(
            obs_f.get(
                "posture_stability",
                obs_f.get("phys_posture_stability", ps0),
            )
        )
        self._student.record_outcome(macro, success, weight=1.0)
        if self._learned.enabled():
            self._learned.learn(
                macro,
                success,
                dict(self._override_start_obs_f),
                d_com_z=cz1 - cz0,
                d_posture=ps1 - ps0,
            )
        ex = _pe_distill_extra(pe_diag or {}, self._recovery_episode_spec)
        self._append_distill(
            tick=sim_tick,
            macro=macro,
            source=f"fallen_override:{source_note}",
            success=success,
            delta={"d_com_z": round(cz1 - cz0, 5), "d_posture": round(ps1 - ps0, 5)},
            obs0=snapshot_obs_for_distill(dict(self._override_start_obs_f)) or None,
            extra=ex,
        )
        sk = self._neuro_streak_key(macro, self._recovery_episode_spec)
        if success:
            self._neuro_streak[sk] = self._neuro_streak.get(sk, 0) + 1
        else:
            self._neuro_streak[sk] = 0
        neuro_new = self._maybe_materialize_macro_concept(
            sim_tick=sim_tick,
            agent=agent,
            macro=macro,
            success=success,
            episode_spec=self._recovery_episode_spec,
        )
        if neuro_new:
            self._last_neuro_node = neuro_new

    def _apply_recover_bundle_no_candidate(
        self,
        sim_tick: int,
        agent: Any,
        base: Any,
        graph: Any,
        node_keys: frozenset[str],
        extra_residuals: dict[str, float] | None,
    ) -> bool:
        macro = "RECOVER_POSTURE"
        bundle = macro_bundle(macro)
        graph_patch: dict[str, float] = dict(bundle.get("graph") or {})
        residuals: dict[str, float] = dict(bundle.get("residuals") or {})
        if extra_residuals:
            for k, v in extra_residuals.items():
                sk = str(k)
                if sk in node_keys:
                    residuals[sk] = residuals.get(sk, 0.0) + float(v)
        nodes = agent.graph.nodes
        for k, v in graph_patch.items():
            if k in nodes:
                nodes[k] = float(max(0.05, min(0.95, float(v))))
        self._sync_self_graph_to_env(base, agent.graph)
        agent.set_system2_candidate(None)
        fn = getattr(base, "apply_motor_intent_residuals", None)
        if not callable(fn) or not residuals:
            return False
        scale = _residual_scale()
        scaled = {k: float(v) * scale for k, v in residuals.items() if k in node_keys}
        if not scaled:
            return False
        try:
            fn(scaled)
            self._last_residual_tick = sim_tick
            self._prev_residual_macro = macro
            return True
        except Exception:
            return False

    def _force_reset_stance_base(self, base: Any) -> None:
        fn = getattr(base, "reset_stance", None)
        if callable(fn):
            try:
                fn()
            except Exception:
                pass

    def _drain_completed_llm_future(
        self,
        sim_tick: int,
        agent: Any,
        obs_f: dict[str, float],
        base: Any,
        graph: Any,
        node_keys: frozenset[str],
        sim: Any | None,
    ) -> dict[str, Any] | None:
        if self._llm_future is not None and self._llm_future.done():
            drained_llm: System2Proposal | None
            slot_labels: dict[str, dict[str, Any]] | None = None
            try:
                pack = self._llm_future.result(timeout=0)
            except Exception as ex:
                pack = None
                drained_llm = None
                logging.getLogger(__name__).warning(
                    "System2 LLM future failed: %s", ex, exc_info=True
                )
            else:
                if isinstance(pack, tuple) and len(pack) >= 1:
                    drained_llm = pack[0]  # type: ignore[assignment]
                    slot_labels = pack[1] if len(pack) >= 2 else None
                else:
                    drained_llm = pack  # type: ignore[assignment]
            self._llm_future = None
            fb = self._llm_slot_lexicon_frame
            self._llm_slot_lexicon_frame = None
            _s2_apply_unified_slot_lexicon(
                sim, labels=slot_labels, sim_tick=sim_tick, frame_b64=fb
            )
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
                plan_tick_bump=False,
            )
        return None

    def _build_override_diag(
        self,
        sim_tick: int,
        fallen: bool,
        age: int,
        max_age: int,
        *,
        recovered: bool = False,
        max_reset: bool = False,
        applied: bool | None = None,
    ) -> dict[str, Any]:
        inflight = (
            self._recovery_llm_future is not None
            and not self._recovery_llm_future.done()
        )
        base_diag: dict[str, Any] = {
            "enabled": True,
            "fallen_override_active": True,
            "fallen_override_ticks": int(max(0, age)) + 1,
            "fallen_override_max_ticks": int(max_age),
            "s2_fallen_streak": int(self._s2_fallen_streak_override),
            "override_recovered": bool(recovered),
            "override_max_reset": bool(max_reset),
            "recovery_llm_inflight": bool(inflight),
            "recovery_steps_loaded": len(self._recovery_steps),
            "recovery_llm_dispatches": int(self._recovery_llm_dispatch_count),
            "macro": "RECOVER_POSTURE",
            "source": "fallen_override",
            "sim_tick": sim_tick,
            "fallen": bool(fallen),
            "blocked": False,
            "neuro_streak": dict(self._neuro_streak),
            "outcome_ema": round(self._outcome_ema, 4),
        }
        if applied is not None:
            base_diag["residuals_applied"] = applied
        return base_diag

    def _maybe_tick_fallen_override(
        self,
        sim_tick: int,
        fallen: bool,
        agent: Any,
        obs_f: dict[str, float],
        base: Any,
        graph: Any,
        node_keys: frozenset[str],
        sim: Any | None,
    ) -> dict[str, Any] | None:
        if not _s2_override_enabled():
            return None
        if fallen:
            self._s2_fallen_streak_override += 1
        else:
            self._s2_fallen_streak_override = 0

        if self._recovery_llm_future is not None and self._recovery_llm_future.done():
            pack: Any = None
            try:
                pack = self._recovery_llm_future.result(timeout=0)
            except Exception:
                pack = None
            self._recovery_llm_future = None
            steps_try: list[dict[str, Any]] | None = None
            if isinstance(pack, tuple) and len(pack) == 3:
                st_l, es_rec, mx_rec = pack
                steps_try = st_l if isinstance(st_l, list) else None
                mx_f: float | None = None
                if mx_rec is not None:
                    try:
                        mx_f = float(max(0.02, min(6.0, float(mx_rec))))
                    except (TypeError, ValueError):
                        mx_f = None
                self._recovery_episode_spec = EpisodeSuccessSpec(
                    expected_state=dict(es_rec or {}),
                    max_prediction_error=mx_f,
                    skill_id="recovery_llm",
                )
            elif isinstance(pack, list) and pack:
                steps_try = pack
            if steps_try:
                self._ingest_recovery_steps(steps_try, sim_tick=sim_tick)

        if not self._s2_override_active:
            if self._s2_fallen_streak_override < _s2_override_fallen_ticks_need():
                return None
            self._s2_override_active = True
            self._s2_override_start_tick = sim_tick
            self._override_start_obs_f = dict(obs_f)
            self._recovery_steps = []
            self._recovery_cumulative = []
            self._recovery_llm_future = None
            self._recovery_episode_spec = EpisodeSuccessSpec()
            self._recovery_schedule_anchor_tick = -1
            self._recovery_llm_dispatch_tick = -1
            self._recovery_llm_dispatch_count = 0
            self._dispatch_recovery_llm_once(sim_tick, agent, base, obs_f, sim)
        else:
            age = int(sim_tick) - int(self._s2_override_start_tick)
            max_age = _s2_override_max_ticks()
            if not fallen:
                ok, pe_diag = self._override_episode_eval(obs_f, fallen=False)
                self._record_override_distill_neuro(
                    sim_tick=sim_tick,
                    agent=agent,
                    obs_f=obs_f,
                    success=ok,
                    source_note="recovered",
                    pe_diag=pe_diag,
                )
                diag = self._build_override_diag(
                    sim_tick,
                    fallen,
                    age,
                    max_age,
                    recovered=True,
                )
                self._clear_override_session()
                return diag
            if age >= max_age:
                self._force_reset_stance_base(base)
                self._record_override_distill_neuro(
                    sim_tick=sim_tick,
                    agent=agent,
                    obs_f=obs_f,
                    success=False,
                    source_note="max_ticks_reset",
                    pe_diag={"max_ticks_reset": True},
                )
                diag = self._build_override_diag(
                    sim_tick,
                    fallen,
                    age,
                    max_age,
                    max_reset=True,
                )
                self._clear_override_session()
                return diag

        if self._s2_override_active and fallen:
            self._maybe_dispatch_recovery_llm_replan(
                sim_tick, agent, base, obs_f, sim
            )

        extra = self._recovery_extra_residuals(sim_tick)
        applied = self._apply_recover_bundle_no_candidate(
            sim_tick,
            agent,
            base,
            graph,
            node_keys,
            extra if extra else None,
        )
        age = int(sim_tick) - int(self._s2_override_start_tick)
        max_age = _s2_override_max_ticks()
        return self._build_override_diag(
            sim_tick,
            fallen,
            age,
            max_age,
            applied=applied,
        )

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
        *,
        plan_tick_bump: bool = True,
    ) -> dict[str, Any]:
        """Один цикл планирования. `plan_tick_bump=False` — ответ async-LLM: не сдвигать
        `_last_plan_tick` (счётчик PLAN_EVERY от момента submit, как при старте старого блокирующего запроса)."""
        if plan_tick_bump:
            self._last_plan_tick = sim_tick
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
        if callable(fn) and residuals and self._should_apply_residuals(
            sim_tick, macro, base=base, sim=sim
        ):
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
        gov = EpisodeSuccessSpec()
        if sim is not None:
            cur = getattr(sim, "_curriculum", None)
            st = getattr(cur, "current_stage", None) if cur is not None else None
            if st is not None:
                cand = curriculum_stage_to_spec(st)
                if should_attach_curriculum_pe_spec(macro, cand):
                    gov = cand
        self._macro_episode_spec = merge_episode_success_specs(
            EpisodeSuccessSpec.from_proposal(proposal_effective), gov
        )
        self._macro_start_obs = dict(obs_f)

        inflight = self._llm_future is not None and not self._llm_future.done()
        hz_exp = self._macro_horizon_expired(sim_tick)
        defer = self._macro_outcome_deferred(sim_tick)
        wt = None
        if inflight and self._llm_submit_tick >= 0:
            wt = int(sim_tick - self._llm_submit_tick)
        self._last_diag = {
            "enabled": True,
            "macro": macro,
            "source": source,
            "until": self._macro_until_tick,
            "sim_tick": sim_tick,
            "macro_horizon_expired": hz_exp,
            "macro_outcome_deferred": defer,
            "llm_wait_ticks": wt,
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

    def _should_apply_residuals(
        self,
        sim_tick: int,
        macro: str,
        *,
        base: Any | None = None,
        sim: Any | None = None,
    ) -> bool:
        if os.environ.get("RKK_SYSTEM2_RESIDUAL_CPG_GUARD", "1").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        ):
            if base is not None and bool(getattr(base, "cpg_owns_legs", False)):
                fnr = getattr(sim, "_locomotion_reward_ema", None) if sim is not None else None
                if callable(fnr):
                    try:
                        thr = float(
                            os.environ.get("RKK_SYSTEM2_RESIDUAL_MIN_LOCOREWARD", "0.22")
                        )
                    except ValueError:
                        thr = 0.22
                    thr = float(max(0.05, min(0.55, thr)))
                    try:
                        if float(fnr()) < thr:
                            return False
                    except Exception:
                        pass
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
        fallen: bool = False,
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

        ov = self._maybe_tick_fallen_override(
            sim_tick, fallen, agent, obs_f, base, graph, node_keys, sim
        )
        if ov is not None:
            self._last_diag = ov
            return self._last_diag

        drained_early = self._drain_completed_llm_future(
            sim_tick, agent, obs_f, base, graph, node_keys, sim
        )
        if drained_early is not None:
            self._last_diag = drained_early
            return self._last_diag

        ending_macro = self._active_macro
        ending_source = self._last_source
        # Завершение макроса: оценка прогресса для студента + лог дистилляции
        # Пока LLM в полёте — не закрывать эпизод по календарю: окно отсчитывается до применения ответа.
        if (
            not self._s2_override_active
            and sim_tick >= self._macro_until_tick
            and self._macro_start_obs
            and not (
                self._llm_future is not None and not self._llm_future.done()
            )
        ):
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
            success, pe_diag = episode_success_with_pe_fallback(
                self._macro_start_obs,
                obs_f,
                self._macro_episode_spec,
                macro=ending_macro,
            )
            self._student.record_outcome(ending_macro, success, weight=1.0)
            self._outcome_ema = float(
                max(0.0, min(1.0, 0.92 * self._outcome_ema + 0.08 * (1.0 if success else 0.0)))
            )
            if self._learned.enabled():
                self._learned.learn(
                    ending_macro,
                    success,
                    dict(self._macro_start_obs),
                    d_com_z=cz1 - cz0,
                    d_posture=ps1 - ps0,
                )
            distill_x = _pe_distill_extra(pe_diag, self._macro_episode_spec)
            self._append_distill(
                tick=sim_tick,
                macro=ending_macro,
                source=ending_source,
                success=success,
                delta={"d_com_z": round(cz1 - cz0, 5), "d_posture": round(ps1 - ps0, 5)},
                obs0=snapshot_obs_for_distill(dict(self._macro_start_obs)) or None,
                extra=distill_x,
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
            sk = self._neuro_streak_key(ending_macro, self._macro_episode_spec)
            if success:
                self._neuro_streak[sk] = self._neuro_streak.get(sk, 0) + 1
            else:
                self._neuro_streak[sk] = 0
            neuro_new = self._maybe_materialize_macro_concept(
                sim_tick=sim_tick,
                agent=agent,
                macro=ending_macro,
                success=success,
                episode_spec=self._macro_episode_spec,
            )
            if neuro_new:
                self._last_neuro_node = neuro_new
            self._macro_start_obs = {}
            self._macro_episode_spec = EpisodeSuccessSpec()

        plan_every = _plan_every_ticks()
        should_plan = (sim_tick - self._last_plan_tick) >= plan_every
        if sim_tick >= self._macro_until_tick:
            should_plan = True

        if not should_plan:
            inflight = self._llm_future is not None and not self._llm_future.done()
            hz_exp = self._macro_horizon_expired(sim_tick)
            defer = self._macro_outcome_deferred(sim_tick)
            wt = None
            if inflight and self._llm_submit_tick >= 0:
                wt = int(sim_tick - self._llm_submit_tick)
            self._last_diag = {
                "enabled": True,
                "macro": self._active_macro,
                "until": self._macro_until_tick,
                "sim_tick": sim_tick,
                "macro_horizon_expired": hz_exp,
                "macro_outcome_deferred": defer,
                "llm_wait_ticks": wt,
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
                hz_exp = self._macro_horizon_expired(sim_tick)
                defer = self._macro_outcome_deferred(sim_tick)
                wt = (
                    int(sim_tick - self._llm_submit_tick)
                    if self._llm_submit_tick >= 0
                    else None
                )
                self._last_diag = {
                    "enabled": True,
                    "macro": self._active_macro,
                    "until": self._macro_until_tick,
                    "sim_tick": sim_tick,
                    "macro_horizon_expired": hz_exp,
                    "macro_outcome_deferred": defer,
                    "llm_wait_ticks": wt,
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
            job = _s2_build_llm_plan_job(
                agent=agent,
                graph=graph,
                sim=sim,
                obs_f=obs_f,
                base=base,
                summary=dict(summary),
                fallen=fallen,
                node_keys=node_keys,
            )
            self._llm_slot_lexicon_frame = (
                (job.get("images") or [None])[0] if job.get("unified_vlm") else None
            )
            self._llm_future = _system2_llm_executor().submit(llm_plan_worker, job)
            self._llm_submit_tick = sim_tick
            self._last_plan_tick = sim_tick
            hz_exp = self._macro_horizon_expired(sim_tick)
            defer = self._macro_outcome_deferred(sim_tick)
            self._last_diag = {
                "enabled": True,
                "macro": self._active_macro,
                "until": self._macro_until_tick,
                "sim_tick": sim_tick,
                "macro_horizon_expired": hz_exp,
                "macro_outcome_deferred": defer,
                "llm_wait_ticks": 0,
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
            job = _s2_build_llm_plan_job(
                agent=agent,
                graph=graph,
                sim=sim,
                obs_f=obs_f,
                base=base,
                summary=dict(summary),
                fallen=fallen,
                node_keys=node_keys,
            )
            self._llm_slot_lexicon_frame = (
                (job.get("images") or [None])[0] if job.get("unified_vlm") else None
            )
            prop_try, slot_try = llm_plan_worker(job)
            self._llm_slot_lexicon_frame = None
            _s2_apply_unified_slot_lexicon(
                sim, labels=slot_try, sim_tick=sim_tick, frame_b64=(job.get("images") or [None])[0]
            )
            proposal_llm = prop_try
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
