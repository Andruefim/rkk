from __future__ import annotations

import bisect
import json
import logging
import os
import uuid
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

from engine.goal_planning import resolve_humanoid_base
from engine.eval_mode import curriculum_context_tags, eval_mode_enabled
from engine.system2.distill_log import (
    DistillHealthTracker,
    compress_recovery_steps,
    distill_enabled,
    distill_log_path,
    pe_distill_extra,
)
from engine.system2.recovery_library import (
    get_recovery_library,
    recovery_library_enabled,
)
from engine.system2.recovery_schedule import (
    default_recovery_fallback_steps,
    enrich_recovery_steps,
    prepare_scripted_getup_steps,
    recovery_fallback_enabled,
    recovery_scripted_enabled,
    recovery_scripted_lock_until_exhausted,
    scripted_getup_episode_spec,
    scripted_getup_phase_at,
)
from engine.system2.learned_student import (
    LearnedMacroStudent,
    WmPlannerStudent,
    snapshot_obs_for_distill,
)
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
    evaluate_override_recovery_exit,
    override_recovered_posture_ok,
    should_attach_curriculum_pe_spec,
)
from engine.system2.student import MacroStudent, choose_macro_from_obs
from engine.working_memory import WorkingMemoryBuffer, working_memory_enabled

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
    return os.environ.get("RKK_SYSTEM2", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


def ensure_sim_system2(sim: Any) -> System2Controller | None:
    """Lazy-init System2 on sim (e.g. human command before first tick)."""
    if not system2_enabled():
        return None
    s2 = getattr(sim, "_system2", None)
    if s2 is None:
        s2 = System2Controller()
        sim._system2 = s2
        s2._rkk_sim = sim
    return s2


def write_human_command_wm(sim: Any, text: str, tick: int) -> None:
    """Persist human-command slots in S2 WM (creates controller if needed)."""
    if not working_memory_enabled():
        return
    s2 = ensure_sim_system2(sim)
    if s2 is None:
        return
    snippet = str(text or "")[:120]
    wm = s2.working_memory
    wm.write(
        "human_task_active",
        1.0,
        text=snippet,
        tick=int(tick),
        source="human_command",
    )
    wm.write(
        "human_task_pe",
        1.0,
        text=snippet[:80],
        tick=int(tick),
        source="human_command",
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


def _s2_wm_override_schedule_only() -> bool:
    return os.environ.get("RKK_S2_WM_OVERRIDE_SCHEDULE_ONLY", "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _s2_override_enabled() -> bool:
    return os.environ.get("RKK_S2_OVERRIDE", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


def _s2_override_fallen_ticks_need() -> int:
    try:
        return max(1, int(os.environ.get("RKK_S2_OVERRIDE_FALLEN_TICKS", "4")))
    except ValueError:
        return 4


def _s2_override_max_ticks() -> int:
    try:
        return max(8, int(os.environ.get("RKK_S2_OVERRIDE_MAX_TICKS", "420")))
    except ValueError:
        return 420


def _s2_learned_recovery_enabled() -> bool:
    return os.environ.get("RKK_S2_LEARNED_RECOVERY", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


def _s2_learned_window_ticks() -> int:
    try:
        return max(1, int(os.environ.get("RKK_S2_LEARNED_WINDOW", "120")))
    except ValueError:
        return 120


def _s2_scripted_fallback_enabled() -> bool:
    return os.environ.get("RKK_S2_SCRIPTED_FALLBACK", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


def _distill_override_sample_every() -> int:
    try:
        return max(0, int(os.environ.get("RKK_S2_DISTILL_OVERRIDE_SAMPLE_EVERY", "240")))
    except ValueError:
        return 240


def _meta_cognition_enabled() -> bool:
    return os.environ.get("RKK_S2_META_COGNITION", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


def _meta_plan_every(base: int, stud_conf: float, *, fallen: bool) -> int:
    """Adaptive compute: uncertain → plan more often; confident → coast."""
    if not _meta_cognition_enabled():
        return base
    if fallen:
        return max(8, base // 3)
    if stud_conf < 0.32:
        return max(8, int(base * 0.55))
    if stud_conf > 0.72:
        return min(int(base * 1.65), 160)
    return base


def _meta_wm_beam_scale(stud_conf: float) -> float:
    if not _meta_cognition_enabled():
        return 1.0
    return float(np.clip(0.55 + 0.9 * (1.0 - stud_conf), 0.45, 1.35))


def _inner_voice_macro_bias(sim: Any | None) -> str | None:
    """Map inner-voice concepts → macro hint for S2-led hierarchy."""
    if sim is None:
        return None
    iv = getattr(sim, "_inner_voice", None)
    if iv is None:
        return None
    concepts = iv.get_active_concepts() if hasattr(iv, "get_active_concepts") else []
    names = {str(n).upper() for n, _ in concepts}
    if any("FALL" in n or "PRONE" in n for n in names):
        return "RECOVER_POSTURE"
    if any("GOAL" in n or "TARGET" in n for n in names):
        return "LOCOMOTE_DELIVERY"
    if any("EXPLORE" in n or "NOVEL" in n or "SURPRISE" in n for n in names):
        return "EXPLORE"
    return None


class System2Controller:
    """
    Медленный контур: раз в N тиков выбирает макрос (студент / эвристики),
    выставляет self_goal_* в графе, мягко сдвигает intent через residuals,
    передаёт один приоритетный кандидат в agent.step.
    """

    def __init__(self) -> None:
        self._student = MacroStudent()
        self._learned = LearnedMacroStudent()
        self._wm_student = WmPlannerStudent()
        self._wm = WorkingMemoryBuffer()
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
        self._s2_override_active: bool = False
        self._s2_override_start_tick: int = -1
        self._s2_fallen_streak_override: int = 0
        self._recovery_steps: list[dict[str, Any]] = []
        self._recovery_cumulative: list[int] = []
        self._override_start_obs_f: dict[str, float] = {}
        self._recovery_schedule_anchor_tick: int = -1
        self._recovery_schedule_source: str = "none"
        self._recovery_fallback_applied: bool = False
        self._recovery_steps_remediated: bool = False
        self._recovery_best_com_z: float = 0.0
        self._recovery_ticks_since_com_z_gain: int = 0
        self._learned_recovery_active: bool = False
        self._last_override_distill_sample_tick: int = -10**9
        self._episode_plan_distill_extra: dict[str, Any] = {}
        self._distill_health = DistillHealthTracker()
        self._autonomy_post_warmup_ticks: int = 0
        self._autonomy_script_ticks: int = 0
        self._autonomy_emergency_ticks: int = 0
        self._meta_stud_conf: float = 0.5
        self._last_wm_plan: dict[str, Any] = {}

    @property
    def working_memory(self) -> WorkingMemoryBuffer:
        return self._wm

    def _sync_human_task_wm_from_sim(self, sim: Any | None, sim_tick: int) -> None:
        """Backfill WM human-command slots when task binding active before lazy S2 tick."""
        if not working_memory_enabled() or sim is None:
            return
        tb = getattr(sim, "_task_binding", None)
        ht = tb.active_task if tb is not None else None
        if ht is None:
            return
        if self._wm.has("human_task_active"):
            return
        snippet = str(getattr(ht, "text", "") or "")[:120]
        self._wm.write(
            "human_task_active",
            1.0,
            text=snippet,
            tick=int(sim_tick),
            source="task_binding_backfill",
        )
        self._wm.write(
            "human_task_pe",
            1.0,
            text=snippet[:80],
            tick=int(sim_tick),
            source="task_binding_backfill",
        )

    def note_wm_plan(
        self,
        var: str,
        val: float,
        *,
        macro: str,
        obs_f: dict[str, float],
        wm_score: float = 0.0,
    ) -> None:
        """Record WM planner action for gradient loop closure."""
        if self._wm_student.enabled():
            self._wm_student.record_plan(
                macro, var, val, obs_f, wm_score=wm_score
            )
        self._last_wm_plan = {
            "var": var,
            "val": val,
            "macro": macro,
            "wm_score": wm_score,
        }
        if working_memory_enabled():
            self._wm.write(
                "wm_last_action",
                float(val),
                text=f"{macro}:{var}",
                tick=int(getattr(self, "_last_plan_tick", 0)),
                source="s2_wm_planner",
            )

    def note_autonomy_sample(self, sim_tick: int) -> None:
        """Track A1/A4 fractions post-warmup (WorldAutonomyContract humanoid probes)."""
        try:
            warmup = int(os.environ.get("RKK_SCORECARD_WARMUP_TICKS", "800"))
        except ValueError:
            warmup = 800
        if int(sim_tick) < warmup:
            return
        self._autonomy_post_warmup_ticks += 1
        macro = (
            "RECOVER_POSTURE"
            if self._s2_override_active
            else str(self._active_macro or "IDLE")
        )
        if macro == "RECOVER_POSTURE" and self._motor_owner_for_tick(sim_tick) == "s2_scripted":
            self._autonomy_script_ticks += 1
        if self._s2_override_active:
            self._autonomy_emergency_ticks += 1

    def autonomy_fields(self) -> dict[str, float]:
        denom = max(1, self._autonomy_post_warmup_ticks)
        script_frac = float(self._autonomy_script_ticks) / float(denom)
        emerg_frac = float(self._autonomy_emergency_ticks) / float(denom)
        return {
            "script_override_frac_post_warmup": round(script_frac, 5),
            "emergency_override_frac_post_warmup": round(emerg_frac, 5),
            "s2_override_frac": round(script_frac, 5),
            "fallen_override_frac_post_warmup": round(emerg_frac, 5),
        }

    @property
    def fallen_override_active(self) -> bool:
        return bool(self._s2_override_active)

    def defer_sim_fall_hard_reset(self) -> bool:
        """
        Пока S2 fallen_override ведёт recovery, mixin_fall не должен вызывать reset_stance.
        Только при активном override (не при простом RKK_S2_LEARNED_RECOVERY=1).
        """
        if not self._s2_override_active:
            return False
        return os.environ.get("RKK_S2_DEFER_FALL_HARD_RESET", "1").strip().lower() not in (
            "0",
            "false",
            "no",
            "off",
        )

    def ollama_busy(self) -> bool:
        """Legacy hook for Ollama yield; macro/recovery LLM paths removed."""
        return False

    def _recovery_schedule_wm_candidate(
        self, sim_tick: int
    ) -> dict[str, Any] | None:
        """Кандидат do() из текущей фазы recovery schedule (антропоморфная фаза, не только торс)."""
        if not self._recovery_steps or not self._recovery_cumulative:
            return None
        anchor = int(self._recovery_schedule_anchor_tick)
        if anchor < 0:
            anchor = int(self._s2_override_start_tick)
        rel = max(0, int(sim_tick) - anchor)
        idx = bisect.bisect_right(self._recovery_cumulative, rel)
        idx = min(idx, len(self._recovery_steps) - 1)
        deltas = self._recovery_steps[idx].get("intent_deltas") or {}
        if not isinstance(deltas, dict) or not deltas:
            return None
        phase_keys = (
            "intent_stop_recover",
            "intent_support_left",
            "intent_support_right",
            "intent_torso_forward",
            "intent_arm_counterbalance",
            "intent_lean_forward",
            "intent_stride",
        )
        var = None
        if isinstance(deltas, dict) and deltas:
            keys_in_phase = [k for k in phase_keys if k in deltas]
            if keys_in_phase:
                var = keys_in_phase[idx % len(keys_in_phase)]
            else:
                var = max(deltas.keys(), key=lambda k: abs(float(deltas[k])))
        base = macro_bundle("RECOVER_POSTURE").get("candidate") or {}
        try:
            base_val = float(base.get("value", 0.5))
        except (TypeError, ValueError):
            base_val = 0.5
        val = float(max(0.06, min(0.94, base_val + float(deltas[var]))))
        if var == "intent_stop_recover":
            try:
                cap = float(os.environ.get("RKK_S2_RECOVER_STOP_RECOVER_GRAPH_MAX", "0.72"))
            except ValueError:
                cap = 0.72
            val = min(val, cap)
        return {
            "variable": str(var),
            "value": val,
            "target": str(base.get("target", "posture_stability")),
            "uncertainty": float(base.get("uncertainty", 0.38)),
            "expected_ig": float(base.get("expected_ig", 0.62)),
        }

    def _intention_from_sim(self, sim: Any | None) -> Any | None:
        if sim is None:
            return None
        return getattr(sim, "_intention_state", None)

    def _sync_active_macro_from_intention(self, sim: Any | None) -> None:
        """Keep S2 macro aligned with Intention Cortex between planning ticks."""
        if self._s2_override_active:
            return
        intent_ctx = self._intention_from_sim(sim)
        if intent_ctx is None:
            return
        hint = str(getattr(intent_ctx, "macro_hint", "") or "").strip().upper()
        if hint not in ("IDLE", "RECOVER_POSTURE", "LOCOMOTE_DELIVERY", "EXPLORE"):
            return
        if hint != str(self._active_macro or "IDLE").upper():
            self._active_macro = hint
            self._last_source = "intention_cortex_sync"
        if hint in ("LOCOMOTE_DELIVERY", "EXPLORE"):
            horizon = int(getattr(intent_ctx, "horizon_ticks", 0) or 0)
            if horizon > 0:
                tick = int(getattr(sim, "tick", 0) if sim is not None else 0)
                self._macro_until_tick = max(self._macro_until_tick, tick + horizon)

    def planning_context_for_wm(
        self, *, fallen: bool = False, sim_tick: int = -1, sim: Any | None = None
    ) -> dict[str, Any]:
        """Контекст для S2-gated WM planner (agent.step после tick)."""
        macro = (
            "RECOVER_POSTURE"
            if self._s2_override_active
            else str(self._active_macro or "IDLE")
        )
        intent_ctx = self._intention_from_sim(sim)
        spec = (
            self._recovery_episode_spec
            if self._s2_override_active
            else self._macro_episode_spec
        )
        bundle = macro_bundle(macro)
        sched_cand = None
        if self._s2_override_active and int(sim_tick) >= 0:
            sched_cand = self._recovery_schedule_wm_candidate(int(sim_tick))
        expected = dict(spec.expected_state or {})
        goal_td = None
        if intent_ctx is not None and not self._s2_override_active:
            if getattr(intent_ctx, "macro_hint", None):
                macro = str(intent_ctx.macro_hint)
            if getattr(intent_ctx, "expected_state", None):
                expected.update(dict(intent_ctx.expected_state))
            primary = getattr(intent_ctx, "primary", None)
            if primary is not None:
                goal_td = float(
                    getattr(primary, "target_val", 0.42)
                )

        human_task_active = False
        human_task_text = ""
        max_pe_out = spec.max_prediction_error
        if sim is not None and not self._s2_override_active:
            tb = getattr(sim, "_task_binding", None)
            ht = tb.active_task if tb is not None else None
            if ht is not None and ht.expected_state:
                expected.update(dict(ht.expected_state))
                human_task_active = True
                human_task_text = str(ht.text or "")[:120]
                if not fallen:
                    macro = "EXPLORE"
                if ht.max_prediction_error is not None:
                    max_pe_out = ht.max_prediction_error

        return {
            "macro": macro,
            "fallen": bool(fallen),
            "fallen_override_active": bool(self._s2_override_active),
            "human_task_active": human_task_active,
            "human_task_text": human_task_text,
            "intention_narrative": (
                str(getattr(intent_ctx, "narrative", "")) if intent_ctx else ""
            ),
            "intention_stack_depth": (
                int(getattr(intent_ctx, "stack_depth", 0)) if intent_ctx else 0
            ),
            "intention_horizon_ticks": (
                int(getattr(intent_ctx, "horizon_ticks", 0)) if intent_ctx else 0
            ),
            "goal_target_dist": goal_td,
            "learned_recovery_active": bool(
                self._learned_recovery_active
                and self._recovery_schedule_source == "learned"
            ),
            "motor_owner": self._motor_owner_for_tick(int(sim_tick)),
            "expected_state": expected,
            "max_prediction_error": max_pe_out,
            "skill_id": spec.skill_id if not human_task_active else "human_command",
            "bundle_candidate": bundle.get("candidate"),
            "recovery_schedule_candidate": sched_cand,
            "wm_override_schedule_only": bool(
                self._s2_override_active and _s2_wm_override_schedule_only()
            ),
            "recovery_schedule_source": str(self._recovery_schedule_source),
            "cpg_recovery_active": bool(
                self._s2_override_active
                and os.environ.get("RKK_S2_CPG_DURING_OVERRIDE", "1").strip().lower()
                not in ("0", "false", "no", "off")
            ),
            "working_memory": (
                self._wm.context_dict(int(sim_tick)) if working_memory_enabled() else {}
            ),
            "meta_stud_conf": round(float(self._meta_stud_conf), 4),
            "meta_wm_beam_scale": round(_meta_wm_beam_scale(self._meta_stud_conf), 4),
            "s2_led_hierarchy": True,
            "source": (
                "fallen_override"
                if self._s2_override_active
                else str(self._last_source or "")
            ),
        }

    def _obs_floats(self, obs: dict[str, Any]) -> dict[str, float]:
        out: dict[str, float] = {}
        for k, v in obs.items():
            try:
                out[str(k)] = float(v)
            except (TypeError, ValueError):
                continue
        return out

    def _macro_outcome_deferred(self, sim_tick: int) -> bool:
        return False

    def _macro_horizon_expired(self, sim_tick: int) -> bool:
        ut = self._macro_until_tick
        if ut < 0 or sim_tick <= ut:
            return False
        if self._macro_outcome_deferred(sim_tick):
            return False
        return not self._macro_start_obs

    def _merge_proposal_goal_into_graph(self, graph: Any, proposal: System2Proposal) -> None:
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

    def _maybe_rolling_macro_distill(
        self,
        sim_tick: int,
        agent: Any,
        obs_f: dict[str, float],
    ) -> None:
        """Rolling window episodes for continuous LOCOMOTE (no terminal required)."""
        try:
            every = int(os.environ.get("RKK_S2_ROLLING_DISTILL_EVERY", "200"))
        except ValueError:
            every = 200
        every = max(40, every)
        if int(sim_tick) % every != 0:
            return
        macro = str(self._active_macro or "IDLE")
        if macro not in ("LOCOMOTE_DELIVERY", "EXPLORE"):
            return
        if not self._macro_start_obs:
            self._macro_start_obs = dict(obs_f)
            return
        cz0 = float(
            self._macro_start_obs.get("com_z", self._macro_start_obs.get("phys_com_z", 0.5))
        )
        cz1 = float(obs_f.get("com_z", obs_f.get("phys_com_z", cz0)))
        ps0 = float(
            self._macro_start_obs.get(
                "posture_stability",
                self._macro_start_obs.get("phys_posture_stability", 0.5),
            )
        )
        ps1 = float(
            obs_f.get("posture_stability", obs_f.get("phys_posture_stability", ps0))
        )
        success, pe_diag = episode_success_with_pe_fallback(
            self._macro_start_obs,
            obs_f,
            self._macro_episode_spec,
            macro=macro,
        )
        self._student.record_outcome(macro, success, weight=0.35)
        stud_conf: float | None = None
        if self._learned.enabled():
            self._learned.learn(
                macro,
                success,
                dict(self._macro_start_obs),
                d_com_z=cz1 - cz0,
                d_posture=ps1 - ps0,
            )
            _, stud_conf = self._learned.predict(dict(self._macro_start_obs))
        distill_x = pe_distill_extra(
            pe_diag,
            dict(self._macro_episode_spec.expected_state or {}),
            self._macro_episode_spec.skill_id,
        )
        distill_x["distill_event"] = "rolling_window"
        self._append_distill(
            tick=sim_tick,
            macro=macro,
            source=str(self._last_source or "rolling"),
            success=success,
            delta={"d_com_z": round(cz1 - cz0, 5), "d_posture": round(ps1 - ps0, 5)},
            obs0=snapshot_obs_for_distill(dict(self._macro_start_obs)) or None,
            extra=distill_x,
            student_conf=stud_conf,
        )
        sim = getattr(self, "_rkk_sim", None)
        if sim is not None:
            sim._s2_episodes_collected_total = int(
                getattr(sim, "_s2_episodes_collected_total", 0)
            ) + 1
        self._macro_start_obs = dict(obs_f)

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
        ending_source: str | None = None,
        student_conf: float | None = None,
        count_health: bool = True,
    ) -> None:
        if not distill_enabled() or eval_mode_enabled():
            return
        path = distill_log_path()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            row: dict[str, Any] = {
                "tick": tick,
                "macro": macro,
                "source": source,
                "success": success,
                "delta": delta,
            }
            if ending_source is not None:
                row["ending_source"] = ending_source
            if obs0:
                row["obs0"] = obs0
            if student_conf is not None:
                row["student_conf"] = round(float(student_conf), 4)
            if extra:
                for k, v in extra.items():
                    if v is not None:
                        row[k] = v
            sim = getattr(self, "_rkk_sim", None)
            if sim is not None and hasattr(sim, "agent"):
                row.update(curriculum_context_tags(sim, sim.agent))
            line = json.dumps(row, ensure_ascii=False)
            with open(path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
            if count_health:
                self._distill_health.record(
                    success=success,
                    macro=macro,
                    student_conf=student_conf,
                )
        except OSError:
            pass

    def _distill_health_diag(self) -> dict[str, Any]:
        return dict(self._distill_health.snapshot())

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
        try:
            from engine.graph_perf import is_large_graph

            if is_large_graph(agent.graph):
                return None
        except ImportError:
            pass
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
        self._override_start_obs_f = {}
        self._recovery_episode_spec = EpisodeSuccessSpec()
        self._recovery_schedule_anchor_tick = -1
        self._recovery_schedule_source = "none"
        self._recovery_fallback_applied = False
        self._recovery_steps_remediated = False
        self._recovery_best_com_z = 0.0
        self._recovery_ticks_since_com_z_gain = 0
        self._learned_recovery_active = False

    def _in_learned_recovery_phase(self, sim_tick: int) -> bool:
        if not self._s2_override_active or not self._learned_recovery_active:
            return False
        if self._recovery_schedule_source != "learned":
            return False
        age = int(sim_tick) - int(self._s2_override_start_tick)
        return age < _s2_learned_window_ticks()

    def _maybe_transition_learned_to_scripted(self, sim_tick: int) -> None:
        if self._in_learned_recovery_phase(sim_tick):
            return
        if (
            not self._learned_recovery_active
            or self._recovery_schedule_source != "learned"
            or not _s2_scripted_fallback_enabled()
        ):
            return
        age = int(sim_tick) - int(self._s2_override_start_tick)
        if age < _s2_learned_window_ticks():
            return
        if not self._apply_scripted_recovery_schedule(sim_tick) and recovery_fallback_enabled():
            self._apply_fallback_recovery_schedule(sim_tick)
        self._learned_recovery_active = False

    def _motor_owner_for_tick(self, sim_tick: int) -> str:
        if not self._s2_override_active:
            return "cpg"
        if self._in_learned_recovery_phase(sim_tick):
            return "s1_learned"
        if self._recovery_schedule_source in ("scripted", "fallback", "library"):
            return "s2_scripted"
        return "cpg"

    @staticmethod
    def _obs_com_z(obs_f: dict[str, float]) -> float:
        return float(obs_f.get("com_z", obs_f.get("phys_com_z", 0.5)))

    def _maybe_refresh_override_obs0(self, obs_f: dict[str, float], *, fallen: bool) -> None:
        """Re-anchor obs0 when override opened on stale standing snapshot."""
        if not fallen or not self._override_start_obs_f:
            return
        try:
            hi = float(os.environ.get("RKK_S2_OVERRIDE_OBS0_REANCHOR_COM_Z", "0.32"))
        except ValueError:
            hi = 0.32
        cz0 = self._obs_com_z(self._override_start_obs_f)
        cz1 = self._obs_com_z(obs_f)
        if cz0 > hi and cz1 < hi:
            self._override_start_obs_f = dict(obs_f)
            self._recovery_best_com_z = cz1
            self._recovery_ticks_since_com_z_gain = 0

    def _recovery_track_com_z_progress(self, obs_f: dict[str, float]) -> None:
        cz = self._obs_com_z(obs_f)
        try:
            eps = float(os.environ.get("RKK_S2_RECOVERY_COM_Z_GAIN_EPS", "0.015"))
        except ValueError:
            eps = 0.015
        if cz > self._recovery_best_com_z + eps:
            self._recovery_best_com_z = cz
            self._recovery_ticks_since_com_z_gain = 0
        else:
            self._recovery_ticks_since_com_z_gain += 1

    def _capture_override_obs0_from_base(self, base: Any, obs_f: dict[str, float]) -> None:
        """Use live physics snapshot so obs0 is prone, not stale standing graph nodes."""
        raw: dict[str, float] = {}
        fn = getattr(base, "observe", None)
        if callable(fn):
            try:
                o = fn()
                if isinstance(o, dict):
                    for k, v in o.items():
                        try:
                            raw[str(k)] = float(v)
                        except (TypeError, ValueError):
                            pass
            except Exception:
                pass
        if not raw:
            gs = getattr(base, "_sim", None)
            if gs is not None and hasattr(gs, "get_state"):
                try:
                    st = gs.get_state()
                    if isinstance(st, dict):
                        raw = {str(k): float(v) for k, v in st.items()}
                except Exception:
                    pass
        if raw:
            merged = dict(obs_f)
            for k, v in raw.items():
                merged[str(k)] = float(v)
            self._override_start_obs_f = self._obs_floats(merged)
        else:
            self._override_start_obs_f = dict(obs_f)

    def _scripted_schedule_locked(self, sim_tick: int) -> bool:
        if not recovery_scripted_lock_until_exhausted():
            return False
        if self._recovery_schedule_source != "scripted":
            return False
        return not self._recovery_schedule_exhausted(sim_tick)

    def _apply_scripted_getup_physics(self, base: Any, sim_tick: int) -> None:
        if self._recovery_schedule_source != "scripted" or not self._recovery_steps:
            return
        anc = int(self._recovery_schedule_anchor_tick)
        if anc < 0:
            anc = int(self._s2_override_start_tick)
        _idx, phase = scripted_getup_phase_at(
            sim_tick, anc, self._recovery_cumulative, self._recovery_steps
        )
        fn = getattr(base, "apply_scripted_getup_physics", None)
        if callable(fn):
            try:
                fn(phase)
            except Exception:
                pass

    def _recovery_schedule_exhausted(self, sim_tick: int) -> bool:
        if not self._recovery_cumulative:
            return True
        anchor = int(self._recovery_schedule_anchor_tick)
        if anchor < 0:
            anchor = int(self._s2_override_start_tick)
        rel = max(0, int(sim_tick) - anchor)
        return rel > int(self._recovery_cumulative[-1])

    def _apply_fallback_recovery_schedule(self, sim_tick: int) -> bool:
        if self._recovery_fallback_applied or not recovery_fallback_enabled():
            return False
        steps = enrich_recovery_steps(default_recovery_fallback_steps())
        if not steps:
            return False
        self._ingest_recovery_steps(steps, sim_tick=sim_tick)
        self._recovery_fallback_applied = True
        self._recovery_schedule_source = "fallback"
        self._recovery_episode_spec = EpisodeSuccessSpec(skill_id="recovery_fallback")
        return True

    def _apply_scripted_recovery_schedule(self, sim_tick: int) -> bool:
        if not recovery_scripted_enabled():
            return False
        steps = prepare_scripted_getup_steps()
        if not steps:
            return False
        self._ingest_recovery_steps(steps, sim_tick=sim_tick)
        self._recovery_schedule_source = "scripted"
        spec = scripted_getup_episode_spec()
        self._recovery_episode_spec = EpisodeSuccessSpec(
            expected_state=dict(spec.get("expected_state") or {}),
            max_prediction_error=spec.get("max_prediction_error"),
            skill_id=str(spec.get("skill_id", "recovery_scripted_getup")),
        )
        return True

    def _ingest_recovery_steps(
        self,
        steps: list[dict[str, Any]] | None,
        *,
        sim_tick: int,
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
        ticks = int(max(1, self._recovery_steps[idx].get("ticks", 1)))
        try:
            rscale = float(os.environ.get("RKK_S2_RECOVERY_RESIDUAL_SCALE", "1.35"))
        except ValueError:
            rscale = 1.35
        if self._recovery_schedule_source == "scripted":
            try:
                rscale = float(
                    os.environ.get("RKK_S2_RECOVERY_SCRIPTED_RESIDUAL_SCALE", "1.65")
                )
            except ValueError:
                rscale = 1.65
        if not isinstance(d, dict):
            return {}
        return {k: float(v) / ticks * rscale for k, v in d.items()}

    def _override_episode_eval(
        self, obs_f: dict[str, float], *, fallen: bool
    ) -> tuple[bool, dict[str, Any], str]:
        """(success, diag, source_note_suffix). source_note empty if no tier exit."""
        if fallen:
            return False, {"fallen": True}, ""
        tier, ok, pe_diag = evaluate_override_recovery_exit(
            obs_f,
            self._override_start_obs_f,
            self._recovery_episode_spec,
            macro="RECOVER_POSTURE",
        )
        if tier == 0:
            return False, pe_diag, ""
        note = "recovered" if tier == 2 else "recovered_tier1"
        pe_diag["recover_tier"] = tier
        return ok, pe_diag, note

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
        if success and self._recovery_steps and recovery_library_enabled():
            try:
                get_recovery_library().add_success(
                    dict(self._override_start_obs_f),
                    list(self._recovery_steps),
                    skill_id=str(self._recovery_episode_spec.skill_id or "recovery_library"),
                )
            except Exception:
                pass
        if self._learned.enabled():
            self._learned.learn(
                macro,
                success,
                dict(self._override_start_obs_f),
                d_com_z=cz1 - cz0,
                d_posture=ps1 - ps0,
            )
        spec = self._recovery_episode_spec
        ex = pe_distill_extra(
            pe_diag or {},
            dict(spec.expected_state or {}),
            spec.skill_id,
        )
        if self._recovery_steps:
            ex["recovery_steps"] = compress_recovery_steps(self._recovery_steps)
        ex["recovery_schedule_source"] = str(self._recovery_schedule_source)
        if self._learned_recovery_active and self._recovery_schedule_source == "learned":
            ex["distill_source"] = "learned_recovery"
        if self._recovery_steps_remediated:
            ex["recovery_steps_remediated"] = True
        if pe_diag and pe_diag.get("recover_tier") is not None:
            ex["recover_tier"] = int(pe_diag["recover_tier"])
        if pe_diag and pe_diag.get("override_exit_block"):
            ex["override_exit_block"] = str(pe_diag["override_exit_block"])
        ex["distill_event"] = f"override_end:{source_note}"
        stud_conf: float | None = None
        if self._learned.enabled():
            _, stud_conf = self._learned.predict(dict(self._override_start_obs_f))
        self._append_distill(
            tick=sim_tick,
            macro=macro,
            source=f"fallen_override:{source_note}",
            success=success,
            delta={"d_com_z": round(cz1 - cz0, 5), "d_posture": round(ps1 - ps0, 5)},
            obs0=snapshot_obs_for_distill(dict(self._override_start_obs_f)) or None,
            extra=ex,
            ending_source=f"fallen_override:{source_note}",
            student_conf=stud_conf,
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
        try:
            stop_rec = float(os.environ.get("RKK_S2_RECOVER_STOP_RECOVER_GRAPH", "0.58"))
        except ValueError:
            stop_rec = 0.58
        torso_fwd = 0.68
        if self._recovery_schedule_source == "scripted":
            try:
                stop_rec = float(
                    os.environ.get("RKK_S2_RECOVERY_SCRIPTED_STOP_RECOVER_GRAPH", "0.64")
                )
            except ValueError:
                stop_rec = 0.64
            try:
                torso_fwd = float(
                    os.environ.get("RKK_S2_RECOVERY_SCRIPTED_TORSO_GRAPH", "0.74")
                )
            except ValueError:
                torso_fwd = 0.74
        if extra_residuals and "intent_stop_recover" in extra_residuals:
            try:
                cap = float(os.environ.get("RKK_S2_RECOVER_STOP_RECOVER_GRAPH_MAX", "0.72"))
            except ValueError:
                cap = 0.72
            stop_rec = min(stop_rec, cap)
        graph_patch.update(
            {
                "intent_stop_recover": stop_rec,
                "intent_torso_forward": torso_fwd,
                "intent_stride": 0.46,
                "intent_gait_coupling": 0.40,
            }
        )
        residuals: dict[str, float] = dict(bundle.get("residuals") or {})
        
        # Divide baseline residuals by the planning interval (48) to scale smoothly
        plan_every = _plan_every_ticks()
        for k in list(residuals.keys()):
            residuals[k] = float(residuals[k]) / plan_every

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
        pe_diag: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        base_diag: dict[str, Any] = {
            "enabled": True,
            "fallen_override_active": True,
            "learned_recovery_active": self._in_learned_recovery_phase(sim_tick),
            "motor_owner": self._motor_owner_for_tick(sim_tick),
            "fallen_override_ticks": int(max(0, age)) + 1,
            "fallen_override_max_ticks": int(max_age),
            "s2_fallen_streak": int(self._s2_fallen_streak_override),
            "override_recovered": bool(recovered),
            "override_max_reset": bool(max_reset),
            "recovery_steps_loaded": len(self._recovery_steps),
            "macro": "RECOVER_POSTURE",
            "source": f"fallen_override:{self._recovery_schedule_source}" if self._recovery_schedule_source and self._recovery_schedule_source != "none" else "fallen_override",
            "sim_tick": sim_tick,
            "fallen": bool(fallen),
            "blocked": False,
            "neuro_streak": dict(self._neuro_streak),
            "outcome_ema": round(self._outcome_ema, 4),
        }
        if applied is not None:
            base_diag["residuals_applied"] = applied
        base_diag["recovery_schedule_source"] = str(self._recovery_schedule_source)
        base_diag["recovery_steps_loaded"] = len(self._recovery_steps)
        base_diag["cpg_recovery_active"] = bool(
            self._s2_override_active
            and os.environ.get("RKK_S2_CPG_DURING_OVERRIDE", "1").strip().lower()
            not in ("0", "false", "no", "off")
        )
        if pe_diag:
            if pe_diag.get("recover_tier") is not None:
                base_diag["recover_tier"] = int(pe_diag["recover_tier"])
            if pe_diag.get("override_exit_block"):
                base_diag["override_exit_block"] = str(pe_diag["override_exit_block"])
        base_diag.update(self._distill_health_diag())
        return base_diag

    def _maybe_distill_override_sample(
        self,
        sim_tick: int,
        obs_f: dict[str, float],
        *,
        fallen: bool,
    ) -> None:
        every = _distill_override_sample_every()
        if every <= 0 or not self._s2_override_active:
            return
        if int(sim_tick) - int(self._last_override_distill_sample_tick) < every:
            return
        self._last_override_distill_sample_tick = int(sim_tick)
        age = int(sim_tick) - int(self._s2_override_start_tick)
        ex: dict[str, Any] = {
            "distill_event": "override_sample",
            "fallen": bool(fallen),
            "override_age_ticks": age,
            "recovery_schedule_source": str(self._recovery_schedule_source),
            "recovery_steps_loaded": len(self._recovery_steps),
        }
        if self._recovery_steps_remediated:
            ex["recovery_steps_remediated"] = True
        if self._recovery_steps:
            ex["recovery_steps"] = compress_recovery_steps(self._recovery_steps)
        stud_conf: float | None = None
        if self._learned.enabled():
            _, stud_conf = self._learned.predict(dict(self._override_start_obs_f))
        self._append_distill(
            tick=sim_tick,
            macro="RECOVER_POSTURE",
            source="fallen_override:sample",
            success=False,
            delta={
                "d_com_z": round(
                    float(obs_f.get("com_z", obs_f.get("phys_com_z", 0.5)))
                    - float(
                        self._override_start_obs_f.get(
                            "com_z",
                            self._override_start_obs_f.get("phys_com_z", 0.5),
                        )
                    ),
                    5,
                ),
                "d_posture": round(
                    float(
                        obs_f.get(
                            "posture_stability",
                            obs_f.get("phys_posture_stability", 0.5),
                        )
                    )
                    - float(
                        self._override_start_obs_f.get(
                            "posture_stability",
                            self._override_start_obs_f.get(
                                "phys_posture_stability", 0.5
                            ),
                        )
                    ),
                    5,
                ),
            },
            obs0=snapshot_obs_for_distill(dict(self._override_start_obs_f)) or None,
            extra=ex,
            student_conf=stud_conf,
            count_health=False,
        )

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

        if not self._s2_override_active:
            if self._s2_fallen_streak_override < _s2_override_fallen_ticks_need():
                return None
            self._s2_override_active = True
            self._s2_override_start_tick = sim_tick
            self._capture_override_obs0_from_base(base, obs_f)
            self._maybe_refresh_override_obs0(obs_f, fallen=True)
            self._recovery_steps = []
            self._recovery_cumulative = []
            self._recovery_episode_spec = EpisodeSuccessSpec()
            self._recovery_schedule_anchor_tick = -1
            self._recovery_best_com_z = self._obs_com_z(obs_f)
            self._recovery_ticks_since_com_z_gain = 0
            self._learned_recovery_active = _s2_learned_recovery_enabled()
            loaded_schedule = False
            if self._learned_recovery_active:
                self._recovery_schedule_source = "learned"
                self._recovery_episode_spec = EpisodeSuccessSpec(skill_id="learned_recovery")
            elif self._apply_scripted_recovery_schedule(sim_tick):
                loaded_schedule = True
            elif recovery_library_enabled():
                lib_hit = get_recovery_library().lookup(obs_f)
                if lib_hit is not None:
                    steps, es_rec, mx_f, skill = lib_hit
                    self._ingest_recovery_steps(
                        enrich_recovery_steps(steps),
                        sim_tick=sim_tick,
                    )
                    self._recovery_schedule_source = "library"
                    self._recovery_episode_spec = EpisodeSuccessSpec(
                        expected_state=dict(es_rec or {}),
                        max_prediction_error=mx_f,
                        skill_id=str(skill),
                    )
                    loaded_schedule = True
            if (
                recovery_fallback_enabled()
                and not self._recovery_steps
                and not self._learned_recovery_active
            ):
                self._apply_fallback_recovery_schedule(sim_tick)
                loaded_schedule = bool(self._recovery_steps)
        else:
            self._maybe_refresh_override_obs0(obs_f, fallen=fallen)
            self._recovery_track_com_z_progress(obs_f)
            age = int(sim_tick) - int(self._s2_override_start_tick)
            max_age = _s2_override_max_ticks()
            ok, pe_diag, exit_note = self._override_episode_eval(
                obs_f, fallen=False
            )
            if exit_note:
                self._record_override_distill_neuro(
                    sim_tick=sim_tick,
                    agent=agent,
                    obs_f=obs_f,
                    success=ok,
                    source_note=exit_note,
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

        if self._s2_override_active:
            self._maybe_distill_override_sample(sim_tick, obs_f, fallen=fallen)

        self._maybe_transition_learned_to_scripted(sim_tick)

        extra = self._recovery_extra_residuals(sim_tick)
        applied = False
        in_learned = self._in_learned_recovery_phase(sim_tick)
        if not in_learned:
            applied = self._apply_recover_bundle_no_candidate(
                sim_tick,
                agent,
                base,
                graph,
                node_keys,
                extra if extra else None,
            )
            self._apply_scripted_getup_physics(base, sim_tick)
        age = int(sim_tick) - int(self._s2_override_start_tick)
        max_age = _s2_override_max_ticks()
        _, pe_diag_live, _ = self._override_episode_eval(obs_f, fallen=False)
        return self._build_override_diag(
            sim_tick,
            fallen,
            age,
            max_age,
            applied=applied,
            pe_diag=pe_diag_live,
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
    ) -> dict[str, Any]:
        """Один цикл планирования (student / learned student only)."""
        self._last_plan_tick = sim_tick
        if (
            self._learned.enabled()
            and not self._bootstrap_attempted
            and os.environ.get("RKK_SYSTEM2_STUDENT_BOOTSTRAP", "0").strip().lower()
            in ("1", "true", "yes", "on")
        ):
            self._learned.bootstrap_from_log(distill_log_path())
        self._bootstrap_attempted = True

        stud_conf = 0.0
        if self._learned.enabled():
            macro, stud_conf = self._learned.predict(obs_f)
            source = "student_learned"
        else:
            macro = choose_macro_from_obs(obs_f)
            source = "student"
        self._meta_stud_conf = float(stud_conf)

        # S2-led hierarchy: inner voice + WM context bias macro before intention cortex
        iv_macro = _inner_voice_macro_bias(sim)
        if iv_macro and not self._s2_override_active:
            macro = iv_macro
            source = "inner_voice_bias"
        if working_memory_enabled():
            self._wm.decay(sim_tick)
            wm_macro = self._wm.read_text("active_macro", "")
            _valid_macros = ("IDLE", "RECOVER_POSTURE", "LOCOMOTE_DELIVERY", "EXPLORE")
            if wm_macro in _valid_macros and not self._s2_override_active:
                if stud_conf < 0.45:
                    macro = wm_macro
                    source = "working_memory"
            for k, v in self._wm.context_dict(sim_tick).items():
                if k.startswith("concept_") and v > 0.55:
                    obs_f[k] = float(v)

        intent_ctx = self._intention_from_sim(sim)
        intent_horizon = _macro_horizon_ticks()
        if intent_ctx is not None and not self._s2_override_active:
            hint = str(getattr(intent_ctx, "macro_hint", "") or "").strip().upper()
            if hint and hint in ("IDLE", "RECOVER_POSTURE", "LOCOMOTE_DELIVERY", "EXPLORE"):
                macro = hint
                source = "intention_cortex"
            if macro == "IDLE":
                try:
                    from engine.locomote_gate import stable_locomote_ready

                    if stable_locomote_ready(obs_f):
                        macro = "LOCOMOTE_DELIVERY"
                        source = "stable_locomote_gate"
                except ImportError:
                    pass
            intent_horizon = max(
                intent_horizon,
                int(getattr(intent_ctx, "horizon_ticks", intent_horizon)),
            )

        self._episode_plan_distill_extra = {}
        proposal_effective: System2Proposal | None = None

        bundle = macro_bundle(macro)
        graph_patch: dict[str, float] = dict(bundle.get("graph") or {})
        cand_tpl = bundle.get("candidate")
        residuals: dict[str, float] = dict(bundle.get("residuals") or {})

        if intent_ctx is not None and not self._s2_override_active:
            for k, v in (getattr(intent_ctx, "graph_patch", None) or {}).items():
                if k in node_keys:
                    graph_patch[k] = float(v)
            for k, dv in (getattr(intent_ctx, "intent_residuals", None) or {}).items():
                if k in node_keys:
                    residuals[k] = residuals.get(k, 0.0) + float(dv)

        if proposal_effective and proposal_effective.intent_deltas:
            for k, dv in proposal_effective.intent_deltas.items():
                if k in node_keys:
                    residuals[k] = residuals.get(k, 0.0) + float(dv)

        nodes = agent.graph.nodes
        for k, v in graph_patch.items():
            if k in nodes:
                nodes[k] = float(max(0.05, min(0.95, float(v))))

        if proposal_effective:
            self._merge_proposal_goal_into_graph(agent.graph, proposal_effective)

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
        self._macro_until_tick = sim_tick + int(intent_horizon)
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
        if working_memory_enabled():
            self._wm.write(
                "active_macro",
                1.0,
                text=macro,
                tick=sim_tick,
                source=source,
            )
            if sim is not None:
                iv = getattr(sim, "_inner_voice", None)
                if iv is not None:
                    for name, act in iv.get_active_concepts()[:5]:
                        self._wm.write(
                            f"concept_{str(name).lower()}",
                            float(act),
                            text=str(name),
                            tick=sim_tick,
                            source="inner_voice",
                        )

        hz_exp = self._macro_horizon_expired(sim_tick)
        defer = self._macro_outcome_deferred(sim_tick)
        self._last_diag = {
            "enabled": True,
            "macro": macro,
            "source": source,
            "until": self._macro_until_tick,
            "sim_tick": sim_tick,
            "macro_horizon_expired": hz_exp,
            "macro_outcome_deferred": defer,
            "has_candidate": candidate is not None,
            "last_candidate_var": last_var,
            "blocked": False,
            "outcome_ema": round(self._outcome_ema, 4),
            "student_conf": round(float(stud_conf), 4) if self._learned.enabled() else None,
            "wm_student_conf": (
                round(self._wm_student.last_confidence(), 4)
                if self._wm_student.enabled()
                else None
            ),
            "meta_plan_every": _meta_plan_every(
                _plan_every_ticks(), stud_conf, fallen=False
            ),
            "residuals_applied": residuals_applied,
            "neuro_streak": dict(self._neuro_streak),
            "online_buf": len(self._online_buf),
            "last_neuro_node": self._last_neuro_node,
            "recovery_steps_loaded": 0,
        }
        self._last_diag.update(self._distill_health_diag())
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
                skip_guard = False
                if macro in ("LOCOMOTE_DELIVERY", "EXPLORE") and sim is not None:
                    try:
                        obs = dict(getattr(sim, "_env_observe_cached", lambda: {})() or {})
                        ps = float(
                            obs.get(
                                "posture_stability",
                                obs.get("phys_posture_stability", 0.5),
                            )
                        )
                        skip_guard = ps >= float(
                            os.environ.get("RKK_SYSTEM2_LOCOMOTE_CPG_GUARD_PS", "0.58")
                        )
                    except Exception:
                        skip_guard = False
                if not skip_guard:
                    fnr = getattr(sim, "_locomotion_reward_ema", None) if sim is not None else None
                    if callable(fnr):
                        try:
                            thr = float(
                                os.environ.get("RKK_SYSTEM2_RESIDUAL_MIN_LOCOREWARD", "0.18")
                            )
                        except ValueError:
                            thr = 0.18
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

        self._sync_human_task_wm_from_sim(sim, sim_tick)

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

        self._maybe_rolling_macro_distill(sim_tick, agent, obs_f)

        ov = self._maybe_tick_fallen_override(
            sim_tick, fallen, agent, obs_f, base, graph, node_keys, sim
        )
        if ov is not None:
            self._last_diag = ov
            return self._last_diag

        ending_macro = self._active_macro
        ending_source = self._last_source
        if (
            not self._s2_override_active
            and sim_tick >= self._macro_until_tick
            and self._macro_start_obs
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
            if self._wm_student.enabled():
                self._wm_student.learn_from_outcome(
                    success,
                    obs_f,
                    d_com_z=cz1 - cz0,
                    d_posture=ps1 - ps0,
                )
            try:
                from engine.intristic_objective import instrumental_task_bonus

                instrumental_task_bonus(ending_macro, success, sim=sim)
            except Exception:
                pass
            spec_e = self._macro_episode_spec
            distill_x = pe_distill_extra(
                pe_diag,
                dict(spec_e.expected_state or {}),
                spec_e.skill_id,
            )
            distill_x.update(self._episode_plan_distill_extra)
            stud_conf_end: float | None = None
            if self._learned.enabled():
                _, stud_conf_end = self._learned.predict(dict(self._macro_start_obs))
            self._append_distill(
                tick=sim_tick,
                macro=ending_macro,
                source=ending_source,
                success=success,
                delta={"d_com_z": round(cz1 - cz0, 5), "d_posture": round(ps1 - ps0, 5)},
                obs0=snapshot_obs_for_distill(dict(self._macro_start_obs)) or None,
                extra=distill_x,
                ending_source=ending_source,
                student_conf=stud_conf_end,
            )
            self._episode_plan_distill_extra = {}
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

        plan_every = _meta_plan_every(
            _plan_every_ticks(),
            self._meta_stud_conf,
            fallen=bool(fallen),
        )
        should_plan = (sim_tick - self._last_plan_tick) >= plan_every
        if sim_tick >= self._macro_until_tick:
            should_plan = True

        if not should_plan:
            self._sync_active_macro_from_intention(sim)
            hz_exp = self._macro_horizon_expired(sim_tick)
            defer = self._macro_outcome_deferred(sim_tick)
            self._last_diag = {
                "enabled": True,
                "macro": self._active_macro,
                "until": self._macro_until_tick,
                "sim_tick": sim_tick,
                "macro_horizon_expired": hz_exp,
                "macro_outcome_deferred": defer,
                "idle": True,
                "outcome_ema": round(self._outcome_ema, 4),
                "student_conf": self._last_diag.get("student_conf"),
                "last_source": self._last_source,
                "last_neuro_node": self._last_neuro_node,
                "recovery_steps_loaded": len(self._recovery_steps),
            }
            self._last_diag.update(self._distill_health_diag())
            return self._last_diag

        return self._apply_planning_step(
            sim_tick,
            agent,
            obs_f,
            base,
            graph,
            node_keys,
            sim,
        )
