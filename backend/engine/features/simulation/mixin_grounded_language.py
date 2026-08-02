"""Simulation mixin: Grounded Language + human task binding (AGI command loop)."""
from __future__ import annotations

import json
import math
import os
import threading
from typing import Any

from engine.grounded_language import (
    GroundedLanguageController,
    command_kind_for_text,
    grounded_language_enabled,
    motor_intents_from_tag,
)
from engine.goal_navigation import (
    navigation_intents,
    navigation_intents_from_bearing_range,
    navigation_intents_from_ego_xy,
)
from engine.manipulation_control import (
    manipulation_intents,
    manipulation_intents_from_bearing_range,
)
from engine.manipulation_verify import ManipulationEpisode, verify_manipulation
from engine.object_resolver import ResolvedObject, resolve_manipulation_target
from engine.object_working_memory import (
    LatentSceneMemory,
    ObjectWorkingMemory,
    ego_from_bearing_range,
    match_latent_slot,
)
from engine.success_predicates import evaluate_goal
from engine.task_binding import TaskBindingController, task_binding_enabled
from engine.task_goal import TaskGoal
from engine.task_logger import summarize_expected_state, task_log_event
from engine.task_observation import (
    build_task_observations,
    contact_reach_m,
    inject_task_observations,
    nav_stop_m,
    reach_start_m,
    sync_task_obs_to_graph,
)
from engine.task_tree import TERMINAL_STATUSES, TaskTreeController, task_tree_enabled
from engine.vision_depth import ArrayDepthCamera, DepthFrame, attach_range_to_target
from engine.vision_resolve import collect_vision_slots, resolve_visual_target, track_visual_target
from engine.vision_target import (
    VisualTarget,
    bearing_from_u,
    sim_oracle_bind_enabled,
    vision_active_percept_enabled,
    vision_active_percept_max_tries,
    vision_resolve_enabled,
)


def _grounded_lang_every() -> int:
    try:
        return max(4, int(os.environ.get("RKK_GROUNDED_LANG_EVERY", "12")))
    except ValueError:
        return 12


def _align_every() -> int:
    try:
        return max(80, int(os.environ.get("RKK_GROUNDED_LANG_ALIGN_EVERY", "400")))
    except ValueError:
        return 400


def _manip_push_force() -> float:
    try:
        return float(os.environ.get("RKK_MANIP_PUSH_FORCE", "38.0"))
    except ValueError:
        return 38.0


def _nav_arrival_streak_needed() -> int:
    try:
        return max(1, int(os.environ.get("RKK_NAV_ARRIVAL_STREAK", "3")))
    except ValueError:
        return 3


def _task_nav_mode() -> str:
    """Production default: wm_ai. Use heuristic for ablation / cold WM fallback tests."""
    raw = os.environ.get("RKK_TASK_NAV_MODE", "wm_ai").strip().lower()
    if raw in ("heuristic", "heur", "bearing", "goal_navigation"):
        return "heuristic"
    return "wm_ai"


def _task_nav_ai_every() -> int:
    try:
        return max(1, int(os.environ.get("RKK_TASK_NAV_AI_EVERY", "2")))
    except ValueError:
        return 2


def _owm_live_refresh_every() -> int:
    try:
        return max(1, int(os.environ.get("RKK_OWM_LIVE_REFRESH_EVERY", "2")))
    except ValueError:
        return 2


def _task_nav_wm_min_steps() -> int:
    try:
        return max(0, int(os.environ.get("RKK_TASK_NAV_WM_MIN_STEPS", "0")))
    except ValueError:
        return 0


def _graph_nid_to_motor_intent(nid: str) -> str | None:
    from engine.features.humanoid.constants import MOTOR_INTENT_VARS

    s = str(nid)
    if s in MOTOR_INTENT_VARS:
        return s
    if s.startswith("phys_intent_"):
        suf = s[len("phys_intent_") :]
        if suf in MOTOR_INTENT_VARS:
            return suf
    return None


_NAV_RANGE_SCALE_M = 5.0


def _encode_nav_bearing_01(bearing: float) -> float:
    """Map normalized bearing [-1, 1] to [0, 1] with 0.5 = facing target."""
    b = float(max(-1.0, min(1.0, bearing)))
    return float(max(0.0, min(1.0, 0.5 + 0.5 * b)))


def _encode_nav_range_01(range_m: float, *, scale_m: float = _NAV_RANGE_SCALE_M) -> float:
    return float(max(0.0, min(1.0, float(range_m) / max(float(scale_m), 0.05))))


def _active_inf_return_all_enabled() -> bool:
    return os.environ.get("RKK_ACTIVE_INF_RETURN_ALL", "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


_KINDS_NEEDING_TARGET = frozenset({"reduce_distance", "contact", "displace"})


class SimulationGroundedLanguageMixin:
    def _ensure_task_binding(self) -> TaskBindingController:
        tb = getattr(self, "_task_binding", None)
        if tb is None:
            tb = TaskBindingController()
            self._task_binding = tb
        return tb

    def _ensure_task_tree(self) -> TaskTreeController:
        tt = getattr(self, "_task_tree_ctrl", None)
        if tt is None:
            tt = TaskTreeController()
            self._task_tree_ctrl = tt
        return tt

    def _task_log_reset_session(self, tick: int) -> None:
        self._task_log_session_start_tick = int(tick)
        self._task_log_fall_count = 0
        self._task_log_last_stage_id = None
        self._task_log_prev_fallen = False
        self._task_log_finished_logged = False

    def _task_log_cancel_if_active(self, tick: int, *, reason: str = "preempted") -> None:
        if getattr(self, "_task_log_finished_logged", False):
            return
        if getattr(self, "_task_log_session_start_tick", None) is None:
            return
        tt = getattr(self, "_task_tree_ctrl", None)
        tb = getattr(self, "_task_binding", None)
        tree_active = tt is not None and tt.is_active
        bind_active = tb is not None and tb.active_task is not None
        if tree_active or bind_active:
            self._task_log_finished(
                tick,
                status="cancelled",
                reason=reason,
            )

    def _task_log_finished(
        self,
        tick: int,
        *,
        status: str,
        reason: str = "",
        final_pe: float | None = None,
    ) -> None:
        if getattr(self, "_task_log_finished_logged", False):
            return
        self._task_log_finished_logged = True
        start = int(getattr(self, "_task_log_session_start_tick", tick) or tick)
        tb = getattr(self, "_task_binding", None)
        task = tb.active_task if tb is not None else None
        if task is None and tb is not None:
            task = getattr(tb, "_active", None)
        if final_pe is None and task is not None:
            try:
                final_pe = float(getattr(task, "last_pe", None))
            except (TypeError, ValueError):
                final_pe = None
        task_log_event(
            "task_finished",
            tick=int(tick),
            status=str(status),
            reason=str(reason or "")[:200],
            duration_ticks=max(0, int(tick) - start),
            fall_count=int(getattr(self, "_task_log_fall_count", 0)),
            final_pe=final_pe,
        )
        self._task_log_session_start_tick = None

    def _task_log_command_received(
        self,
        tick: int,
        text: str,
        *,
        command_kind: str,
        tag: str,
    ) -> None:
        self._task_log_reset_session(tick)
        task_log_event(
            "command_received",
            tick=int(tick),
            text=str(text)[:120],
            command_kind=str(command_kind),
            tag=str(tag),
        )

    def _task_log_target_resolution(
        self,
        tick: int,
        query: str,
        resolved: Any | None,
        diag: dict[str, Any],
    ) -> None:
        fields: dict[str, Any] = {"query": str(query)[:120]}
        if resolved is not None:
            fields["ref"] = getattr(resolved, "ref", None)
            fields["semantic"] = getattr(resolved, "semantic", None)
            fields["movable"] = getattr(resolved, "movable", None)
            pos = {}
            for k in ("x", "y", "z"):
                try:
                    v = getattr(resolved, k, None)
                    if v is not None:
                        pos[k] = float(v)
                except (TypeError, ValueError):
                    pass
            if pos:
                fields["pos"] = pos
        else:
            fields["reason"] = str(diag.get("reason", "no_target"))
        for key in (
            "scene_semantics",
            "semantics",
            "candidates",
            "slot_peakiness",
            "mask_peakiness_min",
            "geometry",
            "geometry_fallback",
            "guided_uv",
            "peak_strength",
            "confidence_pre_floor",
            "ontology_score",
            "range_m",
            "range_conf",
            "u",
            "v",
            "slot_id",
            "label",
            "reason",
            "ontology",
            "best_score",
            "min_conf",
            "objectness_bind",
            "objectness_bind_attempt",
            "refused_geometry_fallback",
            "latent_reid_attempt",
            "source",
        ):
            if key in diag and diag[key] is not None:
                fields[key] = diag[key]
        task_log_event("target_resolution", tick=int(tick), **fields)
        try:
            from engine.neural_logger import neural_log_event, summarize_slot_table

            neural_log_event(
                "vision",
                "resolve",
                tick=int(tick),
                force=True,
                reason=fields.get("reason"),
                slot_id=fields.get("slot_id"),
                label=fields.get("label"),
                geometry=fields.get("geometry") or fields.get("geometry_fallback"),
                objectness_bind=fields.get("objectness_bind"),
                ontology_score=fields.get("ontology_score"),
                best_score=fields.get("best_score"),
                range_m=fields.get("range_m"),
                peak_strength=fields.get("peak_strength"),
                source=fields.get("source"),
                candidates=summarize_slot_table(diag.get("candidates") or fields.get("candidates")),
                slot_peakiness=summarize_slot_table(diag.get("slot_peakiness")),
                objectness_bind_attempt=diag.get("objectness_bind_attempt"),
                latent_reid_attempt=diag.get("latent_reid_attempt"),
            )
        except Exception:
            pass

    def _task_log_tree_bound(self, tick: int, tt: TaskTreeController) -> None:
        tree = tt.tree
        if tree is None:
            return
        nodes: list[dict[str, str]] = []
        root = tree.nodes.get(tree.root_id)
        if root is not None:
            for cid in root.children:
                node = tree.nodes.get(cid)
                if node is None:
                    continue
                nodes.append(
                    {
                        "id": str(node.id),
                        "kind": str(node.kind),
                        "label": str(node.label),
                    }
                )
        task_log_event(
            "tree_bound",
            tick=int(tick),
            session_id=str(tree.session_id),
            nodes=nodes,
        )
        self._task_log_stage_started(tick, tt.active_node)

    def _task_log_imagine_done(self, tick: int, task: Any) -> None:
        es = dict(getattr(task, "expected_state", {}) or {})
        summary = summarize_expected_state(es)
        diag = dict(getattr(task, "last_diag", {}) or {})
        task_log_event(
            "imagine_done",
            tick=int(tick),
            homeo_veto=diag.get("homeo_veto"),
            **summary,
        )

    def _task_log_stage_started(self, tick: int, node: Any | None) -> None:
        if node is None:
            return
        nid = str(getattr(node, "id", "") or "")
        if nid and nid == str(getattr(self, "_task_log_last_stage_id", "") or ""):
            return
        self._task_log_last_stage_id = nid or None
        task_log_event(
            "stage_started",
            tick=int(tick),
            node_id=nid or None,
            kind=str(getattr(node, "kind", "")),
            label=str(getattr(node, "label", "")),
            attempts=int(getattr(node, "attempts", 0) or 0),
        )

    def _task_log_stage_done(
        self,
        tick: int,
        node: Any,
        *,
        diagnostics: dict[str, Any] | None = None,
    ) -> None:
        pe = getattr(node, "last_pe", None)
        if diagnostics:
            pe = diagnostics.get("pe_total", diagnostics.get("last_pe", pe))
        task_log_event(
            "stage_done",
            tick=int(tick),
            node_id=str(getattr(node, "id", "")),
            kind=str(getattr(node, "kind", "")),
            label=str(getattr(node, "label", "")),
            last_pe=pe,
        )

    def _task_log_stage_failed(self, tick: int, node: Any, reason: str) -> None:
        task_log_event(
            "stage_failed",
            tick=int(tick),
            node_id=str(getattr(node, "id", "")),
            kind=str(getattr(node, "kind", "")),
            label=str(getattr(node, "label", "")),
            attempts=int(getattr(node, "attempts", 0) or 0),
            failure_reason=str(reason)[:200],
            last_pe=getattr(node, "last_pe", None),
        )

    def _task_log_stage_retry(self, tick: int, node: Any, reason: str) -> None:
        from engine.task_tree import task_replan_max

        task_log_event(
            "stage_retry",
            tick=int(tick),
            node_id=str(getattr(node, "id", "")),
            kind=str(getattr(node, "kind", "")),
            label=str(getattr(node, "label", "")),
            attempts=int(getattr(node, "attempts", 0) or 0),
            max_retries=int(task_replan_max()),
            failure_reason=str(reason)[:200],
            new_deadline=getattr(node, "tick_deadline", None),
        )
        try:
            self._add_event(
                f"↻ retry {getattr(node, 'kind', '?')}: {reason} "
                f"(attempt {int(getattr(node, 'attempts', 0) or 0)}/"
                f"{int(task_replan_max())})",
                "#ffaa44",
                "value",
            )
        except Exception:
            pass
        # 6C: time-deduped chat REPORT (not every internal retry).
        self._maybe_emit_retry_chat(tick, node, reason)

    def _retry_report_cooldown_ticks(self) -> int:
        try:
            return max(60, int(os.environ.get("RKK_TASK_RETRY_REPORT_COOLDOWN_TICKS", "800")))
        except ValueError:
            return 800

    def _maybe_emit_retry_chat(self, tick: int, node: Any, reason: str) -> None:
        """One chat REPORT per cooldown window while stage retries silently extend."""
        t = int(tick)
        last = int(getattr(self, "_last_retry_chat_tick", -10**9) or -10**9)
        if t - last < self._retry_report_cooldown_ticks():
            return
        self._last_retry_chat_tick = t
        kind = str(getattr(node, "kind", "") or "?")
        attempts = int(getattr(node, "attempts", 0) or 0)
        body = (
            f"Пока не получается на этапе «{kind}» "
            f"(попытка {attempts}): {str(reason)[:80]}. Пробую ещё раз."
        )
        tt = getattr(self, "_task_tree_ctrl", None)
        cmd = ""
        if tt is not None and getattr(tt, "tree", None) is not None:
            cmd = str(tt.tree.command_text or "")
        self._emit_task_report(t, cmd or kind, done=False, body=body)

    def _tt_complete_active(
        self,
        tt: TaskTreeController,
        tick: int,
        diagnostics: dict[str, Any] | None = None,
    ) -> Any:
        active = tt.active_node
        result = tt.complete_active(tick, diagnostics=diagnostics)
        if active is not None:
            self._task_log_stage_done(tick, active, diagnostics=diagnostics)
        nxt = tt.active_node
        if nxt is not None and (active is None or str(nxt.id) != str(active.id)):
            self._task_log_stage_started(tick, nxt)
        return result

    def _tt_fail_active(
        self,
        tt: TaskTreeController,
        tick: int,
        reason: str,
        *,
        retryable: bool = False,
    ) -> Any:
        active = tt.active_node
        result = tt.fail_active(tick, reason, retryable=retryable)
        if active is None:
            return result
        if retryable and tt.active_node is active:
            # Deadline extended in-place — do not pretend the stage failed.
            self._task_log_stage_retry(tick, active, reason)
            return result
        self._task_log_stage_failed(tick, active, reason)
        return result

    def _task_log_human_motor_targets(self) -> dict[str, float]:
        """
        Dump steering-relevant intents actually requested this tick.

        Prefer live arbiter ``navigation`` / ``human_task`` registrations.
        Also surface ``motor_final_*`` from motor_state (what CPG saw last
        finalize) so gait_coupling is observable even when tree fallback
        only has static approach stride/torso.
        """
        out: dict[str, float] = {}
        arb = getattr(self, "_motor_arbiter", None)
        if arb is not None:
            for mi in list(getattr(arb, "_intents", []) or []):
                src = str(getattr(mi, "source", "") or "")
                if src not in ("human_task", "navigation"):
                    continue
                as_map = getattr(mi, "as_field_map", None)
                if callable(as_map):
                    for k, v in as_map().items():
                        sk = str(k)
                        if sk.startswith("intent_"):
                            try:
                                out[sk] = round(float(v), 4)
                            except (TypeError, ValueError):
                                pass
        cached = getattr(self, "_last_nav_intents", None)
        if isinstance(cached, dict):
            for k, v in cached.items():
                sk = str(k)
                if sk.startswith("intent_") and sk not in out:
                    try:
                        out[sk] = round(float(v), 4)
                    except (TypeError, ValueError):
                        pass
        # Applied / previous-tick finals — proves whether coupling reached CPG.
        try:
            agent = getattr(self, "agent", None)
            base = getattr(getattr(agent, "env", None), "base_env", None) or getattr(
                agent, "env", None
            )
            ms = getattr(base, "_motor_state", None) if base is not None else None
            if isinstance(ms, dict):
                for k in (
                    "intent_gait_coupling",
                    "intent_stride",
                    "intent_torso_forward",
                    "intent_support_left",
                    "intent_support_right",
                ):
                    if k in ms:
                        out[f"motor_final_{k[len('intent_'):]}"] = round(
                            float(ms[k]), 4
                        )
        except Exception:
            pass
        if not any(k.startswith("intent_") for k in out):
            try:
                for k, v in self.task_tree_motor_targets().items():
                    sk = str(k)
                    if sk.startswith("intent_"):
                        out[sk] = round(float(v), 4)
            except Exception:
                pass
        return out

    def _task_log_progress(self, tick: int, *, obs: dict[str, float], fallen: bool) -> None:
        try:
            progress_every = max(1, int(os.environ.get("RKK_TASK_LOG_PROGRESS_EVERY", "25")))
        except ValueError:
            progress_every = 25
        if int(tick) % progress_every != 0:
            return
        if getattr(self, "_task_log_session_start_tick", None) is None:
            return
        tt = getattr(self, "_task_tree_ctrl", None)
        tb = getattr(self, "_task_binding", None)
        tree_active = tt is not None and tt.is_active
        bind_active = tb is not None and tb.active_task is not None
        if not tree_active and not bind_active:
            return

        node_kind = ""
        target_ref = ""
        if tree_active and tt is not None:
            active = tt.active_node
            if active is not None:
                node_kind = str(active.kind)
                target_ref = str(getattr(active, "target_ref", "") or "")

        last_pe = None
        max_pe = None
        task_text = ""
        task = tb.active_task if tb is not None else None
        if task is None and tb is not None:
            task = getattr(tb, "_active", None)
        if task is not None:
            task_text = str(getattr(task, "text", "") or "")[:120]
            try:
                last_pe = float(getattr(task, "last_pe", 0.0))
            except (TypeError, ValueError):
                pass
            try:
                max_pe = float(getattr(task, "max_prediction_error", 0.0))
            except (TypeError, ValueError):
                pass

        dist = None
        if target_ref:
            target_xy = self._target_xy(target_ref)
            if target_xy is not None:
                agent_xy, _ = self._agent_xy_forward()
                dist = round(
                    math.hypot(target_xy[0] - agent_xy[0], target_xy[1] - agent_xy[1]),
                    4,
                )

        macro_hint = ""
        try:
            ic = getattr(self, "_intention_state", None) or getattr(
                self, "_intention_cortex", None
            )
            if ic is not None:
                ctx = getattr(ic, "_last_context", ic)
                macro_hint = str(getattr(ctx, "macro_hint", "") or "")
        except Exception:
            pass

        fields: dict[str, Any] = {
            "node_kind": node_kind or None,
            "task_text": task_text or None,
            "macro_hint": macro_hint or None,
            "last_pe": last_pe,
            "max_pe": max_pe,
            "com_x": obs.get("com_x"),
            "com_y": obs.get("com_y"),
            "com_x_vel": obs.get("com_x_vel"),
            "target_dist": dist,
            "target_ref": target_ref or None,
            "fallen": bool(fallen),
            "posture_stability": obs.get("posture_stability"),
            "human_task_motor": self._task_log_human_motor_targets(),
        }
        try:
            nav_meta = getattr(self, "_last_nav_meta", None) or {}
            if isinstance(nav_meta, dict) and nav_meta:
                fields["task_nav_mode"] = str(nav_meta.get("task_nav_mode") or "")
                fields["nav_ai_ok"] = bool(nav_meta.get("nav_ai_ok"))
                if nav_meta.get("nav_ai_reason"):
                    fields["nav_ai_reason"] = str(nav_meta.get("nav_ai_reason"))
        except Exception:
            pass
        try:
            nodes = getattr(getattr(self, "agent", None), "graph", None)
            nodes = getattr(nodes, "nodes", {}) if nodes is not None else {}
            for k in ("task_heading_err", "task_closing_vel", "task_nav_active"):
                if k in obs:
                    fields[k] = obs.get(k)
                elif isinstance(nodes, dict) and k in nodes:
                    fields[k] = nodes.get(k)
        except Exception:
            pass
        try:
            st = self._tick_phys_state()
            if isinstance(st, dict):
                fields["com_x_m"] = round(float(st.get("com_x", 0.0)), 4)
                fields["com_y_m"] = round(float(st.get("com_y", 0.0)), 4)
        except Exception:
            pass
        if obs.get("task_target_dist_m") is not None:
            fields["task_target_dist_m"] = obs.get("task_target_dist_m")
        # Vision OWM range + oracle eval distance + closing velocity
        try:
            owm = getattr(self, "_obj_working_memory", None)
            if owm is not None and float(getattr(owm, "range_m", 0.0) or 0.0) > 0.05:
                fields["vision_range_m"] = round(float(owm.range_m), 4)
                fields["vision_bearing"] = round(float(owm.bearing), 4)
                fields["hard_lock"] = bool(
                    getattr(getattr(owm, "scene", None), "hard_lock_active", False)
                )
                # Drift probe: live scan bearing vs odometry-owned lock bearing.
                try:
                    scene = getattr(owm, "scene", None)
                    act = scene.active() if scene is not None else None
                    diags = dict(getattr(act, "diagnostics", None) or {}) if act else {}
                    if "live_bearing" in diags:
                        fields["live_bearing"] = round(float(diags["live_bearing"]), 4)
                    if "bearing_live_delta" in diags:
                        fields["bearing_live_delta"] = round(
                            float(diags["bearing_live_delta"]), 4
                        )
                    if "bearing_nudge" in diags:
                        fields["bearing_nudge"] = round(float(diags["bearing_nudge"]), 4)
                    if "kalman_gain" in diags:
                        fields["kalman_gain"] = round(float(diags["kalman_gain"]), 4)
                except Exception:
                    pass
                scene = getattr(owm, "scene", None)
                if scene is not None and bool(
                    getattr(scene, "last_odom_discontinuity", False)
                ):
                    fields["odom_discontinuity"] = True
                    fields["odom_jump_m"] = round(
                        float(getattr(scene, "last_odom_jump_m", 0.0) or 0.0), 3
                    )
            oref = None
            resolved = getattr(self, "_manip_resolved", None)
            if resolved is not None:
                oref = str(getattr(resolved, "ref", "") or "")
            if not oref:
                diag = getattr(self, "_manip_diag", None) or {}
                oref = str((diag.get("oracle_eval") or {}).get("ref") or "")
            if oref and not str(oref).startswith("vision:"):
                od = self._oracle_dist_m_for_eval(oref)
                if od is not None:
                    fields["oracle_dist_m"] = round(float(od), 4)
            # Closing: decrease in vision range over progress samples
            prev = getattr(self, "_task_log_prev_vision_range", None)
            if fields.get("vision_range_m") is not None and prev is not None:
                fields["closing_vel"] = round(
                    float(prev) - float(fields["vision_range_m"]), 4
                )
            if fields.get("vision_range_m") is not None:
                self._task_log_prev_vision_range = float(fields["vision_range_m"])
        except Exception:
            pass
        arb = getattr(self, "_motor_arbiter", None)
        if arb is not None:
            snap = getattr(arb, "_last_diag", None) or {}
            if isinstance(snap, dict) and snap.get("sources"):
                fields["motor_sources"] = list(snap.get("sources") or [])
        task_log_event("task_progress", tick=int(tick), **fields)
        try:
            from engine.neural_logger import neural_log_event, summarize_latent

            owm = getattr(self, "_obj_working_memory", None)
            scene = getattr(owm, "scene", None) if owm is not None else None
            act = scene.active() if scene is not None and hasattr(scene, "active") else None
            diags = dict(getattr(act, "diagnostics", None) or {}) if act else {}
            neural_log_event(
                "owm",
                "track",
                tick=int(tick),
                bearing=fields.get("vision_bearing"),
                range_m=fields.get("vision_range_m"),
                hard_lock=fields.get("hard_lock"),
                kalman_gain=fields.get("kalman_gain"),
                live_bearing=fields.get("live_bearing"),
                bearing_nudge=fields.get("bearing_nudge"),
                bearing_live_delta=fields.get("bearing_live_delta"),
                fusion_source=diags.get("source"),
                latent_cos=diags.get("latent_cos"),
                latent=summarize_latent(getattr(act, "latent", None)) if act else None,
                entity_id=getattr(act, "entity_id", None) if act else None,
                slot_id=getattr(act, "slot_id", None) if act else None,
                task_nav_mode=fields.get("task_nav_mode"),
                nav_ai_ok=fields.get("nav_ai_ok"),
                nav_ai_reason=fields.get("nav_ai_reason"),
                node_kind=fields.get("node_kind"),
            )
        except Exception:
            pass

    def _task_log_fall_during_task(self, tick: int) -> None:
        if getattr(self, "_task_log_session_start_tick", None) is None:
            return
        tt = getattr(self, "_task_tree_ctrl", None)
        tb = getattr(self, "_task_binding", None)
        if not (
            (tt is not None and tt.is_active)
            or (tb is not None and tb.active_task is not None)
        ):
            return
        self._task_log_fall_count = int(getattr(self, "_task_log_fall_count", 0)) + 1
        task_log_event(
            "fall_during_task",
            tick=int(tick),
            fall_count=int(self._task_log_fall_count),
        )
        self._arm_nav_hold(int(tick), reason="fall")

    def _humanoid_base_env(self) -> Any | None:
        env = getattr(getattr(self, "agent", None), "env", None)
        if env is None:
            return None
        return getattr(env, "base_env", env)

    def _humanoid_physics_sim(self) -> Any | None:
        """Unwrap to the object that owns PyBullet static body registry / contacts."""
        base = self._humanoid_base_env()
        if base is None:
            return None
        sim = getattr(base, "_sim", None)
        if sim is not None and (
            hasattr(sim, "_static_body_registry")
            or callable(getattr(sim, "find_static_contact_body", None))
            or callable(getattr(sim, "_manip_has_contact", None))
        ):
            return sim
        if hasattr(base, "_static_body_registry") or callable(
            getattr(base, "find_static_contact_body", None)
        ):
            return base
        return base

    def _sandbox_scene_extras(self) -> dict:
        """Scene objects from embodiment (sim or real robot driver), not resolver heuristics."""
        base = self._humanoid_base_env()
        if base is None:
            return {}
        candidates: list[Any] = [base]
        sim = getattr(base, "_sim", None)
        if sim is not None and sim is not base:
            candidates.append(sim)
        for obj in candidates:
            for name in (
                "get_scene_extras",
                "get_sandbox_scene_extras",
                "get_physics_object_positions",
            ):
                fn = getattr(obj, name, None)
                if not callable(fn):
                    continue
                try:
                    return dict(fn() or {})
                except Exception:
                    continue
        return {}

    def _agent_xy_forward(self) -> tuple[tuple[float, float], tuple[float, float]]:
        base = self._humanoid_base_env()
        xy = (0.0, 0.0)
        fwd = (1.0, 0.0)
        if base is not None:
            try:
                pose_fn = getattr(base, "get_task_agent_pose", None)
                if callable(pose_fn):
                    pose = pose_fn()
                    if isinstance(pose, dict):
                        raw_xy = pose.get("xy")
                        raw_fwd = pose.get("forward")
                        if raw_xy is not None and len(raw_xy) >= 2:
                            xy = (float(raw_xy[0]), float(raw_xy[1]))
                        if raw_fwd is not None and len(raw_fwd) >= 2:
                            fwd = (float(raw_fwd[0]), float(raw_fwd[1]))
                            if vision_resolve_enabled():
                                cam_fwd_fn = getattr(base, "get_ego_camera_forward_xy", None)
                                if callable(cam_fwd_fn):
                                    try:
                                        cf = cam_fwd_fn()
                                        if cf is not None and len(cf) >= 2:
                                            fwd = (float(cf[0]), float(cf[1]))
                                    except Exception:
                                        pass
                            return xy, fwd
            except Exception:
                pass
            try:
                st = self._tick_phys_state()
                if isinstance(st, dict):
                    xy = (float(st.get("com_x", 0.0)), float(st.get("com_y", 0.0)))
                    yaw = float(st.get("torso_yaw", st.get("yaw", 0.0)))
                    fwd = (math.cos(yaw), math.sin(yaw))
                    return xy, fwd
            except Exception:
                pass
            try:
                sim = getattr(base, "_sim", base)
                raw = sim.get_state() if hasattr(sim, "get_state") else {}
                if isinstance(raw, dict):
                    xy = (float(raw.get("com_x", 0.0)), float(raw.get("com_y", 0.0)))
                    yaw = float(raw.get("torso_yaw", raw.get("yaw", 0.0)))
                    fwd = (math.cos(yaw), math.sin(yaw))
            except Exception:
                pass
        return xy, fwd

    def _inject_task_obs(self, obs: dict[str, float]) -> dict[str, float]:
        """Merge live task geometry + contact into obs and agent graph."""
        resolved = getattr(self, "_manip_resolved", None)
        goal = getattr(self, "_task_goal", None)
        tt = getattr(self, "_task_tree_ctrl", None)
        tree_active = tt is not None and bool(getattr(tt, "is_active", False))
        if resolved is None and goal is None and not tree_active:
            return obs

        ctx = self._human_task_verify_ctx()
        task_obs = build_task_observations(
            agent_xy=ctx.get("agent_xy"),
            target_xy=ctx.get("target_xy"),
            contact=float(ctx.get("contact", 0.0)),
        )
        # Ego memory distance when vision path has no world target_xy
        if "task_target_dist_m" not in task_obs and ctx.get("distance_m") is not None:
            task_obs["task_target_dist_m"] = float(ctx["distance_m"])
        for k in (
            "task_target_x",
            "task_target_y",
            "task_target_conf",
            "vision_bearing",
            "vision_range_m",
        ):
            if k in ctx:
                task_obs[k] = float(ctx[k])
        merged = inject_task_observations(obs, task_obs)
        try:
            sync_task_obs_to_graph(self.agent.graph, task_obs)
        except Exception:
            pass
        return merged

    def _infer_manip_direction(
        self,
        text: str,
        *,
        target_xy: tuple[float, float] | None = None,
        embed_fn: Any | None = None,
    ) -> tuple[float, float]:
        from engine.goal_grounding import infer_manip_direction

        agent_xy, agent_fwd = self._agent_xy_forward()
        return infer_manip_direction(
            text,
            agent_xy=agent_xy,
            target_xy=target_xy,
            agent_forward=agent_fwd,
            embed_fn=embed_fn,
        )

    def _clear_human_command_state(self, tick: int) -> None:
        self._task_log_cancel_if_active(tick, reason="preempted")
        tb = getattr(self, "_task_binding", None)
        if tb is not None:
            tb.clear()
        tt = getattr(self, "_task_tree_ctrl", None)
        if tt is not None and tt.is_active:
            tt.cancel(int(tick), "preempted")
        self._manip_episode = None
        self._manip_resolved = None
        self._manip_resolved_visual = None
        self._clear_object_working_memory()
        self._manip_diag = {}
        self._task_tree_kind = ""
        self._task_goal = None
        self._task_goal_verified = False
        self._task_tree_reported = False
        self._task_tree_affect_done = False
        s2 = getattr(self, "_system2", None)
        if s2 is not None and hasattr(s2, "working_memory"):
            wm = s2.working_memory
            wm.write("human_task_active", 0.0, tick=int(tick), source="human_command")
            wm.write("human_task_pe", 0.0, tick=int(tick), source="human_command")
        ic = getattr(self, "_intention_cortex", None)
        if ic is not None and hasattr(ic, "clear_human_command"):
            ic.clear_human_command()
        try:
            graph = self.agent.graph
            if "self_goal_active" in graph.nodes:
                graph.nodes["self_goal_active"] = 0.0
        except Exception:
            pass

    def _complete_human_command_cleanup(self, tick: int) -> None:
        self._clear_human_command_state(tick)
        self._manip_episode = None
        self._manip_resolved = None
        self._manip_resolved_visual = None
        self._clear_object_working_memory()
        self._task_tree_kind = ""
        self._task_goal = None
        self._task_goal_verified = False

    def task_tree_motor_targets(self) -> dict[str, float]:
        if not task_tree_enabled():
            return {}
        tt = getattr(self, "_task_tree_ctrl", None)
        if tt is None or not tt.is_active:
            return {}
        return dict(tt.motor_targets())

    def _task_tree_snapshot(self) -> dict[str, Any]:
        if not task_tree_enabled():
            return {"enabled": False, "active": False}
        tt = getattr(self, "_task_tree_ctrl", None)
        tick = int(getattr(self, "tick", 0))
        if tt is None:
            return {"enabled": True, "active": False, "tick": tick}
        snap = dict(tt.snapshot(tick))
        snap["enabled"] = True
        if self._manip_diag:
            snap["manipulation"] = dict(self._manip_diag)
        return snap

    def _ensure_grounded_language(self) -> None:
        if getattr(self, "_grounded_lang_ready", False):
            return
        if not grounded_language_enabled():
            self._grounded_lang = None
            self._grounded_lang_ready = True
            return
        self._grounded_lang = GroundedLanguageController(device=self.device)
        try:
            n = self._grounded_lang.bootstrap_store()
            if n:
                print(f"[GroundedLang] bootstrapped {n} phrase embeddings")
            from engine.goal_grounding import warm_predicate_catalog

            warmed = warm_predicate_catalog(self._grounded_lang.embedder.embed)
            if warmed:
                print(f"[GroundedLang] warmed {warmed} predicate catalog embeddings")
        except Exception as e:
            print(f"[GroundedLang] bootstrap skipped: {e}")
        self._grounded_lang_ready = True

    def _ensure_grounded_motor_drain_hook(self) -> None:
        if getattr(self, "_grounded_motor_hooked", False):
            return
        arb = getattr(self, "_motor_arbiter", None)
        if arb is None:
            return
        orig = arb.begin_tick
        sim = self

        def hooked_begin_tick() -> None:
            orig()
            sim._drain_pending_grounded_motor()

        arb.begin_tick = hooked_begin_tick  # type: ignore[method-assign]
        self._grounded_motor_hooked = True

    def _queue_grounded_motor_intent(self, tag: str) -> None:
        motor = motor_intents_from_tag(tag)
        if not motor:
            self._pending_grounded_motor = None
            return
        self._pending_grounded_motor = dict(motor)

    def _drain_pending_grounded_motor(self) -> None:
        pending = getattr(self, "_pending_grounded_motor", None)
        if not pending:
            return
        arb = getattr(self, "_motor_arbiter", None)
        if arb is not None:
            arb.register_from_dict("grounded_language", dict(pending))
        self._pending_grounded_motor = None

    def _tick_grounded_language(self, *, fallen: bool) -> None:
        if not grounded_language_enabled():
            return
        tick = int(getattr(self, "tick", 0))
        if tick % _grounded_lang_every() != 0:
            return
        self._ensure_grounded_language()
        self._ensure_grounded_motor_drain_hook()
        gl = getattr(self, "_grounded_lang", None)
        if gl is None or not hasattr(self, "agent"):
            return
        graph = self.agent.graph
        obs: dict[str, float] = {}
        try:
            obs = dict(self._graph_vec_cached())
        except Exception:
            try:
                obs = dict(self.agent.env.observe())
            except Exception:
                pass
        gl.ensure_graph_nodes(graph)
        env = getattr(getattr(self, "agent", None), "env", None)
        fallen_flag: bool | None = bool(fallen) if fallen else None
        try:
            base = getattr(env, "base_env", env) if env is not None else None
            if base is not None and callable(getattr(base, "is_fallen", None)):
                fallen_flag = bool(base.is_fallen())
        except Exception:
            pass
        human_task_text = ""
        human_task_stage = ""
        try:
            from engine.task_executive import active_tree_stage_kind

            tb = getattr(self, "_task_binding", None)
            ht = tb.active_task if tb is not None else None
            if ht is None and tb is not None:
                ht = getattr(tb, "_active", None)
            if ht is not None:
                human_task_text = str(getattr(ht, "text", "") or "")
            human_task_stage = active_tree_stage_kind(self)
        except Exception:
            pass
        gl.sync_speak_vector_from_state(
            graph,
            obs,
            fallen=fallen_flag,
            env=env,
            human_task_text=human_task_text,
            human_task_stage=human_task_stage,
        )
        if fallen and not task_binding_enabled():
            gl.ingest_command(graph, "Встань, ты упал", apply_motor_patch=True)
        if tick % _align_every() == 0 and obs:
            try:
                gl.alignment_step(graph, obs, fallen=fallen_flag, env=env)
            except Exception:
                pass

    def _emit_task_speech(
        self,
        tick: int,
        body: str,
        *,
        speech_type: Any,
        curiosity: float = 0.5,
    ) -> None:
        verbal = getattr(self, "_verbal", None)
        if verbal is None or not body or len(str(body).strip()) < 2:
            return
        try:
            from engine.verbal_action import AgentMessage

            msg = AgentMessage(
                tick=int(tick),
                speech_type=speech_type,
                text=str(body).strip(),
                concepts=["HUMAN_TASK"],
                curiosity=float(curiosity),
                posture=0.5,
            )
            verbal._messages.append(msg)
            if str(getattr(speech_type, "name", "")) == "REPORT":
                verbal._last_report_tick = int(tick)
            verbal.total_messages += 1
            for cb in verbal._on_message:
                try:
                    cb(msg)
                except Exception:
                    pass
        except Exception:
            pass

    def _emit_task_report(
        self,
        tick: int,
        command_text: str,
        *,
        done: bool,
        body: str | None = None,
    ) -> None:
        try:
            from engine.verbal_action import SpeechType, ollama_chat_speech_enabled
        except Exception:
            return
        if body is None:
            if ollama_chat_speech_enabled():
                body = self.grounded_lang_generate()
                if not body or len(body.strip()) < 3:
                    return
            elif done:
                body = self.grounded_lang_generate()
                if not body or len(body.strip()) < 3:
                    body = f"Готово: {command_text[:60]}"
            else:
                body = f"Не удалось: {command_text[:60]}"
        self._emit_task_speech(tick, str(body), speech_type=SpeechType.REPORT)

    def _emit_task_ask(self, tick: int, body: str) -> None:
        """6C terminal / escalate: ASK human for clarification (not spam-retry REPORT)."""
        try:
            from engine.verbal_action import SpeechType
        except Exception:
            return
        self._emit_task_speech(
            tick, body, speech_type=SpeechType.ASK, curiosity=0.85
        )

    def _apply_task_outcome_affect(self, success: bool) -> None:
        if getattr(self, "_task_tree_affect_done", False):
            return
        self._task_tree_affect_done = True
        try:
            from engine.intristic_objective import instrumental_task_bonus

            instrumental_task_bonus("human_command", success, sim=self)
        except Exception:
            pass
        base = self._humanoid_base_env()
        if base is not None and hasattr(base, "apply_task_outcome_affect"):
            try:
                base.apply_task_outcome_affect(success)
            except Exception:
                pass

    def _maybe_finalize_task_tree(self, tick: int) -> None:
        if getattr(self, "_task_tree_reported", False):
            return
        tt = getattr(self, "_task_tree_ctrl", None)
        if tt is None or tt.tree is None:
            return
        root_status = str(tt.tree.root_status)
        if root_status not in TERMINAL_STATUSES:
            return
        success = root_status == "done"
        goal = getattr(self, "_task_goal", None)
        if success and goal is not None and any(
            str(p.kind) == "contact" for p in (goal.predicates or [])
        ):
            if not getattr(self, "_task_goal_verified", False):
                try:
                    obs = dict(self._graph_vec_cached())
                except Exception:
                    obs = {}
                obs = self._inject_task_obs(obs)
                ok, _, _ = evaluate_goal(goal, obs, self._human_task_verify_ctx())
                if not ok:
                    success = False
                    tt.tree.root_status = "failed"
                    for node in tt.tree.nodes.values():
                        if node.kind == "verify_goal" and node.status == "done":
                            node.status = "failed"
                            node.failure_reason = "verify_goal_failed"
                            break
        cmd = str(tt.tree.command_text or "")
        self._task_tree_reported = True
        if success:
            self._emit_task_report(tick, cmd, done=True)
        else:
            reason = ""
            for node in tt.tree.nodes.values():
                if node.failure_reason:
                    reason = str(node.failure_reason)
                    break
            vision_uncertain = reason in (
                "uncertain_no_peaked_slot",
                "low_vision_confidence",
                "resolve_failed_vision",
                "no_language_vision_link",
                "missing_or_invalid_range",
                "floor_lock_rejected",
                "no_vision_slots",
                "weak_objectness_peak",
                "active_percept_exhausted",
            )
            if reason.startswith("no_target") or "static" in reason:
                body = "Не вижу цель или не могу сдвинуть объект."
                self._emit_task_report(tick, cmd, done=False, body=body)
            elif vision_uncertain:
                # 6C: terminal ASK — need human clarification, not silent fail.
                self._emit_task_ask(
                    tick,
                    "Не уверен, что правильно вижу цель. "
                    "Уточните объект или направление?",
                )
                self._emit_task_report(
                    tick,
                    cmd,
                    done=False,
                    body="Не вижу цель в камере (зрение не закрепило объект).",
                )
            else:
                body = f"Не удалось: {cmd[:60]}"
                self._emit_task_report(tick, cmd, done=False, body=body)
        self._apply_task_outcome_affect(success)
        reason = ""
        final_pe = None
        for node in tt.tree.nodes.values():
            if node.failure_reason:
                reason = str(node.failure_reason)
            if node.last_pe is not None:
                final_pe = float(node.last_pe)
        tb = getattr(self, "_task_binding", None)
        if tb is not None:
            bound = getattr(tb, "_active", None)
            if bound is not None and getattr(bound, "last_pe", None) is not None:
                final_pe = float(bound.last_pe)
        self._task_log_finished(
            tick,
            status="done" if success else "failed",
            reason=reason,
            final_pe=final_pe,
        )
        s2 = getattr(self, "_system2", None)
        if s2 is not None and hasattr(s2, "working_memory"):
            wm = s2.working_memory
            wm.write(
                "human_task_active",
                0.0,
                text=cmd[:80],
                tick=int(tick),
                source="task_binding",
            )
            wm.write(
                "human_task_status",
                1.0 if success else 0.0,
                text=root_status,
                tick=int(tick),
                source="task_binding",
            )
        ic = getattr(self, "_intention_cortex", None)
        if ic is not None and hasattr(ic, "clear_human_command"):
            ic.clear_human_command()
        try:
            graph = self.agent.graph
            if "self_goal_active" in graph.nodes:
                graph.nodes["self_goal_active"] = 0.0
        except Exception:
            pass
        tt.clear(int(tick))
        self._task_tree_cleared_pending_ack = True

    def _target_xy(self, ref: str) -> tuple[float, float] | None:
        base = self._humanoid_base_env()
        if base is not None:
            pose = base.get_manipulation_target_pose(ref)
            if pose:
                return float(pose.get("x", 0.0)), float(pose.get("y", 0.0))
        try:
            from engine.object_resolver import collect_scene_candidates

            for row in collect_scene_candidates(self._sandbox_scene_extras()):
                if str(row.get("ref")) == str(ref):
                    return float(row["x"]), float(row["y"])
        except Exception:
            pass
        return None

    def _world_xy_from_owm(self, owm: ObjectWorkingMemory) -> tuple[float, float] | None:
        """Egocentric OWM bearing/range → world XY via agent pose."""
        if float(getattr(owm, "range_m", 0.0) or 0.0) < 0.05:
            return None
        agent_xy, agent_fwd = self._agent_xy_forward()
        x_fwd, y_right = ego_from_bearing_range(float(owm.bearing), float(owm.range_m))
        fx, fy = float(agent_fwd[0]), float(agent_fwd[1])
        n = float(math.hypot(fx, fy)) or 1.0
        fx, fy = fx / n, fy / n
        rx, ry = fy, -fx
        ax, ay = float(agent_xy[0]), float(agent_xy[1])
        return (ax + x_fwd * fx + y_right * rx, ay + x_fwd * fy + y_right * ry)

    def _task_ontology_best_key(self) -> str | None:
        """Visual-referent ontology key from active task command / goal text."""
        cmd = ""
        tt = getattr(self, "_task_tree_ctrl", None)
        if tt is not None and tt.tree is not None:
            cmd = str(tt.tree.command_text or "")
        if not cmd:
            goal = getattr(self, "_task_goal", None)
            if goal is not None:
                cmd = str(getattr(goal, "text", "") or "")
        if not cmd.strip():
            return None
        gl = getattr(self, "_grounded_lang", None)
        embed_fn = gl.embedder.embed if gl is not None and getattr(gl, "embedder", None) else None
        if embed_fn is None:
            return None
        try:
            from engine.visual_referent_ontology import match_visual_referent

            _, score, diag = match_visual_referent(cmd, embed_fn)
            if float(score) < 0.25:
                return None
            key = (diag or {}).get("best_key")
            return str(key) if key else None
        except Exception:
            return None

    def _forward_cylinder_contact_body(
        self,
        *,
        vision_range: float | None = None,
        prefer_planter: bool = True,
        max_dist_m: float = 8.0,
    ) -> int | None:
        """Pick a static cylinder ahead of the agent (physics registry, not oracle control).

        For cylinder ontology (`prefer_planter`), only planter-style bodies compete —
        cafe chrome/glass legs must not win on raw proximity.
        """
        base = self._humanoid_physics_sim()
        if base is None:
            return None
        agent_xy, fwd = self._agent_xy_forward()
        ax, ay = float(agent_xy[0]), float(agent_xy[1])
        fx, fy = float(fwd[0]), float(fwd[1])
        fn = float(math.hypot(fx, fy))
        if fn > 1e-6:
            fx, fy = fx / fn, fy / fn
        rows = [
            row
            for row in (getattr(base, "_static_body_registry", []) or [])
            if str(row.get("kind", "")) == "cylinder"
        ]
        if prefer_planter:
            planters = [row for row in rows if str(row.get("style", "")) == "planter"]
            if planters:
                rows = planters
        best_id: int | None = None
        best_score = float("inf")
        for row in rows:
            bx = float(row.get("x", 0.0))
            by = float(row.get("y", 0.0))
            dx, dy = bx - ax, by - ay
            ahead = dx * fx + dy * fy
            horiz = float(math.hypot(dx, dy))
            # Soft behind-gate: allow near-lateral targets; only drop clear rear ones.
            if horiz > 1e-6 and ahead < -0.45 * horiz:
                continue
            radius = float(row.get("radius", 0.0))
            d = max(0.0, horiz - radius)
            if d > float(max_dist_m):
                continue
            score = float(d)
            if vision_range is not None and float(vision_range) > 0.05:
                score += 0.35 * abs(float(d) - float(vision_range))
            # Larger planters are the scene landmark for "cylindrical object".
            if str(row.get("style", "")) == "planter":
                score -= 0.25 * min(radius, 2.0)
            bid = row.get("body_id")
            if bid is not None and score < best_score:
                best_score = score
                best_id = int(bid)
        return best_id

    def _contact_body_id_for_task(self, resolved: ResolvedObject | None) -> int | None:
        body_id = getattr(resolved, "body_id", None) if resolved is not None else None
        if body_id is not None:
            return int(body_id)
        base = self._humanoid_physics_sim()
        if base is None:
            return None
        owm = getattr(self, "_obj_working_memory", None)
        tick = int(getattr(self, "tick", 0))
        fn = getattr(base, "find_static_contact_body", None)
        if not callable(fn):
            return None
        from engine.vision_resolve import _is_visual_concept
        from engine.task_observation import contact_reach_m

        ont_key = str(self._task_ontology_best_key() or "").strip().lower()
        vt = getattr(self, "_manip_resolved_visual", None)
        label_src = str(getattr(vt, "label", "") or getattr(owm, "label", "") or "")
        visual_label = label_src.lower() if _is_visual_concept(label_src) else ""
        prefer_cyl = (
            ont_key == "cylinder"
            or "cylinder" in ont_key
            or any(k in visual_label for k in ("cylinder", "planter", "column"))
        )
        vision_range = None
        if owm is not None and owm.is_usable(tick):
            vision_range = float(owm.range_m)
        elif vt is not None and vt.range_m is not None:
            vision_range = float(vt.range_m)

        if prefer_cyl:
            bid = self._forward_cylinder_contact_body(
                vision_range=vision_range,
                prefer_planter=True,
                max_dist_m=max(8.0, float(vision_range or 0.0) + 3.0),
            )
            if bid is not None:
                return int(bid)

        probe_xy = None
        if owm is not None and owm.is_usable(tick):
            probe_xy = self._world_xy_from_owm(owm)
        if probe_xy is None:
            agent_xy, _ = self._agent_xy_forward()
            probe_xy = (float(agent_xy[0]), float(agent_xy[1]))
        max_d = float(contact_reach_m())
        if prefer_cyl:
            max_d = max(max_d, 2.5)
            bid = fn(probe_xy, kind="cylinder", style="planter", max_dist_m=max_d)
            if bid is None:
                bid = fn(probe_xy, kind="cylinder", max_dist_m=max_d)
            return int(bid) if bid is not None else None
        style = None
        kind = None
        if "chair" in visual_label or ont_key == "chair":
            kind = "box"
        bid = fn(probe_xy, kind=kind, style=style, max_dist_m=max_d)
        return int(bid) if bid is not None else None

    def _cylinder_contact_body_ids_near_agent(self, max_dist_m: float = 2.5) -> list[int]:
        """All cylinder static bodies within reach of agent COM (physics contact probe)."""
        base = self._humanoid_physics_sim()
        if base is None:
            return []
        agent_xy, _ = self._agent_xy_forward()
        ax, ay = float(agent_xy[0]), float(agent_xy[1])
        out: list[int] = []
        for row in getattr(base, "_static_body_registry", []) or []:
            if str(row.get("kind", "")) != "cylinder":
                continue
            bx = float(row.get("x", 0.0))
            by = float(row.get("y", 0.0))
            d = float(math.hypot(ax - bx, ay - by))
            d -= float(row.get("radius", 0.0))
            if d <= float(max_dist_m):
                bid = row.get("body_id")
                if bid is not None:
                    out.append(int(bid))
        return out

    def _static_registry_row_for_body(self, body_id: int) -> dict | None:
        base = self._humanoid_physics_sim()
        if base is None:
            return None
        for row in getattr(base, "_static_body_registry", []) or []:
            bid = row.get("body_id")
            if bid is not None and int(bid) == int(body_id):
                return row
        return None

    def _lock_task_contact_body_on_bind(self, vt: VisualTarget | None = None) -> None:
        """Pin static body_id at vision bind for physics range (not oracle XY control)."""
        prev = getattr(self, "_task_locked_body_id", None)
        if prev is not None:
            prev_row = self._static_registry_row_for_body(int(prev))
            # Keep an existing planter lock across vision rebinds unless it is gone.
            if prev_row is not None and str(prev_row.get("style", "")) == "planter":
                try:
                    task_log_event(
                        "task_body_lock",
                        tick=int(getattr(self, "tick", 0)),
                        locked_body_id=int(prev),
                        label=str(getattr(vt, "label", "") or ""),
                        style="planter",
                        kept=True,
                        radius_m=round(float(prev_row.get("radius", 0.0)), 4),
                    )
                except Exception:
                    pass
                return
        resolved = getattr(self, "_manip_resolved", None)
        body_id = self._contact_body_id_for_task(resolved)
        if body_id is None:
            from engine.vision_resolve import _is_visual_concept

            ont_key = str(self._task_ontology_best_key() or "").strip().lower()
            label_src = ""
            if vt is not None:
                label_src = str(vt.label or "")
            else:
                owm = getattr(self, "_obj_working_memory", None)
                if owm is not None:
                    label_src = str(getattr(owm, "label", "") or "")
            visual_label = label_src.lower() if _is_visual_concept(label_src) else ""
            prefer_cyl = (
                ont_key == "cylinder"
                or "cylinder" in ont_key
                or any(k in visual_label for k in ("cylinder", "planter", "column"))
            )
            if prefer_cyl:
                base = self._humanoid_physics_sim()
                vision_range = None
                if vt is not None and vt.range_m is not None:
                    vision_range = float(vt.range_m)
                body_id = self._forward_cylinder_contact_body(
                    vision_range=vision_range,
                    prefer_planter=True,
                    max_dist_m=8.0,
                )
                if body_id is None:
                    fn = getattr(base, "find_static_contact_body", None) if base is not None else None
                    if callable(fn):
                        agent_xy, _ = self._agent_xy_forward()
                        probe = (float(agent_xy[0]), float(agent_xy[1]))
                        body_id = fn(probe, kind="cylinder", style="planter", max_dist_m=2.5)
                        if body_id is None:
                            body_id = fn(probe, kind="cylinder", max_dist_m=2.5)
                        if body_id is None:
                            body_id = fn(probe, max_dist_m=2.5)
        self._task_locked_body_id = int(body_id) if body_id is not None else None
        row = (
            self._static_registry_row_for_body(int(body_id))
            if body_id is not None
            else None
        )
        try:
            task_log_event(
                "task_body_lock",
                tick=int(getattr(self, "tick", 0)),
                locked_body_id=self._task_locked_body_id,
                label=str(getattr(vt, "label", "") or ""),
                style=str((row or {}).get("style", "") or ""),
                kind=str((row or {}).get("kind", "") or ""),
                x_m=round(float((row or {}).get("x", 0.0)), 4) if row else None,
                y_m=round(float((row or {}).get("y", 0.0)), 4) if row else None,
                radius_m=round(float((row or {}).get("radius", 0.0)), 4) if row else None,
            )
        except Exception:
            pass

    def _physics_range_to_locked_body(self) -> float | None:
        """COM XY distance to bound static body (surface distance for cylinders)."""
        bid = getattr(self, "_task_locked_body_id", None)
        if bid is None:
            return None
        base = self._humanoid_physics_sim()
        if base is None:
            return None
        agent_xy, _ = self._agent_xy_forward()
        ax, ay = float(agent_xy[0]), float(agent_xy[1])
        row = self._static_registry_row_for_body(int(bid))
        kind = str(row.get("kind", "")) if row is not None else ""
        radius = float(row.get("radius", 0.0)) if row is not None else 0.0
        bx: float | None = None
        by: float | None = None
        if row is not None:
            bx = float(row.get("x", 0.0))
            by = float(row.get("y", 0.0))
        try:
            import pybullet as pb

            client = getattr(base, "client", None)
            lock = getattr(base, "_physics_lock", None)
            if lock is not None:
                lock.acquire()
            try:
                p, _ = pb.getBasePositionAndOrientation(int(bid), physicsClientId=client)
                bx, by = float(p[0]), float(p[1])
            finally:
                if lock is not None:
                    lock.release()
        except Exception:
            pass
        if bx is None or by is None:
            return None
        d = float(math.hypot(ax - bx, ay - by))
        if kind in ("cylinder", "sphere"):
            d = max(0.0, d - radius)
        return float(d)

    def _blend_dist_with_physics_range(
        self,
        dist: float,
        owm_range: float | None,
        tick: int,
    ) -> float:
        phys = self._physics_range_to_locked_body()
        if phys is None:
            return float(dist)
        # Physics is ground truth for stage gates once a body is locked.
        # Optimistic vision (much closer than phys) must not complete approach early.
        if float(dist) + 0.45 < float(phys):
            blended = float(phys)
            self._maybe_relock_body_on_optimistic_vision(
                int(tick),
                phys=float(phys),
                vision=float(dist),
            )
        else:
            blended = min(float(dist), float(phys))
        prev_log = int(getattr(self, "_task_physics_range_log_tick", -9999))
        if int(tick) - prev_log >= 30:
            self._task_physics_range_log_tick = int(tick)
            try:
                task_log_event(
                    "task_physics_range",
                    tick=int(tick),
                    phys_m=round(float(phys), 4),
                    owm_range_m=(
                        round(float(owm_range), 4) if owm_range is not None else None
                    ),
                    vision_dist_m=round(float(dist), 4),
                    blended_m=round(float(blended), 4),
                    locked_body_id=int(getattr(self, "_task_locked_body_id", 0) or 0),
                )
            except Exception:
                pass
        return blended

    def _maybe_relock_body_on_optimistic_vision(
        self,
        tick: int,
        *,
        phys: float,
        vision: float,
    ) -> None:
        """When vision << physics, locked body is likely wrong — reselect forward cylinder."""
        until = int(getattr(self, "_body_relock_until_tick", -1) or -1)
        if int(tick) < until:
            return
        if float(phys) - float(vision) < 0.8:
            return
        streak = int(getattr(self, "_optimistic_vision_relock_streak", 0)) + 1
        self._optimistic_vision_relock_streak = streak
        if streak < 2:
            return
        self._optimistic_vision_relock_streak = 0
        self._body_relock_until_tick = int(tick) + 25
        bid = self._forward_cylinder_contact_body(
            vision_range=float(vision),
            prefer_planter=True,
            max_dist_m=8.0,
        )
        if bid is None:
            return
        prev = getattr(self, "_task_locked_body_id", None)
        # Only switch if the candidate is meaningfully closer than the current lock.
        if prev is not None:
            row_new = self._static_registry_row_for_body(int(bid))
            agent_xy, _ = self._agent_xy_forward()
            ax, ay = float(agent_xy[0]), float(agent_xy[1])
            new_phys = float(phys)
            if row_new is not None:
                nx = float(row_new.get("x", 0.0))
                ny = float(row_new.get("y", 0.0))
                new_phys = max(
                    0.0,
                    float(math.hypot(ax - nx, ay - ny))
                    - float(row_new.get("radius", 0.0)),
                )
            if new_phys >= float(phys) - 0.2:
                return
        if prev is not None and int(prev) == int(bid):
            return
        self._task_locked_body_id = int(bid)
        row = self._static_registry_row_for_body(int(bid))
        try:
            task_log_event(
                "task_body_relock",
                tick=int(tick),
                prev_body_id=int(prev) if prev is not None else None,
                locked_body_id=int(bid),
                phys_m=round(float(phys), 4),
                vision_m=round(float(vision), 4),
                reason="optimistic_vision",
                style=str((row or {}).get("style", "") or ""),
                radius_m=round(float((row or {}).get("radius", 0.0)), 4) if row else None,
            )
        except Exception:
            pass

    def _maybe_rebind_on_physics_range_desync(
        self,
        tick: int,
        *,
        phys: float,
        owm_range: float | None,
        kind: str,
        stop: float,
    ) -> None:
        """Soft-unlock + objectness rebind when COM is near locked body but OWM range drifts high."""
        if not vision_resolve_enabled():
            return
        if kind not in ("approach", "approach_target"):
            return
        if float(phys) >= float(stop):
            return
        if owm_range is None or float(owm_range) - float(phys) < 0.6:
            return

        until = int(getattr(self, "_physics_range_rebind_until_tick", -1) or -1)
        if int(tick) < until:
            return

        streak = int(getattr(self, "_physics_range_desync_streak", 0)) + 1
        self._physics_range_desync_streak = streak
        if streak < 3:
            return
        self._physics_range_desync_streak = 0
        self._physics_range_rebind_until_tick = int(tick) + 40

        scene = getattr(self, "_latent_scene", None)
        cam = self._depth_camera_from_sim()
        if scene is not None and cam is not None:
            try:
                scene.refresh_active_from_live_camera(
                    cam,
                    tick=int(tick),
                    range_hint=float(phys),
                    blend=1.0,
                )
            except Exception:
                pass

        owm = getattr(self, "_obj_working_memory", None)
        bearing_hint = float(owm.bearing) if owm is not None else None
        self._rebind_vision_objectness_peak(
            int(tick),
            reason="physics_range_desync",
            oracle_dist=float(phys),
            bearing_hint=bearing_hint,
            allow_full_resolve=True,
        )

    def _manip_has_contact(self, resolved: ResolvedObject | None) -> bool:
        env = getattr(getattr(self, "agent", None), "env", None)
        if env is not None and bool(getattr(env, "_contact_flag", False)):
            return True
        base = self._humanoid_physics_sim()
        if base is None:
            return False
        fn = getattr(base, "_manip_has_contact", None)
        if not callable(fn):
            return False

        body_id = self._contact_body_id_for_task(resolved)
        locked = getattr(self, "_task_locked_body_id", None)
        if locked is not None and bool(fn(int(locked))):
            self._log_task_contact_detected(int(locked))
            return True
        if body_id is not None and bool(fn(int(body_id))):
            self._log_task_contact_detected(int(body_id))
            return True

        owm = getattr(self, "_obj_working_memory", None) or getattr(
            self, "_owm_cached", None
        )
        range_m = float(getattr(owm, "range_m", 0.0) or 0.0) if owm is not None else 999.0
        phys = self._physics_range_to_locked_body()
        near = min(
            range_m,
            float(phys) if phys is not None else range_m,
        )
        ont_key = str(self._task_ontology_best_key() or "").strip().lower()
        raw_label = str(getattr(owm, "label", "") or "") if owm is not None else ""
        from engine.vision_resolve import _is_visual_concept

        visual_label = raw_label.lower() if _is_visual_concept(raw_label) else ""
        prefer_cyl = (
            ont_key == "cylinder"
            or "cylinder" in ont_key
            or any(k in visual_label for k in ("cylinder", "planter", "column"))
        )
        # Scan nearby cylinders when physics/vision say we are in reach (not OWM-only).
        if prefer_cyl and near < max(float(contact_reach_m()), 1.2):
            for bid in self._cylinder_contact_body_ids_near_agent():
                if locked is not None and int(bid) == int(locked):
                    continue
                if body_id is not None and int(bid) == int(body_id):
                    continue
                if bool(fn(int(bid))):
                    self._log_task_contact_detected(int(bid))
                    return True
        return False

    def _log_task_contact_detected(self, body_id: int) -> None:
        prev = int(getattr(self, "_task_contact_logged_body_id", -1) or -1)
        if prev == int(body_id):
            return
        self._task_contact_logged_body_id = int(body_id)
        owm = getattr(self, "_obj_working_memory", None) or getattr(
            self, "_owm_cached", None
        )
        range_m = float(getattr(owm, "range_m", 0.0) or 0.0) if owm is not None else None
        agent_xy, _ = self._agent_xy_forward()
        try:
            task_log_event(
                "task_contact_detected",
                tick=int(getattr(self, "tick", 0)),
                body_id=int(body_id),
                range_m=round(float(range_m), 4) if range_m is not None else None,
                com_x_m=round(float(agent_xy[0]), 4),
                com_y_m=round(float(agent_xy[1]), 4),
            )
        except Exception:
            pass

    def _visual_env_ref(self) -> Any | None:
        return getattr(self, "_visual_env", None)

    def _depth_camera_from_sim(self) -> ArrayDepthCamera | None:
        """Capture ego RGB-D and wrap as DepthCamera (sim backend)."""
        base = self._humanoid_base_env()
        if base is None:
            return None
        fn = getattr(base, "get_ego_rgbd", None)
        if not callable(fn):
            sim = getattr(base, "_sim", None)
            fn = getattr(sim, "get_ego_rgbd", None) if sim is not None else None
        if not callable(fn):
            return None
        try:
            pack = fn(view="ego", width=160, height=120)
        except Exception:
            return None
        if not isinstance(pack, dict) or pack.get("depth_m") is None:
            return None
        try:
            import numpy as np

            frame = DepthFrame(
                depth_m=np.asarray(pack["depth_m"], dtype=np.float32),
                near_m=float(pack.get("near_m", 0.1)),
                far_m=float(pack.get("far_m", 15.0)),
            )
            return ArrayDepthCamera(frame)
        except Exception:
            return None

    def _latent_scene_memory(self) -> LatentSceneMemory:
        scene = getattr(self, "_latent_scene", None)
        if scene is None:
            scene = LatentSceneMemory()
            self._latent_scene = scene
            # Keep facade in sync for any legacy readers
            self._obj_working_memory = ObjectWorkingMemory(scene)
        return scene

    def _object_working_memory(self) -> ObjectWorkingMemory:
        self._latent_scene_memory()
        owm = getattr(self, "_obj_working_memory", None)
        if owm is None:
            owm = ObjectWorkingMemory(self._latent_scene_memory())
            self._obj_working_memory = owm
        return owm

    def _clear_object_working_memory(self) -> None:
        scene = getattr(self, "_latent_scene", None)
        if scene is not None:
            scene.reset()
        else:
            self._latent_scene = LatentSceneMemory()
        self._obj_working_memory = ObjectWorkingMemory(self._latent_scene)
        self._deferred_vision_resolve = None
        self._task_log_prev_vision_range = None
        self._owm_cached_tick = -1
        self._owm_cached = None
        self._task_locked_body_id = None

    def _inject_owm_into_graph(self, owm: ObjectWorkingMemory | LatentSceneMemory) -> None:
        if isinstance(owm, LatentSceneMemory):
            payload = owm.graph_payload(tick=int(getattr(self, "tick", 0)))
        else:
            payload = owm.graph_payload()
        graph = self.agent.graph
        node_ids = set(getattr(graph, "_node_ids", []) or [])
        for k, v in payload.items():
            if k not in node_ids and hasattr(graph, "set_node"):
                try:
                    graph.set_node(k, float(v))
                    node_ids.add(k)
                except Exception:
                    pass
            nodes = graph.nodes
            nodes[k] = float(v)
            pk = f"phys_{k}"
            if pk in nodes:
                nodes[pk] = float(v)
        act = None
        if isinstance(owm, LatentSceneMemory):
            act = owm.active()
        else:
            act = owm.scene.active() if getattr(owm, "scene", None) is not None else None
        if act is not None:
            self._ensure_nav_vision_graph_nodes(graph)
            b01 = _encode_nav_bearing_01(float(act.bearing))
            r01 = _encode_nav_range_01(float(act.range_m))
            graph.nodes["vision_bearing_01"] = b01
            graph.nodes["vision_range_01"] = r01
            graph.nodes["vision_bearing"] = float(act.bearing)
            graph.nodes["vision_range_m"] = float(act.range_m)

    def _ensure_nav_vision_graph_nodes(self, graph: Any) -> None:
        """Register [0,1] vision observation nodes for Active Inference approach."""
        if not hasattr(graph, "set_node"):
            return
        node_ids = set(getattr(graph, "_node_ids", []) or [])
        for nid, default in (
            ("vision_bearing_01", 0.5),
            ("vision_range_01", 0.5),
        ):
            if nid not in node_ids:
                try:
                    graph.set_node(nid, float(default))
                except Exception:
                    pass

    def _dump_bind_frame(
        self,
        tick: int,
        *,
        visual: VisualTarget | None = None,
        diag: dict[str, Any] | None = None,
    ) -> None:
        """RGB+depth+slot table at bind for threshold-vs-model diagnosis."""
        try:
            from engine.task_logger import task_log_dir

            dump_root = task_log_dir() / "bind_dumps" / f"tick_{int(tick)}"
            dump_root.mkdir(parents=True, exist_ok=True)
            meta: dict[str, Any] = {
                "tick": int(tick),
                "mask_peakiness_min": float(
                    (diag or {}).get("mask_peakiness_min")
                    if diag and diag.get("mask_peakiness_min") is not None
                    else 1.8
                ),
                "slot_peakiness": list((diag or {}).get("slot_peakiness") or []),
                "candidates": list((diag or {}).get("candidates") or []),
                "geometry": (diag or {}).get("geometry"),
                "geometry_fallback": (diag or {}).get("geometry_fallback"),
                "guided_uv": (diag or {}).get("guided_uv"),
                "peak_strength": (diag or {}).get("peak_strength"),
                "confidence_pre_floor": (diag or {}).get("confidence_pre_floor"),
                "ontology_score": (diag or {}).get("ontology_score"),
                "range_m": (diag or {}).get("range_m"),
                "reason": (diag or {}).get("reason"),
            }
            if visual is not None:
                meta["visual"] = visual.to_dict()
                meta["confidence_pre_bind_floor"] = round(float(visual.confidence), 4)
            rgbd = None
            try:
                rgbd = self._get_ego_rgbd()
            except Exception:
                rgbd = None
            if isinstance(rgbd, dict):
                rgb = rgbd.get("rgb")
                depth = rgbd.get("depth_m")
                if rgb is not None:
                    try:
                        from PIL import Image
                        import numpy as np

                        Image.fromarray(np.asarray(rgb, dtype=np.uint8)).save(
                            dump_root / "rgb.png"
                        )
                        meta["rgb_shape"] = list(np.asarray(rgb).shape)
                    except Exception as exc:
                        meta["rgb_error"] = str(exc)[:120]
                if depth is not None:
                    try:
                        import numpy as np

                        d = np.asarray(depth, dtype=np.float32)
                        np.save(dump_root / "depth_m.npy", d)
                        # Vis: clip 0–6 m → uint8
                        d_vis = np.clip(d, 0.0, 6.0) / 6.0
                        from PIL import Image

                        Image.fromarray((d_vis * 255.0).astype(np.uint8)).save(
                            dump_root / "depth_vis.png"
                        )
                        meta["depth_shape"] = list(d.shape)
                        meta["depth_min"] = float(np.nanmin(d)) if d.size else None
                        meta["depth_max"] = float(np.nanmax(d)) if d.size else None
                        # Depth at guided / visual UV
                        u = float(
                            ((diag or {}).get("guided_uv") or {}).get("u")
                            or (visual.u if visual else 0.5)
                        )
                        v = float(
                            ((diag or {}).get("guided_uv") or {}).get("v")
                            or (visual.v if visual else 0.5)
                        )
                        h, w = int(d.shape[0]), int(d.shape[1])
                        yi = int(max(0, min(h - 1, round(v * (h - 1)))))
                        xi = int(max(0, min(w - 1, round(u * (w - 1)))))
                        meta["depth_at_guided_uv"] = {
                            "u": u,
                            "v": v,
                            "xi": xi,
                            "yi": yi,
                            "range_m": float(d[yi, xi]),
                        }
                    except Exception as exc:
                        meta["depth_error"] = str(exc)[:120]
            (dump_root / "meta.json").write_text(
                json.dumps(meta, ensure_ascii=False, indent=2, default=str),
                encoding="utf-8",
            )
            task_log_event(
                "bind_dump",
                tick=int(tick),
                path=str(dump_root),
                confidence_pre_floor=(diag or {}).get("confidence_pre_floor"),
                geometry=(diag or {}).get("geometry"),
                peak_strength=(diag or {}).get("peak_strength"),
                range_m=(diag or {}).get("range_m"),
                n_slots=len(meta.get("slot_peakiness") or []),
            )
        except Exception as exc:
            try:
                task_log_event(
                    "bind_dump",
                    tick=int(tick),
                    error=str(exc)[:160],
                )
            except Exception:
                pass

    def _get_ego_rgbd(self) -> dict[str, Any] | None:
        base = self._humanoid_base_env()
        fn = getattr(base, "get_ego_rgbd", None) if base is not None else None
        if not callable(fn):
            sim = getattr(self, "sim", None)
            fn = getattr(sim, "get_ego_rgbd", None) if sim is not None else None
        if not callable(fn):
            return None
        try:
            return fn(view="ego", width=160, height=120)
        except TypeError:
            try:
                return fn(width=160, height=120)
            except Exception:
                return None
        except Exception:
            return None

    def _bind_object_working_memory(self, vt: VisualTarget, tick: int) -> None:
        conf = float(vt.confidence)
        agent_xy, agent_fwd = self._agent_xy_forward()
        scene = self._latent_scene_memory()
        scene.bind_visual_target(
            vt, tick=int(tick), agent_xy=agent_xy, agent_forward=agent_fwd
        )
        # 4A: confidence is stored/logged as-is (no artificial 0.5 floor).
        task_log_event(
            "owm_bind",
            tick=int(tick),
            slot_id=str(vt.slot_id),
            label=str(vt.label or ""),
            u=round(float(vt.u), 4),
            v=round(float(vt.v), 4),
            range_m=None if vt.range_m is None else round(float(vt.range_m), 4),
            confidence=round(conf, 4),
            confidence_pre_floor=round(conf, 4),
            confidence_post_floor=round(conf, 4),
            hard_lock_active=True,
            non_production=bool((vt.diagnostics or {}).get("non_production")),
            source=(vt.diagnostics or {}).get("source"),
            geometry=(vt.diagnostics or {}).get("geometry"),
            peak_strength=(vt.diagnostics or {}).get("objectness_peak_strength"),
            guided_uv=(vt.diagnostics or {}).get("guided_uv"),
        )
        self._inject_owm_into_graph(scene)
        if vt.range_m is not None:
            self._owm_bind_range_m = float(vt.range_m)
            self._owm_range_ema = float(vt.range_m)
        self._vision_recede_streak = 0
        self._owm_cached_tick = -1
        self._owm_cached = None
        self._lock_task_contact_body_on_bind(vt)

    def _collect_scene_percepts(self) -> list[dict[str, Any]]:
        """
        Metric percepts for scene memory.

        Active hard-lock: skip vision refresh for active (odometry only).
        Other slots: only if attention is spatially peaked.
        """
        cam = self._depth_camera_from_sim()
        if cam is None:
            return []
        scene = self._latent_scene_memory()
        slots = collect_vision_slots(self._visual_env_ref())
        by_id = {str(s.get("slot_id") or ""): s for s in slots}
        out: list[dict[str, Any]] = []
        active_ids = set(scene.active_ids)
        hard_lock = bool(getattr(scene, "hard_lock_active", False))

        # 1) Refresh active entities at tracked UV — skipped under hard-lock
        if not hard_lock:
            for eid in list(scene.active_ids):
                ent = scene.entities.get(eid)
                if ent is None:
                    continue
                u, v = float(ent.u), float(ent.v)
                s = by_id.get(eid) or by_id.get(ent.slot_id)
                peaked = bool(s and s.get("uv_valid"))
                range_m = None
                conf = float(ent.confidence)
                try:
                    if peaked and s is not None:
                        mask = s.get("attn_mask")
                        guided = getattr(cam, "range_from_attention", None)
                        if callable(guided) and mask is not None:
                            gu, gv, r, _var, rconf = guided(mask)
                            if r is not None and gu is not None and gv is not None:
                                u = 0.7 * u + 0.3 * float(gu)
                                v = 0.7 * v + 0.3 * float(gv)
                                range_m = float(r)
                                if rconf is not None:
                                    conf = max(conf, float(rconf))
                    if range_m is None:
                        r, _var, rconf = cam.range_at_uv(u, v, window=4)
                        range_m = r
                        if rconf is not None:
                            conf = max(conf, float(rconf))
                except Exception:
                    range_m = None
                if range_m is None or float(range_m) <= 0.05:
                    continue
                label = str((s or {}).get("label") or ent.label or "")
                lat = (s or {}).get("vector")
                if lat is None:
                    lat = getattr(ent, "latent", None) or None
                item: dict[str, Any] = {
                    "slot_id": str(ent.slot_id or eid),
                    "u": u,
                    "v": v,
                    "bearing": bearing_from_u(u),
                    "range_m": float(range_m),
                    "label": label,
                    "activation": float(
                        (s or {}).get("activation") or ent.activation or 0.5
                    ),
                    "confidence": conf,
                }
                if lat is not None:
                    item["vector"] = lat
                    item["latent"] = lat
                out.append(item)

        # 2) Discover other peaked slots (scene context), skip diffuse ones
        for s in slots:
            sid = str(s.get("slot_id") or "")
            if not sid or sid in active_ids:
                continue
            if not s.get("uv_valid"):
                continue
            u = float(s.get("u", 0.5))
            v = float(s.get("v", 0.5))
            range_m = None
            conf = float(s.get("activation") or 0.0)
            try:
                mask = s.get("attn_mask")
                guided = getattr(cam, "range_from_attention", None)
                if callable(guided) and mask is not None:
                    gu, gv, r, _var, rconf = guided(mask)
                    if r is not None:
                        u, v = float(gu), float(gv)
                        range_m = float(r)
                        if rconf is not None:
                            conf = max(conf, float(rconf))
                if range_m is None:
                    r, _var, rconf = cam.range_at_uv(u, v)
                    range_m = r
                    if rconf is not None:
                        conf = max(conf, float(rconf))
            except Exception:
                continue
            if range_m is None or float(range_m) <= 0.05:
                continue
            item = {
                "slot_id": sid,
                "u": u,
                "v": v,
                "bearing": bearing_from_u(u),
                "range_m": float(range_m),
                "label": str(s.get("label") or ""),
                "activation": float(s.get("activation") or 0.0),
                "confidence": conf,
            }
            if s.get("vector") is not None:
                item["vector"] = s.get("vector")
                item["latent"] = s.get("vector")
            out.append(item)

        # 3) Hard-lock: still emit metric+latent candidates for cosine re-ID
        # (slot_id may permute after head turn).
        if hard_lock:
            seen_ids = {str(p.get("slot_id") or "") for p in out}
            for s in slots:
                sid = str(s.get("slot_id") or "")
                if not sid or sid in seen_ids or sid in active_ids:
                    continue
                vec = s.get("vector")
                if vec is None:
                    continue
                u = float(s.get("u", 0.5))
                v = float(s.get("v", 0.5))
                range_m = None
                conf = float(s.get("activation") or 0.35)
                try:
                    r, _var, rconf = cam.range_at_uv(u, v)
                    range_m = r
                    if rconf is not None:
                        conf = max(conf, float(rconf))
                except Exception:
                    continue
                if range_m is None or float(range_m) <= 0.05:
                    continue
                out.append(
                    {
                        "slot_id": sid,
                        "u": u,
                        "v": v,
                        "bearing": bearing_from_u(u),
                        "range_m": float(range_m),
                        "label": str(s.get("label") or ""),
                        "activation": float(s.get("activation") or 0.0),
                        "confidence": conf,
                        "vector": vec,
                        "latent": vec,
                        "_hard_lock_reid_cand": True,
                    }
                )
        return out

    def _update_object_working_memory(self, tick: int) -> ObjectWorkingMemory | None:
        tick_i = int(tick)
        if int(getattr(self, "_owm_cached_tick", -1)) == tick_i:
            cached = getattr(self, "_owm_cached", None)
            if cached is not None:
                return cached

        scene = self._latent_scene_memory()
        agent_xy, agent_fwd = self._agent_xy_forward()
        # Prefer full scene percepts; fall back to active visual track only
        percepts = []
        try:
            percepts = self._collect_scene_percepts()
        except Exception:
            percepts = []
        if not percepts:
            vt = None
            try:
                vt = self._refresh_visual_target()
            except Exception:
                vt = getattr(self, "_manip_resolved_visual", None)
            if vt is not None and vt.is_ready(require_range=True):
                percepts = [
                    {
                        "slot_id": vt.slot_id,
                        "bearing": vt.bearing,
                        "range_m": vt.range_m,
                        "label": vt.label,
                        "confidence": vt.confidence,
                        "activation": vt.confidence,
                    }
                ]
        scene.update(
            tick=int(tick),
            percepts=percepts,
            agent_xy=agent_xy,
            agent_forward=agent_fwd,
        )
        if bool(getattr(scene, "hard_lock_active", False)) and scene.active_ids:
            live_every = _owm_live_refresh_every()
            if int(tick) % live_every == 0:
                try:
                    cam = self._depth_camera_from_sim()
                    if cam is not None:
                        range_hint = self._eval_oracle_dist_m()
                        scene.refresh_active_from_live_camera(
                            cam,
                            tick=int(tick),
                            range_hint=range_hint,
                            blend=0.78,
                        )
                except Exception:
                    pass
        if bool(getattr(scene, "last_odom_discontinuity", False)):
            jump = float(getattr(scene, "last_odom_jump_m", 0.0) or 0.0)
            prev_logged = int(getattr(self, "_last_odom_disc_log_tick", -9999))
            if int(tick) - prev_logged >= 5:
                self._last_odom_disc_log_tick = int(tick)
                try:
                    task_log_event(
                        "com_teleport",
                        tick=int(tick),
                        jump_m=round(jump, 3),
                        hard_lock=bool(scene.hard_lock_active),
                        com_x_m=round(float(agent_xy[0]), 4),
                        com_y_m=round(float(agent_xy[1]), 4),
                    )
                except Exception:
                    pass
                try:
                    self._add_event(
                        f"⚠ COM teleport {jump:.2f}m (stance reset?) — OWM hold",
                        "#ff8866",
                        "value",
                    )
                except Exception:
                    pass
                try:
                    from engine.task_binding import human_task_execution_active

                    if human_task_execution_active(self):
                        self._owm_unlock_after_teleport(int(tick), reason="com_teleport")
                except Exception:
                    pass
        self._inject_owm_into_graph(scene)
        owm = self._object_working_memory()
        if owm is not None and owm.is_usable(tick_i):
            self._maybe_rebind_vision_on_recede(int(tick), scene, float(owm.range_m))
        self._maybe_rebind_after_teleport(int(tick))
        if owm.is_usable(tick_i):
            self._owm_cached_tick = tick_i
            self._owm_cached = owm
            return owm
        self._owm_cached_tick = tick_i
        self._owm_cached = None
        return None

    def _oracle_dist_m_for_eval(self, ref: str) -> float | None:
        """Privileged distance for metrics only — never used as control input in vision mode."""
        agent_xy, _ = self._agent_xy_forward()
        target_xy = self._target_xy(ref) if ref else None
        if target_xy is None:
            return None
        return float(math.hypot(target_xy[0] - agent_xy[0], target_xy[1] - agent_xy[1]))

    def _refresh_visual_target(self) -> VisualTarget | None:
        vt = getattr(self, "_manip_resolved_visual", None)
        if vt is None:
            return None
        cam = self._depth_camera_from_sim()
        embed_fn = None
        gl = getattr(self, "_grounded_lang", None)
        if gl is not None and getattr(gl, "embedder", None) is not None:
            embed_fn = gl.embedder.embed
        try:
            updated = track_visual_target(
                vt,
                visual_env=self._visual_env_ref(),
                depth_camera=cam,
                embed_fn=embed_fn,
            )
        except Exception:
            updated = attach_range_to_target(vt, cam) if cam is not None else vt
        self._manip_resolved_visual = updated
        return updated

    def _task_control_distance(self, oracle_dist: float) -> float:
        """Distance used for stage gates / obs: OWM range, else vision, else oracle."""
        if vision_resolve_enabled():
            owm = getattr(self, "_obj_working_memory", None)
            if owm is not None and owm.range_m > 0.05 and owm.confidence > 0.05:
                return float(owm.range_m)
            vt = getattr(self, "_manip_resolved_visual", None)
            if vt is not None and vt.range_m is not None and float(vt.range_m) > 0.05:
                return float(vt.range_m)
        return float(oracle_dist)

    @staticmethod
    def _task_fall_assist_progress_min_m() -> float:
        try:
            return float(os.environ.get("RKK_TASK_FALL_ASSIST_PROGRESS_M", "0.15"))
        except ValueError:
            return 0.15

    def _task_fall_approach_range_m(self) -> float | None:
        """Current OWM/vision range for fall-assist progress checks."""
        if vision_resolve_enabled():
            owm = getattr(self, "_obj_working_memory", None)
            tick = int(getattr(self, "tick", 0))
            if owm is not None and owm.is_usable(tick):
                r = float(owm.range_m)
                if r > 0.05:
                    return r
            vt = getattr(self, "_manip_resolved_visual", None)
            if vt is not None and vt.range_m is not None and float(vt.range_m) > 0.05:
                return float(vt.range_m)
        bind_r = getattr(self, "_owm_bind_range_m", None)
        if bind_r is not None and float(bind_r) > 0.05:
            return float(bind_r)
        return None

    def _task_fall_approach_com_dist_m(self) -> float | None:
        """Agent COM distance to approach goal (origin for cylinder, else OWM/target XY)."""
        agent_xy, _ = self._agent_xy_forward()
        ax, ay = float(agent_xy[0]), float(agent_xy[1])
        ont_key = str(self._task_ontology_best_key() or "").lower()
        if ont_key == "cylinder" or "cylinder" in ont_key:
            return float(math.hypot(ax, ay))
        owm = getattr(self, "_obj_working_memory", None)
        tick = int(getattr(self, "tick", 0))
        if owm is not None and owm.is_usable(tick):
            wxy = self._world_xy_from_owm(owm)
            if wxy is not None:
                return float(math.hypot(ax - wxy[0], ay - wxy[1]))
        resolved = getattr(self, "_manip_resolved", None)
        ref = str(getattr(resolved, "ref", "") or "")
        target_xy = self._target_xy(ref) if ref and not ref.startswith("vision:") else None
        if target_xy is not None:
            return float(math.hypot(ax - target_xy[0], ay - target_xy[1]))
        return None

    def _capture_task_fall_approach_baseline(self, dist: float) -> None:
        """Record range/COM at fall edge during approach (assist progress baseline).

        Never worsen an existing baseline — a re-fall near the goal must not
        erase earlier approach progress or fall-assist will teleport away.
        """
        prev_r = getattr(self, "_task_fall_start_range", None)
        new_r = float(dist)
        if prev_r is None:
            self._task_fall_start_range = new_r
        else:
            # Keep the farther baseline so re-falls near the goal still count
            # prior approach progress.
            self._task_fall_start_range = max(float(prev_r), new_r)
        # Prefer the farther (worse) bind/start as the progress reference.
        bind_range = getattr(self, "_owm_bind_range_m", None)
        if bind_range is not None:
            self._task_fall_start_range = max(
                float(self._task_fall_start_range), float(bind_range)
            )
        com_d = self._task_fall_approach_com_dist_m()
        if com_d is not None:
            prev_c = getattr(self, "_task_fall_start_com", None)
            if prev_c is None:
                self._task_fall_start_com = float(com_d)
            else:
                # Keep the farther COM baseline so closing still counts as progress.
                self._task_fall_start_com = max(float(prev_c), float(com_d))

    def _task_fall_assist_near_goal(self) -> bool:
        """True when physically / visually near approach stop — never teleport."""
        stop = float(nav_stop_m()) + 0.35
        phys = self._physics_range_to_locked_body()
        if phys is not None and float(phys) <= stop:
            return True
        cur = self._task_fall_approach_range_m()
        if cur is not None and float(cur) <= stop:
            return True
        return False

    def _task_fall_assist_progress_blocks_reset(self) -> bool:
        """True when fallen approach made enough progress — skip assist teleport."""
        if self._task_fall_assist_near_goal():
            return True
        min_gain = self._task_fall_assist_progress_min_m()
        start_range = getattr(self, "_task_fall_start_range", None)
        bind_range = getattr(self, "_owm_bind_range_m", None)
        baseline_range = start_range if start_range is not None else bind_range
        if baseline_range is None and bind_range is not None:
            baseline_range = bind_range
        current_range = self._task_fall_approach_range_m()
        phys = self._physics_range_to_locked_body()
        if phys is not None:
            current_range = (
                float(phys)
                if current_range is None
                else min(float(current_range), float(phys))
            )
        if baseline_range is not None and current_range is not None:
            if float(baseline_range) - float(current_range) >= min_gain:
                return True
        start_com = getattr(self, "_task_fall_start_com", None)
        current_com = self._task_fall_approach_com_dist_m()
        if start_com is not None and current_com is not None:
            if float(start_com) - float(current_com) >= min_gain:
                return True
        return False

    def _slot_concept_project_fn(self) -> Any | None:
        """NeuralConceptProjector: slot_vec → concept names (language space)."""
        nlg = getattr(self, "_neural_lang", None)
        if nlg is None:
            nlg = getattr(self, "_neural_language", None)
        projector = getattr(nlg, "concept_projector", None) if nlg is not None else None
        if projector is None:
            return None
        concept_store = None
        iv = getattr(self, "_inner_voice", None)
        if iv is not None:
            concept_store = getattr(iv, "concept_store", None)
        idx_to_name = getattr(concept_store, "idx_to_name", None) if concept_store else None

        def _project(vec: Any) -> list[tuple[str, float]]:
            import torch

            from engine.vision_resolve import _is_visual_concept

            try:
                t = torch.as_tensor(vec, dtype=torch.float32, device=getattr(nlg, "device", "cpu"))
                hits = projector.project(t, top_k=8, threshold=0.25)
            except Exception:
                return []
            out: list[tuple[str, float]] = []
            for concept_idx, score in hits or []:
                name = None
                if idx_to_name is not None:
                    name = idx_to_name.get(int(concept_idx))
                if not name:
                    name = f"concept_{int(concept_idx)}"
                if not _is_visual_concept(str(name)):
                    continue
                out.append((str(name), float(score)))
            return out

        return _project

    def _bearing_range_from_world_xy(
        self,
        target_xy: tuple[float, float],
        *,
        agent_xy: tuple[float, float] | None = None,
        agent_fwd: tuple[float, float] | None = None,
    ) -> tuple[float, float] | None:
        if agent_xy is None or agent_fwd is None:
            agent_xy, agent_fwd = self._agent_xy_forward()
        ax, ay = float(agent_xy[0]), float(agent_xy[1])
        tx, ty = float(target_xy[0]), float(target_xy[1])
        dx, dy = tx - ax, ty - ay
        dist = float(math.hypot(dx, dy))
        if dist < 0.05:
            return None
        fx, fy = float(agent_fwd[0]), float(agent_fwd[1])
        n = float(math.hypot(fx, fy)) or 1.0
        fx, fy = fx / n, fy / n
        tcx, tcy = dx / dist, dy / dist
        cross = fx * tcy - fy * tcx
        dot = max(-1.0, min(1.0, fx * tcx + fy * tcy))
        bearing_rad = math.atan2(cross, dot)
        bearing = float(max(-1.0, min(1.0, bearing_rad / (math.pi * 0.5))))
        return bearing, dist

    def _try_latent_reid_visual_bind(
        self,
        *,
        vision_diag: dict[str, Any] | None = None,
        reason: str = "uncertain_resolve",
    ) -> tuple[VisualTarget | None, dict[str, Any]]:
        """
        Cosine re-ID against the last bound entity's SlotAttention latent.

        Prefer this over ``sim_oracle`` when peaked UV resolve fails but the
        previous track embedding still matches a live slot.
        """
        diag: dict[str, Any] = {
            "reason": "latent_reid_miss",
            "resolve_mode": "vision",
            "source": "vision_latent_reid",
            "reid_reason": str(reason),
        }
        scene = self._latent_scene_memory()
        ent = scene.active() if scene is not None else None
        query = list(getattr(ent, "latent", None) or []) if ent is not None else []
        if not query:
            vt_prev = getattr(self, "_manip_resolved_visual", None)
            if vt_prev is not None:
                query = list(getattr(vt_prev, "latent", None) or [])
                if not query:
                    query = list((vt_prev.diagnostics or {}).get("latent") or [])
            if ent is None and vt_prev is not None:
                # Seed a temporary query-only path from prior visual target.
                pass
        if not query:
            diag["reason"] = "latent_reid_no_query"
            return None, diag

        cam = self._depth_camera_from_sim()
        slots = collect_vision_slots(self._visual_env_ref())
        if not slots:
            diag["reason"] = "latent_reid_no_slots"
            return None, diag

        from engine.vision_resolve import _apply_metric_geometry, _cap_spatial_confidence

        candidates: list[dict[str, Any]] = []
        for s in slots:
            cand = dict(s)
            try:
                vt_c = _apply_metric_geometry(cand, cam)
                if vt_c is None or not vt_c.is_ready(require_range=True):
                    continue
                cand["u"] = float(vt_c.u)
                cand["v"] = float(vt_c.v)
                cand["bearing"] = float(vt_c.bearing)
                cand["range_m"] = float(vt_c.range_m or 0.0)
                cand["confidence"] = float(vt_c.confidence)
                cand["label"] = str(vt_c.label or cand.get("label") or "")
                if vt_c.latent:
                    cand["latent"] = list(vt_c.latent)
                candidates.append(cand)
            except Exception:
                continue

        hit = match_latent_slot(candidates, query)
        if hit is None:
            diag["n_candidates"] = len(candidates)
            return None, diag

        lat = list(hit.get("latent") or [])
        conf = float(max(0.35, min(0.95, hit.get("confidence", 0.55) or 0.55)))
        vt = VisualTarget(
            slot_id=str(hit.get("slot_id") or "latent_reid"),
            u=float(hit.get("u", 0.5)),
            v=float(hit.get("v", 0.55)),
            label=str(hit.get("label") or (ent.label if ent is not None else "visual_referent")),
            confidence=conf,
            bearing=float(hit.get("bearing", 0.0)),
            range_m=float(hit.get("range_m")),
            range_conf=float(min(1.0, conf)),
            diagnostics={
                "source": "vision_latent_reid",
                "latent_cos": float(hit.get("latent_cos") or 0.0),
                "reid_reason": str(reason),
                "vision_fail_reason": str((vision_diag or {}).get("reason") or ""),
                "geometry": "latent_reid",
                "latent": lat,
                "latent_dim": len(lat),
            },
            latent=lat or None,
        )
        vt = _cap_spatial_confidence(vt)
        if not vt.is_ready(require_range=True):
            diag["reason"] = "latent_reid_not_ready"
            diag["latent_cos"] = hit.get("latent_cos")
            return None, diag

        out_diag: dict[str, Any] = {
            "reason": "ok_latent_reid",
            "resolve_mode": "vision",
            "source": "vision_latent_reid",
            "slot_id": vt.slot_id,
            "label": vt.label,
            "range_m": vt.range_m,
            "confidence": float(vt.confidence),
            "latent_cos": float(hit.get("latent_cos") or 0.0),
            "reid_reason": str(reason),
            "vision_reason": (vision_diag or {}).get("reason"),
        }
        try:
            task_log_event(
                "vision_latent_reid",
                tick=int(getattr(self, "tick", 0) or 0),
                slot_id=str(vt.slot_id),
                range_m=round(float(vt.range_m or 0.0), 4),
                bearing=round(float(vt.bearing), 4),
                latent_cos=round(float(hit.get("latent_cos") or 0.0), 4),
                reason=str(reason),
                vision_reason=str((vision_diag or {}).get("reason") or ""),
            )
            from engine.neural_logger import neural_log_event, summarize_latent

            neural_log_event(
                "latent",
                "reid",
                tick=int(getattr(self, "tick", 0) or 0),
                force=True,
                slot_id=str(vt.slot_id),
                range_m=round(float(vt.range_m or 0.0), 4),
                bearing=round(float(vt.bearing), 4),
                latent_cos=round(float(hit.get("latent_cos") or 0.0), 4),
                reason=str(reason),
                vision_reason=str((vision_diag or {}).get("reason") or ""),
                latent=summarize_latent(vt.latent),
            )
        except Exception:
            pass
        return vt, out_diag

    def _try_sim_oracle_visual_bind(
        self,
        text: str,
        *,
        embed_fn: Any | None,
        require_movable: bool,
        interaction_kinds: frozenset[str],
        vision_diag: dict[str, Any],
    ) -> tuple[ResolvedObject | None, VisualTarget | None, dict[str, Any]]:
        """
        Non-production sim crutch: privileged registry XY → VisualTarget.

        Used only when RKK_SIM_ORACLE_BIND=1 and honest vision resolve is
        uncertain (3B). Logged as source=sim_oracle_bind / non_production=true.
        """
        if not sim_oracle_bind_enabled():
            return None, None, {"reason": "sim_oracle_disabled"}
        agent_xy, agent_fwd = self._agent_xy_forward()
        extras = self._sandbox_scene_extras()
        try:
            oracle, odiag = resolve_manipulation_target(
                text,
                extras,
                agent_xy=agent_xy,
                agent_forward=agent_fwd,
                embed_fn=embed_fn,
                require_movable=require_movable,
                interaction_kinds=interaction_kinds,
            )
        except Exception as exc:
            return None, None, {"reason": f"sim_oracle_error:{exc}"}
        if oracle is None:
            return None, None, {
                "reason": "sim_oracle_no_registry_match",
                "oracle_eval": dict(odiag or {}),
            }
        pos = getattr(oracle, "position", None)
        if pos is None or len(pos) < 2:
            return oracle, None, {"reason": "sim_oracle_no_position"}
        br = self._bearing_range_from_world_xy(
            (float(pos[0]), float(pos[1])),
            agent_xy=agent_xy,
            agent_fwd=agent_fwd,
        )
        if br is None:
            return oracle, None, {"reason": "sim_oracle_too_close"}
        bearing, dist = br
        u = float(max(0.0, min(1.0, 0.5 + 0.5 * bearing)))
        label = str(
            getattr(oracle, "semantic", None)
            or getattr(oracle, "ref", "")
            or "sim_oracle"
        )
        # Deliberately mid confidence — must not look like peaked vision success.
        conf = 0.40
        vt = VisualTarget(
            slot_id="sim_oracle",
            u=u,
            v=0.55,
            label=label,
            confidence=conf,
            bearing=float(bearing),
            range_m=float(dist),
            range_conf=1.0,
            diagnostics={
                "source": "sim_oracle_bind",
                "non_production": True,
                "oracle_ref": str(getattr(oracle, "ref", "")),
                "vision_fail_reason": str(vision_diag.get("reason") or ""),
                "geometry": "sim_oracle_xy",
            },
        )
        diag: dict[str, Any] = {
            "reason": "ok_sim_oracle",
            "resolve_mode": "vision",
            "source": "sim_oracle_bind",
            "non_production": True,
            "slot_id": vt.slot_id,
            "label": vt.label,
            "range_m": vt.range_m,
            "confidence": conf,
            "oracle_eval": {
                "ref": getattr(oracle, "ref", None),
                "reason": (odiag or {}).get("reason"),
            },
            "oracle_dist_m": float(dist),
            "vision_range_m": float(dist),
            "vision_reason": vision_diag.get("reason"),
            "refused_geometry_fallback": vision_diag.get("refused_geometry_fallback"),
        }
        try:
            task_log_event(
                "sim_oracle_bind",
                tick=int(getattr(self, "tick", 0) or 0),
                oracle_ref=str(getattr(oracle, "ref", "")),
                range_m=round(float(dist), 4),
                bearing=round(float(bearing), 4),
                vision_reason=str(vision_diag.get("reason") or ""),
                non_production=True,
            )
        except Exception:
            pass
        return oracle, vt, diag

    def _apply_active_percept_motor(self, pending: dict[str, Any]) -> None:
        """5A: look / slight turn while waiting to re-encode and re-resolve."""
        arb = getattr(self, "_motor_arbiter", None)
        if arb is None:
            return
        sign = float(pending.get("look_sign") or 1.0)
        yaw = float(max(0.05, min(0.95, 0.5 + 0.30 * sign)))
        coupling = float(max(0.05, min(0.95, 0.5 + 0.14 * sign)))
        try:
            arb.register_from_dict(
                "human_task",
                {
                    "intent_head_yaw": yaw,
                    "intent_look_at": 0.72,
                    "intent_gait_coupling": coupling,
                },
                precision=0.70,
            )
        except Exception:
            pass

    def _resolve_uncertain_reason(self, reason: str) -> bool:
        return str(reason or "") in (
            "uncertain_no_peaked_slot",
            "low_vision_confidence",
            "weak_objectness_peak",
            "floor_lock_rejected",
            "missing_or_invalid_range",
            "no_vision_slots",
            "no_language_vision_link",
            "resolve_failed_vision",
        )

    def _resolve_command_target(
        self,
        text: str,
        *,
        embed_fn: Any | None,
        require_movable: bool,
        interaction_kinds: frozenset[str],
    ) -> tuple[ResolvedObject | None, VisualTarget | None, dict[str, Any]]:
        """
        Bind-time resolve. Vision mode never falls back to registry for control
        unless RKK_SIM_ORACLE_BIND=1 (explicit non-production crutch).
        Returns (oracle_resolved_or_None, visual_target_or_None, diag).
        """
        if vision_resolve_enabled():
            # Ensure visual cortex is on for AGI vision path
            if not getattr(self, "_visual_mode", False):
                try:
                    enable = getattr(self, "enable_visual", None)
                    if callable(enable):
                        enable(n_slots=8, mode="hybrid")
                except Exception:
                    pass
            # Refresh encode so slot vectors/attn exist at bind
            ven = self._visual_env_ref()
            if ven is not None:
                try:
                    refresh = getattr(ven, "_refresh", None)
                    if callable(refresh):
                        # force_sync=True can stall the tick/API for tens of seconds
                        refresh(run_encode=True, force_sync=False)
                except Exception:
                    pass
                try:
                    sync = getattr(self, "_phase_m_sync_from_vision", None)
                    if callable(sync):
                        sync()
                except Exception:
                    pass
            cam = self._depth_camera_from_sim()
            if embed_fn is None:
                return None, None, {
                    "reason": "no_embed_fn",
                    "resolve_mode": "vision",
                }
            prev_range = None
            owm_prev = getattr(self, "_obj_working_memory", None)
            if owm_prev is not None and float(getattr(owm_prev, "range_m", 0.0) or 0.0) > 0.05:
                prev_range = float(owm_prev.range_m)
            fallen_flag = False
            base = self._humanoid_base_env()
            if base is not None and callable(getattr(base, "is_fallen", None)):
                fallen_flag = bool(base.is_fallen())
            vt, diag = resolve_visual_target(
                text,
                visual_env=ven,
                depth_camera=cam,
                embed_fn=embed_fn,
                concept_project_fn=self._slot_concept_project_fn(),
                require_range=True,
                fallen=fallen_flag,
                prev_range_m=prev_range,
            )
            diag = dict(diag)
            diag["resolve_mode"] = "vision"
            if vt is None:
                reason = str(diag.get("reason") or "resolve_failed_vision")
                diag["reason"] = reason
                uncertain = self._resolve_uncertain_reason(reason)
                # Phase 2: latent cosine re-ID before sim-oracle / active-percept give-up.
                if uncertain:
                    l_vt, l_diag = self._try_latent_reid_visual_bind(
                        vision_diag=diag,
                        reason=reason,
                    )
                    if l_vt is not None and l_vt.is_ready(require_range=True):
                        merged = {**diag, **l_diag}
                        return None, l_vt, merged
                    diag["latent_reid_attempt"] = l_diag
                # 5A: signal deferred tick to look-around when sim-oracle crutch is off.
                if (
                    uncertain
                    and not sim_oracle_bind_enabled()
                    and vision_active_percept_enabled()
                ):
                    diag["active_percept_candidate"] = True
                    return None, None, diag
                # Explicit sim-only cheat (non-production) after honest 3B refuse.
                if uncertain and sim_oracle_bind_enabled():
                    o_res, o_vt, o_diag = self._try_sim_oracle_visual_bind(
                        text,
                        embed_fn=embed_fn,
                        require_movable=require_movable,
                        interaction_kinds=interaction_kinds,
                        vision_diag=diag,
                    )
                    if o_vt is not None and o_vt.is_ready(require_range=True):
                        merged = {**diag, **o_diag}
                        return o_res, o_vt, merged
                    diag["sim_oracle_attempt"] = o_diag
                return None, None, diag
            # Eval-only: try to attach oracle ref if label matches registry (metrics)
            oracle = None
            try:
                agent_xy, agent_fwd = self._agent_xy_forward()
                extras = self._sandbox_scene_extras()
                oracle, odiag = resolve_manipulation_target(
                    text,
                    extras,
                    agent_xy=agent_xy,
                    agent_forward=agent_fwd,
                    embed_fn=embed_fn,
                    require_movable=require_movable,
                    interaction_kinds=interaction_kinds,
                )
                diag["oracle_eval"] = {
                    "ref": getattr(oracle, "ref", None),
                    "reason": odiag.get("reason"),
                }
                if oracle is not None:
                    od = self._oracle_dist_m_for_eval(oracle.ref)
                    diag["oracle_dist_m"] = od
                    diag["vision_range_m"] = vt.range_m
            except Exception as exc:
                diag["oracle_eval_error"] = str(exc)
            return oracle, vt, diag

        agent_xy, agent_fwd = self._agent_xy_forward()
        extras = self._sandbox_scene_extras()
        resolved, diag = resolve_manipulation_target(
            text,
            extras,
            agent_xy=agent_xy,
            agent_forward=agent_fwd,
            embed_fn=embed_fn,
            require_movable=require_movable,
            interaction_kinds=interaction_kinds,
        )
        diag = dict(diag)
        diag["resolve_mode"] = "oracle"
        return resolved, None, diag

    def _human_task_verify_ctx(self) -> dict[str, Any]:
        """Scene context for goal predicate verification (distance, contact, displace)."""
        resolved = getattr(self, "_manip_resolved", None)
        vt = getattr(self, "_manip_resolved_visual", None)
        owm = getattr(self, "_obj_working_memory", None)
        episode = getattr(self, "_manip_episode", None)
        agent_xy, _ = self._agent_xy_forward()
        ctx: dict[str, Any] = {"agent_xy": agent_xy}
        if vision_resolve_enabled() and owm is not None and owm.range_m > 0.05:
            ctx["distance_m"] = float(owm.range_m)
            ctx["vision_bearing"] = float(owm.bearing)
            ctx["vision_range_m"] = float(owm.range_m)
            ctx["task_target_x"] = float(owm.x_fwd)
            ctx["task_target_y"] = float(owm.y_right)
            ctx["task_target_conf"] = float(owm.confidence)
            ref = str(getattr(resolved, "ref", "") or "")
            if ref:
                od = self._oracle_dist_m_for_eval(ref)
                if od is not None:
                    ctx["oracle_dist_m"] = od
        elif vision_resolve_enabled() and vt is not None and vt.range_m is not None:
            ctx["distance_m"] = float(vt.range_m)
            ctx["vision_bearing"] = float(vt.bearing)
            ctx["vision_range_m"] = float(vt.range_m)
            ref = str(getattr(resolved, "ref", "") or "")
            if ref:
                od = self._oracle_dist_m_for_eval(ref)
                if od is not None:
                    ctx["oracle_dist_m"] = od
        else:
            ref = str(getattr(resolved, "ref", "") or "")
            target_xy = self._target_xy(ref) if ref else None
            if target_xy is not None:
                ctx["target_xy"] = target_xy
                ctx["distance_m"] = math.hypot(
                    target_xy[0] - agent_xy[0], target_xy[1] - agent_xy[1]
                )
        baseline = getattr(episode, "baseline_xy", None) if episode is not None else None
        if baseline is not None:
            ctx["baseline_xy"] = baseline
        if episode is not None and getattr(episode, "displacement_m", None) is not None:
            ctx["displacement_m"] = float(episode.displacement_m)
        ctx["contact"] = 1.0 if self._manip_has_contact(resolved) else 0.0
        return ctx

    def _register_task_manipulation(
        self,
        *,
        active: Any,
        dist: float,
        fallen: bool,
        owm: ObjectWorkingMemory | None = None,
    ) -> None:
        if fallen or active is None:
            # Near locked body: still drive reach/manip so contact can register.
            if fallen and active is not None:
                kind = str(active.kind)
                if kind in ("reach_contact", "reach_target", "verify_goal"):
                    phys = self._physics_range_to_locked_body()
                    near = phys is not None and float(phys) <= float(contact_reach_m())
                    if not near:
                        return
                else:
                    return
            else:
                return
        kind = str(active.kind)
        manip_kinds = ("reach_contact", "reach_target")
        if kind not in manip_kinds:
            if kind != "verify_goal" or float(dist) >= 0.9:
                return

        arb = getattr(self, "_motor_arbiter", None)
        if vision_resolve_enabled():
            tick = int(getattr(self, "tick", 0))
            if owm is None:
                owm = self._update_object_working_memory(tick)
            if owm is None or not owm.is_usable(tick):
                return
            intents = manipulation_intents_from_bearing_range(
                float(owm.bearing),
                float(owm.range_m),
                fallen=fallen,
            )
            if arb is not None and intents:
                intents.pop("vision_bearing", None)
                intents.pop("vision_range_m", None)
                arb.register_from_dict("manipulation", intents, precision=0.85)
            return

        resolved = getattr(self, "_manip_resolved", None)
        ref = str(getattr(resolved, "ref", "") or getattr(active, "target_ref", "") or "")
        target_xy = self._target_xy(ref) if ref else None
        if target_xy is None:
            return

        agent_xy, agent_fwd = self._agent_xy_forward()
        intents = manipulation_intents(
            agent_xy,
            agent_fwd,
            target_xy,
            float(dist),
            fallen=fallen,
        )
        if arb is not None and intents:
            arb.register_from_dict("manipulation", intents, precision=0.85)

    def _set_task_nav_graph_flags(
        self,
        *,
        nav_active: bool,
        heading_err: float | None = None,
    ) -> None:
        nodes = self.agent.graph.nodes
        nodes["task_nav_active"] = 1.0 if nav_active else 0.0
        pk = "phys_task_nav_active"
        if pk in nodes:
            nodes[pk] = 1.0 if nav_active else 0.0
        if heading_err is not None:
            nodes["task_heading_err"] = float(heading_err)
            hk = "phys_task_heading_err"
            if hk in nodes:
                nodes[hk] = float(heading_err)
        elif not nav_active:
            nodes["task_heading_err"] = 0.0
            hk = "phys_task_heading_err"
            if hk in nodes:
                nodes[hk] = 0.0

    def _arm_nav_hold(self, tick: int, *, reason: str = "hold") -> None:
        from engine.motor_arbiter import task_motor_hold_ticks

        # Always anchor to the live sim tick — command handlers may finish
        # many wall-seconds later with a stale bind-time tick.
        live = int(getattr(self, "tick", 0) or 0)
        now = max(int(tick), live)
        hold_ticks = (
            0
            if str(reason) in ("resolve_pending", "post_resolve")
            else int(task_motor_hold_ticks())
        )
        until = now + hold_ticks
        if hold_ticks <= 0:
            # Do not extend a previous freeze across resolve/post_resolve.
            self._nav_hold_until_tick = min(
                int(getattr(self, "_nav_hold_until_tick", -1) or -1), now
            )
            if int(getattr(self, "_nav_hold_until_tick", -1) or -1) < now:
                self._nav_hold_until_tick = now
        else:
            prev = int(getattr(self, "_nav_hold_until_tick", -1) or -1)
            self._nav_hold_until_tick = max(prev, until)
        try:
            task_log_event(
                "nav_hold",
                tick=now,
                until_tick=int(self._nav_hold_until_tick),
                reason=str(reason),
                bind_tick=int(tick),
                live_tick=live,
            )
        except Exception:
            pass

    def _nav_hold_active(self, tick: int) -> bool:
        until = int(getattr(self, "_nav_hold_until_tick", -1) or -1)
        return until > 0 and int(tick) < until

    def _live_tick(self, tick: int | None = None) -> int:
        live = int(getattr(self, "tick", 0) or 0)
        if tick is None:
            return live
        return max(int(tick), live)

    def _eval_oracle_dist_m(self) -> float | None:
        """Privileged eval distance — depth gating / metrics only, not control."""
        resolved = getattr(self, "_manip_resolved", None)
        if resolved is not None:
            od = self._oracle_dist_m_for_eval(str(getattr(resolved, "ref", "") or ""))
            if od is not None:
                return float(od)
        diag = getattr(self, "_manip_diag", None) or {}
        oref = str((diag.get("oracle_eval") or {}).get("ref") or "")
        if oref:
            od = self._oracle_dist_m_for_eval(oref)
            if od is not None:
                return float(od)
        return None

    def _rebind_vision_objectness_peak(
        self,
        tick: int,
        *,
        reason: str,
        oracle_dist: float | None = None,
        bearing_hint: float | None = None,
        allow_full_resolve: bool = False,
    ) -> bool:
        vt = getattr(self, "_manip_resolved_visual", None)
        cam = self._depth_camera_from_sim()
        if vt is None or cam is None:
            return False

        from engine.vision_depth import UvDepthTrack, _uv_dist, live_uv_track_radius
        from engine.vision_resolve import _cap_spatial_confidence

        scene = getattr(self, "_latent_scene", None)
        ent = scene.active() if scene is not None and hasattr(scene, "active") else None
        track = UvDepthTrack.from_list(getattr(ent, "uv_track", None) if ent else None)
        track_pred = track.extrapolate()
        prev_uv = track.prev()
        old_range = float(ent.range_m) if ent is not None else None
        old_u = float(getattr(ent, "u", vt.u) if ent is not None else vt.u)
        old_v = float(getattr(ent, "v", vt.v) if ent is not None else vt.v)

        # Do NOT release hard_lock before continuity/improves gates — a rejected
        # rebind must leave the lock intact (premature unlock → is_usable flip).
        diags = dict(vt.diagnostics or {})
        diags["geometry"] = "objectness_peak"
        diags["rebind_reason"] = reason
        if bearing_hint is not None:
            diags["bearing_hint"] = float(bearing_hint)
        vt2 = VisualTarget(
            slot_id=str(vt.slot_id),
            u=float(vt.u),
            v=float(vt.v),
            label=str(vt.label),
            confidence=float(vt.confidence),
            bearing=float(bearing_hint if bearing_hint is not None else vt.bearing),
            diagnostics=diags,
        )

        vt2 = attach_range_to_target(vt2, cam, attn_mask=None)
        vt2 = _cap_spatial_confidence(vt2)
        if not vt2.is_ready(require_range=True):
            # Phase 2: peaked UV failed → try latent re-ID before giving up.
            l_vt, l_diag = self._try_latent_reid_visual_bind(
                vision_diag={"reason": reason},
                reason=f"rebind_{reason}",
            )
            if l_vt is None or not l_vt.is_ready(require_range=True):
                return False
            vt2 = l_vt
            try:
                task_log_event(
                    "vision_latent_rebind",
                    tick=int(tick),
                    reason=str(reason),
                    slot_id=str(vt2.slot_id),
                    range_m=round(float(vt2.range_m or 0.0), 3),
                    bearing=round(float(vt2.bearing), 3),
                    latent_cos=round(float((l_diag or {}).get("latent_cos") or 0.0), 4),
                )
                from engine.neural_logger import neural_log_event, summarize_latent

                neural_log_event(
                    "latent",
                    "rebind",
                    tick=int(tick),
                    force=True,
                    reason=str(reason),
                    slot_id=str(vt2.slot_id),
                    range_m=round(float(vt2.range_m or 0.0), 3),
                    bearing=round(float(vt2.bearing), 3),
                    latent_cos=round(float((l_diag or {}).get("latent_cos") or 0.0), 4),
                    latent=summarize_latent(getattr(vt2, "latent", None)),
                )
            except Exception:
                pass

        new_u = float(vt2.u)
        new_v = float(vt2.v)
        new_r = float(vt2.range_m) if vt2.range_m is not None else None
        radius = live_uv_track_radius() * 1.6
        ref = track_pred if track_pred is not None else prev_uv
        if ref is None:
            ref = (old_u, old_v)
        cont_ok = _uv_dist(new_u, new_v, ref[0], ref[1]) <= radius

        improves = True
        if new_r is not None and oracle_dist is not None and float(oracle_dist) < 900.0:
            od = float(oracle_dist)
            # Rebind must shrink the vision/oracle gap — otherwise it is the same failure.
            if old_range is not None:
                improves = float(new_r) < float(old_range) * 0.90 and (
                    float(new_r) < od * 1.35 + 0.20
                )
            else:
                improves = float(new_r) < od * 1.35 + 0.20
        elif new_r is not None and old_range is not None:
            improves = float(new_r) < float(old_range) * 0.90

        full_resolve = False
        if not cont_ok or not improves:
            strong_fix = (
                new_r is not None
                and oracle_dist is not None
                and float(oracle_dist) < 900.0
                and float(new_r) < float(oracle_dist) * 1.25 + 0.10
            )
            if allow_full_resolve and strong_fix:
                full_resolve = True
            else:
                try:
                    task_log_event(
                        "vision_rebind_rejected",
                        tick=int(tick),
                        reason=str(reason),
                        continuity_ok=bool(cont_ok),
                        improves=bool(improves),
                        u=round(new_u, 3),
                        v=round(new_v, 3),
                        range_m=round(float(new_r), 3) if new_r is not None else None,
                        oracle_dist_m=(
                            round(float(oracle_dist), 3)
                            if oracle_dist is not None
                            else None
                        ),
                        track_ref_u=round(float(ref[0]), 3),
                        track_ref_v=round(float(ref[1]), 3),
                        hard_lock_preserved=True,
                    )
                except Exception:
                    pass
                return False

        # Gates passed — only now unlock so bind can reseat the track.
        self._release_scene_hard_lock()
        self._manip_resolved_visual = vt2
        self._bind_object_working_memory(vt2, int(tick))
        self._task_log_prev_vision_range = None
        try:
            task_log_event(
                "vision_rebind_full_resolve" if full_resolve else "vision_rebind",
                tick=int(tick),
                reason=str(reason),
                u=round(float(vt2.u), 3),
                v=round(float(vt2.v), 3),
                range_m=round(float(vt2.range_m or 0.0), 3),
                oracle_dist_m=round(float(oracle_dist), 3) if oracle_dist is not None else None,
                continuity_ok=bool(cont_ok),
                full_resolve=bool(full_resolve),
            )
        except Exception:
            pass
        return True

    def _release_scene_hard_lock(self) -> None:
        scene = getattr(self, "_latent_scene", None)
        if scene is not None and hasattr(scene, "release_hard_lock"):
            scene.release_hard_lock()

    def _owm_unlock_after_teleport(self, tick: int, *, reason: str = "com_teleport") -> None:
        """Soft-unlock OWM after COM teleport / assist reset so range rebinds from vision."""
        self._release_scene_hard_lock()
        self._owm_bind_range_m = None
        self._owm_range_ema = None
        self._vision_recede_streak = 0
        self._owm_cached_tick = -1
        self._owm_cached = None
        self._owm_pending_rebind_after_teleport = True
        try:
            task_log_event(
                "owm_unlock_after_teleport",
                tick=int(tick),
                reason=str(reason),
            )
        except Exception:
            pass

    def _maybe_rebind_after_teleport(self, tick: int) -> None:
        if not bool(getattr(self, "_owm_pending_rebind_after_teleport", False)):
            return
        self._owm_pending_rebind_after_teleport = False
        if not vision_resolve_enabled():
            return
        oracle_dist = self._eval_oracle_dist_m()
        owm = getattr(self, "_obj_working_memory", None)
        bearing_hint = float(owm.bearing) if owm is not None else None
        self._rebind_vision_objectness_peak(
            int(tick),
            reason="teleport_rebind",
            oracle_dist=float(oracle_dist) if oracle_dist is not None else None,
            bearing_hint=bearing_hint,
            allow_full_resolve=True,
        )
        self._owm_cached_tick = -1
        self._owm_cached = None

    def _maybe_rebind_vision_on_divergence(
        self,
        tick: int,
        *,
        oracle_dist: float,
        kind: str,
    ) -> None:
        """Re-lock via depth saliency when walking away from oracle eval target."""
        if not vision_resolve_enabled():
            return
        if kind not in ("approach", "approach_target"):
            return
        if float(oracle_dist) >= 900.0:
            return
        diag = getattr(self, "_manip_diag", None) or {}
        if not str((diag.get("oracle_eval") or {}).get("ref") or ""):
            return

        prev = getattr(self, "_vision_oracle_dist_prev", None)
        self._vision_oracle_dist_prev = float(oracle_dist)
        if prev is None:
            return
        if float(oracle_dist) - float(prev) <= 0.10:
            self._vision_walkaway_streak = 0
            return

        owm = getattr(self, "_obj_working_memory", None)
        if owm is None or not owm.is_usable(int(tick)):
            return
        if abs(float(owm.bearing)) > 0.22:
            self._vision_walkaway_streak = 0
            return

        streak = int(getattr(self, "_vision_walkaway_streak", 0)) + 1
        self._vision_walkaway_streak = streak
        if streak < 4:
            return
        self._vision_walkaway_streak = 0

        owm = getattr(self, "_obj_working_memory", None)
        bearing_hint = float(owm.bearing) if owm is not None else None
        if not self._rebind_vision_objectness_peak(
            int(tick),
            reason="oracle_divergence",
            oracle_dist=float(oracle_dist),
            bearing_hint=bearing_hint,
            allow_full_resolve=True,
        ):
            return

    def _maybe_correct_vision_range_mismatch(
        self,
        tick: int,
        *,
        oracle_dist: float,
        kind: str,
    ) -> None:
        """
        Soft-correct inflated floor-lock range via live camera.

        Gated hard_lock release (not blind unlock):
          1) Soft live-camera refresh keeps hard_lock (HUD / range nudge only).
          2) If still mismatched, try objectness rebind — lock releases only if
             continuity/improves gates pass (see ``_rebind_vision_objectness_peak``).
          3) After several consecutive gated rejects, escalate to
             ``allow_full_resolve`` (strong oracle-aligned peak may unlock).
          4) After further failures, force-unlock once so a truly lost target can
             be re-resolved (avoids eternal sticky lock).

        Success criteria for a fresh approach episode (before track_radius work):
          - fewer ``vision_rebind_rejected`` while ``hard_lock`` stays true between
            successful rebinds;
          - ``hard_lock=false`` only right after successful rebind / force-unlock
            fallback — not on every range_correct;
          - ``oracle_dist`` can fall below the ~1.3 m barrier seen in stuck runs;
          - force-unlock path still allows redefinition when the object is gone.
        """
        if not vision_resolve_enabled():
            return
        if kind not in ("approach", "approach_target"):
            return
        if float(oracle_dist) >= 900.0:
            return
        until = int(getattr(self, "_vision_range_correct_until_tick", -1) or -1)
        if int(tick) < until:
            return

        owm = getattr(self, "_obj_working_memory", None)
        if owm is None:
            return
        vr = float(owm.range_m)
        od = float(oracle_dist)
        ratio = vr / max(od, 0.25)
        if vr <= od * 1.32 + 0.30:
            self._vision_range_mismatch_streak = 0
            self._vision_floor_lock_reject_streak = 0
            return

        streak = int(getattr(self, "_vision_range_mismatch_streak", 0)) + 1
        self._vision_range_mismatch_streak = streak
        need_streak = 3 if ratio > 1.75 else 5
        if streak < need_streak:
            return
        self._vision_range_mismatch_streak = 0
        self._vision_range_correct_until_tick = int(tick) + 40

        scene = getattr(self, "_latent_scene", None)
        cam = self._depth_camera_from_sim()
        if scene is None or cam is None:
            return

        hard_before = bool(getattr(scene, "hard_lock_active", False))
        unlocked = False
        rebind_ok = False
        force_unlock = False
        allow_full = False

        # Soft correction under lock — do NOT release here.
        try:
            from engine.object_working_memory import ego_from_bearing_range
            from engine.vision_depth import UvDepthTrack, live_uv_range_at_bearing

            scene.refresh_active_from_live_camera(
                cam,
                tick=int(tick),
                range_hint=float(od),
                blend=1.0,
            )
        except Exception:
            return

        ent = scene.active()
        if ent is not None and float(ent.range_m) > od * 1.25 + 0.20:
            try:
                track = UvDepthTrack.from_list(ent.uv_track)
                _u, _v, lr, _conf = live_uv_range_at_bearing(
                    cam,
                    float(ent.bearing),
                    range_hint=float(od),
                    tick=int(tick),
                    uv_track=track,
                )
                ent.uv_track = track.to_list()
                if lr is not None and float(lr) < float(ent.range_m):
                    ent.range_m = float(lr)
                    ent.x_fwd, ent.y_right = ego_from_bearing_range(
                        float(ent.bearing), float(ent.range_m)
                    )
            except Exception:
                pass

        try:
            escalate_after = max(
                1, int(os.environ.get("RKK_VISION_FLOOR_LOCK_ESCALATE_AFTER", "3"))
            )
        except ValueError:
            escalate_after = 3
        try:
            force_after = max(
                escalate_after + 1,
                int(os.environ.get("RKK_VISION_FLOOR_LOCK_FORCE_UNLOCK_AFTER", "5")),
            )
        except ValueError:
            force_after = 5

        reject_streak = int(getattr(self, "_vision_floor_lock_reject_streak", 0) or 0)
        ent = scene.active()
        still_bad = ent is not None and float(ent.range_m) > od * 1.40 + 0.15
        if still_bad:
            allow_full = reject_streak >= escalate_after
            rebind_ok = bool(
                self._rebind_vision_objectness_peak(
                    int(tick),
                    reason="floor_lock",
                    oracle_dist=float(od),
                    bearing_hint=float(ent.bearing),
                    allow_full_resolve=bool(allow_full),
                )
            )
            if rebind_ok:
                self._vision_floor_lock_reject_streak = 0
                unlocked = True  # successful rebind releases lock inside gate
            else:
                reject_streak += 1
                self._vision_floor_lock_reject_streak = reject_streak
                # Target may be truly gone / unrecoverable under continuity —
                # force unlock so later percept fuse / re-resolve can retarget.
                if reject_streak >= force_after and hard_before:
                    self._release_scene_hard_lock()
                    unlocked = True
                    force_unlock = True
                    self._vision_floor_lock_reject_streak = 0
                    try:
                        task_log_event(
                            "vision_range_correct_force_unlock",
                            tick=int(tick),
                            reason="floor_lock_reject_streak",
                            reject_streak=int(reject_streak),
                            force_after=int(force_after),
                            oracle_dist_m=round(od, 3),
                            vision_range_m=round(vr, 3),
                        )
                    except Exception:
                        pass
        else:
            # Soft path fixed the mismatch without rebind — keep lock.
            self._vision_floor_lock_reject_streak = 0

        self._owm_cached_tick = -1
        self._task_log_prev_vision_range = None
        try:
            ent = scene.active()
            hard_after = bool(getattr(scene, "hard_lock_active", False))
            task_log_event(
                "vision_range_correct",
                tick=int(tick),
                vision_range_m=round(vr, 3),
                oracle_dist_m=round(od, 3),
                corrected_range_m=round(float(ent.range_m), 3) if ent else None,
                hard_lock_before=bool(hard_before),
                hard_lock_after=bool(hard_after),
                unlocked=bool(unlocked),
                rebind_ok=bool(rebind_ok),
                force_unlock=bool(force_unlock),
                allow_full_resolve=bool(allow_full),
                floor_lock_reject_streak=int(
                    getattr(self, "_vision_floor_lock_reject_streak", 0) or 0
                ),
            )
        except Exception:
            pass

    def _maybe_rebind_vision_on_recede(
        self,
        tick: int,
        scene: LatentSceneMemory,
        range_m: float,
    ) -> None:
        """
        Under hard_lock, if live range recedes vs bind/EMA for several ticks,
        refresh live camera and attempt objectness rebind.
        """
        if not vision_resolve_enabled():
            return
        if not bool(getattr(scene, "hard_lock_active", False)):
            self._vision_recede_streak = 0
            return
        try:
            cooldown = max(0, int(os.environ.get("RKK_VISION_RECEDE_COOLDOWN", "15")))
        except ValueError:
            cooldown = 15
        last_tick = int(getattr(self, "_vision_recede_last_tick", -999999) or -999999)
        if cooldown > 0 and int(tick) - last_tick < cooldown:
            return
        try:
            threshold = float(os.environ.get("RKK_VISION_RECEDE_DELTA_M", "0.25"))
        except ValueError:
            threshold = 0.25
        try:
            need_streak = max(1, int(os.environ.get("RKK_VISION_RECEDE_STREAK", "3")))
        except ValueError:
            need_streak = 3

        ema = float(getattr(self, "_owm_range_ema", range_m) or range_m)
        alpha = 0.18
        self._owm_range_ema = (1.0 - alpha) * ema + alpha * float(range_m)
        bind_r = getattr(self, "_owm_bind_range_m", None)
        ref = float(bind_r) if bind_r is not None else ema
        if float(range_m) <= ref + threshold:
            self._vision_recede_streak = 0
            return

        streak = int(getattr(self, "_vision_recede_streak", 0)) + 1
        self._vision_recede_streak = streak
        if streak < need_streak:
            return
        self._vision_recede_streak = 0
        self._vision_recede_last_tick = int(tick)

        cam = self._depth_camera_from_sim()
        if cam is not None:
            try:
                oracle = self._eval_oracle_dist_m()
                scene.refresh_active_from_live_camera(
                    cam,
                    tick=int(tick),
                    range_hint=oracle,
                    blend=1.0,
                )
            except Exception:
                pass

        owm = getattr(self, "_obj_working_memory", None)
        bearing_hint = float(owm.bearing) if owm is not None else None
        oracle_dist = self._eval_oracle_dist_m()
        rebind_ok = bool(
            self._rebind_vision_objectness_peak(
                int(tick),
                reason="vision_recede_rebind",
                oracle_dist=float(oracle_dist) if oracle_dist is not None else None,
                bearing_hint=bearing_hint,
                allow_full_resolve=True,
            )
        )
        self._owm_cached_tick = -1
        self._task_log_prev_vision_range = None
        try:
            task_log_event(
                "vision_recede_rebind",
                tick=int(tick),
                range_m=round(float(range_m), 4),
                bind_range_m=round(float(bind_r), 4) if bind_r is not None else None,
                range_ema_m=round(float(self._owm_range_ema), 4),
                delta_m=round(float(range_m) - ref, 4),
                rebind_ok=bool(rebind_ok),
                hard_lock_after=bool(getattr(scene, "hard_lock_active", False)),
            )
        except Exception:
            pass

    def _wm_train_steps(self) -> int:
        agent = getattr(self, "agent", None)
        g = getattr(agent, "graph", None) if agent is not None else None
        if g is None:
            return 0
        train_calls = int(getattr(g, "_wm_train_calls", 0) or 0)
        if train_calls > 0:
            return train_calls
        return int(getattr(agent, "_notears_steps", 0) or 0)

    def _inject_owm_nav_priors(
        self,
        owm: ObjectWorkingMemory,
        *,
        stop: float,
    ) -> dict[str, float]:
        """Write OWM into graph nodes and return approach target priors for AI."""
        graph = self.agent.graph
        self._ensure_nav_vision_graph_nodes(graph)
        b = float(owm.bearing)
        r = float(owm.range_m)
        b01 = _encode_nav_bearing_01(b)
        r01 = _encode_nav_range_01(r)
        stop01 = _encode_nav_range_01(float(max(0.05, stop)))
        nodes = graph.nodes
        nodes["vision_bearing_01"] = b01
        nodes["vision_range_01"] = r01
        nodes["vision_bearing"] = b
        nodes["vision_range_m"] = r
        if "task_target_dist_m" in nodes:
            nodes["task_target_dist_m"] = r
        priors: dict[str, float] = {
            "phys_posture_stability": 1.0,
            "phys_com_z": 0.82,
            "vision_bearing_01": 0.5,
            "vision_range_01": stop01,
        }
        # Aligned → ask for forward COM velocity; large bearing → stay cautious.
        if abs(b) < 0.35 and r > float(stop) + 0.05:
            priors["phys_com_x_vel"] = 0.38
        else:
            priors["phys_com_x_vel"] = 0.12
        return priors

    def _assert_forward_when_aligned(
        self,
        out: dict[str, float],
        heur: dict[str, float],
        *,
        bearing: float,
        range_m: float,
        stop: float,
        meta: dict[str, Any],
    ) -> dict[str, float]:
        """
        When roughly aligned and still far from stop, floor stride / pull posture
        toward heuristic forward so weak homeostatic intents do not plateau.
        """
        try:
            align = float(os.environ.get("RKK_NAV_ALIGN_BEARING", "0.40"))
            margin = float(os.environ.get("RKK_NAV_FWD_RANGE_MARGIN", "0.12"))
            floor = float(os.environ.get("RKK_NAV_ALIGNED_STRIDE_FLOOR", "0.62"))
            blend = float(os.environ.get("RKK_NAV_ALIGNED_FWD_BLEND", "0.65"))
        except ValueError:
            align, margin, floor, blend = 0.40, 0.12, 0.62, 0.65
        blend = float(max(0.0, min(1.0, blend)))
        if abs(float(bearing)) > align:
            return out
        if float(range_m) <= float(stop) + margin:
            return out

        out = dict(out)
        heur_stride = float(heur.get("intent_stride", floor))
        target_stride = max(float(floor), heur_stride)
        cur = float(out.get("intent_stride", 0.5))
        if cur < target_stride:
            blended = (1.0 - blend) * cur + blend * target_stride
            out["intent_stride"] = float(max(blended, target_stride * 0.90))

        posture_blend = 0.35 * blend
        for key in (
            "intent_gait_coupling",
            "intent_support_left",
            "intent_support_right",
            "intent_torso_forward",
        ):
            if key in heur:
                hv = float(heur[key])
                cv = float(out.get(key, hv))
                out[key] = float((1.0 - posture_blend) * cv + posture_blend * hv)

        meta["nav_fwd_assert"] = True
        meta["nav_fwd_stride"] = float(out.get("intent_stride", 0.0))
        return out

    def _navigation_intents_wm_ai(
        self,
        owm: ObjectWorkingMemory,
        stop: float,
        posture: float,
        fallen: bool,
        *,
        bearing_override: float | None = None,
        range_override: float | None = None,
    ) -> tuple[dict[str, float], dict[str, Any]]:
        """
        WM + Active Inference approach intents (arbiter source ``navigation``).

        Falls back to heuristic bearing/range nav when posture pause,
        AI returns empty, or WM is cold. Fallen still gets heuristic nav
        (downscaled) so crawl-to-target can continue.
        """
        tick = int(getattr(self, "tick", 0))
        every = _task_nav_ai_every()
        cached_tick = int(getattr(self, "_nav_ai_cached_tick", -1))
        cached = getattr(self, "_nav_ai_cached", None)
        if cached is not None and tick - cached_tick < every and not fallen:
            intents_c, meta_c = cached
            return dict(intents_c), dict(meta_c)

        def _store_cache(
            out_intents: dict[str, float], out_meta: dict[str, Any]
        ) -> tuple[dict[str, float], dict[str, Any]]:
            self._nav_ai_cached_tick = tick
            self._nav_ai_cached = (dict(out_intents), dict(out_meta))
            return out_intents, out_meta

        bearing = (
            float(bearing_override)
            if bearing_override is not None
            else float(owm.bearing)
        )
        range_m = (
            float(range_override)
            if range_override is not None
            else float(owm.range_m)
        )
        meta: dict[str, Any] = {
            "task_nav_mode": "wm_ai",
            "nav_ai_ok": False,
            "nav_ai_reason": "",
            "wm_steps": self._wm_train_steps(),
            "wm_min_steps": _task_nav_wm_min_steps(),
            "nav_bearing": round(float(bearing), 4),
            "nav_range_m": round(float(range_m), 4),
        }
        heur = navigation_intents_from_bearing_range(
            float(bearing),
            float(range_m),
            float(stop),
            fallen=fallen,
            posture_stability=float(posture),
        )
        if fallen:
            meta["nav_ai_reason"] = "fallen_heuristic"
            return _store_cache(heur if heur else {}, meta)
        if not heur:
            meta["nav_ai_reason"] = "heuristic_empty_or_posture"
            return _store_cache({}, meta)

        wm_steps = self._wm_train_steps()
        wm_warm = wm_steps >= _task_nav_wm_min_steps()
        meta["wm_warm"] = bool(wm_warm)

        ai_on = os.environ.get("RKK_ACTIVE_INFERENCE", "0").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        if not ai_on:
            meta["nav_ai_reason"] = "active_inference_off_fallback_heur"
            return _store_cache(heur, meta)

        try:
            priors = self._inject_owm_nav_priors(owm, stop=float(stop))
            ensure = getattr(self, "_ensure_homeostatic_ctrl", None)
            if not callable(ensure):
                from engine.active_inference import HomeostaticController, _eff_iters
                import torch

                device = getattr(
                    self.agent.graph._core, "device", torch.device("cpu")
                )
                self._homeostatic_ctrl = HomeostaticController(
                    device=device, learning_rate=0.1, max_iters=min(8, _eff_iters())
                )
                ctrl = self._homeostatic_ctrl
            else:
                ctrl = ensure()

            graph = self.agent.graph
            obs = graph.snapshot_vec_dict()
            b01 = _encode_nav_bearing_01(float(bearing))
            r01 = _encode_nav_range_01(float(range_m))
            obs["vision_bearing_01"] = b01
            obs["vision_range_01"] = r01
            obs["vision_bearing"] = float(bearing)
            obs["vision_range_m"] = float(range_m)
            if "task_target_dist_m" in obs:
                obs["task_target_dist_m"] = float(range_m)

            # Warm-start intent slots from heuristic nav so AI refines a real seed.
            node_ids = list(getattr(graph, "_node_ids", []) or [])
            for key, val in heur.items():
                if not str(key).startswith("intent_"):
                    continue
                fv = float(max(0.0, min(1.0, val)))
                for nid in (str(key), f"phys_{key}"):
                    if nid in node_ids:
                        obs[nid] = fv
                        graph.nodes[nid] = fv

            return_all = _active_inf_return_all_enabled()
            actions = (
                ctrl.optimize_action(
                    obs, graph, priors, return_all=return_all
                )
                or {}
            )
            mapped: dict[str, float] = {}
            for gid, val in actions.items():
                ev = _graph_nid_to_motor_intent(str(gid))
                if ev is not None:
                    mapped[ev] = float(val)

            # Optional short WM beam nudge only when WM has trained enough.
            if not mapped and wm_warm:
                try:
                    from engine.goal_planning import beam_search_first_action

                    state0 = dict(obs)
                    cand_actions: list[tuple[str, float]] = [
                        ("intent_gait_coupling", 0.35),
                        ("intent_gait_coupling", 0.65),
                        ("intent_stride", 0.55),
                        ("intent_stride", 0.72),
                    ]
                    stop01 = _encode_nav_range_01(float(stop))

                    def _score(
                        _s0: dict[str, float],
                        var: str,
                        val: float,
                        s_after: dict[str, float],
                    ) -> float:
                        br = float(
                            s_after.get(
                                "vision_bearing_01",
                                _encode_nav_bearing_01(bearing),
                            )
                        )
                        rg = float(
                            s_after.get(
                                "vision_range_01",
                                _encode_nav_range_01(range_m),
                            )
                        )
                        return -abs(br - 0.5) - 0.15 * abs(rg - stop01)

                    best, sc = beam_search_first_action(
                        self.agent,
                        state0=state0,
                        actions=cand_actions,
                        depth=1,
                        beam_k=4,
                        rollout_horizon=1,
                        score_fn=_score,
                        maximize=True,
                    )
                    if best is not None:
                        var, val = best
                        ev = _graph_nid_to_motor_intent(str(var)) or (
                            str(var) if str(var).startswith("intent_") else None
                        )
                        if ev:
                            mapped[ev] = float(val)
                            meta["nav_ai_reason"] = "wm_beam"
                        else:
                            meta["wm_beam_unmap"] = str(var)
                    else:
                        meta["wm_beam_empty"] = True
                        meta["wm_beam_score"] = float(sc)
                except Exception as exc:
                    meta["wm_beam_error"] = str(exc)
            elif not mapped and not wm_warm:
                meta["wm_beam_skipped"] = "wm_cold"
            elif not mapped:
                meta["wm_beam_skipped"] = "mapped_nonempty" if actions else "ai_empty"

            if mapped and (
                "intent_gait_coupling" in mapped or "intent_stride" in mapped
            ):
                out = dict(heur)
                for k, v in mapped.items():
                    if str(k).startswith("intent_"):
                        out[str(k)] = float(v)
                out = self._assert_forward_when_aligned(
                    out,
                    heur,
                    bearing=float(bearing),
                    range_m=float(range_m),
                    stop=float(stop),
                    meta=meta,
                )
                meta["nav_ai_ok"] = True
                if not meta.get("nav_ai_reason"):
                    meta["nav_ai_reason"] = "homeostatic"
                meta["nav_ai_intents"] = sorted(mapped.keys())
                return _store_cache(out, meta)

            meta["nav_ai_reason"] = "ai_empty_fallback_heur"
            return _store_cache(heur, meta)
        except Exception as exc:
            meta["nav_ai_reason"] = f"ai_error_fallback_heur:{exc}"
            return _store_cache(heur, meta)

    def _register_task_navigation(
        self,
        *,
        active: Any,
        dist: float,
        approach_m: float,
        fallen: bool,
        obs: dict[str, float] | None = None,
        owm: ObjectWorkingMemory | None = None,
    ) -> None:
        tick = int(getattr(self, "tick", 0))
        if active is None or self._nav_hold_active(tick):
            self._set_task_nav_graph_flags(nav_active=False)
            return
        # Fallen still allows WM/AI nav (intents already downscale for fallen).
        # Blocking nav on fallen was freezing crawl-to-target once upright failed.
        kind = str(active.kind)
        if kind not in ("approach", "reach_contact", "approach_target", "reach_target"):
            return

        stop = float(active.expected_state.get("stop_distance", approach_m))
        if kind == "reach_contact":
            stop = min(stop, reach_start_m())

        obs_r = dict(obs or {})
        posture = float(
            obs_r.get(
                "posture_stability",
                obs_r.get("phys_posture_stability", 0.6),
            )
        )
        arb = getattr(self, "_motor_arbiter", None)
        intents: dict[str, float] = {}
        nav_meta: dict[str, Any] = {"task_nav_mode": _task_nav_mode(), "nav_ai_ok": False}

        if vision_resolve_enabled():
            tick = int(getattr(self, "tick", 0))
            if owm is None:
                owm = self._update_object_working_memory(tick)
            if owm is None or not owm.is_usable(tick):
                # Stale / no memory → pause (no oracle fallback)
                self._set_task_nav_graph_flags(nav_active=False)
                return
            range_m = float(owm.range_m)
            # Prefer physics surface range for stop decisions when locked.
            phys = self._physics_range_to_locked_body()
            if phys is not None:
                if float(range_m) + 0.45 < float(phys):
                    range_m = float(phys)
                else:
                    range_m = min(float(range_m), float(phys))
            nav_bearing = float(owm.bearing)
            if phys is not None and abs(float(owm.range_m) - float(phys)) > 0.55:
                row = self._static_registry_row_for_body(
                    int(getattr(self, "_task_locked_body_id", 0) or 0)
                )
                if row is not None:
                    br = self._bearing_range_from_world_xy(
                        (float(row.get("x", 0.0)), float(row.get("y", 0.0)))
                    )
                    if br is not None:
                        nav_bearing = float(br[0])
            has_contact = False
            if kind == "reach_contact":
                has_contact = bool(
                    self._manip_has_contact(getattr(self, "_manip_resolved", None))
                )
            need_nav = kind in ("approach", "approach_target") or (
                kind == "reach_contact"
                and (not has_contact)
                and range_m > 0.12
            ) or (kind == "reach_target" and range_m > approach_m)
            mode = _task_nav_mode()
            if need_nav:
                if mode == "wm_ai":
                    intents, nav_meta = self._navigation_intents_wm_ai(
                        owm,
                        stop,
                        posture,
                        fallen,
                        bearing_override=nav_bearing,
                        range_override=range_m,
                    )
                else:
                    intents = navigation_intents_from_bearing_range(
                        float(nav_bearing),
                        range_m,
                        stop,
                        fallen=fallen,
                        posture_stability=posture,
                    )
                    nav_meta = {
                        "task_nav_mode": "heuristic",
                        "nav_ai_ok": False,
                        "nav_ai_reason": "mode_heuristic",
                    }
            self._last_nav_meta = dict(nav_meta)
            if intents:
                # Cache raw nav intents (incl. gait_coupling) for task_progress dump.
                self._last_nav_intents = {
                    k: float(v)
                    for k, v in intents.items()
                    if str(k).startswith("intent_")
                }
                heading_err = intents.pop("task_heading_err", None)
                intents.pop("task_closing_vel", None)
                intents.pop("task_nav_active", None)
                vb = intents.pop("vision_bearing", None)
                vr = intents.pop("vision_range_m", None)
                intents.pop("task_target_x", None)
                intents.pop("task_target_y", None)
                self._set_task_nav_graph_flags(
                    nav_active=True,
                    heading_err=float(heading_err) if heading_err is not None else None,
                )
                nodes = self.agent.graph.nodes
                if vb is not None:
                    nodes["vision_bearing"] = float(vb)
                if vr is not None:
                    nodes["vision_range_m"] = float(vr)
                    if "task_target_dist_m" in nodes:
                        nodes["task_target_dist_m"] = float(vr)
                try:
                    task_log_event(
                        "task_nav",
                        tick=int(tick),
                        task_nav_mode=str(nav_meta.get("task_nav_mode") or mode),
                        nav_ai_ok=bool(nav_meta.get("nav_ai_ok")),
                        nav_ai_reason=str(nav_meta.get("nav_ai_reason") or ""),
                        wm_steps=int(nav_meta.get("wm_steps") or 0),
                        bearing=round(float(owm.bearing), 4),
                        range_m=round(float(range_m), 4),
                        intent_gait_coupling=(
                            round(float(intents["intent_gait_coupling"]), 4)
                            if "intent_gait_coupling" in intents
                            else None
                        ),
                        intent_stride=(
                            round(float(intents["intent_stride"]), 4)
                            if "intent_stride" in intents
                            else None
                        ),
                    )
                    from engine.neural_logger import neural_log_event

                    neural_log_event(
                        "nav",
                        "intents",
                        tick=int(tick),
                        task_nav_mode=str(nav_meta.get("task_nav_mode") or mode),
                        nav_ai_ok=bool(nav_meta.get("nav_ai_ok")),
                        nav_ai_reason=str(nav_meta.get("nav_ai_reason") or ""),
                        wm_steps=int(nav_meta.get("wm_steps") or 0),
                        bearing=round(float(owm.bearing), 4),
                        range_m=round(float(range_m), 4),
                        intents={
                            k: round(float(v), 4)
                            for k, v in intents.items()
                            if str(k).startswith("intent_")
                        },
                        nav_ai_intents=nav_meta.get("nav_ai_intents"),
                    )
                except Exception:
                    pass
            else:
                self._set_task_nav_graph_flags(nav_active=False)
            if arb is not None and intents:
                prec = 0.88 if kind in ("approach", "approach_target", "reach_contact") else 0.68
                arb.register_from_dict("navigation", intents, precision=prec)
            return

        resolved = getattr(self, "_manip_resolved", None)
        ref = str(getattr(resolved, "ref", "") or getattr(active, "target_ref", "") or "")
        target_xy = self._target_xy(ref) if ref else None
        if target_xy is None:
            self._set_task_nav_graph_flags(nav_active=False)
            return

        agent_xy, agent_fwd = self._agent_xy_forward()
        prev_xy = getattr(self, "_task_nav_prev_xy", None)
        has_contact = False
        if kind == "reach_contact":
            has_contact = bool(
                self._manip_has_contact(getattr(self, "_manip_resolved", None))
            )
        if kind in ("approach", "approach_target") or (
            kind == "reach_contact" and (not has_contact) and float(dist) > 0.12
        ) or (kind == "reach_target" and float(dist) > approach_m):
            intents = navigation_intents(
                agent_xy,
                agent_fwd,
                target_xy,
                stop,
                fallen=fallen,
                posture_stability=posture,
                prev_agent_xy=prev_xy if isinstance(prev_xy, tuple) else None,
            )
        self._task_nav_prev_xy = (float(agent_xy[0]), float(agent_xy[1]))

        if intents:
            self._last_nav_intents = {
                k: float(v)
                for k, v in intents.items()
                if str(k).startswith("intent_")
            }
            heading_err = intents.pop("task_heading_err", None)
            closing_vel = intents.pop("task_closing_vel", None)
            intents.pop("task_nav_active", None)
            self._set_task_nav_graph_flags(
                nav_active=True,
                heading_err=float(heading_err) if heading_err is not None else None,
            )
            nodes = self.agent.graph.nodes
            if closing_vel is not None:
                nodes["task_closing_vel"] = float(closing_vel)
                if "phys_task_closing_vel" in nodes:
                    nodes["phys_task_closing_vel"] = float(closing_vel)
        else:
            self._set_task_nav_graph_flags(nav_active=False)
        if arb is not None and intents:
            prec = 0.88 if kind in ("approach", "approach_target", "reach_contact") else 0.68
            arb.register_from_dict("navigation", intents, precision=prec)

    def _apply_goal_target_ref(self, goal: TaskGoal, resolved: ResolvedObject) -> None:
        goal.target_ref = resolved.ref
        for pred in goal.predicates:
            if pred.kind in _KINDS_NEEDING_TARGET:
                pred.target_ref = resolved.ref
        tt = getattr(self, "_task_tree_ctrl", None)
        if tt is not None and tt.tree is not None:
            for node in tt.tree.nodes.values():
                node.target_ref = resolved.ref

    def _apply_goal_visual_ref(self, goal: TaskGoal, vt: VisualTarget) -> None:
        goal.target_ref = vt.ref
        for pred in goal.predicates:
            if pred.kind in _KINDS_NEEDING_TARGET:
                pred.target_ref = vt.ref
        tt = getattr(self, "_task_tree_ctrl", None)
        if tt is not None and tt.tree is not None:
            for node in tt.tree.nodes.values():
                node.target_ref = vt.ref

    def _ground_command_goal(
        self,
        text: str,
        gl: GroundedLanguageController | None,
    ) -> TaskGoal | None:
        try:
            from engine.goal_grounding import ground_command
        except ImportError:
            return None
        embed_fn = gl.embedder.embed if gl is not None else None
        return ground_command(text, embed_fn)

    def _goal_predicate_kinds(self, goal: TaskGoal | None) -> set[str]:
        if goal is None:
            return set()
        return {str(p.kind) for p in (goal.predicates or [])}

    def _tick_deferred_vision_resolve(self, *, tick: int, fallen: bool) -> None:
        """Finish resolve_target after command (async) / stance recovers.

        5A: when vision is uncertain and sim-oracle crutch is off, look around
        and re-resolve up to RKK_VISION_ACTIVE_PERCEPT_TRIES before failing.
        """
        pending = getattr(self, "_deferred_vision_resolve", None)
        if not isinstance(pending, dict):
            return
        # Only wait for fallen — nav_hold blocks walking, not resolve itself.
        if fallen:
            return
        tt = self._ensure_task_tree()
        active = tt.active_node
        if active is None or str(active.kind) != "resolve_target":
            self._deferred_vision_resolve = None
            return

        live = self._live_tick(tick)
        wait_until = int(pending.get("wait_until_tick") or 0)
        if wait_until > 0 and live < wait_until:
            self._apply_active_percept_motor(pending)
            return

        text = str(pending.get("text") or "")
        gl = getattr(self, "_grounded_lang", None)
        embed_fn = gl.embedder.embed if gl is not None and getattr(gl, "embedder", None) else None
        pred_kinds = set(pending.get("interaction_kinds") or [])
        require_movable = bool(pending.get("require_movable", False))
        try:
            resolved, visual, diag = self._resolve_command_target(
                text,
                embed_fn=embed_fn,
                require_movable=require_movable,
                interaction_kinds=frozenset(str(k) for k in pred_kinds),
            )
        except Exception as exc:
            diag = {"reason": f"resolver_error:{exc}", "resolve_mode": "vision"}
            resolved, visual = None, None
        self._manip_diag = dict(diag)
        self._task_log_target_resolution(live, text, resolved, diag)
        self._dump_bind_frame(live, visual=visual, diag=diag if isinstance(diag, dict) else {})

        vision_ok = visual is not None and visual.is_ready(require_range=True)
        if vision_resolve_enabled() and not vision_ok:
            # 5A active-perception: scan → wait encode → rescan (sim-oracle off only).
            if bool(diag.get("active_percept_candidate")) and vision_active_percept_enabled():
                tries = int(pending.get("active_percept_tries") or 0)
                max_tries = vision_active_percept_max_tries()
                if tries < max_tries:
                    sign = 1.0 if (tries % 2 == 0) else -1.0
                    pending["active_percept_tries"] = tries + 1
                    pending["look_sign"] = sign
                    try:
                        settle = max(
                            8, int(os.environ.get("RKK_VISION_ACTIVE_PERCEPT_SETTLE", "24"))
                        )
                    except ValueError:
                        settle = 24
                    pending["wait_until_tick"] = live + settle
                    self._deferred_vision_resolve = pending
                    self._apply_active_percept_motor(pending)
                    try:
                        task_log_event(
                            "active_percept_scan",
                            tick=live,
                            try_n=tries + 1,
                            max_tries=max_tries,
                            look_sign=sign,
                            reason=str(diag.get("reason") or ""),
                        )
                    except Exception:
                        pass
                    return
                diag = dict(diag)
                diag["reason"] = "active_percept_exhausted"
                self._manip_diag = dict(diag)

            self._deferred_vision_resolve = None
            self._tt_fail_active(
                tt, live, str(diag.get("reason", "resolve_failed_vision")), retryable=False
            )
            self._maybe_finalize_task_tree(live)
            return

        self._deferred_vision_resolve = None
        if vision_resolve_enabled() and visual is not None:
            goal = getattr(self, "_task_goal", None)
            if goal is not None:
                self._apply_goal_visual_ref(goal, visual)
            self._manip_resolved_visual = visual
            self._bind_object_working_memory(visual, live)
            if resolved is not None:
                self._manip_resolved = resolved
                if "displace" in pred_kinds and bool(getattr(resolved, "movable", False)):
                    try:
                        direction = self._infer_manip_direction(
                            text,
                            target_xy=(
                                float(resolved.position[0]),
                                float(resolved.position[1]),
                            ),
                            embed_fn=embed_fn,
                        )
                        self._manip_episode = ManipulationEpisode.begin(
                            resolved, requested_direction=direction
                        )
                    except Exception:
                        pass
            for node in (tt.tree.nodes.values() if tt.tree is not None else ()):
                if not getattr(node, "target_ref", None):
                    node.target_ref = visual.ref
            self._arm_nav_hold(live, reason="post_resolve")
            self._tt_complete_active(
                tt,
                live,
                diagnostics={
                    "resolved": visual.ref,
                    "vision": visual.to_dict(),
                    "deferred": True,
                    "non_production": bool((visual.diagnostics or {}).get("non_production")),
                    "source": (visual.diagnostics or {}).get("source"),
                },
            )
            self._task_tree_stage_enter_tick = live
        elif resolved is not None:
            goal = getattr(self, "_task_goal", None)
            if goal is not None:
                self._apply_goal_target_ref(goal, resolved)
            self._manip_resolved = resolved
            self._arm_nav_hold(live, reason="post_resolve")
            self._tt_complete_active(tt, live, diagnostics={"resolved": resolved.ref, "deferred": True})
            self._task_tree_stage_enter_tick = live

    def _tick_task_tree_goal(self, *, tick: int, obs: dict[str, float], fallen: bool) -> None:
        tt = self._ensure_task_tree()
        active = tt.active_node
        episode = getattr(self, "_manip_episode", None)
        resolved = getattr(self, "_manip_resolved", None)
        if active is None:
            self._maybe_finalize_task_tree(tick)
            return

        kind = str(active.kind)
        if kind == "resolve_target":
            self._tick_deferred_vision_resolve(tick=int(tick), fallen=bool(fallen))
            return

        base = self._humanoid_base_env()
        approach_m = nav_stop_m()
        if base is not None and hasattr(base, "manip_approach_m"):
            try:
                approach_m = min(float(base.manip_approach_m()), reach_start_m() + 0.05)
            except (TypeError, ValueError):
                pass
        reach_min = (
            int(base.manip_reach_min_ticks())
            if base is not None and hasattr(base, "manip_reach_min_ticks")
            else 16
        )
        push_every = (
            int(base.manip_push_every())
            if base is not None and hasattr(base, "manip_push_every")
            else 4
        )

        agent_xy, _ = self._agent_xy_forward()
        ref = str(getattr(resolved, "ref", "") or active.target_ref or "")
        target_xy = self._target_xy(ref) if ref and not str(ref).startswith("vision:") else None
        oracle_eval = self._eval_oracle_dist_m()
        if oracle_eval is not None:
            oracle_dist = float(oracle_eval)
        elif target_xy is not None:
            oracle_dist = float(
                math.hypot(target_xy[0] - agent_xy[0], target_xy[1] - agent_xy[1])
            )
        else:
            oracle_dist = 999.0
        if vision_resolve_enabled():
            owm = self._update_object_working_memory(int(tick))
            if owm is not None and owm.is_usable(int(tick)):
                dist = float(owm.range_m)
            else:
                vt = getattr(self, "_manip_resolved_visual", None)
                if vt is not None and vt.range_m is not None:
                    dist = float(vt.range_m)
                else:
                    dist = float(oracle_dist)
                    owm = None
        else:
            dist = float(oracle_dist)
            owm = None

        owm_range = (
            float(owm.range_m)
            if owm is not None and owm.is_usable(int(tick))
            else None
        )
        dist = self._blend_dist_with_physics_range(float(dist), owm_range, int(tick))

        stop_for_rebind = float(
            active.expected_state.get("stop_distance", nav_stop_m())
            if kind in ("approach", "approach_target")
            else nav_stop_m()
        )
        phys_for_rebind = self._physics_range_to_locked_body()
        if phys_for_rebind is not None:
            self._maybe_rebind_on_physics_range_desync(
                int(tick),
                phys=float(phys_for_rebind),
                owm_range=owm_range,
                kind=str(kind),
                stop=float(stop_for_rebind),
            )

        self._maybe_rebind_vision_on_divergence(
            int(tick),
            oracle_dist=float(oracle_dist),
            kind=str(kind),
        )
        self._maybe_correct_vision_range_mismatch(
            int(tick),
            oracle_dist=float(oracle_dist),
            kind=str(kind),
        )

        kind = active.kind
        stage_enter = int(getattr(self, "_task_tree_stage_enter_tick", tick))

        _approach_fall_kinds = ("approach", "approach_target")
        _reach_verify_fall_kinds = ("reach_contact", "reach_target", "verify_goal", "verify_target")
        if fallen and kind in _approach_fall_kinds + _reach_verify_fall_kinds:
            streak = int(getattr(self, "_task_fall_streak", 0)) + 1
            self._task_fall_streak = streak
            self._task_fallen_ticks = int(getattr(self, "_task_fallen_ticks", 0)) + 1
            if kind in _approach_fall_kinds and streak == 1:
                self._arm_nav_hold(int(tick), reason="fallen_during_approach")
                self._capture_task_fall_approach_baseline(float(dist))
            # Do not fail/clear the task on brief falls — hard reset is already
            # deferred by embodiment protection; keep approach alive.
            protected = False
            try:
                from engine.task_binding import human_task_embodiment_protected

                protected = bool(human_task_embodiment_protected(self))
            except Exception:
                protected = False
            after_assist_ticks = 0
            if bool(getattr(self, "_task_fall_assist_used", False)):
                after_assist_ticks = (
                    int(getattr(self, "_task_fallen_after_assist_ticks", 0)) + 1
                )
                self._task_fallen_after_assist_ticks = after_assist_ticks
            try:
                fail_after_assist_ticks = int(
                    os.environ.get("RKK_TASK_FALL_FAIL_TICKS", "200")
                )
            except ValueError:
                fail_after_assist_ticks = 200
            fail_after_assist_ticks = max(1, min(fail_after_assist_ticks, 5000))
            # Near locked body: keep recovering in place — do not abort the touch stage.
            near_goal = False
            try:
                near_goal = bool(self._task_fall_assist_near_goal())
            except Exception:
                near_goal = False
            if near_goal and kind in _reach_verify_fall_kinds + _approach_fall_kinds:
                fail_after_assist_ticks = max(fail_after_assist_ticks, 800)
            fail_after_assist = (
                bool(getattr(self, "_task_fall_assist_used", False))
                and after_assist_ticks >= fail_after_assist_ticks
                and not near_goal
            )
            fail_reason = (
                "fallen_during_approach"
                if kind in _approach_fall_kinds
                else f"fallen_during_{kind}"
            )
            if kind in _reach_verify_fall_kinds and protected and not fail_after_assist:
                if streak >= fail_after_assist_ticks:
                    self._tt_fail_active(tt, tick, fail_reason, retryable=True)
                    self._maybe_finalize_task_tree(tick)
                    return
            elif fail_after_assist or (
                not protected
                and streak >= 3
                and active.tick_deadline
                and tick > stage_enter + 30
            ):
                self._tt_fail_active(tt, tick, fail_reason, retryable=True)
                self._maybe_finalize_task_tree(tick)
                return
        else:
            self._task_fall_streak = 0
            self._task_fall_start_range = None
            self._task_fall_start_com = None

        self._register_task_navigation(
            active=active,
            dist=dist,
            approach_m=approach_m,
            fallen=fallen,
            obs=obs,
            owm=owm,
        )
        self._register_task_manipulation(
            active=active,
            dist=dist,
            fallen=fallen,
            owm=owm,
        )

        ctx = self._human_task_verify_ctx()
        goal = getattr(self, "_task_goal", None)
        if goal is None:
            tb = getattr(self, "_task_binding", None)
            active_task = tb.active_task if tb is not None else None
            if active_task is None and tb is not None:
                active_task = getattr(tb, "_active", None)
            if active_task is not None:
                goal = getattr(active_task, "goal", None)

        if kind in ("approach", "approach_target"):
            stop = float(active.expected_state.get("stop_distance", nav_stop_m()))
            streak = int(getattr(self, "_nav_arrival_streak", 0))
            if dist <= stop:
                streak += 1
            else:
                streak = 0
            self._nav_arrival_streak = streak
            if streak >= _nav_arrival_streak_needed():
                self._nav_arrival_streak = 0
                self._release_scene_hard_lock()
                self._tt_complete_active(tt, tick)
                self._task_tree_stage_enter_tick = tick
            elif active.tick_deadline and tick > int(active.tick_deadline):
                self._tt_fail_active(tt, tick, "approach_timeout", retryable=True)

        elif kind == "reach_contact":
            min_elapsed = int(tick) - stage_enter >= reach_min
            # Touch / contact stages require physics contact — proximity alone
            # must not skip the neural/physics verify path.
            has_contact = self._manip_has_contact(resolved)
            if min_elapsed and has_contact:
                self._tt_complete_active(tt, tick)
                self._task_tree_stage_enter_tick = tick
            elif active.tick_deadline and tick > int(active.tick_deadline):
                self._tt_fail_active(tt, tick, "contact_timeout", retryable=True)

        elif kind == "verify_goal":
            if goal is not None:
                ok, score, diag = evaluate_goal(goal, obs, ctx)
                active.last_pe = 1.0 - float(score)
                min_elapsed = int(tick) - stage_enter >= reach_min
                if ok and min_elapsed:
                    self._task_goal_verified = True
                    self._tt_complete_active(tt, tick, diagnostics=diag)
                    self._task_tree_stage_enter_tick = tick
                elif active.tick_deadline and tick > int(active.tick_deadline):
                    reason = str(diag.get("reason", "verify_failed"))
                    self._tt_fail_active(tt, tick, reason, retryable=False)
            elif active.tick_deadline and tick > int(active.tick_deadline):
                self._tt_fail_active(tt, tick, "verify_no_goal", retryable=False)

        elif kind == "reach_target":
            in_range = dist <= approach_m
            min_elapsed = int(tick) - stage_enter >= reach_min
            if in_range and min_elapsed:
                self._tt_complete_active(tt, tick)
                self._task_tree_stage_enter_tick = tick
            elif active.tick_deadline and tick > int(active.tick_deadline):
                self._tt_fail_active(tt, tick, "reach_timeout", retryable=True)

        elif kind == "push_target":
            if (
                episode is not None
                and resolved is not None
                and base is not None
                and resolved.body_id is not None
            ):
                last_push = int(getattr(self, "_task_tree_last_push_tick", -push_every))
                if tick - last_push >= push_every:
                    direction = episode.requested_direction or (1.0, 0.0)
                    push_out = base.apply_manipulation_push(
                        int(resolved.body_id), direction, _manip_push_force()
                    )
                    self._manip_diag["last_push"] = push_out
                    self._task_tree_last_push_tick = tick
            if episode is not None and target_xy is not None:
                vdiag = verify_manipulation(episode, target_xy, intent_signals=obs)
                self._manip_diag["verify"] = vdiag
                if vdiag.get("success"):
                    self._tt_complete_active(tt, tick, diagnostics=vdiag)
                    self._task_tree_stage_enter_tick = tick
                elif active.tick_deadline and tick > int(active.tick_deadline):
                    self._tt_fail_active(tt, tick, "push_timeout", retryable=True)
            elif active.tick_deadline and tick > int(active.tick_deadline):
                self._tt_fail_active(tt, tick, "push_timeout", retryable=True)

        elif kind == "verify_target":
            if episode is not None and target_xy is not None:
                vdiag = verify_manipulation(episode, target_xy, intent_signals=obs)
                self._manip_diag["verify"] = vdiag
                if vdiag.get("success"):
                    self._tt_complete_active(tt, tick, diagnostics=vdiag)
                elif active.tick_deadline and tick > int(active.tick_deadline):
                    self._tt_fail_active(tt, tick, "verify_failed", retryable=False)
            elif active.tick_deadline and tick > int(active.tick_deadline):
                self._tt_fail_active(tt, tick, "verify_no_target", retryable=False)

        self._maybe_finalize_task_tree(tick)

    def _tick_task_tree_generic_recover(
        self,
        *,
        tick: int,
        obs: dict[str, float],
        fallen: bool,
    ) -> None:
        tt = self._ensure_task_tree()
        tb = self._ensure_task_binding()
        task = tb.active_task
        active = tt.active_node

        if active is None:
            self._maybe_finalize_task_tree(tick)
            return

        if active.kind in ("execute_goal", "recover_posture"):
            if task is None:
                self._maybe_finalize_task_tree(tick)
                return
            finished = tb.tick_verify(
                obs, tick, fallen=fallen, ctx=self._human_task_verify_ctx()
            )
            if finished is None:
                s2 = getattr(self, "_system2", None)
                if s2 is not None and hasattr(s2, "working_memory"):
                    s2.working_memory.write(
                        "human_task_pe",
                        float(task.last_pe),
                        text=task.text[:80],
                        tick=tick,
                        source="task_binding",
                    )
                return
            if finished.status == "done":
                self._tt_complete_active(tt, tick, diagnostics=finished.last_diag)
                self._task_tree_stage_enter_tick = tick
                s2 = getattr(self, "_system2", None)
                if s2 is not None and hasattr(s2, "working_memory"):
                    wm = s2.working_memory
                    wm.write(
                        "human_task_status",
                        1.0,
                        text=finished.text[:80],
                        tick=tick,
                        source="task_binding",
                    )
                nxt = tt.active_node
                if nxt is not None and nxt.kind in ("verify_goal", "verify_posture"):
                    bound = getattr(tb, "_active", None)
                    if bound is not None and bound.status == "done":
                        self._tt_complete_active(tt, tick)
            else:
                self._tt_fail_active(
                    tt,
                    tick,
                    str(finished.last_diag.get("reason", "failed")),
                )
        elif active.kind in ("verify_goal", "verify_posture"):
            bound = getattr(tb, "_active", None)
            if bound is not None and bound.status == "done":
                self._tt_complete_active(tt, tick)
            elif bound is not None and bound.status == "failed":
                self._tt_fail_active(tt, tick, "verify_failed")
            elif int(tick) > int(getattr(bound, "tick_deadline", tick)):
                self._tt_fail_active(tt, tick, "deadline")

        self._maybe_finalize_task_tree(tick)

    def _tick_task_tree(self, *, fallen: bool, obs: dict[str, float], tick: int) -> None:
        kind = str(getattr(self, "_task_tree_kind", "generic") or "generic")
        if kind in ("manipulate", "goal"):
            self._tick_task_tree_goal(tick=tick, obs=obs, fallen=fallen)
        else:
            self._tick_task_tree_generic_recover(tick=tick, obs=obs, fallen=fallen)

    def _tick_human_task(self, *, fallen: bool) -> None:
        if not task_binding_enabled():
            return

        tick = int(getattr(self, "tick", 0))

        if getattr(self, "_task_tree_cleared_pending_ack", False):
            tt = self._ensure_task_tree()
            tt.acknowledge_clear()
            self._task_tree_cleared_pending_ack = False
            self._complete_human_command_cleanup(tick)
            return

        obs: dict[str, float] = {}
        try:
            obs = dict(self._graph_vec_cached())
        except Exception:
            try:
                obs = dict(self.agent.env.observe())
            except Exception:
                return

        obs = self._inject_task_obs(obs)

        try:
            from engine.task_executive import active_tree_stage_kind, neutralize_blocked_graph_intents

            neutralize_blocked_graph_intents(self.agent.graph, active_tree_stage_kind(self))
        except Exception:
            pass

        if task_tree_enabled():
            tt = getattr(self, "_task_tree_ctrl", None)
            if tt is not None and (tt.is_active or (tt.tree and tt.tree.root_status in TERMINAL_STATUSES)):
                prev_fallen = bool(getattr(self, "_task_log_prev_fallen", False))
                if fallen and not prev_fallen:
                    self._task_log_fall_during_task(tick)
                self._task_log_prev_fallen = bool(fallen)
                # Register navigation BEFORE progress dump so human_task_motor
                # shows gait_coupling / supports — not static tree approach stubs.
                self._tick_task_tree(fallen=fallen, obs=obs, tick=tick)
                try:
                    obs = self._inject_task_obs(obs)
                except Exception:
                    pass
                self._task_log_progress(tick, obs=obs, fallen=fallen)
                return

        tb = self._ensure_task_binding()
        task = tb.active_task
        if task is None:
            return

        prev_fallen = bool(getattr(self, "_task_log_prev_fallen", False))
        if fallen and not prev_fallen:
            self._task_log_fall_during_task(tick)
        self._task_log_prev_fallen = bool(fallen)
        self._task_log_progress(tick, obs=obs, fallen=fallen)

        finished = tb.tick_verify(
            obs, tick, fallen=fallen, ctx=self._human_task_verify_ctx()
        )
        if finished is None:
            s2 = getattr(self, "_system2", None)
            if s2 is not None and hasattr(s2, "working_memory"):
                s2.working_memory.write(
                    "human_task_pe",
                    float(task.last_pe),
                    text=task.text[:80],
                    tick=tick,
                    source="task_binding",
                )
            return

        if task_tree_enabled() and getattr(self, "_task_tree_ctrl", None) is not None:
            return

        self._on_human_task_finished(finished, tick)

    def _on_human_task_finished(self, task: Any, tick: int) -> None:
        s2 = getattr(self, "_system2", None)
        if s2 is not None and hasattr(s2, "working_memory"):
            wm = s2.working_memory
            wm.write(
                "human_task_active",
                0.0,
                text=task.text[:80],
                tick=tick,
                source="task_binding",
            )
            wm.write(
                "human_task_status",
                1.0 if task.status == "done" else 0.0,
                text=task.status,
                tick=tick,
                source="task_binding",
            )

        if task.status == "done":
            self._emit_task_report(tick, task.text, done=True)
        elif task.status == "failed":
            self._emit_task_report(tick, task.text, done=False)

        reason = ""
        if isinstance(getattr(task, "last_diag", None), dict):
            reason = str(task.last_diag.get("reason", ""))
        self._task_log_finished(
            tick,
            status=str(task.status),
            reason=reason or str(task.status),
            final_pe=float(getattr(task, "last_pe", 0.0)),
        )

        tb = getattr(self, "_task_binding", None)
        if tb is not None:
            tb.clear()
        ic = getattr(self, "_intention_cortex", None)
        if ic is not None and hasattr(ic, "clear_human_command"):
            ic.clear_human_command()
        try:
            graph = self.agent.graph
            if "self_goal_active" in graph.nodes:
                graph.nodes["self_goal_active"] = 0.0
        except Exception:
            pass

    def handle_human_command(self, command_text: str) -> dict[str, Any]:
        """HTTP/WS: language hearing → task binding → WM + Intention."""
        if not grounded_language_enabled():
            return {"ok": False, "error": "grounded_language_disabled"}
        text = str(command_text or "").strip()
        if not text:
            return {"ok": False, "error": "empty"}
        self._ensure_grounded_language()
        self._ensure_grounded_motor_drain_hook()
        gl = getattr(self, "_grounded_lang", None)
        if gl is None:
            return {"ok": False, "error": "controller_unavailable"}

        tick = int(getattr(self, "tick", 0))
        self._clear_human_command_state(tick)
        self._task_tree_reported = False
        self._task_tree_affect_done = False
        self._task_tree_cleared_pending_ack = False
        self._task_tree_stage_enter_tick = tick
        self._task_tree_last_push_tick = -999
        self._nav_arrival_streak = 0
        self._task_goal = None
        self._task_goal_verified = False
        self._task_fall_assist_used = False
        self._task_fallen_ticks = 0
        self._task_fallen_after_assist_ticks = 0
        self._task_fall_protected_stall_ticks = 0
        self._task_fall_start_range = None
        self._task_fall_start_com = None

        use_tb = task_binding_enabled()
        use_tree = task_tree_enabled() and use_tb
        result = gl.ingest_command(
            self.agent.graph,
            text,
            apply_motor_patch=not use_tb,
        )
        if not result.get("ok"):
            return result

        if not use_tb:
            self._queue_grounded_motor_intent(str(result.get("tag", "")))
            self._drain_pending_grounded_motor()

        obs: dict[str, float] = {}
        try:
            obs = dict(self._graph_vec_cached())
        except Exception:
            try:
                obs = dict(self.agent.env.observe())
            except Exception:
                obs = {}

        out: dict[str, Any] = dict(result)
        tag = str(result.get("tag", ""))

        fallen_flag = False
        base = self._humanoid_base_env()
        try:
            if base is not None and callable(getattr(base, "is_fallen", None)):
                fallen_flag = bool(base.is_fallen())
        except Exception:
            pass

        cmd_kind = command_kind_for_text(text, tag=tag, fallen=fallen_flag)
        goal = self._ground_command_goal(text, gl)
        use_goal = goal is not None and bool(goal.predicates)
        if use_goal:
            cmd_kind = "goal"
            pred_kinds = self._goal_predicate_kinds(goal)
            if pred_kinds & {"displace", "contact", "reduce_distance"}:
                cmd_kind = "goal"
            elif pred_kinds == {"state_key"}:
                cmd_kind = "generic"
        self._task_tree_kind = cmd_kind
        self._task_goal = goal if use_goal else None
        if use_goal:
            tag = ""
        self._task_log_command_received(
            tick,
            text,
            command_kind=cmd_kind,
            tag=tag,
        )

        if use_tree and use_goal and goal is not None:
            tt = self._ensure_task_tree()
            needs_target = bool(goal.diagnostics.get("needs_target", False))
            tree = tt.bind_goal(
                goal,
                tick,
                needs_target=needs_target,
                target_ref=goal.target_ref,
            )
            self._task_log_tree_bound(tick, tt)
            out["task_goal"] = goal.to_dict()
            out["task_tree"] = tt.snapshot(tick)

            resolved: ResolvedObject | None = getattr(self, "_manip_resolved", None)
            visual: VisualTarget | None = getattr(self, "_manip_resolved_visual", None)
            hard_fail = False
            fail_reason = "no_target"
            diag: dict[str, Any] = {}
            if needs_target:
                # Vision resolve is heavy (encode + depth) and must not block the
                # command/API thread — that freezes UI and lets sim ticks race ahead
                # of a stale nav_hold. Always finish on the tick path.
                if vision_resolve_enabled():
                    reason = "fallen" if fallen_flag else "async_vision"
                    self._deferred_vision_resolve = {
                        "text": text,
                        "require_movable": "displace" in pred_kinds,
                        "interaction_kinds": list(pred_kinds),
                    }
                    live = self._live_tick(tick)
                    self._arm_nav_hold(live, reason="command_while_fallen" if fallen_flag else "resolve_pending")
                    diag = {
                        "reason": f"deferred_{reason}",
                        "resolve_mode": "vision",
                    }
                    self._manip_diag = dict(diag)
                    out["manipulation"] = dict(diag)
                    task_log_event(
                        "resolve_deferred",
                        tick=live,
                        reason=reason,
                    )
                    # Leave resolve_target stage active; tick will finish it.
                else:
                    embed_fn = gl.embedder.embed if gl is not None else None
                    try:
                        resolved, visual, diag = self._resolve_command_target(
                            text,
                            embed_fn=embed_fn,
                            require_movable="displace" in pred_kinds,
                            interaction_kinds=frozenset(pred_kinds),
                        )
                    except Exception as exc:
                        diag = {
                            "reason": f"resolver_error:{exc}",
                            "resolve_mode": "oracle",
                        }
                        resolved, visual = None, None
                    self._manip_diag = dict(diag)
                    out["manipulation"] = dict(diag)
                    self._task_log_target_resolution(tick, text, resolved, diag)

                    if resolved is None or (
                        "displace" in pred_kinds and resolved is not None and not resolved.movable
                    ):
                        if "displace" in pred_kinds:
                            hard_fail = True
                            fail_reason = str(diag.get("reason", "no_target"))
                        elif resolved is None and pred_kinds & {
                            "reduce_distance",
                            "contact",
                            "displace",
                        }:
                            hard_fail = True
                            fail_reason = str(diag.get("reason", "no_target"))

                    if hard_fail:
                        self._tt_fail_active(tt, tick, fail_reason, retryable=False)
                        out["task_tree"] = tt.snapshot(tick)
                        self._task_tree_reported = True
                        fail_body = "Не вижу цель или не могу сдвинуть объект."
                        self._emit_task_report(tick, text, done=False, body=fail_body)
                        self._apply_task_outcome_affect(False)
                        self._task_log_finished(tick, status="failed", reason=fail_reason)
                        tt.clear(tick)
                        self._task_tree_cleared_pending_ack = True
                        out["ok"] = True
                        out["task_binding"] = False
                        return out

                    if resolved is not None:
                        self._apply_goal_target_ref(goal, resolved)
                        self._manip_resolved = resolved
                        self._manip_resolved_visual = None
                        if "displace" in pred_kinds:
                            target_xy = (float(resolved.position[0]), float(resolved.position[1]))
                            embed_fn = gl.embedder.embed if gl is not None else None
                            direction = self._infer_manip_direction(
                                text, target_xy=target_xy, embed_fn=embed_fn
                            )
                            self._manip_episode = ManipulationEpisode.begin(
                                resolved, requested_direction=direction
                            )
                        if tt.active_node is not None and tt.active_node.kind == "resolve_target":
                            self._tt_complete_active(
                                tt,
                                tick,
                                diagnostics={"resolved": resolved.ref},
                            )
                        out["manipulation_target"] = resolved.ref

            if use_tb:
                tb = self._ensure_task_binding()
                embed_fn = gl.embedder.embed if gl is not None else None
                bind_agent_xy, bind_agent_fwd = self._agent_xy_forward()
                bind_target_xy = None
                if resolved is not None:
                    bind_target_xy = (
                        float(resolved.position[0]),
                        float(resolved.position[1]),
                    )
                # Vision path: synthetic target ahead for WM binding only
                if vision_resolve_enabled() and visual is not None and bind_target_xy is None:
                    ax, ay = bind_agent_xy
                    fx, fy = bind_agent_fwd
                    r = float(visual.range_m or 1.0)
                    bind_target_xy = (ax + fx * r, ay + fy * r)
                task = tb.bind_command(
                    self.agent.graph,
                    obs,
                    text,
                    tick,
                    embed_fn=embed_fn,
                    goal=goal,
                    agent_xy=bind_agent_xy,
                    target_xy=bind_target_xy,
                    agent_forward=bind_agent_fwd,
                )
                exp: dict[str, float] = {}
                for p in goal.predicates:
                    if p.kind == "state_key" and p.key:
                        exp[str(p.key)] = float(p.target_value)
                if exp:
                    task.expected_state.update(exp)
                self._task_log_imagine_done(tick, task)
                out["task"] = task.to_dict()
                out["task_binding"] = True
                if tt.active_node is not None and tt.active_node.kind == "imagine_goal":
                    self._tt_complete_active(tt, tick)
                ic = getattr(self, "_intention_cortex", None)
                if ic is None and hasattr(self, "_ensure_intention_cortex"):
                    try:
                        ic = self._ensure_intention_cortex()
                    except Exception:
                        ic = None
                if ic is not None and hasattr(ic, "absorb_human_task"):
                    stage_kind = ""
                    if tt.active_node is not None:
                        stage_kind = str(tt.active_node.kind)
                    ic.absorb_human_task(task, obs, tick, stage_kind=stage_kind)
            else:
                out["task_binding"] = False

            from engine.system2.controller import write_human_command_wm

            write_human_command_wm(self, text, tick)
            try:
                graph = self.agent.graph
                if "self_goal_active" in graph.nodes:
                    graph.nodes["self_goal_active"] = 1.0
            except Exception:
                pass
            out["task_tree"] = tt.snapshot(tick)
            return out

        if use_tree:
            embed_fn = gl.embedder.embed if gl is not None else None
            if embed_fn is not None and not use_goal:
                reason = "no_goal_predicates"
                if goal is not None:
                    fb = goal.diagnostics.get("fallback")
                    if fb:
                        reason = str(fb)
                task_log_event(
                    "grounding_fallback",
                    tick=tick,
                    reason=reason,
                    command_kind=cmd_kind,
                    text=str(text)[:120],
                )
            tt = self._ensure_task_tree()
            tree_kind = cmd_kind
            if cmd_kind == "manipulate":
                tree_kind = "manipulate_object"
            tree = tt.bind_command(
                text,
                tick,
                command_kind=tree_kind,
            )
            self._task_log_tree_bound(tick, tt)
            out["task_tree"] = tt.snapshot(tick)

            if cmd_kind == "manipulate":
                resolved: ResolvedObject | None = None
                visual: VisualTarget | None = None
                diag: dict[str, Any] = {}
                if vision_resolve_enabled():
                    self._deferred_vision_resolve = {
                        "text": text,
                        "require_movable": True,
                        "interaction_kinds": [
                            "displace",
                            "contact",
                            "reduce_distance",
                        ],
                    }
                    live = self._live_tick(tick)
                    self._arm_nav_hold(live, reason="resolve_pending")
                    diag = {"reason": "deferred_async_vision", "resolve_mode": "vision"}
                    self._manip_diag = dict(diag)
                    out["manipulation"] = dict(diag)
                    task_log_event(
                        "resolve_deferred",
                        tick=live,
                        reason="async_vision",
                    )
                else:
                    embed_fn = gl.embedder.embed if gl is not None else None
                    try:
                        resolved, visual, diag = self._resolve_command_target(
                            text,
                            embed_fn=embed_fn,
                            require_movable=True,
                            interaction_kinds=frozenset(
                                {"displace", "contact", "reduce_distance"}
                            ),
                        )
                    except Exception as exc:
                        diag = {"reason": f"resolver_error:{exc}"}
                    self._manip_diag = dict(diag)
                    out["manipulation"] = dict(diag)
                    self._task_log_target_resolution(tick, text, resolved, diag)

                    fail_oracle = resolved is None or not resolved.movable
                    if fail_oracle:
                        reason = str(diag.get("reason", "no_target"))
                        self._tt_fail_active(tt, tick, reason, retryable=False)
                        out["task_tree"] = tt.snapshot(tick)
                        self._task_tree_reported = True
                        fail_body = "Не вижу цель или не могу сдвинуть объект."
                        self._emit_task_report(tick, text, done=False, body=fail_body)
                        self._apply_task_outcome_affect(False)
                        self._task_log_finished(tick, status="failed", reason=reason)
                        tt.clear(tick)
                        self._task_tree_cleared_pending_ack = True
                        out["ok"] = True
                        out["task_binding"] = False
                        return out

                    if resolved is not None and resolved.movable:
                        direction = self._infer_manip_direction(
                            text,
                            target_xy=(
                                float(resolved.position[0]),
                                float(resolved.position[1]),
                            ),
                            embed_fn=embed_fn,
                        )
                        self._manip_episode = ManipulationEpisode.begin(
                            resolved, requested_direction=direction
                        )
                        self._manip_resolved = resolved
                        for node in tree.nodes.values():
                            node.target_ref = resolved.ref
                        if tt.active_node is not None and tt.active_node.kind == "resolve_target":
                            self._tt_complete_active(
                                tt,
                                tick,
                                diagnostics={"resolved": resolved.ref},
                            )
                        out["manipulation_target"] = resolved.ref

                out["task_tree"] = tt.snapshot(tick)

                from engine.system2.controller import write_human_command_wm

                write_human_command_wm(self, text, tick)
                try:
                    graph = self.agent.graph
                    if "self_goal_active" in graph.nodes:
                        graph.nodes["self_goal_active"] = 1.0
                except Exception:
                    pass
                out["task_binding"] = False
                return out

            if cmd_kind == "generic":
                self._tt_complete_active(tt, tick)

        if use_tb:
            tb = self._ensure_task_binding()
            embed_fn = gl.embedder.embed if gl is not None else None
            task = tb.bind_command(
                self.agent.graph, obs, text, tick, embed_fn=embed_fn
            )
            self._task_log_imagine_done(tick, task)
            out["task"] = task.to_dict()
            out["task_binding"] = True

            from engine.system2.controller import write_human_command_wm

            write_human_command_wm(self, text, tick)

            ic = getattr(self, "_intention_cortex", None)
            if ic is None and hasattr(self, "_ensure_intention_cortex"):
                try:
                    ic = self._ensure_intention_cortex()
                except Exception:
                    ic = None
            if ic is not None and hasattr(ic, "absorb_human_task"):
                stage_kind = ""
                tt2 = getattr(self, "_task_tree_ctrl", None)
                if tt2 is not None and tt2.active_node is not None:
                    stage_kind = str(tt2.active_node.kind)
                ic.absorb_human_task(task, obs, tick, stage_kind=stage_kind)

            try:
                graph = self.agent.graph
                if "self_goal_active" in graph.nodes:
                    graph.nodes["self_goal_active"] = 1.0
            except Exception:
                pass

            if use_tree:
                out["task_tree"] = self._ensure_task_tree().snapshot(tick)

        return out

    def grounded_lang_generate(self, obs: dict[str, float] | None = None) -> str:
        """Graph-grounded utterance for verbal layer."""
        self._ensure_grounded_language()
        self._ensure_grounded_motor_drain_hook()
        gl = getattr(self, "_grounded_lang", None)
        if gl is None:
            return ""
        o = obs or {}
        if not o:
            try:
                o = dict(self.agent.env.observe())
            except Exception:
                o = {}
        return gl.generate_utterance(self.agent.graph, o)

    def _grounded_lang_snapshot(self) -> dict[str, Any]:
        if not grounded_language_enabled():
            return {"enabled": False}
        gl = getattr(self, "_grounded_lang", None)
        tb = getattr(self, "_task_binding", None)
        snap: dict[str, Any] = {}
        if gl is None:
            snap = {"enabled": True, "initialized": False}
        else:
            snap = gl.snapshot()
        if tb is not None:
            snap["task_binding"] = tb.snapshot()
        if task_tree_enabled():
            snap["task_tree"] = self._task_tree_snapshot()
        return snap

    def _schedule_grounded_lang_bootstrap(self) -> None:
        """Async Ollama bootstrap so sim init is not blocked."""

        def run() -> None:
            try:
                self._ensure_grounded_language()
                gl = getattr(self, "_grounded_lang", None)
                if gl is not None:
                    gl.bootstrap_store()
            except Exception:
                pass

        threading.Thread(
            target=run, daemon=True, name="rkk-grounded-lang-bootstrap"
        ).start()
