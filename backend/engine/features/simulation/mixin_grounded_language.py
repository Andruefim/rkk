"""Simulation mixin: Grounded Language + human task binding (AGI command loop)."""
from __future__ import annotations

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
from engine.goal_navigation import navigation_intents
from engine.manipulation_control import manipulation_intents
from engine.manipulation_verify import ManipulationEpisode, verify_manipulation
from engine.object_resolver import ResolvedObject, resolve_manipulation_target
from engine.success_predicates import evaluate_goal
from engine.task_binding import TaskBindingController, task_binding_enabled
from engine.task_goal import TaskGoal
from engine.task_logger import summarize_expected_state, task_log_event
from engine.task_observation import (
    build_task_observations,
    inject_task_observations,
    nav_stop_m,
    reach_start_m,
    sync_task_obs_to_graph,
)
from engine.task_tree import TERMINAL_STATUSES, TaskTreeController, task_tree_enabled


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
        for key in ("scene_semantics", "semantics", "candidates"):
            if key in diag and diag[key] is not None:
                fields[key] = diag[key]
        task_log_event("target_resolution", tick=int(tick), **fields)

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
            return result
        self._task_log_stage_failed(tick, active, reason)
        return result

    def _task_log_human_motor_targets(self) -> dict[str, float]:
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
        if not out:
            try:
                out = {
                    k: round(float(v), 4)
                    for k, v in self.task_tree_motor_targets().items()
                    if str(k).startswith("intent_")
                }
            except Exception:
                pass
        return out

    def _task_log_progress(self, tick: int, *, obs: dict[str, float], fallen: bool) -> None:
        if int(tick) % 50 != 0:
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
        task = tb.active_task if tb is not None else None
        if task is None and tb is not None:
            task = getattr(tb, "_active", None)
        if task is not None:
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

        fields: dict[str, Any] = {
            "node_kind": node_kind or None,
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
        arb = getattr(self, "_motor_arbiter", None)
        if arb is not None:
            snap = getattr(arb, "_last_diag", None) or {}
            if isinstance(snap, dict) and snap.get("sources"):
                fields["motor_sources"] = list(snap.get("sources") or [])
        task_log_event("task_progress", tick=int(tick), **fields)

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

    def _humanoid_base_env(self) -> Any | None:
        env = getattr(getattr(self, "agent", None), "env", None)
        if env is None:
            return None
        return getattr(env, "base_env", env)

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

    def _emit_task_report(
        self,
        tick: int,
        command_text: str,
        *,
        done: bool,
        body: str | None = None,
    ) -> None:
        verbal = getattr(self, "_verbal", None)
        if verbal is None:
            return
        try:
            from engine.verbal_action import AgentMessage, SpeechType, ollama_chat_speech_enabled

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
            msg = AgentMessage(
                tick=tick,
                speech_type=SpeechType.REPORT,
                text=str(body).strip(),
                concepts=["HUMAN_TASK"],
                curiosity=0.5,
                posture=0.5,
            )
            verbal._messages.append(msg)
            verbal._last_report_tick = tick
            verbal.total_messages += 1
            for cb in verbal._on_message:
                try:
                    cb(msg)
                except Exception:
                    pass
        except Exception:
            pass

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
            if reason.startswith("no_target") or "static" in reason:
                body = "Не вижу цель или не могу сдвинуть объект."
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

    def _manip_has_contact(self, resolved: ResolvedObject | None) -> bool:
        env = getattr(getattr(self, "agent", None), "env", None)
        if env is not None and bool(getattr(env, "_contact_flag", False)):
            return True
        base = self._humanoid_base_env()
        if base is None or resolved is None:
            return False
        body_id = getattr(resolved, "body_id", None)
        if body_id is None:
            return False
        fn = getattr(base, "_manip_has_contact", None)
        if callable(fn):
            return bool(fn(int(body_id)))
        return False

    def _human_task_verify_ctx(self) -> dict[str, Any]:
        """Scene context for goal predicate verification (distance, contact, displace)."""
        resolved = getattr(self, "_manip_resolved", None)
        episode = getattr(self, "_manip_episode", None)
        agent_xy, _ = self._agent_xy_forward()
        ctx: dict[str, Any] = {"agent_xy": agent_xy}
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
    ) -> None:
        if fallen or active is None:
            return
        if str(active.kind) not in ("reach_contact", "reach_target"):
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
        arb = getattr(self, "_motor_arbiter", None)
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

    def _register_task_navigation(
        self,
        *,
        active: Any,
        dist: float,
        approach_m: float,
        fallen: bool,
        obs: dict[str, float] | None = None,
    ) -> None:
        if fallen or active is None:
            self._set_task_nav_graph_flags(nav_active=False)
            return
        kind = str(active.kind)
        if kind not in ("approach", "reach_contact", "approach_target", "reach_target"):
            return

        resolved = getattr(self, "_manip_resolved", None)
        ref = str(getattr(resolved, "ref", "") or getattr(active, "target_ref", "") or "")
        target_xy = self._target_xy(ref) if ref else None
        if target_xy is None:
            self._set_task_nav_graph_flags(nav_active=False)
            return

        stop = float(active.expected_state.get("stop_distance", approach_m))
        if kind == "reach_contact":
            stop = min(stop, reach_start_m())

        agent_xy, agent_fwd = self._agent_xy_forward()
        obs_r = dict(obs or {})
        posture = float(
            obs_r.get(
                "posture_stability",
                obs_r.get("phys_posture_stability", 0.6),
            )
        )
        prev_xy = getattr(self, "_task_nav_prev_xy", None)
        intents: dict[str, float] = {}
        if kind in ("approach", "approach_target") or (
            kind == "reach_contact" and float(dist) > reach_start_m()
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

        arb = getattr(self, "_motor_arbiter", None)
        if intents:
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

    def _tick_task_tree_goal(self, *, tick: int, obs: dict[str, float], fallen: bool) -> None:
        tt = self._ensure_task_tree()
        active = tt.active_node
        episode = getattr(self, "_manip_episode", None)
        resolved = getattr(self, "_manip_resolved", None)
        if active is None:
            self._maybe_finalize_task_tree(tick)
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
        target_xy = self._target_xy(ref) if ref else None
        dist = (
            math.hypot(target_xy[0] - agent_xy[0], target_xy[1] - agent_xy[1])
            if target_xy is not None
            else 999.0
        )

        kind = active.kind
        stage_enter = int(getattr(self, "_task_tree_stage_enter_tick", tick))

        if fallen and kind in ("approach", "approach_target", "reach_contact", "reach_target"):
            streak = int(getattr(self, "_task_fall_streak", 0)) + 1
            self._task_fall_streak = streak
            if streak >= 3 and active.tick_deadline and tick > stage_enter + 30:
                self._tt_fail_active(tt, tick, "fallen_during_approach", retryable=True)
                self._maybe_finalize_task_tree(tick)
                return
        else:
            self._task_fall_streak = 0

        self._register_task_navigation(
            active=active,
            dist=dist,
            approach_m=approach_m,
            fallen=fallen,
            obs=obs,
        )
        self._register_task_manipulation(
            active=active,
            dist=dist,
            fallen=fallen,
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
                self._tt_complete_active(tt, tick)
                self._task_tree_stage_enter_tick = tick
            elif active.tick_deadline and tick > int(active.tick_deadline):
                self._tt_fail_active(tt, tick, "approach_timeout", retryable=True)

        elif kind == "reach_contact":
            reach_m = reach_start_m()
            min_elapsed = int(tick) - stage_enter >= reach_min
            has_contact = self._manip_has_contact(resolved)
            if min_elapsed and (has_contact or dist <= reach_m):
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
                self._task_log_progress(tick, obs=obs, fallen=fallen)
                self._tick_task_tree(fallen=fallen, obs=obs, tick=tick)
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
            hard_fail = False
            fail_reason = "no_target"
            diag: dict[str, Any] = {}
            if needs_target:
                embed_fn = gl.embedder.embed if gl is not None else None
                agent_xy, agent_fwd = self._agent_xy_forward()
                extras = self._sandbox_scene_extras()
                try:
                    resolved, diag = resolve_manipulation_target(
                        text,
                        extras,
                        agent_xy=agent_xy,
                        agent_forward=agent_fwd,
                        embed_fn=embed_fn,
                        require_movable="displace" in pred_kinds,
                        interaction_kinds=frozenset(pred_kinds),
                    )
                except Exception as exc:
                    diag = {"reason": f"resolver_error:{exc}"}
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
                diag: dict[str, Any] = {}
                embed_fn = gl.embedder.embed if gl is not None else None
                agent_xy, agent_fwd = self._agent_xy_forward()
                extras = self._sandbox_scene_extras()
                try:
                    resolved, diag = resolve_manipulation_target(
                        text,
                        extras,
                        agent_xy=agent_xy,
                        agent_forward=agent_fwd,
                        embed_fn=embed_fn,
                    )
                except Exception as exc:
                    diag = {"reason": f"resolver_error:{exc}"}
                self._manip_diag = dict(diag)
                out["manipulation"] = dict(diag)
                self._task_log_target_resolution(tick, text, resolved, diag)

                if resolved is None or not resolved.movable:
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

                direction = self._infer_manip_direction(
                    text,
                    target_xy=(float(resolved.position[0]), float(resolved.position[1])),
                    embed_fn=embed_fn,
                )
                episode = ManipulationEpisode.begin(
                    resolved, requested_direction=direction
                )
                self._manip_episode = episode
                self._manip_resolved = resolved
                for node in tree.nodes.values():
                    node.target_ref = resolved.ref
                self._tt_complete_active(
                    tt,
                    tick,
                    diagnostics={"resolved": resolved.ref},
                )
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
                out["manipulation_target"] = resolved.ref
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
