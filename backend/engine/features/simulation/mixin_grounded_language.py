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
from engine.manipulation_verify import ManipulationEpisode, verify_manipulation
from engine.object_resolver import ResolvedObject, resolve_manipulation_target
from engine.task_binding import TaskBindingController, task_binding_enabled
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

    def _humanoid_base_env(self) -> Any | None:
        env = getattr(getattr(self, "agent", None), "env", None)
        if env is None:
            return None
        return getattr(env, "base_env", env)

    def _agent_xy_forward(self) -> tuple[tuple[float, float], tuple[float, float]]:
        base = self._humanoid_base_env()
        xy = (0.0, 0.0)
        fwd = (1.0, 0.0)
        if base is not None:
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

    def _infer_manip_direction(self, text: str) -> tuple[float, float]:
        low = str(text or "").lower()
        if any(k in low for k in ("назад", "back", "backward")):
            return (-1.0, 0.0)
        if any(k in low for k in ("влево", "left")):
            return (0.0, 1.0)
        if any(k in low for k in ("вправо", "right")):
            return (0.0, -1.0)
        _, fwd = self._agent_xy_forward()
        return fwd

    def _clear_human_command_state(self, tick: int) -> None:
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
        gl.sync_speak_vector_from_state(graph, obs, fallen=fallen_flag, env=env)
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
        if base is None:
            return None
        pose = base.get_manipulation_target_pose(ref)
        if not pose:
            return None
        return float(pose.get("x", 0.0)), float(pose.get("y", 0.0))

    def _tick_task_tree_manipulate(self, *, tick: int, obs: dict[str, float]) -> None:
        tt = self._ensure_task_tree()
        active = tt.active_node
        episode = getattr(self, "_manip_episode", None)
        resolved = getattr(self, "_manip_resolved", None)
        if active is None:
            self._maybe_finalize_task_tree(tick)
            return

        base = self._humanoid_base_env()
        approach_m = (
            float(base.manip_approach_m())
            if base is not None and hasattr(base, "manip_approach_m")
            else 0.9
        )
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

        if kind == "approach_target":
            if dist <= approach_m:
                tt.complete_active(tick)
                self._task_tree_stage_enter_tick = tick
            elif active.tick_deadline and tick > int(active.tick_deadline):
                tt.fail_active(tick, "approach_timeout", retryable=True)

        elif kind == "reach_target":
            in_range = dist <= approach_m
            min_elapsed = int(tick) - stage_enter >= reach_min
            if in_range and min_elapsed:
                tt.complete_active(tick)
                self._task_tree_stage_enter_tick = tick
            elif active.tick_deadline and tick > int(active.tick_deadline):
                tt.fail_active(tick, "reach_timeout", retryable=True)

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
                    tt.complete_active(tick, diagnostics=vdiag)
                    self._task_tree_stage_enter_tick = tick
                elif active.tick_deadline and tick > int(active.tick_deadline):
                    tt.fail_active(tick, "push_timeout", retryable=True)
            elif active.tick_deadline and tick > int(active.tick_deadline):
                tt.fail_active(tick, "push_timeout", retryable=True)

        elif kind == "verify_target":
            if episode is not None and target_xy is not None:
                vdiag = verify_manipulation(episode, target_xy, intent_signals=obs)
                self._manip_diag["verify"] = vdiag
                if vdiag.get("success"):
                    tt.complete_active(tick, diagnostics=vdiag)
                elif active.tick_deadline and tick > int(active.tick_deadline):
                    tt.fail_active(tick, "verify_failed", retryable=False)
            elif active.tick_deadline and tick > int(active.tick_deadline):
                tt.fail_active(tick, "verify_no_target", retryable=False)

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
            finished = tb.tick_verify(obs, tick, fallen=fallen)
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
                tt.complete_active(tick, diagnostics=finished.last_diag)
                self._task_tree_stage_enter_tick = tick
                nxt = tt.active_node
                if nxt is not None and nxt.kind in ("verify_goal", "verify_posture"):
                    bound = getattr(tb, "_active", None)
                    if bound is not None and bound.status == "done":
                        tt.complete_active(tick)
            else:
                tt.fail_active(tick, str(finished.last_diag.get("reason", "failed")))
        elif active.kind in ("verify_goal", "verify_posture"):
            bound = getattr(tb, "_active", None)
            if bound is not None and bound.status == "done":
                tt.complete_active(tick)
            elif bound is not None and bound.status == "failed":
                tt.fail_active(tick, "verify_failed")
            elif int(tick) > int(getattr(bound, "tick_deadline", tick)):
                tt.fail_active(tick, "deadline")

        self._maybe_finalize_task_tree(tick)

    def _tick_task_tree(self, *, fallen: bool, obs: dict[str, float], tick: int) -> None:
        kind = str(getattr(self, "_task_tree_kind", "generic") or "generic")
        if kind == "manipulate":
            self._tick_task_tree_manipulate(tick=tick, obs=obs)
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

        if task_tree_enabled():
            tt = getattr(self, "_task_tree_ctrl", None)
            if tt is not None and (tt.is_active or (tt.tree and tt.tree.root_status in TERMINAL_STATUSES)):
                self._tick_task_tree(fallen=fallen, obs=obs, tick=tick)
                return

        tb = self._ensure_task_binding()
        task = tb.active_task
        if task is None:
            return

        finished = tb.tick_verify(obs, tick, fallen=fallen)
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
        self._task_tree_kind = cmd_kind

        if use_tree:
            tt = self._ensure_task_tree()
            tree_kind = cmd_kind
            if cmd_kind == "manipulate":
                tree_kind = "manipulate_object"
            tree = tt.bind_command(
                text,
                tick,
                command_kind=tree_kind,
            )
            out["task_tree"] = tt.snapshot(tick)

            if cmd_kind == "manipulate":
                resolved: ResolvedObject | None = None
                diag: dict[str, Any] = {}
                embed_fn = gl.embedder.embed if gl is not None else None
                if base is not None:
                    agent_xy, agent_fwd = self._agent_xy_forward()
                    extras: dict = {}
                    fn = getattr(base, "get_sandbox_scene_extras", None)
                    if callable(fn):
                        try:
                            extras = dict(fn() or {})
                        except Exception:
                            extras = {}
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

                if resolved is None or not resolved.movable:
                    reason = str(diag.get("reason", "no_target"))
                    tt.fail_active(tick, reason, retryable=False)
                    out["task_tree"] = tt.snapshot(tick)
                    self._task_tree_reported = True
                    fail_body = "Не вижу цель или не могу сдвинуть объект."
                    self._emit_task_report(tick, text, done=False, body=fail_body)
                    self._apply_task_outcome_affect(False)
                    tt.clear(tick)
                    self._task_tree_cleared_pending_ack = True
                    out["ok"] = True
                    out["task_binding"] = False
                    return out

                direction = self._infer_manip_direction(text)
                episode = ManipulationEpisode.begin(
                    resolved, requested_direction=direction
                )
                self._manip_episode = episode
                self._manip_resolved = resolved
                for node in tree.nodes.values():
                    node.target_ref = resolved.ref
                tt.complete_active(tick, diagnostics={"resolved": resolved.ref})
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
                tt.complete_active(tick)

        if use_tb:
            tb = self._ensure_task_binding()
            task = tb.bind_command(self.agent.graph, obs, text, tick)
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
                ic.absorb_human_task(task, obs, tick)

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
