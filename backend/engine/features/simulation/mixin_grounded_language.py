"""Simulation mixin: Grounded Language + human task binding (AGI command loop)."""
from __future__ import annotations

import os
import threading
from typing import Any

from engine.grounded_language import (
    GroundedLanguageController,
    grounded_language_enabled,
    motor_intents_from_tag,
)
from engine.task_binding import TaskBindingController, task_binding_enabled


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


class SimulationGroundedLanguageMixin:
    def _ensure_task_binding(self) -> TaskBindingController:
        tb = getattr(self, "_task_binding", None)
        if tb is None:
            tb = TaskBindingController()
            self._task_binding = tb
        return tb

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

    def _tick_human_task(self, *, fallen: bool) -> None:
        if not task_binding_enabled():
            return
        tb = self._ensure_task_binding()
        task = tb.active_task
        if task is None:
            return
        obs: dict[str, float] = {}
        try:
            obs = dict(self._graph_vec_cached())
        except Exception:
            try:
                obs = dict(self.agent.env.observe())
            except Exception:
                return

        tick = int(getattr(self, "tick", 0))
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

    def _emit_task_report(self, tick: int, command_text: str, *, done: bool) -> None:
        verbal = getattr(self, "_verbal", None)
        if verbal is None:
            return
        try:
            from engine.verbal_action import AgentMessage, SpeechType, ollama_chat_speech_enabled

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
                text=body.strip(),
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

        use_tb = task_binding_enabled()
        result = gl.ingest_command(
            self.agent.graph,
            text,
            apply_motor_patch=not use_tb,
        )
        if not result.get("ok"):
            return result

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

        tick = int(getattr(self, "tick", 0))
        out: dict[str, Any] = dict(result)

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
