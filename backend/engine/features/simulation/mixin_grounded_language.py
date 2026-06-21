"""Simulation mixin: Grounded Language (embedding ↔ graph, no heavy LLM loop)."""
from __future__ import annotations

import os
import threading
from typing import Any

from engine.grounded_language import GroundedLanguageController, grounded_language_enabled


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

    def _tick_grounded_language(self, *, fallen: bool) -> None:
        if not grounded_language_enabled():
            return
        tick = int(getattr(self, "tick", 0))
        if tick % _grounded_lang_every() != 0:
            return
        self._ensure_grounded_language()
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
        gl.sync_speak_vector_from_state(graph, obs)
        if fallen:
            gl.ingest_command(graph, "Встань, ты упал")
        if tick % _align_every() == 0 and obs:
            try:
                gl.alignment_step(graph, obs)
            except Exception:
                pass

    def handle_human_command(self, command_text: str) -> dict[str, Any]:
        """HTTP/WS: grounded language command → graph sensory nodes."""
        if not grounded_language_enabled():
            return {"ok": False, "error": "grounded_language_disabled"}
        text = str(command_text or "").strip()
        if not text:
            return {"ok": False, "error": "empty"}
        self._ensure_grounded_language()
        gl = getattr(self, "_grounded_lang", None)
        if gl is None:
            return {"ok": False, "error": "controller_unavailable"}
        result = gl.ingest_command(self.agent.graph, text)
        if result.get("ok"):
            tag = str(result.get("tag", ""))
            if tag == "recover":
                s2 = getattr(self, "_system2", None)
                if s2 is not None and hasattr(s2, "working_memory"):
                    s2.working_memory.write(
                        "active_macro",
                        1.0,
                        text="RECOVER_POSTURE",
                        tick=int(self.tick),
                        source="grounded_lang",
                    )
        return result

    def grounded_lang_generate(self, obs: dict[str, float] | None = None) -> str:
        """Graph-grounded utterance for verbal layer."""
        self._ensure_grounded_language()
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
        if gl is None:
            return {"enabled": True, "initialized": False}
        return gl.snapshot()

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
