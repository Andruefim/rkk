"""Simulation mixin: чат, verbal tick."""
from __future__ import annotations

from engine.core.world import is_humanoid_topology

from engine.features.simulation.mixin_imports import *


class SimulationVerbalMixin:
    async def _async_broadcast_chat_payload(self, payload: dict[str, Any]) -> None:
        """Send JSON to all chat WebSocket clients (must run on uvicorn loop)."""
        import json

        data = json.dumps(payload, ensure_ascii=False)
        dead: list[Any] = []
        for ws in list(self._chat_ws_clients):
            try:
                await ws.send_text(data)
            except Exception:
                dead.append(ws)
        for ws in dead:
            try:
                self._chat_ws_clients.remove(ws)
            except ValueError:
                pass

    def _broadcast_agent_message(self, msg: Any) -> None:
        """Callback from VerbalActionController — push to WS clients."""
        loop = getattr(self, "_uvicorn_loop", None)
        if loop is None or not loop.is_running():
            return
        payload = {"event": "agent_message", "data": msg.to_dict()}
        try:
            asyncio.run_coroutine_threadsafe(
                self._async_broadcast_chat_payload(payload), loop
            )
        except Exception:
            pass

    def _hook_grounded_verbal_decoder(self) -> None:
        """OBSERVE text → grounded_language (nomic + Qwen), not CausalSpeechDecoder GRU."""
        if not _VERBAL_AVAILABLE or self._verbal is None:
            return
        try:
            from engine.grounded_language import (
                grounded_language_enabled,
                state_phrase_for_speech,
            )
        except ImportError:
            return
        if not grounded_language_enabled():
            return

        verbal = self._verbal
        from engine.verbal_action import SpeechDecoder

        base_decode = SpeechDecoder().decode_observe

        def grounded_decode(concepts, obs, curiosity):
            try:
                self._ensure_grounded_language()
                o = dict(obs) if obs else {}
                gl = getattr(self, "_grounded_lang", None)
                env = getattr(getattr(self, "agent", None), "env", None)
                fallen_flag: bool | None = None
                try:
                    base_env = getattr(env, "base_env", env) if env is not None else None
                    if base_env is not None and callable(getattr(base_env, "is_fallen", None)):
                        fallen_flag = bool(base_env.is_fallen())
                except Exception:
                    fallen_flag = None
                human_task = ""
                human_task_stage = ""
                tb = getattr(self, "_task_binding", None)
                ht = tb.active_task if tb is not None else None
                if ht is not None and getattr(ht, "status", "") == "active":
                    human_task = str(getattr(ht, "text", "") or "")
                try:
                    from engine.task_executive import active_tree_stage_kind

                    human_task_stage = active_tree_stage_kind(self)
                except Exception:
                    pass
                # During human tasks, speak the grounded stage phrase directly.
                # Qwen paraphrase was inventing «Я потеряюсь» from store nearest.
                if human_task.strip():
                    phrase = state_phrase_for_speech(
                        o,
                        fallen=fallen_flag,
                        env=env,
                        human_task_text=human_task,
                        human_task_stage=human_task_stage,
                    )
                    if phrase and len(phrase.strip()) >= 2:
                        if gl is not None and hasattr(self, "agent"):
                            gl.sync_speak_vector_from_state(
                                self.agent.graph,
                                o,
                                fallen=fallen_flag,
                                env=env,
                                human_task_text=human_task,
                                human_task_stage=human_task_stage,
                            )
                        return phrase.strip()
                if gl is not None and hasattr(self, "agent"):
                    gl.sync_speak_vector_from_state(
                        self.agent.graph,
                        o,
                        fallen=fallen_flag,
                        env=env,
                        human_task_text=human_task,
                        human_task_stage=human_task_stage,
                    )
                text = self.grounded_lang_generate(o)
                if text and len(text.strip()) >= 3:
                    t = text.strip()
                    if (
                        t.rstrip(".") == "Я упал"
                        and fallen_flag is False
                        and gl is not None
                    ):
                        phrase = state_phrase_for_speech(
                            o,
                            fallen=False,
                            env=env,
                            human_task_text=human_task,
                            human_task_stage=human_task_stage,
                        )
                        gl.sync_speak_vector_from_state(
                            self.agent.graph,
                            o,
                            fallen=False,
                            env=env,
                            human_task_text=human_task,
                            human_task_stage=human_task_stage,
                        )
                        text = gl.generate_utterance(self.agent.graph, o)
                        if text and len(text.strip()) >= 3:
                            return text.strip()
                        return phrase
                    return t
            except Exception:
                pass
            return base_decode(concepts, obs, curiosity)

        verbal.decoder.decode_observe = grounded_decode  # type: ignore[method-assign]

    def _schedule_verbal_tick(self, fallen: bool) -> None:
        """Run verbal tick synchronously on the simulation tick thread."""
        if not _VERBAL_AVAILABLE or self._verbal is None:
            return
        if not is_humanoid_topology(self.current_world) or self._inner_voice is None:
            return
        if not speech_enabled():
            return
        try:
            self._tick_verbal(int(self.tick), fallen)
        except Exception as e:
            print(f"[Verbal] tick error: {e}")

    def _tick_verbal(self, tick: int, fallen: bool) -> None:
        if not _VERBAL_AVAILABLE or self._verbal is None:
            return
        if not is_humanoid_topology(self.current_world) or self._inner_voice is None:
            return

        self._hook_grounded_verbal_decoder()

        obs: dict[str, float] = {}
        try:
            obs = dict(self.agent.env.observe())
        except Exception:
            return

        total_falls = (
            getattr(self._episodic_memory, "total_falls_recorded", 0)
            if self._episodic_memory
            else 0
        )
        fall_history_brief = ""
        if self._episodic_memory and self._episodic_memory._patterns:
            fall_history_brief = self._episodic_memory._patterns[0].description[:80]

        intention_brief = ""
        ctx = getattr(self, "_intention_state", None)
        if ctx is not None and getattr(ctx, "narrative", ""):
            intention_brief = str(ctx.narrative)[:120]
        elif getattr(self, "_intention_cortex", None) is not None:
            lines = getattr(self._intention_cortex, "_narrative_lines", [])
            if lines:
                intention_brief = str(lines[-1])[:120]
        if intention_brief:
            fall_history_brief = (
                f"{fall_history_brief} | intention: {intention_brief}".strip(" |")
            )

        msg = self._verbal.tick_sync(
            tick=tick,
            obs=obs,
            inner_voice_ctrl=self._inner_voice,
            fallen=fallen,
            total_falls=total_falls,
            llm_url=get_ollama_generate_url(),
            llm_model=get_ollama_model(),
            fall_history_brief=fall_history_brief,
        )
        if msg is None:
            return
        icon = {"OBSERVE": "💬", "ASK": "❓", "REPORT": "📊"}.get(
            msg.speech_type.name, "💬"
        )
        self._add_event(f"{icon} {msg.text}", "#88ffcc", "speech")

    def handle_human_reply(self, reply_text: str) -> dict[str, Any]:
        """HTTP/WS: human reply to agent ASK (also grounds free-text as command)."""
        cmd = self.handle_human_command(reply_text)
        if not _VERBAL_AVAILABLE or self._verbal is None:
            return {**cmd, "verbal": False}
        reward = float(self._verbal.on_human_reply(reply_text))

        lc = self._locomotion_controller
        if lc is not None and reward > 0 and hasattr(lc, "_reward_history"):
            lc._reward_history.append(reward)

        self._verbal_reward_total = float(
            getattr(self, "_verbal_reward_total", 0.0)
        ) + float(reward)

        loop = getattr(self, "_uvicorn_loop", None)
        if loop is not None and loop.is_running():
            import time

            payload = {
                "event": "human_message",
                "data": {
                    "text": reply_text,
                    "timestamp": time.time(),
                    "reward_given": round(reward, 3),
                },
            }
            try:
                asyncio.run_coroutine_threadsafe(
                    self._async_broadcast_chat_payload(payload), loop
                )
            except Exception:
                pass

        return {"ok": True, "reward": round(reward, 3), "grounded": cmd}
