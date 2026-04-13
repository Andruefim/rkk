"""Simulation mixin: чат, verbal tick."""
from __future__ import annotations

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

    def _schedule_verbal_tick(self, fallen: bool) -> None:
        """Run async verbal tick in a daemon thread (agent thread has no asyncio loop)."""
        if not _VERBAL_AVAILABLE or self._verbal is None:
            return
        if self.current_world != "humanoid" or self._inner_voice is None:
            return
        if not speech_enabled():
            return
        if self._verbal_tick_running:
            return

        tick_now = int(self.tick)

        def run() -> None:
            self._verbal_tick_running = True
            try:
                asyncio.run(self._tick_verbal(tick_now, fallen))
            except Exception as e:
                print(f"[Verbal] tick error: {e}")
            finally:
                self._verbal_tick_running = False

        threading.Thread(target=run, daemon=True, name="rkk-verbal-tick").start()

    async def _tick_verbal(self, tick: int, fallen: bool) -> None:
        if not _VERBAL_AVAILABLE or self._verbal is None:
            return
        if self.current_world != "humanoid" or self._inner_voice is None:
            return
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

        msg = await self._verbal.tick(
            tick=tick,
            obs=obs,
            inner_voice_ctrl=self._inner_voice,
            fallen=fallen,
            total_falls=total_falls,
            llm_url=get_ollama_generate_url(),
            llm_model=get_ollama_model(),
            fall_history_brief=fall_history_brief,
            visual_voice=self._visual_voice if _PHASE_M_AVAILABLE else None,
        )
        if msg is None:
            return
        icon = {"OBSERVE": "💬", "ASK": "❓", "REPORT": "📊"}.get(
            msg.speech_type.name, "💬"
        )
        self._add_event(f"{icon} {msg.text}", "#88ffcc", "speech")

    def handle_human_reply(self, reply_text: str) -> dict[str, Any]:
        """HTTP/WS: human reply to agent ASK."""
        if not _VERBAL_AVAILABLE or self._verbal is None:
            return {"ok": False, "error": "verbal unavailable"}
        reward = float(self._verbal.on_human_reply(reply_text))

        lc = self._locomotion_controller
        if lc is not None and reward > 0 and hasattr(lc, "_reward_history"):
            lc._reward_history.append(reward)

        if self._reward_coord is not None:
            cur = float(getattr(self._reward_coord, "_total_verbal_reward", 0.0))
            setattr(self._reward_coord, "_total_verbal_reward", cur + reward)

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

        return {"ok": True, "reward": round(reward, 3)}
