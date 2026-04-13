"""Simulation mixin: inner voice, LLM teacher."""
from __future__ import annotations

from engine.features.simulation.mixin_imports import *


class SimulationTeacherMixin:
    def _on_teacher_annotation(self, ann: "TeacherAnnotation") -> None:
        """LLM τ3 teacher callback: distill + intents + GNN seeds + UI event."""
        if ann.error or not ann.primary_concepts:
            return

        if self._inner_voice is not None and hasattr(self.agent, "graph"):
            node_ids = list(self.agent.graph._node_ids)
            state_vec = [float(self.agent.graph.nodes.get(n, 0.5)) for n in node_ids]
            if state_vec:
                cs = self._inner_voice.concept_store
                known_names = set(cs.name_to_idx.keys())
                merged: list[str] = list(ann.primary_concepts)
                if _PHASE_M_AVAILABLE and self._visual_voice is not None:
                    for c, _ in self._visual_voice.get_active_concepts()[:3]:
                        if c not in merged:
                            merged.append(c)
                labels_for_distill = [n for n in merged if n in known_names][:5]
                if not labels_for_distill:
                    labels_for_distill = [
                        n for n in ann.primary_concepts if n in known_names
                    ][:5]
                if not labels_for_distill and "STABLE_BALANCE" in known_names:
                    labels_for_distill = ["STABLE_BALANCE"]
                if labels_for_distill:
                    self._inner_voice.push_distill_sample(state_vec, labels_for_distill)
                if len(self._inner_voice._train_buf) >= 8:
                    self._inner_voice.train_step()

        if ann.intent_adjustments and self._timescale is not None:
            for var, val in ann.intent_adjustments.items():
                self._timescale.set_intent(LEVEL_REFLECT, var, float(val))

        if ann.seeds and hasattr(self, "agent"):
            self.agent.inject_text_priors(ann.seeds)

        verbal = ann.verbal[:80] if ann.verbal else ""
        concepts = ", ".join(ann.primary_concepts[:3])
        mode_icon = {"annotate": "💭", "guide": "🧭", "lesson": "📖"}.get(ann.mode, "💭")
        if verbal:
            self._add_event(
                f"{mode_icon} [{concepts}] {verbal}",
                "#aaaaff",
                "inner_voice",
            )

    def _tick_inner_voice(self, tick: int) -> None:
        """InnerVoiceNet τ2 — no LLM; gated by timescale LEVEL_COGNIT."""
        if not _INNER_VOICE_AVAILABLE or self._inner_voice is None:
            return
        if self.current_world != "humanoid":
            return
        if self._timescale is None or not self._timescale.should_run(LEVEL_COGNIT, tick):
            return

        graph = getattr(self.agent, "graph", None)
        if graph is None:
            return

        inf0 = self._inner_voice.total_inferences
        result = self._inner_voice.tick(tick, graph, self.agent.env)
        if self._inner_voice.total_inferences <= inf0:
            return

        self._timescale.mark_ran(LEVEL_COGNIT, tick)

        active = result.get("active_concepts", []) if result else []
        if active:
            top_concept, top_val = active[0]
            fall_concepts = {
                "FALLING_NOW",
                "HIGH_FALL_RISK",
                "FALLEN",
                "JOINT_CRITICAL",
            }
            if top_concept in fall_concepts and top_val > 0.75:
                pass

    def _tick_llm_teacher(self, tick: int) -> None:
        """LLM teacher τ3 — async, non-blocking."""
        if not _INNER_VOICE_AVAILABLE or self._llm_teacher is None:
            return
        if self.current_world != "humanoid":
            return
        if not self._llm_teacher.should_call():
            return
        if self._timescale is not None and not self._timescale.should_run(LEVEL_REFLECT, tick):
            return

        obs: dict = {}
        try:
            obs = dict(self.agent.env.observe())
        except Exception:
            pass

        valid_intents = [k for k in self.agent.graph.nodes if k.startswith("intent_")]
        valid_graph_vars = list(self.agent.graph.nodes.keys())
        total_falls = 0
        if self._episodic_memory is not None:
            total_falls = int(getattr(self._episodic_memory, "total_falls_recorded", 0))

        import asyncio

        async def _call():
            await self._llm_teacher.call_async(
                tick=tick,
                obs=obs,
                inner_voice_controller=self._inner_voice,
                episodic_memory=self._episodic_memory,
                curriculum=self._curriculum,
                llm_url=get_ollama_generate_url(),
                llm_model=get_ollama_model(),
                valid_intents=valid_intents,
                valid_graph_vars=valid_graph_vars,
                total_ticks=tick,
                total_falls=total_falls,
                visual_voice=self._visual_voice if _PHASE_M_AVAILABLE else None,
            )

        def _run_in_thread() -> None:
            asyncio.run(_call())

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        try:
            if loop is not None:
                asyncio.ensure_future(_call())
            else:
                self._llm_loop_executor.submit(_run_in_thread)
            if self._timescale is not None:
                self._timescale.mark_ran(LEVEL_REFLECT, tick)
        except Exception as e:
            print(f"[Simulation] LLM teacher schedule error: {e}")

