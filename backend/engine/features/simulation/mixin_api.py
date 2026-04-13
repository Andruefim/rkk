"""Simulation mixin: camera, vision API, VLM, Phase3 teacher."""
from __future__ import annotations

from engine.features.simulation.mixin_imports import *


class SimulationApiMixin:
    # ── Camera / Scene ────────────────────────────────────────────────────────
    def get_camera_frame(self, view: str | None = None) -> str | None:
        fn = getattr(self.agent.env, "get_frame_base64", None)
        return fn(view) if callable(fn) else None

    def get_vision_state(self) -> dict:
        """Данные для /vision/slots endpoint (свежий снимок, в т.ч. slot_labels Фазы 2)."""
        if not self._visual_mode or self._visual_env is None:
            return {"visual_mode": False}
        try:
            state = self._visual_env.get_slot_visualization()
        except Exception:
            state = dict(self._last_vision_state)
        state["visual_mode"] = True
        state["n_slots"] = self._visual_env.n_slots
        state["vision_ticks"] = self._vision_ticks
        state["cortex"] = self._visual_env.cortex.snapshot()
        return state

    async def vlm_label_slots(
        self,
        llm_url: str,
        llm_model: str,
        max_mask_images: int = 4,
        text_only: bool = False,
        inject_weak_edges: bool = False,
    ) -> dict:
        """
        Фаза 2: один вызов VLM (или текстовый fallback) → лексикон на EnvironmentVisual.
        Опционально слабые рёбра slot→phys при inject_weak_edges и confidence.

        Идея на будущее (без авто-повтора здесь): вызывать из tick_step при «запутался»
        — например серия fallen, высокий block_rate, длительная стагнация discovery;
        с дебаунсом и env RKK_VLM_ON_CONFUSION=1.
        """
        if not self._visual_mode or self._visual_env is None:
            return {"ok": False, "error": "visual mode off"}

        from engine.slot_lexicon import (
            run_slot_vlm_labeling,
            weak_slot_to_phys_edges,
        )

        vis = self._visual_env
        if vis._last_slots is None or vis._cached_frame_b64 is None:
            vis._refresh(run_encode=True)
            if vis._cached_frame_b64 is None:
                return {
                    "ok": False,
                    "error": "camera frame not available yet — retry after a few ticks",
                }

        snap = vis.get_slot_visualization()
        var_ids = list(self.agent.graph.nodes.keys())

        labels, mode, err = await run_slot_vlm_labeling(
            frame_b64=snap.get("frame"),
            masks_b64=list(snap.get("masks") or []),
            slot_values=list(snap.get("slot_values") or []),
            variability=list(snap.get("variability") or []),
            n_slots=vis.n_slots,
            variable_ids=var_ids,
            llm_url=llm_url,
            llm_model=llm_model,
            max_mask_images=max_mask_images,
            text_only=text_only,
        )

        if not labels:
            return {
                "ok": False,
                "mode": mode,
                "error": err or "empty labels",
            }

        vis.set_slot_lexicon(labels, self.tick, snap.get("frame"))

        # Phase M (PATCH 4): единственное место с полным set_slot_lexicon() из VLM API
        if _PHASE_M_AVAILABLE:
            self._phase_m_sync_from_vision()

        injected = 0
        skipped: list[str] = []
        if inject_weak_edges:
            edges = weak_slot_to_phys_edges(labels)
            if edges:
                with self._sim_step_lock:
                    r = self.agent.inject_text_priors(edges)
                    injected = int(r.get("injected", 0))
                    skipped = list(r.get("skipped") or [])

        self._add_event(
            f"🔬 VLM slots: {mode}, {len(labels)} labels"
            + (f", +{injected} weak edges" if inject_weak_edges else ""),
            "#44ccff",
            "phase",
        )

        return {
            "ok": True,
            "mode": mode,
            "n_slots_labeled": len(labels),
            "warning": err,
            "slot_lexicon_tick": self.tick,
            "weak_edges_injected": injected,
            "weak_edges_skipped": skipped,
        }

    async def refresh_phase3_teacher_llm(self) -> dict:
        """
        Фаза 3: один вызов Ollama → правила IG-бонуса для System1 + TTL-дельты Value Layer.
        """
        from engine.phase3_teacher import (
            fetch_phase3_teacher_bundle,
            build_phase3_digest,
            top_uncertain_vars_from_agent,
            _slot_lexicon_summary,
        )

        llm_url = get_ollama_generate_url()
        model = get_ollama_model()
        agent = self.agent
        valid = set(agent.graph.nodes.keys())
        if not valid:
            return {"ok": False, "error": "no graph nodes"}

        fallen = False
        is_fn = getattr(agent.env, "is_fallen", None)
        if callable(is_fn) and not self._fixed_root_active:
            try:
                fallen = bool(is_fn())
            except Exception:
                fallen = False

        pn = (
            PHASE_NAMES[self.phase]
            if 0 <= self.phase < len(PHASE_NAMES)
            else ""
        )
        digest = build_phase3_digest(
            variable_ids=sorted(valid),
            nodes=dict(agent.graph.nodes),
            phase_idx=self.phase,
            phase_name=pn,
            fallen=fallen,
            block_rate=agent.value_layer.block_rate,
            total_interventions=agent._total_interventions,
            top_uncertain_vars=top_uncertain_vars_from_agent(agent),
            slot_lexicon=_slot_lexicon_summary(self._visual_env),
        )

        rules, ov, err, teacher_insight = await fetch_phase3_teacher_bundle(
            llm_url=llm_url,
            llm_model=model,
            digest=digest,
            valid_vars=valid,
            current_tick=self.tick,
        )

        if err and not rules and ov is None:
            return {"ok": False, "error": err, "insight": teacher_insight or ""}

        try:
            tmax = int(os.environ.get("RKK_TEACHER_T_MAX", "140"))
        except ValueError:
            tmax = 140
        tw = max(0.0, 1.0 - (agent._total_interventions / max(1, tmax)))
        with self._sim_step_lock:
            self._phase3_teacher_rules = rules
            self._phase3_vl_overlay = ov
            agent.set_teacher_state(rules, tw)
            if ov is not None and self.tick <= ov.expires_at_tick and tw > 0:
                agent.value_layer.set_teacher_vl_overlay(ov)
            else:
                agent.value_layer.set_teacher_vl_overlay(None)

            msg = f"📚 Phase3 teacher: {len(rules)} rules"
            if ov is not None:
                msg += f", VL overlay ttl={ov.expires_at_tick - self.tick}t"
            if teacher_insight:
                msg += f" — {teacher_insight[:220]}{'…' if len(teacher_insight) > 220 else ''}"
            self._add_event(msg, "#ddaa44", "phase")

        print("[Phase3 Teacher] ────────────────────────────────────────────────")
        if teacher_insight:
            print(f"[Phase3 Teacher] Insight:\n{teacher_insight}\n")
        else:
            print("[Phase3 Teacher] (no insight string in LLM JSON — check model output)\n")
        for i, r in enumerate(rules):
            if r.when_var:
                cond = f"when {r.when_var}∈[{r.when_min},{r.when_max}]"
            else:
                cond = "always"
            print(
                f"[Phase3 Teacher] IG rule {i + 1}: target={r.target_var} "
                f"{cond} bonus={r.bonus:.3f}"
            )
        if ov is not None:
            print(
                "[Phase3 Teacher] VL overlay deltas: "
                f"φ_minΔ={ov.phi_min_delta:+.4f} "
                f"entropy_maxΔ={ov.env_entropy_max_delta:+.4f} "
                f"h_slowΔ={ov.h_slow_max_delta:+.4f} "
                f"pred_loΔ={ov.predict_lo_delta:+.4f} pred_hiΔ={ov.predict_hi_delta:+.4f}"
            )
        if err:
            print(f"[Phase3 Teacher] warning: {err}")
        print("[Phase3 Teacher] ────────────────────────────────────────────────")

        return {
            "ok": True,
            "n_rules": len(rules),
            "vl_overlay": ov is not None,
            "warning": err,
            "insight": teacher_insight or "",
            "expires_at_tick": ov.expires_at_tick if ov is not None else None,
        }

