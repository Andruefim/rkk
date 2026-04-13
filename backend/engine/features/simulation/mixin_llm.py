"""Simulation mixin: LLM loop L2/L3."""
from __future__ import annotations

from engine.features.simulation.mixin_imports import *


class SimulationLlmLoopMixin:
    # ── Этап D: LLM в петле ────────────────────────────────────────────────────
    def _llm_loop_enabled(self) -> bool:
        return os.environ.get("RKK_LLM_LOOP", "").strip().lower() in ("1", "true", "yes", "on")

    def _apply_pending_llm_bundle(self) -> None:
        b = self._pending_llm_bundle
        if not b:
            return
        self._pending_llm_bundle = None
        l2 = b.get("l2") or {}
        if l2.get("ok"):
            edges = l2.get("candidate_edges") or []
            ex = (l2.get("explanation") or "").strip()
            next_probe = l2.get("next_probe") or {}
            if ex or edges or next_probe:
                edge_preview = [
                    f"{e.get('from_')}->{e.get('to')}({float(e.get('weight', 0.0)):+.2f})"
                    for e in list(edges)[:6]
                    if isinstance(e, dict)
                ]
                print(
                    "[LLM L2] "
                    f"explanation={ex or '-'} | "
                    f"next_probe={next_probe or {}} | "
                    f"edges={len(edges)} | "
                    f"preview={edge_preview}"
                )
            if edges:
                inj = self.agent.inject_text_priors(edges)
                n = inj.get("injected", 0)
                self._add_event(f"🧠 LLM L2 (+{n} priors): {ex}", "#9bdcff", "phase")
            self._llm_loop_stats["level2_runs"] = int(self._llm_loop_stats.get("level2_runs", 0)) + 1
            self._llm_loop_stats["last_level2_explanation"] = (l2.get("explanation") or "").strip()
        elif l2.get("error"):
            print(f"[LLM L2] error={l2.get('error')}")
            self._add_event(f"LLM L2 error: {l2.get('error')}", "#cc6666", "phase")

        l3 = b.get("l3")
        if isinstance(l3, dict) and l3.get("ok"):
            edges3 = l3.get("edges") or []
            if edges3:
                edge_preview3 = [
                    f"{e.get('from_')}->{e.get('to')}({float(e.get('weight', 0.0)):+.2f})"
                    for e in list(edges3)[:8]
                    if isinstance(e, dict)
                ]
                print(
                    "[LLM L3] "
                    f"edges={len(edges3)} | "
                    f"preview={edge_preview3}"
                )
            if edges3:
                inj3 = self.agent.inject_text_priors(edges3)
                self._add_event(
                    f"🧬 LLM L3 restructure +{inj3.get('injected', 0)} hypotheses",
                    "#66ddff",
                    "phase",
                )
            self._last_level3_tick = self.tick
            self._llm_loop_stats["level3_runs"] = int(self._llm_loop_stats.get("level3_runs", 0)) + 1
        elif isinstance(l3, dict) and l3.get("error"):
            print(f"[LLM L3] error={l3.get('error')}")

    def _rolling_block_rate(self) -> float:
        if len(self._rolling_block_bits) < 24:
            return float(self.agent.value_layer.block_rate)
        return float(np.mean(self._rolling_block_bits))

    def _prediction_surprise_trigger(self, result: dict) -> bool:
        if result.get("skipped") or result.get("blocked"):
            return False
        pe = float(result.get("prediction_error", 0.0))
        self._pe_history.append(pe)
        if len(self._pe_history) < 40:
            return False
        arr = np.array(self._pe_history, dtype=np.float64)
        mu, sd = float(arr.mean()), float(arr.std())
        if sd < 1e-8:
            return False
        return pe > mu + 3.0 * sd

    def _vlm_unknown_slot_trigger(self) -> bool:
        """Слот с заметной динамикой, но без привязки к phys / подозрительный лейбл."""
        if not self._visual_mode or self._visual_env is None:
            return False
        try:
            vis = self._visual_env.get_slot_visualization()
        except Exception:
            return False
        lex = getattr(self._visual_env, "_slot_lexicon", None) or {}
        varib = list(vis.get("variability") or [])
        n = int(getattr(self._visual_env, "n_slots", 0) or 0)
        bad_labs = frozenset({"?", "unknown", "unlabeled", "", "thing", "object"})
        for i in range(n):
            sk = f"slot_{i}"
            entry = lex.get(sk) or {}
            lab = str(entry.get("label", "")).strip().lower()
            likely = entry.get("likely_phys") or []
            vscore = float(varib[i]) if i < len(varib) else 0.0
            if vscore > 0.32 and not likely:
                return True
            if vscore > 0.25 and lab in bad_labs:
                return True
        return False

    def _should_run_level3(self, triggers: list[str]) -> bool:
        if self.current_world != "humanoid":
            return False
        try:
            interval = int(os.environ.get("RKK_LLM_LEVEL3_INTERVAL", "4200"))
        except ValueError:
            interval = 4200
        if self.tick - self._last_level3_tick < interval:
            return False
        tset = " ".join(triggers)
        return ("stagnation" in tset) or ("block_rate" in tset)

    def _maybe_schedule_llm_loop(self, result: dict, snap: dict) -> None:
        if not self._llm_loop_enabled():
            return
        _inner_voice_active = (
            _INNER_VOICE_AVAILABLE
            and self._inner_voice is not None
            and self._inner_voice.total_inferences > 20
        )
        if _inner_voice_active:
            return
        if self._llm_level2_inflight or self._pending_llm_bundle is not None:
            return
        try:
            cooldown = int(os.environ.get("RKK_LLM_LEVEL2_COOLDOWN", "1200"))
        except ValueError:
            cooldown = 1200
        if self.tick - self._last_level2_schedule_tick < cooldown:
            return

        try:
            stagnation_ticks = int(os.environ.get("RKK_LLM_STAGNATION_TICKS", "1400"))
        except ValueError:
            stagnation_ticks = 1400
        try:
            min_iv = int(os.environ.get("RKK_LLM_MIN_INTERVENTIONS", "96"))
        except ValueError:
            min_iv = 96

        triggers: list[str] = []
        if (
            self.agent._total_interventions >= min_iv
            and (self.tick - self._last_dr_gain_tick) >= stagnation_ticks
        ):
            triggers.append("discovery_stagnation")

        vl = self.agent.value_layer
        if (
            vl.total_checked >= 128
            and len(self.agent.graph.edges) >= 6
            and self._rolling_block_rate() > 0.72
        ):
            triggers.append("block_rate")

        if self._vlm_unknown_slot_trigger():
            triggers.append("vlm_unknown_object")

        if self._prediction_surprise_trigger(result) and self.tick >= max(240, min_iv):
            triggers.append("prediction_surprise_3sigma")

        if not triggers:
            return

        self._last_level2_schedule_tick = self.tick
        self._llm_level2_inflight = True
        self._llm_loop_stats["last_triggers"] = list(triggers)

        from engine.phase3_teacher import _slot_lexicon_summary

        try:
            _ctx_val = float(result.get("value", 0.0))
        except (TypeError, ValueError):
            _ctx_val = 0.0
        try:
            _ctx_pe = float(result.get("prediction_error", 0.0))
        except (TypeError, ValueError):
            _ctx_pe = 0.0
        ctx = {
            "variable_ids": list(self.agent.graph.nodes.keys()),
            "triggers": triggers,
            "variable": result.get("variable"),
            "value": _ctx_val,
            "prediction_error": _ctx_pe,
            "discovery_rate": float(snap.get("discovery_rate", 0.0)),
            "block_rate": float(vl.block_rate),
            "cf_predicted": result.get("cf_predicted") or {},
            "cf_observed": result.get("cf_observed") or {},
            "slot_lexicon": _slot_lexicon_summary(self._visual_env),
            "fall_history": (
                self._episodic_memory.get_llm_context_block(max_falls=5)
                if self._episodic_memory is not None
                else ""
            ),
            "curriculum_stage": (
                self._curriculum.current_stage.name
                if self._curriculum is not None
                else ""
            ),
            "temporal_context": (
                self._timescale.build_llm_temporal_context()
                if self._timescale is not None
                else ""
            ),
            "reward_breakdown": (
                self._reward_coord.snapshot().get("last_signal", {})
                if self._reward_coord is not None
                else {}
            ),
            "proprio_abstracts": (
                self._proprio.snapshot().get("abstracts", {})
                if self._proprio is not None
                else {}
            ),
            "inner_voice_concepts": (
                self._inner_voice.get_concept_str(max_concepts=5)
                if self._inner_voice is not None
                else ""
            ),
            "inner_voice_verbal": (
                (self._llm_teacher.get_latest().verbal or "")
                if self._llm_teacher is not None and self._llm_teacher.get_latest()
                else ""
            ),
            "run_level3": self._should_run_level3(triggers),
            "llm_url": get_ollama_generate_url(),
            "llm_model": get_ollama_model(),
        }

        self._llm_loop_executor.submit(self._llm_bundle_worker, ctx)

    def _llm_bundle_worker(self, ctx: dict) -> None:
        try:
            from engine.llm_loop import consult_counterfactual_sync, structure_revision_sync

            valid = set(ctx.get("variable_ids") or [])
            l2 = consult_counterfactual_sync(
                ctx["llm_url"],
                ctx["llm_model"],
                ctx,
                valid,
            )
            l3 = None
            if ctx.get("run_level3") and l2.get("ok") and ctx.get("variable_ids"):
                l3 = structure_revision_sync(
                    ctx["llm_url"],
                    ctx["llm_model"],
                    list(ctx["variable_ids"]),
                )
                if not (isinstance(l3, dict) and l3.get("ok")):
                    l3 = None
            self._pending_llm_bundle = {"l2": l2, "l3": l3}
        except Exception as e:
            self._pending_llm_bundle = {"l2": {"ok": False, "error": str(e)}, "l3": None}
        finally:
            self._llm_level2_inflight = False
