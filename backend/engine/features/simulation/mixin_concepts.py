"""Simulation mixin: концепты, память (save/load), мета-снимок."""
from __future__ import annotations

from engine.features.simulation.mixin_imports import *


class SimulationConceptsMixin:
    def _annotate_concepts_with_graph_nodes(self) -> None:
        for c in self._concepts_cache:
            did = str(c.get("id", ""))
            gn = None
            for nid, meta in self.agent.graph._concept_meta.items():
                if str(meta.get("detector_id", "")) == did:
                    gn = nid
                    break
            c["graph_node"] = gn

    def _maybe_materialize_concept_macros(self, concepts: list[dict]) -> None:
        try:
            max_n = int(os.environ.get("RKK_CONCEPT_MACRO_MAX", "3"))
        except ValueError:
            max_n = 3
        if max_n <= 0:
            return
        try:
            dplat = int(os.environ.get("RKK_CONCEPT_DISCOVERY_PLATEAU_TICKS", "0"))
        except ValueError:
            dplat = 0
        if dplat > 0 and self._discovery_plateau_count < dplat:
            return
        try:
            amin_edge = float(os.environ.get("RKK_CONCEPT_EDGE_ALPHA_MIN", "0"))
        except ValueError:
            amin_edge = 0.0
        n_macro = sum(1 for x in self.agent.graph._node_ids if str(x).startswith("concept_"))
        g = self.agent.graph
        for c in concepts:
            if n_macro >= max_n:
                break
            did = str(c.get("id", ""))
            if not did or did in self._materialized_detector_concept_ids:
                continue
            members = list(c.get("pattern_nodes_example") or [])
            if not members:
                continue
            if amin_edge > 0.0 and g._core is not None:
                if g.path_min_alpha_trust_on_path(members) < amin_edge:
                    continue
            if did.startswith("c") and did[1:].isdigit():
                macro = f"concept_{int(did[1:])}"
            else:
                macro = f"concept_{n_macro}"
            if macro in g.nodes:
                self._materialized_detector_concept_ids.add(did)
                continue
            ok = g.materialize_concept_macro(
                macro,
                members,
                detector_id=did,
                pattern=list(c.get("pattern") or []),
            )
            if ok:
                self._materialized_detector_concept_ids.add(did)
                n_macro += 1
                self._add_event(
                    f"🧩 Macro-node {macro} ← {len(members)} vars (detector {did})",
                    "#88aaff",
                    "phase",
                )

    def _maybe_refresh_concepts_cache(self) -> None:
        try:
            every = int(os.environ.get("RKK_CONCEPT_EVERY", "24"))
        except ValueError:
            every = 24
        if every <= 0 or self.tick % every != 0:
            return
        from engine.concept_detector import detect_proto_concepts

        try:
            pt = int(os.environ.get("RKK_CONCEPT_PLATEAU_TICKS", "0"))
        except ValueError:
            pt = 0
        self._concepts_cache = detect_proto_concepts(
            self.agent.graph,
            agent_plateau_counter=self.agent._rsi_plateau_count,
            plateau_ticks_required=pt,
        )
        self._maybe_materialize_concept_macros(self._concepts_cache)
        self._annotate_concepts_with_graph_nodes()

    def _init_persistence(self, rkk_path: str | None = None) -> None:
        """Init PersistenceManager (same path as .rkk)."""
        if not _PHASE_K_AVAILABLE:
            return
        from pathlib import Path

        from engine.persistence import default_memory_path

        resolved = (
            str(Path(rkk_path).resolve())
            if rkk_path
            else str(Path(default_memory_path()).resolve())
        )
        if self._persist is not None and getattr(self._persist, "_rkk_path", "") == resolved:
            return
        self._persist = PersistenceManager(resolved)

    def _try_restore_meta(self) -> None:
        """Restore autosave.meta.json after .rkk load."""
        if not _PHASE_K_AVAILABLE or self._persist is None:
            return
        if self._meta_restored:
            return
        meta = self._persist.try_load()
        if meta is not None:
            restore_meta_to_simulation(self, meta)
        self._meta_restored = True

    def _maybe_autosave_memory(self) -> None:
        from pathlib import Path

        from engine.persistence import autosave_every_ticks, default_memory_path, save_simulation

        n = autosave_every_ticks()
        if n <= 0 or self.tick <= 0 or self.tick % n != 0:
            return
        try:
            path = default_memory_path()
            save_simulation(self, path)
            if _PHASE_K_AVAILABLE:
                self._init_persistence(str(Path(path).resolve()))
                if self._persist is not None:
                    self._persist.save(collect_meta_from_simulation(self))
        except Exception as e:
            print(f"[RKK] memory autosave: {e}")

    def memory_save(self, path: str | None = None) -> dict:
        from pathlib import Path

        from engine.persistence import default_memory_path, save_simulation

        with self._sim_step_lock:
            pth = Path(path) if path else default_memory_path()
            out = save_simulation(self, pth)
            if out.get("ok") and _PHASE_K_AVAILABLE:
                self._init_persistence(str(pth.resolve()))
                if self._persist is not None:
                    self._persist.save(collect_meta_from_simulation(self))
            return out

    def memory_load(self, path: str | None = None) -> dict:
        from pathlib import Path

        from engine.persistence import default_memory_path, load_simulation

        p = Path(path) if path else default_memory_path()
        target_world = None
        if p.is_file():
            try:
                payload = torch.load(p, map_location="cpu", weights_only=False)
                if isinstance(payload, dict):
                    cw = payload.get("current_world")
                    if isinstance(cw, str) and cw in WORLDS:
                        target_world = cw
            except Exception:
                target_world = None

        if target_world and target_world != self.current_world:
            sw = self.switch_world(target_world)
            if sw.get("error"):
                return {
                    "ok": False,
                    "error": f"failed to switch world to {target_world!r} before load: {sw.get('error')}",
                }

        with self._sim_step_lock:
            # Worker-safe load: останавливаем L4 воркер и чистим его очереди,
            # чтобы не применить устаревшие концепты после миграции графа.
            self._stop_l4_worker()
            self._l4_last_snapshot = {"n_concepts": 0, "concepts": []}
            self._l4_last_submit_tick = 0
            self._l4_last_apply_tick = 0
            self._l3_next_due_ts = 0.0
            self._motor_state = MotorState()
            self._clear_fall_recovery()
            self._drain_simple_queue(self._l1_motor_q)
            self._l1_last_cmd_tick = 0
            self._l1_last_apply_tick = 0
            out = load_simulation(self, p)
            if out.get("ok"):
                self._annotate_concepts_with_graph_nodes()
                self._ensure_phase2()
                try:
                    auto_fr = int(os.environ.get("RKK_AUTO_FIXED_ROOT_TICKS", "0"))
                except ValueError:
                    auto_fr = 0
                if (
                    auto_fr > 0
                    and self.current_world == "humanoid"
                    and self.tick >= auto_fr
                ):
                    self._curriculum_auto_fr_released = True
                    if self._fixed_root_active:
                        self.disable_fixed_root()
                self._meta_restored = False
                self._init_persistence(str(p.resolve()))
                self._try_restore_meta()
            return out

    def concepts_list_payload(self) -> dict:
        phase2 = (
            dict(self._l4_last_snapshot)
            if _l4_worker_enabled()
            else (
                self._concept_store.snapshot()
                if self._concept_store is not None
                else {"n_concepts": 0, "concepts": []}
            )
        )
        return {
            "concepts": list(self._concepts_cache),  # legacy / Phase 1
            "concept_store": phase2,                 # Phase 2 Part 3
            "phase2_concepts": list(phase2.get("concepts", [])),
        }

    def concept_subgraph_payload(self, cid: str) -> dict:
        from engine.concept_detector import concept_by_id

        c = concept_by_id(self._concepts_cache, cid)
        if c is None:
            return {"ok": False, "error": f"unknown concept {cid!r}"}
        nodes: list[str] = []
        for e in c.get("edges", []):
            for k in ("from_", "to"):
                if k in e and e[k] not in nodes:
                    nodes.append(e[k])
        out = {k: v for k, v in c.items()}
        out["ok"] = True
        out["nodes"] = nodes
        gn = c.get("graph_node")
        if gn and gn in self.agent.graph.nodes:
            out["graph_node_value"] = round(float(self.agent.graph.nodes[gn]), 4)
        return out

    def _memory_snapshot_meta(self) -> dict:
        try:
            from engine.persistence import autosave_every_ticks, default_memory_path

            return {
                "autosave_every": autosave_every_ticks(),
                "default_path": str(default_memory_path().resolve()),
            }
        except Exception:
            return {"autosave_every": 0, "default_path": ""}

