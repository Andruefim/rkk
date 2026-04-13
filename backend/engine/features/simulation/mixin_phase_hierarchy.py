"""Simulation mixin: фазы снимка, L2/L3/L4, иерархия."""
from __future__ import annotations

from engine.features.simulation.mixin_imports import *


class SimulationPhaseHierarchyMixin:
    def _tick_discovery_plateau(self, dr: float) -> None:
        try:
            eps = float(os.environ.get("RKK_CONCEPT_DR_EPS", "0.0015"))
        except ValueError:
            eps = 0.0015
        ref = self._last_dr_snapshot
        if ref is not None and abs(float(dr) - float(ref)) < eps:
            self._discovery_plateau_count += 1
        else:
            self._discovery_plateau_count = 0
        self._last_dr_snapshot = float(dr)

    def _phase1_snapshot_meta(self) -> dict:
        from engine.local_reflex import snapshot_chains_metadata

        try:
            dpt = int(os.environ.get("RKK_CONCEPT_DISCOVERY_PLATEAU_TICKS", "0"))
        except ValueError:
            dpt = 0
        try:
            amin = float(os.environ.get("RKK_CONCEPT_EDGE_ALPHA_MIN", "0"))
        except ValueError:
            amin = 0.0
        dag_mask_frozen = os.environ.get("RKK_DAG_MASK_FROZEN", "1").strip().lower() not in (
            "0",
            "false",
            "no",
            "off",
        )
        return {
            "read_only_macro_prefix": "concept_",
            "discovery_plateau_ticks": self._discovery_plateau_count,
            "discovery_plateau_required": dpt,
            "concept_edge_alpha_min": amin,
            "urdf_frozen_pairs": len(
                getattr(self.agent.graph, "_frozen_edge_set", set()) or set()
            ),
            "dag_mask_frozen": dag_mask_frozen,
            "local_reflex": snapshot_chains_metadata(list(self.agent.graph._node_ids)),
            "local_reflex_train": getattr(self.agent, "_last_local_reflex_train", None),
        }

    def _phase2_snapshot_meta(self) -> dict:
        hg = self._hierarchical_graph
        base = {
            "hierarchical_graph_env": hierarchical_graph_enabled(),
            "active": hg is not None,
        }
        concept_snapshot = (
            dict(self._l4_last_snapshot)
            if _l4_worker_enabled()
            else (
                self._concept_store.snapshot()
                if self._concept_store is not None
                else {"n_concepts": 0, "concepts": []}
            )
        )
        if hg is None:
            base["concept_store"] = concept_snapshot
            if _l4_worker_enabled():
                base["l4_worker"] = {
                    "enabled": True,
                    "pending": bool(self._l4_task_pending),
                    "last_submit_tick": int(self._l4_last_submit_tick),
                    "last_apply_tick": int(self._l4_last_apply_tick),
                }
            return base
        out = {**base, **hg.snapshot()}
        out["concept_store"] = concept_snapshot
        if _l4_worker_enabled():
            out["l4_worker"] = {
                "enabled": True,
                "pending": bool(self._l4_task_pending),
                "last_submit_tick": int(self._l4_last_submit_tick),
                "last_apply_tick": int(self._l4_last_apply_tick),
            }
        return out

    def _ensure_phase2(self) -> None:
        """Ленивая инициализация компонентов фазы 2 (humanoid)."""
        if self.current_world != "humanoid":
            return
        if hierarchical_graph_enabled():
            if self._hierarchical_graph is None:
                self._hierarchical_graph = HierarchicalGraph(self.agent.graph, self.device)
        cs_en = os.environ.get("RKK_CONCEPT_STORE", "1").strip().lower()
        if cs_en in ("0", "false", "no", "off"):
            self._stop_l4_worker()
            self._concept_store = None
            return
        if not self._visual_mode or self._visual_env is None:
            self._stop_l4_worker()
            return
        if _l4_worker_enabled():
            self._ensure_l4_worker()
            self._concept_store = None
            return
        if self._concept_store is None:
            vids = [
                v for v in self.agent.graph._node_ids
                if not str(v).startswith("slot_") and not str(v).startswith("concept_")
            ]
            self._concept_store = VisualConceptStore(
                n_slots=int(self._visual_env.n_slots),
                variable_ids=vids,
            )

    def _maybe_step_hierarchical_l1(self) -> None:
        if not hierarchical_graph_enabled():
            return
        if self.current_world != "humanoid":
            return
        if self._hierarchical_graph is None:
            return
        if self._visual_mode and self._visual_env is not None:
            base_env = getattr(self._visual_env, "base_env", None)
            if base_env is None:
                return
            raw_obs = dict(base_env.observe())
        else:
            raw_obs = dict(self.agent.env.observe())
        self._hierarchical_graph.step_l1(raw_obs)
        self._hierarchical_graph.inject_l1_virtual_nodes()

    def _l3_planning_due(self) -> bool:
        """
        Разрешение запуска L3 (goal_planning + imagination horizon) в текущем тике.
        Single-writer: только флаг для agent.step, без фоновых мутаций graph/env.
        """
        hz = _l3_loop_hz_from_env()
        if hz <= 0.0:
            self._l3_last_tick = self.tick
            return True
        now = time.perf_counter()
        if now >= self._l3_next_due_ts:
            self._l3_next_due_ts = now + (1.0 / hz)
            self._l3_last_tick = self.tick
            return True
        return False

    def _ensure_l4_worker(self) -> None:
        if self._l4_thread is not None and self._l4_thread.is_alive():
            return
        self._l4_stop.clear()
        self._l4_task_pending = False
        self._l4_thread = threading.Thread(
            target=self._l4_worker_loop,
            daemon=True,
            name="rkk-l4-concepts",
        )
        self._l4_thread.start()
        print("[Simulation] L4 concept worker enabled (single-writer apply)")

    def _stop_l4_worker(self) -> None:
        self._l4_stop.set()
        th = self._l4_thread
        if th is not None and th.is_alive():
            th.join(timeout=1.5)
        self._l4_thread = None
        self._l4_stop.clear()
        self._l4_task_pending = False
        self._drain_simple_queue(self._l4_in_q)
        self._drain_simple_queue(self._l4_out_q)

    @staticmethod
    def _drain_simple_queue(q: queue.SimpleQueue) -> None:
        while True:
            try:
                q.get_nowait()
            except Exception:
                break

    def _enqueue_l4_task(
        self,
        *,
        slot_vecs: torch.Tensor,
        slot_values: list[float],
        variability: list[float],
        phys_obs: dict[str, float],
    ) -> None:
        if self._l4_task_pending:
            return
        payload = {
            "tick": int(self.tick),
            "slot_vecs": slot_vecs.detach().cpu().float().numpy(),
            "slot_values": [float(x) for x in list(slot_values)],
            "variability": [float(x) for x in list(variability)],
            "phys_obs": {str(k): float(v) for k, v in dict(phys_obs).items()},
            "graph_node_ids": list(self.agent.graph._node_ids),
            "n_slots": int(self._visual_env.n_slots) if self._visual_env is not None else 8,
        }
        self._l4_in_q.put(payload)
        self._l4_task_pending = True
        self._l4_last_submit_tick = self.tick

    @staticmethod
    def _serialize_l4_concept(c) -> dict:
        return {
            "cid": str(c.cid),
            "label": c.label,
            "slot_idx": int(c.slot_idx),
            "phys_vars": list(c.phys_vars),
            "corr_scores": {str(k): float(v) for k, v in dict(c.corr_scores).items()},
            "uses": int(c.uses),
            "stable_frames": int(c.stable_frames),
            "created_tick": int(c.created_tick),
        }

    def _l4_worker_loop(self) -> None:
        store: VisualConceptStore | None = None
        while not self._l4_stop.is_set():
            try:
                task = self._l4_in_q.get(timeout=0.05)
            except Exception:
                continue
            try:
                n_slots = int(task.get("n_slots", 8))
                graph_node_ids = list(task.get("graph_node_ids") or [])
                if store is None or store.n_slots != n_slots:
                    vids = [
                        v for v in graph_node_ids
                        if not str(v).startswith("slot_") and not str(v).startswith("concept_")
                    ]
                    store = VisualConceptStore(n_slots=n_slots, variable_ids=vids)
                slot_vecs_np = task.get("slot_vecs")
                slot_vecs = torch.from_numpy(slot_vecs_np).float()
                new_concepts = store.update(
                    slot_vecs=slot_vecs,
                    slot_values=list(task.get("slot_values") or []),
                    variability=list(task.get("variability") or []),
                    phys_obs=dict(task.get("phys_obs") or {}),
                    tick=int(task.get("tick", 0)),
                    graph_node_ids=graph_node_ids,
                )
                self._l4_out_q.put({
                    "tick": int(task.get("tick", 0)),
                    "snapshot": store.snapshot(),
                    "new_concepts": [self._serialize_l4_concept(c) for c in new_concepts],
                })
            except Exception as ex:
                self._l4_out_q.put({"error": str(ex)})

    def _apply_l4_concepts(self, concepts: list[dict]) -> int:
        added = 0
        for c in concepts:
            cid = str(c.get("cid", ""))
            if not cid:
                continue
            node_name = f"concept_{cid[:4]}"
            if node_name in self.agent.graph.nodes:
                continue
            uses = int(c.get("uses", 0))
            val = float(uses / (uses + 10)) if uses >= 0 else 0.0
            self.agent.graph.set_node(node_name, val)
            slot_idx = int(c.get("slot_idx", -1))
            slot_key = f"slot_{slot_idx}"
            if slot_key in self.agent.graph.nodes:
                self.agent.graph.set_edge(slot_key, node_name, 0.15, 0.05)
            corrs = dict(c.get("corr_scores") or {})
            for phys_var, corr in corrs.items():
                if phys_var not in self.agent.graph.nodes:
                    continue
                corr_f = float(corr)
                w = float(np.clip(abs(corr_f) * 0.5, 0.06, 0.4))
                sign = 1.0 if corr_f > 0 else -1.0
                self.agent.graph.set_edge(node_name, phys_var, sign * w, 0.06)
            added += 1
        return added

    def _drain_l4_results(self) -> None:
        while True:
            try:
                msg = self._l4_out_q.get_nowait()
            except Exception:
                break
            self._l4_task_pending = False
            if not isinstance(msg, dict):
                continue
            err = msg.get("error")
            if err:
                print(f"[Simulation] L4 worker: {err}")
                continue
            snap = msg.get("snapshot")
            if isinstance(snap, dict):
                self._l4_last_snapshot = snap
            new_concepts = list(msg.get("new_concepts") or [])
            if new_concepts:
                added = self._apply_l4_concepts(new_concepts)
                c0 = new_concepts[0]
                self._add_event(
                    f"Concept formed: {str(c0.get('cid',''))[:4]}, "
                    f"slot_{int(c0.get('slot_idx', -1))}, +{added} nodes",
                    "#EF9F27",
                    "phase",
                )
                self._l4_last_apply_tick = self.tick

