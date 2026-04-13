"""Simulation mixin: visual grounding, Phase M."""
from __future__ import annotations

from engine.features.simulation.mixin_imports import *


class SimulationVisualGroundingMixin:
    def _maybe_run_visual_grounding(self) -> None:
        """Level 1-B: Update visual-body grounding (slot → joint mapping)."""
        if not _VISUAL_GROUNDING_AVAILABLE or self._visual_grounding_ctrl is None:
            return
        if not self._visual_mode or self._visual_env is None:
            return
        if not self._visual_grounding_ctrl.should_run(self.tick):
            return

        # Get PyBullet state
        base_env = self._base_env_ref
        physics_client, robot_id, link_names = None, None, []
        if base_env is not None:
            physics_client, robot_id, link_names = get_pybullet_state_from_humanoid_env(
                base_env
            )

        result = self._visual_grounding_ctrl.update(
            tick=self.tick,
            visual_env=self._visual_env,
            agent_graph=self.agent.graph,
            physics_client=physics_client,
            robot_id=robot_id,
            link_names=link_names,
        )

        if result.get("ok") and result.get("edges_injected", 0) > 0:
            slot_map = result.get("slot_to_joint", {})
            if slot_map:
                mapping_str = ", ".join(
                    f"{k}→{v}" for k, v in list(slot_map.items())[:4]
                )
                self._add_event(
                    f"👁 Grounding: +{result['edges_injected']} edges [{mapping_str}]",
                    "#44ffcc",
                    "phase",
                )

        # Phase M (PATCH 4): VisualGroundingController.update() может дописать
        # visual_env._slot_lexicon[slot_*] in-place (см. visual_grounding.py).
        # Сразу после этого — SlotLabeler + VisualInnerVoice.
        if result.get("ok") and _PHASE_M_AVAILABLE:
            self._phase_m_sync_from_vision()

    @staticmethod
    def _phase_m_slot_positions_from_visual(visual_env: Any) -> dict[str, tuple[float, float]]:
        """Center-of-mass of each attention mask → normalized [0,1]×[0,1]."""
        attn = getattr(visual_env, "_last_attn", None)
        if attn is None:
            return {}
        try:
            if hasattr(attn, "detach"):
                a = attn.detach().cpu().numpy()
            else:
                a = np.asarray(attn)
        except Exception:
            return {}
        if a.ndim != 3 or a.shape[0] < 1:
            return {}
        K, H, W = int(a.shape[0]), int(a.shape[1]), int(a.shape[2])
        pos: dict[str, tuple[float, float]] = {}
        for k in range(K):
            mk = np.maximum(a[k], 0.0)
            s = float(mk.sum()) + 1e-8
            yy, xx = np.mgrid[0:H, 0:W]
            cx = float((mk * xx).sum() / s / max(W, 1))
            cy = float((mk * yy).sum() / s / max(H, 1))
            pos[f"slot_{k}"] = (cx, cy)
        return pos

    def _phase_m_sync_from_vision(self) -> None:
        """
        Phase M (PATCH 4): VLM lexicon + slot masks/vectors → SlotLabeler → VisualInnerVoice.

        Вызывается из:
          1) vlm_label_slots — сразу после EnvironmentVisual.set_slot_lexicon() (полный VLM batch);
          2) _maybe_run_visual_grounding — после update(), если result['ok']
             (in-place правки _slot_lexicon из engine/visual_grounding.py);
          3) каждый тик агента после _maybe_run_visual_grounding — чтобы обновлять
             позиции/векторы слотов по _last_attn / _last_slot_vecs даже без нового VLM.
        """
        if not _PHASE_M_AVAILABLE or self._slot_labeler is None:
            return
        if not self._visual_mode or self._visual_env is None:
            return
        vis = self._visual_env
        lex = getattr(vis, "_slot_lexicon", None) or {}
        vlm_str: dict[str, str] = {}
        slot_confidences: dict[str, float] = {}
        for sid, entry in lex.items():
            if not isinstance(entry, dict):
                continue
            lab = entry.get("label")
            if lab:
                vlm_str[str(sid)] = str(lab)
            slot_confidences[str(sid)] = float(entry.get("confidence", 0.6))

        slot_positions = self._phase_m_slot_positions_from_visual(vis)
        slot_vectors: dict[str, list[float]] = {}
        vecs = getattr(vis, "_last_slot_vecs", None)
        if vecs is not None and hasattr(vecs, "shape"):
            try:
                v = vecs.detach().cpu().numpy() if hasattr(vecs, "detach") else np.asarray(vecs)
                for k in range(int(vecs.shape[0])):
                    sid = f"slot_{k}"
                    row = v[k].flatten()
                    slot_vectors[sid] = [float(x) for x in row[:96]]
            except Exception:
                pass

        try:
            visual_concepts = self._slot_labeler.process_slots(
                vlm_labels=vlm_str,
                slot_positions=slot_positions,
                slot_vectors=slot_vectors,
                slot_confidences=slot_confidences,
                tick=self.tick,
            )
        except Exception as e:
            print(f"[Simulation] Phase M process_slots: {e}")
            return

        world_desc = self._slot_labeler.get_world_description()
        if self._visual_voice is not None:
            try:
                self._visual_voice.update(
                    visual_concepts=visual_concepts,
                    world_desc=world_desc,
                    graph=self.agent.graph if hasattr(self.agent, "graph") else None,
                    inner_voice_ctrl=self._inner_voice,
                    tick=self.tick,
                )
            except Exception as e:
                print(f"[Simulation] Phase M visual_voice.update: {e}")

