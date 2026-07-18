"""Simulation mixin: camera and vision API."""
from __future__ import annotations

from engine.features.simulation.mixin_imports import *


class SimulationApiMixin:
    # ── Camera / Scene ────────────────────────────────────────────────────────
    def get_camera_frame(self, view: str | None = None) -> str | None:
        """PyBullet render is expensive — serve cached JPEG between UI polls."""
        key = (view or "default").strip() or "default"
        cache: dict = getattr(self, "_camera_frame_cache", None) or {}
        if not hasattr(self, "_camera_frame_cache"):
            self._camera_frame_cache = cache
        try:
            min_sec = float(os.environ.get("RKK_CAMERA_MIN_INTERVAL_SEC", "1.0"))
        except ValueError:
            min_sec = 1.0
        min_sec = max(0.25, min(5.0, min_sec))
        now = time.monotonic()
        hit = cache.get(key)
        if hit is not None and (now - float(hit[0])) < min_sec:
            return hit[1]
        fn = getattr(self.agent.env, "get_frame_base64", None)
        frame = None
        if callable(fn):
            with self._sim_step_lock:
                frame = fn(view)
        cache[key] = (now, frame)
        return frame

    def get_vision_state(self) -> dict:
        """Данные для /vision/slots endpoint (свежий снимок слотов)."""
        if not self._visual_mode or self._visual_env is None:
            return {"visual_mode": False}
        try:
            min_sec = float(os.environ.get("RKK_VISION_UI_MIN_INTERVAL_SEC", "1.25"))
        except ValueError:
            min_sec = 1.25
        min_sec = max(0.3, min(5.0, min_sec))
        now = time.monotonic()
        last_at = float(getattr(self, "_vision_ui_served_at", 0.0) or 0.0)
        last_state = getattr(self, "_last_vision_state", None)
        if last_state and (now - last_at) < min_sec:
            return dict(last_state)
        try:
            state = self._visual_env.get_slot_visualization()
        except Exception:
            state = dict(last_state or {})
        state["visual_mode"] = True
        state["n_slots"] = self._visual_env.n_slots
        state["vision_ticks"] = self._vision_ticks
        state["cortex"] = self._visual_env.cortex.snapshot()
        self._last_vision_state = state
        self._vision_ui_served_at = now
        return state

    def get_vision_overlay(self) -> dict:
        """
        Camera HUD overlay from LatentSceneMemory (+ visual target fallback).
        Agent-centric only — no privileged registry XY.
        """
        tick = int(getattr(self, "tick", 0) or 0)
        out: dict = {
            "tick": tick,
            "n_entities": 0,
            "n_active": 0,
            "active": None,
            "entities": [],
            "stage": None,
        }
        try:
            scene_fn = getattr(self, "_latent_scene_memory", None)
            cam_fn = getattr(self, "_depth_camera_from_sim", None)
            lock = getattr(self, "_sim_step_lock", None)
            if callable(scene_fn) and callable(cam_fn):
                if lock is not None:
                    with lock:
                        scene = scene_fn()
                        cam = cam_fn()
                        if cam is not None and scene.active_ids:
                            scene.refresh_active_from_live_camera(
                                cam, tick=tick, blend=0.85
                            )
                        payload = scene.overlay_payload(tick=tick)
                else:
                    scene = scene_fn()
                    cam = cam_fn()
                    if cam is not None and scene.active_ids:
                        scene.refresh_active_from_live_camera(
                            cam, tick=tick, blend=0.85
                        )
                    payload = scene.overlay_payload(tick=tick)
                out.update(payload)
            elif callable(scene_fn):
                scene = scene_fn()
                payload = scene.overlay_payload(tick=tick)
                out.update(payload)
        except Exception:
            pass

        # Fallback: bound visual target if scene has no active focus yet
        if out.get("active") is None:
            vt = getattr(self, "_manip_resolved_visual", None)
            if vt is not None and getattr(vt, "range_m", None) is not None:
                try:
                    from engine.vision_resolve import hud_safe_label

                    hud_label = hud_safe_label(
                            str(vt.label or ""), fallback=str(vt.slot_id or "target")
                        )
                    diags = getattr(vt, "diagnostics", None) or {}
                    conf = float(vt.confidence)
                    if vt.range_conf is not None:
                        conf = min(conf, float(vt.range_conf))
                    if diags.get("geometry") == "objectness_peak":
                        pstr = float(diags.get("objectness_peak_strength") or 0.0)
                        conf = min(conf, 0.15 + 0.85 * pstr)
                    out["active"] = {
                        "id": str(vt.slot_id),
                        "slot_id": str(vt.slot_id),
                        "label": hud_label,
                        "u": float(max(0.0, min(1.0, vt.u))),
                        "v": float(max(0.0, min(1.0, vt.v))),
                        "range_m": round(float(vt.range_m), 2),
                        "bearing": round(float(vt.bearing), 3),
                        "conf": round(float(conf), 3),
                        "holding": False,
                    }
                    out["n_active"] = 1
                except Exception:
                    pass

        # Task stage hint for HUD
        try:
            tt = getattr(self, "_task_tree_ctrl", None)
            if tt is not None and getattr(tt, "is_active", False):
                node = getattr(tt, "active_node", None)
                progress = 0.0
                try:
                    snap = tt.snapshot(tick)
                    progress = float(snap.get("progress") or 0.0)
                except Exception:
                    progress = 0.0
                if node is not None:
                    out["stage"] = {
                        "kind": str(getattr(node, "kind", "") or ""),
                        "label": str(getattr(node, "label", "") or "")[:32],
                        "progress": progress,
                    }
        except Exception:
            pass
        return out
