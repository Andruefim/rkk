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
