"""Simulation mixin: snapshot, public_state, shutdown."""
from __future__ import annotations

import os
import time

from engine.core.constants import agent_loop_hz_from_env
from engine.features.simulation.mixin_imports import *
from engine.features.simulation.snapshot import build_simulation_snapshot


class SimulationSnapshotShutdownMixin:
    # ── Snapshot ──────────────────────────────────────────────────────────────
    def _build_snapshot(self, snap: dict, graph_deltas: dict,
                        smoothed: float, scene: dict) -> dict:
        return build_simulation_snapshot(self, snap, graph_deltas, smoothed, scene)

    def public_state(self, *, force: bool = False) -> dict:
        """
        UI/HTTP снимок. При RKK_AGENT_LOOP_HZ>0 отдаём кэш фонового тика (без второго
        agent.snapshot + get_full_scene). TTL — чтобы параллельные GET /api/snapshot
        не копили очередь на lock.
        """
        try:
            ttl_ms = int(os.environ.get("RKK_PUBLIC_STATE_CACHE_MS", "400"))
        except ValueError:
            ttl_ms = 400
        ttl_ms = max(0, min(5000, ttl_ms))
        now = time.monotonic()
        if not force and ttl_ms > 0:
            cached = getattr(self, "_public_state_cache", None)
            at = float(getattr(self, "_public_state_cache_at", 0.0) or 0.0)
            if cached is not None and (now - at) * 1000.0 < ttl_ms:
                return cached

        if not force and agent_loop_hz_from_env() > 0:
            with self._sim_step_lock:
                ac = self._agent_step_response
            if ac is not None:
                self._public_state_cache = ac
                self._public_state_cache_at = now
                return ac

        with self._sim_step_lock:
            snap = self._last_snapshot or self.agent.snapshot()
            smoothed = float(np.mean(self._dr_window)) if self._dr_window else 0.0
            scene = dict(getattr(self, "_cached_scene", {}) or {})
            if not scene:
                fn = getattr(self.agent.env, "get_full_scene", None)
                scene = fn() if callable(fn) else {}
            out = self._build_snapshot(snap, {}, smoothed, scene)
        if ttl_ms > 0:
            self._public_state_cache = out
            self._public_state_cache_at = now
        return out

    def shutdown(self):
        self._bg.stop_rkk_agent_loop()
        self._stop_cpg_background_loop()
        try:
            self._llm_loop_executor.shutdown(wait=False, cancel_futures=False)
        except TypeError:
            self._llm_loop_executor.shutdown(wait=False)
        except Exception:
            pass