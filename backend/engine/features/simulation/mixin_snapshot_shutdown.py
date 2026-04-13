"""Simulation mixin: snapshot, public_state, shutdown."""
from __future__ import annotations

from engine.features.simulation.mixin_imports import *
from engine.features.simulation.snapshot import build_simulation_snapshot


class SimulationSnapshotShutdownMixin:
    # ── Snapshot ──────────────────────────────────────────────────────────────
    def _build_snapshot(self, snap: dict, graph_deltas: dict,
                        smoothed: float, scene: dict) -> dict:
        return build_simulation_snapshot(self, snap, graph_deltas, smoothed, scene)

    def public_state(self) -> dict:
        snap     = self._last_snapshot or self.agent.snapshot()
        smoothed = float(np.mean(self._dr_window)) if self._dr_window else 0.0
        fn       = getattr(self.agent.env, "get_full_scene", None)
        scene    = fn() if callable(fn) else {}
        return self._build_snapshot(snap, {}, smoothed, scene)

    def shutdown(self):
        self._bg.stop_rkk_agent_loop()
        self._stop_cpg_background_loop()
        try:
            self._llm_loop_executor.shutdown(wait=False, cancel_futures=False)
        except TypeError:
            self._llm_loop_executor.shutdown(wait=False)
        except Exception:
            pass