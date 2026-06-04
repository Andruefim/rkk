"""Unified neurogenesis scheduling: request during tick, apply only when safe."""
from __future__ import annotations

from engine.core.world import is_humanoid_topology

import os
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from engine.agent import RKKAgent

from engine.rsi_structural import NeurogenesisEngine


def _env_int(key: str, default: int) -> int:
    try:
        return max(1, int(os.environ.get(key, str(default))))
    except ValueError:
        return default


def _env_flag(key: str, default: bool = False) -> bool:
    raw = os.environ.get(key)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


class NeurogenesisCoordinator:
    def __init__(self, engine: NeurogenesisEngine | None = None):
        self.engine = engine or NeurogenesisEngine(
            min_interventions=_env_int("RKK_NEURO_MIN_INTERVENTIONS", 1500),
            error_threshold=float(os.environ.get("RKK_INTRINSIC_DISCOVERY_EIG", "0.3") or 0.3),
        )
        self.engine._cooldown = _env_int("RKK_NEURO_COOLDOWN", 2000)
        self._step3_entry_tick: int = -1

    def note_step3_entry(self, tick: int) -> None:
        if self._step3_entry_tick < 0:
            self._step3_entry_tick = int(tick)

    def _in_step3_grace(self, sim: Any, tick: int) -> bool:
        grace = _env_int("RKK_NEURO_STEP3_GRACE", 400)
        if self._step3_entry_tick < 0:
            return False
        return (int(tick) - self._step3_entry_tick) < grace

    def _motor_blocks_growth(self, sim: Any) -> bool:
        s2 = getattr(sim, "_system2_last", None) or {}
        owner = str(s2.get("motor_owner", "") or "")
        if owner == "s2_scripted":
            return True
        if bool(s2.get("fallen_override_active")):
            return True
        try:
            if getattr(sim.agent.env, "is_fallen", lambda: False)():
                return True
        except Exception:
            pass
        return False

    def _edge_growth_blocked(self, sim: Any) -> bool:
        hist = getattr(sim, "_edge_delta_hist", None)
        if not hist:
            return False
        try:
            max_delta = _env_int("RKK_MAX_EDGE_DELTA_PER_WINDOW", 200)
            window = _env_int("RKK_EDGE_DELTA_WINDOW", 100)
        except ValueError:
            return False
        recent = list(hist)[-window:]
        if len(recent) < window:
            return False
        return sum(int(x) for x in recent) > max_delta

    def request_growth(
        self,
        sim: Any,
        *,
        tick: int,
        trigger: str = "stress",
        high_error_nodes: list[tuple[str, float]] | None = None,
    ) -> dict | None:
        """Queue growth request; returns pending descriptor or None."""
        if not is_humanoid_topology(sim.current_world):
            return None
        if getattr(sim, "_fixed_root_active", False) is False and self._in_step3_grace(sim, tick):
            return None
        if self._motor_blocks_growth(sim):
            return None
        if self._edge_growth_blocked(sim):
            return None
        if getattr(sim, "_wm_warmup_until", 0) and int(tick) <= int(sim._wm_warmup_until):
            return None

        agent = sim.agent
        obs: dict = {}
        try:
            obs = dict(agent.env.observe())
        except Exception:
            pass

        pending = self.engine.queue_growth(agent, int(tick), obs=obs, trigger=trigger)
        if pending is None and high_error_nodes and len(high_error_nodes) >= 2:
            # VariableDiscovery path: force pair from high-error nodes
            n0, n1 = high_error_nodes[0][0], high_error_nodes[1][0]
            pending = self.engine.queue_growth_pair(agent, int(tick), n0, n1, trigger="discovery")
        return pending

    def apply_if_safe(self, sim: Any, *, tick: int) -> dict | None:
        """Apply pending growth when body/sim state is safe."""
        if self._motor_blocks_growth(sim):
            return None
        if self._edge_growth_blocked(sim):
            return None

        fixed = bool(getattr(sim, "_fixed_root_active", False))
        obs: dict = {}
        try:
            obs = dict(sim.agent.env.observe())
        except Exception:
            pass

        result = self.engine.apply_pending_growth(
            sim.agent,
            int(tick),
            obs=obs,
            fixed_root=fixed,
            stable_ticks_required=_env_int("RKK_NEURO_STABLE_TICKS", 20),
            posture_min=0.80,
        )
        if result is not None:
            warmup = _env_int("RKK_NEURO_WARMUP_TICKS", 50)
            sim._wm_warmup_until = int(tick) + warmup
            sim.agent._wm_warmup_until = sim._wm_warmup_until
            sim._neuro_pending = False
            ens = getattr(sim.agent.graph, "_ensemble", None)
            core = getattr(sim.agent.graph, "_core", None)
            if ens is not None and core is not None:
                try:
                    ens.sync_from_executive(core.W, idx=0)
                    import torch

                    with torch.no_grad():
                        ens.log_weights.mul_(0.85)
                except Exception:
                    pass
        return result

    def request_or_apply(self, sim: Any, *, tick: int) -> dict | None:
        """Try apply first, else attempt new request from stress scan."""
        applied = self.apply_if_safe(sim, tick=tick)
        if applied is not None:
            return applied
        pending = self.engine.scan_and_queue(sim.agent, int(tick))
        if pending is not None:
            sim._neuro_pending = True
            return {"type": "neurogenesis_pending", **pending}
        return None

    def apply_after_sleep(self, sim: Any, *, tick: int) -> dict | None:
        if not _env_flag("RKK_NEURO_APPLY_IN_SLEEP", True):
            return None
        return self.apply_if_safe(sim, tick=tick)
