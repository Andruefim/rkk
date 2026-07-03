"""
Human command → intentional goal via world-model imagination (no tag/motor tables).

Flow:
  1. Grounded language embed → sensory_audio_semantic_* (hearing)
  2. GNN free-rollout imagines post-command state
  3. expected_state = imagined observe keys → WM + Intention + S2 WM planner PE
  4. Each tick: verify PE vs expected; REPORT when homeostatic veto + PE ok
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from engine.grounded_language import motor_interventions_for_command
from engine.system2.schema import filter_expected_state_raw
from engine.system2.success_predicates import (
    EpisodeSuccessSpec,
    evaluate_macro_success,
    homeostatic_veto,
    prediction_error_total,
    resolve_max_prediction_error,
)


def task_binding_enabled() -> bool:
    return os.environ.get("RKK_TASK_BINDING", "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _env_int(key: str, default: int) -> int:
    try:
        return max(1, int(os.environ.get(key, str(default))))
    except ValueError:
        return default


def _env_float(key: str, default: float) -> float:
    try:
        return float(os.environ.get(key, str(default)))
    except ValueError:
        return default


def task_observation_keys(obs: dict[str, float]) -> list[str]:
    """All numeric observe keys the agent may target (no keyword routing)."""
    keys: list[str] = []
    for k in sorted(obs.keys()):
        sk = str(k)
        if sk.startswith("_"):
            continue
        keys.append(sk)
    cap = _env_int("RKK_TASK_OBS_KEY_CAP", 48)
    return keys[:cap]


@dataclass
class HumanTask:
    text: str
    expected_state: dict[str, float]
    max_prediction_error: float | None
    tick_started: int
    tick_deadline: int
    status: str = "active"  # active | done | failed
    last_pe: float = 1.0
    last_diag: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text[:120],
            "status": self.status,
            "tick_started": self.tick_started,
            "tick_deadline": self.tick_deadline,
            "last_pe": round(float(self.last_pe), 6),
            "n_expected_keys": len(self.expected_state),
            "expected_state": {
                k: round(float(v), 4) for k, v in list(self.expected_state.items())[:16]
            },
            "diag": dict(self.last_diag),
        }


class TaskBindingController:
    """Bind natural-language commands to PE-verifiable expected_state."""

    def __init__(self) -> None:
        self._active: HumanTask | None = None

    @property
    def active_task(self) -> HumanTask | None:
        return self._active if self._active and self._active.status == "active" else None

    def imagine_expected_state(
        self,
        graph: Any,
        obs: dict[str, float],
        *,
        horizon: int | None = None,
        text: str = "",
        motor_interventions: dict[str, float] | None = None,
    ) -> dict[str, float]:
        """
        Counterfactual goal: do(intent_*) from command semantics, then H-step free rollout.
        Sensory language nodes must already be in graph (ingest_command before bind).
        """
        steps = horizon if horizon is not None else _env_int("RKK_TASK_IMAGINE_STEPS", 16)
        motor = dict(motor_interventions or {})
        if not motor and str(text or "").strip():
            motor = motor_interventions_for_command(str(text))

        def _merge_graph(o: dict[str, float]) -> dict[str, float]:
            state = dict(o)
            nodes = getattr(graph, "nodes", None) or {}
            for k, v in nodes.items():
                sk = str(k)
                if sk in state or sk.startswith(
                    ("slot_", "sensory_", "intent_", "posture", "com_", "target", "self_goal")
                ):
                    try:
                        state[sk] = float(v)
                    except (TypeError, ValueError):
                        pass
            return state

        state = _merge_graph(obs)
        propagate = getattr(graph, "propagate_from", None)
        rollout = getattr(graph, "rollout_step_free", None)

        if motor and callable(propagate):
            for var, val in sorted(motor.items()):
                try:
                    state = propagate(state, str(var), float(val))
                except Exception:
                    break

        if not callable(rollout):
            raw = {k: float(v) for k, v in state.items()}
            return filter_expected_state_raw(raw, obs_keys=list(state.keys()))

        for _ in range(steps):
            try:
                state = rollout(state)
            except Exception:
                break

        keys = task_observation_keys(state)
        raw = {k: float(state[k]) for k in keys if k in state}
        return filter_expected_state_raw(raw, obs_keys=keys)

    def bind_command(
        self,
        graph: Any,
        obs: dict[str, float],
        text: str,
        tick: int,
    ) -> HumanTask:
        expected = self.imagine_expected_state(graph, obs, text=text)
        if not expected:
            # Fallback: at least anchor proprio + slots from current (command modulates via graph)
            expected = filter_expected_state_raw(
                {k: float(v) for k, v in obs.items()},
                obs_keys=list(obs.keys()),
            )

        n_keys = len(expected)
        max_pe = resolve_max_prediction_error(
            None,
            n_keys=n_keys,
            macro="EXPLORE",
            skill_id="human_command",
        )
        horizon = _env_int("RKK_TASK_DEADLINE_TICKS", 2400)
        task = HumanTask(
            text=str(text).strip(),
            expected_state=expected,
            max_prediction_error=max_pe,
            tick_started=int(tick),
            tick_deadline=int(tick) + horizon,
            status="active",
        )
        self._active = task
        return task

    def verify(self, obs: dict[str, float], task: HumanTask | None = None) -> tuple[bool, float, dict[str, Any]]:
        t = task or self._active
        if t is None or t.status != "active":
            return False, 1.0, {"reason": "no_active_task"}

        spec = EpisodeSuccessSpec(
            expected_state=dict(t.expected_state),
            max_prediction_error=t.max_prediction_error,
            skill_id="human_command",
        )
        ok, diag = evaluate_macro_success(obs, spec, macro="EXPLORE")
        pe = float(diag.get("pe_total", prediction_error_total(obs, t.expected_state)))
        t.last_pe = pe
        t.last_diag = diag
        return ok, pe, diag

    def tick_verify(
        self,
        obs: dict[str, float],
        tick: int,
        *,
        fallen: bool = False,
    ) -> HumanTask | None:
        """Update task status; returns task if just completed or failed."""
        t = self._active
        if t is None or t.status != "active":
            return None

        if int(tick) > t.tick_deadline:
            t.status = "failed"
            t.last_diag = {"reason": "deadline", "tick": tick}
            return t

        if fallen and _env_float("RKK_TASK_FAIL_ON_FALLEN", 0.0) > 0.5:
            t.status = "failed"
            t.last_diag = {"reason": "fallen"}
            return t

        ok_h, veto_reason = homeostatic_veto(obs)
        if not ok_h and int(tick) - t.tick_started < _env_int("RKK_TASK_HOME0_GRACE", 40):
            return None

        ok, pe, diag = self.verify(obs, t)
        if ok:
            min_ticks = _env_int("RKK_TASK_MIN_TICKS", 60)
            if int(tick) - t.tick_started < min_ticks:
                return None
            t.status = "done"
            t.last_diag = diag
            return t

        if not ok_h and int(tick) - t.tick_started > _env_int("RKK_TASK_HOME0_GRACE", 40):
            # Stuck in bad homeostasis — keep trying until deadline
            diag["homeo_note"] = veto_reason

        return None

    def clear(self) -> None:
        self._active = None

    def snapshot(self) -> dict[str, Any]:
        return {
            "enabled": task_binding_enabled(),
            "active": self._active.to_dict() if self._active else None,
        }
