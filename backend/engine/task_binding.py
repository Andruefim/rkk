"""
Human command → intentional goal via world-model imagination (no tag/motor tables).

Flow:
  1. goal_grounding: text → TaskGoal predicates (embedding similarity)
  2. Anchored WM rollout imagines post-command state (do() interventions)
  3. expected_state narrowed to goal-relevant keys
  4. Verify: predicate satisfaction when WM untrusted; PE on goal keys when trusted
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from engine.goal_grounding import goal_observation_keys, ground_command
from engine.goal_interventions import interventions_for_goal
from engine.grounded_language import FallbackEmbeddingClient, motor_interventions_for_command
from engine.success_predicates import (
    evaluate_goal,
    evaluate_macro_success,
    expected_state_keys_for_goal,
    homeostatic_veto,
    prediction_error_total,
    resolve_max_prediction_error,
)
from engine.system2.schema import EpisodeSuccessSpec, filter_expected_state_raw
from engine.task_goal import TaskGoal
from engine.task_observation import (
    CONTACT_SIGNAL,
    GRASP_CONTACT,
    TASK_CONTACT,
    TASK_TARGET_DIST,
)


def task_binding_enabled() -> bool:
    return os.environ.get("RKK_TASK_BINDING", "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def task_protect_embodiment_enabled() -> bool:
    """When on, active human tasks defer hard pose reset and fixed_root re-attach."""
    return os.environ.get("RKK_TASK_PROTECT_EMBODIMENT", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


def human_task_execution_active(sim: Any) -> bool:
    """True while a human command is bound and executing (tree or flat binding)."""
    if not task_binding_enabled():
        return False
    try:
        from engine.task_tree import task_tree_enabled

        if task_tree_enabled():
            tt = getattr(sim, "_task_tree_ctrl", None)
            if tt is not None and bool(getattr(tt, "is_active", False)):
                return True
    except Exception:
        pass
    tb = getattr(sim, "_task_binding", None)
    ht = tb.active_task if tb is not None else None
    return ht is not None and str(getattr(ht, "status", "active")) == "active"


def human_task_embodiment_protected(sim: Any) -> bool:
    """Embodiment curriculum / fall reset must not interrupt an in-flight human task."""
    return task_protect_embodiment_enabled() and human_task_execution_active(sim)


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


def _merge_graph_obs(graph: Any, obs: dict[str, float]) -> dict[str, float]:
    """Anchor rollout start state to current observations + graph nodes."""
    state = {k: float(v) for k, v in obs.items()}
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


def _is_degenerate(values: list[float], *, eps: float = 0.008) -> bool:
    if len(values) < 2:
        return False
    return (max(values) - min(values)) < eps


def _assess_wm_trust(
    obs: dict[str, float],
    anchored: dict[str, float],
    imagined: dict[str, float],
    keys: list[str],
) -> tuple[bool, dict[str, Any]]:
    """WM trust gate: anchored state must match obs; imagined must not be degenerate."""
    thresh = _env_float("RKK_WM_TRUST_PE", 0.35)
    diag: dict[str, Any] = {"wm_trust_pe_thresh": thresh}
    check_keys = [k for k in keys if k in obs or k in anchored]
    if not check_keys:
        check_keys = [k for k in list(obs.keys())[:12] if k in anchored]

    errs0: list[float] = []
    for k in check_keys:
        if k in obs and k in anchored:
            errs0.append(abs(float(obs[k]) - float(anchored[k])))
    mean0 = sum(errs0) / len(errs0) if errs0 else 0.0
    diag["anchor_pe_mean"] = round(mean0, 4)

    img_vals = [float(imagined[k]) for k in check_keys if k in imagined]
    diag["degenerate"] = _is_degenerate(img_vals)
    if img_vals:
        diag["imagined_spread"] = round(max(img_vals) - min(img_vals), 4)

    trusted = mean0 <= thresh and not diag["degenerate"]
    if not trusted:
        if mean0 > thresh:
            diag["wm_trust_reason"] = "anchor_mismatch"
        elif diag["degenerate"]:
            diag["wm_trust_reason"] = "degenerate_imagination"
    return trusted, diag


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
    goal: TaskGoal | None = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
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
        if self.goal is not None:
            out["goal"] = self.goal.to_dict()
        return out


class TaskBindingController:
    """Bind natural-language commands to PE-verifiable expected_state."""

    def __init__(self) -> None:
        self._active: HumanTask | None = None
        self._fallback_embedder = FallbackEmbeddingClient()

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
        interventions: dict[str, float] | None = None,
        motor_interventions: dict[str, float] | None = None,
        goal: TaskGoal | None = None,
        agent_xy: tuple[float, float] | None = None,
        target_xy: tuple[float, float] | None = None,
        agent_forward: tuple[float, float] | None = None,
    ) -> tuple[dict[str, float], bool, dict[str, Any]]:
        """
        Counterfactual goal: anchor to current obs, apply do(intent_*), H-step rollout.

        Returns (expected_state, wm_trusted, trust_diag).
        """
        steps = horizon if horizon is not None else _env_int("RKK_TASK_IMAGINE_STEPS", 16)
        motor = dict(interventions or motor_interventions or {})
        if not motor:
            if goal is not None and goal.predicates:
                motor = interventions_for_goal(
                    goal,
                    agent_xy=agent_xy,
                    target_xy=target_xy,
                    agent_forward=agent_forward,
                )
            elif str(text or "").strip():
                motor = motor_interventions_for_command(str(text))

        state = _merge_graph_obs(graph, obs)
        anchored = dict(state)
        propagate = getattr(graph, "propagate_from", None)
        rollout = getattr(graph, "rollout_step_free", None)

        if motor and callable(propagate):
            for var, val in sorted(motor.items()):
                try:
                    state = propagate(state, str(var), float(val))
                except Exception:
                    break

        state_after_intervention = dict(state)

        if callable(rollout):
            state_one = state
            try:
                state_one = rollout(dict(state))
            except Exception:
                state_one = state
            for _ in range(steps):
                try:
                    state = rollout(state)
                except Exception:
                    break
        else:
            state_one = state

        goal_keys = goal_observation_keys(goal)
        if goal is not None:
            goal_keys = list(dict.fromkeys(goal_keys + expected_state_keys_for_goal(goal)))

        if goal is not None and goal_keys:
            raw = {k: float(state[k]) for k in goal_keys if k in state}
        else:
            keys = task_observation_keys(state)
            raw = {k: float(state[k]) for k in keys if k in state}

        expected = filter_expected_state_raw(raw, obs_keys=list(state.keys()))

        trust_keys = goal_keys if goal_keys else list(expected.keys())[:16]
        wm_trusted, trust_diag = _assess_wm_trust(obs, anchored, expected, trust_keys)

        # Step-0/1 sanity: intervention rollout should not drift far from obs on anchor keys.
        step1_errs: list[float] = []
        for k in trust_keys:
            if k in obs and k in state_one:
                step1_errs.append(abs(float(obs[k]) - float(state_one[k])))
        if step1_errs:
            mean1 = sum(step1_errs) / len(step1_errs)
            trust_diag["step1_pe_mean"] = round(mean1, 4)
            if mean1 > _env_float("RKK_WM_TRUST_PE", 0.35):
                wm_trusted = False
                trust_diag["wm_trust_reason"] = "step1_drift"

        trust_diag["n_expected_keys"] = len(expected)
        trust_diag["interventions"] = sorted(motor.keys())
        trust_diag["anchored_keys"] = len(anchored)
        return expected, wm_trusted, trust_diag

    def bind_command(
        self,
        graph: Any,
        obs: dict[str, float],
        text: str,
        tick: int,
        *,
        embed_fn: Any | None = None,
        goal: TaskGoal | None = None,
        agent_xy: tuple[float, float] | None = None,
        target_xy: tuple[float, float] | None = None,
        agent_forward: tuple[float, float] | None = None,
    ) -> HumanTask:
        ef = embed_fn
        if ef is None:
            ef = self._fallback_embedder.embed

        if goal is None:
            goal = ground_command(str(text).strip(), ef)
        expected, wm_trusted, trust_diag = self.imagine_expected_state(
            graph,
            obs,
            text=text,
            goal=goal,
            agent_xy=agent_xy,
            target_xy=target_xy,
            agent_forward=agent_forward,
        )
        goal.wm_trusted = wm_trusted
        goal.diagnostics.update(trust_diag)

        if not expected:
            fallback_keys = goal_observation_keys(goal) or list(obs.keys())[:16]
            expected = filter_expected_state_raw(
                {k: float(obs[k]) for k in fallback_keys if k in obs},
                obs_keys=list(obs.keys()),
            )
        # Metric predicates: seed target values when rollout never saw task_* keys
        # (mock sims / early bind before OWM injects task_target_dist_m).
        if not expected and goal is not None and goal.predicates:
            seeded: dict[str, float] = {}
            for p in goal.predicates:
                kind = str(p.kind)
                if kind == "reduce_distance":
                    seeded[TASK_TARGET_DIST] = float(p.target_value)
                elif kind == "contact":
                    seeded[TASK_CONTACT] = float(p.target_value)
                    seeded[CONTACT_SIGNAL] = float(p.target_value)
                    seeded[GRASP_CONTACT] = float(p.target_value)
                elif kind == "state_key" and p.key:
                    seeded[str(p.key)] = float(p.target_value)
            if seeded:
                expected = filter_expected_state_raw(
                    seeded,
                    obs_keys=list(obs.keys()) + list(seeded.keys()),
                )

        n_keys = len(expected) if expected else max(1, len(goal.predicates))
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
            goal=goal,
        )
        self._active = task
        return task

    def verify(
        self,
        obs: dict[str, float],
        task: HumanTask | None = None,
        *,
        ctx: Any = None,
    ) -> tuple[bool, float, dict[str, Any]]:
        t = task or self._active
        if t is None or t.status != "active":
            return False, 1.0, {"reason": "no_active_task"}

        goal = t.goal
        if goal is not None and not goal.wm_trusted:
            ok, score, diag = evaluate_goal(goal, obs, ctx)
            t.last_pe = 1.0 - float(score)
            t.last_diag = diag
            return ok, t.last_pe, diag

        pe_keys = expected_state_keys_for_goal(goal) if goal else list(t.expected_state.keys())
        narrowed_expected = (
            {k: t.expected_state[k] for k in pe_keys if k in t.expected_state}
            if pe_keys
            else dict(t.expected_state)
        )
        if not narrowed_expected and t.expected_state:
            narrowed_expected = dict(t.expected_state)

        spec = EpisodeSuccessSpec(
            expected_state=narrowed_expected,
            max_prediction_error=t.max_prediction_error,
            skill_id="human_command",
        )
        ok, diag = evaluate_macro_success(obs, spec, macro="EXPLORE")
        pe = float(diag.get("pe_total", prediction_error_total(obs, narrowed_expected)))
        t.last_pe = pe
        if goal is not None:
            diag["wm_trusted"] = bool(goal.wm_trusted)
            diag["goal_score_path"] = "pe"
        t.last_diag = diag
        return ok, pe, diag

    def tick_verify(
        self,
        obs: dict[str, float],
        tick: int,
        *,
        fallen: bool = False,
        ctx: Any = None,
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

        ok, pe, diag = self.verify(obs, t, ctx=ctx)
        if ok:
            min_ticks = _env_int("RKK_TASK_MIN_TICKS", 60)
            if int(tick) - t.tick_started < min_ticks:
                return None
            t.status = "done"
            t.last_diag = diag
            return t

        if not ok_h and int(tick) - t.tick_started > _env_int("RKK_TASK_HOME0_GRACE", 40):
            diag["homeo_note"] = veto_reason

        return None

    def clear(self) -> None:
        self._active = None

    def snapshot(self) -> dict[str, Any]:
        active = self._active.to_dict() if self._active else None
        goal_summary = None
        if self._active and self._active.goal is not None:
            g = self._active.goal
            goal_summary = {
                "confidence": g.confidence,
                "wm_trusted": g.wm_trusted,
                "n_predicates": len(g.predicates),
                "kinds": [p.kind for p in g.predicates],
                "needs_target": bool(g.diagnostics.get("needs_target")),
            }
        return {
            "enabled": task_binding_enabled(),
            "active": active,
            "goal": goal_summary,
        }
