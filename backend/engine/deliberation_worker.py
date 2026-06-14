"""
GPU/async deliberation worker — deep WM imagination off the hot tick path.

Biology: default-mode / slow System-2 thinking (seconds-scale) while motor loop runs.
Main tick reads cached DeliberationResult; worker never blocks PyBullet.
"""
from __future__ import annotations

import copy
import os
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from engine.goal_planning import (
    beam_search_first_action,
    imagination_steps_default,
    parse_plan_value_levels,
    plan_beam_k,
    plan_branch_per_beam,
    plan_depth,
    planning_graph_motor_vars,
    subsample_actions,
)
from engine.graph_constants import is_read_only_macro_var
from engine.wm_neural_ode import integrate_world_model_step


def deliberation_enabled() -> bool:
    return os.environ.get("RKK_DELIBERATION_ENABLED", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


def deliberation_device() -> torch.device:
    raw = os.environ.get("RKK_DELIBERATION_DEVICE", "").strip().lower()
    if not raw:
        raw = os.environ.get("RKK_DEVICE", "cpu").strip().lower()
    use_gpu = os.environ.get("RKK_DELIBERATION_GPU", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )
    if use_gpu and raw in ("cuda", "gpu") and torch.cuda.is_available():
        return torch.device("cuda")
    if use_gpu and torch.cuda.is_available() and raw == "cpu":
        return torch.device("cuda")
    return torch.device("cpu")


def _ei(key: str, default: int) -> int:
    try:
        return max(1, int(os.environ.get(key, str(default))))
    except ValueError:
        return default


def _eb(key: str, default: bool = True) -> bool:
    raw = os.environ.get(key, "1" if default else "0").strip().lower()
    return raw not in ("0", "false", "no", "off")


def deliberation_intent_only() -> bool:
    return _eb("RKK_DELIBERATION_INTENT_ONLY", True)


def deliberation_plan_value_levels() -> list[float]:
    raw = os.environ.get("RKK_DELIBERATION_PLAN_VALUES", "0.42,0.62")
    levels: list[float] = []
    for part in raw.split(","):
        p = part.strip()
        if not p:
            continue
        try:
            levels.append(float(np.clip(float(p), 0.06, 0.94)))
        except ValueError:
            continue
    return levels if levels else [0.42, 0.62]


@dataclass
class DeliberationResult:
    tick: int = -1
    macro_hint: str = "IDLE"
    first_action: tuple[str, float] | None = None
    score: float = 0.0
    expected_state: dict[str, float] = field(default_factory=dict)
    graph_patch: dict[str, float] = field(default_factory=dict)
    intent_residuals: dict[str, float] = field(default_factory=dict)
    narrative: str = ""
    plan_depth: int = 0
    imagination_horizon: int = 0
    device: str = "cpu"
    latency_ms: float = 0.0
    stale: bool = True

    def to_dict(self) -> dict[str, Any]:
        fa = None
        if self.first_action is not None:
            fa = {"variable": self.first_action[0], "value": round(self.first_action[1], 4)}
        return {
            "tick": self.tick,
            "macro_hint": self.macro_hint,
            "first_action": fa,
            "score": round(self.score, 5),
            "expected_state": {k: round(float(v), 4) for k, v in self.expected_state.items()},
            "graph_patch": {k: round(float(v), 4) for k, v in self.graph_patch.items()},
            "intent_residuals": {
                k: round(float(v), 4) for k, v in self.intent_residuals.items()
            },
            "narrative": self.narrative,
            "plan_depth": self.plan_depth,
            "imagination_horizon": self.imagination_horizon,
            "device": self.device,
            "latency_ms": round(self.latency_ms, 2),
            "stale": self.stale,
        }


class DeliberationGraphView:
    """Read-only WM forward on deliberation device (CUDA mirror of GNN core)."""

    __slots__ = ("_node_ids", "_d", "MAX_D", "device", "_core")

    def __init__(self, source: Any, device: torch.device) -> None:
        self._node_ids = list(getattr(source, "_node_ids", []) or [])
        self._d = int(getattr(source, "_d", 0) or 0)
        self.MAX_D = int(getattr(source, "MAX_D", 256))
        self.device = device
        self._core: Any = None

    def _pad_to(self, x: torch.Tensor, target_d: int) -> torch.Tensor:
        pad_len = target_d - x.shape[-1]
        if pad_len > 0:
            import torch.nn.functional as F

            return F.pad(x, (0, pad_len))
        return x

    def _wm_inputs_for_core(
        self, X: torch.Tensor, A: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self._core is None:
            return X, A
        cd = int(getattr(self._core, "d", self.MAX_D))
        if cd > X.shape[-1]:
            return self._pad_to(X, cd), self._pad_to(A, cd)
        if cd < X.shape[-1]:
            return X[..., :cd], A[..., :cd]
        return X, A

    def forward_dynamics(self, X: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        if self._core is None:
            return X
        X_in, a_in = self._wm_inputs_for_core(X, a)
        if hasattr(self._core, "forward_dynamics"):
            pred = self._core.forward_dynamics(X_in, a_in)
        else:
            import torch.nn.functional as F

            m = (torch.abs(a_in) > 1e-8).float()
            x_in = X_in * (1.0 - m) + a_in * m
            pred = self._core(x_in)
        return pred[..., : self._d]

    def sync_from(self, source: Any) -> bool:
        core = getattr(source, "_core", None)
        if core is None:
            return False
        with torch.inference_mode():
            if self._core is None:
                self._core = copy.deepcopy(core).to(self.device)
            else:
                self._core.load_state_dict(core.state_dict())
            self._core.eval()
        return True

    def _state_matrix_from_dicts(self, bases: list[dict[str, float]]) -> torch.Tensor:
        rows = [[float(b.get(nid, 0.0)) for nid in self._node_ids] for b in bases]
        return torch.tensor(rows, dtype=torch.float32, device=self.device)

    def _dicts_from_pred_batch(self, pred: torch.Tensor) -> list[dict[str, float]]:
        nids = self._node_ids
        d_out = pred.shape[1]
        out: list[dict[str, float]] = []
        for i in range(pred.shape[0]):
            out.append({nids[j]: float(pred[i, j].item()) for j in range(min(d_out, len(nids)))})
        return out

    def propagate_from_batch(
        self,
        base: dict[str, float],
        interventions: list[tuple[str, float]],
    ) -> list[dict[str, float]]:
        if not interventions or self._core is None:
            return [dict(base) for _ in interventions]
        n = len(interventions)
        state_vec = self._state_matrix_from_dicts([base] * n)
        a_vec = torch.zeros(n, self._d, dtype=torch.float32, device=self.device)
        nid_to_i = {nid: i for i, nid in enumerate(self._node_ids)}
        for i, (variable, value) in enumerate(interventions):
            if is_read_only_macro_var(variable):
                continue
            if variable in nid_to_i:
                a_vec[i, nid_to_i[variable]] = float(value)
        with torch.inference_mode():
            pred = integrate_world_model_step(self, state_vec, a_vec)
        return self._dicts_from_pred_batch(pred)

    def propagate_from_multi_batch(
        self,
        bases: list[dict[str, float]],
        interventions: list[tuple[str, float]],
    ) -> list[dict[str, float]]:
        if not bases or not interventions or len(bases) != len(interventions):
            return []
        if self._core is None:
            return [dict(b) for b in bases]
        state_vec = self._state_matrix_from_dicts(bases)
        n = len(interventions)
        a_vec = torch.zeros(n, self._d, dtype=torch.float32, device=self.device)
        nid_to_i = {nid: i for i, nid in enumerate(self._node_ids)}
        for i, (variable, value) in enumerate(interventions):
            if is_read_only_macro_var(variable):
                continue
            if variable in nid_to_i:
                a_vec[i, nid_to_i[variable]] = float(value)
        with torch.inference_mode():
            pred = integrate_world_model_step(self, state_vec, a_vec)
        return self._dicts_from_pred_batch(pred)

    def rollout_step_free_batch(
        self, bases: list[dict[str, float]]
    ) -> list[dict[str, float]]:
        if not bases or self._core is None:
            return [dict(b) for b in bases]
        state_vec = self._state_matrix_from_dicts(bases)
        z = torch.zeros_like(state_vec)
        with torch.inference_mode():
            pred = integrate_world_model_step(self, state_vec, z)
        return self._dicts_from_pred_batch(pred)


class _DelibAgentProxy:
    def __init__(self, graph_view: DeliberationGraphView, horizon: int) -> None:
        self.graph = graph_view
        self._horizon = horizon

    def _batch_rollout_imagination_states(
        self,
        base: dict[str, float],
        actions: list[tuple[str, float]],
        *,
        row_bases: list[dict[str, float]] | None = None,
        horizon: int | None = None,
    ) -> list[dict[str, float]]:
        h = self._horizon if horizon is None else max(0, int(horizon))
        if not actions:
            return []
        if row_bases is None:
            states = self.graph.propagate_from_batch(dict(base), actions)
        else:
            states = self.graph.propagate_from_multi_batch(row_bases, actions)
        for _ in range(h):
            states = self.graph.rollout_step_free_batch(states)
        return states


@dataclass
class _DelibJob:
    tick: int
    state0: dict[str, float]
    macro: str
    expected_state: dict[str, float]
    goal_target_dist: float | None
    primary_var: str
    graph_ref: Any


class DeliberationService:
    """Background deliberation queue + result cache."""

    def __init__(self, sim: Any) -> None:
        self._sim = sim
        self._queue: queue.Queue[_DelibJob | None] = queue.Queue(maxsize=2)
        self._result = DeliberationResult()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._device = deliberation_device()
        self._graph_view = DeliberationGraphView(sim.agent.graph, self._device)
        self._last_request_tick = -10**9
        self._last_delib_macro = ""
        self._plan_cache_key: tuple[str, float] | None = None
        self._coalesced: _DelibJob | None = None
        self._coalesce_lock = threading.Lock()
        self._busy = False

    def ensure_started(self) -> None:
        if not deliberation_enabled():
            return
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._worker_loop,
            daemon=True,
            name="rkk-deliberation-gpu",
        )
        self._thread.start()
        print(
            f"[Deliberation] Worker started on {self._device} "
            f"(RKK_DELIBERATION_GPU / RKK_DELIBERATION_DEVICE)"
        )

    def stop(self) -> None:
        self._stop.set()
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass
        th = self._thread
        if th is not None and th.is_alive():
            th.join(timeout=2.0)
        self._thread = None
        self._stop.clear()

    def latest(self, *, max_age_ticks: int | None = None) -> DeliberationResult | None:
        with self._lock:
            r = self._result
        if r.tick < 0 or r.stale:
            return None
        if max_age_ticks is not None:
            tick = int(getattr(self._sim, "tick", 0))
            if tick - r.tick > max_age_ticks:
                return None
        return r

    def request_if_due(self, *, tick: int, macro: str, intention_ctx: Any | None) -> bool:
        if not deliberation_enabled():
            return False
        sim = self._sim
        pe_fwd = float(getattr(sim, "_hai_pe_fwd_ema", 0.0))
        last_macro = str(getattr(self, "_last_delib_macro", "") or "")
        macro_changed = last_macro != str(macro)
        pe_spike = pe_fwd < -0.6
        if not pe_spike and not macro_changed:
            return False
        plan_key = (str(macro), round(pe_fwd, 2))
        if getattr(self, "_plan_cache_key", None) == plan_key and self.latest(max_age_ticks=120):
            return False
        every = _ei("RKK_DELIBERATION_EVERY", 60)
        if macro.upper() in ("LOCOMOTE_DELIVERY", "EXPLORE"):
            every = max(every, _ei("RKK_DELIB_LOCOMOTE_EVERY", 90))
        if tick - self._last_request_tick < every:
            return False
        self._last_request_tick = tick
        self._last_delib_macro = str(macro)
        self._plan_cache_key = plan_key
        return self.enqueue(tick=tick, macro=macro, intention_ctx=intention_ctx)

    def enqueue(
        self,
        *,
        tick: int,
        macro: str,
        intention_ctx: Any | None,
    ) -> bool:
        if not deliberation_enabled():
            return False
        sim = self._sim
        agent = sim.agent
        with sim._sim_step_lock:
            state0 = dict(agent.graph.nodes)
            graph_ref = agent.graph
        primary_var = ""
        expected: dict[str, float] = {}
        goal_td = None
        if intention_ctx is not None:
            expected = dict(getattr(intention_ctx, "expected_state", None) or {})
            primary = getattr(intention_ctx, "primary", None)
            if primary is not None:
                primary_var = str(getattr(primary, "var_id", ""))
                if primary_var == "target_dist":
                    goal_td = float(getattr(primary, "target_val", 0.42))
        job = _DelibJob(
            tick=tick,
            state0=state0,
            macro=macro,
            expected_state=expected,
            goal_target_dist=goal_td,
            primary_var=primary_var,
            graph_ref=graph_ref,
        )
        with self._coalesce_lock:
            self._coalesced = job
        try:
            while not self._queue.empty():
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    break
            self._queue.put_nowait(job)
            return True
        except queue.Full:
            return True

    def _take_job(self) -> _DelibJob | None:
        with self._coalesce_lock:
            if self._coalesced is not None:
                job = self._coalesced
                self._coalesced = None
                return job
        try:
            job = self._queue.get(timeout=0.25)
        except queue.Empty:
            return None
        if job is None:
            self._stop.set()
            return None
        with self._coalesce_lock:
            if self._coalesced is not None:
                newer = self._coalesced
                self._coalesced = None
                return newer
        return job

    def _worker_loop(self) -> None:
        while not self._stop.is_set():
            job = self._take_job()
            if job is None:
                continue
            self._busy = True
            try:
                result = self._run_job(job)
                with self._lock:
                    self._result = result
            except Exception as ex:
                print(f"[Deliberation] Worker error: {ex}")
            finally:
                self._busy = False

    def _run_job(self, job: _DelibJob) -> DeliberationResult:
        t0 = time.perf_counter()
        if not self._graph_view.sync_from(job.graph_ref):
            return DeliberationResult(tick=job.tick, stale=True)

        depth = _ei("RKK_DELIBERATION_PLAN_DEPTH", 3)
        beam_k = min(plan_beam_k(), _ei("RKK_DELIBERATION_BEAM", 4))
        horizon = _ei(
            "RKK_DELIBERATION_IMAGINATION_STEPS",
            max(6, min(12, imagination_steps_default())),
        )
        proxy = _DelibAgentProxy(self._graph_view, horizon)

        motor = planning_graph_motor_vars(
            self._sim.agent.env, list(self._graph_view._node_ids)
        )
        if deliberation_intent_only():
            intent_motor = [
                v
                for v in motor
                if v.startswith("intent_") or v.startswith("phys_intent_")
            ]
            if intent_motor:
                motor = intent_motor
        if not motor:
            return DeliberationResult(tick=job.tick, stale=True)

        levels = deliberation_plan_value_levels()
        actions = [(v, x) for v in motor for x in levels]
        max_b = _ei("RKK_DELIBERATION_MAX_BRANCH", 32)
        if len(actions) > max_b:
            actions = subsample_actions(actions, max_b)

        from engine.system2.wm_planner import S2WmTask, score_wm_trajectory

        task = S2WmTask(
            macro=job.macro,
            expected_state=dict(job.expected_state),
            goal_target_dist=float(
                job.goal_target_dist
                if job.goal_target_dist is not None
                else job.state0.get("self_goal_target_dist", 0.42)
            ),
            self_goal_active=float(job.state0.get("self_goal_active", 0.0)),
        )

        def _score(_s0, var, val, sfin):
            sc = score_wm_trajectory(
                job.state0, sfin, task, action_var=var, action_val=val
            )
            try:
                from engine.neuro_symbolic.engine import symbolic_engine_enabled
                from engine.neuro_symbolic.predicates import ground_humanoid_state
                from engine.neuro_symbolic.engine import SymbolicCognitiveEngine

                if symbolic_engine_enabled():
                    eng = getattr(self._sim, "_ns_engine", None)
                    if eng is None:
                        eng = SymbolicCognitiveEngine()
                    veto = eng.veto_prediction(
                        sfin,
                        self._sim.agent.env,
                        fuzzy_state=ground_humanoid_state(sfin),
                    )
                    if not veto.allowed:
                        sc -= float(veto.penalty)
            except Exception:
                pass
            return sc

        best, sc = beam_search_first_action(
            proxy,
            state0=job.state0,
            actions=actions,
            depth=depth,
            beam_k=beam_k,
            rollout_horizon=horizon,
            score_fn=_score,
            maximize=True,
        )

        graph_patch: dict[str, float] = {}
        residuals: dict[str, float] = {}
        if job.goal_target_dist is not None:
            graph_patch["self_goal_target_dist"] = float(job.goal_target_dist)
            graph_patch["self_goal_active"] = 0.9
        if best is not None:
            var, val = best
            if var.startswith("intent_"):
                residuals[var] = (float(val) - 0.5) * 0.3

        narrative = (
            f"delib[{job.macro}] depth={depth} h={horizon} "
            f"action={best[0] if best else '—'} score={sc:.3f}"
        )
        ms = (time.perf_counter() - t0) * 1000.0
        try:
            from engine.tick_profiler import get_tick_profiler

            get_tick_profiler().record("bg.deliberation", ms)
        except Exception:
            pass

        return DeliberationResult(
            tick=job.tick,
            macro_hint=job.macro,
            first_action=best,
            score=float(sc),
            expected_state=dict(job.expected_state),
            graph_patch=graph_patch,
            intent_residuals=residuals,
            narrative=narrative,
            plan_depth=depth,
            imagination_horizon=horizon,
            device=str(self._device),
            latency_ms=ms,
            stale=False,
        )
