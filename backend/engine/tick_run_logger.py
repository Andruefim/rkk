"""
Structured per-tick run log for post-hoc LLM analysis (JSON Lines).

RKK_TICK_RUN_LOG=1              — enable (default off)
RKK_TICK_RUN_LOG_PATH=logs/rkk_run.jsonl  — relative to repo root; truncated each run
RKK_TICK_RUN_LOG_EVERY=1        — log every N engine ticks
RKK_TICK_RUN_LOG_PHYS=1         — include humanoid posture/com/feet + motor intents
RKK_TICK_RUN_LOG_EVENTS=1       — include causal-stream events for this tick
RKK_TICK_RUN_LOG_NOTEAR=1       — include last train_step loss dict when present
RKK_TICK_RUN_LOG_SYSTEM2=1     — include sim._system2_last snapshot per tick (default on)
"""
from __future__ import annotations

import json
import os
import statistics
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from engine.core.constants import cpg_during_fixed_root_enabled

_REPO_ROOT = Path(__file__).resolve().parents[2]

# Rolling inner tick duration (ms) for slowdown detection in log
_INNER_MS_WINDOW: deque[float] = deque(maxlen=120)

_LOGGER: "TickRunLogger | None" = None


def tick_run_log_enabled() -> bool:
    return os.environ.get("RKK_TICK_RUN_LOG", "0").strip().lower() in (
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


def _env_flag(key: str, default: bool = True) -> bool:
    raw = os.environ.get(key)
    if raw is None:
        return default
    return raw.strip().lower() not in ("0", "false", "no", "off")


# Subset of System2Controller._last_diag — JSON-safe, bounded size for jsonl.
_TICK_LOG_SYSTEM2_KEYS: frozenset[str] = frozenset(
    {
        "enabled",
        "macro",
        "source",
        "until",
        "idle",
        "sim_tick",
        "macro_horizon_expired",
        "macro_outcome_deferred",
        "llm_inflight",
        "llm_submit_tick",
        "llm_submitted",
        "llm_wait_ticks",
        "has_candidate",
        "last_candidate_var",
        "residuals_applied",
        "last_neuro_node",
        "blocked",
        "skipped",
        "error",
        "outcome_ema",
        "student_conf",
        "last_source",
        "online_buf",
        "neuro_streak",
    }
)


def _system2_tick_log_snapshot(sim: Any) -> dict[str, Any] | None:
    """Per-tick System2 diagnostics (separate from action.from_system2)."""
    if not _env_flag("RKK_TICK_RUN_LOG_SYSTEM2", True):
        return None
    raw = getattr(sim, "_system2_last", None)
    if raw is None:
        return None
    if not isinstance(raw, dict):
        return {"parse_error": "not_a_dict"}
    out: dict[str, Any] = {}
    for k in _TICK_LOG_SYSTEM2_KEYS:
        if k not in raw:
            continue
        v = raw[k]
        if v is None or isinstance(v, (bool, int, str)):
            out[k] = v
        elif isinstance(v, float):
            out[k] = round(v, 5)
        elif k == "neuro_streak" and isinstance(v, dict):
            items = list(v.items())[:48]
            out[k] = {str(a): round(float(b), 4) for a, b in items}
    return out or {"enabled": bool(raw.get("enabled", False))}


def _log_path() -> Path:
    raw = os.environ.get("RKK_TICK_RUN_LOG_PATH", "logs/rkk_run.jsonl").strip()
    p = Path(raw)
    if not p.is_absolute():
        p = _REPO_ROOT / p
    return p


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def _obs_key(obs: dict, *keys: str) -> float | None:
    for k in keys:
        if k in obs:
            return _safe_float(obs[k], 0.5)
    return None


def _unwrap_base_env(env: Any) -> Any:
    e = env
    for _ in range(8):
        nxt = getattr(e, "base_env", None)
        if nxt is None or nxt is e:
            break
        e = nxt
    return e


def _phys_from_sim_raw(base: Any) -> dict[str, float]:
    """com_* / torso в логе даже в fixed_root (observe их не включает)."""
    sim_obj = getattr(base, "_sim", None)
    norm = getattr(base, "_norm", None)
    if sim_obj is None or not callable(norm):
        return {}
    try:
        raw = sim_obj.get_state()
        if not isinstance(raw, dict):
            return {}
    except Exception:
        return {}
    out: dict[str, float] = {}
    for k in ("com_x", "com_y", "com_z", "torso_pitch", "torso_roll"):
        if k not in raw:
            continue
        try:
            out[k] = round(float(norm(k, raw[k])), 4)
        except (TypeError, ValueError):
            continue
    return out


class TickRunLogger:
    def __init__(self) -> None:
        self._path = _log_path()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self._path.open("w", encoding="utf-8", buffering=1)
        self._every = _env_int("RKK_TICK_RUN_LOG_EVERY", 1)
        self._tick_count = 0
        self._started = time.perf_counter()
        self._max_inner_ms = 0.0
        self._sum_inner_ms = 0.0
        self._write(
            {
                "type": "run_start",
                "ts": datetime.now(timezone.utc).isoformat(),
                "path": str(self._path),
                "log_every": self._every,
                "config": _run_config_snapshot(),
            }
        )

    def _write(self, obj: dict[str, Any]) -> None:
        try:
            self._fh.write(json.dumps(obj, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"[TickRunLog] write failed: {e}")

    def record_tick(
        self,
        sim: Any,
        *,
        result: dict[str, Any],
        snap: dict[str, Any],
        inner_ms: float,
        obs: dict[str, float] | None = None,
        fallen: bool = False,
        posture: float | None = None,
    ) -> None:
        tick = int(getattr(sim, "tick", 0))
        if tick <= 0 or (tick % self._every) != 0:
            return

        self._tick_count += 1
        self._sum_inner_ms += inner_ms
        if inner_ms > self._max_inner_ms:
            self._max_inner_ms = inner_ms

        agent = sim.agent
        vl = snap.get("value_layer") or {}
        prog = snap.get("progressive_scope") or {}
        traj = snap.get("trajectory") or {}
        notears = result.get("notears") if _env_flag("RKK_TICK_RUN_LOG_NOTEAR", True) else None

        edge_count = int(snap.get("edge_count", 0))
        prev_edges = int(getattr(sim, "_tick_log_prev_edges", edge_count))
        edge_delta = edge_count - prev_edges
        sim._tick_log_prev_edges = edge_count

        _INNER_MS_WINDOW.append(inner_ms)
        med_ms = float(sorted(_INNER_MS_WINDOW)[len(_INNER_MS_WINDOW) // 2]) if _INNER_MS_WINDOW else inner_ms
        slowdown = inner_ms > max(500.0, 2.5 * med_ms) if len(_INNER_MS_WINDOW) >= 8 else inner_ms > 2000.0

        agent_timings = dict(getattr(agent, "_last_step_timings", {}) or {})
        agent_total = float(
            agent_timings.get("total_ms")
            or sum(
                float(v)
                for k, v in agent_timings.items()
                if k != "total_ms" and isinstance(v, (int, float))
            )
        )
        inner_phases = dict(getattr(sim, "_inner_phase_ms", {}) or {})
        sim_post_ms = round(max(0.0, inner_ms - agent_total), 2)

        record: dict[str, Any] = {
            "type": "tick",
            "ts": datetime.now(timezone.utc).isoformat(),
            "tick": tick,
            "world": getattr(sim, "current_world", "?"),
            "perf": {
                "inner_ms": round(inner_ms, 2),
                "inner_median_ms": round(med_ms, 2),
                "slowdown_flag": slowdown,
                "agent_step_ms": agent_timings,
                "agent_total_ms": round(agent_total, 2),
                "sim_post_ms": sim_post_ms,
                "inner_phases_ms": inner_phases,
            },
            "hud": {
                "phi": snap.get("phi"),
                "compression_gain": snap.get("compression_gain"),
                "alpha_mean": snap.get("alpha_mean"),
                "h_W": snap.get("h_W"),
                "gnn_d": len(agent.graph._node_ids),
                "edge_count": edge_count,
                "edge_delta": edge_delta,
                "graph_mdl": snap.get("graph_mdl"),
                "do_total": snap.get("total_interventions"),
                "blocked_total": snap.get("total_blocked"),
                "block_rate": vl.get("block_rate"),
                "discovery_rate": snap.get("discovery_rate"),
                "peak_discovery_rate": snap.get("peak_discovery_rate"),
                "phase": getattr(sim, "phase", 0),
                "smoothed_dr": (
                    round(float(statistics.mean(sim._dr_window)), 4)
                    if getattr(sim, "_dr_window", None)
                    else None
                ),
            },
            "scope": {
                "phase": prog.get("phase"),
                "mastery_quality": prog.get(
                    "mastery_quality", prog.get("mastery", 0.0)
                ),
                "ticks_in_phase": prog.get("ticks_in_phase"),
            }
            if prog
            else None,
            "trajectory": traj if traj else None,
            "value_layer": {
                "vl_phase": vl.get("vl_phase"),
                "vl_strictness": vl.get("vl_strictness"),
                "fixed_root_mode": vl.get("fixed_root_mode"),
                "block_reasons": vl.get("block_reasons"),
            },
            "body": {
                "fallen": fallen,
                "fall_count": snap.get("fall_count"),
                "fixed_root": bool(getattr(sim, "_fixed_root_active", False)),
                "cpg_blocked_by_fixed_root": bool(
                    getattr(sim, "_fixed_root_active", False)
                    and not cpg_during_fixed_root_enabled()
                ),
                "visual_mode": bool(getattr(sim, "_visual_mode", False)),
                "posture_stability": posture,
            },
            "action": {
                "variable": result.get("variable"),
                "value": result.get("value"),
                "blocked": bool(result.get("blocked")),
                "skipped": bool(result.get("skipped")),
                "reason": result.get("reason") or result.get("message"),
                "prediction_error": result.get("prediction_error"),
                "compression_delta": result.get("compression_delta"),
                "goal_planned": result.get("goal_planned"),
                "from_cem": result.get("from_cem"),
                "from_system2": result.get("from_system2"),
                "hierarchy": result.get("hierarchy"),
                "skill": result.get("skill"),
            },
            "learning": {
                "notears": notears,
                "rsi_lite": result.get("rsi_lite"),
                "system1_mean_loss": (snap.get("system1") or {}).get("mean_loss"),
            },
            "sleep": _sleep_snapshot(sim),
            "locomotion": _locomotion_snapshot(sim),
        }

        s2snap = _system2_tick_log_snapshot(sim)
        if s2snap is not None:
            record["system2"] = s2snap

        if _env_flag("RKK_TICK_RUN_LOG_PHYS", True) and obs:
            record["phys"] = _phys_snapshot(obs, agent, sim=sim)

        if _env_flag("RKK_TICK_RUN_LOG_EVENTS", True):
            record["events"] = _tick_events(sim, tick)

        self._write(record)

    def finalize(self, sim: Any | None = None) -> None:
        elapsed = time.perf_counter() - self._started
        n = max(1, self._tick_count)
        summary: dict[str, Any] = {
            "type": "run_end",
            "ts": datetime.now(timezone.utc).isoformat(),
            "elapsed_sec": round(elapsed, 2),
            "ticks_logged": self._tick_count,
            "inner_ms_avg": round(self._sum_inner_ms / n, 2),
            "inner_ms_max": round(self._max_inner_ms, 2),
        }
        if sim is not None:
            try:
                snap = sim.agent.snapshot()
                summary["final"] = {
                    "tick": int(sim.tick),
                    "edge_count": snap.get("edge_count"),
                    "gnn_d": len(sim.agent.graph._node_ids),
                    "fall_count": getattr(sim, "_fall_count", 0),
                    "discovery_rate": snap.get("discovery_rate"),
                    "phi": snap.get("phi"),
                }
            except Exception:
                pass
        self._write(summary)
        try:
            self._fh.close()
        except Exception:
            pass
        print(f"[TickRunLog] wrote {self._tick_count} ticks → {self._path}")


def _run_config_snapshot() -> dict[str, str]:
    keys = (
        "RKK_MEMORY_RESUME_ON_START",
        "RKK_SLEEP_ENABLED",
        "RKK_SLEEP_MOCAP_DREAMS",
        "RKK_AGENT_LOOP_HZ",
        "RKK_LOCOMOTION_CPG",
        "RKK_AUTO_FIXED_ROOT_TICKS",
        "RKK_EDGE_THRESH",
        "RKK_CEM_PLANNING",
        "RKK_TRAJECTORY_ENABLED",
        "RKK_SYSTEM2",
        "RKK_SYSTEM2_PLAN_EVERY",
        "RKK_SYSTEM2_MACRO_TICKS",
    )
    return {k: os.environ.get(k, "") for k in keys if os.environ.get(k) is not None}


def _phys_snapshot(obs: dict, agent: Any, *, sim: Any | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {
        "posture_stability": _obs_key(obs, "posture_stability", "phys_posture_stability"),
        "com_z": _obs_key(obs, "com_z", "phys_com_z"),
        "com_x": _obs_key(obs, "com_x", "phys_com_x"),
        "foot_contact_l": _obs_key(obs, "foot_contact_l", "phys_foot_contact_l"),
        "foot_contact_r": _obs_key(obs, "foot_contact_r", "phys_foot_contact_r"),
        "torso_pitch": _obs_key(obs, "torso_pitch", "phys_torso_pitch"),
    }
    base = _unwrap_base_env(getattr(agent, "env", None))
    if base is not None:
        for k, v in _phys_from_sim_raw(base).items():
            if out.get(k) is None:
                out[k] = v
        out["cpg_owns_legs"] = bool(getattr(base, "cpg_owns_legs", False))
    ms = getattr(base, "_motor_state", None) if base is not None else None
    if ms is None:
        ms = getattr(getattr(agent, "env", None), "_motor_state", None)
    if isinstance(ms, dict):
        intents = {
            k: round(_safe_float(v), 4)
            for k, v in ms.items()
            if str(k).startswith("intent_")
        }
        if intents:
            out["motor_intents"] = intents
    if sim is not None:
        out["cpg_l1_last_apply_tick"] = int(getattr(sim, "_l1_last_apply_tick", -1))
        out["cpg_decoupled_hz"] = round(
            float(os.environ.get("RKK_CPG_LOOP_HZ", "0") or 0), 1
        )
    return out


def _sleep_snapshot(sim: Any) -> dict[str, Any] | None:
    ctrl = getattr(sim, "_sleep_ctrl", None)
    if ctrl is None:
        return None
    return {
        "sleeping": bool(getattr(ctrl, "is_sleeping", False)),
        "sleep_count": int(getattr(ctrl, "sleep_count", 0)),
        "falls_since_sleep": int(getattr(ctrl, "_falls_since_sleep", 0)),
    }


def _locomotion_snapshot(sim: Any) -> dict[str, Any] | None:
    lc = getattr(sim, "_locomotion_controller", None)
    reward = round(
        _safe_float(sim._locomotion_reward_ema())
        if callable(getattr(sim, "_locomotion_reward_ema", None))
        else 0.0,
        4,
    )
    if lc is None:
        return {
            "controller_ready": False,
            "cpg_blocked_by_fixed_root": bool(
                getattr(sim, "_fixed_root_active", False)
                and not cpg_during_fixed_root_enabled()
            ),
            "reward_ema": reward,
        }
    return {
        "controller_ready": True,
        "cpg_weight": round(_safe_float(getattr(lc, "cpg_weight", 0.0)), 4),
        "reward_ema": reward,
    }


def _tick_events(sim: Any, tick: int) -> list[dict[str, Any]]:
    events = getattr(sim, "events", None)
    if not events:
        return []
    out: list[dict[str, Any]] = []
    for ev in events:
        if not isinstance(ev, dict):
            continue
        if int(ev.get("tick", -1)) != tick:
            continue
        out.append(
            {
                "text": str(ev.get("text", ""))[:500],
                "type": ev.get("type"),
            }
        )
    return out


def get_tick_run_logger() -> TickRunLogger | None:
    global _LOGGER
    if not tick_run_log_enabled():
        return None
    if _LOGGER is None:
        try:
            _LOGGER = TickRunLogger()
        except Exception as e:
            print(f"[TickRunLog] init failed: {e}")
            return None
    return _LOGGER


def record_sim_tick(
    sim: Any,
    *,
    result: dict[str, Any],
    snap: dict[str, Any],
    inner_ms: float,
    obs: dict[str, float] | None = None,
    fallen: bool = False,
    posture: float | None = None,
) -> None:
    lg = get_tick_run_logger()
    if lg is None:
        return
    try:
        lg.record_tick(
            sim,
            result=result,
            snap=snap,
            inner_ms=inner_ms,
            obs=obs,
            fallen=fallen,
            posture=posture,
        )
    except Exception as e:
        print(f"[TickRunLog] record_tick: {e}")


def finalize_tick_run_log(sim: Any | None = None) -> None:
    global _LOGGER
    if _LOGGER is None:
        return
    try:
        _LOGGER.finalize(sim)
    except Exception as e:
        print(f"[TickRunLog] finalize: {e}")
    _LOGGER = None
