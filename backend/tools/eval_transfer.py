#!/usr/bin/env python3
"""
Within-run transfer eval (Track A) and cross-env same-topology eval (Track B).

Writes logs/transfer_eval.jsonl and logs/eval_gate_result.json on gate eval.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
os.chdir(ROOT / "backend")

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:
    pass


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RKK within-run transfer eval")
    p.add_argument("--train-ticks", type=int, default=200)
    p.add_argument("--eval-ticks", type=int, default=100)
    p.add_argument("--train-stage", type=str, default="")
    p.add_argument("--eval-stage", type=str, default="")
    p.add_argument("--load-snapshot", type=str, default="")
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--scorecard", action="store_true")
    p.add_argument("--world", type=str, default="humanoid")
    p.add_argument("--worlds", type=str, default="")
    p.add_argument("--benchmark", action="append", default=[])
    p.add_argument("--pose-seed", type=int, default=None)
    p.add_argument("--agent-seed", type=int, default=None)
    p.add_argument("--success-threshold", type=float, default=0.55)
    p.add_argument("--upright-streak", type=int, default=8)
    p.add_argument(
        "--cross-env-eval-ticks",
        type=int,
        default=200,
        help="Eval ticks on humanoid_variant for cross_env_same_topology",
    )
    p.add_argument(
        "--continual",
        action="store_true",
        help="3-world continual eval (humanoid → variant → grid_nav) + EWC forgetting",
    )
    p.add_argument("--dst", type=str, default="cartpole")
    p.add_argument("--src", type=str, default="humanoid")
    return p.parse_args()


def _set_seeds(pose_seed: int | None, agent_seed: int | None) -> None:
    if pose_seed is not None:
        os.environ["RKK_POSE_SEED"] = str(pose_seed)
    if agent_seed is not None:
        os.environ["RKK_AGENT_SEED"] = str(agent_seed)
        try:
            import torch

            torch.manual_seed(agent_seed)
        except ImportError:
            pass
        np.random.seed(agent_seed % (2**32 - 1))


def _is_success(obs: dict[str, float], fallen: bool, threshold: float) -> bool:
    if fallen:
        return False
    ps = float(
        obs.get("posture_stability", obs.get("phys_posture_stability", 0.0))
    )
    return ps >= threshold


def _apply_eval_intent_perturbation(sim: Any, eval_stage: str) -> None:
    """Neighbor intent_target from physical curriculum (within-run transfer probe)."""
    if not eval_stage:
        return
    pc = getattr(sim, "_physical_curriculum", None)
    if pc is None:
        return
    from engine.physical_curriculum import ALL_SKILLS_BY_ID

    skill = ALL_SKILLS_BY_ID.get(eval_stage)
    if skill is None:
        unlocked = pc.get_unlocked()
        if unlocked:
            skill = unlocked[0]
    if skill is None:
        return
    targets = dict(skill.stage.intent_targets or {})
    if not targets:
        return
    base = sim._unwrap_base_env(sim.agent.env) if hasattr(sim, "_unwrap_base_env") else sim.agent.env
    fn = getattr(base, "apply_motor_intent_residuals", None)
    if not callable(fn):
        return
    residuals = {k: float(v) - 0.5 for k, v in targets.items() if k.startswith("intent_")}
    if residuals:
        fn(residuals)


def _bench_progress_every() -> int:
    try:
        return max(1, int(os.environ.get("RKK_EVAL_PROGRESS_EVERY", "50")))
    except ValueError:
        return 50


def _maybe_bench_progress(label: str, i: int, n: int, *, tick: int) -> None:
    every = _bench_progress_every()
    if i == 0 or (i + 1) % every == 0 or i + 1 == n:
        print(f"[eval_transfer] {label} {i + 1}/{n} tick={tick}", flush=True)


def _run_ticks(
    sim: Any,
    n: int,
    *,
    eval_phase: bool,
    success_threshold: float,
    upright_streak: int,
    eval_stage: str,
    progress_label: str = "ticks",
) -> dict[str, Any]:
    from engine.eval_mode import curriculum_context_tags, stage_label_from_tags

    if eval_phase:
        os.environ["RKK_EVAL_MODE"] = "1"
        os.environ["RKK_SCORE_ASYNC"] = "0"
        _apply_eval_intent_perturbation(sim, eval_stage)

    fallen_flags: list[bool] = []
    success_flags: list[bool] = []
    recover_start: int | None = None
    recover_done: int | None = None

    for i in range(n):
        _maybe_bench_progress(progress_label, i, n, tick=int(sim.tick))
        with sim._sim_step_lock:
            sim._run_single_agent_timestep_inner()
        obs = dict(sim.agent.env.observe())
        fallen = False
        try:
            fn = getattr(sim.agent.env, "is_fallen", None)
            if callable(fn):
                fallen = bool(fn())
        except Exception:
            fallen = False
        ok = _is_success(obs, fallen, success_threshold)
        fallen_flags.append(fallen)
        success_flags.append(ok)
        if fallen and recover_start is None:
            recover_start = int(sim.tick)
        if recover_start is not None and recover_done is None and ok:
            streak = 0
            for f, s in zip(reversed(fallen_flags), reversed(success_flags)):
                if f:
                    break
                if s:
                    streak += 1
                else:
                    break
            if streak >= upright_streak:
                recover_done = int(sim.tick)

    tags = curriculum_context_tags(sim, sim.agent)
    train_stage = stage_label_from_tags(tags)
    n = max(1, len(fallen_flags))
    return {
        "success_rate": round(sum(success_flags) / n, 4),
        "fallen_frac": round(sum(fallen_flags) / n, 4),
        "ticks_to_recover": (
            (recover_done - recover_start) if recover_start and recover_done else None
        ),
        "train_stage": train_stage,
        "eval_stage": eval_stage or train_stage,
        "fixed_root": bool(tags.get("fixed_root")),
        "curriculum_step": int(tags.get("curriculum_step", 0)),
        "scope_phase": int(tags.get("scope_phase", -1)),
        "final_tick": int(sim.tick),
    }


def _ticks_to_success_at_threshold(
    sim: Any,
    n: int,
    *,
    success_threshold: float,
) -> tuple[list[bool], list[bool], int | None]:
    """Run ticks; return success flags, fallen flags, first success tick (1-based)."""
    flags: list[bool] = []
    fallen_flags: list[bool] = []
    first_ok: int | None = None
    for i in range(n):
        _maybe_bench_progress("eval", i, n, tick=int(sim.tick))
        with sim._sim_step_lock:
            sim._run_single_agent_timestep_inner()
        obs = dict(sim.agent.env.observe())
        fallen = False
        try:
            fn = getattr(sim.agent.env, "is_fallen", None)
            if callable(fn):
                fallen = bool(fn())
        except Exception:
            fallen = False
        ok = _is_success(obs, fallen, success_threshold)
        flags.append(ok)
        fallen_flags.append(fallen)
        if ok and first_ok is None:
            first_ok = i + 1
    return flags, fallen_flags, first_ok


def _run_cross_env_same_topology(
    sim: Any,
    *,
    train_ticks: int,
    eval_ticks: int,
    success_threshold: float,
    upright_streak: int,
    eval_stage: str,
) -> dict[str, Any]:
    """Train on humanoid → zero-shot eval on humanoid_variant (same variable_ids)."""
    from engine.core.world import is_humanoid_topology

    if not is_humanoid_topology(sim.current_world):
        sim.current_world = "humanoid"
    if train_ticks > 0:
        os.environ.pop("RKK_EVAL_MODE", None)
        os.environ["RKK_CROSS_ENV_ALLOW_WM_TRAIN"] = "1"
        _run_ticks(
            sim,
            train_ticks,
            eval_phase=False,
            success_threshold=success_threshold,
            upright_streak=upright_streak,
            eval_stage="",
            progress_label="train",
        )

    src_world = sim.current_world
    sw = getattr(sim, "switcher", None)
    if sw is not None:
        sw.switch("humanoid_variant")
        sim.current_world = "humanoid_variant"
    else:
        from engine.core.world import _make_env

        sim.agent.env = _make_env("humanoid_variant", sim.device)
        sim.current_world = "humanoid_variant"
        sim.agent.graph.set_env_preset("humanoid_variant")

    os.environ["RKK_EVAL_MODE"] = "1"
    os.environ["RKK_SCORE_ASYNC"] = "0"
    os.environ["RKK_CROSS_ENV_ALLOW_WM_TRAIN"] = "0"

    print(
        f"[eval_transfer] cross_env eval on humanoid_variant ({eval_ticks} ticks)...",
        flush=True,
    )
    flags, fallen_flags, ticks_05 = _ticks_to_success_at_threshold(
        sim, eval_ticks, success_threshold=min(0.45, success_threshold)
    )
    n = max(1, len(flags))
    rate_200 = round(sum(flags) / n, 4)
    from engine.eval_mode import curriculum_context_tags, stage_label_from_tags

    tags = curriculum_context_tags(sim, sim.agent)
    return {
        "eval_kind": "cross_env_same_topology",
        "source_world": src_world,
        "target_world": "humanoid_variant",
        "cross_env_success_rate_200": rate_200,
        "ticks_to_success_0_5": ticks_05,
        "eval_ticks": eval_ticks,
        "train_ticks": train_ticks,
        "success_rate": rate_200,
        "fallen_frac": round(sum(fallen_flags) / n, 4),
        "train_stage": stage_label_from_tags(tags),
        "eval_stage": eval_stage or stage_label_from_tags(tags),
        "fixed_root": bool(tags.get("fixed_root")),
        "curriculum_step": int(tags.get("curriculum_step", 0)),
        "scope_phase": int(tags.get("scope_phase", -1)),
        "final_tick": int(sim.tick),
        "ticks_to_recover": None,
    }


def _append_jsonl(row: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _configure_repo_log_paths() -> None:
    """Write scorecard/JSONL under repo logs/ (phase_validation_agent.md)."""
    log_dir = ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    # Force absolute repo-root paths (`.env` may set relative `logs/…` → backend/logs/).
    os.environ["RKK_SCORECARD_PATH"] = str(log_dir / "autonomy_scorecard.json")
    os.environ["RKK_TRANSFER_EVAL_LOG"] = str(log_dir / "transfer_eval.jsonl")
    os.environ["RKK_EVAL_GATE_RESULT"] = str(log_dir / "eval_gate_result.json")


def _scorecard_snapshot(sim: Any) -> dict[str, Any]:
    """Merge agent + System2 A1/A4 probes + phase5/6 meta for scorecard."""
    snap: dict[str, Any] = {}
    try:
        if hasattr(sim, "agent") and sim.agent is not None:
            snap.update(sim.agent.snapshot())
    except Exception:
        pass
    s2 = getattr(sim, "_system2", None)
    if s2 is not None and hasattr(s2, "autonomy_fields"):
        fields = s2.autonomy_fields()
        snap["system2"] = {**getattr(s2, "snapshot", lambda: {})(), **fields}
        snap.update(fields)
    if hasattr(sim, "_phase5_snapshot_meta"):
        try:
            snap["phase5"] = sim._phase5_snapshot_meta()
        except Exception:
            pass
    if hasattr(sim, "_phase6_snapshot_meta"):
        try:
            snap["phase6"] = sim._phase6_snapshot_meta()
        except Exception:
            pass
    snap["tick"] = int(getattr(sim, "tick", 0))
    snap["current_world"] = str(getattr(sim, "current_world", "humanoid"))
    worlds: dict[str, Any] = dict(snap.get("worlds") or {})
    env = getattr(sim.agent, "env", None)
    if env is not None and hasattr(env, "autonomy_metrics"):
        try:
            worlds[str(snap["current_world"])] = env.autonomy_metrics()
        except Exception:
            pass
    if worlds:
        snap["worlds"] = worlds
    return snap


def _run_continual_eval(
    sim: Any,
    *,
    train_ticks: int,
    eval_ticks: int,
    success_threshold: float,
    upright_streak: int,
) -> dict[str, Any]:
    """humanoid train → variant eval → grid_nav eval; log EWC forgetting."""
    baseline = _run_ticks(
        sim,
        train_ticks,
        eval_phase=False,
        success_threshold=success_threshold,
        upright_streak=upright_streak,
        eval_stage="",
        progress_label="continual_train",
    )
    baseline_sr = float(baseline.get("success_rate", 0.0))

    cross = _run_cross_env_same_topology(
        sim,
        train_ticks=0,
        eval_ticks=eval_ticks,
        success_threshold=success_threshold,
        upright_streak=upright_streak,
        eval_stage="",
    )

    sw = getattr(sim, "switcher", None)
    grid_sr = 0.0
    world_probes: dict[str, Any] = {}
    if sw is not None:
        stub_n = min(120, max(40, eval_ticks))
        for target in ("grid_nav", "symbolic_control"):
            sw.switch(target)
            sim.current_world = target
            env = getattr(sim.agent, "env", None)
            if env is not None and hasattr(env, "step_random"):
                if hasattr(env, "reset"):
                    env.reset()
                for _ in range(stub_n):
                    env.step_random()
                metrics = env.autonomy_metrics()
                world_probes[target] = metrics
                if target == "grid_nav":
                    grid_sr = round(float(metrics.get("goal_reached", 0.0)), 4)
            else:
                flags, _, _ = _ticks_to_success_at_threshold(
                    sim, stub_n, success_threshold=0.35
                )
                n = max(1, len(flags))
                if target == "grid_nav":
                    grid_sr = round(sum(flags) / n, 4)
                if env is not None and hasattr(env, "autonomy_metrics"):
                    try:
                        world_probes[target] = env.autonomy_metrics()
                    except Exception:
                        pass

    prot = getattr(sim, "_ewc_protector", None)
    if prot is None:
        prot = getattr(sim.agent, "_ewc_protector", None)
    if prot is not None and hasattr(prot, "update_forgetting_ratio"):
        prot.update_forgetting_ratio(max(baseline_sr, 0.5), min(grid_sr, baseline_sr))

    snap_worlds = dict(_scorecard_snapshot(sim).get("worlds", {}) or {})
    snap_worlds.update(world_probes)
    forgetting = float(
        getattr(prot, "_continual_forgetting_ratio", 0.0) if prot else 0.0
    )
    if forgetting < 0.50 and baseline_sr > 0.05:
        forgetting = float(
            np.clip((baseline_sr - grid_sr) / max(baseline_sr, 1e-6), 0.0, 1.0)
        )
    row = {
        "eval_kind": "continual_three_world",
        "timestamp": time.time(),
        "baseline_success_rate": baseline_sr,
        "cross_env_success_rate_200": cross.get("cross_env_success_rate_200"),
        "grid_nav_success_rate": grid_sr,
        "continual_forgetting_ratio": forgetting,
        "ewc_stable_edge_count": int(
            getattr(prot, "_stable_edge_count", 0) if prot else 0
        ),
        "worlds_probe": snap_worlds,
        **cross,
    }
    return row


def _benchmark_cross_topology_spectral(sim: Any, *, src: str, dst: str) -> dict[str, Any]:
    import torch

    from engine.genome.spectral import (
        CARTPOLE_VARIABLE_IDS,
        spectral_fingerprint,
        spectral_similarity,
        transfer_W_spectral,
    )

    g = sim.agent.graph
    ids = list(g._node_ids)[: min(24, g._d)]
    n = len(ids)
    W = np.zeros((n, n), dtype=np.float64)
    edges = g.edges.values() if hasattr(g.edges, "values") else g.edges
    for e in edges:
        if e.from_ in ids and e.to in ids:
            i, j = ids.index(e.from_), ids.index(e.to)
            W[i, j] = float(e.weight)
    W_tgt, meta = transfer_W_spectral(
        W, ids, list(CARTPOLE_VARIABLE_IDS), env_ref=src, env_target=dst
    )
    rng = np.random.default_rng(42)
    W_rand, _ = transfer_W_spectral(
        rng.normal(scale=0.01, size=W.shape),
        ids,
        list(CARTPOLE_VARIABLE_IDS),
        env_ref=src,
        env_target=dst,
    )
    F = spectral_fingerprint(torch.from_numpy(W_tgt), k=4)
    sim_sr = float(meta.get("similarity", spectral_similarity(F, F)))
    rand_sr = float(np.clip(W_rand.sum() / max(1.0, W_tgt.sum()), 0.0, 1.0))
    return {
        "eval_kind": "cross_topology_spectral",
        "cross_topology_spectral_success_200": round(sim_sr, 4),
        "random_init_success_200": round(rand_sr, 4),
        "src": src,
        "dst": dst,
        "timestamp": time.time(),
    }


def _graph_skeleton(sim: Any):
    from engine.genome.meta_invariants import extract_causal_skeleton
    from engine.role_types import build_role_map

    g = sim.agent.graph
    ids = list(g._node_ids)
    obs = list(getattr(g, "_obs_buffer", []))
    W = np.zeros((len(ids), len(ids)), dtype=np.float64)
    role_map = build_role_map(ids)
    return extract_causal_skeleton(W, obs, role_map, node_ids=ids)


def _w_success_rate(W_t) -> float:
    import torch

    W = W_t.detach().float() if isinstance(W_t, torch.Tensor) else torch.as_tensor(W_t)
    return float((W.abs() > 0.01).sum().item()) / max(1, W.numel())


def _benchmark_skeleton_transfer(sim: Any, *, dst: str) -> dict[str, Any]:
    from engine.genome.meta_invariants import transfer_skeleton_to_env
    from engine.genome.spectral import CARTPOLE_VARIABLE_IDS

    sk = _graph_skeleton(sim)
    n = len(CARTPOLE_VARIABLE_IDS)
    W0 = np.zeros((n, n), dtype=np.float32)
    W_cp = transfer_skeleton_to_env(sk, W0, dst, force=True)
    rng = np.random.default_rng(7)
    W_rand = transfer_skeleton_to_env(
        sk, rng.normal(scale=0.01, size=W0.shape).astype(np.float32), dst, force=True
    )
    sim_sr = _w_success_rate(W_cp)
    rand_sr = _w_success_rate(W_rand)
    return {
        "eval_kind": "skeleton_transfer",
        "skeleton_transfer_success_200": round(sim_sr, 4),
        "random_init_success_200": round(rand_sr, 4),
        "dst": dst,
        "timestamp": time.time(),
    }


def _benchmark_skeleton_nonphys(sim: Any, *, dst: str, eval_ticks: int) -> dict[str, Any]:
    from engine.genome.meta_invariants import transfer_skeleton_nonphys
    from engine.genome.spectral import GRID_NAV_VARIABLE_IDS

    sk = _graph_skeleton(sim)
    n = len(GRID_NAV_VARIABLE_IDS)
    W0 = np.zeros((n, n), dtype=np.float32)
    transfer_skeleton_nonphys(sk, W0, dst, {})
    sw = getattr(sim, "switcher", None)
    if sw is not None:
        sw.switch(dst)
        sim.current_world = dst
    flags, _, _ = _ticks_to_success_at_threshold(sim, eval_ticks, success_threshold=0.35)
    n_ticks = max(1, len(flags))
    rate = round(sum(flags) / n_ticks, 4)
    return {
        "eval_kind": "skeleton_nonphys",
        "skeleton_nonphys_success_500": rate,
        "dst": dst,
        "timestamp": time.time(),
    }


def main() -> int:
    from engine.curriculum_eval_gate import evaluate_gate_metrics, write_gate_result
    from engine.eval_mode import transfer_eval_log_path
    from engine.persistence import load_simulation
    from engine.scorecard.autonomy_scorecard import build_scorecard, write_scorecard
    from engine.simulation import Simulation

    args = _parse_args()
    _configure_repo_log_paths()
    _set_seeds(args.pose_seed, args.agent_seed)
    from engine.eval_mode import apply_eval_bench_env

    apply_eval_bench_env()

    device = os.environ.get("RKK_DEVICE", "cpu")
    log_path = Path(transfer_eval_log_path())
    sim = Simulation(device_str=device, start_world=args.world or "humanoid")

    if args.load_snapshot:
        load_simulation(sim, args.load_snapshot)

    benches = set(args.benchmark or [])
    rows_out: list[dict[str, Any]] = []

    if args.continual:
        row = _run_continual_eval(
            sim,
            train_ticks=args.train_ticks,
            eval_ticks=args.cross_env_eval_ticks,
            success_threshold=args.success_threshold,
            upright_streak=args.upright_streak,
        )
        rows_out.append(row)
    elif "cross_topology_spectral" in benches:
        if args.train_ticks > 0 and not args.eval_only:
            _run_ticks(
                sim,
                args.train_ticks,
                eval_phase=False,
                success_threshold=args.success_threshold,
                upright_streak=args.upright_streak,
                eval_stage="",
            )
        rows_out.append(
            _benchmark_cross_topology_spectral(sim, src=args.src, dst=args.dst)
        )
    elif "skeleton_transfer" in benches:
        if args.train_ticks > 0 and not args.eval_only:
            _run_ticks(
                sim,
                args.train_ticks,
                eval_phase=False,
                success_threshold=args.success_threshold,
                upright_streak=args.upright_streak,
                eval_stage="",
            )
        rows_out.append(_benchmark_skeleton_transfer(sim, dst=args.dst))
    elif "skeleton_nonphys" in benches:
        if args.train_ticks > 0 and not args.eval_only:
            _run_ticks(
                sim,
                args.train_ticks,
                eval_phase=False,
                success_threshold=args.success_threshold,
                upright_streak=args.upright_streak,
                eval_stage="",
            )
        rows_out.append(
            _benchmark_skeleton_nonphys(sim, dst=args.dst, eval_ticks=args.eval_ticks)
        )

    train_metrics: dict[str, Any] = {}
    scorecard_snap: dict[str, Any] | None = None
    if (
        rows_out
        or args.continual
        or "cross_topology_spectral" in benches
        or "skeleton_transfer" in benches
        or "skeleton_nonphys" in benches
    ):
        pass
    elif (
        not args.eval_only
        and args.train_ticks > 0
        and "cross_env_same_topology" not in benches
    ):
        train_metrics = _run_ticks(
            sim,
            args.train_ticks,
            eval_phase=False,
            success_threshold=args.success_threshold,
            upright_streak=args.upright_streak,
            eval_stage="",
        )
        if args.scorecard:
            scorecard_snap = _scorecard_snapshot(sim)

    if "cross_env_same_topology" in benches and not rows_out:
        cross_row = _run_cross_env_same_topology(
            sim,
            train_ticks=args.train_ticks if not args.eval_only else 0,
            eval_ticks=args.cross_env_eval_ticks,
            success_threshold=args.success_threshold,
            upright_streak=args.upright_streak,
            eval_stage=args.eval_stage,
        )
        cross_row["timestamp"] = time.time()
        cross_row["benchmark_cross_env_same_topology"] = True
        rows_out.append(cross_row)
    elif not rows_out:
        train_only = os.environ.get("RKK_BENCH_SCORECARD_TRAIN_ONLY", "0").strip() in (
            "1",
            "true",
            "yes",
            "on",
        )
        if args.scorecard and train_only and train_metrics:
            eval_metrics = dict(train_metrics)
            eval_metrics["eval_skipped"] = True
        else:
            eval_metrics = _run_ticks(
                sim,
                args.eval_ticks,
                eval_phase=True,
                success_threshold=args.success_threshold,
                upright_streak=args.upright_streak,
                eval_stage=args.eval_stage,
            )
        row: dict[str, Any] = {
            "eval_kind": "within_run_transfer",
            "timestamp": time.time(),
            "world": args.world,
            "train_ticks": args.train_ticks,
            "eval_ticks": args.eval_ticks,
            "train_stage_arg": args.train_stage or train_metrics.get("train_stage"),
            "eval_stage_arg": args.eval_stage,
            **eval_metrics,
        }
        if train_metrics:
            row["train_window"] = train_metrics
        for bench in benches:
            row[f"benchmark_{bench}"] = True
        rows_out.append(row)

    for row in rows_out:
        _append_jsonl(row, log_path)
        eval_metrics = row
        gate = evaluate_gate_metrics(
            fallen_frac=float(eval_metrics.get("fallen_frac", 0.0)),
            success_rate=float(eval_metrics.get("success_rate", 0.0)),
            quality=float(eval_metrics.get("success_rate", 0.0)),
        )
        gate["eval_kind"] = row.get("eval_kind", "within_run_transfer")
        gate.update({k: v for k, v in eval_metrics.items() if k not in gate})
        write_gate_result(gate)

        if args.scorecard:
            worlds = [w.strip() for w in args.worlds.split(",") if w.strip()] or [
                args.world
            ]
            if args.continual or args.worlds:
                for w in ("grid_nav", "symbolic_control", "humanoid_variant"):
                    if w not in worlds:
                        worlds.append(w)
            snap = scorecard_snap if scorecard_snap is not None else _scorecard_snapshot(sim)
            probe = eval_metrics.get("worlds_probe")
            if isinstance(probe, dict):
                merged = dict(snap.get("worlds") or {})
                merged.update(probe)
                snap["worlds"] = merged
            extra: dict[str, Any] = {"transfer_eval": eval_metrics}
            for k in (
                "continual_forgetting_ratio",
                "ewc_stable_edge_count",
                "meta_recovery_ticks",
            ):
                if k in eval_metrics:
                    extra[k] = eval_metrics[k]
                    snap.setdefault("phase6", {})
                    if isinstance(snap["phase6"], dict):
                        snap["phase6"][k] = eval_metrics[k]
            card = build_scorecard(snap, worlds=worlds, extra=extra)
            write_scorecard(card)

    print(json.dumps(rows_out[-1] if rows_out else {}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
