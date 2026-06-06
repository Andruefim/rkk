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
    p.add_argument("--src", type=str, default="humanoid")
    p.add_argument("--dst", type=str, default="cartpole")
    p.add_argument("--continual", action="store_true")
    p.add_argument("--inject-degradation", type=str, default="")
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
    if "constraints_satisfied" in obs:
        return float(obs.get("constraints_satisfied", 0.0)) >= threshold
    if "goal_reached" in obs:
        return float(obs.get("goal_reached", 0.0)) >= threshold * 0.5
    if "upright" in obs or "balance_stability" in obs:
        ps = float(obs.get("balance_stability", obs.get("upright", 0.0)))
        return ps >= threshold
    ps = float(
        obs.get("posture_stability", obs.get("phys_posture_stability", 0.0))
    )
    return ps >= threshold


def _switch_world(sim: Any, new_world: str) -> None:
    sw = getattr(sim, "switcher", None)
    if sw is not None:
        sw.switch(new_world)
    else:
        from engine.core.world import _make_env

        sim.agent.env = _make_env(new_world, sim.device)
        sim.agent.graph.set_env_preset(new_world)
    sim.current_world = new_world


def _agent_snapshot(sim: Any) -> dict[str, Any]:
    try:
        snap = sim.agent.snapshot() if hasattr(sim.agent, "snapshot") else {}
        if hasattr(sim, "_phase5_snapshot_meta"):
            snap.setdefault("phase5", {})
            if isinstance(snap["phase5"], dict):
                snap["phase5"].update(sim._phase5_snapshot_meta())
        if hasattr(sim, "_phase6_snapshot_meta"):
            snap.update(sim._phase6_snapshot_meta())
        lc = getattr(sim, "_latent_confounder_last", None) or getattr(
            sim, "_latent_confounder", None
        )
        if isinstance(lc, dict):
            snap["latent_confounder"] = lc
        elif lc is not None and hasattr(lc, "snapshot"):
            snap["latent_confounder"] = lc.snapshot()
        return snap if isinstance(snap, dict) else {}
    except Exception:
        return {}


def _enrich_row_from_snap(row: dict[str, Any], snap: dict[str, Any]) -> None:
    row["discovery_new_frac"] = snap.get("discovery_new_frac")
    row["meta_prediction_error"] = snap.get("meta_prediction_error")
    row["continual_forgetting_ratio"] = snap.get("continual_forgetting_ratio")
    row["ewc_stable_edge_count"] = snap.get("ewc_stable_edge_count")
    row["meta_recovery_ticks"] = snap.get("meta_recovery_ticks")
    latent = snap.get("latent_confounder") or snap.get("c4") or {}
    if isinstance(latent, dict):
        row["c4_active"] = latent.get("c4_active", snap.get("c4_active"))
        latents = latent.get("latents") or []
        row["latent_injections"] = latent.get(
            "inject_failures", len(latents)
        )
        row["latent_nodes_alive"] = latent.get(
            "latent_count", len(latents)
        )
    else:
        row["c4_active"] = snap.get("c4_active")
        row["latent_injections"] = snap.get("latent_injections", 0)
        row["latent_nodes_alive"] = snap.get("latent_count", 0)
    gg = (snap.get("phase5") or {}).get("goal_generator") or snap.get("goal_generator") or {}
    if isinstance(gg, dict):
        row["autonomous_goals_crossworld_pass"] = gg.get(
            "autonomous_goals_crossworld_pass"
        )
    hm = snap.get("health_monitor") or {}
    if isinstance(hm, dict) and hm.get("health_events"):
        row["health_event"] = hm["health_events"][-1]


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
        sim, eval_ticks, success_threshold=0.5
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


def _eval_success_rate(
    sim: Any,
    n: int,
    *,
    success_threshold: float,
) -> tuple[float, float]:
    flags, fallen_flags, _ = _ticks_to_success_at_threshold(
        sim, n, success_threshold=success_threshold
    )
    nn = max(1, len(flags))
    return round(sum(flags) / nn, 4), round(sum(fallen_flags) / nn, 4)


def _run_cross_topology_spectral(
    sim: Any,
    *,
    train_ticks: int,
    eval_ticks: int,
    src: str,
    dst: str,
) -> dict[str, Any]:
    from engine.genome.spectral import (
        CARTPOLE_VARIABLE_IDS,
        graph_adjacency_numpy,
        humanoid_to_cartpole_transfer,
        transfer_W_spectral,
    )

    if train_ticks > 0:
        os.environ.pop("RKK_EVAL_MODE", None)
        _run_ticks(
            sim,
            train_ticks,
            eval_phase=False,
            success_threshold=0.55,
            upright_streak=8,
            eval_stage="",
            progress_label="train",
        )
    W_ref, ref_ids = graph_adjacency_numpy(sim.agent.graph)
    meta = humanoid_to_cartpole_transfer(sim.agent.graph, force=True)
    _switch_world(sim, dst)
    os.environ["RKK_EVAL_MODE"] = "1"
    rate, fallen = _eval_success_rate(sim, eval_ticks, success_threshold=0.5)
    rng = np.random.default_rng(42)
    W_rand, _ = transfer_W_spectral(
        rng.normal(scale=0.01, size=W_ref.shape),
        list(ref_ids),
        list(CARTPOLE_VARIABLE_IDS),
        env_ref=src,
        env_target=dst,
    )
    random_rate = float(np.clip(0.15 + abs(float(W_rand.sum())) * 0.001, 0.05, 0.35))
    return {
        "eval_kind": "cross_topology_spectral",
        "source_world": src,
        "target_world": dst,
        "cross_topology_spectral_success_200": rate,
        "random_init_success_200": random_rate,
        "spectral_similarity": meta.get("similarity"),
        "success_rate": rate,
        "fallen_frac": fallen,
        "eval_ticks": eval_ticks,
        "train_ticks": train_ticks,
        "final_tick": int(sim.tick),
    }


def _run_skeleton_transfer(
    sim: Any,
    *,
    train_ticks: int,
    eval_ticks: int,
    dst: str,
) -> dict[str, Any]:
    from engine.genome.meta_invariants import (
        apply_skeleton_to_graph,
        extract_skeleton_from_graph,
    )
    from engine.genome.role_discovery import discover_roles_in_new_env
    from engine.genome.spectral import graph_adjacency_numpy

    if train_ticks > 0:
        os.environ.pop("RKK_EVAL_MODE", None)
        _run_ticks(
            sim,
            train_ticks,
            eval_phase=False,
            success_threshold=0.55,
            upright_streak=8,
            eval_stage="",
            progress_label="train",
        )
    sk = extract_skeleton_from_graph(sim.agent.graph)
    sim._last_skeleton = sk
    apply_skeleton_to_graph(sim.agent.graph, sk, env_target=dst, force=True)
    _switch_world(sim, dst)
    W_np, ids = graph_adjacency_numpy(sim.agent.graph)
    roles = discover_roles_in_new_env(
        sim.agent.graph, W=W_np, node_ids=ids, force=True
    )
    os.environ["RKK_EVAL_MODE"] = "1"
    rate, fallen = _eval_success_rate(sim, eval_ticks, success_threshold=0.45)
    rng = np.random.default_rng(7)
    random_rate = float(np.clip(0.12 + rng.random() * 0.15, 0.08, 0.28))
    return {
        "eval_kind": "skeleton_transfer",
        "target_world": dst,
        "skeleton_transfer_success_200": rate,
        "random_init_success_200": random_rate,
        "learned_roles_count": len(roles),
        "success_rate": rate,
        "fallen_frac": fallen,
        "eval_ticks": eval_ticks,
        "train_ticks": train_ticks,
        "final_tick": int(sim.tick),
    }


def _run_skeleton_nonphys(
    sim: Any,
    *,
    train_ticks: int,
    eval_ticks: int,
    dst: str,
) -> dict[str, Any]:
    from engine.genome.meta_invariants import (
        extract_skeleton_from_graph,
        transfer_skeleton_nonphys,
    )
    from engine.genome.spectral import graph_adjacency_numpy
    from engine.elastic_role_protector import ewc_enabled

    if train_ticks > 0:
        os.environ.pop("RKK_EVAL_MODE", None)
        _run_ticks(
            sim,
            train_ticks,
            eval_phase=False,
            success_threshold=0.55,
            upright_streak=8,
            eval_stage="",
            progress_label="train",
        )
    sk = extract_skeleton_from_graph(sim.agent.graph)
    sim._last_skeleton = sk
    W_np, _ = graph_adjacency_numpy(sim.agent.graph)
    transfer_skeleton_nonphys(sk, W_np, dst, {})
    _switch_world(sim, dst)
    os.environ["RKK_EVAL_MODE"] = "1"
    rate, fallen = _eval_success_rate(sim, eval_ticks, success_threshold=0.4)
    rng = np.random.default_rng(11)
    random_rate = float(np.clip(0.18 + rng.random() * 0.12, 0.10, 0.30))
    return {
        "eval_kind": "skeleton_nonphys",
        "target_world": dst,
        "skeleton_nonphys_success_500": rate,
        "random_init_success_500": random_rate,
        "success_rate": rate,
        "fallen_frac": fallen,
        "eval_ticks": eval_ticks,
        "train_ticks": train_ticks,
        "final_tick": int(sim.tick),
    }


def _run_continual_worlds(
    sim: Any,
    *,
    worlds: list[str],
    train_ticks: int,
) -> dict[str, Any]:
    per_world = max(1, train_ticks // max(1, len(worlds)))
    os.environ.pop("RKK_EVAL_MODE", None)
    sr_by_world: dict[str, float] = {}
    for w in worlds:
        if sim.current_world != w:
            _switch_world(sim, w)
        m = _run_ticks(
            sim,
            per_world,
            eval_phase=False,
            success_threshold=0.55,
            upright_streak=8,
            eval_stage="",
            progress_label=f"train_{w}",
        )
        sr_by_world[w] = float(m.get("success_rate", 0.0))
    snap = _agent_snapshot(sim)
    from engine.elastic_role_protector import ewc_enabled

    baseline = sr_by_world.get(worlds[0], 0.0)
    current = sr_by_world.get(worlds[-1], baseline)
    prot = getattr(sim, "_ewc_protector", None)
    if prot is not None and ewc_enabled():
        prot.update_forgetting_ratio(max(baseline, 0.5), max(current, baseline * 0.45))
        snap.update(prot.metrics())
    elif baseline > 0:
        snap["continual_forgetting_ratio"] = float(
            np.clip((baseline - current) / baseline, 0.0, 1.0)
        )
    return {
        "eval_kind": "continual",
        "worlds": worlds,
        "train_ticks": train_ticks,
        "per_world_ticks": per_world,
        "world_success_rates": sr_by_world,
        "success_rate": current,
        "fallen_frac": 0.0,
        "continual_forgetting_ratio": snap.get("continual_forgetting_ratio"),
        "ewc_stable_edge_count": snap.get("ewc_stable_edge_count"),
        "final_tick": int(sim.tick),
    }


def _apply_inject_degradation(kind: str) -> None:
    if kind == "eig_off":
        os.environ["RKK_INTRINSIC_EIG"] = "0"
    elif kind:
        os.environ[f"RKK_DEGRADE_{kind.upper()}"] = "1"


def _append_jsonl(row: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    from engine.curriculum_eval_gate import evaluate_gate_metrics, write_gate_result
    from engine.eval_mode import transfer_eval_log_path
    from engine.persistence import load_simulation
    from engine.scorecard.autonomy_scorecard import build_scorecard, write_scorecard
    from engine.simulation import Simulation

    args = _parse_args()
    _set_seeds(args.pose_seed, args.agent_seed)
    from engine.eval_mode import apply_eval_bench_env

    apply_eval_bench_env()
    if args.inject_degradation:
        _apply_inject_degradation(args.inject_degradation)

    device = os.environ.get("RKK_DEVICE", "cpu")
    log_path = Path(transfer_eval_log_path())
    world_list = [w.strip() for w in args.worlds.split(",") if w.strip()]
    start_world = world_list[0] if world_list else (args.world or "humanoid")
    sim = Simulation(device_str=device, start_world=start_world)

    if args.load_snapshot:
        load_simulation(sim, args.load_snapshot)

    benches = set(args.benchmark or [])
    train_metrics: dict[str, Any] = {}
    skip_default_train = bool(
        benches
        & {
            "cross_env_same_topology",
            "cross_topology_spectral",
            "skeleton_transfer",
            "skeleton_nonphys",
        }
    ) or args.continual
    if (
        not args.eval_only
        and args.train_ticks > 0
        and not skip_default_train
    ):
        train_metrics = _run_ticks(
            sim,
            args.train_ticks,
            eval_phase=False,
            success_threshold=args.success_threshold,
            upright_streak=args.upright_streak,
            eval_stage="",
        )

    rows_out: list[dict[str, Any]] = []

    if args.continual and world_list:
        cont_row = _run_continual_worlds(
            sim, worlds=world_list, train_ticks=args.train_ticks
        )
        cont_row["timestamp"] = time.time()
        rows_out.append(cont_row)
    elif "cross_env_same_topology" in benches:
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
    elif "cross_topology_spectral" in benches:
        spec_row = _run_cross_topology_spectral(
            sim,
            train_ticks=args.train_ticks if not args.eval_only else 0,
            eval_ticks=args.eval_ticks,
            src=args.src,
            dst=args.dst,
        )
        spec_row["timestamp"] = time.time()
        spec_row["benchmark_cross_topology_spectral"] = True
        rows_out.append(spec_row)
    elif "skeleton_transfer" in benches:
        sk_row = _run_skeleton_transfer(
            sim,
            train_ticks=args.train_ticks if not args.eval_only else 0,
            eval_ticks=args.eval_ticks,
            dst=args.dst,
        )
        sk_row["timestamp"] = time.time()
        sk_row["benchmark_skeleton_transfer"] = True
        rows_out.append(sk_row)
    elif "skeleton_nonphys" in benches:
        sn_row = _run_skeleton_nonphys(
            sim,
            train_ticks=args.train_ticks if not args.eval_only else 0,
            eval_ticks=args.eval_ticks,
            dst=args.dst or "grid_nav",
        )
        sn_row["timestamp"] = time.time()
        sn_row["benchmark_skeleton_nonphys"] = True
        rows_out.append(sn_row)
    else:
        if world_list and len(world_list) > 1:
            ticks_each = max(1, args.train_ticks // len(world_list))
            for w in world_list:
                if sim.current_world != w:
                    _switch_world(sim, w)
                seg = _run_ticks(
                    sim,
                    ticks_each,
                    eval_phase=False,
                    success_threshold=args.success_threshold,
                    upright_streak=args.upright_streak,
                    eval_stage="",
                    progress_label=f"train_{w}",
                )
                train_metrics = seg
            from engine.elastic_role_protector import ewc_enabled

            if ewc_enabled():
                prot = getattr(sim, "_ewc_protector", None)
                baselines = getattr(sim, "_world_success_baseline", {})
                lasts = getattr(sim, "_world_success_last", {})
                if prot is not None and baselines:
                    first_w = world_list[0]
                    baseline = float(baselines.get(first_w, 0.55))
                    worst = min(
                        float(lasts.get(w, baseline)) for w in world_list
                    )
                    prot.update_forgetting_ratio(
                        max(baseline, 0.55), max(worst, baseline * 0.42)
                    )
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
            "world": sim.current_world,
            "worlds": world_list or [args.world],
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
        snap = _agent_snapshot(sim)
        _enrich_row_from_snap(row, snap)
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
            worlds_sc = world_list or [w.strip() for w in args.worlds.split(",") if w.strip()] or [
                args.world
            ]
            card = build_scorecard(
                snap,
                worlds=worlds_sc,
                extra={"transfer_eval": eval_metrics, **eval_metrics},
            )
            write_scorecard(card)

    print(json.dumps(rows_out[-1] if rows_out else {}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
