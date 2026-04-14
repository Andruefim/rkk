"""Сборка WS/HTTP снимка состояния (вынесено из Simulation для SRP)."""
from __future__ import annotations

from typing import Any

from engine.core.constants import (
    agent_loop_hz_from_env as _agent_loop_hz_from_env,
    cpg_loop_hz_from_env as _cpg_loop_hz_from_env,
    l3_loop_hz_from_env as _l3_loop_hz_from_env,
    l4_worker_enabled as _l4_worker_enabled,
)
from engine.core.world import WORLDS
from engine.features.simulation.imports import (
    _INNER_VOICE_AVAILABLE,
    _PHASE_K_AVAILABLE,
    _PHASE_M_AVAILABLE,
    _RSSM_AVAILABLE,
    _TIMESCALE_AVAILABLE,
    _VERBAL_AVAILABLE,
    _WORLD_BRIDGE_AVAILABLE,
    rssm_enabled,
    timescale_enabled,
)


def build_simulation_snapshot(
    sim: Any,
    snap: dict,
    graph_deltas: dict,
    smoothed: float,
    scene: dict,
) -> dict:
    winfo = WORLDS.get(sim.current_world, {"color": "#cc44ff", "label": sim.current_world})

    vision_summary = None
    if sim._visual_mode and sim._visual_env is not None:
        vision_summary = sim._visual_env.cortex.snapshot()

    return {
        "tick": sim.tick,
        "phase": sim.phase,
        "max_phase": sim.max_phase,
        "entropy": round((1 - snap.get("peak_discovery_rate", 0)) * 100, 1),
        "smoothed_dr": round(smoothed, 3),
        "agents": [snap],
        "n_agents": 1,
        "demon": sim.demon.snapshot,
        "tom_links": [],
        "events": list(sim.events),
        "graph_deltas": graph_deltas,
        "value_layer": {
            "total_blocked_all": snap.get("total_blocked", 0),
            "block_rates": [round(snap.get("value_layer", {}).get("block_rate", 0), 3)],
            "imagination_horizon": snap.get("value_layer", {}).get("imagination_horizon", 0),
            "imagination_checks": snap.get("value_layer", {}).get("imagination_checks", 0),
            "imagination_blocks": snap.get("value_layer", {}).get("imagination_blocks", 0),
        },
        "byzantine": None,
        "motif": None,
        "multiprocess": False,
        "singleton": True,
        "current_world": sim.current_world,
        "world_label": winfo["label"],
        "world_color": winfo["color"],
        "worlds": WORLDS,
        "switch_history": sim.switcher.history[-5:],
        "gnn_d": sim.agent.graph._d,
        "fallen": snap.get("fallen", False),
        "fall_count": snap.get("fall_count", 0),
        "fall_recovery": {
            "active": bool(sim._fall_recovery_active),
            "start_tick": int(sim._fall_recovery_start_tick),
            "last_progress_tick": int(sim._fall_recovery_last_progress_tick),
            "best_score": round(float(sim._fall_recovery_best_score), 4),
        },
        "fixed_root": sim._fixed_root_active,
        "scene": scene,
        "visual_mode": sim._visual_mode,
        "vision_ticks": sim._vision_ticks,
        "vision": vision_summary,
        "llm_loop": {
            "enabled": sim._llm_loop_enabled(),
            "level2_inflight": sim._llm_level2_inflight,
            "pending_bundle": sim._pending_llm_bundle is not None,
            "last_schedule_tick": sim._last_level2_schedule_tick,
            "last_dr_gain_tick": sim._last_dr_gain_tick,
            "rolling_block_rate": round(sim._rolling_block_rate(), 4),
            "stats": dict(sim._llm_loop_stats),
        },
        "agent_loop": {
            "hz": round(_agent_loop_hz_from_env(), 1),
            "decoupled": _agent_loop_hz_from_env() > 0.0,
            "l3_hz": round(_l3_loop_hz_from_env(), 1),
            "l3_last_tick": int(sim._l3_last_tick),
            "l4_worker": bool(_l4_worker_enabled()),
            "l4_pending": bool(sim._l4_task_pending),
            "l4_last_submit_tick": int(sim._l4_last_submit_tick),
            "l4_last_apply_tick": int(sim._l4_last_apply_tick),
            "l1_last_cmd_tick": int(sim._l1_last_cmd_tick),
            "l1_last_apply_tick": int(sim._l1_last_apply_tick),
        },
        "locomotion": (
            {
                **sim._locomotion_controller.snapshot(),
                "decoupled_loop_hz": round(_cpg_loop_hz_from_env(), 1)
                if sim._cpg_decoupled_enabled()
                else 0.0,
            }
            if sim._locomotion_controller is not None
            else None
        ),
        "motor_state": sim._motor_state_snapshot(),
        "skills": sim._skill_snapshot(),
        "rsi_full": sim._rsi_full.snapshot()
        if sim._rsi_full_enabled() and sim._rsi_full is not None
        else None,
        "motor_cortex": (
            sim._motor_cortex.snapshot() if sim._motor_cortex is not None else None
        ),
        "concepts": [
            {
                "id": c["id"],
                "pattern": c["pattern"],
                "uses": c["uses"],
                "alpha_mean": c["alpha_mean"],
                "graph_node": c.get("graph_node"),
            }
            for c in sim._concepts_cache
        ],
        "memory": sim._memory_snapshot_meta(),
        "embodied_reward": None,
        "visual_grounding": (
            sim._visual_grounding_ctrl.snapshot()
            if sim._visual_grounding_ctrl is not None
            else None
        ),
        "episodic_memory": (
            sim._episodic_memory.snapshot()
            if sim._episodic_memory is not None
            else None
        ),
        "curriculum": (
            sim._curriculum.snapshot() if sim._curriculum is not None else None
        ),
        "rssm": (
            sim._rssm_trainer.snapshot()
            if sim._rssm_trainer is not None
            else {"enabled": rssm_enabled() if _RSSM_AVAILABLE else False}
        ),
        "proprioception": (
            sim._proprio.snapshot() if sim._proprio is not None else None
        ),
        "reward_coordinator": {"enabled": False, "replaced_by": "intrinsic_objective"},
        "intrinsic_objective": (
            sim._intrinsic.snapshot()
            if getattr(sim, "_intrinsic", None) is not None
            else None
        ),
        "timescale": (
            sim._timescale.snapshot()
            if sim._timescale is not None
            else (
                {"enabled": timescale_enabled()}
                if _TIMESCALE_AVAILABLE
                else {"enabled": False}
            )
        ),
        "inner_voice": (
            sim._inner_voice.snapshot()
            if sim._inner_voice is not None
            else {"enabled": _INNER_VOICE_AVAILABLE}
        ),
        "llm_teacher": (
            sim._llm_teacher.snapshot()
            if sim._llm_teacher is not None
            else {"enabled": False}
        ),
        "sleep": (
            sim._sleep_ctrl.snapshot()
            if _PHASE_K_AVAILABLE and sim._sleep_ctrl is not None
            else {"enabled": False}
        ),
        "physical_curriculum": (
            sim._physical_curriculum.snapshot()
            if _PHASE_K_AVAILABLE and sim._physical_curriculum is not None
            else None
        ),
        "persistence": (
            sim._persist.snapshot()
            if _PHASE_K_AVAILABLE and sim._persist is not None
            else None
        ),
        "verbal": (
            sim._verbal.snapshot()
            if _VERBAL_AVAILABLE and sim._verbal is not None
            else {"enabled": False}
        ),
        "visual_voice": (
            sim._visual_voice.snapshot()
            if _PHASE_M_AVAILABLE and sim._visual_voice is not None
            else {"enabled": False}
        ),
        "slot_labeler": (
            sim._slot_labeler.snapshot()
            if _PHASE_M_AVAILABLE and sim._slot_labeler is not None
            else None
        ),
        "world_bridge": (
            sim._world_bridge.snapshot()
            if _WORLD_BRIDGE_AVAILABLE and sim._world_bridge is not None
            else {"enabled": False}
        ),
        "phase1": sim._phase1_snapshot_meta(),
        "phase2": sim._phase2_snapshot_meta(),
    }
