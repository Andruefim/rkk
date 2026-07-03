"""Integration: human command → task binding → WM / motor / verbal REPORT (mock, no PyBullet)."""
from __future__ import annotations

import os
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

from engine.grounded_language import sensory_node_ids
from engine.motor_arbiter import MotorArbiter
from engine.system2.controller import System2Controller
from engine.system2.wm_planner import task_from_planning_context
from tests.conftest import AgiLoopSim, _default_humanoid_obs


def _patch_fallback_embed(sim: AgiLoopSim) -> None:
    """Deterministic embed (no Ollama) for ingest_command."""
    sim._ensure_grounded_language()
    gl = sim._grounded_lang
    assert gl is not None
    fake = np.random.RandomState(42).randn(64).astype(np.float32)
    fake /= np.linalg.norm(fake) + 1e-9
    gl.embedder.embed = lambda _t: fake  # type: ignore[method-assign]
    gl.store.add("подойди", "approach", fake)


def test_command_ingest_bind_writes_wm(agi_loop_sim: AgiLoopSim) -> None:
    """handle_human_command: ingest → bind → non-empty expected_state + WM slots."""
    sim = agi_loop_sim
    _patch_fallback_embed(sim)

    out = sim.handle_human_command("подойди ближе")

    assert out["ok"] is True
    assert out.get("task_binding") is True
    task = out.get("task") or {}
    assert task.get("n_expected_keys", 0) > 0
    assert task.get("expected_state")

    wm = sim._system2.working_memory
    assert wm.has("human_task_active")
    assert wm.read("human_task_active") == 1.0
    assert wm.has("human_task_pe")

    tb = sim._task_binding
    assert tb is not None
    active = tb.active_task
    assert active is not None
    assert active.text.startswith("подойди")
    for nid in sensory_node_ids(8):
        assert nid in sim.agent.graph.nodes


def test_pe_verify_improves_toward_expected_state(agi_loop_sim: AgiLoopSim) -> None:
    """bind_command + verify: far obs → high PE; close obs → success path."""
    sim = agi_loop_sim
    _patch_fallback_embed(sim)
    sim.handle_human_command("иди вперёд")

    tb = sim._ensure_task_binding()
    task = tb.active_task
    assert task is not None
    task.expected_state = {"intent_stride": 0.66, "intent_torso_forward": 0.58}
    task.max_prediction_error = 0.2
    expected = dict(task.expected_state)

    far_obs = dict(sim._obs)
    for k, tgt in expected.items():
        far_obs[k] = float(tgt) + 0.45
    far_obs.setdefault("intero_energy", 0.85)
    far_obs.setdefault("intero_stress", 0.1)

    ok_far, pe_far, _ = tb.verify(far_obs, task)
    assert pe_far > 0.15
    assert not ok_far

    close_obs = dict(far_obs)
    for k, tgt in expected.items():
        close_obs[k] = float(tgt)

    ok_close, pe_close, diag = tb.verify(close_obs, task)
    assert pe_close < pe_far
    assert ok_close
    assert pe_close <= float(diag.get("max_pe", 1.0))


def test_wm_planner_receives_expected_state_from_command(agi_loop_sim: AgiLoopSim) -> None:
    """S2 planning_context + WM task carry human-task expected_state after command."""
    sim = agi_loop_sim
    _patch_fallback_embed(sim)
    sim.handle_human_command("осмотрись")

    ic = sim._intention_cortex
    ic.absorb_human_task(sim._task_binding.active_task, sim._obs, sim.tick)
    ctx = ic.tick_pre_control(sim, tick=sim.tick + 1, obs=sim._obs, fallen=False)
    sim._intention_state = ctx

    s2 = System2Controller()
    s2._wm = sim._system2.working_memory
    plan_ctx = s2.planning_context_for_wm(fallen=False, sim_tick=sim.tick, sim=sim)

    assert plan_ctx.get("human_task_active") is True
    assert plan_ctx.get("skill_id") == "human_command"
    assert plan_ctx.get("macro") in ("EXPLORE", "LOCOMOTE_DELIVERY")
    es = plan_ctx.get("expected_state") or {}
    assert es
    active = sim._task_binding.active_task
    assert active is not None
    for k in active.expected_state:
        assert k in es

    wm_task = task_from_planning_context(plan_ctx, sim.agent.graph.nodes)
    assert wm_task.expected_state
    assert wm_task.macro in ("EXPLORE", "LOCOMOTE_DELIVERY")
    assert any(k in wm_task.expected_state for k in active.expected_state)


def test_intention_registers_motor_arbiter_intent(agi_loop_env: None) -> None:
    """Human command with intent delta → motor arbiter receives register_from_dict."""
    obs = _default_humanoid_obs()
    obs["intent_stride"] = 0.2
    sim = AgiLoopSim(obs=obs, tick=200)
    _patch_fallback_embed(sim)
    sim.handle_human_command("иди вперёд")

    task = sim._task_binding.active_task
    assert task is not None
    task.expected_state = {"intent_stride": 0.85}

    sim._motor_arbiter = MotorArbiter()
    sim._motor_arbiter.begin_tick()
    ic = sim._intention_cortex
    ic.absorb_human_task(task, obs, sim.tick)
    primary = ic._stack[0]
    ctx = ic._last_context
    ctx.macro_hint = "EXPLORE"
    ic._project_intent_motor(sim.agent, ctx, primary, fallen=False, sim=sim)

    assert len(sim._motor_arbiter._intents) >= 1
    srcs = {i.source for i in sim._motor_arbiter._intents}
    assert "intention_cortex" in srcs
    assert primary.source == "human_command"


def test_task_done_emits_verbal_report(agi_loop_sim: AgiLoopSim) -> None:
    """tick_verify success → _emit_task_report appends SpeechType.REPORT."""
    from engine.verbal_action import SpeechType

    sim = agi_loop_sim
    _patch_fallback_embed(sim)
    sim.handle_human_command("готово")

    tb = sim._ensure_task_binding()
    task = tb.active_task
    assert task is not None
    task.tick_started = 10
    task.expected_state = {"target_dist": 0.35, "posture_stability": 0.75}
    task.max_prediction_error = 0.25

    verbal = SimpleNamespace(
        _messages=[],
        _on_message=[],
        total_messages=0,
        _last_report_tick=-1,
    )
    sim._verbal = verbal
    sim.grounded_lang_generate = lambda obs=None: "Готово."  # type: ignore[method-assign]

    match_obs = dict(sim._obs)
    for k, tgt in task.expected_state.items():
        match_obs[k] = float(tgt)
    match_obs.setdefault("intero_energy", 0.9)
    match_obs.setdefault("intero_stress", 0.05)
    sim.set_obs(match_obs)

    with mock.patch("engine.verbal_action.ollama_chat_speech_enabled", return_value=False):
        sim.tick = 80
        sim._tick_human_task(fallen=False)

    assert task.status == "done"
    assert len(verbal._messages) == 1
    msg = verbal._messages[0]
    assert msg.speech_type == SpeechType.REPORT
    assert "HUMAN_TASK" in msg.concepts
    assert verbal.total_messages == 1

    wm = sim._system2.working_memory
    assert wm.read("human_task_active") == 0.0
    assert wm.read("human_task_status") == 1.0


@pytest.mark.skipif(
    os.environ.get("RKK_RUN_PYBULLET_E2E", "0").strip() not in ("1", "true", "yes"),
    reason="set RKK_RUN_PYBULLET_E2E=1 for full PyBullet e2e",
)
def test_pybullet_human_command_smoke() -> None:
    """Optional full-stack smoke (PyBullet); skipped by default."""
    from engine.features.simulation.simulation_main import Simulation

    with mock.patch.dict(
        os.environ,
        {"RKK_GROUNDED_LANG": "1", "RKK_TASK_BINDING": "1"},
        clear=False,
    ):
        sim = Simulation()
        out = sim.handle_human_command("тест")
    assert isinstance(out, dict)
