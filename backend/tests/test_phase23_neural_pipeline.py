"""Phase 2–3: latent slot re-ID + WM/Active Inference navigation."""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from engine.object_working_memory import (
    LatentSceneMemory,
    SceneEntity,
    latent_cosine,
    match_latent_slot,
)
from engine.vision_target import VisualTarget, sim_oracle_bind_enabled


def test_latent_cosine_match_and_reject() -> None:
    a = [1.0, 0.0, 0.0, 0.5]
    b = [1.0, 0.0, 0.0, 0.5]
    c = [0.0, 1.0, 0.0, 0.0]
    assert latent_cosine(a, b) == pytest.approx(1.0, abs=1e-5)
    assert latent_cosine(a, c) < 0.2


def test_match_latent_slot_survives_slot_id_permutation() -> None:
    query = [0.9, 0.1, 0.2, 0.3]
    candidates = [
        {"slot_id": "slot_7", "vector": [0.0, 1.0, 0.0, 0.0], "range_m": 2.0},
        {"slot_id": "slot_2", "vector": [0.9, 0.1, 0.2, 0.3], "range_m": 1.8},
        {"slot_id": "slot_0", "vector": [0.1, 0.8, 0.1, 0.0], "range_m": 3.0},
    ]
    hit = match_latent_slot(candidates, query, min_cos=0.9)
    assert hit is not None
    assert hit["slot_id"] == "slot_2"
    assert float(hit["latent_cos"]) >= 0.9

    miss = match_latent_slot(
        [{"slot_id": "x", "vector": [0.0, 1.0, 0.0, 0.0], "range_m": 1.0}],
        query,
        min_cos=0.7,
    )
    assert miss is None


def test_scene_memory_latent_reid_under_hard_lock(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RKK_LATENT_REID_MIN_COS", "0.7")
    scene = LatentSceneMemory()
    ent = SceneEntity(entity_id="bound_0")
    ent.seed_from_bearing_range(
        bearing=0.1,
        range_m=2.2,
        tick=1,
        label="chair",
        confidence=0.8,
        activation=0.8,
        slot_id="slot_0",
        u=0.55,
        v=0.5,
        latent=[1.0, 0.0, 0.2, 0.1],
    )
    scene.entities["bound_0"] = ent
    scene.focus("bound_0", exclusive=True)
    scene.hard_lock_active = True
    scene._prev_xy = (0.0, 0.0)
    scene._prev_fwd = (1.0, 0.0)
    assert scene.hard_lock_active is True

    # SlotAttention permutes: same latent now lives on slot_5.
    scene.update(
        tick=2,
        agent_xy=(0.0, 0.0),
        agent_forward=(1.0, 0.0),
        percepts=[
            {
                "slot_id": "slot_5",
                "bearing": 0.12,
                "range_m": 2.15,
                "u": 0.56,
                "v": 0.5,
                "confidence": 0.75,
                "activation": 0.75,
                "label": "chair",
                "vector": [1.0, 0.0, 0.2, 0.1],
            }
        ],
    )
    act = scene.active()
    assert act is not None
    assert act.diagnostics.get("source") == "hard_lock_latent_reid"
    assert float(act.diagnostics.get("latent_cos") or 0.0) >= 0.7
    assert len(act.latent) == 4


def test_bind_persists_latent_from_visual_target() -> None:
    scene = LatentSceneMemory()
    vt = VisualTarget(
        slot_id="slot_3",
        u=0.5,
        v=0.5,
        label="box",
        confidence=0.7,
        bearing=0.0,
        range_m=1.5,
        range_conf=0.9,
        latent=[0.2, 0.4, 0.6, 0.8],
        diagnostics={"latent": [0.2, 0.4, 0.6, 0.8]},
    )
    scene.bind_visual_target(
        vt, tick=5, agent_xy=(0.0, 0.0), agent_forward=(1.0, 0.0)
    )
    act = scene.active()
    assert act is not None
    assert act.latent == pytest.approx([0.2, 0.4, 0.6, 0.8])


def test_sim_oracle_bind_defaults_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RKK_SIM_ORACLE_BIND", raising=False)
    assert sim_oracle_bind_enabled() is False


def test_resolve_prefers_latent_reid_over_oracle(monkeypatch: pytest.MonkeyPatch) -> None:
    """With oracle off, uncertain resolve uses latent re-ID before giving up."""
    monkeypatch.setenv("RKK_TASK_RESOLVE", "vision")
    monkeypatch.setenv("RKK_SIM_ORACLE_BIND", "0")
    monkeypatch.setenv("RKK_VISION_ACTIVE_PERCEPT", "0")
    monkeypatch.setenv("RKK_LATENT_REID_MIN_COS", "0.5")

    from engine.features.simulation.mixin_grounded_language import (
        SimulationGroundedLanguageMixin,
    )

    class _Harness(SimulationGroundedLanguageMixin):
        def __init__(self) -> None:
            self.tick = 10
            self._latent_scene = LatentSceneMemory()
            ent = SceneEntity(entity_id="bound_0")
            ent.seed_from_bearing_range(
                bearing=0.0,
                range_m=2.0,
                tick=1,
                label="chair",
                confidence=0.8,
                slot_id="slot_0",
                latent=[1.0, 0.0, 0.0, 0.0],
            )
            self._latent_scene.entities["bound_0"] = ent
            self._latent_scene.focus("bound_0", exclusive=True)
            self._manip_resolved_visual = VisualTarget(
                slot_id="slot_0",
                u=0.5,
                v=0.5,
                label="chair",
                confidence=0.8,
                bearing=0.0,
                range_m=2.0,
                range_conf=0.9,
                latent=[1.0, 0.0, 0.0, 0.0],
            )
            self._oracle_called = False

        def _latent_scene_memory(self) -> LatentSceneMemory:
            return self._latent_scene

        def _depth_camera_from_sim(self) -> Any:
            return object()

        def _visual_env_ref(self) -> Any:
            return object()

        def _try_sim_oracle_visual_bind(self, *args: Any, **kwargs: Any):
            self._oracle_called = True
            return None, None, {"reason": "should_not_run"}

    h = _Harness()

    def _fake_resolve(*args: Any, **kwargs: Any):
        return None, {"reason": "uncertain_no_peaked_slot", "resolve_mode": "vision"}

    def _fake_latent_bind(*, vision_diag=None, reason=""):
        vt = VisualTarget(
            slot_id="slot_4",
            u=0.52,
            v=0.5,
            label="chair",
            confidence=0.7,
            bearing=0.04,
            range_m=1.9,
            range_conf=0.8,
            latent=[1.0, 0.0, 0.0, 0.0],
            diagnostics={"source": "vision_latent_reid", "latent_cos": 0.99},
        )
        return vt, {
            "reason": "ok_latent_reid",
            "source": "vision_latent_reid",
            "latent_cos": 0.99,
        }

    monkeypatch.setattr(
        "engine.features.simulation.mixin_grounded_language.resolve_visual_target",
        _fake_resolve,
    )
    monkeypatch.setattr(h, "_try_latent_reid_visual_bind", _fake_latent_bind)
    monkeypatch.setattr(h, "_slot_concept_project_fn", lambda: None)
    monkeypatch.setattr(h, "_visual_mode", True, raising=False)

    # Avoid enable_visual / refresh side effects
    monkeypatch.setattr(h, "enable_visual", None, raising=False)

    _oracle, vt, diag = h._resolve_command_target(
        "approach the chair",
        embed_fn=lambda _t: None,
        require_movable=False,
        interaction_kinds=frozenset(),
    )
    assert vt is not None
    assert vt.slot_id == "slot_4"
    assert diag.get("source") == "vision_latent_reid"
    assert h._oracle_called is False


def test_wm_ai_nav_returns_intents_and_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RKK_TASK_NAV_MODE", "wm_ai")
    monkeypatch.setenv("RKK_ACTIVE_INFERENCE", "1")
    monkeypatch.setenv("RKK_TASK_NAV_WM_MIN_STEPS", "1")

    from engine.features.simulation.mixin_grounded_language import (
        SimulationGroundedLanguageMixin,
    )
    from engine.object_working_memory import ObjectWorkingMemory

    class _FakeCtrl:
        def optimize_action(self, *_a: Any, **_k: Any) -> dict[str, float]:
            return {
                "intent_gait_coupling": 0.72,
                "intent_stride": 0.61,
            }

    class _Harness(SimulationGroundedLanguageMixin):
        def __init__(self) -> None:
            self.tick = 20
            self._homeostatic_ctrl = _FakeCtrl()
            self.agent = SimpleNamespace(
                graph=SimpleNamespace(
                    nodes={
                        "vision_bearing": 0.0,
                        "vision_range_m": 2.0,
                        "task_target_dist_m": 2.0,
                    },
                    _core=SimpleNamespace(device="cpu"),
                    _wm_train_calls=12,
                    snapshot_vec_dict=lambda: {
                        "vision_bearing": 0.2,
                        "vision_range_m": 2.0,
                        "task_target_dist_m": 2.0,
                        "phys_posture_stability": 0.8,
                        "phys_com_z": 0.8,
                        "phys_com_x_vel": 0.1,
                    },
                )
            )

        def _ensure_homeostatic_ctrl(self):
            return self._homeostatic_ctrl

        def _wm_train_steps(self) -> int:
            return 12

    h = _Harness()
    vt = VisualTarget(
        slot_id="nav_t",
        u=0.6,
        v=0.5,
        label="chair",
        confidence=0.85,
        bearing=0.25,
        range_m=2.4,
        range_conf=0.9,
    )
    owm = ObjectWorkingMemory()
    owm.bind_from_visual(
        vt, tick=20, agent_xy=(0.0, 0.0), agent_forward=(1.0, 0.0)
    )
    owm.scene.release_hard_lock()

    intents, meta = h._navigation_intents_wm_ai(owm, stop=0.7, posture=0.9, fallen=False)
    assert intents
    assert "intent_gait_coupling" in intents
    assert "intent_stride" in intents
    assert meta["nav_ai_ok"] is True
    assert meta["task_nav_mode"] == "wm_ai"

    # Empty AI → heuristic fallback still has coupling/stride.
    class _EmptyCtrl:
        def optimize_action(self, *_a: Any, **_k: Any) -> dict[str, float]:
            return {}

    h._homeostatic_ctrl = _EmptyCtrl()
    intents2, meta2 = h._navigation_intents_wm_ai(owm, stop=0.7, posture=0.9, fallen=False)
    assert intents2
    assert "intent_gait_coupling" in intents2
    assert meta2["nav_ai_ok"] is False
    assert "fallback" in str(meta2.get("nav_ai_reason") or "")

    # Fallen → empty
    intents3, meta3 = h._navigation_intents_wm_ai(owm, stop=0.7, posture=0.9, fallen=True)
    assert intents3 == {}
    assert meta3["nav_ai_ok"] is False


def test_wm_ai_assert_forward_when_aligned(monkeypatch: pytest.MonkeyPatch) -> None:
    """Aligned + far → floor weak AI stride toward heuristic forward."""
    monkeypatch.setenv("RKK_TASK_NAV_MODE", "wm_ai")
    monkeypatch.setenv("RKK_ACTIVE_INFERENCE", "1")
    monkeypatch.setenv("RKK_TASK_NAV_WM_MIN_STEPS", "0")
    monkeypatch.setenv("RKK_NAV_ALIGN_BEARING", "0.40")
    monkeypatch.setenv("RKK_NAV_FWD_RANGE_MARGIN", "0.12")
    monkeypatch.setenv("RKK_NAV_ALIGNED_STRIDE_FLOOR", "0.62")
    monkeypatch.setenv("RKK_NAV_ALIGNED_FWD_BLEND", "0.80")

    from engine.features.simulation.mixin_grounded_language import (
        SimulationGroundedLanguageMixin,
    )
    from engine.object_working_memory import ObjectWorkingMemory

    class _WeakCtrl:
        def optimize_action(self, *_a: Any, **_k: Any) -> dict[str, float]:
            # Near-neutral AI — historically plateaued approach ~1.9 m.
            return {
                "intent_gait_coupling": 0.50,
                "intent_stride": 0.48,
            }

    class _Harness(SimulationGroundedLanguageMixin):
        def __init__(self) -> None:
            self.tick = 20
            self._homeostatic_ctrl = _WeakCtrl()
            self.agent = SimpleNamespace(
                graph=SimpleNamespace(
                    nodes={
                        "vision_bearing": 0.0,
                        "vision_range_m": 1.9,
                        "task_target_dist_m": 1.9,
                    },
                    _core=SimpleNamespace(device="cpu"),
                    _wm_train_calls=0,
                    snapshot_vec_dict=lambda: {
                        "vision_bearing": 0.1,
                        "vision_range_m": 1.9,
                        "task_target_dist_m": 1.9,
                        "phys_posture_stability": 0.9,
                        "phys_com_z": 0.82,
                        "phys_com_x_vel": 0.05,
                    },
                )
            )

        def _ensure_homeostatic_ctrl(self):
            return self._homeostatic_ctrl

        def _wm_train_steps(self) -> int:
            return 0

    h = _Harness()
    vt = VisualTarget(
        slot_id="nav_align",
        u=0.55,
        v=0.5,
        label="ball",
        confidence=0.85,
        bearing=0.12,
        range_m=1.9,
        range_conf=0.9,
    )
    owm = ObjectWorkingMemory()
    owm.bind_from_visual(
        vt, tick=20, agent_xy=(0.0, 0.0), agent_forward=(1.0, 0.0)
    )
    owm.scene.release_hard_lock()

    intents, meta = h._navigation_intents_wm_ai(owm, stop=0.7, posture=0.9, fallen=False)
    assert intents
    assert meta.get("nav_fwd_assert") is True
    assert float(intents["intent_stride"]) >= 0.56
    assert float(intents["intent_stride"]) > 0.48

    # Large bearing → no forward assert floor.
    class _TurnCtrl:
        def optimize_action(self, *_a: Any, **_k: Any) -> dict[str, float]:
            return {"intent_gait_coupling": 0.72, "intent_stride": 0.50}

    h._homeostatic_ctrl = _TurnCtrl()
    vt2 = VisualTarget(
        slot_id="nav_turn",
        u=0.85,
        v=0.5,
        label="ball",
        confidence=0.85,
        bearing=0.72,
        range_m=1.9,
        range_conf=0.9,
    )
    owm2 = ObjectWorkingMemory()
    owm2.bind_from_visual(
        vt2, tick=21, agent_xy=(0.0, 0.0), agent_forward=(1.0, 0.0)
    )
    intents_t, meta_t = h._navigation_intents_wm_ai(
        owm2, stop=0.7, posture=0.9, fallen=False
    )
    assert meta_t.get("nav_fwd_assert") is not True
    assert abs(float(intents_t["intent_stride"]) - 0.50) < 0.02
