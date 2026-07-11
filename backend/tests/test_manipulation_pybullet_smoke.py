"""Optional PyBullet smoke for manipulation chair slice."""
from __future__ import annotations

import pytest

from engine.features.humanoid.deps import PYBULLET_AVAILABLE


@pytest.mark.skipif(not PYBULLET_AVAILABLE, reason="PyBullet not installed")
def test_manip_chair_registry_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RKK_MANIP_CHAIR", "1")
    from engine.features.humanoid.environment import EnvironmentHumanoid
    from engine.grounded_language import FallbackEmbeddingClient
    from engine.object_resolver import resolve_manipulation_target

    fb = FallbackEmbeddingClient(embed_dim=64)
    env = EnvironmentHumanoid(fixed_root=True)
    try:
        extras = env._sim.get_sandbox_scene_extras()
        assert any(r.get("ref") == "manip_chair_front" for r in extras.get("registry", []))
        raw = env._sim.get_state()
        agent_xy = (float(raw["com_x"]), float(raw["com_y"]))
        resolved, diag = resolve_manipulation_target(
            "передвинь стул перед тобой",
            extras,
            agent_xy=agent_xy,
            embed_fn=fb.embed,
        )
        assert resolved is not None, diag
        assert resolved.movable is True
        assert resolved.ref == "manip_chair_front"
    finally:
        del env


def test_fallback_manip_chair_push(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RKK_MANIP_CHAIR", "1")
    from engine.features.humanoid.fallback import _FallbackHumanoid
    from engine.manipulation_verify import ManipulationEpisode, verify_manipulation
    from engine.object_resolver import resolve_manipulation_target

    sim = _FallbackHumanoid(fixed_root=True)
    extras = sim.get_sandbox_scene_extras()
    resolved, _ = resolve_manipulation_target("move chair", extras, agent_xy=(0.0, 0.0))
    assert resolved is not None
    ep = ManipulationEpisode.begin(resolved)
    push = sim.apply_manipulation_push(9001, (1.0, 0.0), 80.0)
    assert push.get("applied") is True
    pose = sim.get_manipulation_target_pose("manip_chair_front")
    assert pose is not None
    out = verify_manipulation(ep, (float(pose["x"]), float(pose["y"])))
    assert out["displacement_m"] > 0.0
