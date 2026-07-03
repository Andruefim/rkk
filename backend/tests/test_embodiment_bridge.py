"""EmbodiedEnv protocol compliance and embodiment bridge (sim-to-real prep)."""
from __future__ import annotations

import torch

from engine.core.world import EmbodiedEnv, embodied_is_fallen
from engine.embodiment_bridge import (
    PassthroughEmbodimentBridge,
    StubRosEmbodimentBridge,
    wrap_embodiment,
)
from engine.environment_humanoid import EnvironmentHumanoid


def _make_fallback_humanoid() -> EnvironmentHumanoid:
    return EnvironmentHumanoid(device=torch.device("cpu"))


def test_humanoid_implements_embodied_env_protocol() -> None:
    env = _make_fallback_humanoid()
    assert isinstance(env, EmbodiedEnv)
    obs0 = env.reset()
    assert isinstance(obs0, dict)
    assert len(obs0) > 0
    obs1 = env.step({"intent_lean_forward": 0.55})
    assert isinstance(obs1, dict)
    assert len(obs1) > 0
    assert isinstance(embodied_is_fallen(env), bool)


def test_passthrough_bridge_matches_env() -> None:
    env = _make_fallback_humanoid()
    bridge = PassthroughEmbodimentBridge(env)
    assert bridge.env is env
    assert bridge.backend == "sim"
    obs_reset = bridge.reset()
    assert obs_reset
    obs_step = bridge.step({})
    assert isinstance(obs_step, dict)
    assert isinstance(bridge.is_fallen(), bool)


def test_wrap_embodiment_factory() -> None:
    env = _make_fallback_humanoid()
    sim_bridge = wrap_embodiment(env)
    assert isinstance(sim_bridge, PassthroughEmbodimentBridge)
    ros_bridge = wrap_embodiment(env, backend="ros_stub")
    assert isinstance(ros_bridge, StubRosEmbodimentBridge)
    assert ros_bridge.backend == "ros_stub"
