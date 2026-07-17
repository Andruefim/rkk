"""
Thin adapter between RKK and future hardware drivers (ROS, UDP, etc.).

Extension point: subclass :class:`EmbodimentBridge` and override
``_on_action_out`` / ``_on_observation_in`` for real robot I/O.
Simulation uses :class:`PassthroughEmbodimentBridge` (no transformation).
"""
from __future__ import annotations

from engine.core.world import EmbodiedEnv, embodied_is_fallen


class EmbodimentBridge:
    """Wraps any :class:`EmbodiedEnv`; hook point for sim-to-real."""

    def __init__(self, env: EmbodiedEnv, *, backend: str = "sim"):
        self._env = env
        self.backend = backend

    @property
    def env(self) -> EmbodiedEnv:
        return self._env

    def observe(self) -> dict[str, float]:
        obs = dict(self._env.observe())
        self._on_observation_in(obs)
        return obs

    def step(self, action: dict[str, float]) -> dict[str, float]:
        self._on_action_out(action)
        obs = dict(self._env.step(action))
        self._on_observation_in(obs)
        return obs

    def reset(self) -> dict[str, float]:
        obs = dict(self._env.reset())
        self._on_observation_in(obs)
        return obs

    def is_fallen(self) -> bool:
        return embodied_is_fallen(self._env)

    def get_scene_extras(self) -> dict:
        """Portable scene observation API for task target resolution."""
        env = self._env
        for name in (
            "get_scene_extras",
            "get_sandbox_scene_extras",
            "get_physics_object_positions",
        ):
            fn = getattr(env, name, None)
            if callable(fn):
                try:
                    return dict(fn() or {})
                except Exception:
                    continue
        sim = getattr(env, "_sim", None)
        if sim is not None and sim is not env:
            for name in (
                "get_scene_extras",
                "get_sandbox_scene_extras",
                "get_physics_object_positions",
            ):
                fn = getattr(sim, name, None)
                if callable(fn):
                    try:
                        return dict(fn() or {})
                    except Exception:
                        continue
        return {}

    def _on_action_out(self, action: dict[str, float]) -> None:
        """Override to publish commands to hardware (ROS topics, etc.)."""

    def _on_observation_in(self, obs: dict[str, float]) -> None:
        """Override to ingest sensor streams from hardware."""


class PassthroughEmbodimentBridge(EmbodimentBridge):
    """Default sim bridge — identity pass-through."""


class StubRosEmbodimentBridge(EmbodimentBridge):
    """Stub ROS bridge; hooks are no-ops until a real driver is wired."""

    def __init__(self, env: EmbodiedEnv):
        super().__init__(env, backend="ros_stub")


def wrap_embodiment(env: EmbodiedEnv, *, backend: str = "sim") -> EmbodimentBridge:
    """Factory used at simulation boot for humanoid topology worlds."""
    if backend == "ros_stub":
        return StubRosEmbodimentBridge(env)
    return PassthroughEmbodimentBridge(env, backend=backend)
