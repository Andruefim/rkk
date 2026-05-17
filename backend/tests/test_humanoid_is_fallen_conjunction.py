"""is_fallen debounce requires normalized gate AND raw com_z < FALLEN_Z."""
from __future__ import annotations

import pytest

from engine.features.humanoid.environment import EnvironmentHumanoid


@pytest.fixture
def env() -> EnvironmentHumanoid:
    return EnvironmentHumanoid()


def test_is_fallen_streak_requires_raw_below(monkeypatch: pytest.MonkeyPatch, env: EnvironmentHumanoid) -> None:
    monkeypatch.setenv("RKK_FALLEN_CONFIRM_TICKS", "2")
    env._fixed_root = False
    monkeypatch.setattr(env, "_fallen_z_below_threshold", lambda: True)
    monkeypatch.setattr(env, "_com_z_raw_below_fallen", lambda: False)
    assert env.is_fallen() is False
    assert env.is_fallen() is False
    assert env._fallen_low_z_streak == 0


def test_is_fallen_streak_when_conjunction(monkeypatch: pytest.MonkeyPatch, env: EnvironmentHumanoid) -> None:
    monkeypatch.setenv("RKK_FALLEN_CONFIRM_TICKS", "2")
    env._fixed_root = False
    monkeypatch.setattr(env, "_fallen_z_below_threshold", lambda: True)
    monkeypatch.setattr(env, "_com_z_raw_below_fallen", lambda: True)
    assert env.is_fallen() is False
    assert env._fallen_low_z_streak == 1
    assert env.is_fallen() is True
    assert env._fallen_low_z_streak >= 2


def test_is_fallen_fixed_root_clears_streak(env: EnvironmentHumanoid) -> None:
    env._fallen_low_z_streak = 5
    assert env.is_fallen() is False
    assert env._fallen_low_z_streak == 0


def test_effective_fallen_z_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    from engine.features.humanoid import environment as env_mod
    from engine.features.humanoid.constants import FALLEN_Z

    monkeypatch.setenv("RKK_FALLEN_Z", "0.24")
    assert abs(env_mod.effective_fallen_z_m() - 0.24) < 1e-9
    monkeypatch.delenv("RKK_FALLEN_Z", raising=False)
    assert abs(env_mod.effective_fallen_z_m() - float(FALLEN_Z)) < 1e-9
