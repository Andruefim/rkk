"""
Integration test fixtures (Phase 0+): CPU device, fixed seeds, gate env defaults.
"""
from __future__ import annotations

import os

import pytest


@pytest.fixture(scope="session", autouse=True)
def _rkk_integration_env() -> None:
    os.environ.setdefault("RKK_DEVICE", "cpu")
    os.environ.setdefault("RKK_SKIP_ALL_LLM", "1")
    os.environ.setdefault("RKK_AGENT_SEED", "42")
    os.environ.setdefault("RKK_POSE_SEED", "7")
    os.environ.setdefault("RKK_CURRICULUM_EVAL_GATE", "0")
    yield


@pytest.fixture
def integration_seeds() -> dict[str, int]:
    return {"agent_seed": 42, "pose_seed": 7}
