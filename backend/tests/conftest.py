"""
Pytest отключён по умолчанию, чтобы не мешать ручному прогону run.py / UI.

  RKK_RUN_TESTS=1 pytest
  RKK_RUN_TESTS=1 pytest tests/test_graph_perf.py -q
"""
from __future__ import annotations

import os

import pytest


def _tests_enabled() -> bool:
    return os.environ.get("RKK_RUN_TESTS", "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def pytest_collection_modifyitems(config, items) -> None:
    if _tests_enabled():
        return
    skip = pytest.mark.skip(
        reason="pytest off (set RKK_RUN_TESTS=1 to run backend tests)"
    )
    for item in items:
        item.add_marker(skip)
