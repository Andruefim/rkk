"""Минимальные контракты для слабой связности подсистем."""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class GraphHolder(Protocol):
    """Всё, что даёт доступ к графу агента (снимки, узлы)."""

    @property
    def agent(self) -> Any: ...
