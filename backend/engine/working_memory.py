"""
Semantic working memory — prefrontal-style slot buffer for System 2 deliberation.

Unlike episodic memory (long-term structured episodes), WM holds active context
that S2 reads/writes during a task: current goal, plan head, inner-voice concepts,
and recent macro outcomes. ~16 slots with TTL decay.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any


def working_memory_enabled() -> bool:
    return os.environ.get("RKK_WORKING_MEMORY", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


def _wm_capacity() -> int:
    try:
        return max(4, min(32, int(os.environ.get("RKK_WM_SLOTS", "16"))))
    except ValueError:
        return 16


def _wm_default_ttl() -> int:
    try:
        return max(120, int(os.environ.get("RKK_WM_TTL_TICKS", "2400")))
    except ValueError:
        return 2400


def _wm_human_ttl() -> int:
    try:
        return max(_wm_default_ttl(), int(os.environ.get("RKK_WM_HUMAN_TTL_TICKS", "12000")))
    except ValueError:
        return 12000


def _is_human_task_key(key: str) -> bool:
    k = str(key)
    return k.startswith("human_task_") or k.startswith("goal_human")


@dataclass
class WmSlot:
    key: str
    value: float = 0.5
    text: str = ""
    tick_written: int = 0
    ttl_ticks: int = 2400
    source: str = ""

    def alive(self, tick: int) -> bool:
        return (tick - self.tick_written) <= self.ttl_ticks


class WorkingMemoryBuffer:
    """Fixed-capacity slot buffer with LRU eviction and TTL decay."""

    def __init__(self, capacity: int | None = None) -> None:
        self._capacity = capacity if capacity is not None else _wm_capacity()
        self._slots: dict[str, WmSlot] = {}
        self._order: list[str] = []

    def write(
        self,
        key: str,
        value: float,
        *,
        text: str = "",
        tick: int = 0,
        ttl_ticks: int | None = None,
        source: str = "",
    ) -> None:
        k = str(key).strip()
        if not k:
            return
        if ttl_ticks is None and _is_human_task_key(k):
            ttl = _wm_human_ttl()
        else:
            ttl = int(ttl_ticks if ttl_ticks is not None else _wm_default_ttl())
        self._slots[k] = WmSlot(
            key=k,
            value=float(value),
            text=str(text or ""),
            tick_written=int(tick),
            ttl_ticks=ttl,
            source=str(source or ""),
        )
        if k in self._order:
            self._order.remove(k)
        self._order.append(k)
        while len(self._order) > self._capacity:
            evict = next(
                (ek for ek in self._order if not _is_human_task_key(ek)),
                None,
            )
            if evict is None:
                evict = self._order.pop(0)
            else:
                self._order.remove(evict)
            self._slots.pop(evict, None)

    def read(self, key: str, default: float = 0.5) -> float:
        slot = self._slots.get(str(key))
        return float(slot.value) if slot is not None else float(default)

    def read_text(self, key: str, default: str = "") -> str:
        slot = self._slots.get(str(key))
        return str(slot.text) if slot is not None else default

    def has(self, key: str) -> bool:
        return str(key) in self._slots

    def keys(self) -> list[str]:
        return list(self._order)

    def active_goals(self, tick: int) -> list[str]:
        out: list[str] = []
        for k in self._order:
            s = self._slots.get(k)
            if s is None or not s.alive(tick):
                continue
            if k.startswith("goal_") or k in ("active_macro", "human_task_active"):
                if s.text:
                    out.append(s.text)
                else:
                    out.append(k)
        return out

    def context_dict(self, tick: int) -> dict[str, float]:
        """Compact float context for S2 student / WM planner."""
        out: dict[str, float] = {}
        for k in self._order:
            s = self._slots.get(k)
            if s is None or not s.alive(tick):
                continue
            out[k] = float(s.value)
        return out

    def decay(self, tick: int) -> int:
        """Remove expired slots; returns count removed."""
        dead = [k for k, s in self._slots.items() if not s.alive(tick)]
        for k in dead:
            self._slots.pop(k, None)
            if k in self._order:
                self._order.remove(k)
        return len(dead)

    def snapshot(self) -> dict[str, Any]:
        return {
            "enabled": working_memory_enabled(),
            "capacity": self._capacity,
            "n_slots": len(self._slots),
            "slots": [
                {
                    "key": s.key,
                    "value": round(s.value, 4),
                    "text": s.text[:64],
                    "tick": s.tick_written,
                    "ttl": s.ttl_ticks,
                    "source": s.source,
                }
                for s in (self._slots[k] for k in self._order if k in self._slots)
            ],
        }
