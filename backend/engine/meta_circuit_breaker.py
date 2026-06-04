"""
Track I3: MetaCircuitBreaker — CLOSED/OPEN/HALF_OPEN state machine over W_meta (no do-calculus).
"""
from __future__ import annotations

import os
from typing import Any


def _env_float(key: str, default: float) -> float:
    try:
        return float(os.environ.get(key, str(default)))
    except ValueError:
        return default


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.environ.get(key, str(default)))
    except ValueError:
        return default


def meta_cb_enabled() -> bool:
    raw = os.environ.get("RKK_META_CB_ENABLED", "0")
    return raw.strip().lower() in ("1", "true", "yes", "on")


def meta_cb_pe_open() -> float:
    return _env_float("RKK_META_CB_PE_OPEN", 0.25)


def meta_cb_pe_close() -> float:
    return _env_float("RKK_META_CB_PE_CLOSE", 0.12)


def meta_cb_age_open() -> int:
    return _env_int("RKK_META_CB_AGE_OPEN", 2000)


def meta_cb_reset_after() -> int:
    return _env_int("RKK_META_CB_RESET_AFTER", 500)


class MetaCircuitBreaker:
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

    def __init__(self) -> None:
        self.state = self.CLOSED
        self.ema_pe = 0.0
        self.ticks_in_open = 0
        self.recovery_start_tick: int | None = None
        self._meta_recovery_ticks: int | None = None
        self._transition_log: list[str] = []

    def observe(self, meta_pe: float, meta_age: int, tick: int) -> None:
        if not meta_cb_enabled():
            return
        self.ema_pe = 0.9 * self.ema_pe + 0.1 * float(meta_pe)

        if self.state == self.CLOSED:
            if self.ema_pe > meta_cb_pe_open() or int(meta_age) > meta_cb_age_open():
                self._transition_open(tick)

        elif self.state == self.OPEN:
            self.ticks_in_open += 1
            if self.ticks_in_open >= meta_cb_reset_after():
                self._transition_half_open(tick)

        elif self.state == self.HALF_OPEN:
            if self.ema_pe < meta_cb_pe_close():
                self._transition_closed(tick)
            elif self.ema_pe > meta_cb_pe_open():
                self._transition_open(tick)

    def _transition_open(self, tick: int) -> None:
        self.state = self.OPEN
        self.ticks_in_open = 0
        self.recovery_start_tick = int(tick)
        self._transition_log.append(f"open@{tick}")

    def _transition_half_open(self, tick: int) -> None:
        self.state = self.HALF_OPEN
        self.ticks_in_open = 0
        self.ema_pe = meta_cb_pe_close()
        self._transition_log.append(f"half_open@{tick}")

    def _transition_closed(self, tick: int) -> None:
        if self.recovery_start_tick is not None:
            self._meta_recovery_ticks = int(tick) - int(self.recovery_start_tick)
        self.state = self.CLOSED
        self.recovery_start_tick = None
        self._transition_log.append(f"closed@{tick}")

    def force_open(self, tick: int) -> None:
        self._transition_open(tick)

    def force_half_open(self, tick: int) -> None:
        self._transition_half_open(tick)

    def force_closed(self, tick: int) -> None:
        self._transition_closed(tick)

    @property
    def wmeta_active(self) -> bool:
        return self.state != self.OPEN

    def recovery_ticks(self, current_tick: int) -> int | None:
        if self.recovery_start_tick is None:
            return self._meta_recovery_ticks
        return int(current_tick) - int(self.recovery_start_tick)

    def reset_w_meta_if_needed(self, w_meta: Any) -> bool:
        """Fresh-init W_meta when entering HALF_OPEN."""
        if self.state != self.HALF_OPEN or w_meta is None:
            return False
        w_meta.load_dict({})
        return True

    def snapshot(self, tick: int = 0) -> dict[str, Any]:
        return {
            "enabled": meta_cb_enabled(),
            "state": self.state,
            "ema_pe": round(self.ema_pe, 4),
            "ticks_in_open": self.ticks_in_open,
            "wmeta_active": self.wmeta_active,
            "recovery_start_tick": self.recovery_start_tick,
            "meta_recovery_ticks": self.recovery_ticks(tick),
            "transitions": self._transition_log[-8:],
        }
