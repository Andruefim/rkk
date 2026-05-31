"""
Per-tick performance profiler: cumulative + ranked spans for agent, simulation, and background loops.

Enable: RKK_TICK_PROFILE=1 (default on).
API: GET /api/tick_profile
Console: ranked report every RKK_TICK_PROFILE_REPORT_EVERY ticks and on slow ticks.
"""
from __future__ import annotations

import os
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator

_EMA_ALPHA = 0.08


def _env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in ("1", "true", "yes", "on")


def _env_int(name: str, default: int, lo: int, hi: int) -> int:
    try:
        v = int(os.environ.get(name, str(default)))
    except ValueError:
        v = default
    return max(lo, min(hi, v))


def _env_float(name: str, default: float, lo: float, hi: float) -> float:
    try:
        v = float(os.environ.get(name, str(default)))
    except ValueError:
        v = default
    return max(lo, min(hi, v))


@dataclass
class SpanStats:
    count: int = 0
    total_ms: float = 0.0
    max_ms: float = 0.0
    ema_ms: float = 0.0
    last_ms: float = 0.0

    def add(self, ms: float) -> None:
        self.count += 1
        self.total_ms += ms
        self.last_ms = ms
        if ms > self.max_ms:
            self.max_ms = ms
        if self.count == 1:
            self.ema_ms = ms
        else:
            self.ema_ms = (1.0 - _EMA_ALPHA) * self.ema_ms + _EMA_ALPHA * ms

    @property
    def avg_ms(self) -> float:
        return self.total_ms / max(1, self.count)

    def to_dict(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "avg_ms": round(self.avg_ms, 3),
            "ema_ms": round(self.ema_ms, 3),
            "last_ms": round(self.last_ms, 3),
            "max_ms": round(self.max_ms, 3),
            "total_ms": round(self.total_ms, 3),
        }


@dataclass
class _TickRecord:
    tick: int
    wall_ms: float
    spans: dict[str, float] = field(default_factory=dict)


class TickProfiler:
    """Thread-safe rolling profiler; span names use prefixes: agent.*, sim.*, bg.*, api.*."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._stats: dict[str, SpanStats] = {}
        self._window: list[_TickRecord] = []
        self._window_max = _env_int("RKK_TICK_PROFILE_WINDOW", 200, 20, 5000)
        self._current_tick = -1
        self._current_spans: dict[str, float] = {}
        self._tick_t0 = 0.0
        self._last_report_tick = -1
        self._report_every = _env_int("RKK_TICK_PROFILE_REPORT_EVERY", 100, 1, 100000)
        self._slow_ms = _env_float("RKK_TICK_PROFILE_SLOW_MS", 800.0, 50.0, 120000.0)
        self._min_pct = _env_float("RKK_TICK_PROFILE_MIN_PCT", 0.3, 0.0, 50.0)
        self._last_effective_hz = 0.0
        self._last_wall_ms = 0.0

    @staticmethod
    def enabled() -> bool:
        return _env_flag("RKK_TICK_PROFILE", "1")

    def begin_tick(self, tick: int) -> None:
        if not self.enabled():
            return
        with self._lock:
            self._current_tick = int(tick)
            self._current_spans = {}
            self._tick_t0 = time.perf_counter()

    def record(self, name: str, ms: float) -> None:
        if not self.enabled() or ms < 0:
            return
        name = str(name).strip()
        if not name:
            return
        with self._lock:
            st = self._stats.setdefault(name, SpanStats())
            st.add(ms)
            prev = self._current_spans.get(name, 0.0)
            self._current_spans[name] = prev + ms

    def record_seconds(self, name: str, sec: float) -> None:
        self.record(name, float(sec) * 1000.0)

    def merge_dict_seconds(self, timings: dict[str, float], *, prefix: str = "") -> None:
        """Merge a {name: seconds} map (e.g. agent _slow_t) into profiler."""
        pfx = prefix if prefix.endswith(".") or not prefix else f"{prefix}."
        for k, v in timings.items():
            try:
                sec = float(v)
            except (TypeError, ValueError):
                continue
            self.record_seconds(f"{pfx}{k}", sec)

    def end_tick(self) -> None:
        if not self.enabled():
            return
        wall_ms = (time.perf_counter() - self._tick_t0) * 1000.0
        with self._lock:
            tick = self._current_tick
            spans = dict(self._current_spans)
            if "sim.wall" not in spans:
                spans["sim.wall"] = wall_ms
                st = self._stats.setdefault("sim.wall", SpanStats())
                st.add(wall_ms)
            rec = _TickRecord(tick=tick, wall_ms=wall_ms, spans=spans)
            self._window.append(rec)
            if len(self._window) > self._window_max:
                self._window = self._window[-self._window_max :]
            self._last_wall_ms = wall_ms
            if wall_ms > 0:
                self._last_effective_hz = 1000.0 / wall_ms
        self.maybe_report(tick, wall_ms)

    def maybe_report(self, tick: int, wall_ms: float | None = None) -> None:
        if not self.enabled():
            return
        w = wall_ms if wall_ms is not None else self._last_wall_ms
        periodic = tick > 0 and (tick - self._last_report_tick) >= self._report_every
        slow = w >= self._slow_ms
        if not periodic and not slow:
            return
        with self._lock:
            self._last_report_tick = tick
        line = self.format_report(tick=tick, wall_ms=w, top_n=20)
        if line:
            print(line, flush=True)

    def ranked(
        self,
        *,
        top_n: int = 40,
        use_ema: bool = True,
        scope_prefix: str | None = None,
    ) -> list[dict[str, Any]]:
        with self._lock:
            items: list[tuple[str, SpanStats]] = list(self._stats.items())
            window = list(self._window)
        if scope_prefix:
            items = [(n, s) for n, s in items if n.startswith(scope_prefix)]
        if not items:
            return []
        total_avg = sum(s.ema_ms if use_ema else s.avg_ms for _, s in items)
        if total_avg <= 0:
            total_avg = 1.0
        ranked_list: list[dict[str, Any]] = []
        for name, st in items:
            avg = st.ema_ms if use_ema else st.avg_ms
            pct = 100.0 * avg / total_avg
            if pct < self._min_pct and st.count > 3:
                continue
            ranked_list.append(
                {
                    "name": name,
                    "avg_ms": round(avg, 3),
                    "pct": round(pct, 2),
                    "max_ms": round(st.max_ms, 3),
                    "last_ms": round(st.last_ms, 3),
                    "count": st.count,
                }
            )
        ranked_list.sort(key=lambda x: (-x["avg_ms"], x["name"]))
        return ranked_list[: max(1, top_n)]

    def last_tick_spans(self) -> dict[str, float]:
        with self._lock:
            if not self._window:
                return {}
            return dict(self._window[-1].spans)

    def snapshot(self) -> dict[str, Any]:
        ranked = self.ranked(top_n=50)
        with self._lock:
            tick = self._current_tick
            window_n = len(self._window)
            wall = self._last_wall_ms
            hz = self._last_effective_hz
            last_spans = dict(self._window[-1].spans) if self._window else {}
        return {
            "enabled": True,
            "tick": tick,
            "window_ticks": window_n,
            "window_max": self._window_max,
            "last_wall_ms": round(wall, 2),
            "effective_hz": round(hz, 3),
            "last_tick_spans_ms": {k: round(v, 2) for k, v in sorted(
                last_spans.items(), key=lambda x: -x[1]
            )},
            "ranked": ranked,
        }

    def format_report(self, *, tick: int, wall_ms: float, top_n: int = 15) -> str:
        ranked = self.ranked(top_n=top_n)
        if not ranked:
            return ""
        hz = 1000.0 / wall_ms if wall_ms > 0 else 0.0
        lines = [
            f"[TICK PROFILE tick={tick} wall={wall_ms:.0f}ms ~{hz:.2f}Hz window={self._window_max}]"
        ]
        for i, row in enumerate(ranked[:top_n], 1):
            lines.append(
                f"  {i:2d}. {row['name']:<28} "
                f"ema={row['avg_ms']:7.1f}ms  pct={row['pct']:5.1f}%  "
                f"max={row['max_ms']:7.1f}ms  n={row['count']}"
            )
        return "\n".join(lines)


_profiler: TickProfiler | None = None
_profiler_lock = threading.Lock()


def get_tick_profiler() -> TickProfiler:
    global _profiler
    with _profiler_lock:
        if _profiler is None:
            _profiler = TickProfiler()
        return _profiler


@contextmanager
def tick_profile(name: str) -> Iterator[None]:
    """Context manager: record span wall time in milliseconds."""
    if not TickProfiler.enabled():
        yield
        return
    t0 = time.perf_counter()
    try:
        yield
    finally:
        get_tick_profiler().record(name, (time.perf_counter() - t0) * 1000.0)


def profile_snapshot() -> dict[str, Any]:
    if not TickProfiler.enabled():
        return {"enabled": False}
    return get_tick_profiler().snapshot()
