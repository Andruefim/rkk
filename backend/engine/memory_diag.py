"""
Диагностика памяти процесса (RSS) и крупных подсистем RKK.

Включение:
  RKK_MEMORY_DIAG=1          — логи в консоль
  RKK_MEMORY_DIAG_INTERVAL=0 — если >0, каждые N тиков симуляции (см. mixin_tick)
  RKK_MEMORY_TRACE=1         — tracemalloc: дифф топ-N аллокаций между снапшотами сна

Без psutil на Windows пробуем короткий ctypes-вызов к Working Set.
"""
from __future__ import annotations

import os
import sys
import tracemalloc
from typing import Any


def memory_diag_enabled() -> bool:
    return os.environ.get("RKK_MEMORY_DIAG", "0").strip().lower() in (
        "1", "true", "yes", "on",
    )


def memory_trace_enabled() -> bool:
    return os.environ.get("RKK_MEMORY_TRACE", "0").strip().lower() in (
        "1", "true", "yes", "on",
    )


def _env_int(key: str, default: int) -> int:
    try:
        return max(0, int(os.environ.get(key, str(default))))
    except ValueError:
        return default


def rss_mb() -> float | None:
    """Текущий working set / RSS в МБ (по возможности)."""
    try:
        import psutil

        return float(psutil.Process().memory_info().rss) / (1024 * 1024)
    except Exception:
        pass
    if sys.platform == "win32":
        try:
            import ctypes
            from ctypes import wintypes

            class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            pmc = PROCESS_MEMORY_COUNTERS()
            pmc.cb = ctypes.sizeof(pmc)
            h = ctypes.windll.kernel32.GetCurrentProcess()
            if ctypes.windll.psapi.GetProcessMemoryInfo(
                h, ctypes.byref(pmc), pmc.cb
            ):
                return float(pmc.WorkingSetSize) / (1024 * 1024)
        except Exception:
            pass
    try:
        import resource

        ru = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        if sys.platform == "darwin":
            return ru / (1024 * 1024)
        return ru / 1024.0
    except Exception:
        return None


def _tensor_bytes(t: Any) -> int:
    try:
        import torch

        if isinstance(t, torch.Tensor):
            return int(t.numel() * t.element_size())
    except Exception:
        pass
    return 0


def _module_param_bytes(module: Any) -> int:
    try:
        import torch.nn as nn

        if not isinstance(module, nn.Module):
            return 0
        n = 0
        for p in module.parameters():
            n += _tensor_bytes(p)
        for b in module.buffers():
            n += _tensor_bytes(b)
        return n
    except Exception:
        return 0


def _log_line(msg: str) -> None:
    print(msg, flush=True)


def log_sim_memory(sim: Any, tag: str) -> None:
    """Один снимок: RSS + оценки по графу, мостам, episodic, inner voice, CUDA."""
    if not memory_diag_enabled():
        return

    parts: list[str] = [f"[MemDiag] {tag}"]

    r = rss_mb()
    if r is not None:
        parts.append(f"RSS={r:.0f}MB")

    try:
        import torch

        if torch.cuda.is_available():
            parts.append(
                f"CUDA alloc={torch.cuda.memory_allocated() / 1e6:.1f}MB "
                f"reserved={torch.cuda.memory_reserved() / 1e6:.1f}MB"
            )
    except Exception:
        pass

    g = getattr(getattr(sim, "agent", None), "graph", None)
    if g is not None:
        n_nodes = len(getattr(g, "_node_ids", []) or [])
        d = int(getattr(g, "_d", n_nodes) or 0)
        core = getattr(g, "_core", None)
        w_bytes = 0
        if core is not None:
            w_bytes = _module_param_bytes(core)
        parts.append(f"GNN nodes={n_nodes} d={d} core_params~{w_bytes / 1e6:.2f}MB")

    ag = getattr(sim, "agent", None)
    if ag is not None:
        s1 = getattr(ag, "system1", None)
        if s1 is not None:
            net = getattr(s1, "net", None)
            if net is not None:
                parts.append(f"system1~{_module_param_bytes(net) / 1e6:.2f}MB")

    mc = getattr(sim, "_motor_cortex", None)
    if mc is not None:
        net = getattr(mc, "net", None) or getattr(mc, "_net", None)
        if net is not None:
            parts.append(f"motor_net~{_module_param_bytes(net) / 1e6:.2f}MB")

    iv = getattr(sim, "_inner_voice", None)
    if iv is not None:
        net = getattr(iv, "net", None)
        buf = 0
        if net is not None:
            tb = getattr(net, "_train_buf", None)
            if tb is not None:
                buf = len(tb)
            parts.append(
                f"inner_voice net~{_module_param_bytes(net) / 1e6:.2f}MB train_buf={buf}"
            )

    bridge = getattr(sim, "_world_bridge", None)
    if bridge is not None:
        tr = getattr(bridge, "_transitions", None)
        ln = len(tr) if tr is not None else 0
        parts.append(f"world_bridge={ln}/{getattr(bridge, '_maxlen', '?')}")

    ep = getattr(sim, "_episodic_memory", None)
    if ep is not None:
        falls = getattr(ep, "falls", None)
        parts.append(f"episodic_falls={len(falls) if falls is not None else 0}")

    nlg = getattr(sim, "_neural_lang", None)
    if nlg is not None:
        nb = 0
        for attr in ("speech_decoder", "concept_projector", "spatial_memory"):
            m = getattr(nlg, attr, None)
            if m is not None:
                nb += _module_param_bytes(m)
        parts.append(f"neural_lang~{nb / 1e6:.2f}MB")

    _log_line(" | ".join(parts))


_tracemalloc_started = False
_last_snapshot: tracemalloc.Snapshot | None = None


def trace_start_if_needed() -> None:
    global _tracemalloc_started
    if not memory_trace_enabled() or _tracemalloc_started:
        return
    tracemalloc.start(25)
    _tracemalloc_started = True
    _log_line("[MemDiag] tracemalloc started (RKK_MEMORY_TRACE=1)")


def trace_snapshot(tag: str) -> None:
    """Сравнить с предыдущим снапшотом и вывести топ аллокаций Python."""
    global _last_snapshot
    if not memory_trace_enabled():
        return
    trace_start_if_needed()
    snap = tracemalloc.take_snapshot()
    top_n = _env_int("RKK_MEMORY_TRACE_TOP", 12)
    if _last_snapshot is not None:
        stats = snap.compare_to(_last_snapshot, "lineno")
        lines = [f"[MemDiag] tracemalloc diff → {tag} (top {top_n})"]
        for s in stats[:top_n]:
            lines.append(f"  {s}")
        _log_line("\n".join(lines))
    _last_snapshot = snap


def trace_clear() -> None:
    global _last_snapshot, _tracemalloc_started
    _last_snapshot = None
    if _tracemalloc_started:
        tracemalloc.stop()
        _tracemalloc_started = False
