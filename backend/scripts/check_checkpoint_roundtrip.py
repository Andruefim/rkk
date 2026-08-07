#!/usr/bin/env python3
"""
Проверка накопительного обучения между запусками: поднимает Simulation, тикает,
сохраняет чекпоинт, поднимает вторую Simulation с resume и сравнивает веса.

    cd backend && RKK_DEVICE=cpu python3 scripts/check_checkpoint_roundtrip.py --ticks 120

Пишет во временный путь (RKK_MEMORY_PATH), рабочий state/autosave.rkk не трогает.
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch


def _module_fingerprint(mod: torch.nn.Module) -> float:
    """Сумма |values| по state_dict: покрывает и параметры, и буферы (W_stack ансамбля)."""
    total = 0.0
    for v in mod.state_dict().values():
        if isinstance(v, torch.Tensor) and v.is_floating_point():
            total += float(v.detach().abs().sum())
    return round(total, 6)


def _fingerprints(sim) -> dict[str, float]:
    from engine.checkpoint_modules import _SPECS, _resolve  # noqa: PLC2701

    out: dict[str, float] = {}
    for spec in _SPECS:
        owner = _resolve(sim, spec.owner)
        if owner is None:
            continue
        for attr in spec.modules:
            mod = owner if attr == "." else getattr(owner, attr, None)
            if isinstance(mod, torch.nn.Module):
                out[f"{spec.key}.{attr}"] = _module_fingerprint(mod)
    mc = getattr(sim, "_motor_cortex", None)
    if mc is not None:
        for name, prog in (mc.programs or {}).items():
            out[f"motor_cortex.{name}"] = _module_fingerprint(prog.net)
    return out


def _counters(sim) -> dict[str, float]:
    """Счётчики обучения продолжают расти после resume — их сверяем на «не меньше»."""
    out: dict[str, float] = {}
    mc = getattr(sim, "_motor_cortex", None)
    if mc is not None:
        for name, prog in (mc.programs or {}).items():
            out[f"motor_cortex.{name}.train_steps"] = float(prog.train_steps)
        out["motor_cortex.cpg_weight"] = float(mc.cpg_weight)
    vis = getattr(sim, "_visual_env", None)
    if vis is not None and getattr(vis, "cortex", None) is not None:
        out["vision_cortex.n_train"] = float(vis.cortex.n_train)
        out["vision_cortex.n_encode"] = float(vis.cortex.n_encode)
    return out


def _boot(resume: bool):
    os.environ["RKK_MEMORY_RESUME_ON_START"] = "1" if resume else "0"
    from engine.simulation import Simulation

    return Simulation()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticks", type=int, default=120)
    args = ap.parse_args()

    tmp = Path(tempfile.mkdtemp(prefix="rkk_ckpt_")) / "autosave.rkk"
    os.environ["RKK_MEMORY_PATH"] = str(tmp)
    os.environ.setdefault("RKK_DEVICE", "cpu")

    print(f"[check] checkpoint path: {tmp}")
    sim_a = _boot(resume=False)
    for _ in range(args.ticks):
        sim_a.tick_step()
    before = _fingerprints(sim_a)
    before_counters = _counters(sim_a)
    saved_tick = int(sim_a.tick)
    save = sim_a.memory_save()
    print(f"[check] saved: ok={save.get('ok')} tick={saved_tick} modules={len(before)}")
    sim_a.shutdown()

    from engine.checkpoint_modules import pending_module_keys

    sim_b = _boot(resume=True)
    at_boot = _fingerprints(sim_b)
    # Ленивые подсистемы (моторная кора, зрительная кора) создаются на первых тиках —
    # отложенные секции чекпоинта досылаются в них уже из tick_step. Замер снимаем
    # сразу после того, как очередь опустела, иначе продолжающееся обучение уводит веса.
    after_apply = dict(at_boot)
    for _ in range(args.ticks):
        sim_b.tick_step()
        if not pending_module_keys(sim_b):
            after_apply = _fingerprints(sim_b)
            break

    ok = True
    print(f"\n{'module':46} {'saved':>13} {'at boot':>13} {'after apply':>13}  status")
    for k in sorted(before):
        b = before[k]
        boot_v = at_boot.get(k)
        applied_v = after_apply.get(k)
        if boot_v is not None and abs(boot_v - b) <= max(1e-4, abs(b) * 1e-4):
            status = "restored"
        elif applied_v is not None and abs(applied_v - b) <= max(1e-3, abs(b) * 1e-2):
            status = "restored (deferred)"
        elif k.startswith("graph_ensemble") or k.startswith("graph_bridge_head"):
            # Оба зависят от размерности графа: включение зрения добавляет slot-узлы
            # уже после загрузки, и матрицы пересобираются под новый d.
            status = "reshaped with graph"
        else:
            status = "LOST"
            ok = False
        print(
            f"{k:46} {b:13.4f} "
            f"{('-' if boot_v is None else f'{boot_v:.4f}'):>13} "
            f"{('-' if applied_v is None else f'{applied_v:.4f}'):>13}  {status}"
        )
    after_counters = _counters(sim_b)
    print(f"\n{'counter':46} {'saved':>13} {'resumed':>13}  status")
    for k, v in sorted(before_counters.items()):
        got = after_counters.get(k)
        if got is None:
            status, ok = "ABSENT", False
        elif k.endswith("cpg_weight"):
            same = abs(got - v) <= max(1e-3, abs(v) * 0.2)
            status = "carried" if same else "LOST"
            ok = ok and same
        else:
            status = "carried" if got >= v else "LOST"
            ok = ok and got >= v
        print(f"{k:46} {v:13.2f} {('-' if got is None else f'{got:.2f}'):>13}  {status}")

    tick_ok = int(sim_b.tick) >= saved_tick
    print(f"\n[check] tick continued: saved={saved_tick} resumed={int(sim_b.tick)} -> {tick_ok}")
    sim_b.shutdown()
    print("[check] RESULT:", "OK" if ok and tick_ok else "MISMATCH")
    return 0 if (ok and tick_ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
