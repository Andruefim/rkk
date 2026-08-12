"""
checkpoint_modules.py — сохранение/восстановление весов обучаемых подсистем.

`persistence.py` исторически сохранял только каузальный граф и TemporalBlankets,
поэтому при рестарте бэкенда всё остальное обучение (зрительная кора, System1,
моторная кора, мозжечок, рефлексы, CPG, внутренний голос, проприоцепция,
ансамбль графа) начиналось с нуля.

Здесь описан декларативный реестр подсистем: каждая спека знает, где живёт
модуль, какой у него оптимизатор и какие счётчики обучения нужно перенести.

Часть подсистем создаётся лениво (моторная кора — при первом шаге локомоции,
зрительная кора — при auto-visual, мозжечок — по env-флагу), поэтому на момент
загрузки чекпоинта их ещё нет. Такие секции складываются в отложенную очередь
и применяются позже через `apply_pending_learnable_modules`.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn

CHECKPOINT_MODULES_VERSION = 1

_PENDING_ATTR = "_pending_module_state"
_SELF = "."


def optim_state_enabled() -> bool:
    """Моменты Adam восстанавливать по умолчанию: без них обучение «дёргается» после рестарта."""
    return os.environ.get("RKK_CKPT_OPTIM", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


def _verbose() -> bool:
    return os.environ.get("RKK_CKPT_VERBOSE", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


@dataclass(frozen=True)
class ModuleSpec:
    """Описание одной обучаемой подсистемы."""

    key: str
    owner: str
    modules: tuple[str, ...] = ()
    optims: tuple[str, ...] = ()
    counters: tuple[str, ...] = ()
    tensors: tuple[str, ...] = field(default=())


# Пути указаны от объекта Simulation; "agent." уходит в RKKAgent.
_SPECS: tuple[ModuleSpec, ...] = (
    ModuleSpec(
        key="vision_cortex",
        owner="_visual_env.cortex",
        modules=(_SELF,),
        optims=("optim",),
        counters=("n_encode", "n_train", "n_recon_train"),
    ),
    ModuleSpec(
        key="system1",
        owner="agent.system1",
        modules=("net",),
        optims=("optim",),
        counters=("train_steps", "n_calls"),
    ),
    ModuleSpec(
        key="temporal_optim",
        owner="agent.temporal",
        optims=("optim",),
    ),
    ModuleSpec(
        key="graph_optim",
        owner="agent.graph",
        optims=("_optim",),
    ),
    ModuleSpec(
        key="graph_ensemble",
        owner="agent.graph",
        modules=("_ensemble",),
    ),
    ModuleSpec(
        key="graph_traj_head",
        owner="agent.graph",
        modules=("_traj_head",),
    ),
    ModuleSpec(
        key="graph_bridge_head",
        owner="agent.graph",
        modules=("_concept_bridge_head",),
    ),
    ModuleSpec(
        key="cerebellum",
        owner="_cerebellum",
        modules=("forward_model", "inverse_model"),
        optims=("opt_fwd", "opt_inv"),
        counters=("train_steps", "n_calls"),
    ),
    ModuleSpec(
        key="reflex_stabilizer",
        owner="_reflex_stabilizer",
        modules=("net",),
        optims=("opt",),
        counters=("train_steps", "n_calls"),
    ),
    ModuleSpec(
        key="cpg",
        owner="_locomotion_controller",
        modules=("cpg",),
        optims=("optim",),
        counters=("train_steps",),
    ),
    ModuleSpec(
        key="inner_voice",
        owner="_inner_voice",
        modules=("net", "concept_store"),
        optims=("optim",),
        counters=("train_steps",),
    ),
    ModuleSpec(
        key="proprioception",
        owner="_proprio",
        modules=("net",),
        optims=("optim",),
        counters=("train_steps",),
    ),
    ModuleSpec(
        key="demon",
        owner="_demon",
        modules=("policy",),
        optims=("optim",),
        counters=("train_steps",),
    ),
    ModuleSpec(
        key="slot_dynamics",
        owner="_slot_dynamics",
        modules=(_SELF,),
        optims=("optim",),
        counters=("train_steps", "n_predict"),
    ),
)


# ── Резолв путей ──────────────────────────────────────────────────────────────
def _resolve(root: Any, path: str) -> Any:
    if not path:
        return root
    cur = root
    for part in path.split("."):
        cur = getattr(cur, part, None)
        if cur is None:
            return None
    return cur


def _cpu_state_dict(module: nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}


def _load_compatible(module: nn.Module, sd: dict[str, torch.Tensor]) -> int:
    """Копируем только тензоры совпадающей формы: архитектура могла измениться между сейвами."""
    cur = module.state_dict()
    loaded = 0
    with torch.no_grad():
        for k, v in sd.items():
            t = cur.get(k)
            if t is None or not isinstance(v, torch.Tensor):
                continue
            if tuple(t.shape) != tuple(v.shape):
                continue
            t.copy_(v.to(device=t.device, dtype=t.dtype))
            loaded += 1
    return loaded


# ── Моторная кора: словарь программ, каждая со своей сетью ────────────────────
def _pack_motor_cortex(sim: Any) -> dict[str, Any] | None:
    mc = getattr(sim, "_motor_cortex", None)
    if mc is None or not getattr(mc, "programs", None):
        return None
    programs: dict[str, Any] = {}
    for name, prog in list(mc.programs.items()):
        entry: dict[str, Any] = {
            "net": _cpu_state_dict(prog.net),
            "uses": int(getattr(prog, "uses", 0)),
            "train_steps": int(getattr(prog, "train_steps", 0)),
            "mean_reward": float(getattr(prog, "mean_reward", 0.0)),
            "active": bool(getattr(prog, "active", True)),
        }
        if optim_state_enabled():
            try:
                entry["optim"] = prog.optim.state_dict()
            except Exception:
                pass
        programs[str(name)] = entry
    return {
        "programs": programs,
        "cpg_weight": float(getattr(mc, "cpg_weight", 1.0)),
        "annealing_enabled": bool(getattr(mc, "_annealing_enabled", False)),
        "annealing_ticks": int(getattr(mc, "_annealing_ticks", 0)),
        "quality_ema": float(getattr(mc, "_quality_ema", 0.0)),
        "posture_ema": float(getattr(mc, "_posture_ema", 0.0)),
        "contact_ema": float(getattr(mc, "_contact_ema", 0.0)),
        "total_uses": int(getattr(mc, "_total_uses", 0)),
    }


def _unpack_motor_cortex(sim: Any, data: dict[str, Any]) -> bool:
    mc = getattr(sim, "_motor_cortex", None)
    if mc is None:
        return False
    for name, entry in (data.get("programs") or {}).items():
        try:
            prog = mc.ensure_program(str(name))
        except Exception:
            continue
        _load_compatible(prog.net, entry.get("net") or {})
        if optim_state_enabled() and entry.get("optim"):
            try:
                prog.optim.load_state_dict(entry["optim"])
            except Exception:
                pass
        prog.uses = int(entry.get("uses", prog.uses))
        prog.train_steps = int(entry.get("train_steps", prog.train_steps))
        prog.mean_reward = float(entry.get("mean_reward", prog.mean_reward))
        prog.active = bool(entry.get("active", prog.active))
    mc.cpg_weight = float(data.get("cpg_weight", mc.cpg_weight))
    mc._annealing_enabled = bool(data.get("annealing_enabled", mc._annealing_enabled))
    mc._annealing_ticks = int(data.get("annealing_ticks", mc._annealing_ticks))
    mc._quality_ema = float(data.get("quality_ema", mc._quality_ema))
    mc._posture_ema = float(data.get("posture_ema", mc._posture_ema))
    mc._contact_ema = float(data.get("contact_ema", mc._contact_ema))
    mc._total_uses = int(data.get("total_uses", mc._total_uses))
    return True


def _motor_cortex_ready(sim: Any) -> bool:
    return getattr(sim, "_motor_cortex", None) is not None


_CUSTOM_SECTIONS: dict[str, dict[str, Any]] = {
    "motor_cortex": {
        "pack": _pack_motor_cortex,
        "unpack": _unpack_motor_cortex,
        "ready": _motor_cortex_ready,
    },
}


# ── Pack / unpack по спекам ───────────────────────────────────────────────────
def _pack_spec(sim: Any, spec: ModuleSpec) -> dict[str, Any] | None:
    owner = _resolve(sim, spec.owner)
    if owner is None:
        return None
    out: dict[str, Any] = {}
    modules: dict[str, Any] = {}
    for attr in spec.modules:
        mod = owner if attr == _SELF else getattr(owner, attr, None)
        if isinstance(mod, nn.Module):
            modules[attr] = _cpu_state_dict(mod)
    if modules:
        out["modules"] = modules
    if optim_state_enabled():
        optims: dict[str, Any] = {}
        for attr in spec.optims:
            opt = getattr(owner, attr, None)
            if isinstance(opt, torch.optim.Optimizer):
                try:
                    optims[attr] = opt.state_dict()
                except Exception:
                    continue
        if optims:
            out["optims"] = optims
    counters: dict[str, Any] = {}
    for attr in spec.counters:
        val = getattr(owner, attr, None)
        if isinstance(val, (int, float, bool)):
            counters[attr] = val
    if counters:
        out["counters"] = counters
    tensors: dict[str, Any] = {}
    for attr in spec.tensors:
        val = getattr(owner, attr, None)
        if isinstance(val, torch.Tensor):
            tensors[attr] = val.detach().cpu().clone()
    if tensors:
        out["tensors"] = tensors
    return out or None


def _unpack_spec(sim: Any, spec: ModuleSpec, data: dict[str, Any]) -> bool:
    owner = _resolve(sim, spec.owner)
    if owner is None:
        return False
    ready = False
    for attr, sd in (data.get("modules") or {}).items():
        mod = owner if attr == _SELF else getattr(owner, attr, None)
        if isinstance(mod, nn.Module):
            _load_compatible(mod, sd)
            ready = True
    if optim_state_enabled():
        for attr, osd in (data.get("optims") or {}).items():
            opt = getattr(owner, attr, None)
            if isinstance(opt, torch.optim.Optimizer):
                try:
                    opt.load_state_dict(osd)
                    ready = True
                except Exception:
                    pass
    for attr, val in (data.get("counters") or {}).items():
        try:
            setattr(owner, attr, type(getattr(owner, attr, val))(val))
            ready = True
        except Exception:
            pass
    for attr, val in (data.get("tensors") or {}).items():
        cur = getattr(owner, attr, None)
        if isinstance(val, torch.Tensor):
            dev = cur.device if isinstance(cur, torch.Tensor) else torch.device("cpu")
            setattr(owner, attr, val.to(dev))
            ready = True
    if not data.get("modules") and not data.get("optims"):
        # Секция без весов (например, только счётчики) считается применённой.
        ready = ready or bool(data.get("counters") or data.get("tensors"))
    return ready


def _spec_ready(sim: Any, spec: ModuleSpec, data: dict[str, Any]) -> bool:
    """Подсистема готова принять веса, только если её модули уже созданы."""
    owner = _resolve(sim, spec.owner)
    if owner is None:
        return False
    wanted = list((data.get("modules") or {}).keys())
    if not wanted:
        return True
    for attr in wanted:
        mod = owner if attr == _SELF else getattr(owner, attr, None)
        if not isinstance(mod, nn.Module):
            return False
    return True


def pack_learnable_modules(sim: Any) -> dict[str, Any]:
    """Собрать веса всех существующих на данный момент обучаемых подсистем."""
    sections: dict[str, Any] = {}
    for spec in _SPECS:
        try:
            packed = _pack_spec(sim, spec)
        except Exception as e:
            if _verbose():
                print(f"[Ckpt] pack {spec.key} skipped: {type(e).__name__}: {e}")
            continue
        if packed:
            sections[spec.key] = packed
    for key, hooks in _CUSTOM_SECTIONS.items():
        try:
            packed = hooks["pack"](sim)
        except Exception as e:
            if _verbose():
                print(f"[Ckpt] pack {key} skipped: {type(e).__name__}: {e}")
            continue
        if packed:
            sections[key] = packed
    return {"version": CHECKPOINT_MODULES_VERSION, "sections": sections}


def unpack_learnable_modules(sim: Any, data: dict[str, Any] | None) -> dict[str, Any]:
    """
    Восстановить веса. Секции для ещё не созданных подсистем откладываются
    и применяются позже (`apply_pending_learnable_modules`).
    """
    if not isinstance(data, dict):
        return {"applied": [], "deferred": []}
    sections = data.get("sections")
    if not isinstance(sections, dict):
        return {"applied": [], "deferred": []}

    pending: dict[str, Any] = dict(getattr(sim, _PENDING_ATTR, {}) or {})
    applied: list[str] = []
    for key, payload in sections.items():
        pending[str(key)] = payload
    setattr(sim, _PENDING_ATTR, pending)
    applied = apply_pending_learnable_modules(sim)
    deferred = sorted(getattr(sim, _PENDING_ATTR, {}) or {})
    if _verbose():
        print(
            f"[Ckpt] modules restored: {sorted(applied)}"
            + (f", deferred: {deferred}" if deferred else "")
        )
    return {"applied": applied, "deferred": deferred}


def apply_pending_learnable_modules(sim: Any) -> list[str]:
    """Применить отложенные секции к подсистемам, которые уже успели создаться."""
    pending = getattr(sim, _PENDING_ATTR, None)
    if not pending:
        return []
    spec_by_key = {s.key: s for s in _SPECS}
    applied: list[str] = []
    for key in list(pending.keys()):
        payload = pending[key]
        try:
            if key in _CUSTOM_SECTIONS:
                hooks = _CUSTOM_SECTIONS[key]
                if not hooks["ready"](sim):
                    continue
                if hooks["unpack"](sim, payload):
                    applied.append(key)
                    pending.pop(key, None)
                continue
            spec = spec_by_key.get(key)
            if spec is None:
                pending.pop(key, None)
                continue
            if not _spec_ready(sim, spec, payload):
                continue
            if _unpack_spec(sim, spec, payload):
                applied.append(key)
                pending.pop(key, None)
        except Exception as e:
            if _verbose():
                print(f"[Ckpt] restore {key} failed: {type(e).__name__}: {e}")
            pending.pop(key, None)
    if not pending:
        setattr(sim, _PENDING_ATTR, {})
    return applied


def pending_module_keys(sim: Any) -> list[str]:
    return sorted(getattr(sim, _PENDING_ATTR, {}) or {})
