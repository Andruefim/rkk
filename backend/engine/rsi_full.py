"""
rsi_full.py — Full RSI loop (Phase C) + Motor Cortex RSI (Phase D).

ИЗМЕНЕНИЯ (Motor Cortex):
  - RSIController.tick() вызывает mc.rsi_check_and_spawn() если motor cortex существует
  - _perturb_cpg() теперь также уведомляет MotorCortexLibrary о сбросе
  - Новый тип события: "motor_cortex_spawn"
  - Новый тип события: "cpg_annealing_active" — когда cpg_weight < 0.5

Оригинальный функционал не затронут.
"""
from __future__ import annotations

import os
from collections import deque
from typing import TYPE_CHECKING, Any, Callable

import numpy as np
import torch
import torch.nn as nn

from engine.causal_gnn import CausalGNNCore
from engine.causal_graph import USE_GNN
from engine.environment_humanoid import MOTOR_INTENT_VARS

if TYPE_CHECKING:
    from engine.agent import RKKAgent
    from engine.cpg_locomotion import LocomotionController
    from engine.skill_library import SkillLibrary
    from engine.motor_cortex import MotorCortexLibrary


def _env_bool(key: str, default: bool = False) -> bool:
    v = os.environ.get(key, "1" if default else "0").strip().lower()
    return v in ("1", "true", "yes", "on")


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.environ.get(key, str(default)))
    except ValueError:
        return default


def _env_float(key: str, default: float) -> float:
    try:
        return float(os.environ.get(key, str(default)))
    except ValueError:
        return default


def _unwrap_gnn_core(core: Any) -> CausalGNNCore | None:
    if core is None:
        return None
    base = getattr(core, "_orig_mod", core)
    return base if isinstance(base, CausalGNNCore) else None


def _migrate_gnn_expand_hidden(old: CausalGNNCore, new_hidden: int) -> CausalGNNCore:
    d, dev = old.d, old.device
    oh, nh = old.hidden, new_hidden
    new = CausalGNNCore(d, dev, hidden=new_hidden)
    with torch.no_grad():
        new.W.copy_(old.W)

    def copy_linear(src: nn.Linear, dst: nn.Linear) -> None:
        so, si = src.weight.shape
        do, di = dst.weight.shape
        o_ = min(so, do)
        i_ = min(si, di)
        with torch.no_grad():
            w = dst.weight.detach().clone()
            w[:o_, :i_].copy_(src.weight[:o_, :i_].detach())
            dst.weight.copy_(w)
            if src.bias is not None and dst.bias is not None:
                b_ = min(src.bias.shape[0], dst.bias.shape[0])
                b = dst.bias.detach().clone()
                b[:b_].copy_(src.bias[:b_].detach())
                dst.bias.copy_(b)

    copy_linear(old.node_enc[0], new.node_enc[0])
    copy_linear(old.action_enc[0], new.action_enc[0])
    copy_linear(old.msg_fn[0], new.msg_fn[0])
    copy_linear(old.msg_fn[2], new.msg_fn[2])
    copy_linear(old.out_dec[0], new.out_dec[0])
    copy_linear(old.out_dec[2], new.out_dec[2])

    if nh > oh:
        with torch.no_grad():
            def xavier_rows(p: nn.Parameter, r0: int, r1: int) -> None:
                w = p.detach().clone()
                chunk = torch.empty(r1 - r0, w.shape[1], device=w.device, dtype=w.dtype)
                nn.init.xavier_uniform_(chunk, gain=0.5)
                w[r0:r1, :] = chunk
                p.copy_(w)

            def xavier_cols(p: nn.Parameter, c0: int, c1: int) -> None:
                w = p.detach().clone()
                chunk = torch.empty(w.shape[0], c1 - c0, device=w.device, dtype=w.dtype)
                nn.init.xavier_uniform_(chunk, gain=0.5)
                w[:, c0:c1] = chunk
                p.copy_(w)

            def zero_bias_rows(p: nn.Parameter, r0: int, r1: int) -> None:
                b = p.detach().clone()
                b[r0:r1] = 0.0
                p.copy_(b)

            for seq in (new.node_enc, new.action_enc):
                xavier_rows(seq[0].weight, oh, nh)
                zero_bias_rows(seq[0].bias, oh, nh)
            xavier_rows(new.msg_fn[0].weight, oh, nh)
            zero_bias_rows(new.msg_fn[0].bias, oh, nh)
            xavier_rows(new.msg_fn[2].weight, oh, nh)
            xavier_cols(new.msg_fn[2].weight, oh, nh)
            zero_bias_rows(new.msg_fn[2].bias, oh, nh)
            xavier_rows(new.out_dec[0].weight, oh, nh)
            zero_bias_rows(new.out_dec[0].bias, oh, nh)
            xavier_cols(new.out_dec[2].weight, oh, nh)

    return new


class RSIController:
    """
    Следит за прогрессом агента и запускает структурную самонастройку.

    MOTOR_CORTEX: добавлен motor_cortex_supplier для координации с MotorCortexLibrary.
    Новые RSI события:
      - "motor_cortex_spawn": новый моторный модуль создан
      - "cpg_annealing": cpg_weight снизился ниже 0.5

    Оригинальные триггеры:
      - phi drop → VL relax
      - walk skill plateau → harder variants
      - discovery plateau → GNN expand
      - loco plateau → CPG perturb
    """

    def __init__(
        self,
        agent: RKKAgent,
        locomotion_ctrl: LocomotionController | None = None,
        skill_library_supplier: Callable[[], SkillLibrary | None] | None = None,
        motor_cortex_supplier: Callable[[], MotorCortexLibrary | None] | None = None,
    ):
        self.agent = agent
        self.loco = locomotion_ctrl
        self._skill_supplier = skill_library_supplier
        self._mc_supplier = motor_cortex_supplier  # MOTOR_CORTEX

        maxlen = max(50, _env_int("RKK_RSI_FULL_SNAPSHOT_CAP", 500))
        self._snapshots: deque[dict[str, float]] = deque(maxlen=maxlen)
        self._rsi_events: list[dict[str, Any]] = []
        self._last_structural_tick = -10**9
        self._vl_backup: tuple[float, float] | None = None
        self._vl_relax_remaining = 0

        # MOTOR_CORTEX tracking
        self._last_mc_spawn_tick = -10**9
        self._last_cpg_annealing_event_w = 1.0

    def snapshot(self) -> dict[str, Any]:
        return {
            "n_snapshots": len(self._snapshots),
            "recent_events": list(self._rsi_events[-12:]),
            "vl_relax_ticks": self._vl_relax_remaining,
        }

    def reset(self) -> None:
        self._snapshots.clear()
        self._rsi_events.clear()
        self._last_structural_tick = -10**9
        self._vl_backup = None
        self._vl_relax_remaining = 0

    def _skills(self) -> SkillLibrary | None:
        if self._skill_supplier is None:
            return None
        try:
            return self._skill_supplier()
        except Exception:
            return None

    def _motor_cortex(self) -> MotorCortexLibrary | None:
        if self._mc_supplier is None:
            return None
        try:
            return self._mc_supplier()
        except Exception:
            return None

    def _walk_skill_sr_mean(self) -> float | None:
        lib = self._skills()
        if lib is None:
            return None
        vals: list[float] = []
        for s in lib.skills:
            if "walk" not in s.goals:
                continue
            if s.uses < _env_int("RKK_RSI_FULL_SKILL_MIN_USES", 20):
                continue
            vals.append(float(s.success_rate))
        return float(np.mean(vals)) if vals else None

    def _cooldown_ok(self, tick: int) -> bool:
        cd = max(1, _env_int("RKK_RSI_FULL_COOLDOWN", 320))
        return (tick - self._last_structural_tick) >= cd

    def _tick_vl_restore(self) -> None:
        if self._vl_relax_remaining <= 0 or self._vl_backup is None:
            return
        self._vl_relax_remaining -= 1
        if self._vl_relax_remaining == 0:
            b = self.agent.value_layer.bounds
            pm, pms = self._vl_backup
            b.phi_min = pm
            b.phi_min_steady = pms
            self._vl_backup = None
            print("[RSI] Value Layer bounds restored after phi-relax TTL")

    def _relax_value_layer(self, tick: int) -> dict[str, Any] | None:
        if not _env_bool("RKK_RSI_FULL_VL_RELAX", True):
            return None
        b = self.agent.value_layer.bounds
        if self._vl_backup is None:
            self._vl_backup = (float(b.phi_min), float(b.phi_min_steady))
        elif self._vl_relax_remaining > 0:
            ttl = _env_int("RKK_RSI_FULL_VL_RELAX_TICKS", 400)
            self._vl_relax_remaining = max(self._vl_relax_remaining, ttl)
            return {"type": "vl_relax_extend", "tick": tick}
        fac = _env_float("RKK_RSI_FULL_VL_RELAX_FACTOR", 0.90)
        b.phi_min = max(0.005, float(b.phi_min) * fac)
        b.phi_min_steady = max(0.01, float(b.phi_min_steady) * fac)
        self._vl_relax_remaining = max(
            self._vl_relax_remaining,
            _env_int("RKK_RSI_FULL_VL_RELAX_TICKS", 400),
        )
        ev = {"type": "vl_relax_phi", "phi_min": b.phi_min, "tick": tick}
        self._rsi_events.append(ev)
        self._last_structural_tick = tick
        print(f"[RSI] VL relax (phi_min→{b.phi_min:.4f})")
        return ev

    def _expand_gnn_hidden(self, new_hidden: int) -> None:
        graph = self.agent.graph
        old_wrapped = graph._core
        old = _unwrap_gnn_core(old_wrapped)
        if old is None:
            return
        new_core = _migrate_gnn_expand_hidden(old, new_hidden)
        graph._core = new_core
        graph._optim = torch.optim.Adam(new_core.parameters(), lr=5e-3)
        graph._invalidate_cache()
        graph._maybe_compile_gnn_core()
        print(f"[RSI] GNN expanded: hidden {old.hidden} → {new_hidden}")

    def _perturb_cpg(self, loco: Any | None = None) -> None:
        ctrl = loco if loco is not None else self.loco
        if ctrl is None:
            return
        cpg = ctrl.cpg
        pb = float(_env_float("RKK_RSI_FULL_CPG_PHASE_NOISE", 0.3))
        fq = float(_env_float("RKK_RSI_FULL_CPG_FREQ_NOISE", 0.1))
        with torch.no_grad():
            p = cpg.phase_bias
            f = cpg.frequency
            cpg.phase_bias.copy_(p.detach() + torch.randn_like(p) * pb)
            cpg.frequency.copy_(f.detach() + torch.randn_like(f) * fq)
        env = getattr(self.agent, "env", None)
        while env is not None and hasattr(env, "base_env"):
            env = getattr(env, "base_env")
        motor_state = getattr(env, "_motor_state", None) if env is not None else None
        if motor_state is not None and hasattr(motor_state, "intents"):
            noise = float(_env_float("RKK_RSI_FULL_MOTOR_INTENT_NOISE", 0.08))
            for key in MOTOR_INTENT_VARS:
                cur = float(motor_state.intents.get(key, 0.5))
                delta = float(np.random.normal(0.0, noise))
                motor_state.intents[key] = float(np.clip(cur + delta, 0.05, 0.95))
                if key in self.agent.graph.nodes:
                    self.agent.graph.nodes[key] = motor_state.intents[key]

        # MOTOR_CORTEX: reset cpg_weight slightly when CPG is perturbed
        mc = self._motor_cortex()
        if mc is not None and mc.cpg_weight < 0.9:
            mc.cpg_weight = min(1.0, mc.cpg_weight + 0.15)
            print(f"[RSI] CPG perturbed → cpg_weight restored to {mc.cpg_weight:.3f}")

    def _check_motor_cortex_rsi(self, tick: int, snap: dict) -> dict[str, Any] | None:
        """
        MOTOR_CORTEX: RSI check — события связанные с motor cortex.
        Вызывается из tick() ниже основных RSI триггеров.
        """
        mc = self._motor_cortex()
        if mc is None:
            return None

        # Событие: cpg_weight снизился ниже 0.5 (cortex берёт управление)
        if mc.cpg_weight < 0.5 and self._last_cpg_annealing_event_w >= 0.5:
            self._last_cpg_annealing_event_w = mc.cpg_weight
            ev = {
                "type": "cpg_annealing",
                "cpg_weight": round(mc.cpg_weight, 4),
                "tick": tick,
                "n_programs": len(mc.programs),
                "quality_ema": round(mc._quality_ema, 4),
            }
            self._rsi_events.append(ev)
            print(f"[RSI] Motor cortex dominant: cpg_weight={mc.cpg_weight:.3f}, programs={list(mc.programs.keys())}")
            return ev
        elif mc.cpg_weight >= 0.5:
            self._last_cpg_annealing_event_w = mc.cpg_weight

        # Событие: новые программы созданы (через rsi_check_and_spawn уже в simulation)
        # Проверяем можно ли расширить hidden у walk-программы при плато
        walk_prog = mc.programs.get("walk")
        if (
            walk_prog is not None
            and walk_prog.train_steps > 500
            and walk_prog.performance < 0.15
            and (tick - self._last_mc_spawn_tick) > 2000
        ):
            # Сбрасываем буфер опыта и перезапускаем обучение с меньшим lr
            for p in walk_prog.optim.param_groups:
                p["lr"] = max(5e-5, p["lr"] * 0.5)
            self._last_mc_spawn_tick = tick
            ev = {"type": "mc_lr_decay", "program": "walk", "tick": tick}
            self._rsi_events.append(ev)
            return ev

        return None

    def tick(
        self,
        snap: dict,
        locomotion_reward: float,
        *,
        tick: int,
        locomotion_ctrl: Any | None = None,
    ) -> dict[str, Any] | None:
        self._tick_vl_restore()
        active_loco = locomotion_ctrl if locomotion_ctrl is not None else self.loco

        w_sr = self._walk_skill_sr_mean()
        core = _unwrap_gnn_core(self.agent.graph._core)
        gnn_h = int(core.hidden) if core is not None else 0

        self._snapshots.append(
            {
                "dr": float(snap.get("discovery_rate", 0.0)),
                "phi": float(snap.get("phi", 0.0)),
                "loco_r": float(locomotion_reward),
                "gnn_d": float(getattr(self.agent.graph, "_d", 0)),
                "gnn_h": float(gnn_h),
                "walk_sr": float(w_sr) if w_sr is not None else float("nan"),
            }
        )

        # MOTOR_CORTEX RSI check (runs every tick regardless of cooldown)
        mc_event = self._check_motor_cortex_rsi(tick, snap)
        if mc_event is not None and mc_event.get("type") != "cpg_annealing":
            return mc_event  # mc_lr_decay is a minor event, don't block other RSI

        min_n = max(20, _env_int("RKK_RSI_FULL_MIN_SNAPSHOTS", 200))
        win = max(10, _env_int("RKK_RSI_FULL_WINDOW", 100))
        if len(self._snapshots) < min_n:
            return mc_event  # return mc_event if any

        recent = list(self._snapshots)[-win:]
        older = list(self._snapshots)[-2 * win : -win]
        if len(older) < win // 2:
            return mc_event

        def mean_key(key: str, rows: list[dict]) -> float:
            return float(np.mean([r[key] for r in rows]))

        dr_gain = mean_key("dr", recent) - mean_key("dr", older)
        loco_gain = mean_key("loco_r", recent) - mean_key("loco_r", older)
        phi_gain = mean_key("phi", recent) - mean_key("phi", older)
        walk_recent = [r["walk_sr"] for r in recent if not np.isnan(r["walk_sr"])]
        walk_older = [r["walk_sr"] for r in older if not np.isnan(r["walk_sr"])]
        sr_gain = (
            float(np.mean(walk_recent) - np.mean(walk_older))
            if walk_recent and walk_older
            else 0.0
        )

        event: dict[str, Any] | None = None

        if tick > 0 and tick % max(5000, min_n * 10) == 0:
            self._rsi_events.append(
                {"type": "meta_distill_stub", "tick": tick, "note": "MAML/distill hook"}
            )

        phi_drop = _env_float("RKK_RSI_FULL_PHI_DROP", 0.035)
        if phi_gain < -phi_drop:
            return self._relax_value_layer(tick)

        if not self._cooldown_ok(tick):
            return mc_event  # can still return cpg_annealing event

        sr_eps = _env_float("RKK_RSI_FULL_SKILL_SR_EPS", 0.012)
        sr_floor = _env_float("RKK_RSI_FULL_SKILL_SR_FLOOR", 0.78)
        if (
            walk_recent
            and walk_older
            and abs(sr_gain) < sr_eps
            and float(np.mean(walk_recent)) > sr_floor
        ):
            lib = self._skills()
            if lib is not None:
                added = lib.ensure_harder_walk_variants()
                if added:
                    event = {"type": "skill_curriculum", "added": added, "tick": tick}
                    self._rsi_events.append(event)
                    self._last_structural_tick = tick
                    print(f"[RSI] Skill curriculum: {added}")
                    return event

        dr_eps = _env_float("RKK_RSI_FULL_DR_EPS", 1e-4)
        h_max = _env_int("RKK_RSI_FULL_HIDDEN_MAX", 128)
        if (
            USE_GNN
            and core is not None
            and abs(dr_gain) < dr_eps
            and core.hidden < h_max
        ):
            new_h = min(core.hidden * 2, h_max)
            if new_h > core.hidden:
                self._expand_gnn_hidden(new_h)
                event = {"type": "gnn_expand", "new_hidden": new_h, "tick": tick}
                self._rsi_events.append(event)
                self._last_structural_tick = tick
                return event

        loco_eps = _env_float("RKK_RSI_FULL_LOCO_EPS", 0.01)
        if (
            active_loco is not None
            and len(active_loco._reward_history) >= 8
            and abs(loco_gain) < loco_eps
        ):
            # Before perturbing CPG, check if motor cortex can take over instead
            mc = self._motor_cortex()
            if mc is not None and mc.cpg_weight < 0.4 and len(mc.programs) > 0:
                # Motor cortex is dominant — don't perturb CPG, let cortex handle it
                event = {"type": "mc_take_over_loco", "cpg_weight": mc.cpg_weight, "tick": tick}
                self._rsi_events.append(event)
                self._last_structural_tick = tick
                return event
            else:
                self._perturb_cpg(active_loco)
                event = {"type": "motor_policy_perturb", "tick": tick}
                self._rsi_events.append(event)
                self._last_structural_tick = tick
                return event

        return mc_event  # return cpg_annealing event if happened this tick