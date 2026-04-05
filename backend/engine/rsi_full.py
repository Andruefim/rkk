"""
Full RSI loop for embodied AGI (Phase C):
1. Meta-learning: hooks for MAML-like adaptation (stub events / future extension).
2. Architecture search: NAS lite — expand GNN hidden dim when discovery plateaus.
3. Self-curriculum: harder walk skills when motor skills plateau.
4. Knowledge distillation: placeholder for compressing skills (robot deployment).

Отдельно от rsi_lite (L1/buffer/imagination). Включается RKK_RSI_FULL=1.
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

if TYPE_CHECKING:
    from engine.agent import RKKAgent
    from engine.cpg_locomotion import LocomotionController
    from engine.skill_library import SkillLibrary


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
    """Новый core с большим hidden; W и пересекающиеся слои MLP копируются."""
    d, dev = old.d, old.device
    oh, nh = old.hidden, new_hidden
    nmin = min(oh, nh)

    new = CausalGNNCore(d, dev, hidden=new_hidden)
    with torch.no_grad():
        new.W.copy_(old.W)

    def copy_linear(src: nn.Linear, dst: nn.Linear) -> None:
        so, si = src.weight.shape
        do, di = dst.weight.shape
        o_ = min(so, do)
        i_ = min(si, di)
        dst.weight[:o_, :i_].copy_(src.weight[:o_, :i_])
        if src.bias is not None and dst.bias is not None:
            b_ = min(src.bias.shape[0], dst.bias.shape[0])
            dst.bias[:b_].copy_(src.bias[:b_])

    # node_enc / action_enc: Linear(1, hidden)
    copy_linear(old.node_enc[0], new.node_enc[0])
    copy_linear(old.action_enc[0], new.action_enc[0])

    # msg_fn: Linear(2*h,h) -> Linear(h,h)
    copy_linear(old.msg_fn[0], new.msg_fn[0])
    copy_linear(old.msg_fn[2], new.msg_fn[2])

    # out_dec: Linear(2*h,h) -> Linear(h,1)
    copy_linear(old.out_dec[0], new.out_dec[0])
    copy_linear(old.out_dec[2], new.out_dec[2])

    # Остальные параметры уже инициализированы Xavier; при nh>oh новые строки — свежие.
    if nh > oh:
        with torch.no_grad():
            for seq in (new.node_enc, new.action_enc):
                nn.init.xavier_uniform_(seq[0].weight[oh:nh], gain=0.5)
                nn.init.zeros_(seq[0].bias[oh:nh])
            nn.init.xavier_uniform_(new.msg_fn[0].weight[oh:nh], gain=0.5)
            nn.init.zeros_(new.msg_fn[0].bias[oh:nh])
            nn.init.xavier_uniform_(new.msg_fn[2].weight[oh:nh, :], gain=0.5)
            nn.init.xavier_uniform_(new.msg_fn[2].weight[:, oh:nh], gain=0.5)
            nn.init.zeros_(new.msg_fn[2].bias[oh:nh])
            nn.init.xavier_uniform_(new.out_dec[0].weight[oh:nh], gain=0.5)
            nn.init.zeros_(new.out_dec[0].bias[oh:nh])
            nn.init.xavier_uniform_(new.out_dec[2].weight[:, oh:nh], gain=0.5)

    return new


class RSIController:
    """
    Следит за прогрессом агента и запускает структурную самонастройку.

    Триггеры:
    - падение phi → временное смягчение VL (phi_min, phi_min_steady)
    - плато success walk-скиллов → более жёсткие варианты шага
    - плато discovery_rate при GNN → расширение hidden (NAS lite)
    - плато locomotion reward → шум по CPG (phase_bias, frequency)
    """

    def __init__(
        self,
        agent: RKKAgent,
        locomotion_ctrl: LocomotionController | None = None,
        skill_library_supplier: Callable[[], SkillLibrary | None] | None = None,
    ):
        self.agent = agent
        self.loco = locomotion_ctrl
        self._skill_supplier = skill_library_supplier

        maxlen = max(50, _env_int("RKK_RSI_FULL_SNAPSHOT_CAP", 500))
        self._snapshots: deque[dict[str, float]] = deque(maxlen=maxlen)
        self._rsi_events: list[dict[str, Any]] = []
        self._last_structural_tick = -10**9
        self._vl_backup: tuple[float, float] | None = None
        self._vl_relax_remaining = 0

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
            cpg.phase_bias.add_(torch.randn_like(cpg.phase_bias) * pb)
            cpg.frequency.add_(torch.randn_like(cpg.frequency) * fq)

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

        min_n = max(20, _env_int("RKK_RSI_FULL_MIN_SNAPSHOTS", 200))
        win = max(10, _env_int("RKK_RSI_FULL_WINDOW", 100))
        if len(self._snapshots) < min_n:
            return None

        recent = list(self._snapshots)[-win:]
        older = list(self._snapshots)[-2 * win : -win]
        if len(older) < win // 2:
            return None

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

        # Периодический stub для мета-learning / distillation (расширение без тяжёлой логики)
        if tick > 0 and tick % max(5000, min_n * 10) == 0:
            self._rsi_events.append(
                {"type": "meta_distill_stub", "tick": tick, "note": "MAML/distill hook"}
            )

        phi_drop = _env_float("RKK_RSI_FULL_PHI_DROP", 0.035)
        if phi_gain < -phi_drop:
            return self._relax_value_layer(tick)

        if not self._cooldown_ok(tick):
            return None

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
            self._perturb_cpg(active_loco)
            event = {"type": "cpg_perturb", "tick": tick}
            self._rsi_events.append(event)
            self._last_structural_tick = tick
            return event

        return None
