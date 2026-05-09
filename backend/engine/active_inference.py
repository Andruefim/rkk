"""
active_inference.py — Homeostatic Causal Control (System 1).

Минимизация расхождения предсказания WM с гомеостатическими целями через градиент
только по узлам намерений, которые отображаются в MOTOR_INTENT_VARS среды humanoid.
"""
from __future__ import annotations

import os
from typing import Any

import torch
import torch.nn.functional as F

from engine.features.humanoid.constants import MOTOR_INTENT_VARS
from engine.graph_constants import is_read_only_macro_var


def _resolve_target_pairs(
    node_ids: list[str], target_priors: dict[str, float]
) -> list[tuple[int, float]]:
    """Сопоставление ключей приоров с индексами узлов (short / phys_ / без префикса)."""
    pairs: list[tuple[int, float]] = []
    seen: set[int] = set()
    for raw_k, raw_v in target_priors.items():
        key = str(raw_k)
        candidates = [key]
        if key.startswith("phys_"):
            candidates.append(key[5:])
        else:
            candidates.append("phys_" + key)
        idx = None
        for c in candidates:
            if c in node_ids:
                idx = node_ids.index(c)
                break
        if idx is None:
            continue
        if idx in seen:
            continue
        seen.add(idx)
        pairs.append((idx, float(raw_v)))
    return pairs


def _intent_parameter_indices(node_ids: list[str]) -> list[int]:
    """
    Индексы узлов motor intent в графе, совпадающих с именами в humanoid MOTOR_INTENT_VARS
    (короткое имя или phys_intent_<name>). Агрегаты l1_* не оптимизируем — в среду не мапятся.
    """
    out: list[int] = []
    for i, nid in enumerate(node_ids):
        if is_read_only_macro_var(nid):
            continue
        s = str(nid)
        if s in MOTOR_INTENT_VARS:
            out.append(i)
            continue
        if s.startswith("phys_intent_"):
            suf = s[len("phys_intent_") :]
            if suf in MOTOR_INTENT_VARS:
                out.append(i)
    return out


def _eff_lr() -> float:
    try:
        return float(os.environ.get("RKK_ACTIVE_INF_LR", "0.12"))
    except ValueError:
        return 0.12


def _eff_iters() -> int:
    try:
        return max(1, int(os.environ.get("RKK_ACTIVE_INF_ITERS", "12")))
    except ValueError:
        return 12


class HomeostaticController:
    def __init__(self, device: torch.device, learning_rate: float = 0.1, max_iters: int = 15):
        self.device = device
        self.lr = float(learning_rate)
        self.max_iters = int(max_iters)
        self._last_free: torch.Tensor | None = None
        self._last_free_iv: list[int] | None = None

    def reset(self) -> None:
        self._last_free = None
        self._last_free_iv = None

    def optimize_action(
        self,
        current_state: dict[str, float],
        graph: Any,
        target_priors: dict[str, float],
    ) -> dict[str, float]:
        if getattr(graph, "_core", None) is None or not target_priors:
            return {}

        node_ids = list(graph._node_ids)
        d = len(node_ids)
        if d < 1:
            return {}

        iv = _intent_parameter_indices(node_ids)
        if not iv:
            return {}

        pairs = _resolve_target_pairs(node_ids, target_priors)
        if not pairs:
            return {}

        target_ix = [p[0] for p in pairs]
        target_vals = torch.tensor(
            [[p[1] for p in pairs]], dtype=torch.float32, device=self.device
        )
        ix_t = torch.tensor(target_ix, dtype=torch.long, device=self.device)

        x_list = [float(current_state.get(nid, graph.nodes.get(nid, 0.5))) for nid in node_ids]
        X = torch.tensor([x_list], dtype=torch.float32, device=self.device)

        iv_t = torch.tensor(iv, dtype=torch.long, device=self.device)
        # Тёплый старт: предыдущий свободный вектор или текущее наблюдение по intent-слотам
        if (
            self._last_free is not None
            and self._last_free_iv == iv
            and self._last_free.numel() == len(iv)
        ):
            init_free = self._last_free.clone().to(self.device)
        else:
            init_free = X[0, iv_t].detach().clone()

        A_free = torch.nn.Parameter(torch.clamp(init_free, 0.0, 1.0))

        lr = _eff_lr()
        iters = min(self.max_iters, _eff_iters())
        optimizer = torch.optim.Adam([A_free], lr=lr)

        pen_w = float(os.environ.get("RKK_ACTIVE_INF_ACTION_PEN", "0.015"))
        try:
            pen_w = float(pen_w)
        except ValueError:
            pen_w = 0.015

        initial_free = A_free.detach().clone()

        for it in range(iters):
            optimizer.zero_grad()
            A_full = X.detach().clone()
            A_full[0, iv_t] = A_free
            predicted_X = graph.forward_dynamics(X, A_full)
            pred_t = predicted_X[0, ix_t].unsqueeze(0)
            pred_c = torch.clamp(pred_t, 0.0, 1.0)
            targ_c = torch.clamp(target_vals, 0.0, 1.0)
            loss = F.mse_loss(pred_c, targ_c)
            action_penalty = pen_w * ((A_free - 0.5) ** 2).mean()
            total_loss = loss + action_penalty
            total_loss.backward()
            if it == 0:
                gn = (
                    float(A_free.grad.abs().max().item())
                    if A_free.grad is not None
                    else 0.0
                )
                print(
                    f"[ACTIVE INF] Iter 0: Loss {total_loss.item():.4f}, Max Grad {gn:.6f}"
                )
            optimizer.step()
            with torch.no_grad():
                A_free.clamp_(0.0, 1.0)

        self._last_free = A_free.detach().clone()
        self._last_free_iv = list(iv)

        final_free = A_free.detach().cpu().numpy()
        initial_np = initial_free.detach().cpu().numpy()
        optimized_actions: dict[str, float] = {}
        for k, idx in enumerate(iv):
            nid = node_ids[idx]
            val = float(final_free[k])
            if abs(val - float(initial_np[k])) > 1e-4:
                optimized_actions[nid] = val

        # Лог после последней итерации
        with torch.no_grad():
            A_full = X.detach().clone()
            A_full[0, iv_t] = A_free.detach()
            predicted_X = graph.forward_dynamics(X, A_full)
            curr_vals = X[0, ix_t].cpu().numpy()
            pred_vals = torch.clamp(predicted_X[0, ix_t], 0.0, 1.0).cpu().numpy()
            pred_raw = predicted_X[0, ix_t].cpu().numpy()
            tl = F.mse_loss(
                torch.clamp(predicted_X[0, ix_t], 0.0, 1.0),
                torch.clamp(target_vals.squeeze(0), 0.0, 1.0),
            )
            active_targets = [node_ids[i] for i in target_ix]
            print(
                f"[ACTIVE INF] Targets: {active_targets} | Curr: {curr_vals.tolist()} | "
                f"Pred: {pred_raw.tolist()} (clipped {pred_vals.tolist()}) | Loss: {tl.item():.6f}"
            )
            if optimized_actions:
                top_acts = sorted(
                    optimized_actions.items(),
                    key=lambda x: abs(x[1] - 0.5),
                    reverse=True,
                )[:3]
                print(f"[ACTIVE INF] Top Actions: {top_acts}")

        return optimized_actions
