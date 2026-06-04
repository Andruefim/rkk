"""
Track C4 / C4b: latent confounder injection, online EM, TTL, ensemble sync.

High residual on a role cluster → inject latent_X (binary/k-ary) → online EM
with optional weak language prior (verbal / inner_voice / S2 text, not LLM oracle).
"""
from __future__ import annotations

import os
import re
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from engine.role_types import VALID_ROLE_TYPES


def _env_flag(key: str, default: bool = False) -> bool:
    raw = os.environ.get(key)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


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


def c4_enabled() -> bool:
    return _env_flag("RKK_C4_ENABLED", False)


def c4_active_global() -> bool:
    """Module-level fallback flag; manager also tracks per-instance active."""
    return _C4_ACTIVE_GLOBAL


_C4_ACTIVE_GLOBAL: bool = True


def set_c4_active_global(active: bool) -> None:
    global _C4_ACTIVE_GLOBAL
    _C4_ACTIVE_GLOBAL = bool(active)


@dataclass
class LatentRecord:
    node_id: str
    k_states: int
    value: int = 0
    inject_tick: int = 0
    role_cluster: str = ""
    target_nodes: list[str] = field(default_factory=list)
    baseline_residual: float = 0.0
    residual_at_inject: float = 0.0
    em_obs: deque = field(default_factory=lambda: deque(maxlen=32))
    inject_failures: int = 0
    ttl_passed: bool | None = None
    worlds_survived: list[str] = field(default_factory=list)
    edge_pairs: list[tuple[str, str, float]] = field(default_factory=list)

    def signature_vector(self, graph: Any) -> np.ndarray:
        """Compact edge-pattern signature for C5 promotion matching."""
        feats: list[float] = []
        for fr, to, w in self.edge_pairs[:12]:
            feats.extend(
                [
                    float(w),
                    float(graph.nodes.get(fr, 0.5)),
                    float(graph.nodes.get(to, 0.5)),
                ]
            )
        feats.append(float(self.k_states))
        feats.append(float(self.value))
        if not feats:
            return np.zeros(4, dtype=np.float64)
        v = np.asarray(feats, dtype=np.float64)
        n = np.linalg.norm(v)
        return v / max(n, 1e-8)


def signature_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return 0.0
    n = min(a.size, b.size)
    return float(np.dot(a[:n], b[:n]) / max(np.linalg.norm(a[:n]) * np.linalg.norm(b[:n]), 1e-8))


def _lang_prior_weight() -> float:
    return _env_float("RKK_LATENT_LANG_PRIOR_WEIGHT", 0.10)


def _lang_prior_min_corr() -> float:
    return _env_float("RKK_LATENT_LANG_PRIOR_MIN_CORR", 0.25)


def compute_cluster_pe(
    graph: Any,
    obs: dict[str, float] | None,
) -> dict[str, float]:
    """Per-node |obs - graph node| proxy for role-cluster residual detection."""
    obs = obs or {}
    out: dict[str, float] = {}
    for nid in graph._node_ids:
        if nid.startswith("latent_X_"):
            continue
        gv = float(graph.nodes.get(nid, 0.5))
        ov = float(obs.get(nid, gv))
        out[nid] = abs(ov - gv)
    return out


def collect_language_context(sim: Any | None) -> str:
    """Weak prior text: verbal + inner_voice + S2 macro (no LLM oracle)."""
    parts: list[str] = []
    if sim is None:
        return ""
    verbal = getattr(sim, "_verbal", None)
    if verbal is not None:
        snap = verbal.snapshot() if hasattr(verbal, "snapshot") else {}
        last = (snap or {}).get("last_message") or {}
        if isinstance(last, dict):
            parts.append(str(last.get("text", "")))
    iv = getattr(sim, "_inner_voice", None)
    if iv is not None and hasattr(iv, "get_concept_str"):
        parts.append(str(iv.get_concept_str()))
    s2 = getattr(sim, "_system2", None)
    if s2 is not None:
        parts.append(str(getattr(s2, "_active_macro", "")))
        parts.append(str(getattr(s2, "_last_source", "")))
    return " ".join(p for p in parts if p).strip()


def _tokenize(text: str) -> set[str]:
    return {t.lower() for t in re.findall(r"[a-zA-Z_]{3,}", text) if len(t) >= 3}


def language_state_prior(
    text: str,
    cluster_nodes: list[str],
    obs_row: dict[str, float],
    k_states: int,
) -> np.ndarray:
    """
    Map verbal/S2 tokens to a soft bias over latent states (length k).
    Returns log-prior offsets (sum need not be 1).
    """
    k = max(2, int(k_states))
    prior = np.zeros(k, dtype=np.float64)
    weight = _lang_prior_weight()
    if weight <= 0.0 or not text.strip():
        return prior
    tokens = _tokenize(text)
    if not tokens:
        return prior
    hits = 0
    for nid in cluster_nodes:
        nid_l = nid.lower()
        for tok in tokens:
            if tok in nid_l or nid_l in tok:
                hits += 1
                break
    if hits == 0:
        return prior
    mean_v = float(np.mean([float(obs_row.get(n, 0.5)) for n in cluster_nodes])) if cluster_nodes else 0.5
    corr_proxy = min(1.0, hits / max(len(cluster_nodes), 1))
    if corr_proxy < _lang_prior_min_corr():
        return prior
    # Split states by mean observation: low vs high (k=2) or tertiles (k=3)
    for s in range(k):
        center = (s + 0.5) / k
        prior[s] = weight * (1.0 - abs(mean_v - center))
    return prior


class LatentConfounderManager:
    """Per-graph latent confounder pipeline (C4 + C4b)."""

    def __init__(self) -> None:
        self.active: bool = True
        self._pe_baseline: float = 0.15
        self._pe_history: deque[float] = deque(maxlen=128)
        self._latents: dict[str, LatentRecord] = {}
        self._inject_failures: int = 0
        self._k_escalated: bool = False

    @property
    def c4_active(self) -> bool:
        return self.active and c4_active_global()

    def disable(self) -> None:
        self.active = False
        set_c4_active_global(False)

    def _update_baseline(self, pe: float) -> None:
        pe = max(0.0, float(pe))
        self._pe_history.append(pe)
        if self._pe_history:
            self._pe_baseline = max(
                0.05,
                0.92 * self._pe_baseline + 0.08 * float(np.mean(list(self._pe_history))),
            )

    def role_cluster_mean_pe(
        self,
        graph: Any,
        role: str,
        cluster_pe: dict[str, float] | None = None,
    ) -> float:
        """Mean per-node PE proxy for nodes in ``role`` (from last tick cluster_pe)."""
        if cluster_pe:
            vals = [
                float(cluster_pe[nid])
                for nid in graph._node_ids
                if graph.get_role_type(nid) == role and nid in cluster_pe
            ]
            if vals:
                return float(np.mean(vals))
        return self._pe_baseline

    def _high_residual_roles(
        self,
        graph: Any,
        cluster_pe: dict[str, float] | None,
    ) -> list[str]:
        thresh = _env_float("RKK_LATENT_RESIDUAL_THRESH", 0.30)
        baseline = max(self._pe_baseline, 0.05)
        cutoff = baseline * (1.0 + thresh)
        roles: list[str] = []
        seen: set[str] = set()
        for nid in graph._node_ids:
            role = graph.get_role_type(nid)
            if role not in VALID_ROLE_TYPES or role in seen:
                continue
            seen.add(role)
            rpe = self.role_cluster_mean_pe(graph, role, cluster_pe)
            if rpe > cutoff:
                roles.append(role)
        return roles

    def _cluster_nodes(self, graph: Any, role: str) -> list[str]:
        return [nid for nid in graph._node_ids if graph.get_role_type(nid) == role]

    def _sync_ensemble_latent(
        self,
        graph: Any,
        latent_id: str,
        edge_pairs: list[tuple[str, str, float]],
    ) -> None:
        ens = getattr(graph, "_ensemble", None)
        if ens is None or graph._core is None:
            return
        nids = graph._node_ids
        if latent_id not in nids:
            return
        fn = getattr(ens, "sync_latent_edges", None)
        if callable(fn):
            fn(nids, edge_pairs, latent_id=latent_id)

    def inject_latent(
        self,
        graph: Any,
        role: str,
        *,
        tick: int,
        residual: float,
        k_states: int | None = None,
    ) -> LatentRecord | None:
        if not self.c4_active or not c4_enabled():
            return None
        nodes = self._cluster_nodes(graph, role)
        if len(nodes) < 2:
            return None
        k = k_states if k_states is not None else _env_int("RKK_LATENT_MAX_STATES", 2)
        k = max(2, min(3, int(k)))
        latent_id = f"latent_X_{role}_{uuid.uuid4().hex[:6]}"
        targets = nodes[: min(6, len(nodes))]
        current_ids = list(graph._node_ids)
        if latent_id in current_ids:
            return None
        values = {nid: float(graph.nodes.get(nid, 0.5)) for nid in current_ids}
        values[latent_id] = 0.5
        graph.rebind_variables(current_ids + [latent_id], values, preserve_state=True)
        if hasattr(graph, "_maybe_init_ensemble"):
            graph._maybe_init_ensemble()
        edge_pairs: list[tuple[str, str, float]] = []
        for tgt in targets[:3]:
            graph.set_edge(latent_id, tgt, 0.35, 0.12)
            edge_pairs.append((latent_id, tgt, 0.35))
            graph.set_edge(tgt, latent_id, 0.18, 0.10)
            edge_pairs.append((tgt, latent_id, 0.18))
        self._sync_ensemble_latent(graph, latent_id, edge_pairs)
        rec = LatentRecord(
            node_id=latent_id,
            k_states=k,
            inject_tick=int(tick),
            role_cluster=role,
            target_nodes=targets,
            baseline_residual=float(self._pe_baseline),
            residual_at_inject=float(residual),
            edge_pairs=edge_pairs,
        )
        rec.em_obs = deque(maxlen=_env_int("RKK_LATENT_EM_WINDOW", 32))
        self._latents[latent_id] = rec
        return rec

    def _emission_log_prob(
        self,
        obs: dict[str, float],
        state: int,
        rec: LatentRecord,
    ) -> float:
        k = rec.k_states
        s = int(state) % k
        center = (s + 0.5) / k
        ll = 0.0
        for nid in rec.target_nodes:
            v = float(obs.get(nid, 0.5))
            sigma = 0.18
            ll += -0.5 * ((v - center) ** 2) / (sigma**2)
        return ll

    def infer_latent_value(
        self,
        obs: dict[str, float],
        rec: LatentRecord,
        lang_text: str = "",
    ) -> int:
        window = list(rec.em_obs)[-_env_int("RKK_LATENT_EM_WINDOW", 32) :]
        if not window:
            window = [obs]
        log_p = np.zeros(rec.k_states, dtype=np.float64)
        for row in window:
            for s in range(rec.k_states):
                log_p[s] += self._emission_log_prob(row, s, rec)
        lang_off = language_state_prior(
            lang_text, rec.target_nodes, obs, rec.k_states
        )
        log_p += lang_off
        val = int(np.argmax(log_p))
        rec.value = val
        if rec.node_id in obs:
            pass
        return val

    def _residual_reduction(self, rec: LatentRecord, current_residual: float) -> float:
        base = max(rec.residual_at_inject, 1e-6)
        return max(0.0, (rec.residual_at_inject - current_residual) / base)

    def prune_latent(self, graph: Any, latent_id: str) -> None:
        rec = self._latents.pop(latent_id, None)
        if rec is None:
            return
        for fr, to, _ in rec.edge_pairs:
            graph.remove_edge(fr, to)
        if latent_id not in graph._node_ids:
            return
        new_ids = [n for n in graph._node_ids if n != latent_id]
        values = {nid: float(graph.nodes.get(nid, 0.5)) for nid in new_ids}
        graph.rebind_variables(new_ids, values, preserve_state=True)
        if hasattr(graph, "_maybe_init_ensemble"):
            graph._maybe_init_ensemble()

    def _check_ttl(
        self,
        graph: Any,
        rec: LatentRecord,
        *,
        tick: int,
        cluster_pe: dict[str, float] | None,
        world_id: str = "humanoid",
    ) -> bool:
        age = int(tick) - int(rec.inject_tick)
        ttl = _env_int("RKK_LATENT_TTL_TICKS", 500)
        if age < ttl:
            return False
        rpe = self.role_cluster_mean_pe(graph, rec.role_cluster, cluster_pe)
        ig = self._residual_reduction(rec, rpe)
        min_ig = _env_float("RKK_LATENT_MIN_IG", 0.05)
        if ig >= min_ig:
            rec.ttl_passed = True
            if world_id and world_id not in rec.worlds_survived:
                rec.worlds_survived.append(world_id)
            return True
        rec.ttl_passed = False
        self.prune_latent(graph, rec.node_id)
        self._inject_failures += 1
        max_fail = _env_int("RKK_LATENT_MAX_INJECT_FAILURES", 5)
        if self._inject_failures >= max_fail:
            self.disable()
        elif _env_flag("RKK_LATENT_K_RETRY", True) and not self._k_escalated:
            self._k_escalated = True
            role = rec.role_cluster
            self.inject_latent(
                graph,
                role,
                tick=tick,
                residual=rpe,
                k_states=3,
            )
        return False

    def tick(
        self,
        graph: Any,
        *,
        engine_tick: int,
        prediction_error: float = 0.0,
        obs: dict[str, float] | None = None,
        cluster_pe: dict[str, float] | None = None,
        lang_text: str = "",
        world_id: str = "humanoid",
    ) -> dict[str, Any]:
        if not c4_enabled():
            return {"c4_enabled": False, "c4_active": False}
        obs = dict(obs or {})
        self._update_baseline(prediction_error)
        out: dict[str, Any] = {
            "c4_enabled": True,
            "c4_active": self.c4_active,
            "latent_count": len(self._latents),
            "inject_failures": self._inject_failures,
            "pe_baseline": round(self._pe_baseline, 5),
        }
        if not self.c4_active:
            return out

        # Infer existing latents
        for lid, rec in list(self._latents.items()):
            if lid not in graph._node_ids:
                self._latents.pop(lid, None)
                continue
            rec.em_obs.append(dict(obs))
            val = self.infer_latent_value(obs, rec, lang_text)
            graph.nodes[lid] = float(val) / max(rec.k_states - 1, 1)
            if rec.node_id in graph._node_meta:
                graph._node_meta[lid].value = graph.nodes[lid]
            self._check_ttl(
                graph, rec, tick=engine_tick, cluster_pe=cluster_pe, world_id=world_id
            )

        # Inject new latents for high-residual roles (one per tick max)
        if len(self._latents) < 3:
            for role in self._high_residual_roles(graph, cluster_pe):
                if any(r.role_cluster == role for r in self._latents.values()):
                    continue
                rpe = self.role_cluster_mean_pe(graph, role, cluster_pe)
                rec = self.inject_latent(
                    graph, role, tick=engine_tick, residual=rpe
                )
                if rec is not None:
                    out["injected"] = rec.node_id
                    out["role_cluster"] = role
                    break

        if self._latents:
            last = next(reversed(self._latents.values()))
            out["latent_X"] = last.node_id
            out["latent_value"] = int(last.value)
            out["latent_k"] = int(last.k_states)
        out["c4_active"] = self.c4_active
        out["latent_count"] = len(self._latents)
        return out

    def snapshot(self) -> dict[str, Any]:
        latents = []
        for rec in self._latents.values():
            latents.append(
                {
                    "node_id": rec.node_id,
                    "k": rec.k_states,
                    "value": rec.value,
                    "role_cluster": rec.role_cluster,
                    "ttl_passed": rec.ttl_passed,
                    "worlds_survived": list(rec.worlds_survived),
                    "inject_failures": rec.inject_failures,
                }
            )
        return {
            "c4_active": self.c4_active,
            "c4_enabled": c4_enabled(),
            "inject_failures": self._inject_failures,
            "latents": latents,
            "pe_baseline": round(self._pe_baseline, 5),
        }

    def records_passed_ttl(self) -> list[LatentRecord]:
        return [r for r in self._latents.values() if r.ttl_passed]
