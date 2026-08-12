"""Bind-time visual object resolve — language↔vision via embeddings only.

No keyword / deictic / typo / spatial-language heuristics.
Selection score is cosine similarity in embedding space:
  command ↔ slot label  and/or  command ↔ concept names projected from slot vectors.
Geometry (UV → bearing, depth → range_m) uses attention×foreground metric depth.
"""
from __future__ import annotations

import os
from typing import Any, Callable

import numpy as np

from engine.vision_depth import DepthCamera, attach_range_to_target
from engine.vision_target import VisualTarget, bearing_from_u

EmbedFn = Callable[[str], np.ndarray | None]
# slot_vector → [(concept_name, score), ...]
ConceptProjectFn = Callable[[np.ndarray], list[tuple[str, float]]]

# Agent interoceptive / curriculum / body-gait concepts — not visual object names.
_NON_VISUAL_CONCEPTS = frozenset(
    {
        "BREAKTHROUGH",
        "HIGH_EMPOWERMENT",
        "EMPOWER",
        "EMPOWERMENT",
        "TEMPORAL",
        "INTENT",
        "CURIOSITY",
        "NOVELTY",
        "SURPRISE",
        "REWARD",
        "PAIN",
        "FATIGUE",
        "BALANCE",
        "FALLEN",
        "AUTONOMY",
        "HOME",
        "SELF",
        "SWING",
        "STANCE",
        "PHASE",
        "CPG",
        "GAIT",
        "STRIDE",
        "SUPPORT",
        "ANKLE",
        "KNEE",
        "HIP",
        "TORSO",
        "FOOT",
        "LEG",
        "ARM",
        "SHOULDER",
        "WRIST",
        "ELBOW",
        "HEAD",
        "NECK",
        "HAND",
        "JOINT",
        "LEARNING",
        "OPPORTUNITY",
        "OPPORTUNITIES",
        # Skill / curriculum leakage from NeuralConceptProjector
        "SKILL",
        "NONE",
        "WALK",
        "STAND",
        "ACTIVE",
        "TRANSITION",
        "WEIGHT",
        "ANOMALY",
        "NOMINAL",
        "STRESS",
        "CRITICAL",
        "PASSIVE",
        "EXPLOITING",
        "KNOWN",
        "IMPROVING",
        "DEGRADING",
        "PLATEAU",
        "LATERAL",
        "TILT",
        "ROLL",
        "PITCH",
        "YAW",
    }
)

_NON_VISUAL_PREFIXES = (
    "SKILL_",
    "INTENT_",
    "WEIGHT_",
    "JOINT_",
    "LATENT_",
    "HIGH_",
    "LOW_",
    "LEARNING_",
    "LATERAL_",
    "TILT_",
)


def _is_visual_concept(name: str) -> bool:
    s = str(name or "").strip()
    if not s or s.startswith("LATENT_") or s.startswith("[EGO]"):
        return False
    upper = s.upper().replace("-", "_").replace(" ", "_")
    if any(upper.startswith(p) for p in _NON_VISUAL_PREFIXES):
        return False
    tokens = [t for t in upper.split("_") if t]
    if any(t in _NON_VISUAL_CONCEPTS for t in tokens):
        return False
    if upper in _NON_VISUAL_CONCEPTS:
        return False
    for bad in _NON_VISUAL_CONCEPTS:
        if upper.startswith(bad + "_") or upper.endswith("_" + bad):
            return False
    return True


def hud_safe_label(label: str, *, fallback: str = "target") -> str:
    """Strip body/gait/concept leakage for camera HUD."""
    s = str(label or "").strip()
    if s and _is_visual_concept(s):
        return s[:24]
    return str(fallback or "target")[:24]


def _env_float(key: str, default: float) -> float:
    try:
        return float(os.environ.get(key, str(default)))
    except ValueError:
        return float(default)


def vision_min_confidence() -> float:
    return _env_float("RKK_VISION_RESOLVE_MIN_CONF", 0.35)


def mask_peakiness_min() -> float:
    """min (max/mean) for treating a slot mask as spatially localized."""
    return _env_float("RKK_SLOT_MASK_PEAKINESS_MIN", 1.8)


def objectness_bind_enabled() -> bool:
    """
    When SlotAttention UV is diffuse, allow depth objectness-peak bind if
    language↔ontology matched. Camera-only (not sim_oracle). Default on —
    required for neural-primary resolve with RKK_SIM_ORACLE_BIND=0.
    """
    raw = os.environ.get("RKK_VISION_OBJECTNESS_BIND", "1").strip().lower()
    return raw not in ("0", "false", "no", "off")


def objectness_bind_min_peak() -> float:
    return _env_float("RKK_VISION_OBJECTNESS_BIND_MIN_PEAK", 0.18)


def _normalize(v: np.ndarray | None) -> np.ndarray | None:
    if v is None:
        return None
    a = np.asarray(v, dtype=np.float32).reshape(-1)
    n = float(np.linalg.norm(a))
    if n < 1e-9:
        return None
    return a / n


def _cosine(a: np.ndarray | None, b: np.ndarray | None) -> float:
    aa, bb = _normalize(a), _normalize(b)
    if aa is None or bb is None or aa.shape != bb.shape:
        return 0.0
    return float(np.clip(np.dot(aa, bb), -1.0, 1.0))


def _mask_peakiness(m: np.ndarray) -> float:
    flat = np.asarray(m, dtype=np.float64).reshape(-1)
    mean = float(flat.mean()) + 1e-9
    return float(flat.max() / mean)


def _mask_bbox(
    mask: np.ndarray | None, *, thresh: float = 0.25
) -> tuple[float, float, float, float] | None:
    """Attention-mask bbox in normalized UV (u_min, v_min, u_max, v_max)."""
    if mask is None:
        return None
    try:
        m = np.asarray(mask, dtype=np.float64)
    except Exception:
        return None
    if m.ndim != 2 or m.size == 0:
        return None
    peak = float(m.max())
    if peak <= 1e-9:
        return None
    ys, xs = np.where(m >= (thresh * peak))
    if xs.size == 0 or ys.size == 0:
        return None
    h, w = m.shape
    u_min = float(xs.min()) / max(w - 1, 1)
    u_max = float(xs.max()) / max(w - 1, 1)
    v_min = float(ys.min()) / max(h - 1, 1)
    v_max = float(ys.max()) / max(h - 1, 1)
    return (u_min, v_min, u_max, v_max)


def is_self_vision_slot(slot_or_meta: dict[str, Any] | None, label: str = "") -> bool:
    """True for body/self slots that must not enter objectness / scene bind."""
    meta = dict(slot_or_meta or {})
    if bool(meta.get("self_slot")):
        return True
    src = str(meta.get("source") or "").strip().lower()
    if src == "grounding":
        return True
    lab = str(label or meta.get("label") or "").strip()
    if lab.lower().startswith("[ego]"):
        return True
    return False


def collect_vision_slots(visual_env: Any | None) -> list[dict[str, Any]]:
    """Build slot candidates from EnvironmentVisual-like object."""
    if visual_env is None:
        return []

    slots: list[dict[str, Any]] = []
    lexicon = dict(getattr(visual_env, "_slot_lexicon", None) or {})
    attn = getattr(visual_env, "_last_attn", None)
    vals = getattr(visual_env, "_last_slots", None)
    vecs = getattr(visual_env, "_last_slot_vecs", None)

    positions: dict[str, tuple[float, float, float, np.ndarray | None]] = {}
    if attn is not None:
        try:
            if hasattr(attn, "detach"):
                masks = attn.detach().float().cpu().numpy()
            else:
                masks = np.asarray(attn, dtype=np.float32)
            if masks.ndim == 3:
                k, h, w = masks.shape
                ys = np.linspace(0.0, 1.0, h, dtype=np.float64)[:, None]
                xs = np.linspace(0.0, 1.0, w, dtype=np.float64)[None, :]
                peak_min = mask_peakiness_min()
                for i in range(k):
                    m = masks[i]
                    tot = float(m.sum())
                    if tot < 1e-6:
                        continue
                    peak = _mask_peakiness(m)
                    u = float((m * xs).sum() / tot)
                    v = float((m * ys).sum() / tot)
                    # Diffuse / untrained masks collapse to center — mark invalid UV
                    uv_ok = peak >= peak_min
                    positions[f"slot_{i}"] = (u, v, peak, np.asarray(m, dtype=np.float32))
                    if not uv_ok:
                        # Keep mask for attention-guided depth; UV will be recomputed
                        positions[f"slot_{i}"] = (u, v, peak, np.asarray(m, dtype=np.float32))
        except Exception:
            positions = {}

    n = 0
    if vals is not None:
        try:
            if hasattr(vals, "detach"):
                n = int(vals.numel())
            else:
                n = int(np.asarray(vals).size)
        except Exception:
            n = 0
    if n <= 0:
        n = max(len(lexicon), len(positions), 0)

    peak_min = mask_peakiness_min()
    for i in range(n):
        sid = f"slot_{i}"
        meta = lexicon.get(sid) or {}
        label = str(meta.get("label") or "")
        # Skip body/self even when grounding is off but an [EGO] label already exists.
        if is_self_vision_slot(meta, label):
            continue
        if sid in positions:
            u, v, peak, mask = positions[sid]
            uv_valid = bool(peak >= peak_min)
        else:
            u, v, peak, mask = 0.5, 0.55, 0.0, None
            uv_valid = False
        # Drop non-visual lexicon labels from concept store leakage
        if label and not _is_visual_concept(label):
            label = ""
        act = 0.5
        if vals is not None:
            try:
                if hasattr(vals, "detach"):
                    act = float(vals.detach().flatten()[i].item())
                else:
                    act = float(np.asarray(vals).reshape(-1)[i])
            except Exception:
                act = 0.5
        vec = None
        if vecs is not None:
            try:
                if hasattr(vecs, "detach"):
                    vec = vecs.detach().float().cpu().numpy()[i]
                else:
                    vec = np.asarray(vecs, dtype=np.float32)[i]
            except Exception:
                vec = None
        slots.append(
            {
                "slot_id": sid,
                "u": float(u),
                "v": float(v),
                "label": label,
                "activation": float(act),
                "vector": vec,
                "lex_conf": float(meta.get("confidence") or 0.0),
                "uv_valid": bool(uv_valid),
                "mask_peakiness": float(peak),
                "attn_mask": mask,
                "bbox": _mask_bbox(mask),
                "self_slot": bool(meta.get("self_slot")),
                "source": str(meta.get("source") or ""),
            }
        )
    return slots


def _embed_score(
    cmd_emb: np.ndarray,
    text: str,
    embed_fn: EmbedFn,
    cache: dict[str, np.ndarray | None],
) -> float:
    key = str(text or "").strip()
    if not key:
        return 0.0
    if key not in cache:
        try:
            cache[key] = _normalize(embed_fn(key))
        except Exception:
            cache[key] = None
    return max(0.0, _cosine(cmd_emb, cache[key]))


def score_slots_for_command(
    slots: list[dict[str, Any]],
    command_text: str,
    *,
    embed_fn: EmbedFn,
    concept_project_fn: ConceptProjectFn | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Score each slot by embedding similarity to the command.

    Language↔vision links (any one suffices):
      1. embed(command) ↔ embed(slot label)
      2. embed(command) ↔ embed(concept) from NeuralConceptProjector(slot_vec)
      3. embed(command) ↔ visual-referent ontology, then bind to percepts by
         SlotAttention activation (cortical saliency — not linguistic rules)
    """
    meta: dict[str, Any] = {}
    text = str(command_text or "").strip()
    if not text:
        meta["reason"] = "empty_command"
        return [], meta

    try:
        cmd_emb = _normalize(embed_fn(text))
    except Exception:
        cmd_emb = None
    if cmd_emb is None:
        meta["reason"] = "command_embed_failed"
        return [], meta

    emb_cache: dict[str, np.ndarray | None] = {text: cmd_emb}
    out: list[dict[str, Any]] = []
    any_language_link = False

    # Ontology referent (same style as goal_grounding predicate catalog).
    ont_entry = None
    ont_sc = 0.0
    try:
        from engine.visual_referent_ontology import match_visual_referent

        ont_entry, ont_sc, ont_diag = match_visual_referent(text, embed_fn)
        meta["ontology"] = ont_diag
    except Exception as exc:
        meta["ontology_error"] = str(exc)

    acts = [max(0.0, float(s.get("activation") or 0.0)) for s in slots]
    act_max = max(acts) if acts else 0.0
    act_sum = sum(acts) + 1e-9

    for s in slots:
        s2 = dict(s)
        label = str(s.get("label") or "").strip()
        if label and not _is_visual_concept(label):
            label = ""
        label_sc = 0.0
        if label and not label.lower().startswith("[ego]"):
            label_sc = _embed_score(cmd_emb, label, embed_fn, emb_cache)
            if label_sc > 0.0:
                any_language_link = True

        concept_sc = 0.0
        concept_hit = ""
        vec = s.get("vector")
        if concept_project_fn is not None and vec is not None:
            try:
                projected = concept_project_fn(np.asarray(vec, dtype=np.float32))
            except Exception:
                projected = []
            for name, pscore in projected or []:
                name_s = str(name or "").strip()
                if not _is_visual_concept(name_s):
                    continue
                sc = _embed_score(cmd_emb, name_s, embed_fn, emb_cache)
                sc = sc * float(max(0.0, min(1.0, pscore)))
                if sc > concept_sc:
                    concept_sc = sc
                    concept_hit = name_s
                    any_language_link = True

        # Ontology × cortical saliency (activation / winner-take-all mass).
        ont_slot_sc = 0.0
        act = float(s.get("activation") or 0.0)
        if ont_entry is not None and ont_sc >= 0.25 and act_max > 1e-6:
            salience = act / act_max
            # Soft mass so a single dominant slot wins without linguistic rules.
            mass = act / act_sum
            ont_slot_sc = float(ont_sc) * (0.65 * salience + 0.35 * mass)
            if ont_slot_sc > 0.0:
                any_language_link = True

        score = float(max(label_sc, concept_sc, ont_slot_sc))
        if act < 0.02 and score < 0.5 and ont_slot_sc <= 0.0:
            score = 0.0

        # Diffuse SlotAttention masks (uv_valid=false) are not object locks —
        # they collapse to image center / floor. Strongly down-weight them so a
        # peaked slot can win, and ontology×activation alone cannot pick noise.
        uv_raw = s.get("uv_valid")
        if uv_raw is None:
            # Legacy slots without mask metadata — no penalty
            uv_valid = True
        else:
            uv_valid = bool(uv_raw)
        peak = float(s.get("mask_peakiness") or 0.0)
        peak_min = mask_peakiness_min()
        if uv_raw is False:
            # Ontology-only on a blob of noise ≈ random walk to floor.
            if label_sc < 0.35 and concept_sc < 0.35:
                score *= 0.15
            else:
                score *= 0.45
        elif uv_valid and peak_min > 1e-6 and peak > 0.0:
            score *= float(0.55 + 0.45 * min(1.0, peak / peak_min))

        s2["match_score"] = score
        s2["match_label"] = float(label_sc)
        s2["match_concept"] = float(concept_sc)
        s2["match_ontology"] = float(ont_slot_sc)
        s2["uv_valid"] = uv_valid
        if not label:
            if concept_hit:
                s2["label"] = concept_hit
            elif ont_entry is not None and ont_slot_sc >= max(label_sc, concept_sc):
                s2["label"] = str(ont_entry.key)
        out.append(s2)

    if not any_language_link:
        meta["reason"] = "no_language_vision_link"
    return out, meta


def _cap_spatial_confidence(target: VisualTarget) -> VisualTarget:
    """HUD/control confidence: semantic match × depth × spatial peak strength."""
    diags = dict(target.diagnostics or {})
    match = float(target.confidence)
    rc = float(target.range_conf) if target.range_conf is not None else match
    conf = min(match, rc)
    if diags.get("geometry") == "objectness_peak":
        pstr = float(diags.get("objectness_peak_strength") or 0.0)
        conf = min(conf, 0.15 + 0.85 * pstr)
    return VisualTarget(
        slot_id=target.slot_id,
        u=target.u,
        v=target.v,
        label=target.label,
        confidence=float(max(0.0, min(1.0, conf))),
        bearing=target.bearing,
        range_m=target.range_m,
        range_var=target.range_var,
        range_conf=target.range_conf,
        bbox=target.bbox,
        diagnostics=diags,
        latent=list(target.latent) if target.latent else None,
    )


def _slot_latent_list(cand: dict[str, Any]) -> list[float] | None:
    raw = cand.get("latent")
    if raw is None:
        raw = cand.get("vector")
    if raw is None:
        return None
    try:
        if hasattr(raw, "detach"):
            arr = raw.detach().float().cpu().numpy().reshape(-1)
        else:
            arr = np.asarray(raw, dtype=np.float64).reshape(-1)
        out = [float(x) for x in arr.tolist()]
        return out if out else None
    except Exception:
        return None


def _apply_metric_geometry(
    cand: dict[str, Any],
    depth_camera: DepthCamera | None,
) -> VisualTarget:
    """Build VisualTarget and attach attention-guided (or UV) metric range."""
    u = float(cand.get("u", 0.5))
    v = float(cand.get("v", 0.55))
    lat = _slot_latent_list(cand)
    diags: dict[str, Any] = {
        "match_label": cand.get("match_label"),
        "match_concept": cand.get("match_concept"),
        "match_ontology": cand.get("match_ontology"),
        "uv_valid": cand.get("uv_valid"),
        "mask_peakiness": cand.get("mask_peakiness"),
    }
    if lat:
        diags["latent"] = lat
        diags["latent_dim"] = len(lat)
    bbox = cand.get("bbox")
    if bbox is None:
        bbox = _mask_bbox(cand.get("attn_mask"))
    target = VisualTarget(
        slot_id=str(cand["slot_id"]),
        u=u,
        v=v,
        label=str(cand.get("label") or "visual_referent"),
        confidence=float(cand.get("match_score") or 0.0),
        bearing=bearing_from_u(u),
        bbox=bbox if isinstance(bbox, tuple) else None,
        diagnostics=diags,
        latent=lat,
    )
    if depth_camera is None:
        return target
    mask = cand.get("attn_mask")
    # Diffuse / invalid UV: ignore SlotAttention mask — lock to depth objectness
    # peak (protrusions), not the floor strip under the image center.
    if not bool(cand.get("uv_valid")) and cand.get("uv_valid") is not None:
        mask = None
        diags = dict(target.diagnostics or {})
        diags["geometry"] = "objectness_peak"
        target = VisualTarget(
            slot_id=target.slot_id,
            u=target.u,
            v=target.v,
            label=target.label,
            confidence=target.confidence,
            bearing=target.bearing,
            range_m=target.range_m,
            range_var=target.range_var,
            range_conf=target.range_conf,
            bbox=target.bbox,
            diagnostics=diags,
            latent=list(target.latent) if target.latent else None,
        )
    out = attach_range_to_target(target, depth_camera, attn_mask=mask)
    return _cap_spatial_confidence(out)


def resolve_visual_target(
    command_text: str,
    *,
    visual_env: Any | None = None,
    slots: list[dict[str, Any]] | None = None,
    depth_camera: DepthCamera | None = None,
    embed_fn: EmbedFn | None = None,
    concept_project_fn: ConceptProjectFn | None = None,
    require_range: bool = True,
) -> tuple[VisualTarget | None, dict[str, Any]]:
    """
    Resolve a VisualTarget from ego vision via embedding grounding only.
    Never reads scene registry / body_id.
    """
    diag: dict[str, Any] = {"mode": "vision", "reason": ""}
    if embed_fn is None:
        diag["reason"] = "no_embed_fn"
        return None, diag

    raw = list(slots) if slots is not None else collect_vision_slots(visual_env)
    raw = [s for s in raw if not is_self_vision_slot(s)]
    if not raw:
        diag["reason"] = "no_vision_slots"
        return None, diag

    scored, score_meta = score_slots_for_command(
        raw,
        command_text,
        embed_fn=embed_fn,
        concept_project_fn=concept_project_fn,
    )
    if score_meta.get("reason") == "command_embed_failed":
        diag["reason"] = "command_embed_failed"
        return None, diag
    if score_meta.get("reason") == "empty_command":
        diag["reason"] = "empty_command"
        return None, diag

    # Prefer spatially peaked slots; break ontology×activation ties on noise.
    scored.sort(
        key=lambda r: (
            float(r.get("match_score") or 0.0),
            1.0 if r.get("uv_valid") else 0.0,
            float(r.get("mask_peakiness") or 0.0),
        ),
        reverse=True,
    )
    diag["candidates"] = [
        {
            "slot_id": c["slot_id"],
            "score": round(float(c.get("match_score") or 0.0), 4),
            "label": c.get("label"),
            "u": c.get("u"),
            "v": c.get("v"),
            "match_label": round(float(c.get("match_label") or 0.0), 4),
            "match_concept": round(float(c.get("match_concept") or 0.0), 4),
            "match_ontology": round(float(c.get("match_ontology") or 0.0), 4),
            "uv_valid": c.get("uv_valid"),
            "mask_peakiness": round(float(c.get("mask_peakiness") or 0.0), 4),
            "activation": round(float(c.get("activation") or 0.0), 4),
        }
        for c in scored[:5]
    ]
    # Full slot table for threshold-vs-model diagnosis (all K slots).
    diag["slot_peakiness"] = [
        {
            "slot_id": c.get("slot_id"),
            "mask_peakiness": round(float(c.get("mask_peakiness") or 0.0), 4),
            "uv_valid": bool(c.get("uv_valid")),
            "u": round(float(c.get("u") or 0.5), 4),
            "v": round(float(c.get("v") or 0.5), 4),
            "activation": round(float(c.get("activation") or 0.0), 4),
            "label": c.get("label"),
            "match_score": round(float(c.get("match_score") or 0.0), 4),
            "match_label": round(float(c.get("match_label") or 0.0), 4),
            "match_concept": round(float(c.get("match_concept") or 0.0), 4),
            "match_ontology": round(float(c.get("match_ontology") or 0.0), 4),
        }
        for c in scored
    ]
    diag["mask_peakiness_min"] = float(mask_peakiness_min())
    if score_meta.get("ontology"):
        diag["ontology"] = score_meta.get("ontology")

    if score_meta.get("reason") == "no_language_vision_link":
        diag["reason"] = "no_language_vision_link"
        return None, diag

    min_c = vision_min_confidence()
    # Try top candidates until one has valid foreground metric range
    best_target: VisualTarget | None = None
    for cand in scored[:5]:
        conf = float(cand.get("match_score") or 0.0)
        if conf < min_c:
            break
        target = _apply_metric_geometry(cand, depth_camera)
        if require_range and not target.is_ready(require_range=True):
            continue
        # Prefer valid range; first ready candidate wins (already score-sorted)
        best_target = target
        break

    if best_target is None and depth_camera is not None:
        # Diffuse SlotAttention: try gated depth objectness bind when language
        # ontology matched. This is camera-only (not sim_oracle). Refuse only
        # when peak is weak / floor / objectness bind disabled.
        ont_diag = score_meta.get("ontology") if isinstance(score_meta.get("ontology"), dict) else {}
        ont_best = float((ont_diag or {}).get("best_score") or 0.0)
        best_ont_slot = max(
            scored,
            key=lambda r: (
                float(r.get("match_ontology") or 0.0),
                float(r.get("activation") or 0.0),
            ),
        )
        ont_slot = float(best_ont_slot.get("match_ontology") or 0.0)
        if ont_best >= 0.25 or ont_slot >= 0.20:
            if objectness_bind_enabled():
                cand = dict(best_ont_slot)
                cand["uv_valid"] = False  # force objectness geometry path
                ont_key = str((ont_diag or {}).get("best_key") or "").strip()
                if ont_key and not str(cand.get("label") or "").strip():
                    cand["label"] = ont_key
                elif ont_key and str(cand.get("label") or "").lower() in (
                    "object",
                    "com_high",
                    "",
                ):
                    cand["label"] = ont_key
                # Boost match_score so spatial confidence isn't crushed to ~0
                # when ontology is the only language link (flat SA scores ~0.07).
                cand["match_score"] = float(
                    max(
                        float(cand.get("match_score") or 0.0),
                        max(ont_best, ont_slot) * 0.85,
                        0.40,
                    )
                )
                target = _apply_metric_geometry(cand, depth_camera)
                pstr = float(
                    (target.diagnostics or {}).get("objectness_peak_strength") or 0.0
                )
                floorish = (
                    float(target.v) > 0.72
                    and float(target.confidence or 0.0) < 0.55
                )
                try:
                    from engine.vision_depth import objectness_edge_u_margin

                    edge_m = float(objectness_edge_u_margin())
                except Exception:
                    edge_m = 0.08
                edgeish = float(target.u) < edge_m or float(target.u) > (1.0 - edge_m)
                if (
                    target.is_ready(require_range=True)
                    and pstr >= objectness_bind_min_peak()
                    and not floorish
                    and not edgeish
                ):
                    diags = dict(target.diagnostics or {})
                    diags["geometry"] = "objectness_peak"
                    diags["source"] = "vision_objectness_bind"
                    diags["ontology_score"] = round(max(ont_best, ont_slot), 4)
                    target.diagnostics = diags
                    best_target = target
                    diag["geometry_fallback"] = "objectness_peak"
                    diag["objectness_bind"] = True
                    diag["ontology_score"] = round(max(ont_best, ont_slot), 4)
                else:
                    diag["objectness_bind_attempt"] = {
                        "peak_strength": round(pstr, 4),
                        "ready": bool(target.is_ready(require_range=True)),
                        "floorish": bool(floorish),
                        "edgeish": bool(edgeish),
                        "u": round(float(target.u), 4),
                        "range_m": target.range_m,
                        "v": round(float(target.v), 4),
                    }
            if best_target is None:
                diag["reason"] = "uncertain_no_peaked_slot"
                diag["ontology_score"] = round(ont_best, 4)
                diag["ontology"] = ont_diag
                diag["best_score"] = float(best_ont_slot.get("match_score") or 0.0)
                diag["min_conf"] = min_c
                diag["refused_geometry_fallback"] = "objectness_peak"
                return None, diag

    if best_target is None:
        # Fall back: best score without range gate (for diagnostics)
        best = scored[0]
        conf = float(best.get("match_score") or 0.0)
        if conf < min_c:
            diag["reason"] = "low_vision_confidence"
            diag["best_score"] = conf
            diag["min_conf"] = min_c
            return None, diag
        target = _apply_metric_geometry(best, depth_camera)
        if require_range and not target.is_ready(require_range=True):
            diag["reason"] = "missing_or_invalid_range"
            diag["target_partial"] = target.to_dict()
            diag["range_m"] = target.range_m
            diag["range_conf"] = target.range_conf
            return None, diag
        best_target = target

    target = best_target
    pstr = float((target.diagnostics or {}).get("objectness_peak_strength") or 0.0)
    if (target.diagnostics or {}).get("geometry") == "objectness_peak" and pstr < 0.12:
        diag["reason"] = "weak_objectness_peak"
        diag["peak_strength"] = round(pstr, 4)
        diag["target_partial"] = target.to_dict()
        return None, diag
    # Refuse obvious floor-center locks after geometry (still no peaked mask).
    if (
        depth_camera is not None
        and not bool((target.diagnostics or {}).get("uv_valid"))
        and float(target.v) > 0.72
        and float(target.confidence or 0.0) < 0.55
    ):
        diag["reason"] = "floor_lock_rejected"
        diag["target_partial"] = target.to_dict()
        return None, diag

    diag["reason"] = "ok"
    diag["slot_id"] = target.slot_id
    diag["label"] = target.label
    diag["range_m"] = target.range_m
    diag["range_conf"] = target.range_conf
    diag["u"] = target.u
    diag["v"] = target.v
    diag["resolved"] = target.ref
    diag["guided_uv"] = {"u": float(target.u), "v": float(target.v)}
    diag["geometry"] = (target.diagnostics or {}).get("geometry")
    diag["peak_strength"] = round(pstr, 4)
    # Raw confidence after spatial caps (no bind-time floor — 4A).
    diag["confidence_pre_floor"] = round(float(target.confidence), 4)
    diag["confidence"] = round(float(target.confidence), 4)
    if (target.diagnostics or {}).get("guided_uv"):
        diag["guided_uv"] = dict((target.diagnostics or {}).get("guided_uv") or {})
    ont = score_meta.get("ontology") if isinstance(score_meta.get("ontology"), dict) else {}
    if ont and "ontology_score" not in diag:
        diag["ontology_score"] = round(float(ont.get("best_score") or 0.0), 4)

    if visual_env is not None and hasattr(visual_env, "set_slot_lexicon"):
        try:
            lex = dict(getattr(visual_env, "_slot_lexicon", None) or {})
            sid = str(target.slot_id)
            prev = dict(lex.get(sid) or {})
            lex[sid] = {
                **prev,
                "label": str(target.label or prev.get("label") or "visual_referent"),
                "confidence": float(target.confidence),
            }
            tick = int(getattr(visual_env, "_slot_lexicon_tick", -1) or -1)
            visual_env.set_slot_lexicon(lex, tick=max(0, tick), frame_b64=None)
        except Exception:
            pass

    return target, diag


def track_visual_target(
    prev: VisualTarget,
    *,
    visual_env: Any | None = None,
    slots: list[dict[str, Any]] | None = None,
    depth_camera: DepthCamera | None = None,
    embed_fn: EmbedFn | None = None,
) -> VisualTarget:
    """Re-acquire by slot_id, else re-ground label↔slots via embeddings."""
    raw = list(slots) if slots is not None else collect_vision_slots(visual_env)
    by_id = {str(s["slot_id"]): s for s in raw}
    hit = by_id.get(prev.slot_id)
    if hit is None and prev.label and embed_fn is not None:
        scored, _ = score_slots_for_command(
            raw, str(prev.label), embed_fn=embed_fn, concept_project_fn=None
        )
        scored.sort(key=lambda r: float(r.get("match_score") or 0.0), reverse=True)
        if scored and float(scored[0].get("match_score") or 0.0) >= vision_min_confidence():
            hit = scored[0]
    if hit is None:
        updated = prev
        mask = None
    else:
        u, v = float(hit["u"]), float(hit["v"])
        updated = VisualTarget(
            slot_id=str(hit["slot_id"]),
            u=u,
            v=v,
            label=str(hit.get("label") or prev.label),
            confidence=max(
                float(prev.confidence),
                float(hit.get("match_score") or hit.get("activation") or 0.0),
            ),
            bearing=bearing_from_u(u),
            range_m=prev.range_m,
            range_var=prev.range_var,
            range_conf=prev.range_conf,
            bbox=prev.bbox,
            diagnostics=dict(prev.diagnostics),
        )
        mask = hit.get("attn_mask")
        if not bool(hit.get("uv_valid")) and hit.get("uv_valid") is not None:
            mask = None
            diags = dict(updated.diagnostics or {})
            diags["geometry"] = "objectness_peak"
            updated = VisualTarget(
                slot_id=updated.slot_id,
                u=updated.u,
                v=updated.v,
                label=updated.label,
                confidence=updated.confidence,
                bearing=updated.bearing,
                range_m=updated.range_m,
                range_var=updated.range_var,
                range_conf=updated.range_conf,
                bbox=updated.bbox,
                diagnostics=diags,
            )
    if depth_camera is not None:
        updated = attach_range_to_target(updated, depth_camera, attn_mask=mask)
        updated = _cap_spatial_confidence(updated)
    return updated
