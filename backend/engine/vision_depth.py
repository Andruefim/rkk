"""Depth / stereo range-at-UV API — sim PyBullet today, RGB-D/stereo on robot later."""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Protocol, runtime_checkable

import numpy as np


def _ef(key: str, default: float) -> float:
    try:
        return float(os.environ.get(key, str(default)))
    except ValueError:
        return float(default)


def camera_near_m() -> float:
    return _ef("RKK_EGO_CAM_NEAR", 0.1)


def camera_far_m() -> float:
    return _ef("RKK_EGO_CAM_FAR", 15.0)


def depth_sample_window() -> int:
    try:
        return max(1, int(os.environ.get("RKK_DEPTH_UV_WINDOW", "3")))
    except ValueError:
        return 3


def live_uv_fov_base() -> float:
    return _ef("RKK_LIVE_UV_FOV_HALF", 0.22)


def live_uv_continuity_k() -> float:
    return _ef("RKK_LIVE_UV_CONTINUITY_K", 9.0)


def live_uv_track_radius() -> float:
    return _ef("RKK_LIVE_UV_TRACK_RADIUS", 0.14)


def live_uv_track_len() -> int:
    try:
        return max(2, int(os.environ.get("RKK_LIVE_UV_TRACK_LEN", "5")))
    except ValueError:
        return 5


@dataclass
class UvDepthTrack:
    """Compact UV history for depth-lock identity continuity."""

    history: list[tuple[float, float]] = field(default_factory=list)
    max_len: int = 5

    @classmethod
    def from_list(cls, pts: list | tuple | None) -> UvDepthTrack:
        out = cls(max_len=live_uv_track_len())
        if not pts:
            return out
        for item in pts:
            try:
                u, v = float(item[0]), float(item[1])
            except (TypeError, ValueError, IndexError):
                continue
            out.push(u, v)
        return out

    def to_list(self) -> list[list[float]]:
        return [[float(u), float(v)] for u, v in self.history]

    def prev(self) -> tuple[float, float] | None:
        return self.history[-1] if self.history else None

    def push(self, u: float, v: float) -> None:
        self.history.append((float(u), float(v)))
        if len(self.history) > int(self.max_len):
            self.history = self.history[-int(self.max_len) :]

    def extrapolate(self) -> tuple[float, float] | None:
        if not self.history:
            return None
        if len(self.history) == 1:
            return self.history[-1]
        (u0, v0), (u1, v1) = self.history[-2], self.history[-1]
        return (
            float(max(0.02, min(0.98, u1 + (u1 - u0)))),
            float(max(0.02, min(0.98, v1 + (v1 - v0)))),
        )


def track_motion_uv(uv_track: UvDepthTrack | None) -> float:
    """Max recent UV step magnitude — proxy for localization uncertainty."""
    if uv_track is None or len(uv_track.history) < 2:
        return 0.0
    motions: list[float] = []
    hist = uv_track.history
    for i in range(1, len(hist)):
        u0, v0 = hist[i - 1]
        u1, v1 = hist[i]
        motions.append(_uv_dist(u1, v1, u0, v0))
    return float(max(motions[-3:])) if motions else 0.0


def track_search_fov_u_half(
    uv_track: UvDepthTrack | None = None,
    *,
    base: float | None = None,
) -> float:
    """
    Horizontal search half-width from track uncertainty, not 1/range.

    No object-size / range heuristics: unknown task & geometry.
    - no track / single point → full base (discovery)
    - stable track → modest tighten toward track_radius scale
    - fast UV motion → widen up to base
    """
    b = float(base if base is not None else live_uv_fov_base())
    if uv_track is None or len(uv_track.history) < 2:
        return b
    sigma = track_motion_uv(uv_track)
    radius = live_uv_track_radius()
    # Cover predicted step + identity radius; never below ~half base when tracked.
    half = max(0.55 * b, min(b, 2.5 * sigma + 0.65 * radius))
    return float(max(0.08, half))


def adaptive_fov_u_half(
    range_hint: float | None = None,
    *,
    base: float | None = None,
    uv_track: UvDepthTrack | None = None,
) -> float:
    """Back-compat wrapper: range_hint ignored; FOV follows track uncertainty."""
    _ = range_hint
    return track_search_fov_u_half(uv_track, base=base)


def _uv_dist(u: float, v: float, u_ref: float, v_ref: float) -> float:
    du = float(u) - float(u_ref)
    dv = float(v) - float(v_ref)
    return float(math.hypot(du, 1.15 * dv))


def _continuity_factor(
    u: float,
    v: float,
    *,
    prev_uv: tuple[float, float] | None,
    track_pred: tuple[float, float] | None,
) -> float:
    k = live_uv_continuity_k()
    factors: list[float] = []
    if prev_uv is not None:
        factors.append(math.exp(-k * _uv_dist(u, v, prev_uv[0], prev_uv[1])))
    if track_pred is not None:
        factors.append(
            math.exp(-0.85 * k * _uv_dist(u, v, track_pred[0], track_pred[1]))
        )
    if not factors:
        return 1.0
    return float(max(factors))


def _score_live_uv_candidate(
    u: float,
    v: float,
    r: float,
    conf: float,
    *,
    rel: float,
    prev_uv: tuple[float, float] | None,
    track_pred: tuple[float, float] | None,
) -> float:
    base = float(conf) * (1.0 / max(float(r), 0.25)) * (1.0 / (1.0 + 2.0 * rel))
    return base * _continuity_factor(u, v, prev_uv=prev_uv, track_pred=track_pred)


def depth_far_reject_frac() -> float:
    """Reject metric depths above far * frac (sky / empty buffer ≈ far plane)."""
    return float(max(0.35, min(0.95, _ef("RKK_DEPTH_FAR_REJECT", 0.65))))


def depth_max_control_m() -> float:
    """Hard ceiling for control-path range (room-scale AGI / indoor robot)."""
    return float(max(1.0, _ef("RKK_DEPTH_MAX_M", 6.0)))


def depth_hi_m(far_m: float) -> float:
    return float(min(float(far_m) * depth_far_reject_frac(), depth_max_control_m()))


@runtime_checkable
class DepthCamera(Protocol):
    """Robot-transferable depth provider."""

    def range_at_uv(
        self,
        u: float,
        v: float,
        *,
        window: int | None = None,
    ) -> tuple[float | None, float | None, float | None]:
        """
        Return (range_m, range_var, range_conf).
        range_m None → invalid / hole.
        """
        ...


@dataclass
class DepthFrame:
    """Metric-capable depth map aligned to an RGB ego frame."""

    depth_m: np.ndarray  # (H, W) float32 meters
    near_m: float
    far_m: float
    valid_mask: np.ndarray | None = None  # (H, W) bool

    @property
    def height(self) -> int:
        return int(self.depth_m.shape[0])

    @property
    def width(self) -> int:
        return int(self.depth_m.shape[1])


def buffer_to_metric_depth(
    depth_buffer: np.ndarray,
    *,
    near_m: float | None = None,
    far_m: float | None = None,
) -> np.ndarray:
    """
    Convert PyBullet-style depth buffer [0,1] to metric depth (meters).

    Formula: Z = far * near / (far - (far - near) * d)
    Values near 1.0 (far plane) and 0.0 (near) are treated carefully.
    """
    near = float(near_m if near_m is not None else camera_near_m())
    far = float(far_m if far_m is not None else camera_far_m())
    d = np.asarray(depth_buffer, dtype=np.float64)
    # Some renderers already return metric depths > 1
    if float(np.nanmax(d)) > 1.5:
        return np.asarray(d, dtype=np.float32)
    denom = far - (far - near) * np.clip(d, 0.0, 1.0)
    denom = np.maximum(denom, 1e-6)
    z = (far * near) / denom
    return np.asarray(z, dtype=np.float32)


def depth_at_uv(
    depth: DepthFrame | np.ndarray,
    u: float,
    v: float,
    *,
    window: int | None = None,
    near_m: float | None = None,
    far_m: float | None = None,
) -> tuple[float | None, float | None, float | None]:
    """
    Median metric range in a window around normalized UV.
    Returns (range_m, variance, confidence in [0,1]).
    """
    if isinstance(depth, DepthFrame):
        z = depth.depth_m
        near = depth.near_m
        far = depth.far_m
        mask = depth.valid_mask
    else:
        z = np.asarray(depth, dtype=np.float32)
        near = float(near_m if near_m is not None else camera_near_m())
        far = float(far_m if far_m is not None else camera_far_m())
        mask = None

    if z.ndim != 2 or z.size == 0:
        return None, None, None

    h, w = int(z.shape[0]), int(z.shape[1])
    uu = max(0.0, min(1.0, float(u)))
    vv = max(0.0, min(1.0, float(v)))
    cx = int(round(uu * (w - 1)))
    cy = int(round(vv * (h - 1)))
    rad = int(window if window is not None else depth_sample_window())
    rad = max(0, rad)

    x0, x1 = max(0, cx - rad), min(w, cx + rad + 1)
    y0, y1 = max(0, cy - rad), min(h, cy + rad + 1)
    patch = z[y0:y1, x0:x1].astype(np.float64).reshape(-1)
    if mask is not None:
        mpatch = mask[y0:y1, x0:x1].reshape(-1)
        patch = patch[mpatch.astype(bool)]

    lo = near * 1.05
    hi = depth_hi_m(far)
    valid = patch[np.isfinite(patch) & (patch > lo) & (patch < hi)]
    if valid.size < 1:
        return None, None, None

    med = float(np.median(valid))
    # Far-plane / sky samples must not seed navigation
    if med >= hi * 0.98:
        return None, None, None
    var = float(np.var(valid)) if valid.size > 1 else 0.0
    # Confidence: fraction valid in window + inverse relative variance
    frac = float(valid.size) / float(max(1, (2 * rad + 1) ** 2))
    rel = var / max(med * med, 1e-6)
    conf = float(max(0.0, min(1.0, frac * (1.0 / (1.0 + 8.0 * rel)))))
    # Penalize depths that sit near the reject ceiling (ambiguous void)
    near_far = med / max(hi, 1e-6)
    if near_far > 0.85:
        conf *= float(max(0.0, 1.0 - (near_far - 0.85) / 0.15))
    if conf < 0.05:
        return None, None, None
    return med, var, conf


def _upsample_mask(mask: np.ndarray, h: int, w: int) -> np.ndarray:
    m = np.asarray(mask, dtype=np.float64)
    if m.ndim != 2:
        return np.ones((h, w), dtype=np.float64)
    mh, mw = int(m.shape[0]), int(m.shape[1])
    if mh == h and mw == w:
        return m
    # Nearest-neighbor upsample (no scipy dependency)
    ys = (np.arange(h, dtype=np.float64) + 0.5) * mh / h
    xs = (np.arange(w, dtype=np.float64) + 0.5) * mw / w
    yi = np.clip(ys.astype(np.int64), 0, mh - 1)
    xi = np.clip(xs.astype(np.int64), 0, mw - 1)
    return m[yi[:, None], xi[None, :]]


def _objectness_map(z: np.ndarray, foreground: np.ndarray, *, near: float) -> np.ndarray:
    """
    Geometric objectness: how much closer a pixel is than its local surround.

    Flat floor → ~0; tree/prop protruding in depth → high.
    Uses a cheap box blur via integral-style stride subsample (no scipy).
    """
    h, w = int(z.shape[0]), int(z.shape[1])
    obj = np.zeros((h, w), dtype=np.float64)
    if h < 5 or w < 5:
        return obj
    # Coarse local mean via strided blocks
    by, bx = max(3, h // 12), max(3, w // 12)
    z_safe = np.where(foreground, z, np.nan)
    for y0 in range(0, h, by):
        for x0 in range(0, w, bx):
            y1, x1 = min(h, y0 + by * 2), min(w, x0 + bx * 2)
            patch = z_safe[y0:y1, x0:x1]
            if not np.any(np.isfinite(patch)):
                continue
            local = float(np.nanmedian(patch))
            block = z[y0:y1, x0:x1]
            fg = foreground[y0:y1, x0:x1]
            # Closer than surround → positive objectness
            delta = np.where(fg, local - block, 0.0)
            obj[y0:y1, x0:x1] = np.maximum(obj[y0:y1, x0:x1], np.maximum(delta, 0.0))
    # Normalize
    m = float(obj.max())
    if m > 1e-6:
        obj = obj / m
    # Mild mid-FOV preference — do not crush short ground objects (v>0.7).
    ys = np.linspace(0.0, 1.0, h, dtype=np.float64)[:, None]
    v_w = np.clip(1.05 - 0.25 * ys, 0.70, 1.0)
    return obj * v_w


def attention_guided_range(
    depth: DepthFrame | np.ndarray,
    attn_mask: np.ndarray | None,
    *,
    near_m: float | None = None,
    far_m: float | None = None,
    prefer_objects: bool = True,
) -> tuple[float | None, float | None, float | None, float | None, float | None]:
    """
    Metric range + UV from attention × foreground depth.

    When prefer_objects=True, weight by geometric objectness (depth protrusion)
    so flat floor does not lock range to a constant ground strip.

    Returns (u, v, range_m, variance, confidence). Any None → invalid.
    """
    if isinstance(depth, DepthFrame):
        z = np.asarray(depth.depth_m, dtype=np.float64)
        near = float(depth.near_m)
        far = float(depth.far_m)
        vmask = depth.valid_mask
    else:
        z = np.asarray(depth, dtype=np.float64)
        near = float(near_m if near_m is not None else camera_near_m())
        far = float(far_m if far_m is not None else camera_far_m())
        vmask = None

    if z.ndim != 2 or z.size == 0:
        return None, None, None, None, None

    h, w = int(z.shape[0]), int(z.shape[1])
    lo = near * 1.05
    hi = depth_hi_m(far)
    foreground = np.isfinite(z) & (z > lo) & (z < hi)
    if vmask is not None:
        foreground &= np.asarray(vmask, dtype=bool)

    if attn_mask is None:
        attn = np.ones((h, w), dtype=np.float64)
    else:
        attn = _upsample_mask(attn_mask, h, w)
        attn = np.maximum(attn, 0.0)

    inv_z = 1.0 / np.maximum(z, near)
    if prefer_objects:
        obj = _objectness_map(z, foreground, near=near)
        # Blend: objectness dominates when present; else mild 1/Z
        surf = 0.15 + 0.85 * obj
        weights = attn * foreground.astype(np.float64) * inv_z * surf
    else:
        weights = attn * foreground.astype(np.float64) * inv_z

    mass = float(weights.sum())
    if mass < 1e-9 or int(foreground.sum()) < 3:
        return None, None, None, None, None

    ys = np.linspace(0.0, 1.0, h, dtype=np.float64)[:, None]
    xs = np.linspace(0.0, 1.0, w, dtype=np.float64)[None, :]
    u = float((weights * xs).sum() / mass)
    v = float((weights * ys).sum() / mass)

    # Weighted median depth via sorted samples
    flat_z = z[foreground]
    flat_w = weights[foreground]
    order = np.argsort(flat_z)
    flat_z = flat_z[order]
    flat_w = flat_w[order]
    cdf = np.cumsum(flat_w)
    mid = 0.5 * float(cdf[-1])
    idx = int(np.searchsorted(cdf, mid))
    idx = max(0, min(int(flat_z.size) - 1, idx))
    med = float(flat_z[idx])
    if med >= hi * 0.98 or med <= lo:
        return None, None, None, None, None

    mean_z = float((weights * z).sum() / mass)
    var = float(((weights * (z - mean_z) ** 2).sum()) / mass)
    fg_frac = float(foreground.sum()) / float(h * w)
    rel = var / max(med * med, 1e-6)
    conf = float(max(0.0, min(1.0, (0.35 + 0.65 * min(1.0, mass / (h * w * 0.05))) * (1.0 / (1.0 + 4.0 * rel)))))
    conf *= float(max(0.2, min(1.0, fg_frac * 8.0)))
    if conf < 0.05:
        return None, None, None, None, None
    return u, v, med, var, conf


def objectness_floor_v_max() -> float:
    """Soft band where planarity gate is required (legacy name kept for env)."""
    return float(max(0.35, min(0.75, _ef("RKK_OBJECTNESS_FLOOR_V", 0.58))))


def objectness_edge_u_margin() -> float:
    """Reject / suppress peaks within this margin of left/right frame edge."""
    # 0.15: FOV-border depth discontinuities sit around u≈0.08–0.12 and
    # lock approach into a turn that never centers a real object.
    return float(max(0.02, min(0.25, _ef("RKK_OBJECTNESS_EDGE_U", 0.15))))


def _apply_frame_edge_suppression(score: np.ndarray, xs: np.ndarray) -> np.ndarray:
    """Zero FOV-border columns; mild ramp so argmax does not sit on the kill edge."""
    edge_m = objectness_edge_u_margin()
    if edge_m <= 1e-6:
        return score
    out = np.where((xs < edge_m) | (xs > (1.0 - edge_m)), 0.0, score)
    ramp = 0.06
    left = np.clip((xs - edge_m) / ramp, 0.0, 1.0)
    right = np.clip(((1.0 - edge_m) - xs) / ramp, 0.0, 1.0)
    return out * (0.55 + 0.45 * left) * (0.55 + 0.45 * right)


def floor_protrusion_min() -> float:
    """Min plane residual weight in lower FOV to accept a candidate (0–1)."""
    return float(max(0.05, min(0.6, _ef("RKK_FLOOR_PROTRUSION_MIN", 0.18))))


def _floor_protrusion_weight(
    z: np.ndarray,
    foreground: np.ndarray,
    *,
    near: float,
    hi: float,
) -> np.ndarray:
    """
    Per-pixel weight in [0, 1]: closer than expected floor / surround → high.

    Fits z ≈ a + b·v from robust lower-FOV row medians, plus a wide
    horizontal local-surround residual (catches cylinders whose top sits
    near plane height). Flat floor → ~0; grounded props → high even at
    v > floor_v_max.
    """
    h, w = int(z.shape[0]), int(z.shape[1])
    weight = np.zeros((h, w), dtype=np.float64)
    if h < 8 or w < 8:
        return weight

    ys = np.linspace(0.0, 1.0, h, dtype=np.float64)[:, None]
    row_v: list[float] = []
    row_z: list[float] = []
    for yi in range(h):
        v = float(yi) / max(h - 1, 1)
        if v < 0.45:
            continue
        row = z[yi]
        fg = foreground[yi] & np.isfinite(row) & (row < hi * 0.90)
        if int(fg.sum()) < max(4, w // 10):
            continue
        med = float(np.median(row[fg]))
        if not np.isfinite(med) or med <= near * 1.05:
            continue
        row_v.append(v)
        row_z.append(med)
    if len(row_v) < 4:
        return np.where(foreground, 0.35, 0.0).astype(np.float64)

    vv = np.asarray(row_v, dtype=np.float64)
    zz = np.asarray(row_z, dtype=np.float64)
    A = np.column_stack([np.ones_like(vv), vv])
    try:
        coef, *_ = np.linalg.lstsq(A, zz, rcond=None)
        pred = A @ coef
        resid = np.abs(zz - pred)
        keep = resid <= max(0.25, float(np.median(resid)) * 2.5)
        if int(keep.sum()) >= 3:
            coef, *_ = np.linalg.lstsq(A[keep], zz[keep], rcond=None)
    except np.linalg.LinAlgError:
        return np.where(foreground, 0.35, 0.0).astype(np.float64)

    z_pred = coef[0] + coef[1] * ys

    # Per-row background: robust median of the row, then local residual.
    # (Wide block medians self-contaminate when the object fills the block.)
    surround = np.full_like(z, np.nan, dtype=np.float64)
    for yi in range(h):
        row = z[yi]
        fg = foreground[yi]
        if int(fg.sum()) < max(4, w // 10):
            continue
        bg = float(np.median(row[fg]))
        surround[yi, :] = bg
        # Also try left/right half medians and take the farther (background).
        mid = w // 2
        left = row[:mid][fg[:mid]]
        right = row[mid:][fg[mid:]]
        cands = [bg]
        if left.size >= 3:
            cands.append(float(np.median(left)))
        if right.size >= 3:
            cands.append(float(np.median(right)))
        bg2 = float(max(cands))
        surround[yi, :] = bg2

    residual_plane = np.where(foreground, z_pred - z, 0.0)
    residual_local = np.where(
        foreground & np.isfinite(surround),
        surround - z,
        0.0,
    )
    floor_v = objectness_floor_v_max()
    # Plane residual only in the lower band — never mid/upper FOV.
    lowerish = ys >= floor_v
    residual = np.where(
        lowerish,
        np.maximum(residual_plane, residual_local),
        np.maximum(residual_local, 0.0),
    )
    scale = np.maximum(
        0.18 * np.maximum(np.where(np.isfinite(surround), surround, z_pred), near),
        0.12,
    )
    weight = np.clip(residual / scale, 0.0, 1.0)
    weight = np.where(foreground, weight, 0.0)
    pmin = floor_protrusion_min()
    lower = ys >= floor_v
    weight = np.where(lower & (weight < pmin), 0.0, weight)
    return weight


def _apply_planarity_to_score(
    score: np.ndarray,
    z: np.ndarray,
    foreground: np.ndarray,
    *,
    near: float,
    hi: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Blend objectness score with protrusion weight; return (score, weight)."""
    wmap = _floor_protrusion_weight(z, foreground, near=near, hi=hi)
    inv_z = 1.0 / np.maximum(z, near)
    # In lower FOV protrusions dominate; upper FOV keeps objectness×1/Z.
    out = score * (0.15 + 0.85 * np.maximum(wmap, 0.05)) + inv_z * wmap * 0.55
    out = np.where(foreground, out, 0.0)
    return out, wmap


def salient_objectness_peak(
    depth: DepthFrame | np.ndarray,
    *,
    near_m: float | None = None,
    far_m: float | None = None,
    floor_v_max: float | None = None,
) -> tuple[float | None, float | None, float | None, float | None, float | None, float]:
    """
    Argmax salient depth protrusion (planarity-gated, full FOV).

    Returns (u, v, range_m, variance, confidence, peak_strength).
    peak_strength in [0, 1] — use to gate HUD / bind confidence.

    ``floor_v_max`` kept for API compat; binary cutoff removed — lower FOV
    accepted only when plane residual indicates a protrusion.
    """
    del floor_v_max  # planarity replaces hard cutoff; arg retained for callers
    if isinstance(depth, DepthFrame):
        z = np.asarray(depth.depth_m, dtype=np.float64)
        near = float(depth.near_m)
        far = float(depth.far_m)
        vmask = depth.valid_mask
        frame = depth
    else:
        z = np.asarray(depth, dtype=np.float64)
        near = float(near_m if near_m is not None else camera_near_m())
        far = float(far_m if far_m is not None else camera_far_m())
        vmask = None
        frame = DepthFrame(depth_m=np.asarray(z, dtype=np.float32), near_m=near, far_m=far)

    if z.ndim != 2 or z.size == 0:
        return None, None, None, None, None, 0.0

    h, w = int(z.shape[0]), int(z.shape[1])
    lo = near * 1.05
    hi = depth_hi_m(far)
    foreground = np.isfinite(z) & (z > lo) & (z < hi)
    if vmask is not None:
        foreground &= np.asarray(vmask, dtype=bool)

    obj = _objectness_map(z, foreground, near=near)
    inv_z = 1.0 / np.maximum(z, near)
    score = obj * inv_z * foreground.astype(np.float64)
    score, _wmap = _apply_planarity_to_score(score, z, foreground, near=near, hi=hi)

    ys = np.linspace(0.0, 1.0, h, dtype=np.float64)[:, None]
    xs = np.linspace(0.0, 1.0, w, dtype=np.float64)[None, :]

    # Soft-penalize diffuse lower-center floor strip (not a hard ban).
    cx, cy, gw, gh = 0.5, 0.72, 0.12, 0.10
    center = np.exp(-(((xs - cx) / gw) ** 2 + ((ys - cy) / gh) ** 2))
    score = score * (1.0 - 0.75 * center * (1.0 - _wmap))

    score = _apply_frame_edge_suppression(score, xs)

    peak_val = float(score.max())
    if peak_val < 1e-8 or int(np.count_nonzero(score > 0)) < 3:
        return None, None, None, None, None, 0.0

    yi, xi = np.unravel_index(int(np.argmax(score)), score.shape)
    u = float(xs[0, xi])
    v = float(ys[yi, 0])

    # Hard reject residual edge peaks (numerical / thin FOV).
    edge_m = objectness_edge_u_margin()
    if u < edge_m or u > (1.0 - edge_m):
        return None, None, None, None, None, 0.0

    # Reject empty-floor locks: lower-FOV peak without protrusion, or
    # globally flat scene (no protrusion + weak objectness).
    pmin = floor_protrusion_min()
    if float(_wmap.max()) < pmin * 0.85 and float(obj.max()) < 0.45:
        return None, None, None, None, None, 0.0
    if v >= objectness_floor_v_max() and float(_wmap[yi, xi]) < pmin:
        return None, None, None, None, None, 0.0

    pos = score > 0
    med = float(np.median(score[pos])) if np.any(pos) else peak_val
    peak_strength = float(min(1.0, peak_val / max(med * 2.5, 1e-6)))

    r, var, conf = depth_at_uv(frame, u, v, window=3)
    z_peak = float(z[yi, xi])
    # Edge of a protrusion: 3×3 window often mixes background and reports far
    # plane — prefer the peak pixel when it is a clear closer foreground sample.
    if (
        np.isfinite(z_peak)
        and lo < z_peak < hi * 0.98
        and (
            r is None
            or (
                float(r) > z_peak * 1.25 + 0.15
                and (var is None or float(var) > 0.25)
            )
        )
    ):
        r, var, conf = z_peak, 0.0, 0.55 * peak_strength
    if r is None:
        return None, None, None, None, None, peak_strength
    if conf is not None:
        conf = float(max(0.0, min(1.0, float(conf) * (0.25 + 0.75 * peak_strength))))
    return u, v, float(r), var, conf, peak_strength


def salient_objectness_peak_near_bearing(
    depth: DepthFrame | np.ndarray,
    bearing: float,
    *,
    fov_u_half: float = 0.28,
    near_m: float | None = None,
    far_m: float | None = None,
    floor_v_max: float | None = None,
) -> tuple[float | None, float | None, float | None, float | None, float | None, float]:
    """Salient peak restricted to a horizontal window around ego bearing."""
    del floor_v_max
    if isinstance(depth, DepthFrame):
        z = np.asarray(depth.depth_m, dtype=np.float64)
        near = float(depth.near_m)
        far = float(depth.far_m)
        vmask = depth.valid_mask
        frame = depth
    else:
        z = np.asarray(depth, dtype=np.float64)
        near = float(near_m if near_m is not None else camera_near_m())
        far = float(far_m if far_m is not None else camera_far_m())
        vmask = None
        frame = DepthFrame(depth_m=np.asarray(z, dtype=np.float32), near_m=near, far_m=far)

    if z.ndim != 2 or z.size == 0:
        return None, None, None, None, None, 0.0

    h, w = int(z.shape[0]), int(z.shape[1])
    lo = near * 1.05
    hi = depth_hi_m(far)
    foreground = np.isfinite(z) & (z > lo) & (z < hi)
    if vmask is not None:
        foreground &= np.asarray(vmask, dtype=bool)

    obj = _objectness_map(z, foreground, near=near)
    inv_z = 1.0 / np.maximum(z, near)
    score = obj * inv_z * foreground.astype(np.float64)
    score, wmap = _apply_planarity_to_score(score, z, foreground, near=near, hi=hi)

    ys = np.linspace(0.0, 1.0, h, dtype=np.float64)[:, None]
    xs = np.linspace(0.0, 1.0, w, dtype=np.float64)[None, :]

    u0 = max(0.05, min(0.95, 0.5 + 0.5 * float(bearing)))
    u_lo = max(0.0, u0 - float(fov_u_half))
    u_hi = min(1.0, u0 + float(fov_u_half))
    score = np.where((xs >= u_lo) & (xs <= u_hi), score, 0.0)
    score = _apply_frame_edge_suppression(score, xs)

    cx, cy, gw, gh = 0.5, 0.72, 0.12, 0.10
    center = np.exp(-(((xs - cx) / gw) ** 2 + ((ys - cy) / gh) ** 2))
    score = score * (1.0 - 0.75 * center * (1.0 - wmap))

    peak_val = float(score.max())
    if peak_val < 1e-8 or int(np.count_nonzero(score > 0)) < 3:
        return salient_objectness_peak(frame, near_m=near, far_m=far)

    yi, xi = np.unravel_index(int(np.argmax(score)), score.shape)
    u = float(xs[0, xi])
    v = float(ys[yi, 0])
    pos = score > 0
    med = float(np.median(score[pos])) if np.any(pos) else peak_val
    peak_strength = float(min(1.0, peak_val / max(med * 2.5, 1e-6)))

    r, var, conf = depth_at_uv(frame, u, v, window=3)
    z_peak = float(z[yi, xi])
    if (
        np.isfinite(z_peak)
        and lo < z_peak < hi * 0.98
        and (
            r is None
            or (
                float(r) > z_peak * 1.25 + 0.15
                and (var is None or float(var) > 0.25)
            )
        )
    ):
        r, var, conf = z_peak, 0.0, 0.55 * peak_strength
    if r is None:
        return None, None, None, None, None, peak_strength
    u_col = max(0.05, min(0.95, u0))
    u_lo_i = max(0, int((u_col - float(fov_u_half)) * (w - 1)))
    u_hi_i = min(w - 1, int((u_col + float(fov_u_half)) * (w - 1)))
    # Full vertical search; planarity already gates floor pixels.
    v_hi_i = h - 1
    best_r: float | None = None
    best_u = best_v = 0.5
    pmin = floor_protrusion_min()
    for yi2 in range(0, v_hi_i + 1):
        for xi2 in range(u_lo_i, u_hi_i + 1):
            if float(wmap[yi2, xi2]) < pmin and (yi2 / max(h - 1, 1)) >= objectness_floor_v_max():
                continue
            ru = float(xi2) / max(w - 1, 1)
            rv = float(yi2) / max(h - 1, 1)
            rr, _vr, _rc = depth_at_uv(frame, ru, rv, window=1)
            if rr is None or float(rr) <= lo or float(rr) >= hi * 0.98:
                continue
            if best_r is None or float(rr) < best_r:
                best_r = float(rr)
                best_u, best_v = ru, rv
    if best_r is not None and float(best_r) < float(r) * 0.82:
        u, v = best_u, best_v
        r = float(best_r)
        _r2, var, conf = depth_at_uv(frame, u, v, window=1)
        if _r2 is not None:
            r = float(_r2)
    if conf is not None:
        conf = float(max(0.0, min(1.0, float(conf) * (0.25 + 0.75 * peak_strength))))
    return u, v, float(r), var, conf, peak_strength


def _live_uv_cand_log_enabled(tick: int | None) -> bool:
    """TEMP: dump scored window candidates (default: any tick while enabled)."""
    raw = os.environ.get("RKK_LIVE_UV_CAND_LOG", "1").strip().lower()
    if raw in ("0", "false", "no", "off"):
        return False
    if tick is None:
        return False
    # Default: log all ticks. Narrow with RKK_LIVE_UV_CAND_LOG_TICK_LO/HI if needed.
    try:
        lo = int(os.environ.get("RKK_LIVE_UV_CAND_LOG_TICK_LO", "0"))
        hi = int(os.environ.get("RKK_LIVE_UV_CAND_LOG_TICK_HI", "999999999"))
    except ValueError:
        lo, hi = 0, 999999999
    return lo <= int(tick) <= hi


_LIVE_UV_CAND_PATH_LOGGED = False


def _log_live_uv_candidates(
    *,
    tick: int,
    bearing: float,
    range_hint: float | None,
    fov_u_half: float,
    u_lo: int,
    u_hi: int,
    v_lo: int,
    v_hi: int,
    candidates: list[dict[str, float]],
    best: tuple[float, float, float, float] | None,
    best_score: float,
    min_hit: tuple[float, float, float, float] | None,
    chosen: tuple[float, float, float, float] | None,
    chosen_via: str,
) -> None:
    """TEMP diagnostic — write one JSONL record per live_uv call."""
    global _LIVE_UV_CAND_PATH_LOGGED
    try:
        from engine.task_logger import task_log_dir, task_log_event

        path = task_log_dir() / "live_uv_candidates.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        if not _LIVE_UV_CAND_PATH_LOGGED:
            _LIVE_UV_CAND_PATH_LOGGED = True
            try:
                task_log_event(
                    "live_uv_cand_log_open",
                    tick=int(tick),
                    path=str(path.resolve()),
                )
            except Exception:
                pass
        rec = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "tick": int(tick),
            "event": "live_uv_candidates",
            "bearing": round(float(bearing), 4),
            "range_hint": round(float(range_hint), 4) if range_hint is not None else None,
            "fov_u_half": float(fov_u_half),
            "window_px": {
                "u_lo": int(u_lo),
                "u_hi": int(u_hi),
                "v_lo": int(v_lo),
                "v_hi": int(v_hi),
            },
            "n_candidates": len(candidates),
            "candidates": candidates,
            "best": (
                {
                    "u": round(best[0], 4),
                    "v": round(best[1], 4),
                    "r": round(best[2], 4),
                    "conf": round(best[3], 4),
                    "score": round(float(best_score), 6),
                }
                if best is not None
                else None
            ),
            "min_hit": (
                {
                    "u": round(min_hit[0], 4),
                    "v": round(min_hit[1], 4),
                    "r": round(min_hit[2], 4),
                    "conf": round(min_hit[3], 4),
                }
                if min_hit is not None
                else None
            ),
            "chosen": (
                {
                    "u": round(chosen[0], 4),
                    "v": round(chosen[1], 4),
                    "r": round(chosen[2], 4),
                    "conf": round(chosen[3], 4),
                }
                if chosen is not None
                else None
            ),
            "chosen_via": chosen_via,
        }
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:
        pass


def live_uv_range_at_bearing(
    camera: DepthCamera | ArrayDepthCamera,
    bearing: float,
    *,
    range_hint: float | None = None,
    fov_u_half: float | None = None,
    tick: int | None = None,
    uv_track: UvDepthTrack | None = None,
) -> tuple[float | None, float | None, float | None, float | None]:
    """
  Scan upper-FOV column near bearing for the closest valid depth (live camera lock).

  Returns (u, v, range_m, confidence). Used to keep HUD crosshair on the image
  as the robot turns — not frozen at bind-time UV.

  Spatial window follows track uncertainty (not 1/range). If a tight range_hint
  yields no hits inside the track gate, retry once with range filter relaxed
  while keeping identity continuity — empty tight scan must not force rebind.
    """
    frame = getattr(camera, "_frame", None)
    if frame is None:
        u0 = max(0.05, min(0.95, 0.5 + 0.5 * float(bearing)))
        r, _var, conf = camera.range_at_uv(u0, 0.42, window=4)
        if uv_track is not None and u0 is not None and r is not None:
            uv_track.push(float(u0), 0.42)
        return u0, 0.42 if r is not None else None, r, conf

    z = np.asarray(frame.depth_m, dtype=np.float64)
    h, w = int(z.shape[0]), int(z.shape[1])
    near = float(frame.near_m)
    far = float(frame.far_m)
    lo = near * 1.05
    hi = depth_hi_m(far)
    hint = float(range_hint) if range_hint is not None and float(range_hint) > 0.1 else None
    fov = (
        float(fov_u_half)
        if fov_u_half is not None
        else track_search_fov_u_half(uv_track)
    )
    track_pred = uv_track.extrapolate() if uv_track is not None else None
    prev_uv = uv_track.prev() if uv_track is not None else None
    u_bearing = max(0.05, min(0.95, 0.5 + 0.5 * float(bearing)))
    if track_pred is not None:
        u0 = max(0.05, min(0.95, 0.32 * u_bearing + 0.68 * float(track_pred[0])))
    else:
        u0 = u_bearing
    track_radius = live_uv_track_radius()
    log_cands = _live_uv_cand_log_enabled(tick)
    fg = np.isfinite(z) & (z > lo) & (z < hi)
    protrude = _floor_protrusion_weight(z, fg, near=near, hi=hi)
    pmin = floor_protrusion_min()
    floor_v = objectness_floor_v_max()

    def _scan(
        *,
        track_gate: bool,
        range_gate: bool,
    ) -> tuple[
        list[dict[str, float]],
        tuple[float, float, float, float] | None,
        float,
        tuple[float, float, float, float] | None,
    ]:
        u_lo = max(0, int((u0 - fov) * (w - 1)))
        u_hi = min(w - 1, int((u0 + fov) * (w - 1)))
        # Full vertical FOV; planarity rejects flat floor in the lower band.
        v_lo = max(0, int(0.08 * (h - 1)))
        v_hi = min(h - 1, int(0.96 * (h - 1)))
        if track_pred is not None:
            v_mid = float(track_pred[1])
            v_half = max(0.12, min(0.30, fov * 1.15))
            v_lo = max(0, int((v_mid - v_half) * (h - 1)))
            v_hi = min(h - 1, int((v_mid + v_half) * (h - 1)))
            v_lo = max(v_lo, int(0.08 * (h - 1)))
            v_hi = min(h - 1, max(v_hi, int(0.08 * (h - 1))))
            # Never re-impose the old 0.62 ceiling — short objects live below it.

        best: tuple[float, float, float, float] | None = None
        best_score = -1.0
        min_hit: tuple[float, float, float, float] | None = None
        cands: list[dict[str, float]] = []
        for yi in range(v_lo, v_hi + 1):
            v = float(yi) / max(h - 1, 1)
            pw = float(protrude[yi].max()) if w > 0 else 0.0
            # Fast reject empty lower rows with no protrusion anywhere in row.
            if v >= floor_v and pw < pmin * 0.5:
                # Still allow sparse checks if any column in window protrudes.
                pass
            for xi in range(u_lo, u_hi + 1):
                u = float(xi) / max(w - 1, 1)
                p_w = float(protrude[yi, xi])
                if v >= floor_v and p_w < pmin:
                    continue
                if track_gate and track_pred is not None:
                    if _uv_dist(u, v, track_pred[0], track_pred[1]) > track_radius:
                        continue
                r, _var, conf = depth_at_uv(frame, u, v, window=1)
                if r is None or conf is None:
                    continue
                if float(r) >= hi * 0.98 or float(r) <= lo:
                    continue
                rel = 0.0
                if hint is not None:
                    rel = abs(float(r) - hint) / max(hint, 0.3)
                    if range_gate:
                        if float(r) > hint * 1.32 and rel > 0.22:
                            continue
                        if float(r) >= hint * 0.78 and rel > 0.55:
                            continue
                score = _score_live_uv_candidate(
                    u,
                    v,
                    float(r),
                    float(conf),
                    rel=rel,
                    prev_uv=prev_uv,
                    track_pred=track_pred,
                )
                # Prefer protrusions over flat floor matches.
                score = float(score) * (0.25 + 0.75 * max(p_w, 0.15 if v < floor_v else 0.0))
                if log_cands:
                    cands.append(
                        {
                            "u": round(float(u), 4),
                            "v": round(float(v), 4),
                            "r": round(float(r), 4),
                            "conf": round(float(conf), 4),
                            "score": round(float(score), 6),
                            "protrusion": round(float(p_w), 4),
                            "track_gate": bool(track_gate),
                            "range_gate": bool(range_gate),
                        }
                    )
                if score > best_score:
                    best_score = score
                    best = (u, v, float(r), float(conf))
                if min_hit is None or float(r) < min_hit[2]:
                    min_hit = (u, v, float(r), float(conf))
        return cands, best, best_score, min_hit

    candidates, best, best_score, min_hit = _scan(track_gate=True, range_gate=True)
    chosen_via = "none"
    # Tight hint empty inside identity gate → relax range filter, keep track gate.
    if best is None and hint is not None and track_pred is not None:
        soft_cands, best, best_score, min_hit = _scan(
            track_gate=True, range_gate=False
        )
        if log_cands:
            candidates.extend(soft_cands)
        if best is not None:
            chosen_via = "best_score_track_relax_range"
    if best is None:
        loose_cands, best, best_score, min_hit = _scan(
            track_gate=False, range_gate=True
        )
        if log_cands:
            candidates.extend(loose_cands)
        if best is not None:
            chosen_via = "best_score_loose"
    if best is None and hint is not None:
        loose_soft, best, best_score, min_hit = _scan(
            track_gate=False, range_gate=False
        )
        if log_cands:
            candidates.extend(loose_soft)
        if best is not None:
            chosen_via = "best_score_loose_relax_range"
    if best is None and min_hit is not None:
        best = min_hit
        chosen_via = "min_hit_fallback"
    if best is None:
        if log_cands and tick is not None:
            _log_live_uv_candidates(
                tick=int(tick),
                bearing=float(bearing),
                range_hint=hint,
                fov_u_half=float(fov),
                u_lo=0,
                u_hi=w - 1,
                v_lo=0,
                v_hi=h - 1,
                candidates=candidates,
                best=None,
                best_score=best_score,
                min_hit=min_hit,
                chosen=None,
                chosen_via="empty",
            )
        return None, None, None, None
    if chosen_via == "none":
        chosen_via = "best_score_track"
    chosen = best
    if (
        hint is not None
        and min_hit is not None
        and min_hit[2] < best[2] * 0.82
        and (
            track_pred is None
            or _uv_dist(min_hit[0], min_hit[1], track_pred[0], track_pred[1])
            <= track_radius * 1.25
        )
    ):
        chosen = min_hit
        chosen_via = "min_hit_override"
    if uv_track is not None:
        uv_track.push(chosen[0], chosen[1])
    if log_cands and tick is not None:
        u_lo = max(0, int((u0 - fov) * (w - 1)))
        u_hi = min(w - 1, int((u0 + fov) * (w - 1)))
        v_lo = max(0, int(0.08 * (h - 1)))
        v_hi = min(h - 1, int(0.96 * (h - 1)))
        _log_live_uv_candidates(
            tick=int(tick),
            bearing=float(bearing),
            range_hint=hint,
            fov_u_half=float(fov),
            u_lo=u_lo,
            u_hi=u_hi,
            v_lo=v_lo,
            v_hi=v_hi,
            candidates=candidates,
            best=best,
            best_score=best_score,
            min_hit=min_hit,
            chosen=chosen,
            chosen_via=chosen_via,
        )
    return chosen[0], chosen[1], chosen[2], chosen[3]


class ArrayDepthCamera:
    """In-memory DepthCamera over a DepthFrame / ndarray (tests + sim cache)."""

    def __init__(
        self,
        depth: DepthFrame | np.ndarray,
        *,
        near_m: float | None = None,
        far_m: float | None = None,
    ) -> None:
        if isinstance(depth, DepthFrame):
            self._frame = depth
        else:
            self._frame = DepthFrame(
                depth_m=np.asarray(depth, dtype=np.float32),
                near_m=float(near_m if near_m is not None else camera_near_m()),
                far_m=float(far_m if far_m is not None else camera_far_m()),
            )

    def range_at_uv(
        self,
        u: float,
        v: float,
        *,
        window: int | None = None,
    ) -> tuple[float | None, float | None, float | None]:
        return depth_at_uv(self._frame, u, v, window=window)

    def range_from_attention(
        self,
        attn_mask: np.ndarray | None,
    ) -> tuple[float | None, float | None, float | None, float | None, float | None]:
        return attention_guided_range(self._frame, attn_mask)

    def range_from_objectness_peak(
        self,
    ) -> tuple[float | None, float | None, float | None, float | None, float | None, float]:
        return salient_objectness_peak(self._frame)

    def range_from_objectness_peak_near_bearing(
        self,
        bearing: float,
        *,
        fov_u_half: float = 0.28,
    ) -> tuple[float | None, float | None, float | None, float | None, float | None, float]:
        return salient_objectness_peak_near_bearing(
            self._frame, float(bearing), fov_u_half=fov_u_half
        )

    def live_at_bearing(
        self,
        bearing: float,
        *,
        range_hint: float | None = None,
        tick: int | None = None,
        uv_track: UvDepthTrack | None = None,
        fov_u_half: float | None = None,
    ) -> tuple[float | None, float | None, float | None, float | None]:
        return live_uv_range_at_bearing(
            self,
            bearing,
            range_hint=range_hint,
            tick=tick,
            uv_track=uv_track,
            fov_u_half=fov_u_half,
        )


class StereoDepthProvider:
    """
    Placeholder RGB-D / stereo provider for real robots.

    Expects a disparity→depth callable or a live depth frame supplier.
    """

    def __init__(self, frame_fn: Any) -> None:
        self._frame_fn = frame_fn

    def range_at_uv(
        self,
        u: float,
        v: float,
        *,
        window: int | None = None,
    ) -> tuple[float | None, float | None, float | None]:
        frame = self._frame_fn()
        if frame is None:
            return None, None, None
        if isinstance(frame, DepthFrame):
            return depth_at_uv(frame, u, v, window=window)
        if isinstance(frame, np.ndarray):
            return depth_at_uv(frame, u, v, window=window)
        return None, None, None

    def range_from_attention(
        self,
        attn_mask: np.ndarray | None,
    ) -> tuple[float | None, float | None, float | None, float | None, float | None]:
        frame = self._frame_fn()
        if frame is None:
            return None, None, None, None, None
        return attention_guided_range(frame, attn_mask)

    def range_from_objectness_peak(
        self,
    ) -> tuple[float | None, float | None, float | None, float | None, float | None, float]:
        frame = self._frame_fn()
        if frame is None:
            return None, None, None, None, None, 0.0
        return salient_objectness_peak(frame)


def attach_range_to_target(
    target: Any,
    camera: DepthCamera | None,
    *,
    window: int | None = None,
    attn_mask: np.ndarray | None = None,
) -> Any:
    """
    Attach metric range. Prefer attention×foreground depth when a mask is given;
    fall back to point sample at UV. Updates UV when attention guidance moves it.
    """
    if target is None or camera is None:
        return target
    if not callable(getattr(target, "with_range", None)):
        return target

    guided = getattr(camera, "range_from_attention", None)
    peak_fn = getattr(camera, "range_from_objectness_peak", None)
    geom = (getattr(target, "diagnostics", None) or {}).get("geometry")

    def _finish(tgt: Any, r: float | None, var: float | None, conf: float | None) -> Any:
        return tgt.with_range(r, range_var=var, range_conf=conf)

    if geom == "objectness_peak" and callable(peak_fn):
        diags = dict(getattr(target, "diagnostics", None) or {})
        bearing_hint = diags.get("bearing_hint")
        near_fn = getattr(camera, "range_from_objectness_peak_near_bearing", None)
        try:
            if bearing_hint is not None and callable(near_fn):
                gu, gv, r, var, conf, pstr = near_fn(float(bearing_hint))
            else:
                gu, gv, r, var, conf, pstr = peak_fn()
        except Exception:
            gu = gv = r = var = conf = None
            pstr = 0.0
        if r is not None and gu is not None and gv is not None:
            diags = dict(getattr(target, "diagnostics", None) or {})
            diags["objectness_peak_strength"] = float(pstr)
            diags["guided_uv"] = {"u": float(gu), "v": float(gv)}
            target.diagnostics = diags
            with_uv = getattr(target, "with_uv", None)
            if callable(with_uv):
                target = with_uv(float(gu), float(gv))
            return _finish(target, r, var, conf)

    if callable(guided):
        try:
            gu, gv, r, var, conf = guided(attn_mask)
        except Exception:
            gu = gv = r = var = conf = None
        if r is not None and gu is not None and gv is not None:
            with_uv = getattr(target, "with_uv", None)
            if callable(with_uv):
                target = with_uv(float(gu), float(gv))
            return _finish(target, r, var, conf)

    r, var, conf = camera.range_at_uv(float(target.u), float(target.v), window=window)
    return _finish(target, r, var, conf)
