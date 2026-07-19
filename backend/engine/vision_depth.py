"""Depth / stereo range-at-UV API — sim PyBullet today, RGB-D/stereo on robot later."""
from __future__ import annotations

import math
import os
from dataclasses import dataclass
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
    # Mild preference for mid/upper FOV (objects) vs extreme bottom (feet/floor)
    ys = np.linspace(0.0, 1.0, h, dtype=np.float64)[:, None]
    v_w = np.clip(1.15 - ys, 0.25, 1.0)
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
    """Exclude bottom band (floor/feet) from salient-object search."""
    return float(max(0.35, min(0.75, _ef("RKK_OBJECTNESS_FLOOR_V", 0.58))))


def salient_objectness_peak(
    depth: DepthFrame | np.ndarray,
    *,
    near_m: float | None = None,
    far_m: float | None = None,
    floor_v_max: float | None = None,
) -> tuple[float | None, float | None, float | None, float | None, float | None, float]:
    """
    Argmax salient depth protrusion in upper FOV (not full-image centroid).

    Returns (u, v, range_m, variance, confidence, peak_strength).
    peak_strength in [0, 1] — use to gate HUD / bind confidence.
    """
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

    ys = np.linspace(0.0, 1.0, h, dtype=np.float64)[:, None]
    xs = np.linspace(0.0, 1.0, w, dtype=np.float64)[None, :]
    floor_v = float(floor_v_max if floor_v_max is not None else objectness_floor_v_max())
    score = np.where(ys < floor_v, score, 0.0)

    # Penalize lower-center floor dead-zone (common false lock when slots are diffuse).
    cx, cy, gw, gh = 0.5, 0.55, 0.10, 0.12
    center = np.exp(-(((xs - cx) / gw) ** 2 + ((ys - cy) / gh) ** 2))
    score = score * (1.0 - 0.92 * center)

    peak_val = float(score.max())
    if peak_val < 1e-8 or int(np.count_nonzero(score > 0)) < 3:
        return None, None, None, None, None, 0.0

    yi, xi = np.unravel_index(int(np.argmax(score)), score.shape)
    u = float(xs[0, xi])
    v = float(ys[yi, 0])

    upper = ys < floor_v
    med = float(np.median(score[upper & (score > 0)])) if np.any(upper & (score > 0)) else peak_val
    peak_strength = float(min(1.0, peak_val / max(med * 2.5, 1e-6)))

    r, var, conf = depth_at_uv(frame, u, v, window=3)
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

    ys = np.linspace(0.0, 1.0, h, dtype=np.float64)[:, None]
    xs = np.linspace(0.0, 1.0, w, dtype=np.float64)[None, :]
    floor_v = float(floor_v_max if floor_v_max is not None else objectness_floor_v_max())
    score = np.where(ys < floor_v, score, 0.0)

    u0 = max(0.05, min(0.95, 0.5 + 0.5 * float(bearing)))
    u_lo = max(0.0, u0 - float(fov_u_half))
    u_hi = min(1.0, u0 + float(fov_u_half))
    score = np.where((xs >= u_lo) & (xs <= u_hi), score, 0.0)

    cx, cy, gw, gh = 0.5, 0.55, 0.10, 0.12
    center = np.exp(-(((xs - cx) / gw) ** 2 + ((ys - cy) / gh) ** 2))
    score = score * (1.0 - 0.92 * center)

    peak_val = float(score.max())
    if peak_val < 1e-8 or int(np.count_nonzero(score > 0)) < 3:
        return salient_objectness_peak(
            frame, near_m=near, far_m=far, floor_v_max=floor_v_max
        )

    yi, xi = np.unravel_index(int(np.argmax(score)), score.shape)
    u = float(xs[0, xi])
    v = float(ys[yi, 0])
    upper = ys < floor_v
    med = float(np.median(score[upper & (score > 0)])) if np.any(upper & (score > 0)) else peak_val
    peak_strength = float(min(1.0, peak_val / max(med * 2.5, 1e-6)))

    r, var, conf = depth_at_uv(frame, u, v, window=3)
    if r is None:
        return None, None, None, None, None, peak_strength
    u_col = max(0.05, min(0.95, u0))
    u_lo_i = max(0, int((u_col - float(fov_u_half)) * (w - 1)))
    u_hi_i = min(w - 1, int((u_col + float(fov_u_half)) * (w - 1)))
    v_hi_i = min(h - 1, int(floor_v * (h - 1)))
    best_r: float | None = None
    best_u = best_v = 0.5
    for yi2 in range(0, v_hi_i + 1):
        for xi2 in range(u_lo_i, u_hi_i + 1):
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


def live_uv_range_at_bearing(
    camera: DepthCamera | ArrayDepthCamera,
    bearing: float,
    *,
    range_hint: float | None = None,
    fov_u_half: float = 0.22,
) -> tuple[float | None, float | None, float | None, float | None]:
    """
  Scan upper-FOV column near bearing for the closest valid depth (live camera lock).

  Returns (u, v, range_m, confidence). Used to keep HUD crosshair on the image
  as the robot turns — not frozen at bind-time UV.
    """
    frame = getattr(camera, "_frame", None)
    if frame is None:
        u0 = max(0.05, min(0.95, 0.5 + 0.5 * float(bearing)))
        r, _var, conf = camera.range_at_uv(u0, 0.42, window=4)
        return u0, 0.42 if r is not None else None, r, conf

    z = np.asarray(frame.depth_m, dtype=np.float64)
    h, w = int(z.shape[0]), int(z.shape[1])
    near = float(frame.near_m)
    far = float(frame.far_m)
    lo = near * 1.05
    hi = depth_hi_m(far)
    u0 = max(0.05, min(0.95, 0.5 + 0.5 * float(bearing)))
    u_lo = max(0, int((u0 - fov_u_half) * (w - 1)))
    u_hi = min(w - 1, int((u0 + fov_u_half) * (w - 1)))
    v_lo = max(0, int(0.10 * (h - 1)))
    v_hi = min(h - 1, int(0.58 * (h - 1)))

    best: tuple[float, float, float, float] | None = None
    best_score = -1.0
    min_hit: tuple[float, float, float, float] | None = None
    hint = float(range_hint) if range_hint is not None and float(range_hint) > 0.1 else None
    for yi in range(v_lo, v_hi + 1):
        v = float(yi) / max(h - 1, 1)
        for xi in range(u_lo, u_hi + 1):
            u = float(xi) / max(w - 1, 1)
            r, _var, conf = depth_at_uv(frame, u, v, window=1)
            if r is None or conf is None:
                continue
            if float(r) >= hi * 0.98 or float(r) <= lo:
                continue
            rel = 0.0
            if hint is not None:
                rel = abs(float(r) - hint) / max(hint, 0.3)
                # Reject floor/background lock when eval hint says target is closer.
                if float(r) > hint * 1.32 and rel > 0.22:
                    continue
                if float(r) >= hint * 0.78 and rel > 0.55:
                    continue
            score = float(conf) * (1.0 / max(float(r), 0.25)) * (1.0 / (1.0 + 2.0 * rel))
            if score > best_score:
                best_score = score
                best = (u, v, float(r), float(conf))
            if min_hit is None or float(r) < min_hit[2]:
                min_hit = (u, v, float(r), float(conf))
    if best is None and min_hit is not None:
        best = min_hit
    if best is None:
        return None, None, None, None
    if hint is not None and min_hit is not None and min_hit[2] < best[2] * 0.82:
        best = min_hit
    return best[0], best[1], best[2], best[3]


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
    ) -> tuple[float | None, float | None, float | None, float | None]:
        return live_uv_range_at_bearing(self, bearing, range_hint=range_hint)


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
