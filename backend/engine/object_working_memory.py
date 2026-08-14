"""Latent egocentric scene memory: many entities + attentional active set.

Architecture:
  - LatentSceneMemory holds a dictionary of SceneEntity tracks (slot-backed).
  - Each tick: odometry warps all tracks, vision fuses matching slots (EMA).
  - active_ids = objects the agent is currently attending / tasked with.
  - Navigation / reach read the primary active entity — not a one-off heuristic buffer.

Object permanence is **dead reckoning** (OWM + odometry + growing bearing_sigma),
optionally blended with object-centric ``SlotDynamics`` (JEPA on slot embeddings +
ego residual) when ``RKK_SLOT_DYNAMICS=1``. GNN ``slot_*`` / ``phys_nav_*`` remain
scalar predictive-coding targets — not a Dreamer XYZ rollout inside the graph.

Robot-transferable: ego (x_fwd, y_right) + range/bearing, no privileged world XY.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Any, Iterable

from engine.vision_target import VisualTarget


def _ef(key: str, default: float) -> float:
    try:
        return float(os.environ.get(key, str(default)))
    except ValueError:
        return float(default)


def _ei(key: str, default: int) -> int:
    try:
        return int(os.environ.get(key, str(default)))
    except ValueError:
        return int(default)


def _bayesian_vision_kalman_gain(
    live_delta: float,
    conf: float | None,
    *,
    sigma_odom2: float | None = None,
    outlier_scale: float | None = None,
) -> tuple[float, float]:
    """
    Continuous vision↔odom Kalman gain (no binary lock-out).

    Returns ``(k_gain, sigma_vis2)``. Large bearing residuals inflate σ_vision² so
    gain → 0 smoothly (robust / Cauchy-like), avoiding the old yank↔freeze binary.
    """
    c_val = float(max(0.05, min(1.0, float(conf if conf is not None else 0.5))))
    d = float(live_delta)
    s = float(
        outlier_scale
        if outlier_scale is not None
        else _ef("RKK_HARD_LOCK_BEARING_SLACK", 0.18)
    )
    s = max(0.05, s)
    resid = (abs(d) / s) ** 2
    sigma_vis2 = (1.0 / c_val) * (1.0 + 2.0 * resid + 2.5 * (resid**2))
    so2 = float(
        sigma_odom2 if sigma_odom2 is not None else _ef("RKK_HARD_LOCK_ODOM_VAR", 0.35)
    )
    so2 = max(1e-4, so2)
    pi_vis = 1.0 / max(1e-6, sigma_vis2)
    pi_odom = 1.0 / so2
    k_gain = float(pi_vis / (pi_vis + pi_odom))
    return k_gain, float(sigma_vis2)


def _inject_vision_precision(
    k_gain: float, conf: float | None, live_delta: float
) -> None:
    """Soft-update global π_vision from track quality (precision_groups hook)."""
    try:
        from engine.precision_groups import get_precision_state, precision_groups_enabled

        if not precision_groups_enabled():
            return
        c_val = float(max(0.05, min(1.0, float(conf if conf is not None else 0.5))))
        quality = float(max(0.0, min(1.0, float(k_gain) * c_val)))
        quality *= float(1.0 / (1.0 + 4.0 * (float(live_delta) ** 2)))
        target_pi = 0.25 + 1.75 * quality
        st = get_precision_state()
        st.vision = float(
            max(0.05, min(4.0, 0.82 * float(st.vision) + 0.18 * target_pi))
        )
    except Exception:
        pass


def scene_ema_alpha() -> float:
    return float(max(0.05, min(0.95, _ef("RKK_SCENE_EMA_ALPHA", _ef("RKK_OWM_EMA_ALPHA", 0.35)))))


def scene_hold_ticks() -> int:
    """Characteristic hold horizon (eviction of *non-active* tracks).

    Not a hard cutoff for the locked target: ``is_usable`` follows
    ``bearing_sigma`` / confidence. At zero turn, idle process noise reaches
    ``scene_sigma_max`` on roughly this many ticks.
    """
    return max(1, _ei("RKK_SCENE_HOLD_TICKS", _ei("RKK_OWM_HOLD_TICKS", 45)))


def scene_min_conf() -> float:
    return _ef("RKK_SCENE_MIN_CONF", _ef("RKK_OWM_MIN_VISION_CONF", 0.15))


def scene_sigma_max() -> float:
    """Usable cutoff on bearing_sigma (bearing units in [-1, 1])."""
    return max(0.05, _ef("RKK_SCENE_SIGMA_MAX", 0.45))


def scene_sigma_yaw() -> float:
    return max(0.0, _ef("RKK_SCENE_SIGMA_YAW", 0.35))


def scene_sigma_step() -> float:
    return max(0.0, _ef("RKK_SCENE_SIGMA_STEP", 0.08))


def scene_sigma_idle() -> float:
    """Per-tick process noise with no live UV (standing occlusion)."""
    return max(0.0, _ef("RKK_SCENE_SIGMA_IDLE", 0.009))


def scene_hold_decay() -> float:
    """Per-tick confidence multiplier on odom-only / unseen tracks (was 0.995)."""
    return float(max(0.90, min(0.999, _ef("RKK_SCENE_HOLD_DECAY", 0.985))))


def scene_max_entities() -> int:
    return max(2, _ei("RKK_SCENE_MAX_ENTITIES", 12))


# Back-compat aliases for older env / tests
def owm_ema_alpha() -> float:
    return scene_ema_alpha()


def owm_hold_ticks() -> int:
    return scene_hold_ticks()


def owm_min_vision_conf() -> float:
    return scene_min_conf()


def bearing_to_angle_rad(bearing: float) -> float:
    b = float(max(-1.0, min(1.0, bearing)))
    return b * math.pi * 0.5


def angle_to_bearing(angle_rad: float) -> float:
    return float(max(-1.0, min(1.0, float(angle_rad) / (math.pi * 0.5))))


def ego_from_bearing_range(bearing: float, range_m: float) -> tuple[float, float]:
    r = float(max(0.05, range_m))
    ang = bearing_to_angle_rad(bearing)
    return r * math.cos(ang), r * math.sin(ang)


def bearing_range_from_ego(x_fwd: float, y_right: float) -> tuple[float, float]:
    r = float(math.hypot(x_fwd, y_right))
    if r < 1e-6:
        return 0.0, 0.05
    ang = math.atan2(float(y_right), float(x_fwd))
    return angle_to_bearing(ang), max(0.05, r)


def _yaw_delta(fwd_a: tuple[float, float], fwd_b: tuple[float, float]) -> float:
    ax, ay = float(fwd_a[0]), float(fwd_a[1])
    bx, by = float(fwd_b[0]), float(fwd_b[1])
    na = math.hypot(ax, ay) + 1e-9
    nb = math.hypot(bx, by) + 1e-9
    ax, ay = ax / na, ay / na
    bx, by = bx / nb, by / nb
    cross = ax * by - ay * bx
    dot = max(-1.0, min(1.0, ax * bx + ay * by))
    return math.atan2(cross, dot)


def scene_odom_max_step_m() -> float:
    """Max COM step (m) treated as continuous walk; larger = teleport/reset."""
    return float(max(0.15, min(3.0, _ef("RKK_SCENE_ODOM_MAX_STEP_M", 0.75))))


def latent_reid_min_cos() -> float:
    return float(max(0.15, min(0.95, _ef("RKK_LATENT_REID_MIN_COS", 0.55))))


def _as_latent_list(vec: Any) -> list[float]:
    if vec is None:
        return []
    try:
        if hasattr(vec, "detach"):
            arr = vec.detach().float().cpu().numpy().reshape(-1)
        else:
            import numpy as np

            arr = np.asarray(vec, dtype=np.float64).reshape(-1)
        out = [float(x) for x in arr.tolist() if math.isfinite(float(x))]
        return out
    except Exception:
        return []


def latent_cosine(a: list[float] | None, b: list[float] | None) -> float:
    """Cosine similarity in [-1, 1]; 0 if either latent is empty/mismatched."""
    aa = list(a or [])
    bb = list(b or [])
    if not aa or not bb or len(aa) != len(bb):
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(aa, bb):
        xf, yf = float(x), float(y)
        dot += xf * yf
        na += xf * xf
        nb += yf * yf
    denom = math.sqrt(na) * math.sqrt(nb)
    if denom < 1e-12:
        return 0.0
    return float(max(-1.0, min(1.0, dot / denom)))


def match_latent_slot(
    candidates: list[dict[str, Any]],
    query_latent: list[float] | None,
    *,
    min_cos: float | None = None,
) -> dict[str, Any] | None:
    """
    Pick the candidate whose ``vector``/``latent`` best matches ``query_latent``.

    Returns the winning candidate dict (mutated with ``latent_cos``) or None.
    """
    q = list(query_latent or [])
    if not q or not candidates:
        return None
    thr = float(latent_reid_min_cos() if min_cos is None else min_cos)
    best: dict[str, Any] | None = None
    best_cos = -2.0
    for cand in candidates:
        vec = cand.get("latent")
        if vec is None:
            vec = cand.get("vector")
        lat = _as_latent_list(vec)
        if not lat:
            continue
        cos = latent_cosine(q, lat)
        if cos > best_cos:
            best_cos = cos
            best = dict(cand)
            best["latent"] = lat
            best["latent_cos"] = float(cos)
    if best is None or float(best.get("latent_cos", -1.0)) < thr:
        return None
    return best


def _apply_odometry_to_ego(
    x_fwd: float,
    y_right: float,
    *,
    prev_xy: tuple[float, float],
    prev_fwd: tuple[float, float],
    agent_xy: tuple[float, float],
    agent_forward: tuple[float, float],
) -> tuple[float, float]:
    px, py = float(prev_xy[0]), float(prev_xy[1])
    ax, ay = float(agent_xy[0]), float(agent_xy[1])
    fpx, fpy = float(prev_fwd[0]), float(prev_fwd[1])
    fn = math.hypot(fpx, fpy) + 1e-9
    fpx, fpy = fpx / fn, fpy / fn

    # Target position in world coordinates (using prev agent pose)
    # Forward = (fpx, fpy), Right = (fpy, -fpx)
    twx = px + float(x_fwd) * fpx + float(y_right) * fpy
    twy = py + float(x_fwd) * fpy - float(y_right) * fpx

    # Current agent forward and right unit vectors
    fcx, fcy = float(agent_forward[0]), float(agent_forward[1])
    fcn = math.hypot(fcx, fcy) + 1e-9
    fcx, fcy = fcx / fcn, fcy / fcn

    # Target relative to current agent position, projected onto current (Forward, Right)
    dtx = twx - ax
    dty = twy - ay
    new_x = dtx * fcx + dty * fcy
    new_y = dtx * fcy - dty * fcx
    return float(new_x), float(new_y)


def _odometry_motion(
    prev_xy: tuple[float, float],
    prev_fwd: tuple[float, float],
    agent_xy: tuple[float, float],
    agent_forward: tuple[float, float],
) -> tuple[float, float]:
    """Return (dtheta_rad, ds_m) of the agent since the previous pose."""
    px, py = prev_xy
    ax, ay = float(agent_xy[0]), float(agent_xy[1])
    ds = math.hypot(ax - px, ay - py)
    dtheta = _yaw_delta(prev_fwd, agent_forward)
    return float(dtheta), float(ds)


def _grow_bearing_sigma(sigma: float, *, dtheta: float, ds: float, idle: bool = True) -> float:
    grown = (
        float(sigma)
        + scene_sigma_yaw() * abs(float(dtheta))
        + scene_sigma_step() * max(0.0, float(ds))
        + (scene_sigma_idle() if idle else 0.0)
    )
    return float(min(2.0, max(0.0, grown)))


def _shrink_bearing_sigma(sigma: float, *, k_gain: float, vis_sigma: float = 0.05) -> float:
    k = float(max(0.0, min(1.0, k_gain)))
    mixed = (1.0 - k) * float(sigma) + k * float(max(0.02, vis_sigma))
    return float(max(0.02, mixed))


@dataclass
class SceneEntity:
    """One tracked scene entity in egocentric coordinates."""

    entity_id: str
    slot_id: str = ""
    label: str = ""
    bearing: float = 0.0
    range_m: float = 1.0
    x_fwd: float = 1.0
    y_right: float = 0.0
    u: float = 0.5
    v: float = 0.55
    confidence: float = 0.0
    activation: float = 0.0
    last_vision_tick: int = -1
    last_update_tick: int = -1
    last_live_uv_tick: int = -1
    bearing_sigma: float = 0.04
    holding: bool = False
    diagnostics: dict[str, Any] = field(default_factory=dict)
    uv_track: list[list[float]] = field(default_factory=list)
    # SlotAttention embedding for re-ID across slot_id permutations.
    latent: list[float] = field(default_factory=list)

    def is_fresh(self, tick: int) -> bool:
        if self.last_vision_tick < 0:
            return False
        return (int(tick) - int(self.last_vision_tick)) <= scene_hold_ticks()

    def is_usable(self, tick: int) -> bool:
        """Soft hold: confidence + range + bearing_sigma. Age is in sigma, not a 45-tick cliff."""
        _ = tick  # age is encoded in bearing_sigma / confidence decay
        return (
            self.confidence >= scene_min_conf()
            and self.range_m > 0.05
            and float(self.bearing_sigma) < scene_sigma_max()
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "slot_id": self.slot_id,
            "label": self.label,
            "bearing": float(self.bearing),
            "range_m": float(self.range_m),
            "x_fwd": float(self.x_fwd),
            "y_right": float(self.y_right),
            "u": float(self.u),
            "v": float(self.v),
            "confidence": float(self.confidence),
            "activation": float(self.activation),
            "last_vision_tick": int(self.last_vision_tick),
            "last_live_uv_tick": int(self.last_live_uv_tick),
            "bearing_sigma": float(self.bearing_sigma),
            "holding": bool(self.holding),
            "latent_dim": len(self.latent),
        }

    def set_latent(self, vec: Any) -> None:
        lat = _as_latent_list(vec)
        if lat:
            self.latent = lat

    def seed_from_bearing_range(
        self,
        *,
        bearing: float,
        range_m: float,
        tick: int,
        label: str = "",
        confidence: float = 0.5,
        activation: float = 0.5,
        slot_id: str = "",
        u: float | None = None,
        v: float | None = None,
        latent: Any = None,
    ) -> None:
        b = float(bearing)
        r = float(max(0.05, range_m))
        xf, yr = ego_from_bearing_range(b, r)
        self.slot_id = str(slot_id or self.entity_id)
        self.label = str(label or self.label)
        self.bearing = b
        self.range_m = r
        self.x_fwd = xf
        self.y_right = yr
        self.u = float(u) if u is not None else float(0.5 + 0.5 * b)
        self.v = float(v) if v is not None else 0.55
        # 4A: no artificial confidence floor — pass through calibrated/raw score.
        self.confidence = float(max(0.0, min(1.0, confidence)))
        self.activation = float(max(0.0, min(1.0, activation)))
        self.last_vision_tick = int(tick)
        self.last_update_tick = int(tick)
        self.last_live_uv_tick = int(tick)
        self.bearing_sigma = 0.04
        self.holding = False
        self.diagnostics = {"source": "seed"}
        self.uv_track = [[float(self.u), float(self.v)]]
        if latent is not None:
            self.set_latent(latent)

    def fuse_observation(
        self,
        *,
        bearing: float,
        range_m: float,
        tick: int,
        label: str = "",
        confidence: float = 0.5,
        activation: float = 0.5,
        slot_id: str = "",
        u: float | None = None,
        v: float | None = None,
        gate: bool = False,
        latent: Any = None,
    ) -> bool:
        """
        EMA-fuse a vision observation. If gate=True, reject outliers that would
        yank the track onto a different surface (e.g. floor re-lock).
        Returns True if fused, False if held.
        """
        b_obs = float(bearing)
        r_obs = float(max(0.05, range_m))
        if gate and self.range_m > 0.05 and self.confidence >= scene_min_conf() * 0.5:
            # Relative range jump or large bearing flip → keep odometry hold
            rel = abs(r_obs - self.range_m) / max(self.range_m, 0.2)
            db = abs(b_obs - self.bearing)
            if rel > 0.45 or db > 0.55:
                self.holding = True
                self.last_update_tick = int(tick)
                self.diagnostics = {
                    "source": "gate_reject",
                    "rel_range": rel,
                    "dbearing": db,
                }
                return False

        alpha = scene_ema_alpha()
        xf_obs, yr_obs = ego_from_bearing_range(b_obs, r_obs)
        self.x_fwd = (1.0 - alpha) * self.x_fwd + alpha * xf_obs
        self.y_right = (1.0 - alpha) * self.y_right + alpha * yr_obs
        self.bearing, self.range_m = bearing_range_from_ego(self.x_fwd, self.y_right)
        if slot_id:
            self.slot_id = str(slot_id)
        if label:
            self.label = str(label)
        if u is not None:
            self.u = (1.0 - alpha) * self.u + alpha * float(u)
        else:
            self.u = float(0.5 + 0.5 * self.bearing)
        if v is not None:
            self.v = (1.0 - alpha) * self.v + alpha * float(v)
        self.activation = float(activation)
        self.confidence = min(
            1.0, (1.0 - alpha) * self.confidence + alpha * float(confidence)
        )
        self.last_vision_tick = int(tick)
        self.last_update_tick = int(tick)
        self.last_live_uv_tick = int(tick)
        self.bearing_sigma = _shrink_bearing_sigma(
            self.bearing_sigma, k_gain=alpha, vis_sigma=0.04
        )
        self.holding = False
        self.diagnostics = {"source": "vision_ema", "alpha": alpha}
        if latent is not None:
            self.set_latent(latent)
        return True


@dataclass
class LatentSceneMemory:
    """
    Latent representation of surrounding entities + attentional focus.

    entities: full tracked set (the 'world' slice available to the agent)
    active_ids: subset currently in goal / thought (command binding, etc.)
    hard_lock_active: when True, active entity is warped by odometry only
      (no vision re-fuse) until release — prevents floor re-lock during approach.
    last_odom_discontinuity: True when the last update saw a COM/yaw jump
      (reset_stance / teleport) and skipped continuous odometry.
    """

    entities: dict[str, SceneEntity] = field(default_factory=dict)
    active_ids: list[str] = field(default_factory=list)
    hard_lock_active: bool = False
    last_odom_discontinuity: bool = False
    last_odom_jump_m: float = 0.0
    last_odom_dtheta: float = 0.0
    last_odom_ds: float = 0.0
    _live_diverge_streak: int = field(default=0, repr=False)
    _prev_xy: tuple[float, float] | None = field(default=None, repr=False)
    _prev_fwd: tuple[float, float] | None = field(default=None, repr=False)

    def reset(self) -> None:
        self.entities.clear()
        self.active_ids.clear()
        self.hard_lock_active = False
        self.last_odom_discontinuity = False
        self.last_odom_jump_m = 0.0
        self.last_odom_dtheta = 0.0
        self.last_odom_ds = 0.0
        self._live_diverge_streak = 0
        self._prev_xy = None
        self._prev_fwd = None

    def release_hard_lock(self) -> None:
        self.hard_lock_active = False
        self._live_diverge_streak = 0

    def refresh_active_from_live_camera(
        self,
        camera: Any,
        *,
        tick: int | None = None,
        range_hint: float | None = None,
        blend: float = 0.65,
    ) -> bool:
        """
        Project active target onto the current ego RGB-D frame (HUD + soft track).

        Hard-lock: continuous Bayesian vision↔odom fusion (Kalman gain from
        confidence + residual size). Outliers get near-zero gain — no binary
        freeze and no full rebind yank. Unlocked: live UV may rewrite bearing.
        """
        act = self.active()
        if act is None or camera is None:
            return False
        try:
            from engine.vision_depth import UvDepthTrack, live_uv_range_at_bearing
            from engine.vision_target import bearing_from_u
        except ImportError:
            return False

        locked = bool(self.hard_lock_active)
        prev_b = float(act.bearing)
        locked_u = float(max(0.05, min(0.95, 0.5 + 0.5 * prev_b)))
        lock_bearing_slack = _ef("RKK_HARD_LOCK_BEARING_SLACK", 0.18)

        track = UvDepthTrack.from_list(act.uv_track)
        live_kwargs = {
            "range_hint": range_hint or act.range_m,
            "tick": tick,
            "uv_track": track,
        }
        live_fn = getattr(camera, "live_at_bearing", None)
        if callable(live_fn):
            try:
                u, v, r, conf = live_fn(act.bearing, **live_kwargs)
            except TypeError:
                u, v, r, conf = live_fn(
                    act.bearing, range_hint=range_hint or act.range_m
                )
                if u is not None and v is not None:
                    track.push(float(u), float(v))
        else:
            u, v, r, conf = live_uv_range_at_bearing(
                camera,
                act.bearing,
                **live_kwargs,
            )
        act.uv_track = track.to_list()
        if u is None or v is None:
            peak_fn = getattr(camera, "range_from_objectness_peak_near_bearing", None)
            if callable(peak_fn):
                try:
                    pu, pv, pr, _var, pconf, _pstr = peak_fn(float(act.bearing))
                except Exception:
                    pu = pv = pr = pconf = None
                if pu is not None and pr is not None:
                    if abs(bearing_from_u(float(pu)) - float(act.bearing)) < 0.4:
                        u, v, r, conf = float(pu), float(pv), float(pr), pconf
            # Global peak fallback only when unlocked — under hard_lock it yanks
            # navigation toward unrelated protrusions.
            if u is None or v is None:
                if locked:
                    act.u = locked_u
                    act.diagnostics = {
                        **dict(act.diagnostics or {}),
                        "source": "hard_lock_hud",
                        "live_conf": 0.0,
                        "live_bearing": None,
                        "bearing_live_delta": None,
                        "bearing_nudge": 0.0,
                        "bearing_sigma": round(float(act.bearing_sigma), 4),
                    }
                    if tick is not None:
                        act.last_update_tick = int(tick)
                    return True
                peak_fn = getattr(camera, "range_from_objectness_peak", None)
                if callable(peak_fn):
                    try:
                        pu, pv, pr, _var, pconf, _pstr = peak_fn()
                    except Exception:
                        pu = pv = pr = pconf = None
                    if pu is not None and pr is not None:
                        if abs(bearing_from_u(float(pu)) - float(act.bearing)) < 0.4:
                            u, v, r, conf = float(pu), float(pv), float(pr), pconf
        if u is None or v is None:
            return False

        a = float(max(0.15, min(0.95, blend)))
        live_b = float(bearing_from_u(float(u)))
        live_delta = float(live_b - prev_b)

        if locked:
            k_gain, sigma_vis2 = _bayesian_vision_kalman_gain(
                live_delta, conf, outlier_scale=lock_bearing_slack
            )
            # Soft floor-depth downweight: lower FOV hits inflate σ further.
            try:
                from engine.vision_depth import objectness_floor_v_max

                floor_v = float(objectness_floor_v_max())
                if float(v) >= floor_v:
                    sigma_vis2 *= 4.0
                    pi_vis = 1.0 / max(1e-6, sigma_vis2)
                    pi_odom = 1.0 / max(1e-4, _ef("RKK_HARD_LOCK_ODOM_VAR", 0.35))
                    k_gain = float(pi_vis / (pi_vis + pi_odom))
            except Exception:
                pass

            nudge = float(k_gain * live_delta)
            # Persistent wrong-edge lock: lock extreme (+/−) while live camera
            # says near-center → force trust live (Kalman alone goes to ~0).
            # Also: sustained |live−lock| disagreement over several ticks.
            force_live = False
            soft_unlock = False
            try:
                live_ok = float(conf or 0.0) >= _ef(
                    "RKK_HARD_LOCK_FORCE_MIN_CONF", 0.15
                )
                diverge_thr = _ef("RKK_HARD_LOCK_DIVERGE_B", 0.40)
                if live_ok and abs(float(live_delta)) >= diverge_thr:
                    self._live_diverge_streak = int(self._live_diverge_streak) + 1
                else:
                    self._live_diverge_streak = 0

                extreme_lock = abs(float(prev_b)) >= _ef(
                    "RKK_HARD_LOCK_EXTREME_B", 0.55
                )
                live_centered = abs(float(live_b)) <= _ef(
                    "RKK_HARD_LOCK_LIVE_CENTER", 0.35
                )
                big_delta = abs(float(live_delta)) >= _ef(
                    "RKK_HARD_LOCK_FORCE_DELTA", 0.45
                )
                sustained_n = _ei("RKK_HARD_LOCK_DIVERGE_TICKS", 3)
                soft_n = _ei("RKK_HARD_LOCK_SOFT_UNLOCK_TICKS", 6)
                sustained = int(self._live_diverge_streak) >= sustained_n
                if live_ok and (
                    (extreme_lock and live_centered and big_delta) or sustained
                ):
                    force_live = True
                    k_force = _ef("RKK_HARD_LOCK_FORCE_LIVE_GAIN", 0.45)
                    max_step = _ef("RKK_HARD_LOCK_FORCE_MAX_STEP", 0.28)
                    if int(self._live_diverge_streak) >= soft_n:
                        soft_unlock = True
                        k_force = max(
                            float(k_force),
                            _ef("RKK_HARD_LOCK_SOFT_UNLOCK_GAIN", 0.75),
                        )
                        max_step = max(
                            float(max_step),
                            _ef("RKK_HARD_LOCK_SOFT_UNLOCK_MAX_STEP", 0.55),
                        )
                    k_gain = max(float(k_gain), float(k_force))
                    nudge = float(k_gain * live_delta)
                    # Cap single-tick snap so we don't oscillate, but escape edge.
                    nudge = float(max(-max_step, min(max_step, nudge)))
            except Exception:
                force_live = False
                soft_unlock = False

            act.bearing = float(prev_b + nudge)
            if soft_unlock:
                # Aggressive snap toward live while keeping hard-lock identity;
                # cool diverge streak so we do not yank every subsequent tick.
                snap = _ef("RKK_HARD_LOCK_SOFT_UNLOCK_SNAP", 0.70)
                act.bearing = (1.0 - snap) * float(act.bearing) + snap * float(
                    live_b
                )
                act.bearing = float(max(-1.0, min(1.0, act.bearing)))
                self._live_diverge_streak = max(
                    0, int(self._live_diverge_streak) // 2
                )
            # HUD follows live only in proportion to trust — outliers stay glued
            # to the locked column instead of yanking the crosshair.
            uv_w = float(a * max(k_gain, 0.35 if force_live else 0.0))
            act.u = (1.0 - uv_w) * float(act.u) + uv_w * float(u)
            act.v = (1.0 - uv_w) * float(act.v) + uv_w * float(v)
            locked_u = float(max(0.05, min(0.95, 0.5 + 0.5 * float(act.bearing))))
            act.u = (1.0 - 0.35) * float(act.u) + 0.35 * locked_u

            if r is not None and float(r) > 0.05 and (k_gain > 0.05 or force_live):
                prev_r = float(act.range_m)
                new_r = float(r)
                hint = (
                    float(range_hint)
                    if range_hint is not None and float(range_hint) > 0.1
                    else None
                )
                accept = (
                    new_r < prev_r * 0.82
                    or abs(new_r - prev_r) / max(prev_r, 0.3) < 0.45
                    or new_r < prev_r * 0.95
                    or force_live
                )
                if hint is not None and prev_r > hint * 1.25 and new_r < prev_r * 0.90:
                    accept = True
                if accept:
                    r_gain = min(0.55, k_gain * a + 0.08)
                    if force_live:
                        r_gain = max(r_gain, 0.25)
                    act.range_m = (1.0 - r_gain) * prev_r + r_gain * new_r
            act.x_fwd, act.y_right = ego_from_bearing_range(act.bearing, act.range_m)
            _inject_vision_precision(k_gain, conf, live_delta)
            if tick is not None:
                vis_sigma = min(0.20, math.sqrt(max(1e-6, float(sigma_vis2))) * 0.08)
                act.bearing_sigma = _shrink_bearing_sigma(
                    act.bearing_sigma, k_gain=float(k_gain), vis_sigma=vis_sigma
                )
                act.last_update_tick = int(tick)
                act.last_live_uv_tick = int(tick)
                act.last_vision_tick = int(tick)
            act.diagnostics = {
                **dict(act.diagnostics or {}),
                "source": (
                    "hard_lock_soft_unlock"
                    if soft_unlock
                    else (
                        "hard_lock_force_live"
                        if force_live
                        else "hard_lock_bayesian_kalman"
                    )
                ),
                "live_conf": float(conf or 0.0),
                "kalman_gain": float(k_gain),
                "sigma_vis2": round(float(sigma_vis2), 4),
                "near_locked": bool(abs(live_delta) <= lock_bearing_slack),
                "live_bearing": float(live_b),
                "bearing_live_delta": float(live_delta),
                "bearing_nudge": float(nudge),
                "force_live": bool(force_live),
                "soft_unlock": bool(soft_unlock),
                "live_diverge_streak": int(self._live_diverge_streak),
                "bearing_sigma": round(float(act.bearing_sigma), 4),
            }
            return True

        act.u = (1.0 - a) * float(act.u) + a * float(u)
        act.v = (1.0 - a) * float(act.v) + a * float(v)
        new_b = bearing_from_u(act.u)
        max_delta = 0.16 if a >= 0.7 else 0.22
        if abs(new_b - prev_b) > max_delta:
            new_b = prev_b + max(-max_delta, min(max_delta, new_b - prev_b))
            act.u = max(0.05, min(0.95, 0.5 + 0.5 * new_b))
        act.bearing = float(new_b)

        if r is not None and float(r) > 0.05:
            prev_r = float(act.range_m)
            new_r = float(r)
            hint = float(range_hint) if range_hint is not None and float(range_hint) > 0.1 else None
            accept = (
                new_r < prev_r * 0.82
                or abs(new_r - prev_r) / max(prev_r, 0.3) < 0.35
                or new_r < prev_r * 0.92
            )
            if hint is not None and prev_r > hint * 1.25 and new_r < prev_r * 0.90:
                accept = True
            if accept:
                act.range_m = (1.0 - a) * prev_r + a * new_r

        # Control path must match camera bearing even when depth gate rejects range.
        act.x_fwd, act.y_right = ego_from_bearing_range(act.bearing, act.range_m)

        act.diagnostics = {
            **dict(act.diagnostics or {}),
            "source": "live_camera_uv",
            "live_conf": float(conf or 0.0),
            "live_bearing": float(live_b),
            "bearing_live_delta": float(live_delta),
            "bearing_sigma": round(float(act.bearing_sigma), 4),
        }
        if tick is not None:
            act.bearing_sigma = _shrink_bearing_sigma(
                act.bearing_sigma, k_gain=float(a), vis_sigma=0.04
            )
            act.last_update_tick = int(tick)
            act.last_live_uv_tick = int(tick)
            act.last_vision_tick = int(tick)
        return True

    def active(self) -> SceneEntity | None:
        for eid in self.active_ids:
            ent = self.entities.get(eid)
            if ent is not None:
                return ent
        return None

    def set_active(self, entity_ids: Iterable[str]) -> None:
        seen: list[str] = []
        for eid in entity_ids:
            s = str(eid)
            if s and s not in seen:
                seen.append(s)
        self.active_ids = seen

    def focus(self, entity_id: str, *, exclusive: bool = True) -> None:
        eid = str(entity_id)
        if exclusive:
            self.active_ids = [eid] if eid else []
        elif eid and eid not in self.active_ids:
            self.active_ids.append(eid)

    def bind_visual_target(
        self,
        vt: VisualTarget,
        *,
        tick: int,
        agent_xy: tuple[float, float] | None = None,
        agent_forward: tuple[float, float] | None = None,
    ) -> SceneEntity:
        """Seed/update entity from resolve and mark it as the sole active focus."""
        eid = str(vt.slot_id or vt.ref)
        ent = self.entities.get(eid)
        if ent is None:
            ent = SceneEntity(entity_id=eid)
            self.entities[eid] = ent
        r = float(vt.range_m) if vt.range_m is not None else 1.0
        conf = float(max(0.0, min(1.0, float(vt.confidence))))
        ent.seed_from_bearing_range(
            bearing=float(vt.bearing),
            range_m=r,
            tick=int(tick),
            label=str(vt.label or ""),
            confidence=conf,
            activation=conf,
            slot_id=str(vt.slot_id),
            u=float(vt.u),
            v=float(vt.v),
            latent=getattr(vt, "latent", None)
            or (vt.diagnostics or {}).get("latent")
            or (vt.diagnostics or {}).get("vector"),
        )
        ent.diagnostics = {"source": "bind", **dict(vt.diagnostics or {})}
        if ent.latent:
            ent.diagnostics["latent_dim"] = len(ent.latent)
        self.focus(eid, exclusive=True)
        self.hard_lock_active = True
        if agent_xy is not None:
            self._prev_xy = (float(agent_xy[0]), float(agent_xy[1]))
        if agent_forward is not None:
            self._prev_fwd = (float(agent_forward[0]), float(agent_forward[1]))
        return ent

    def update(
        self,
        *,
        tick: int,
        percepts: list[dict[str, Any]] | None,
        agent_xy: tuple[float, float],
        agent_forward: tuple[float, float],
        dynamics: Any | None = None,
        action: Any | None = None,
    ) -> None:
        """
        Warp all entities by odometry, then fuse vision percepts.

        percepts items: slot_id, bearing, range_m, label?, activation?, confidence?

        Large COM/yaw jumps (reset_stance) skip odometry so hard-lock range is
        not inflated; locked active may reseed once from vision after a jump.
        """
        self.last_odom_discontinuity = False
        self.last_odom_jump_m = 0.0
        self.last_odom_dtheta = 0.0
        self.last_odom_ds = 0.0
        dtheta = 0.0
        ds = 0.0
        if self._prev_xy is not None and self._prev_fwd is not None:
            px, py = self._prev_xy
            ax, ay = float(agent_xy[0]), float(agent_xy[1])
            jump = float(math.hypot(ax - px, ay - py))
            self.last_odom_jump_m = jump
            dtheta, ds = _odometry_motion(
                self._prev_xy, self._prev_fwd, agent_xy, agent_forward
            )
            self.last_odom_dtheta = float(dtheta)
            self.last_odom_ds = float(ds)
            # Position-only gate: yaw spins are normal; reset_stance teleports XY.
            # Skip odom warp on discontinuity so hard-lock range is not inflated;
            # locked active reseeds from vision below (adaptive recovery).
            if jump > scene_odom_max_step_m():
                self.last_odom_discontinuity = True
            else:
                ego_prev_by_id: dict[str, tuple[float, float]] = {}
                sigma_grown_by_id: dict[str, float] = {}
                for ent in self.entities.values():
                    if ent.confidence < 1e-6:
                        continue
                    ego_prev_by_id[ent.entity_id] = (float(ent.x_fwd), float(ent.y_right))
                    sigma_before = float(ent.bearing_sigma)
                    ent.x_fwd, ent.y_right = _apply_odometry_to_ego(
                        ent.x_fwd,
                        ent.y_right,
                        prev_xy=self._prev_xy,
                        prev_fwd=self._prev_fwd,
                        agent_xy=agent_xy,
                        agent_forward=agent_forward,
                    )
                    ent.bearing, ent.range_m = bearing_range_from_ego(
                        ent.x_fwd, ent.y_right
                    )
                    locked = self.hard_lock_active and ent.entity_id in self.active_ids
                    if not locked:
                        ent.u = float(max(0.0, min(1.0, 0.5 + 0.5 * ent.bearing)))
                    ent.bearing_sigma = _grow_bearing_sigma(
                        ent.bearing_sigma, dtheta=dtheta, ds=ds, idle=True
                    )
                    sigma_grown_by_id[ent.entity_id] = float(ent.bearing_sigma) - sigma_before
                    ent.last_update_tick = int(tick)
                if dynamics is not None and not self.last_odom_discontinuity:
                    self._apply_active_slot_dynamics(
                        dynamics,
                        action,
                        tick=int(tick),
                        ego_prev_by_id=ego_prev_by_id,
                        sigma_grown_by_id=sigma_grown_by_id,
                    )

        # Hard-lock: odometry-only unless a teleport discontinuity needs reseed.
        # Do NOT refresh last_vision_tick here — that made is_fresh never expire.
        locked_ids = set(self.active_ids) if self.hard_lock_active else set()
        reseed_locked = bool(self.last_odom_discontinuity and locked_ids)
        if locked_ids and not reseed_locked:
            for eid in locked_ids:
                ent = self.entities.get(eid)
                if ent is None:
                    continue
                ent.holding = True
                live_ref = (
                    int(ent.last_live_uv_tick)
                    if ent.last_live_uv_tick >= 0
                    else int(ent.last_vision_tick)
                )
                age = int(tick) - live_ref if live_ref >= 0 else 9999
                extra = min(
                    0.06,
                    0.20 * abs(float(dtheta))
                    + 0.015 * max(0.0, float(ds))
                    + 0.0008 * max(0, age),
                )
                ent.confidence = float(
                    ent.confidence * max(0.90, scene_hold_decay() - extra)
                )
                ent.diagnostics = {
                    "source": "hard_lock_odom",
                    "bearing_sigma": round(float(ent.bearing_sigma), 4),
                    "age_live": int(age),
                    "dtheta": round(float(dtheta), 4),
                    "ds": round(float(ds), 4),
                    **{
                        k: v
                        for k, v in dict(ent.diagnostics or {}).items()
                        if str(k).startswith("slot_dyn")
                    },
                }

        seen: set[str] = set() if reseed_locked else set(locked_ids)
        # Latent re-ID: remap permuted slot_ids onto the active entity when
        # SlotAttention reorders slots but the embedding still matches.
        remapped: list[dict[str, Any]] = list(percepts or [])
        act_ent = self.active()
        if act_ent is not None and act_ent.latent:
            hit = match_latent_slot(remapped, act_ent.latent)
            if hit is not None:
                orig_sid = str(hit.get("slot_id") or "")
                hit = dict(hit)
                hit["slot_id"] = str(act_ent.entity_id)
                hit["_latent_reid"] = True
                drop = {
                    str(act_ent.entity_id),
                    str(act_ent.slot_id or ""),
                    orig_sid,
                }
                remapped = [hit] + [
                    p
                    for p in remapped
                    if str(p.get("slot_id") or "") not in drop
                ]

        for p in remapped:
            sid = str(p.get("slot_id") or "")
            if not sid:
                continue
            # Soft-lock: allow latent-matched re-fuse onto active even under hard_lock
            # when cosine re-ID succeeded (keeps track without full unlock).
            latent_ok = bool(p.get("_latent_reid"))
            if sid in locked_ids and not reseed_locked and not latent_ok:
                continue  # never re-fuse active while hard-locked
            r = p.get("range_m")
            if r is None or float(r) <= 0.05:
                continue
            conf = float(p.get("confidence", p.get("activation", 0.5)) or 0.5)
            if conf < scene_min_conf() * 0.5:
                continue
            bearing = float(p.get("bearing", 0.0))
            act = float(p.get("activation", conf) or conf)
            label = str(p.get("label") or "")
            u = p.get("u")
            v = p.get("v")
            u_f = float(u) if u is not None else None
            v_f = float(v) if v is not None else None
            lat = p.get("latent", p.get("vector"))
            eid = sid
            ent = self.entities.get(eid)
            if ent is None and latent_ok and act_ent is not None:
                ent = act_ent
                eid = act_ent.entity_id
            if ent is None:
                if len(self.entities) >= scene_max_entities():
                    self._evict_weakest(keep=seen | set(self.active_ids))
                if len(self.entities) >= scene_max_entities():
                    continue
                ent = SceneEntity(entity_id=eid)
                self.entities[eid] = ent
                ent.seed_from_bearing_range(
                    bearing=bearing,
                    range_m=float(r),
                    tick=int(tick),
                    label=label,
                    confidence=conf,
                    activation=act,
                    slot_id=sid,
                    u=u_f,
                    v=v_f,
                    latent=lat,
                )
            elif (sid in locked_ids or latent_ok) and reseed_locked:
                # After teleport: replace corrupted ego with a fresh depth sample
                ent.seed_from_bearing_range(
                    bearing=bearing,
                    range_m=float(r),
                    tick=int(tick),
                    label=label or ent.label,
                    confidence=max(conf, float(ent.confidence)),
                    activation=max(act, float(ent.activation)),
                    slot_id=str(ent.slot_id or sid),
                    u=u_f if u_f is not None else ent.u,
                    v=v_f if v_f is not None else ent.v,
                    latent=lat,
                )
                ent.diagnostics = {
                    "source": "hard_lock_reseed",
                    "odom_jump_m": round(float(self.last_odom_jump_m), 3),
                    "latent_reid": bool(latent_ok),
                    "latent_cos": p.get("latent_cos"),
                }
            elif latent_ok and sid in locked_ids and not reseed_locked:
                # Soft identity refresh under lock — update latent/UV lightly, keep ego.
                if lat is not None:
                    ent.set_latent(lat)
                if u_f is not None:
                    ent.u = 0.7 * float(ent.u) + 0.3 * float(u_f)
                if v_f is not None:
                    ent.v = 0.7 * float(ent.v) + 0.3 * float(v_f)
                ent.last_vision_tick = int(tick)
                ent.last_live_uv_tick = int(tick)
                ent.bearing_sigma = _shrink_bearing_sigma(
                    ent.bearing_sigma, k_gain=0.25, vis_sigma=0.06
                )
                ent.diagnostics = {
                    "source": "hard_lock_latent_reid",
                    "latent_cos": p.get("latent_cos"),
                    "bearing_sigma": round(float(ent.bearing_sigma), 4),
                }
            else:
                ent.fuse_observation(
                    bearing=bearing,
                    range_m=float(r),
                    tick=int(tick),
                    label=label,
                    confidence=conf,
                    activation=act,
                    slot_id=sid,
                    u=u_f,
                    v=v_f,
                    gate=bool(eid in self.active_ids),
                    latent=lat,
                )
            seen.add(eid)

        if locked_ids and reseed_locked:
            for eid in locked_ids:
                if eid in seen:
                    continue
                ent = self.entities.get(eid)
                if ent is None:
                    continue
                # No matching percept: keep prior ego, mark discontinuity
                ent.holding = True
                ent.diagnostics = {
                    "source": "hard_lock_odom_skip",
                    "odom_jump_m": round(float(self.last_odom_jump_m), 3),
                    "bearing_sigma": round(float(ent.bearing_sigma), 4),
                }
                seen.add(eid)

        # Hold / decay unseen
        for eid, ent in list(self.entities.items()):
            if eid in seen:
                continue
            age = (
                int(tick) - int(ent.last_vision_tick)
                if ent.last_vision_tick >= 0
                else 9999
            )
            ent.holding = age > 0
            extra = min(
                0.06,
                0.20 * abs(float(self.last_odom_dtheta))
                + 0.015 * max(0.0, float(self.last_odom_ds))
                + 0.0008 * max(0, age),
            )
            ent.confidence = float(
                ent.confidence * max(0.90, scene_hold_decay() - extra)
            )
            ent.last_update_tick = int(tick)
            ent.diagnostics = {
                "source": "hold" if ent.is_fresh(tick) else "stale",
                "age_ticks": age,
                "bearing_sigma": round(float(ent.bearing_sigma), 4),
            }
            if not ent.is_fresh(tick) and ent.confidence < scene_min_conf() * 0.5:
                if eid not in self.active_ids:
                    del self.entities[eid]

        # Drop active pointers to deleted entities
        self.active_ids = [a for a in self.active_ids if a in self.entities]

        self._prev_xy = (float(agent_xy[0]), float(agent_xy[1]))
        self._prev_fwd = (float(agent_forward[0]), float(agent_forward[1]))

    def _apply_active_slot_dynamics(
        self,
        dynamics: Any,
        action: Any,
        *,
        tick: int,
        ego_prev_by_id: dict[str, tuple[float, float]],
        sigma_grown_by_id: dict[str, float],
    ) -> None:
        if dynamics is None or not self.hard_lock_active:
            return
        try:
            from engine.slot_dynamics import apply_slot_dynamics_hold, pack_action
        except Exception:
            return
        act = action if action is not None else pack_action()
        for eid in list(self.active_ids):
            ent = self.entities.get(eid)
            if ent is None or not getattr(ent, "latent", None):
                continue
            diag = apply_slot_dynamics_hold(
                ent,
                dynamics,
                act,
                ego_prev=ego_prev_by_id.get(eid),
                tick=int(tick),
                sigma_grown=float(sigma_grown_by_id.get(eid, 0.0)),
            )
            if diag:
                ent.diagnostics = {**dict(ent.diagnostics or {}), **diag}

    def _evict_weakest(self, *, keep: set[str]) -> None:
        candidates = [
            (eid, ent)
            for eid, ent in self.entities.items()
            if eid not in keep
        ]
        if not candidates:
            return
        candidates.sort(key=lambda kv: (kv[1].confidence, kv[1].activation))
        del self.entities[candidates[0][0]]

    def graph_payload(self, tick: int | None = None) -> dict[str, float]:
        """Active focus + scene summary for GNN / obs."""
        t = int(tick) if tick is not None else -1
        act = self.active()
        scale = 5.0
        n_fresh = sum(
            1
            for e in self.entities.values()
            if t < 0 or e.is_fresh(t)
        )
        out: dict[str, float] = {
            "scene_n_entities": float(len(self.entities)),
            "scene_n_fresh": float(n_fresh),
            "scene_n_active": float(len(self.active_ids)),
        }
        if act is None:
            out.update(
                {
                    "task_target_x": 0.0,
                    "task_target_y": 0.0,
                    "task_target_dist_m": 0.0,
                    "task_target_bearing": 0.0,
                    "task_target_conf": 0.0,
                    "task_target_holding": 0.0,
                    "self_goal_target_dist": 0.0,
                    "self_goal_active": 0.0,
                }
            )
            return out

        out.update(
            {
                "task_target_x": float(max(-1.0, min(1.0, act.x_fwd / scale))),
                "task_target_y": float(max(-1.0, min(1.0, act.y_right / scale))),
                "task_target_dist_m": float(act.range_m),
                "task_target_bearing": float(act.bearing),
                "task_target_conf": float(max(0.0, min(1.0, act.confidence))),
                "task_target_holding": 1.0 if act.holding else 0.0,
                "self_goal_target_dist": float(max(0.0, min(1.0, act.range_m / 3.0))),
                "self_goal_active": 1.0 if act.confidence > scene_min_conf() else 0.0,
            }
        )
        # Top-k scene slots for WM (by confidence)
        ranked = sorted(
            self.entities.values(),
            key=lambda e: e.confidence,
            reverse=True,
        )[:4]
        for i, ent in enumerate(ranked):
            out[f"scene_e{i}_x"] = float(max(-1.0, min(1.0, ent.x_fwd / scale)))
            out[f"scene_e{i}_y"] = float(max(-1.0, min(1.0, ent.y_right / scale)))
            out[f"scene_e{i}_dist"] = float(min(1.0, ent.range_m / 5.0))
            out[f"scene_e{i}_conf"] = float(max(0.0, min(1.0, ent.confidence)))
        return out

    def snapshot(self, tick: int | None = None) -> dict[str, Any]:
        return {
            "n_entities": len(self.entities),
            "active_ids": list(self.active_ids),
            "entities": [e.to_dict() for e in self.entities.values()],
            "active": self.active().to_dict() if self.active() is not None else None,
        }

    def overlay_payload(self, tick: int | None = None) -> dict[str, Any]:
        """Compact HUD payload for camera preview (normalized UV + metric range)."""
        try:
            from engine.vision_resolve import hud_safe_label
        except Exception:
            def hud_safe_label(label: str, *, fallback: str = "target") -> str:  # type: ignore
                return str(fallback)[:24]

        t = int(tick) if tick is not None else -1

        def _hud_conf(ent: SceneEntity) -> float:
            conf = float(ent.confidence)
            diags = ent.diagnostics or {}
            if diags.get("geometry") == "objectness_peak":
                pstr = float(diags.get("objectness_peak_strength") or 0.0)
                conf = min(conf, 0.15 + 0.85 * pstr)
            return conf

        act = self.active()
        entities: list[dict[str, Any]] = []
        ranked = sorted(
            self.entities.values(),
            key=lambda e: (e.entity_id in self.active_ids, e.confidence),
            reverse=True,
        )[:8]
        for ent in ranked:
            if t >= 0 and not ent.is_fresh(t) and ent.confidence < scene_min_conf():
                continue
            fallback = ent.slot_id or ent.entity_id or "entity"
            label = hud_safe_label(ent.label, fallback=fallback)
            entities.append(
                {
                    "id": ent.entity_id,
                    "slot_id": ent.slot_id,
                    "label": label,
                    "u": float(max(0.0, min(1.0, ent.u))),
                    "v": float(max(0.0, min(1.0, ent.v))),
                    "range_m": round(float(ent.range_m), 2),
                    "bearing": round(float(ent.bearing), 3),
                    "bearing_sigma": round(float(ent.bearing_sigma), 3),
                    "conf": round(_hud_conf(ent), 3),
                    "active": ent.entity_id in self.active_ids,
                    "holding": bool(ent.holding),
                }
            )
        active_payload = None
        if act is not None:
            fallback = act.slot_id or "target"
            active_payload = {
                "id": act.entity_id,
                "slot_id": act.slot_id,
                "label": hud_safe_label(act.label, fallback=fallback),
                "u": float(max(0.0, min(1.0, act.u))),
                "v": float(max(0.0, min(1.0, act.v))),
                "range_m": round(float(act.range_m), 2),
                "bearing": round(float(act.bearing), 3),
                "bearing_sigma": round(float(act.bearing_sigma), 3),
                "conf": round(_hud_conf(act), 3),
                "holding": bool(act.holding),
            }
        return {
            "n_entities": len(self.entities),
            "n_active": len(self.active_ids),
            "hard_lock": bool(self.hard_lock_active),
            "odom_discontinuity": bool(self.last_odom_discontinuity),
            "odom_jump_m": round(float(self.last_odom_jump_m), 3),
            "active": active_payload,
            "entities": entities,
        }


# ---------------------------------------------------------------------------
# Back-compat: ObjectWorkingMemory as a view over LatentSceneMemory.active
# ---------------------------------------------------------------------------


class ObjectWorkingMemory:
    """
    Compatibility facade: single-target API over LatentSceneMemory.active.

    Prefer LatentSceneMemory directly for multi-entity / attentional sets.
    """

    def __init__(self, scene: LatentSceneMemory | None = None) -> None:
        self.scene = scene if scene is not None else LatentSceneMemory()

    def _ent(self) -> SceneEntity | None:
        return self.scene.active()

    @property
    def slot_id(self) -> str:
        e = self._ent()
        return e.slot_id if e else ""

    @property
    def label(self) -> str:
        e = self._ent()
        return e.label if e else ""

    @property
    def bearing(self) -> float:
        e = self._ent()
        return float(e.bearing) if e else 0.0

    @property
    def range_m(self) -> float:
        e = self._ent()
        return float(e.range_m) if e else 0.0

    @property
    def x_fwd(self) -> float:
        e = self._ent()
        return float(e.x_fwd) if e else 0.0

    @property
    def y_right(self) -> float:
        e = self._ent()
        return float(e.y_right) if e else 0.0

    @property
    def confidence(self) -> float:
        e = self._ent()
        return float(e.confidence) if e else 0.0

    @property
    def last_vision_tick(self) -> int:
        e = self._ent()
        return int(e.last_vision_tick) if e else -1

    @property
    def bearing_sigma(self) -> float:
        e = self._ent()
        return float(e.bearing_sigma) if e else 0.0

    @property
    def holding(self) -> bool:
        e = self._ent()
        return bool(e.holding) if e else False

    def is_fresh(self, tick: int) -> bool:
        e = self._ent()
        return bool(e and e.is_fresh(tick))

    def is_usable(self, tick: int) -> bool:
        e = self._ent()
        return bool(e and e.is_usable(tick))

    def to_dict(self) -> dict[str, Any]:
        e = self._ent()
        return e.to_dict() if e else {}

    def reset(self) -> None:
        self.scene.reset()

    def bind_from_visual(
        self,
        vt: VisualTarget,
        *,
        tick: int,
        agent_xy: tuple[float, float] | None = None,
        agent_forward: tuple[float, float] | None = None,
    ) -> None:
        self.scene.bind_visual_target(
            vt, tick=tick, agent_xy=agent_xy, agent_forward=agent_forward
        )

    def observe_vision(
        self,
        vt: VisualTarget | None,
        *,
        tick: int,
        agent_xy: tuple[float, float],
        agent_forward: tuple[float, float],
        dynamics: Any | None = None,
        action: Any | None = None,
    ) -> None:
        percepts: list[dict[str, Any]] = []
        if vt is not None and vt.is_ready(require_range=True):
            percepts.append(
                {
                    "slot_id": vt.slot_id,
                    "bearing": vt.bearing,
                    "range_m": vt.range_m,
                    "label": vt.label,
                    "confidence": vt.confidence,
                    "activation": vt.confidence,
                }
            )
        self.scene.update(
            tick=tick,
            percepts=percepts,
            agent_xy=agent_xy,
            agent_forward=agent_forward,
            dynamics=dynamics,
            action=action,
        )

    def graph_payload(self) -> dict[str, float]:
        return self.scene.graph_payload()
