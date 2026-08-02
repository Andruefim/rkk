"""Latent egocentric scene memory: many entities + attentional active set.

Architecture:
  - LatentSceneMemory holds a dictionary of SceneEntity tracks (slot-backed).
  - Each tick: odometry warps all tracks, vision fuses matching slots (EMA).
  - active_ids = objects the agent is currently attending / tasked with.
  - Navigation / reach read the primary active entity — not a one-off heuristic buffer.

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


def scene_ema_alpha() -> float:
    return float(max(0.05, min(0.95, _ef("RKK_SCENE_EMA_ALPHA", _ef("RKK_OWM_EMA_ALPHA", 0.35)))))


def scene_hold_ticks() -> int:
    return max(1, _ei("RKK_SCENE_HOLD_TICKS", _ei("RKK_OWM_HOLD_TICKS", 45)))


def scene_min_conf() -> float:
    return _ef("RKK_SCENE_MIN_CONF", _ef("RKK_OWM_MIN_VISION_CONF", 0.15))


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


def _apply_odometry_to_ego(
    x_fwd: float,
    y_right: float,
    *,
    prev_xy: tuple[float, float],
    prev_fwd: tuple[float, float],
    agent_xy: tuple[float, float],
    agent_forward: tuple[float, float],
) -> tuple[float, float]:
    px, py = prev_xy
    ax, ay = float(agent_xy[0]), float(agent_xy[1])
    fpx, fpy = prev_fwd
    fn = math.hypot(fpx, fpy) + 1e-9
    fpx, fpy = fpx / fn, fpy / fn
    dx, dy = ax - px, ay - py
    ds_fwd = dx * fpx + dy * fpy
    ds_right = dy * fpx - dx * fpy
    dtheta = _yaw_delta(prev_fwd, agent_forward)
    tx = float(x_fwd) - ds_fwd
    ty = float(y_right) - ds_right
    c, s = math.cos(-dtheta), math.sin(-dtheta)
    return c * tx - s * ty, s * tx + c * ty


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
    holding: bool = False
    diagnostics: dict[str, Any] = field(default_factory=dict)
    uv_track: list[list[float]] = field(default_factory=list)

    def is_fresh(self, tick: int) -> bool:
        if self.last_vision_tick < 0:
            return False
        return (int(tick) - int(self.last_vision_tick)) <= scene_hold_ticks()

    def is_usable(self, tick: int) -> bool:
        return (
            self.confidence >= scene_min_conf()
            and self.is_fresh(tick)
            and self.range_m > 0.05
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
            "holding": bool(self.holding),
        }

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
        self.holding = False
        self.diagnostics = {"source": "seed"}
        self.uv_track = [[float(self.u), float(self.v)]]

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
        self.holding = False
        self.diagnostics = {"source": "vision_ema", "alpha": alpha}
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
    _prev_xy: tuple[float, float] | None = field(default=None, repr=False)
    _prev_fwd: tuple[float, float] | None = field(default=None, repr=False)

    def reset(self) -> None:
        self.entities.clear()
        self.active_ids.clear()
        self.hard_lock_active = False
        self.last_odom_discontinuity = False
        self.last_odom_jump_m = 0.0
        self._prev_xy = None
        self._prev_fwd = None

    def release_hard_lock(self) -> None:
        self.hard_lock_active = False

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

        Hard-lock: control bearing is odometry-owned with a *bounded* live nudge
        when the live hit stays near the lock (EMA + max step). Outliers are
        rejected — no full rebind yank. Unlocked: live UV may rewrite bearing.
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
        # Accept live hits only inside this slack for nudge / range soft-correct.
        lock_bearing_slack = _ef("RKK_HARD_LOCK_BEARING_SLACK", 0.18)
        nudge_ema = _ef("RKK_HARD_LOCK_BEARING_NUDGE_EMA", 0.18)
        nudge_max = _ef("RKK_HARD_LOCK_BEARING_NUDGE_MAX", 0.035)

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
        near_locked = abs(live_delta) <= lock_bearing_slack

        if locked:
            nudge = 0.0
            if near_locked:
                act.u = (1.0 - a) * float(act.u) + a * float(u)
                act.v = (1.0 - a) * float(act.v) + a * float(v)
                # Bounded EMA correction — kills odom drift without full rebind.
                raw = float(nudge_ema) * live_delta
                nudge = float(max(-nudge_max, min(nudge_max, raw)))
                act.bearing = float(prev_b + nudge)
                locked_u = float(max(0.05, min(0.95, 0.5 + 0.5 * float(act.bearing))))
                act.u = (1.0 - 0.35) * float(act.u) + 0.35 * locked_u
            else:
                act.u = (1.0 - a) * float(act.u) + a * locked_u
                act.bearing = prev_b
            if near_locked and r is not None and float(r) > 0.05:
                prev_r = float(act.range_m)
                new_r = float(r)
                hint = (
                    float(range_hint)
                    if range_hint is not None and float(range_hint) > 0.1
                    else None
                )
                accept = (
                    new_r < prev_r * 0.82
                    or abs(new_r - prev_r) / max(prev_r, 0.3) < 0.35
                    or new_r < prev_r * 0.92
                )
                if hint is not None and prev_r > hint * 1.25 and new_r < prev_r * 0.90:
                    accept = True
                if accept:
                    act.range_m = (1.0 - a) * prev_r + a * new_r
            act.x_fwd, act.y_right = ego_from_bearing_range(act.bearing, act.range_m)
            act.diagnostics = {
                **dict(act.diagnostics or {}),
                "source": "hard_lock_live_nudge" if near_locked and nudge else "hard_lock_live_hud",
                "live_conf": float(conf or 0.0),
                "near_locked": bool(near_locked),
                "live_bearing": float(live_b),
                "bearing_live_delta": float(live_delta),
                "bearing_nudge": float(nudge),
            }
            if tick is not None:
                act.last_update_tick = int(tick)
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
        }
        if tick is not None:
            act.last_update_tick = int(tick)
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
        )
        ent.diagnostics = {"source": "bind", **dict(vt.diagnostics or {})}
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
    ) -> None:
        """
        Warp all entities by odometry, then fuse vision percepts.

        percepts items: slot_id, bearing, range_m, label?, activation?, confidence?

        Large COM/yaw jumps (reset_stance) skip odometry so hard-lock range is
        not inflated; locked active may reseed once from vision after a jump.
        """
        self.last_odom_discontinuity = False
        self.last_odom_jump_m = 0.0
        if self._prev_xy is not None and self._prev_fwd is not None:
            px, py = self._prev_xy
            ax, ay = float(agent_xy[0]), float(agent_xy[1])
            jump = float(math.hypot(ax - px, ay - py))
            self.last_odom_jump_m = jump
            # Position-only gate: yaw spins are normal; reset_stance teleports XY.
            if jump > scene_odom_max_step_m():
                self.last_odom_discontinuity = True
            else:
                for ent in self.entities.values():
                    if ent.confidence < 1e-6:
                        continue
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
                    ent.last_update_tick = int(tick)

        # Hard-lock: odometry-only unless a teleport discontinuity needs reseed
        locked_ids = set(self.active_ids) if self.hard_lock_active else set()
        reseed_locked = bool(self.last_odom_discontinuity and locked_ids)
        if locked_ids and not reseed_locked:
            for eid in locked_ids:
                ent = self.entities.get(eid)
                if ent is None:
                    continue
                ent.last_vision_tick = int(tick)
                ent.holding = True
                ent.diagnostics = {"source": "hard_lock_odom"}

        seen: set[str] = set() if reseed_locked else set(locked_ids)
        for p in percepts or []:
            sid = str(p.get("slot_id") or "")
            if not sid:
                continue
            if sid in locked_ids and not reseed_locked:
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
            eid = sid
            ent = self.entities.get(eid)
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
                )
            elif sid in locked_ids and reseed_locked:
                # After teleport: replace corrupted ego with a fresh depth sample
                ent.seed_from_bearing_range(
                    bearing=bearing,
                    range_m=float(r),
                    tick=int(tick),
                    label=label or ent.label,
                    confidence=max(conf, float(ent.confidence)),
                    activation=max(act, float(ent.activation)),
                    slot_id=sid,
                    u=u_f if u_f is not None else ent.u,
                    v=v_f if v_f is not None else ent.v,
                )
                ent.diagnostics = {
                    "source": "hard_lock_reseed",
                    "odom_jump_m": round(float(self.last_odom_jump_m), 3),
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
                ent.last_vision_tick = int(tick)
                ent.holding = True
                ent.diagnostics = {
                    "source": "hard_lock_odom_skip",
                    "odom_jump_m": round(float(self.last_odom_jump_m), 3),
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
            ent.confidence = float(ent.confidence * 0.995)
            ent.last_update_tick = int(tick)
            ent.diagnostics = {
                "source": "hold" if ent.is_fresh(tick) else "stale",
                "age_ticks": age,
            }
            if not ent.is_fresh(tick) and ent.confidence < scene_min_conf() * 0.5:
                if eid not in self.active_ids:
                    del self.entities[eid]

        # Drop active pointers to deleted entities
        self.active_ids = [a for a in self.active_ids if a in self.entities]

        self._prev_xy = (float(agent_xy[0]), float(agent_xy[1]))
        self._prev_fwd = (float(agent_forward[0]), float(agent_forward[1]))

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
        )

    def graph_payload(self) -> dict[str, float]:
        return self.scene.graph_payload()
