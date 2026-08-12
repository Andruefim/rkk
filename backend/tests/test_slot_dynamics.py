"""Object-centric SlotDynamics (JEPA) + OWM hold blend."""
from __future__ import annotations

import math

import pytest
import torch

from engine.object_working_memory import LatentSceneMemory, ObjectWorkingMemory
from engine.slot_dynamics import (
    SlotDynamics,
    pack_action,
    slot_dyn_sigma_scale,
)
from engine.vision_target import VisualTarget


def _vt(**kwargs) -> VisualTarget:
    base = dict(
        slot_id="slot_0",
        u=0.5,
        v=0.5,
        label="object",
        confidence=0.9,
        bearing=0.0,
        range_m=2.0,
        latent=[0.1] * 8,
    )
    base.update(kwargs)
    return VisualTarget(**base)


def test_untrained_predict_is_near_identity() -> None:
    dyn = SlotDynamics(slot_dim=8, device="cpu")
    z = [0.2] * 8
    z_hat, ego_hat = dyn.predict(z, (2.0, 0.1), pack_action(0.0, 0.0))
    assert ego_hat[0] == pytest.approx(2.0, abs=0.05)
    assert ego_hat[1] == pytest.approx(0.1, abs=0.05)
    assert all(abs(a - b) < 0.05 for a, b in zip(z_hat, z))


def test_train_on_synthetic_forward_shrinks_x_fwd() -> None:
    dyn = SlotDynamics(slot_dim=8, device="cpu", lr=1e-2)
    z = [0.15] * 8
    for _ in range(48):
        dyn.push_pair(
            z_t=z,
            ego_t=(2.0, 0.0),
            action=pack_action(0.0, 0.2),
            z_next=z,
            ego_next=(1.8, 0.0),
            has_live=True,
        )
    last = None
    for _ in range(40):
        last = dyn.train_step(batch_size=16)
    assert last is not None
    assert last["loss"] >= 0.0
    _, ego_hat = dyn.predict(z, (2.0, 0.0), pack_action(0.0, 0.2))
    assert ego_hat[0] < 2.0


def test_rollout_keeps_z_norm_bounded() -> None:
    dyn = SlotDynamics(slot_dim=8, device="cpu")
    z = [0.3] * 8
    acts = [pack_action(0.05, 0.1) for _ in range(8)]
    traj = dyn.rollout(z, (2.0, 0.0), acts, horizon=8)
    assert len(traj) == 8
    cap = math.sqrt(8.0) * 4.0 + 1e-6
    for z_i, ego_i in traj:
        nrm = math.sqrt(sum(v * v for v in z_i))
        assert nrm <= cap
        assert math.isfinite(ego_i[0]) and math.isfinite(ego_i[1])


class _AgreeOdom:
    """Predictor that matches forward odometry: x_fwd decreases by ds."""

    def predict(self, z, ego, action):
        ds = float(action[1]) if action is not None and len(action) > 1 else 0.0
        return list(z), (float(ego[0]) - ds, float(ego[1]))


def test_owm_blend_does_not_zero_bearing_in_one_step(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RKK_SLOT_DYN_BLEND", "0.35")
    owm = ObjectWorkingMemory()
    owm.bind_from_visual(
        _vt(bearing=0.2, range_m=3.0),
        tick=1,
        agent_xy=(0.0, 0.0),
        agent_forward=(1.0, 0.0),
    )
    b0 = float(owm.bearing)
    owm.observe_vision(
        None,
        tick=2,
        agent_xy=(0.15, 0.0),
        agent_forward=(1.0, 0.0),
        dynamics=_AgreeOdom(),
        action=pack_action(0.0, 0.15),
    )
    assert owm.bearing == pytest.approx(b0, abs=0.15)
    assert owm.range_m > 0.5
    assert owm.is_usable(2)


def test_owm_sigma_grows_slower_when_predictor_agrees(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RKK_SLOT_DYN_BLEND", "0.35")
    monkeypatch.setenv("RKK_SLOT_DYN_SIGMA_SCALE", "0.4")
    monkeypatch.setenv("RKK_SLOT_DYN_AGREE_M", "0.3")

    def _run(dyn) -> float:
        scene = LatentSceneMemory()
        scene.bind_visual_target(
            _vt(),
            tick=1,
            agent_xy=(0.0, 0.0),
            agent_forward=(1.0, 0.0),
        )
        xy = (0.0, 0.0)
        for t in range(2, 8):
            xy = (0.05 * (t - 1), 0.0)
            scene.update(
                tick=t,
                percepts=[],
                agent_xy=xy,
                agent_forward=(1.0, 0.0),
                dynamics=dyn,
                action=pack_action(0.0, 0.05),
            )
        ent = scene.active()
        assert ent is not None
        return float(ent.bearing_sigma)

    sigma_plain = _run(None)
    sigma_agree = _run(_AgreeOdom())
    assert sigma_agree < sigma_plain
    assert slot_dyn_sigma_scale() == pytest.approx(0.4)


def test_slot_dynamics_checkpoint_roundtrip() -> None:
    from engine.checkpoint_modules import pack_learnable_modules, unpack_learnable_modules

    src_dyn = SlotDynamics(slot_dim=8, device="cpu", lr=1e-2)
    z = [0.2] * 8
    for _ in range(16):
        src_dyn.push_pair(
            z_t=z,
            ego_t=(2.0, 0.0),
            action=pack_action(0.0, 0.1),
            z_next=z,
            ego_next=(1.9, 0.0),
            has_live=True,
        )
    src_dyn.train_step(batch_size=8)
    src_dyn.train_step(batch_size=8)

    class _Sim:
        pass

    src = _Sim()
    src._slot_dynamics = src_dyn
    payload = pack_learnable_modules(src)
    assert "slot_dynamics" in (payload.get("sections") or {})

    dst_dyn = SlotDynamics(slot_dim=8, device="cpu")
    dst = _Sim()
    dst._slot_dynamics = dst_dyn
    out = unpack_learnable_modules(dst, payload)
    assert "slot_dynamics" in out["applied"]
    w_src = src_dyn.z_head.weight.detach().clone()
    w_dst = dst_dyn.z_head.weight.detach().clone()
    assert torch.allclose(w_src, w_dst)
    assert dst_dyn.train_steps == src_dyn.train_steps
