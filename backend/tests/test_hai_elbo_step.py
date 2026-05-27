"""HAI ELBO step tests."""
from __future__ import annotations

from engine.hierarchical_active_inference import hai_elbo_step


def test_hai_elbo_step_updates_posterior():
    obs = {
        "com_x": 0.6,
        "com_z": 0.7,
        "intent_stride": 0.4,
        "intent_gait_coupling": 0.5,
    }
    prior = {k: 0.5 for k in obs}
    q = hai_elbo_step(obs, prior_mean=prior, log_precision=0.0)
    assert "intent_stride" in q
    assert abs(q["com_x"] - 0.6) < abs(prior["com_x"] - 0.6)
