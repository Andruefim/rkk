"""
humanoid_variant — same variable_ids and role map as humanoid; perturbed physics (Track B1).
"""
from __future__ import annotations

import os

import numpy as np
import torch

from engine.features.humanoid.environment import EnvironmentHumanoid


def variant_mass_scale() -> float:
    try:
        return float(np.clip(float(os.environ.get("RKK_VARIANT_MASS_SCALE", "1.30")), 0.5, 2.5))
    except ValueError:
        return 1.30


def variant_friction_scale() -> float:
    try:
        return float(np.clip(float(os.environ.get("RKK_VARIANT_FRICTION_SCALE", "0.70")), 0.1, 2.0))
    except ValueError:
        return 0.70


def variant_com_offset_z() -> float:
    try:
        return float(np.clip(float(os.environ.get("RKK_VARIANT_COM_OFFSET_Z", "0.02")), -0.08, 0.08))
    except ValueError:
        return 0.02


class EnvironmentHumanoidVariant(EnvironmentHumanoid):
    """Humanoid with same topology/variables; mass, friction, and COM offset differ."""

    PRESET = "humanoid_variant"

    def __init__(
        self,
        device: torch.device | None = None,
        steps_per_do: int = 10,
        fixed_root: bool = False,
    ):
        super().__init__(device=device, steps_per_do=steps_per_do, fixed_root=fixed_root)
        self.preset = self.PRESET
        fn = getattr(self._sim, "apply_variant_physics", None)
        if callable(fn):
            fn(
                mass_scale=variant_mass_scale(),
                friction_scale=variant_friction_scale(),
                com_offset_z=variant_com_offset_z(),
            )
        print(
            f"[HumanoidVariant] mass x{variant_mass_scale():.2f}, "
            f"friction x{variant_friction_scale():.2f}, "
            f"com_z+{variant_com_offset_z():.3f}"
        )
