"""Опциональные зависимости: PyBullet, PIL."""
from __future__ import annotations

try:
    import pybullet as pb
    import pybullet_data as pbd

    PYBULLET_AVAILABLE = True
except ImportError:
    PYBULLET_AVAILABLE = False
    pb = None  # type: ignore[assignment]
    pbd = None  # type: ignore[assignment]
    print("[HumanoidEnv] pybullet not installed, using fallback")

try:
    from PIL import Image as PILImage

    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    PILImage = None  # type: ignore[misc, assignment]
