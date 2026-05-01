"""JSON-safe значения: float nan/inf недопустимы в стандартном JSON (RFC 8259)."""
from __future__ import annotations

import math
from typing import Any


def sanitize_for_json(obj: Any) -> Any:
    """
    Рекурсивно заменяет nan и ±inf на None; приводит нестандартные числа к float/int.
    Используйте перед json.dumps / FastAPI JSONResponse / websocket.send_json.
    """
    if obj is None:
        return None
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, int) and not isinstance(obj, bool):
        return obj
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, str):
        return obj
    if isinstance(obj, dict):
        return {str(k): sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_for_json(v) for v in obj]
    try:
        import numpy as np

        if isinstance(obj, np.ndarray):
            if obj.size == 0:
                return []
            if obj.ndim == 0 or obj.size == 1:
                return sanitize_for_json(obj.item())
            return [sanitize_for_json(x) for x in obj.tolist()]
        if isinstance(obj, np.generic):
            return sanitize_for_json(obj.item())
    except ImportError:
        pass
    try:
        import torch

        if isinstance(obj, torch.Tensor):
            t = obj.detach().cpu()
            if t.numel() == 0:
                return None
            if t.numel() == 1:
                return sanitize_for_json(float(t.item()))
            # Крупные тензоры в JSON не кладём — только сводка (иначе WS гигабайты).
            if t.numel() <= 64:
                return [sanitize_for_json(float(x)) for x in t.flatten().tolist()]
            return {
                "_tensor": "large",
                "shape": list(t.shape),
                "mean": sanitize_for_json(float(t.mean().item())),
            }
    except ImportError:
        pass
    return obj
