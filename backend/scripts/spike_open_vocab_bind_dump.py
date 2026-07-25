#!/usr/bin/env python3
"""1C spike: offline open-vocab scoring on bind-dump RGB (no hot-path wire-up).

Compares command text against spatial/color heatmap proxies on the dump frame.
Tries optional CLIP / torchvision if installed; otherwise falls back to a
deterministic spatial prior + ontology paraphrase cosine (existing embedder).

This is a measurement spike — outputs JSON scores; does not bind targets.

Usage:
  cd backend && python scripts/spike_open_vocab_bind_dump.py
  python scripts/spike_open_vocab_bind_dump.py --dump logs/bind_dumps/tick_18678 \\
      --text "цилиндр перед тобой"
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_rgb(dump_dir: Path) -> np.ndarray | None:
    path = dump_dir / "rgb.png"
    if not path.is_file():
        return None
    try:
        from PIL import Image

        img = Image.open(path).convert("RGB")
        return np.asarray(img, dtype=np.float32) / 255.0
    except Exception:
        try:
            import imageio.v2 as imageio

            arr = imageio.imread(path)
            if arr.ndim == 2:
                arr = np.stack([arr] * 3, axis=-1)
            return np.asarray(arr[..., :3], dtype=np.float32) / 255.0
        except Exception as exc:
            print(f"warn: cannot load rgb.png: {exc}", file=sys.stderr)
            return None


def _spatial_objectness(rgb: np.ndarray) -> np.ndarray:
    """Cheap proxy: local contrast + lower-FOV bias (standing object)."""
    gray = rgb.mean(axis=2)
    # Local absolute deviation from 3x3 mean
    from numpy.lib.stride_tricks import sliding_window_view

    pad = np.pad(gray, 1, mode="edge")
    windows = sliding_window_view(pad, (3, 3))
    local = windows.mean(axis=(-1, -2))
    contrast = np.abs(gray - local)
    h, w = contrast.shape
    yy = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
    # Prefer mid-lower FOV (objects on floor plane) without hard floor cut.
    v_weight = np.clip(1.0 - np.abs(yy - 0.62) * 2.2, 0.15, 1.0)
    heat = contrast * v_weight
    heat = heat / float(heat.max() + 1e-8)
    return heat.astype(np.float32)


def _peak_uv(heat: np.ndarray) -> tuple[float, float, float]:
    idx = int(np.argmax(heat))
    h, w = heat.shape
    r, c = divmod(idx, w)
    return float(c / max(w - 1, 1)), float(r / max(h - 1, 1)), float(heat[r, c])


def _try_clip_heatmap(rgb: np.ndarray, text: str) -> dict | None:
    """Optional CLIP: if open_clip / transformers available."""
    try:
        import torch
        import open_clip  # type: ignore
    except Exception:
        return None
    try:
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="openai"
        )
        tokenizer = open_clip.get_tokenizer("ViT-B-32")
        model.eval()
        from PIL import Image

        img = Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8))
        # Coarse grid of patches
        h, w, _ = rgb.shape
        scores = np.zeros((4, 4), dtype=np.float32)
        with torch.no_grad():
            text_t = tokenizer([text])
            text_f = model.encode_text(text_t)
            text_f = text_f / text_f.norm(dim=-1, keepdim=True)
            for i in range(4):
                for j in range(4):
                    y0, y1 = i * h // 4, (i + 1) * h // 4
                    x0, x1 = j * w // 4, (j + 1) * w // 4
                    patch = img.crop((x0, y0, x1, y1))
                    img_t = preprocess(patch).unsqueeze(0)
                    img_f = model.encode_image(img_t)
                    img_f = img_f / img_f.norm(dim=-1, keepdim=True)
                    scores[i, j] = float((img_f @ text_f.T).item())
        u, v, peak = _peak_uv(scores)
        return {
            "backend": "open_clip_ViT-B-32",
            "grid": "4x4",
            "peak_u": u,
            "peak_v": v,
            "peak_score": peak,
            "score_map": scores.tolist(),
        }
    except Exception as exc:
        return {"backend": "open_clip", "error": str(exc)}


def _ontology_text_score(text: str) -> dict:
    from engine.grounded_language import FallbackEmbeddingClient
    from engine.visual_referent_ontology import clear_visual_referent_cache, match_visual_referent

    clear_visual_referent_cache()
    emb = FallbackEmbeddingClient(embed_dim=64)
    entry, score, meta = match_visual_referent(text, emb.embed)
    return {
        "best_key": getattr(entry, "key", None) if entry is not None else None,
        "best_description": getattr(entry, "description", None) if entry is not None else None,
        "best_score": float(score),
        "meta": {
            k: (float(v) if isinstance(v, (float, int)) else v)
            for k, v in (meta or {}).items()
            if k in ("catalog_size", "best_key", "best_score", "reason", "best_description")
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--dump",
        type=str,
        default="logs/bind_dumps/tick_18678",
        help="Bind dump directory with rgb.png + meta.json",
    )
    ap.add_argument(
        "--text",
        type=str,
        default="подойди к цилиндрическому объекту перед тобой",
    )
    args = ap.parse_args()

    dump_dir = Path(args.dump)
    if not dump_dir.is_absolute():
        dump_dir = ROOT / dump_dir

    out: dict = {
        "dump": str(dump_dir),
        "text": args.text,
        "spike": "1C_open_vocab_offline",
        "hot_path": False,
    }

    meta_path = dump_dir / "meta.json"
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        out["bind_meta_reason"] = meta.get("reason")
        out["bind_geometry_fallback"] = meta.get("geometry_fallback")
        slots = meta.get("slot_peakiness") or []
        out["slot_peakiness_max"] = max(
            (float(s.get("mask_peakiness") or 0.0) for s in slots), default=None
        )

    rgb = _load_rgb(dump_dir)
    if rgb is None:
        out["error"] = "rgb.png missing or unreadable"
        print(json.dumps(out, indent=2, ensure_ascii=False))
        return 1

    heat = _spatial_objectness(rgb)
    u, v, peak = _peak_uv(heat)
    out["spatial_proxy"] = {
        "peak_u": round(u, 4),
        "peak_v": round(v, 4),
        "peak_score": round(peak, 4),
        "note": "contrast×mid-lower FOV; not production open-vocab",
    }

    clip = _try_clip_heatmap(rgb, args.text)
    if clip is not None:
        out["clip"] = clip
    else:
        out["clip"] = {"backend": None, "note": "open_clip not installed — skipped"}

    try:
        out["ontology_text"] = _ontology_text_score(args.text)
    except Exception as exc:
        out["ontology_text"] = {"error": str(exc)}

    print(json.dumps(out, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
