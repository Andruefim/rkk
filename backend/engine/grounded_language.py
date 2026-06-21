"""
grounded_language.py — Embodied language without heavy LLM decoders.

Text ↔ CausalGraph via frozen Ollama embeddings (nomic-embed-text) and
learnable LanguageAdapterIn/Out (causal_gnn.py). Speech output uses a small
local model (qwen3.5:0.8b) only as a grammar layer over graph-grounded vectors.

RKK_GROUNDED_LANG=1
RKK_GROUNDED_LANG_CHANNELS=8
RKK_GROUNDED_LANG_EMBED_DIM=64
RKK_GROUNDED_LANG_EVERY=12          — tick cadence for speak-vector sync
RKK_GROUNDED_LANG_ALIGN_EVERY=400   — teacher-forcing alignment step
"""
from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from engine.causal_gnn import LanguageAdapterIn, LanguageAdapterOut
from engine.ollama_env import (
    get_ollama_embed_model,
    get_ollama_embeddings_url,
    get_ollama_generate_url,
    get_ollama_speech_model,
    ollama_think_disabled_payload,
)

SENSORY_PREFIX = "sensory_audio_semantic_"
INTENT_SPEAK_PREFIX = "intent_speak_"

# Motor targets for downward causal edges from language sensory nodes.
_LANG_MOTOR_TARGETS = (
    "intent_stop_recover",
    "intent_torso_forward",
    "intent_stride",
    "intent_support_left",
    "intent_support_right",
)

_BOOTSTRAP_PHRASES: tuple[tuple[str, str], ...] = (
    ("Встань, ты упал", "recover"),
    ("Get up, you fell", "recover"),
    ("Я упал", "fallen"),
    ("I fell down", "fallen"),
    ("Иди вперёд", "locomote"),
    ("Step forward", "locomote"),
    ("Повернись", "turn"),
    ("Turn around", "turn"),
    ("Помоги мне", "help"),
    ("Help me", "help"),
    ("Стабилен", "stable"),
    ("Иду вперёд", "locomote"),
)


def grounded_language_enabled() -> bool:
    return os.environ.get("RKK_GROUNDED_LANG", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


def _env_int(key: str, default: int) -> int:
    try:
        return max(1, int(os.environ.get(key, str(default))))
    except ValueError:
        return default


def _env_float(key: str, default: float) -> float:
    try:
        return float(os.environ.get(key, str(default)))
    except ValueError:
        return default


def language_channel_count() -> int:
    return _env_int("RKK_GROUNDED_LANG_CHANNELS", 8)


def language_embed_dim() -> int:
    return _env_int("RKK_GROUNDED_LANG_EMBED_DIM", 64)


def sensory_node_ids(n: int | None = None) -> list[str]:
    k = n if n is not None else language_channel_count()
    return [f"{SENSORY_PREFIX}{i}" for i in range(k)]


def intent_speak_node_ids(n: int | None = None) -> list[str]:
    k = n if n is not None else language_channel_count()
    return [f"{INTENT_SPEAK_PREFIX}{i}" for i in range(k)]


@dataclass
class PhraseEntry:
    text: str
    tag: str
    embedding: np.ndarray


class SemanticVectorStore:
    """Lightweight cosine index over phrase embeddings (no FAISS)."""

    def __init__(self) -> None:
        self._entries: list[PhraseEntry] = []

    def add(self, text: str, tag: str, embedding: np.ndarray) -> None:
        emb = np.asarray(embedding, dtype=np.float32).reshape(-1)
        n = float(np.linalg.norm(emb)) + 1e-9
        emb = emb / n
        self._entries.append(PhraseEntry(text=text, tag=tag, embedding=emb))

    def nearest(self, query: np.ndarray, *, top_k: int = 1) -> list[tuple[str, str, float]]:
        if not self._entries:
            return []
        q = np.asarray(query, dtype=np.float32).reshape(-1)
        q = q / (float(np.linalg.norm(q)) + 1e-9)
        scores: list[tuple[str, str, float]] = []
        for e in self._entries:
            sim = float(np.dot(q, e.embedding))
            scores.append((e.text, e.tag, sim))
        scores.sort(key=lambda t: -t[2])
        return scores[:top_k]

    def __len__(self) -> int:
        return len(self._entries)


class OllamaEmbeddingClient:
    """Sync Ollama /api/embeddings → numpy vector (truncated to embed_dim)."""

    def __init__(self, *, embed_dim: int | None = None) -> None:
        self.embed_dim = embed_dim or language_embed_dim()
        self._raw_dim: int | None = None
        self._lock = threading.Lock()

    def embed(self, text: str) -> np.ndarray | None:
        text = str(text or "").strip()
        if not text:
            return None
        url = get_ollama_embeddings_url()
        model = get_ollama_embed_model()
        payload = {"model": model, "prompt": text}
        try:
            import httpx

            with self._lock:
                with httpx.Client(timeout=20.0) as client:
                    resp = client.post(url, json=payload)
            if resp.status_code != 200:
                return None
            data = resp.json()
            vec = data.get("embedding")
            if not isinstance(vec, list) or not vec:
                return None
            arr = np.asarray(vec, dtype=np.float32)
            self._raw_dim = int(arr.size)
            if arr.size >= self.embed_dim:
                out = arr[: self.embed_dim]
            else:
                out = np.zeros(self.embed_dim, dtype=np.float32)
                out[: arr.size] = arr
            n = float(np.linalg.norm(out)) + 1e-9
            return out / n
        except Exception:
            return None


class OllamaSpeechDecoder:
    """Ultra-light LLM grammar layer: graph-grounded state → short utterance."""

    def __init__(self) -> None:
        self._lock = threading.Lock()

    def decode(
        self,
        *,
        nearest_phrase: str,
        lang: str = "ru",
    ) -> str:
        """Soft-prompt decode via small local model."""
        model = get_ollama_speech_model()
        url = get_ollama_generate_url().strip().rstrip("/")
        if not url.endswith("/generate"):
            url = url + "/api/generate" if "/api/" not in url else url
        lang_hint = "Russian" if lang.startswith("ru") else "English"
        prompt = (
            "You are the inner voice of a robotic agent. "
            f"Your current grounded intent concept is: [{nearest_phrase}].\n"
            f"Verbalize this concept into ONE natural, short first-person sentence in {lang_hint}. "
            "Do not add extra information or JSON.\n"
            "Utterance:"
        )
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            **ollama_think_disabled_payload(),
            "options": {"temperature": 0.35, "num_predict": 48},
        }
        try:
            import httpx

            with self._lock:
                with httpx.Client(timeout=25.0) as client:
                    resp = client.post(url, json=payload)
            if resp.status_code != 200:
                return nearest_phrase
            raw = (resp.json().get("response") or "").strip()
            if len(raw) < 2:
                return nearest_phrase
            return raw.split("\n")[0].strip().strip('"').strip("'")
        except Exception:
            return nearest_phrase


class GroundedLanguageController:
    """
    Bidirectional language ↔ graph bridge for embodied AGI.

    Input:  human text → embed → adapter_in → sensory_audio_semantic_* nodes
    Output: intent_speak_* → adapter_out → vector lookup + optional Qwen decode
    """

    def __init__(self, device: torch.device | None = None) -> None:
        self.device = device or torch.device("cpu")
        self.n_channels = language_channel_count()
        self.embed_dim = language_embed_dim()
        self.adapter_in = LanguageAdapterIn(
            self.embed_dim, self.n_channels, hidden=64
        ).to(self.device)
        self.adapter_out = LanguageAdapterOut(
            self.n_channels, self.embed_dim, hidden=64
        ).to(self.device)
        self._embed_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False).to(
            self.device
        )
        nn.init.eye_(self._embed_proj.weight)
        self.optim = torch.optim.Adam(
            list(self.adapter_in.parameters())
            + list(self.adapter_out.parameters())
            + list(self._embed_proj.parameters()),
            lr=3e-4,
        )
        self.embedder = OllamaEmbeddingClient(embed_dim=self.embed_dim)
        self.speech = OllamaSpeechDecoder()
        self.store = SemanticVectorStore()
        self._nodes_wired = False
        self._last_input_text = ""
        self._last_output_text = ""
        self._align_steps = 0
        self._bootstrap_done = False
        self._lang = os.environ.get("RKK_SPEECH_LANG", "ru")

    def bootstrap_store(self) -> int:
        if self._bootstrap_done:
            return len(self.store)
        n = 0
        for phrase, tag in _BOOTSTRAP_PHRASES:
            emb = self.embedder.embed(phrase)
            if emb is not None:
                self.store.add(phrase, tag, emb)
                n += 1
        self._bootstrap_done = n > 0
        return n

    def ensure_graph_nodes(self, graph: Any) -> None:
        """Phase 2: register sensory + speak nodes and causal edges to motor intents."""
        if self._nodes_wired:
            return
        sensory = sensory_node_ids(self.n_channels)
        speak = intent_speak_node_ids(self.n_channels)
        for nid in sensory + speak:
            if nid not in graph.nodes:
                graph.set_node(nid, 0.5)
        edge_w = _env_float("RKK_GROUNDED_LANG_EDGE_W", 0.42)
        alpha = _env_float("RKK_GROUNDED_LANG_EDGE_ALPHA", 0.12)
        for s_nid in sensory:
            for target in _LANG_MOTOR_TARGETS:
                if target in graph.nodes:
                    try:
                        graph.set_edge(s_nid, target, edge_w, alpha=alpha)
                    except Exception:
                        pass
        self._nodes_wired = True

    def _apply_attention_to_motor(self, graph: Any) -> None:
        """Boost motor precision when language sensory channels are active."""
        sensory = sensory_node_ids(self.n_channels)
        peak = max(float(graph.nodes.get(n, 0.5)) for n in sensory)
        if peak < 0.58:
            return
        weights: dict[str, float] = {}
        for t in _LANG_MOTOR_TARGETS:
            if t in graph.nodes:
                weights[t] = 1.0 + 2.2 * (peak - 0.5)
        fn = getattr(graph, "apply_symbolic_precision", None)
        if callable(fn) and weights:
            fn(weights)

    def ingest_command(self, graph: Any, text: str) -> dict[str, Any]:
        """
        Direction 1 (hearing): text → embedding → sensory graph nodes.
        """
        self.ensure_graph_nodes(graph)
        emb_np = self.embedder.embed(text)
        if emb_np is None:
            return {"ok": False, "error": "embed_failed", "text": text}
        self._last_input_text = text
        with torch.no_grad():
            emb_t = torch.tensor(emb_np, dtype=torch.float32, device=self.device)
            channels = self.adapter_in(emb_t.unsqueeze(0)).squeeze(0)
        sensory = sensory_node_ids(self.n_channels)
        for i, nid in enumerate(sensory):
            graph.nodes[nid] = float(channels[i].item())
        self._apply_attention_to_motor(graph)
        # Tag-based motor nudge from nearest bootstrap phrase
        hits = self.store.nearest(emb_np, top_k=1)
        tag = hits[0][1] if hits else ""
        motor_patch = self._motor_patch_for_tag(tag)
        for k, v in motor_patch.items():
            if k in graph.nodes:
                cur = float(graph.nodes[k])
                graph.nodes[k] = float(np.clip(0.65 * cur + 0.35 * v, 0.05, 0.95))
        return {
            "ok": True,
            "text": text,
            "tag": tag,
            "channels": [round(float(channels[i]), 4) for i in range(self.n_channels)],
            "nearest": hits[0][0] if hits else "",
        }

    @staticmethod
    def _motor_patch_for_tag(tag: str) -> dict[str, float]:
        t = str(tag or "").lower()
        if t == "recover":
            return {
                "intent_stop_recover": 0.72,
                "intent_torso_forward": 0.64,
            }
        if t == "locomote":
            return {"intent_stride": 0.66, "intent_torso_forward": 0.58}
        if t == "turn":
            return {"intent_stride": 0.48, "intent_gait_coupling": 0.72}
        if t == "help":
            return {"intent_stop_recover": 0.55}
        return {}

    def sync_speak_vector_from_state(self, graph: Any, obs: dict[str, float]) -> None:
        """Write intent_speak_* from physical state (inverse grounding for output path)."""
        self.ensure_graph_nodes(graph)
        cz = float(obs.get("com_z", obs.get("phys_com_z", 0.5)))
        ps = float(obs.get("posture_stability", obs.get("phys_posture_stability", 0.5)))
        phrase = "Я упал" if cz < 0.38 or ps < 0.35 else "Стабилен"
        emb = self.embedder.embed(phrase)
        if emb is None:
            return
        with torch.no_grad():
            emb_t = torch.tensor(emb, dtype=torch.float32, device=self.device)
            ch = self.adapter_in(emb_t.unsqueeze(0)).squeeze(0)
        for i, nid in enumerate(intent_speak_node_ids(self.n_channels)):
            graph.nodes[nid] = float(ch[i].item())

    def generate_utterance(self, graph: Any, obs: dict[str, float]) -> str:
        """
        Direction 2 (speech): intent_speak_* → adapter_out → vector store → Qwen decode.
        """
        self.ensure_graph_nodes(graph)
        speak = intent_speak_node_ids(self.n_channels)
        vals = [float(graph.nodes.get(n, 0.5)) for n in speak]
        ch_t = torch.tensor(vals, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            emb_t = self.adapter_out(ch_t).squeeze(0)
            emb_np = emb_t.cpu().numpy()
        hits = self.store.nearest(emb_np, top_k=1)
        nearest = hits[0][0] if hits else "..."
        text = self.speech.decode(
            nearest_phrase=nearest,
            lang=self._lang,
        )
        self._last_output_text = text
        return text

    def alignment_step(self, graph: Any, obs: dict[str, float]) -> float | None:
        """
        Phase 3: teacher forcing — physical state ↔ phrase embedding alignment.
        """
        cz = float(obs.get("com_z", obs.get("phys_com_z", 0.5)))
        ps = float(obs.get("posture_stability", obs.get("phys_posture_stability", 0.5)))
        if cz < 0.38 or ps < 0.35:
            target_phrase = "Я упал"
        elif ps > 0.72:
            target_phrase = "Стабилен"
        else:
            target_phrase = "Иду вперёд"
        emb_np = self.embedder.embed(target_phrase)
        if emb_np is None:
            return None
        target = torch.tensor(emb_np, dtype=torch.float32, device=self.device)
        target = F.normalize(self._embed_proj(target.unsqueeze(0)), dim=-1).squeeze(0)
        ch_tgt = self.adapter_in(target.unsqueeze(0))
        speak = intent_speak_node_ids(self.n_channels)
        cur = torch.tensor(
            [float(graph.nodes.get(n, 0.5)) for n in speak],
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)
        loss_ch = F.mse_loss(cur.detach(), ch_tgt)
        recon = self.adapter_out(ch_tgt)
        loss_emb = 1.0 - F.cosine_similarity(
            F.normalize(recon, dim=-1), target.unsqueeze(0), dim=-1
        ).mean()
        loss = loss_ch + 0.6 * loss_emb
        self.optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.adapter_in.parameters()) + list(self.adapter_out.parameters()),
            1.0,
        )
        self.optim.step()
        self._align_steps += 1
        return float(loss.item())

    def snapshot(self) -> dict[str, Any]:
        return {
            "enabled": grounded_language_enabled(),
            "n_channels": self.n_channels,
            "embed_dim": self.embed_dim,
            "store_size": len(self.store),
            "nodes_wired": self._nodes_wired,
            "align_steps": self._align_steps,
            "last_input": self._last_input_text[:80],
            "last_output": self._last_output_text[:80],
            "embed_model": get_ollama_embed_model(),
            "speech_model": get_ollama_speech_model(),
        }
