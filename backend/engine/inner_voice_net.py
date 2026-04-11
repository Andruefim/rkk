"""
inner_voice_net.py — InnerVoiceNet: быстрая внутренняя речь агента.

Заменяет LLM в real-time когнитивном цикле (τ2, каждые 300 тиков).
Обучается через дистилляцию от LLM teacher (раз в 60-120 секунд).

Architecture:
  StateEncoder: GNN state vector (d) → z (128)
  GRU:          (h_{t-1}, z) → h  ← continuous internal monologue
  ThoughtHead:  h → thought_embedding (64)  ← semantic internal state
  MonologueHead: h → monologue_logits (vocab=512)  ← concept predictions

Training (offline, from LLM distillation):
  LLM смотрит на GNN state → генерирует verbal annotation
  → LLM embedding → text_proj → target_embedding
  → MSE loss: ThoughtHead ≈ target_embedding (semantic alignment)
  → ConceptStore.contrastive_step: top concepts aligned

Runtime (no LLM):
  Every 300 ticks:
    h = GRU(h_prev, encode(GNN_state))
    thought = ThoughtHead(h)
    active_concepts = ConceptStore.query(thought, k=5)
    → inject concept_* nodes into GNN
    → update τ2 buffer in MultiscaleTimeController

GNN injection:
  concept_FALLING_BACKWARD = 0.87
  concept_HIGH_FALL_RISK   = 0.81
  concept_LOW_STABILITY    = 0.78
  ... (top-5 active concepts as GNN nodes)
  → GNN теперь понимает ситуацию семантически, не только числами

RKK_INNER_VOICE_ENABLED=1
RKK_INNER_VOICE_HIDDEN=128
RKK_INNER_VOICE_CONCEPTS=5     — top-K concepts active at once
RKK_INNER_VOICE_EVERY=300      — тиков между monologue updates (τ2)
RKK_INNER_VOICE_TRAIN_EVERY=60 — секунд между LLM teacher calls
"""
from __future__ import annotations

import os
from collections import deque
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from engine.concept_store import CONCEPT_DIM, VOCAB_SIZE, SemanticConceptStore


def inner_voice_enabled() -> bool:
    return os.environ.get("RKK_INNER_VOICE_ENABLED", "1").strip().lower() not in (
        "0", "false", "no", "off"
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


def _safe_act(x: object) -> float:
    """Concept activation may be numpy / str from upstream."""
    try:
        return float(x)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0


# ── State encoder ─────────────────────────────────────────────────────────────
class StateEncoder(nn.Module):
    """
    Encodes GNN state vector → latent z.
    Handles variable-length input (d changes as nodes are added).
    Uses attention over node embeddings rather than fixed MLP.
    """

    def __init__(self, max_d: int = 128, hidden: int = 128, device: torch.device | None = None):
        super().__init__()
        dev = device or torch.device("cpu")
        self.max_d = max_d
        self.hidden = hidden

        # Node-wise embedding (each scalar node → hidden)
        self.node_emb = nn.Sequential(
            nn.Linear(1, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
        )

        # Self-attention over nodes
        self.attn_q = nn.Linear(32, 32)
        self.attn_k = nn.Linear(32, 32)
        self.attn_v = nn.Linear(32, 32)

        # Pool → hidden
        self.pool_proj = nn.Sequential(
            nn.Linear(32, hidden),
            nn.LayerNorm(hidden),
            nn.Tanh(),
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.4)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        self.to(dev)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, d) — GNN state, d can vary
        Returns: (B, hidden)
        """
        B, d = x.shape
        # Reshape to (B, d, 1) for node-wise embedding
        x_nodes = x.unsqueeze(-1)          # (B, d, 1)
        # Flatten batch for Linear
        x_flat = x_nodes.view(B * d, 1)
        node_embs = self.node_emb(x_flat).view(B, d, 32)  # (B, d, 32)

        # Scaled dot-product attention
        Q = self.attn_q(node_embs)  # (B, d, 32)
        K = self.attn_k(node_embs)
        V = self.attn_v(node_embs)

        scale = 32 ** 0.5
        attn = torch.softmax(Q @ K.transpose(-2, -1) / scale, dim=-1)  # (B, d, d)
        attended = (attn @ V).mean(dim=1)  # (B, 32) — mean pool over nodes

        return self.pool_proj(attended)     # (B, hidden)


# ── InnerVoiceNet ──────────────────────────────────────────────────────────────
class InnerVoiceNet(nn.Module):
    """
    Internal monologue network (~2M parameters).

    Continuous GRU-based internal state that runs every 300 ticks
    without LLM involvement. Produces:
      thought_embedding (64): semantic position in concept space
      monologue_logits (512):  soft concept activations

    Trained offline by LLM teacher through distillation.
    """

    def __init__(
        self,
        max_d: int = 128,
        hidden: int = 128,
        concept_dim: int = CONCEPT_DIM,
        device: torch.device | None = None,
    ):
        super().__init__()
        dev = device or torch.device("cpu")
        self.device = dev
        self.hidden = hidden
        self.concept_dim = concept_dim

        # State encoder
        self.encoder = StateEncoder(max_d=max_d, hidden=hidden, device=dev)

        # Recurrent core — continuous internal monologue
        self.gru = nn.GRUCell(hidden, hidden)

        # Thought head: h → thought_embedding (concept space)
        self.thought_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.Tanh(),
            nn.Linear(hidden // 2, concept_dim),
        )

        # Monologue head: h → soft concept activations
        self.monologue_head = nn.Sequential(
            nn.Linear(hidden, VOCAB_SIZE),
        )

        # Attention gate: thought modulates GNN node attention
        self.gate_head = nn.Sequential(
            nn.Linear(concept_dim, max_d),
            nn.Sigmoid(),
        )

        for m in self.modules():
            if isinstance(m, (nn.Linear,)):
                nn.init.orthogonal_(m.weight, gain=0.3)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        self.to(dev)

        # Internal hidden state
        self._h = torch.zeros(1, hidden, device=dev)
        self._last_thought: torch.Tensor = torch.zeros(1, concept_dim, device=dev)
        self._last_concepts: list[tuple[str, float]] = []

    def reset_hidden(self) -> None:
        self._h = torch.zeros(1, self.hidden, device=self.device)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        x: (1, d) GNN state
        Returns: (thought_emb, monologue_logits, gru_h)
        """
        z = self.encoder(x)                          # (1, hidden)
        h = self.gru(z, self._h)                     # (1, hidden)
        self._h = h.detach()

        thought = self.thought_head(h)               # (1, concept_dim)
        monologue = self.monologue_head(h)           # (1, vocab_size)
        return thought, monologue, h

    def forward_stateless(
        self, x: torch.Tensor, h: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """For batch training without updating internal state."""
        z = self.encoder(x)
        h_next = self.gru(z, h)
        thought = self.thought_head(h_next)
        monologue = self.monologue_head(h_next)
        return thought, monologue, h_next

    @torch.no_grad()
    def infer(
        self,
        state_vec: list[float],
        concept_store: SemanticConceptStore,
        k: int | None = None,
    ) -> dict[str, Any]:
        """
        Full inference pass. Returns thought_embedding + active concepts.
        """
        self.eval()
        k = k or _env_int("RKK_INNER_VOICE_CONCEPTS", 5)

        d = len(state_vec)
        x = torch.tensor(state_vec, dtype=torch.float32, device=self.device).view(1, d)
        thought, monologue, _ = self(x)

        # Query concept store
        active_concepts = concept_store.query(thought.squeeze(0), k=k)

        self._last_thought = thought
        self._last_concepts = active_concepts

        # Attention gate for GNN (which nodes to pay attention to)
        gate = self.gate_head(thought.squeeze(0))  # (max_d,)

        return {
            "thought_embedding": thought.squeeze(0).cpu().tolist(),
            "active_concepts": active_concepts,
            "monologue_logits": monologue.squeeze(0).cpu().tolist(),
            "gate_vector": gate.cpu().tolist()[:d],  # clipped to actual d
        }

    def get_attention_gate(self, d: int) -> list[float]:
        """Get current attention gate vector for GNN (d nodes)."""
        with torch.no_grad():
            gate = self.gate_head(self._last_thought.squeeze(0))
            return gate.cpu().tolist()[:d]


# ── Inner Voice Controller ─────────────────────────────────────────────────────
class InnerVoiceController:
    """
    Полный контроллер внутреннего монолога.

    Основной цикл (без LLM, каждые RKK_INNER_VOICE_EVERY тиков):
      1. Собирает GNN state vector
      2. InnerVoiceNet.infer() → thought_embedding + active_concepts
      3. Инжектирует concept_* узлы в GNN
      4. Предоставляет thought_embedding для RewardCoordinator

    Distillation (с LLM, каждые ~60 секунд офлайн):
      LLMVoiceTeacher.generate_annotation() → verbal + concept_labels
      → InnerVoiceTrainer.step() → обновляет веса InnerVoiceNet

    Интеграция в simulation.py:
      self._inner_voice = InnerVoiceController(device)
      # В тик-цикле:
      if self._timescale.should_run(LEVEL_COGNIT, tick):
          result = self._inner_voice.tick(tick, agent.graph, agent.env)
    """

    def __init__(self, device: torch.device, max_d: int = 128):
        self.device = device
        self.max_d = max_d
        hidden = _env_int("RKK_INNER_VOICE_HIDDEN", 128)

        self.net = InnerVoiceNet(max_d=max_d, hidden=hidden, device=device)
        self.concept_store = SemanticConceptStore(device=device)
        self.optim = torch.optim.Adam(
            list(self.net.parameters()) + list(self.concept_store.parameters()),
            lr=3e-4,
        )

        self._every = _env_int("RKK_INNER_VOICE_EVERY", 300)
        self._last_tick = -9999
        self._last_result: dict[str, Any] = {}
        self._active_concept_nodes: set[str] = set()

        # Training state
        self._train_buf: deque[tuple[list[float], list[str]]] = deque(maxlen=256)
        self.train_steps = 0
        self._distill_loss_history: deque[float] = deque(maxlen=50)

        # Concept injection history (for UI)
        self._concept_history: deque[list[tuple[str, float]]] = deque(maxlen=20)

        self.total_inferences = 0

    def tick(
        self,
        tick: int,
        graph,
        env=None,
    ) -> dict[str, Any]:
        """
        Main update. Call when timescale LEVEL_COGNIT fires.
        """
        if not inner_voice_enabled():
            return {}
        if (tick - self._last_tick) < self._every:
            return self._last_result

        self._last_tick = tick

        # Build state vector from GNN
        node_ids = list(graph._node_ids) if hasattr(graph, '_node_ids') else list(graph.nodes.keys())
        state_vec = [float(graph.nodes.get(n, 0.5)) for n in node_ids]

        if not state_vec:
            return {}

        # Infer thought + concepts
        k = _env_int("RKK_INNER_VOICE_CONCEPTS", 5)
        result = self.net.infer(state_vec, self.concept_store, k=k)
        self._last_result = result
        self.total_inferences += 1

        active = result.get("active_concepts", [])
        self._concept_history.append(active)

        # Inject active concepts into GNN
        self._inject_concepts_to_graph(graph, active)

        # Apply attention gate to GNN (weight causal edges by thought relevance)
        gate = result.get("gate_vector", [])
        if gate and hasattr(graph, '_attention_gate'):
            graph._attention_gate = gate[:len(node_ids)]

        return result

    def _inject_concepts_to_graph(
        self,
        graph,
        active_concepts: list[tuple[str, float]],
    ) -> None:
        """
        Add/update concept_* nodes in GNN with activation values.
        Remove stale concepts not in current top-k.
        """
        current_names = {f"concept_{name.lower()}" for name, _ in active_concepts}

        # Remove stale concept nodes
        for node_name in list(self._active_concept_nodes):
            if node_name not in current_names:
                if node_name in graph.nodes:
                    graph.nodes[node_name] = 0.0  # fade out
        self._active_concept_nodes = current_names

        # Set/update active concept nodes
        for name, activation in active_concepts:
            node_key = f"concept_{name.lower()}"
            if node_key not in graph.nodes:
                try:
                    graph.set_node(node_key, float(activation))
                    # Add semantic edges based on concept domain
                    self._add_concept_edges(graph, name, node_key, activation)
                except Exception:
                    pass
            else:
                graph.nodes[node_key] = float(activation)

    def _add_concept_edges(
        self,
        graph,
        concept_name: str,
        node_key: str,
        activation: float,
    ) -> None:
        """Add causal edges from concept nodes to relevant GNN variables."""
        from engine.concept_store import CONCEPT_DEFS

        # Find domain of this concept
        domain = next(
            (d for n, d, _ in CONCEPT_DEFS if n == concept_name), "LATENT"
        )

        edges: list[tuple[str, str, float]] = []

        if domain == "FALL":
            edges = [
                (node_key, "intent_stop_recover", 0.35),
                (node_key, "posture_stability", -0.20),
            ]
        elif domain == "STABILITY":
            edges = [
                (node_key, "posture_stability", 0.25),
                (node_key, "com_z", 0.20),
            ]
        elif domain == "GAIT":
            edges = [
                (node_key, "intent_stride", 0.22),
                (node_key, "intent_gait_coupling", 0.18),
            ]
        elif domain == "RECOVERY":
            edges = [
                (node_key, "intent_stop_recover", 0.30),
            ]
        elif domain == "EMPOWER":
            edges = [
                (node_key, "proprio_empowerment", 0.20),
            ]

        for fr, to, w in edges:
            if fr in graph.nodes and to in graph.nodes:
                try:
                    graph.set_edge(fr, to, w, alpha=0.06)
                except Exception:
                    pass

    # ── Training ───────────────────────────────────────────────────────────────
    def push_distill_sample(
        self,
        state_vec: list[float],
        concept_labels: list[str],
    ) -> None:
        """Push a (state, concept_labels) pair for training."""
        self._train_buf.append((list(state_vec), list(concept_labels)))

    def train_step(self) -> float | None:
        """One distillation training step."""
        if len(self._train_buf) < 8:
            return None

        batch = list(self._train_buf)[-32:]
        B = len(batch)

        # Build tensors
        d = len(batch[0][0])
        X = torch.zeros(B, d, device=self.device)
        for i, (sv, _) in enumerate(batch):
            l = min(len(sv), d)
            X[i, :l] = torch.tensor(sv[:l], dtype=torch.float32)

        # Forward through InnerVoiceNet
        h = torch.zeros(1, self.net.hidden, device=self.device)
        thoughts_list = []
        for i in range(B):
            t, m, h = self.net.forward_stateless(X[i:i+1], h)
            thoughts_list.append(t)
        thoughts = torch.cat(thoughts_list, dim=0)  # (B, concept_dim)

        # Contrastive loss on concept_store
        all_labels = [labels[0] if labels else "STABLE_BALANCE" for _, labels in batch]
        contr_loss = self.concept_store.contrastive_step(
            thoughts.detach(), all_labels, temperature=0.07
        )

        # Monologue distribution loss: thought should have high cosine with target concept emb
        target_embs = []
        for labels in [b[1] for b in batch]:
            if labels:
                emb = self.concept_store.get_embedding(labels[0])
                if emb is not None:
                    target_embs.append(emb)
                else:
                    target_embs.append(torch.zeros(CONCEPT_DIM, device=self.device))
            else:
                target_embs.append(torch.zeros(CONCEPT_DIM, device=self.device))
        target_embs_t = torch.stack(target_embs[:B], dim=0).to(self.device)

        self.optim.zero_grad()
        cosine_loss = (1.0 - F.cosine_similarity(thoughts, target_embs_t.detach(), dim=-1)).mean()
        cosine_loss.backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)
        self.optim.step()

        total_loss = contr_loss + float(cosine_loss.item())
        self._distill_loss_history.append(total_loss)
        self.train_steps += 1
        return total_loss

    def get_thought_embedding(self) -> list[float]:
        """Get current thought embedding for RewardCoordinator curiosity."""
        t = self.net._last_thought
        if t is not None:
            return t.squeeze(0).detach().cpu().tolist()
        return [0.0] * CONCEPT_DIM

    def get_active_concepts(self) -> list[tuple[str, float]]:
        return self.net._last_concepts

    def get_concept_str(self, max_concepts: int = 3) -> str:
        """Human-readable concept description for LLM/UI."""
        concepts = self.net._last_concepts[:max_concepts]
        if not concepts:
            return "UNKNOWN"
        return " + ".join(f"{n}({_safe_act(v):.2f})" for n, v in concepts)

    def snapshot(self) -> dict[str, Any]:
        active = self.net._last_concepts
        return {
            "enabled": inner_voice_enabled(),
            "total_inferences": self.total_inferences,
            "train_steps": self.train_steps,
            "distill_loss_mean": round(
                float(np.mean(list(self._distill_loss_history))), 5
            ) if self._distill_loss_history else 0.0,
            "active_concepts": [(n, round(_safe_act(v), 3)) for n, v in active],
            "concept_str": self.get_concept_str(),
            "active_concept_nodes": len(self._active_concept_nodes),
            "train_buf_len": len(self._train_buf),
            "concept_store": self.concept_store.snapshot(),
        }
