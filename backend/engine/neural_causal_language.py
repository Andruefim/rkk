"""
neural_causal_language.py — Фаза P: Neural Causal Language Emergence.

Полная замена хардкода на нейросети:
  - TEMPLATES_RU/EN → CausalSpeechDecoder (GRU + vocab head)
  - SLOT_PROPERTY_TO_CONCEPTS → NeuralConceptProjector (learned mapping)
  - position_to_spatial_concepts → InterventionalSpatialMemory (causal)
  - text_to_concepts keyword matching → LearnedSlotSemantics (contrastive)

Принцип: язык, зрение, движение — один поток каузального обучения.
Агент учится описывать мир через интервенции, а не через lookup table.

Обучение:
  τ3 LLM teacher → verbal annotation → дистилляция в CausalSpeechDecoder
  Slot vectors × GNN interventional co-variance → LearnedSlotSemantics
  Agent actions × slot_delta → InterventionalSpatialMemory

RKK_NEURAL_LANG_ENABLED=1
RKK_NEURAL_LANG_VOCAB=512         — размер токенного словаря
RKK_NEURAL_LANG_HIDDEN=128        — hidden size декодера
RKK_NEURAL_LANG_MAX_LEN=32        — макс длина генерируемой фразы
RKK_NEURAL_LANG_TEMP=0.8          — температура сэмплинга
RKK_NEURAL_LANG_DISTILL_LR=3e-4   — lr дистилляции от LLM
RKK_SPATIAL_MEM_SLOTS=32          — ячеек топологической памяти
RKK_SPATIAL_MEM_DECAY=0.95        — decay незанятых ячеек
"""
from __future__ import annotations

import os
import time
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── Env helpers ──────────────────────────────────────────────────────────────
def neural_lang_enabled() -> bool:
    return os.environ.get("RKK_NEURAL_LANG_ENABLED", "1").strip().lower() not in (
        "0", "false", "no", "off"
    )

def _ei(key: str, default: int) -> int:
    try: return max(1, int(os.environ.get(key, str(default))))
    except ValueError: return default

def _ef(key: str, default: float) -> float:
    try: return float(os.environ.get(key, str(default)))
    except ValueError: return default


# ─── Learned vocabulary ───────────────────────────────────────────────────────
# Начальный биомеханический словарь — prior, не template.
# Сеть учится его переписывать через дистилляцию.
BIOMECH_VOCAB_SEEDS = [
    # Body state tokens
    "<pad>", "<sos>", "<eos>", "<unk>",
    "я", "я падаю", "я встаю", "я стою", "я иду", "я теряю",
    "равновесие", "баланс", "ноги", "тело", "торс", "колено", "бедро",
    "левый", "правый", "впереди", "сзади", "слева", "справа",
    "устойчиво", "неустойчиво", "падение", "восстановление", "шаг",
    # Perception tokens
    "вижу", "замечаю", "объект", "поверхность", "препятствие", "путь",
    "близко", "далеко", "сверху", "снизу", "рядом", "открыто",
    "новый", "незнакомый", "знакомый", "неожиданный", "изменился",
    # Action tokens  
    "пробую", "хочу", "могу", "нужно", "попробую", "интересно",
    "что", "если", "почему", "как", "этот", "тот", "здесь", "там",
    # State descriptors
    "высокий", "низкий", "быстрый", "медленный", "сильный", "слабый",
    "хорошо", "плохо", "лучше", "хуже", "нормально", "критично",
    # Causal connectors
    "потому что", "значит", "поэтому", "когда", "после", "до",
    "если бы", "тогда", "следовательно", "вызывает", "влияет",
]

# Дополняем до VOCAB_SIZE пустыми слотами (сеть их заполнит через обучение)
_BASE_VOCAB_SIZE = _ei("RKK_NEURAL_LANG_VOCAB", 512)
while len(BIOMECH_VOCAB_SEEDS) < _BASE_VOCAB_SIZE:
    BIOMECH_VOCAB_SEEDS.append(f"<tok_{len(BIOMECH_VOCAB_SEEDS)}>")

VOCAB_SIZE = len(BIOMECH_VOCAB_SEEDS)
TOKEN_TO_IDX = {tok: i for i, tok in enumerate(BIOMECH_VOCAB_SEEDS)}
IDX_TO_TOKEN = {i: tok for tok, i in TOKEN_TO_IDX.items()}
PAD_IDX = TOKEN_TO_IDX["<pad>"]
SOS_IDX = TOKEN_TO_IDX["<sos>"]
EOS_IDX = TOKEN_TO_IDX["<eos>"]


# ─── CausalSpeechDecoder ──────────────────────────────────────────────────────
class CausalSpeechDecoder(nn.Module):
    """
    GRU-based speech decoder: thought_embedding → token sequence.

    Обучается ОНЛАЙН через дистилляцию от LLM teacher (τ3).
    
    LLM teacher видит float состояния → генерирует текст →
    текст токенизируется → cross-entropy loss обновляет декодер.
    
    Без LLM: автономная генерация из thought vector через greedy/sampling.
    
    Ключевое: НЕТ ЕДИНОГО ЗАХАРДКОЖЕННОГО ШАБЛОНА.
    Всё что декодер говорит — он выучил сам из interventional experience.
    """

    def __init__(
        self,
        concept_dim: int = 64,
        hidden: int = 128,
        vocab_size: int = VOCAB_SIZE,
        max_len: int = 32,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.device = device or torch.device("cpu")
        self.concept_dim = concept_dim
        self.hidden = hidden
        self.vocab_size = vocab_size
        self.max_len = max_len

        # Thought embedding → decoder init hidden
        self.thought_proj = nn.Sequential(
            nn.Linear(concept_dim, hidden),
            nn.Tanh(),
        )

        # Token embedding
        self.token_emb = nn.Embedding(vocab_size, hidden, padding_idx=PAD_IDX)

        # Causal condition: concat thought context at each step
        self.context_proj = nn.Linear(concept_dim, hidden)

        # GRU decoder
        self.gru = nn.GRUCell(hidden * 2, hidden)  # [token_emb; context] → h

        # Output projection
        self.out_proj = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, vocab_size),
        )

        # Causal state: GNN node history influences word choice
        # This makes speech causally grounded in body state
        self.state_attention = nn.MultiheadAttention(
            hidden, num_heads=4, batch_first=True, dropout=0.0
        )
        self.state_proj = nn.Linear(1, hidden)  # scalar node → hidden

        self._init_weights()
        self.to(self.device)

        # Training state
        self.train_steps = 0
        self._loss_history: deque[float] = deque(maxlen=100)
        self.optim = torch.optim.Adam(self.parameters(), lr=_ef("RKK_NEURAL_LANG_DISTILL_LR", 3e-4))

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.4)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def _encode_state(
        self,
        state_vec: torch.Tensor | None,
        thought: torch.Tensor,
    ) -> torch.Tensor:
        """
        Attend over GNN state nodes, conditioned on thought.
        Returns context vector (1, hidden).
        """
        if state_vec is None or state_vec.numel() == 0:
            return thought.unsqueeze(0)  # (1, hidden)

        state_vec = state_vec.to(device=self.device, dtype=torch.float32)

        # Project each scalar node to hidden dim
        d = state_vec.shape[-1]
        sv = state_vec.view(1, d, 1)  # (1, d, 1)
        kv = self.state_proj(sv)       # (1, d, hidden)

        q = thought.unsqueeze(0).unsqueeze(0)  # (1, 1, hidden)
        out, _ = self.state_attention(q, kv, kv)  # (1, 1, hidden)
        return out.squeeze(1)  # (1, hidden)

    @torch.no_grad()
    def generate(
        self,
        thought_emb: torch.Tensor,
        state_vec: torch.Tensor | None = None,
        temperature: float | None = None,
        max_len: int | None = None,
    ) -> str:
        """
        Autoregressive generation from thought embedding.
        No templates — pure neural decoding.
        """
        self.eval()
        T = temperature or _ef("RKK_NEURAL_LANG_TEMP", 0.8)
        L = max_len or self.max_len

        thought = thought_emb.to(self.device).view(1, -1)
        if thought.shape[-1] != self.concept_dim:
            # Pad or truncate
            if thought.shape[-1] < self.concept_dim:
                thought = F.pad(thought, (0, self.concept_dim - thought.shape[-1]))
            else:
                thought = thought[:, :self.concept_dim]

        h = self.thought_proj(thought).squeeze(0)          # (hidden,)
        ctx = self._encode_state(state_vec, h)             # (1, hidden)
        context = (ctx + self.context_proj(thought)).squeeze(0)  # (hidden,)

        tokens: list[int] = []
        token_id = SOS_IDX

        for _ in range(L):
            tok_emb = self.token_emb(
                torch.tensor([token_id], device=self.device)
            ).squeeze(0)  # (hidden,)

            inp = torch.cat([tok_emb, context], dim=-1).unsqueeze(0)  # (1, hidden*2)
            h = self.gru(inp.squeeze(0), h)

            logits = self.out_proj(h)  # (vocab_size,)

            # Temperature sampling
            if T > 0:
                probs = F.softmax(logits / T, dim=-1)
                token_id = int(torch.multinomial(probs, 1).item())
            else:
                token_id = int(logits.argmax().item())

            if token_id == EOS_IDX:
                break
            if token_id != PAD_IDX:
                tokens.append(token_id)

        words = [IDX_TO_TOKEN.get(t, "<unk>") for t in tokens]
        # Filter internal tokens
        words = [w for w in words if not w.startswith("<")]
        return " ".join(words) if words else ""

    def distill_step(
        self,
        thought_emb: torch.Tensor,
        target_text: str,
        state_vec: torch.Tensor | None = None,
    ) -> float | None:
        """
        Teacher-forced training: thought → target_text (from LLM).
        This is the ONLY way decoder learns — no hardcoded knowledge.
        """
        if not target_text.strip():
            return None

        # Tokenize target text
        target_tokens = self._tokenize(target_text)
        if len(target_tokens) < 2:
            return None

        self.train()
        thought = thought_emb.to(self.device).view(1, -1)
        if thought.shape[-1] != self.concept_dim:
            if thought.shape[-1] < self.concept_dim:
                thought = F.pad(thought, (0, self.concept_dim - thought.shape[-1]))
            else:
                thought = thought[:, :self.concept_dim]

        h = self.thought_proj(thought).squeeze(0)
        ctx = self._encode_state(state_vec, h)
        context = (ctx + self.context_proj(thought)).squeeze(0)

        # Teacher forcing: feed target tokens, predict next
        input_ids = [SOS_IDX] + target_tokens[:-1]
        target_ids = target_tokens

        total_loss = torch.tensor(0.0, device=self.device)
        for inp_id, tgt_id in zip(input_ids, target_ids):
            tok_emb = self.token_emb(torch.tensor([inp_id], device=self.device)).squeeze(0)
            inp = torch.cat([tok_emb, context], dim=-1).unsqueeze(0)
            h = self.gru(inp.squeeze(0), h)
            logits = self.out_proj(h)
            total_loss = total_loss + F.cross_entropy(
                logits.unsqueeze(0),
                torch.tensor([tgt_id], device=self.device),
            )

        loss = total_loss / max(len(target_ids), 1)
        self.optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), 0.5)
        self.optim.step()

        v = float(loss.item())
        self._loss_history.append(v)
        self.train_steps += 1
        return v

    def _tokenize(self, text: str) -> list[int]:
        """Simple word-level tokenization against learned vocab."""
        tokens: list[int] = []
        words = text.lower().strip().split()
        for w in words:
            if w in TOKEN_TO_IDX:
                tokens.append(TOKEN_TO_IDX[w])
            else:
                # Try partial match (multi-word tokens)
                tokens.append(TOKEN_TO_IDX.get("<unk>", 3))
        tokens.append(EOS_IDX)
        return tokens

    def snapshot(self) -> dict[str, Any]:
        mean_loss = float(np.mean(list(self._loss_history))) if self._loss_history else 0.0
        return {
            "train_steps": self.train_steps,
            "mean_loss": round(mean_loss, 5),
            "vocab_size": self.vocab_size,
            "max_len": self.max_len,
        }


# ─── NeuralConceptProjector ───────────────────────────────────────────────────
class NeuralConceptProjector(nn.Module):
    """
    Slot vectors → semantic concept activations.
    
    Заменяет захардкоженный SLOT_PROPERTY_TO_CONCEPTS.
    
    Обучается через contrastive loss:
      - Positive pairs: (slot_vec, concept_emb) когда slot causally связан с concept
      - Negative pairs: всё остальное
    
    Causally linked = slot_k изменился одновременно с тем что GNN 
    зарегистрировал определённый паттерн состояний (через IntervCovariance).
    """

    def __init__(
        self,
        slot_dim: int = 64,
        concept_dim: int = 64,
        n_concepts: int = 512,
        hidden: int = 128,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.device = device or torch.device("cpu")
        self.slot_dim = slot_dim
        self.concept_dim = concept_dim
        self.n_concepts = n_concepts

        # Slot encoder: slot_vec → concept space
        self.slot_enc = nn.Sequential(
            nn.Linear(slot_dim, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, concept_dim),
        )

        # Interventional modulation:
        # GNN state delta conditions the projection
        # "what changed in body state when this slot became active?"
        self.intervention_gate = nn.Sequential(
            nn.Linear(slot_dim + concept_dim, concept_dim),
            nn.Sigmoid(),
        )

        # Concept embeddings (shared with SemanticConceptStore if available)
        self.concept_embs = nn.Embedding(n_concepts, concept_dim)

        self._init_weights()
        self.to(self.device)

        self.train_steps = 0
        self._loss_history: deque[float] = deque(maxlen=100)
        self.optim = torch.optim.Adam(self.parameters(), lr=3e-4)

        # Interventional co-variance tracking
        # slot_k → {concept_idx: co-occurrence_count}
        self._slot_concept_covar: dict[int, dict[int, float]] = {}
        self._covar_decay = _ef("RKK_SPATIAL_MEM_DECAY", 0.95)

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.4)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.1)

    def project(
        self,
        slot_vec: torch.Tensor,
        state_delta: torch.Tensor | None = None,
        top_k: int = 5,
        threshold: float = 0.45,
    ) -> list[tuple[int, float]]:
        """
        Slot vector → [(concept_idx, activation), ...]
        No keyword matching. Pure learned projection.
        """
        with torch.no_grad():
            self.eval()
            sv = slot_vec.to(self.device).view(1, -1)
            if sv.shape[-1] != self.slot_dim:
                sv = F.pad(sv, (0, max(0, self.slot_dim - sv.shape[-1])))[:, :self.slot_dim]

            # Encode to concept space
            z = self.slot_enc(sv)  # (1, concept_dim)

            # Interventional modulation: if we know what changed, condition on it
            if state_delta is not None:
                delta_norm = state_delta.to(self.device).view(1, -1)
                if delta_norm.shape[-1] > self.slot_dim:
                    delta_norm = delta_norm[:, :self.slot_dim]
                elif delta_norm.shape[-1] < self.slot_dim:
                    delta_norm = F.pad(delta_norm, (0, self.slot_dim - delta_norm.shape[-1]))
                gate_in = torch.cat([delta_norm, z], dim=-1)
                gate = self.intervention_gate(gate_in)
                z = z * gate

            # Cosine similarity to all concept embeddings
            z_norm = F.normalize(z, dim=-1)
            c_norm = F.normalize(self.concept_embs.weight, dim=-1)
            sims = (z_norm @ c_norm.T).squeeze(0)  # (n_concepts,)

            top_vals, top_idx = sims.topk(min(top_k * 2, self.n_concepts))
            results: list[tuple[int, float]] = []
            for val, idx in zip(top_vals.tolist(), top_idx.tolist()):
                if val >= threshold:
                    results.append((int(idx), float(val)))
                if len(results) >= top_k:
                    break
            return results

    def contrastive_step(
        self,
        slot_vec: torch.Tensor,
        positive_concept_idxs: list[int],
        negative_concept_idxs: list[int] | None = None,
        state_delta: torch.Tensor | None = None,
        temperature: float = 0.07,
    ) -> float | None:
        """
        Contrastive learning: slot_vec should be close to positive concepts,
        far from negatives.
        
        Called when we know (from LLM annotation or bodily co-occurrence)
        that this slot is associated with certain concepts.
        """
        if not positive_concept_idxs:
            return None

        self.train()
        sv = slot_vec.to(self.device).view(1, -1)
        if sv.shape[-1] != self.slot_dim:
            sv = F.pad(sv, (0, max(0, self.slot_dim - sv.shape[-1])))[:, :self.slot_dim]

        z = self.slot_enc(sv)  # (1, concept_dim)

        if state_delta is not None:
            delta_norm = state_delta.to(self.device).view(1, -1)
            if delta_norm.shape[-1] > self.slot_dim:
                delta_norm = delta_norm[:, :self.slot_dim]
            elif delta_norm.shape[-1] < self.slot_dim:
                delta_norm = F.pad(delta_norm, (0, self.slot_dim - delta_norm.shape[-1]))
            gate_in = torch.cat([delta_norm, z], dim=-1)
            gate = self.intervention_gate(gate_in)
            z = z * gate

        z_norm = F.normalize(z, dim=-1)

        # Positive loss: push toward positive concept embeddings
        pos_idx = torch.tensor(positive_concept_idxs, device=self.device)
        pos_embs = F.normalize(self.concept_embs(pos_idx), dim=-1)  # (P, concept_dim)
        pos_sims = (z_norm @ pos_embs.T) / temperature  # (1, P)
        pos_loss = -pos_sims.mean()

        # Negative loss (contrastive): push away from all others
        all_embs = F.normalize(self.concept_embs.weight, dim=-1)
        all_sims = (z_norm @ all_embs.T) / temperature  # (1, n_concepts)
        neg_loss = torch.logsumexp(all_sims, dim=-1).mean()

        loss = pos_loss + neg_loss * 0.1

        self.optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), 0.5)
        self.optim.step()

        v = float(loss.item())
        self._loss_history.append(v)
        self.train_steps += 1
        return v

    def record_covariance(self, slot_k: int, concept_idxs: list[int]) -> None:
        """Track which concepts co-occur with which slots over time."""
        if slot_k not in self._slot_concept_covar:
            self._slot_concept_covar[slot_k] = {}
        cov = self._slot_concept_covar[slot_k]
        # Decay all
        for c in list(cov.keys()):
            cov[c] *= self._covar_decay
            if cov[c] < 0.01:
                del cov[c]
        # Increment observed
        for c in concept_idxs:
            cov[c] = cov.get(c, 0.0) + 1.0

    def get_slot_concept_prior(self, slot_k: int, top_k: int = 5) -> list[tuple[int, float]]:
        """Get strongest historical concept associations for slot (interventional prior)."""
        cov = self._slot_concept_covar.get(slot_k, {})
        if not cov:
            return []
        sorted_cov = sorted(cov.items(), key=lambda x: -x[1])
        total = sum(v for _, v in sorted_cov[:top_k])
        return [(c, v / max(total, 1e-8)) for c, v in sorted_cov[:top_k]]

    def snapshot(self) -> dict[str, Any]:
        mean_loss = float(np.mean(list(self._loss_history))) if self._loss_history else 0.0
        return {
            "train_steps": self.train_steps,
            "mean_loss": round(mean_loss, 5),
            "n_slots_tracked": len(self._slot_concept_covar),
        }


# ─── InterventionalSpatialMemory ─────────────────────────────────────────────
class InterventionalSpatialMemory(nn.Module):
    """
    Топологическая карта пространства через интервенции.
    
    Заменяет хардкоженый position_to_spatial_concepts().
    
    Принцип: пространственные концепты возникают из interventional history.
    "Объект слева" = когда агент делал do(intent_stride=0.7),
    slot_k ВСЕГДА увеличивался И foot_contact_l уменьшался.
    Это causal signature — не правило "x < 0.35 → LEFT".
    
    Архитектура:
      - M memory slots (32 by default)
      - Каждый слот: (slot_signature, action_signature, outcome_signature, position_belief)
      - Write: при каждом do() записываем (slot_before, action, slot_after, body_after)
      - Read: при новом slot_vec → find nearest slot → decode spatial context
    """

    def __init__(
        self,
        slot_dim: int = 64,
        body_dim: int = 16,
        action_dim: int = 16,
        n_memory: int = 32,
        hidden: int = 64,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.device = device or torch.device("cpu")
        self.slot_dim = slot_dim
        self.body_dim = body_dim
        self.action_dim = action_dim
        self.n_memory = n_memory
        self.hidden = hidden

        # Memory: each cell stores a compressed signature
        self.memory_keys = nn.Parameter(
            torch.randn(n_memory, slot_dim, device=device) * 0.1
        )

        # Encoder: (slot_before, action, slot_after) → memory write vector
        self.write_enc = nn.Sequential(
            nn.Linear(slot_dim * 2 + action_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, slot_dim),
        )

        # Decoder: memory cell → spatial concept logits
        # Output: [close, far, left, right, ahead, behind, above, below,
        #          blocking, clear, novel, familiar, moving, static, dangerous, safe]
        self.SPATIAL_DIMS = 16
        self.spatial_dec = nn.Sequential(
            nn.Linear(slot_dim + body_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, self.SPATIAL_DIMS),
        )

        # Spatial concept names (learned meanings, not hardcoded positions)
        # These are OUTPUT NEURONS — their MEANING is learned, not assigned
        self.spatial_concept_names = [
            "spatial_0", "spatial_1", "spatial_2", "spatial_3",
            "spatial_4", "spatial_5", "spatial_6", "spatial_7",
            "spatial_8", "spatial_9", "spatial_10", "spatial_11",
            "spatial_12", "spatial_13", "spatial_14", "spatial_15",
        ]
        # Mapping from spatial neuron → SemanticConceptStore name
        # Initialized randomly, learned through co-occurrence
        self._spatial_to_concept: dict[int, str] = {}

        self._init_weights()
        self.to(self.device)

        # Memory state: (slot_dim,) per cell, usage count
        self._memory_values = torch.zeros(n_memory, slot_dim, device=device)
        self._memory_usage = torch.zeros(n_memory, device=device)
        self._memory_body = torch.zeros(n_memory, body_dim, device=device)
        self._memory_action = torch.zeros(n_memory, action_dim, device=device)

        self.write_count = 0
        self.read_count = 0
        self.train_steps = 0
        self._loss_history: deque[float] = deque(maxlen=100)
        self.optim = torch.optim.Adam(self.parameters(), lr=1e-3)

        # Decay for memory
        self._decay = _ef("RKK_SPATIAL_MEM_DECAY", 0.95)

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.4)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _encode_action(self, action_var: str | None, action_val: float | None) -> torch.Tensor:
        """Encode (var, val) as a learned feature vector."""
        a = torch.zeros(self.action_dim, device=self.device)
        if action_var is not None and action_val is not None:
            # Use hash of var name for reproducible encoding
            var_hash = hash(action_var) % (self.action_dim // 2)
            a[var_hash] = float(action_val)
            a[self.action_dim // 2 + var_hash % (self.action_dim // 2)] = 1.0
        return a

    def _encode_body(self, obs: dict[str, float]) -> torch.Tensor:
        """Encode key body state variables."""
        keys = [
            "posture_stability", "com_z", "com_x",
            "foot_contact_l", "foot_contact_r", "support_bias",
            "gait_phase_l", "gait_phase_r",
            "intent_stride", "intent_torso_forward",
        ]
        vals = []
        for k in keys:
            v = obs.get(k, obs.get(f"phys_{k}", 0.5))
            vals.append(float(v) - 0.5)  # center around 0
        # Pad or truncate to body_dim
        while len(vals) < self.body_dim:
            vals.append(0.0)
        return torch.tensor(vals[:self.body_dim], dtype=torch.float32, device=self.device)

    @torch.no_grad()
    def write(
        self,
        slot_before: torch.Tensor,
        slot_after: torch.Tensor,
        action_var: str | None,
        action_val: float | None,
        body_after: dict[str, float],
    ) -> int:
        """
        Write an interventional experience to memory.
        Returns memory cell index written to.
        """
        sb = slot_before.to(self.device).view(1, -1)
        sa = slot_after.to(self.device).view(1, -1)

        # Pad if needed
        def _pad(t: torch.Tensor, d: int) -> torch.Tensor:
            if t.shape[-1] < d:
                return F.pad(t, (0, d - t.shape[-1]))
            return t[:, :d]

        sb = _pad(sb, self.slot_dim)
        sa = _pad(sa, self.slot_dim)

        a = self._encode_action(action_var, action_val).unsqueeze(0)
        b = self._encode_body(body_after)

        # Write vector
        write_in = torch.cat([sb, sa, a], dim=-1)
        write_vec = self.write_enc(write_in).squeeze(0)  # (slot_dim,)

        # Find least used memory cell (LRU-style)
        least_used = int(self._memory_usage.argmin().item())

        # Decay existing usage
        self._memory_usage *= self._decay

        # Write
        alpha = 0.3  # soft write
        self._memory_values[least_used] = (
            (1 - alpha) * self._memory_values[least_used] + alpha * write_vec.detach()
        )
        self._memory_body[least_used] = b.detach()
        self._memory_action[least_used] = a.squeeze(0).detach()
        self._memory_usage[least_used] = 1.0

        self.write_count += 1
        return least_used

    @torch.no_grad()
    def read(
        self,
        slot_vec: torch.Tensor,
        body_obs: dict[str, float],
        top_k: int = 3,
    ) -> list[tuple[str, float]]:
        """
        Read spatial concepts from memory given current slot.
        Returns [(spatial_concept_name, activation), ...].
        No rules. Pure associative memory.
        """
        self.eval()
        sv = slot_vec.to(self.device).view(1, -1)
        if sv.shape[-1] < self.slot_dim:
            sv = F.pad(sv, (0, self.slot_dim - sv.shape[-1]))
        sv = sv[:, :self.slot_dim]

        # Attention over memory keys
        keys_norm = F.normalize(self.memory_keys, dim=-1)
        sv_norm = F.normalize(sv, dim=-1)
        attn = (sv_norm @ keys_norm.T).squeeze(0)  # (n_memory,)

        # Weight by usage (don't read from empty cells)
        usage_mask = (self._memory_usage > 0.05).float()
        attn = attn * usage_mask

        if attn.sum() < 1e-8:
            return []  # Memory is empty — no spatial concepts yet

        # Soft attention read
        weights = F.softmax(attn * 5.0, dim=-1)  # sharpen
        read_vec = (weights.unsqueeze(-1) * self._memory_values).sum(0)  # (slot_dim,)
        body_vec = (weights.unsqueeze(-1) * self._memory_body).sum(0)     # (body_dim,)

        # Decode spatial concepts
        dec_in = torch.cat([read_vec, body_vec], dim=-1).unsqueeze(0)
        spatial_logits = self.spatial_dec(dec_in).squeeze(0)  # (SPATIAL_DIMS,)
        spatial_probs = torch.sigmoid(spatial_logits)

        # Return top-k active spatial neurons with their concept names
        results: list[tuple[str, float]] = []
        top_vals, top_idx = spatial_probs.topk(top_k)
        for val, idx in zip(top_vals.tolist(), top_idx.tolist()):
            if val > 0.5:
                # Map spatial neuron to semantic concept name (learned)
                concept_name = self._spatial_to_concept.get(
                    idx,
                    f"SPATIAL_{idx}"  # unnamed until learned
                )
                results.append((concept_name, float(val)))

        self.read_count += 1
        return results

    def assign_spatial_concept(self, neuron_idx: int, concept_name: str) -> None:
        """
        Called by LLM teacher when it identifies what a spatial neuron means.
        This is the ONLY way spatial neurons get names — through distillation.
        """
        self._spatial_to_concept[neuron_idx] = concept_name

    def train_step(
        self,
        slot_vec: torch.Tensor,
        body_obs: dict[str, float],
        target_spatial: list[tuple[int, float]],
    ) -> float | None:
        """
        Supervised step: spatial neuron i should activate at value v.
        Targets come from LLM teacher annotation.
        """
        if not target_spatial:
            return None

        self.train()
        sv = slot_vec.to(self.device).view(1, -1)
        if sv.shape[-1] < self.slot_dim:
            sv = F.pad(sv, (0, self.slot_dim - sv.shape[-1]))
        sv = sv[:, :self.slot_dim]

        body_vec = self._encode_body(body_obs)
        read_vec = sv.squeeze(0)

        dec_in = torch.cat([read_vec, body_vec], dim=-1).unsqueeze(0)
        logits = self.spatial_dec(dec_in).squeeze(0)

        target = torch.zeros(self.SPATIAL_DIMS, device=self.device)
        for idx, val in target_spatial:
            if 0 <= idx < self.SPATIAL_DIMS:
                target[idx] = float(val)

        loss = F.binary_cross_entropy_with_logits(logits, target)
        self.optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), 0.5)
        self.optim.step()

        v = float(loss.item())
        self._loss_history.append(v)
        self.train_steps += 1
        return v

    def snapshot(self) -> dict[str, Any]:
        active_cells = int((self._memory_usage > 0.1).sum().item())
        named_neurons = len(self._spatial_to_concept)
        return {
            "write_count": self.write_count,
            "read_count": self.read_count,
            "train_steps": self.train_steps,
            "active_memory_cells": active_cells,
            "named_spatial_neurons": named_neurons,
            "spatial_concepts": dict(self._spatial_to_concept),
            "mean_loss": round(
                float(np.mean(list(self._loss_history))), 5
            ) if self._loss_history else 0.0,
        }


# ─── NeuralLanguageGrounding ──────────────────────────────────────────────────
class NeuralLanguageGrounding:
    """
    Единая точка интеграции: зрение + тело + язык без хардкода.
    
    Заменяет:
      - VerbalActionController с TEMPLATES_RU/EN
      - SlotLabeler с SLOT_PROPERTY_TO_CONCEPTS  
      - VisualInnerVoice с VISUAL_TEMPLATES_RU
      - position_to_spatial_concepts
      - text_to_concepts keyword matching
      
    Всё что генерируется — выучено из experience, дистиллировано от LLM.
    
    Цикл обучения:
      1. Агент действует (do(var, val))
      2. Slot vectors до/после записываются в InterventionalSpatialMemory
      3. InnerVoiceNet генерирует thought_embedding
      4. CausalSpeechDecoder генерирует текст из thought
      5. Каждые τ3 тиков: LLM teacher получает (float_state, generated_text)
         → корректирует через дистилляцию
      6. NeuralConceptProjector обновляется через contrastive от LLM labels
    """

    def __init__(
        self,
        concept_dim: int = 64,
        slot_dim: int = 64,
        device: torch.device | None = None,
    ):
        self.device = device or torch.device("cpu")
        self.concept_dim = concept_dim
        self.slot_dim = slot_dim

        vocab_size = _ei("RKK_NEURAL_LANG_VOCAB", VOCAB_SIZE)
        hidden = _ei("RKK_NEURAL_LANG_HIDDEN", 128)
        max_len = _ei("RKK_NEURAL_LANG_MAX_LEN", 32)
        n_memory = _ei("RKK_SPATIAL_MEM_SLOTS", 32)

        self.speech_decoder = CausalSpeechDecoder(
            concept_dim=concept_dim,
            hidden=hidden,
            vocab_size=vocab_size,
            max_len=max_len,
            device=device,
        )

        self.concept_projector = NeuralConceptProjector(
            slot_dim=slot_dim,
            concept_dim=concept_dim,
            n_concepts=vocab_size,
            hidden=hidden,
            device=device,
        )

        self.spatial_memory = InterventionalSpatialMemory(
            slot_dim=slot_dim,
            body_dim=16,
            action_dim=16,
            n_memory=n_memory,
            hidden=hidden // 2,
            device=device,
        )

        # Last generated text (for UI)
        self._last_text: str = ""
        self._last_tick: int = -999
        self._speak_every = _ei("RKK_SPEECH_OBSERVE_EVERY", 150)

        # Distillation buffer: (thought_emb, target_text, state_vec)
        self._distill_buf: deque[tuple[list[float], str, list[float]]] = deque(maxlen=128)

        self.total_utterances = 0
        self.total_distill_steps = 0

    def on_agent_step(
        self,
        tick: int,
        slot_vec_before: torch.Tensor | None,
        slot_vec_after: torch.Tensor | None,
        action_var: str | None,
        action_val: float | None,
        body_obs: dict[str, float],
    ) -> None:
        """
        Call after every agent action to update spatial memory.
        This is the core causal learning loop.
        """
        if not neural_lang_enabled():
            return
        if slot_vec_before is None or slot_vec_after is None:
            return
        self.spatial_memory.write(
            slot_before=slot_vec_before,
            slot_after=slot_vec_after,
            action_var=action_var,
            action_val=action_val,
            body_after=body_obs,
        )

    def generate_utterance(
        self,
        thought_emb: torch.Tensor,
        slot_vec: torch.Tensor | None,
        body_obs: dict[str, float],
        state_vec: torch.Tensor | None = None,
        tick: int = 0,
    ) -> str:
        """
        Generate speech from current internal state.
        No templates. Neural generation only.
        """
        if not neural_lang_enabled():
            return ""
        if (tick - self._last_tick) < self._speak_every:
            return self._last_text

        self._last_tick = tick

        # Get spatial context from memory
        spatial_ctx: list[tuple[str, float]] = []
        if slot_vec is not None:
            spatial_ctx = self.spatial_memory.read(slot_vec, body_obs)

        # Generate text
        text = self.speech_decoder.generate(
            thought_emb=thought_emb,
            state_vec=state_vec,
        )

        # Append spatial context if spatial neurons have learned names
        if spatial_ctx:
            named = [(n, s) for n, s in spatial_ctx if not n.startswith("SPATIAL_")]
            if named:
                top_name = named[0][0].lower().replace("_", " ")
                if text and top_name:
                    text = text + " " + top_name

        self._last_text = text
        self.total_utterances += 1
        return text

    def push_distill_sample(
        self,
        thought_emb: torch.Tensor,
        llm_text: str,
        state_vec: list[float],
    ) -> None:
        """
        Store LLM-generated annotation for offline distillation.
        This is the TEACHER SIGNAL — the only supervised input.
        """
        self._distill_buf.append((
            thought_emb.detach().cpu().tolist(),
            llm_text,
            state_vec,
        ))

    def distill_step(self, batch_size: int = 8) -> float | None:
        """
        Batch distillation from LLM teacher annotations.
        """
        if len(self._distill_buf) < 2:
            return None

        batch = list(self._distill_buf)[-batch_size:]
        total_loss = 0.0
        n_valid = 0

        for thought_list, text, state_list in batch:
            thought = torch.tensor(thought_list, dtype=torch.float32, device=self.device)
            state = torch.tensor(state_list, dtype=torch.float32, device=self.device)
            loss = self.speech_decoder.distill_step(thought, text, state)
            if loss is not None:
                total_loss += loss
                n_valid += 1

        self.total_distill_steps += 1
        return total_loss / max(n_valid, 1) if n_valid > 0 else None

    def on_llm_spatial_annotation(
        self,
        slot_vec: torch.Tensor,
        body_obs: dict[str, float],
        neuron_concept_pairs: list[tuple[int, str, float]],
    ) -> None:
        """
        LLM teacher tells us: "spatial neuron 3 = OBJECT_LEFT with 0.8 confidence".
        This is how spatial neurons learn their semantic meaning.
        """
        for neuron_idx, concept_name, confidence in neuron_concept_pairs:
            if confidence > 0.5:
                self.spatial_memory.assign_spatial_concept(neuron_idx, concept_name)
                # Also train spatial memory on this example
                self.spatial_memory.train_step(
                    slot_vec=slot_vec,
                    body_obs=body_obs,
                    target_spatial=[(neuron_idx, confidence)],
                )

    def snapshot(self) -> dict[str, Any]:
        return {
            "enabled": neural_lang_enabled(),
            "total_utterances": self.total_utterances,
            "total_distill_steps": self.total_distill_steps,
            "last_text": self._last_text[:120],
            "distill_buf_len": len(self._distill_buf),
            "speech_decoder": self.speech_decoder.snapshot(),
            "concept_projector": self.concept_projector.snapshot(),
            "spatial_memory": self.spatial_memory.snapshot(),
        }
