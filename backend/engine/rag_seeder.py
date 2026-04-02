"""
rag_seeder.py — RAG Pipeline для генерации каузальных seeds.

Архитектура:
  1. Wikipedia Search → получаем релевантный текст по теме среды
  2. Causal Extraction → паттерны "X affects Y", "X causes Y", "X increases Y"
  3. Variable Mapping → маппим слова на переменные агента
  4. inject_seeds() → загружаем с alpha=0.05

Опциональная LLM интеграция (фаза 11+):
  Если задан LLM_URL (OpenAI-совместимый API, напр. Ollama),
  используем его вместо regex для извлечения.

Поддерживаемые backends:
  - "wikipedia" : Wikipedia REST API (бесплатно, без LLM)
  - "ollama"    : локальная LLM через Ollama (qwen2.5:3b и т.д.)
  - "openai"    : OpenAI-compatible API
"""
from __future__ import annotations

import re
import json
import asyncio
import httpx
import numpy as np
from dataclasses import dataclass, field
from typing import Literal


# ─── Causal edge из RAG ───────────────────────────────────────────────────────
@dataclass
class CausalHypothesis:
    from_:  str
    to:     str
    weight: float = 0.3
    alpha:  float = 0.05
    source: str   = "rag"      # "wikipedia", "llm", "manual"
    confidence: float = 0.3    # насколько уверен в гипотезе

    def to_dict(self) -> dict:
        return {
            "from_":  self.from_,
            "to":     self.to,
            "weight": round(self.weight, 3),
            "alpha":  self.alpha,
        }


# ─── Словари синонимов для маппинга слов → переменные ────────────────────────
ENV_VARIABLE_SYNONYMS = {
    # Physics env
    "Temp": ["temperature", "temp", "heat", "thermal", "hot", "cold", "warm"],
    "Pressure": ["pressure", "stress", "force", "compression", "pascal"],
    "Volume": ["volume", "space", "expansion", "size", "capacity"],
    "Energy": ["energy", "power", "work", "kinetic", "potential", "joule"],
    "StateChange": ["state", "phase", "transition", "melting", "boiling", "solid", "liquid", "gas"],
    "Entropy": ["entropy", "disorder", "chaos", "randomness"],

    # Chemistry env
    "Reactant_A": ["reactant", "substrate", "reagent", "molecule", "compound"],
    "Reactant_B": ["reagent", "reactant_b", "second reactant"],
    "Catalyst": ["catalyst", "enzyme", "accelerator", "activator"],
    "Rate": ["rate", "speed", "velocity", "kinetics", "frequency"],
    "Product": ["product", "yield", "output", "result", "synthesis"],

    # Logic env
    "Input": ["input", "signal", "trigger", "stimulus"],
    "Condition": ["condition", "state", "flag", "control"],
    "Branch_A": ["branch", "path", "route", "option_a"],
    "Branch_B": ["alternative", "option_b", "else"],
    "Output": ["output", "result", "response", "return"],
    "Error": ["error", "exception", "fault", "failure"],
}

# Паттерны каузальных отношений в тексте
CAUSAL_PATTERNS = [
    # "X affects Y", "X increases Y"
    (r'(\w[\w\s]+?)\s+(?:affects?|influences?|impacts?)\s+(\w[\w\s]+)', 0.6),
    (r'(\w[\w\s]+?)\s+(?:increases?|raises?|boosts?)\s+(\w[\w\s]+)', 0.7),
    (r'(\w[\w\s]+?)\s+(?:decreases?|reduces?|lowers?)\s+(\w[\w\s]+)', -0.6),
    # "X causes Y"
    (r'(\w[\w\s]+?)\s+(?:causes?|leads? to|results? in)\s+(\w[\w\s]+)', 0.7),
    # "when X, Y"
    (r'when\s+(\w[\w\s]+?)\s+(?:is|are)?\s+(?:high|increased|raised),\s+(\w[\w\s]+)',0.6),
    (r'when\s+(\w[\w\s]+?)\s+(?:is|are)?\s+(?:low|decreased|reduced),\s+(\w[\w\s]+)',-0.5),
    # "X is proportional to Y"
    (r'(\w[\w\s]+?)\s+(?:is|are)\s+(?:directly)?\s+proportional to\s+(\w[\w\s]+)', 0.7),
    (r'(\w[\w\s]+?)\s+(?:is|are)\s+inversely proportional to\s+(\w[\w\s]+)', -0.6),
    # "higher X → higher Y"
    (r'higher\s+(\w[\w\s]+?)\s+(?:means?|leads? to|results? in)\s+(?:higher|more)\s+(\w[\w\s]+)', 0.6),
    (r'higher\s+(\w[\w\s]+?)\s+(?:means?|leads? to|results? in)\s+(?:lower|less)\s+(\w[\w\s]+)', -0.5),
]

# Слова-якоря для определения темы среды из запроса
ENV_KEYWORDS = {
    "physics":   ["temperature", "pressure", "volume", "gas", "thermodynamics", "entropy", "energy", "heat"],
    "chemistry": ["reaction", "catalyst", "reactant", "chemical", "synthesis", "yield", "rate", "enzyme"],
    "logic":     ["logic", "boolean", "condition", "branch", "algorithm", "input", "output", "state"],
}


# ─── Wikipedia fetcher ────────────────────────────────────────────────────────
async def fetch_wikipedia(topic: str, sentences: int = 5) -> str:
    """Получаем краткое описание из Wikipedia REST API."""
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{topic.replace(' ', '_')}"
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            resp = await client.get(url, headers={"User-Agent": "RKK-RAG/1.0"})
            if resp.status_code == 200:
                data = resp.json()
                return data.get("extract", "")[:2000]
        except Exception:
            pass

    # Fallback: Wikipedia search API
    search_url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "list":   "search",
        "srsearch": topic,
        "format": "json",
        "srlimit": 3,
    }
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            resp = await client.get(search_url, params=params)
            if resp.status_code == 200:
                results = resp.json().get("query", {}).get("search", [])
                if results:
                    # Берём первый результат
                    page_title = results[0]["title"]
                    return await fetch_wikipedia(page_title, sentences)
        except Exception:
            pass
    return ""


# ─── Causal extractor (regex) ─────────────────────────────────────────────────
def extract_causal_pairs(text: str, target_vars: list[str]) -> list[tuple[str, str, float]]:
    """
    Извлекаем каузальные пары из текста через regex паттерны.
    Возвращает [(word_from, word_to, weight), ...]
    """
    text_lower = text.lower()
    pairs = []

    for pattern, weight in CAUSAL_PATTERNS:
        matches = re.findall(pattern, text_lower, re.IGNORECASE)
        for match in matches:
            from_word = match[0].strip()[:30]
            to_word   = match[1].strip()[:30]
            if len(from_word) > 2 and len(to_word) > 2:
                pairs.append((from_word, to_word, weight))

    return pairs


def map_to_variables(
    word: str,
    available_vars: list[str],
    threshold: float = 0.3,
) -> str | None:
    """Маппим слово на переменную агента через словарь синонимов."""
    word_lower = word.lower()

    # Прямое совпадение с именем переменной
    for var in available_vars:
        if var.lower() in word_lower or word_lower in var.lower():
            return var

    # Поиск через синонимы
    best_var   = None
    best_score = 0.0

    for var, synonyms in ENV_VARIABLE_SYNONYMS.items():
        if var not in available_vars:
            continue
        for syn in synonyms:
            if syn in word_lower or word_lower in syn:
                # Простая метрика: длина пересечения
                score = len(set(syn) & set(word_lower)) / max(len(syn), len(word_lower))
                if score > best_score and score > threshold:
                    best_score = score
                    best_var   = var

    return best_var


# ─── LLM extractor (опционально) ─────────────────────────────────────────────
async def extract_via_llm(
    text:     str,
    var_names: list[str],
    llm_url:  str,
    model:    str = "qwen2.5:3b",
) -> list[tuple[str, str, float]]:
    """
    Используем LLM (Ollama/OpenAI-compatible) для извлечения каузальных пар.
    Возвращает [(from_var, to_var, weight), ...]
    """
    prompt = f"""You are a causal reasoning expert. Given this text and a list of variables, 
extract causal relationships between the variables.

Variables: {var_names}

Text: {text[:800]}

Output ONLY a JSON array of causal edges. Example:
[{{"from_": "Temp", "to": "Pressure", "weight": 0.8}},
 {{"from_": "Pressure", "to": "Volume", "weight": -0.6}}]

Rules:
- Only use variables from the list above
- weight: positive = increases, negative = decreases, range [-1, 1]
- Include only edges you are confident about
- Output ONLY the JSON array, no other text"""

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.1, "num_predict": 300},
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            resp = await client.post(llm_url, json=payload)
            if resp.status_code == 200:
                raw = resp.json().get("response", "")
                # Извлекаем JSON из ответа
                json_match = re.search(r'\[.*?\]', raw, re.DOTALL)
                if json_match:
                    edges = json.loads(json_match.group())
                    pairs = []
                    for e in edges:
                        from_ = e.get("from_") or e.get("from")
                        to    = e.get("to")
                        w     = float(e.get("weight", 0.3))
                        if from_ in var_names and to in var_names:
                            pairs.append((from_, to, w))
                    return pairs
        except Exception as e:
            print(f"[RAG] LLM extraction failed: {e}")

    return []


# ─── Главный класс ────────────────────────────────────────────────────────────
class RAGSeeder:
    """
    Автоматическая генерация text priors из Wikipedia + опционального LLM.

    Использование:
      seeder = RAGSeeder()
      hypotheses = await seeder.generate(
          env_preset="physics",
          available_vars=["Temp","Pressure","Volume","Energy","StateChange","Entropy"],
      )
    """

    def __init__(
        self,
        llm_url:  str | None = None,   # None = только regex (без LLM)
        llm_model: str = "qwen2.5:3b",
        backend: Literal["wikipedia", "ollama", "openai"] = "wikipedia",
    ):
        self.llm_url   = llm_url
        self.llm_model = llm_model
        self.backend   = backend

        # Темы поиска для каждой среды
        self.env_topics = {
            "physics":   ["ideal gas law", "thermodynamics laws", "Boyle's law",
                          "Charles's law", "heat transfer", "entropy thermodynamics"],
            "chemistry": ["chemical kinetics", "reaction rate", "catalysis chemistry",
                          "Arrhenius equation", "activation energy"],
            "logic":     ["boolean logic", "conditional branching", "logic gates",
                          "control flow programming"],
        }

    async def generate(
        self,
        env_preset:      str,
        available_vars:  list[str],
        max_hypotheses:  int = 8,
    ) -> list[CausalHypothesis]:
        """
        Генерируем каузальные гипотезы для среды.
        Возвращает список CausalHypothesis готовых к inject_seeds().
        """
        topics = self.env_topics.get(env_preset, [env_preset])
        all_text = ""

        # 1. Собираем тексты из Wikipedia
        for topic in topics[:3]:   # берём первые 3 темы
            text = await fetch_wikipedia(topic)
            if text:
                all_text += f"\n{text}"
                await asyncio.sleep(0.1)   # rate limit

        if not all_text:
            print(f"[RAG] No Wikipedia text found for {env_preset}")
            return []

        # 2. Извлекаем каузальные пары
        if self.llm_url:
            # LLM экстракция (точнее)
            raw_pairs = await extract_via_llm(
                all_text, available_vars, self.llm_url, self.llm_model
            )
            source = "llm"
        else:
            # Regex экстракция (быстрее, без LLM)
            raw_pairs_text = extract_causal_pairs(all_text, available_vars)
            # Маппим слова → переменные
            raw_pairs: list[tuple[str, str, float]] = []
            for from_word, to_word, weight in raw_pairs_text:
                from_var = map_to_variables(from_word, available_vars)
                to_var   = map_to_variables(to_word,   available_vars)
                if from_var and to_var and from_var != to_var:
                    raw_pairs.append((from_var, to_var, weight))
            source = "wikipedia"

        # 3. Дедупликация и нормализация весов
        seen   = set()
        hypotheses = []

        for from_, to, weight in raw_pairs:
            key = (from_, to)
            if key in seen:
                continue
            seen.add(key)

            # Нормализуем вес к [-0.9, 0.9]
            w = float(np.clip(weight, -0.9, 0.9))

            # Оцениваем уверенность по частоте упоминания
            count = sum(1 for f,t,_ in raw_pairs if f == from_ and t == to)
            confidence = min(0.9, 0.3 + count * 0.15)

            hypotheses.append(CausalHypothesis(
                from_=from_, to=to,
                weight=w * 0.35,   # уменьшаем вес — seeds должны быть слабыми
                alpha=0.05,
                source=source,
                confidence=confidence,
            ))

        # Сортируем по уверенности
        hypotheses.sort(key=lambda h: -h.confidence)

        result = hypotheses[:max_hypotheses]
        print(f"[RAG] {env_preset}: generated {len(result)} hypotheses "
              f"from {len(all_text)} chars text")
        return result

    async def generate_all_agents(
        self,
        agents_config: list[dict],  # [{"preset": "physics", "vars": [...]}]
    ) -> dict[int, list[CausalHypothesis]]:
        """Генерируем seeds для всех агентов параллельно."""
        tasks = [
            self.generate(
                env_preset=cfg["preset"],
                available_vars=cfg["vars"],
            )
            for cfg in agents_config
        ]
        results = await asyncio.gather(*tasks)
        return {i: hyps for i, hyps in enumerate(results)}


# ─── Жёстко заданные seeds (fallback без интернета) ──────────────────────────
HARDCODED_SEEDS = {
    "physics": [
        {"from_": "Temp",     "to": "Pressure",    "weight": 0.25, "alpha": 0.05},
        {"from_": "Temp",     "to": "Energy",      "weight": 0.22, "alpha": 0.05},
        {"from_": "Pressure", "to": "Volume",      "weight":-0.20, "alpha": 0.05},
        {"from_": "Energy",   "to": "Entropy",     "weight": 0.18, "alpha": 0.05},
    ],
    "chemistry": [
        {"from_": "Catalyst",   "to": "Rate",    "weight": 0.28, "alpha": 0.05},
        {"from_": "Temp",       "to": "Rate",    "weight": 0.22, "alpha": 0.05},
        {"from_": "Rate",       "to": "Product", "weight": 0.30, "alpha": 0.05},
        {"from_": "Reactant_A", "to": "Rate",    "weight": 0.20, "alpha": 0.05},
    ],
    "logic": [
        {"from_": "Input",     "to": "Condition", "weight": 0.28, "alpha": 0.05},
        {"from_": "Condition", "to": "Branch_A",  "weight": 0.30, "alpha": 0.05},
        {"from_": "Branch_A",  "to": "Output",    "weight": 0.25, "alpha": 0.05},
        {"from_": "Condition", "to": "Branch_B",  "weight":-0.28, "alpha": 0.05},
    ],
    "pybullet": [
        {"from_": "obj0_vx", "to": "obj0_x", "weight": 0.25, "alpha": 0.05},
        {"from_": "obj0_vy", "to": "obj0_y", "weight": 0.25, "alpha": 0.05},
        {"from_": "obj1_vx", "to": "obj1_x", "weight": 0.25, "alpha": 0.05},
        {"from_": "obj1_vy", "to": "obj1_y", "weight": 0.25, "alpha": 0.05},
        {"from_": "obj2_vx", "to": "obj2_x", "weight": 0.25, "alpha": 0.05},
        {"from_": "obj2_vy", "to": "obj2_y", "weight": 0.25, "alpha": 0.05},
    ],
}
