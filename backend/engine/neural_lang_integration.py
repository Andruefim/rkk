"""
neural_lang_integration.py — Патч интеграции NeuralLanguageGrounding в Simulation.

Что заменяется:
  verbal_action.py длинные TEMPLATES_* → CausalSpeechDecoder (нейросеть) + короткий fallback
  slot_labeler.py text_to_concepts / position_to_spatial_concepts → projector + spatial memory
  visual_inner_voice.py VISUAL_TEMPLATES_* → тот же нейросетевой путь (get_template не используется)

Включается из `Simulation.__init__` при RKK_NEURAL_LANG=1 (по умолчанию), если доступен
`engine.neural_causal_language`.
"""
from __future__ import annotations

import os
import asyncio
import threading
from typing import Any

import numpy as np
import torch


def _neural_lang_available() -> bool:
    try:
        from engine.neural_causal_language import NeuralLanguageGrounding  # noqa: F401
        return True
    except ImportError:
        return False


def apply_neural_lang_patch(sim) -> bool:
    """
    Патч существующего экземпляра Simulation:
      1. Инициализирует NeuralLanguageGrounding
      2. Перехватывает _run_single_agent_timestep_inner для on_agent_step
      3. Перехватывает verbal tick для generate_utterance
      4. Подключает дистилляцию к LLM teacher callback
    
    Возвращает True если патч успешно применён.
    """
    if not _neural_lang_available():
        print("[NeuralLang] neural_causal_language.py not found")
        return False

    from engine.neural_causal_language import NeuralLanguageGrounding

    # Инициализация
    device = getattr(sim, "device", torch.device("cpu"))
    nlg = NeuralLanguageGrounding(
        concept_dim=64,
        slot_dim=64,
        device=device,
    )
    sim._neural_lang = nlg
    print(f"[NeuralLang] NeuralLanguageGrounding initialized on {device}")

    # Подключаем к LLM teacher callback
    _patch_llm_teacher(sim, nlg)

    # Подключаем к verbal action
    _patch_verbal_action(sim, nlg)

    # Подключаем к slot_labeler (заменяем text_to_concepts)
    _patch_slot_labeler(sim, nlg)

    # Добавляем перехват on_agent_step (через monkey-patch метода тика)
    _patch_tick_step(sim, nlg)

    return True


def _patch_llm_teacher(sim, nlg) -> None:
    """
    Перехватываем LLM teacher callback:
    annotation.verbal → distill_step для CausalSpeechDecoder.
    """
    original_callback = getattr(sim, "_on_teacher_annotation", None)
    if original_callback is None:
        return

    def patched_on_teacher_annotation(ann) -> None:
        # Оригинальный callback
        original_callback(ann)

        # Дистилляция verbal текста в CausalSpeechDecoder
        if ann.verbal and not ann.error:
            iv = getattr(sim, "_inner_voice", None)
            if iv is None:
                return
            thought = iv.get_thought_embedding()
            if not thought:
                return
            thought_t = torch.tensor(thought, dtype=torch.float32)

            # State vector из графа
            graph = sim.agent.graph
            node_ids = list(graph._node_ids)
            state_vec = [float(graph.nodes.get(n, 0.5)) for n in node_ids]

            nlg.push_distill_sample(thought_t, ann.verbal, state_vec)

            # Обучаем если накопилось достаточно
            if len(nlg._distill_buf) >= 4 and sim.tick % 4 == 0:
                loss = nlg.distill_step(batch_size=4)
                if loss is not None and sim.tick % 100 == 0:
                    print(f"[NeuralLang] Distill loss: {loss:.4f} tick={sim.tick}")

        # Spatial annotation: если LLM упоминает пространственные концепты,
        # учим spatial memory их значению
        if ann.primary_concepts:
            _extract_spatial_from_annotation(sim, nlg, ann)

    sim._on_teacher_annotation = patched_on_teacher_annotation


def _extract_spatial_from_annotation(sim, nlg, ann) -> None:
    """
    Из аннотации LLM teacher извлекаем пространственные пары (neuron, concept, conf).
    Это обучает InterventionalSpatialMemory знать значение своих нейронов.
    """
    SPATIAL_CONCEPTS = {
        "OBJECT_LEFT", "OBJECT_RIGHT", "OBJECT_AHEAD", "OBJECT_VERY_CLOSE",
        "OBJECT_FAR", "CLEAR_PATH_AHEAD", "OBJECT_BLOCKING_PATH",
        "WALL_NEARBY", "OPEN_SCENE", "CLUTTERED_SCENE",
    }
    spatial_found = [c for c in ann.primary_concepts if c in SPATIAL_CONCEPTS]
    if not spatial_found:
        return

    vis_env = getattr(sim, "_visual_env", None)
    if vis_env is None:
        return
    slot_vecs = getattr(vis_env, "_last_slot_vecs", None)
    if slot_vecs is None:
        return

    obs = {}
    try:
        obs = dict(sim.agent.env.observe())
    except Exception:
        return

    # Назначаем: первый spatial концепт → нейрон 0, второй → нейрон 1, ...
    # Это упрощение — в идеале нужен attention matching, но начнём с этого
    pairs = [
        (i % nlg.spatial_memory.SPATIAL_DIMS, c, 0.7)
        for i, c in enumerate(spatial_found)
    ]
    # Усредняем slot vectors
    mean_slot = slot_vecs.mean(dim=0)
    nlg.on_llm_spatial_annotation(mean_slot, obs, pairs)


def _patch_verbal_action(sim, nlg) -> None:
    """
    Перехватываем verbal generation:
    Вместо TEMPLATES_RU → nlg.generate_utterance()
    
    Патчим метод _tick_verbal в VerbalActionController.
    """
    verbal = getattr(sim, "_verbal", None)
    if verbal is None:
        return

    original_decode_observe = getattr(verbal.decoder, "decode_observe", None)
    if original_decode_observe is None:
        return

    def neural_decode_observe(concepts, obs, curiosity):
        # Пробуем нейросетевую генерацию
        iv = getattr(sim, "_inner_voice", None)
        if iv is None or not iv.net._last_thought.any():
            # Fallback к оригиналу если InnerVoice ещё не обучен
            return original_decode_observe(concepts, obs, curiosity)

        thought = iv.net._last_thought.squeeze(0).detach()

        # Получаем slot vector если доступен
        vis_env = getattr(sim, "_visual_env", None)
        slot_vec = None
        if vis_env is not None:
            sv = getattr(vis_env, "_last_slot_vecs", None)
            if sv is not None:
                slot_vec = sv.mean(dim=0)

        graph = sim.agent.graph
        node_ids = list(graph._node_ids)
        state_vec = torch.tensor(
            [float(graph.nodes.get(n, 0.5)) for n in node_ids],
            dtype=torch.float32,
        )

        text = nlg.generate_utterance(
            thought_emb=thought,
            slot_vec=slot_vec,
            body_obs=obs,
            state_vec=state_vec,
            tick=sim.tick,
        )

        # Если декодер ещё не обучен (пустой вывод) — используем оригинал
        if not text or len(text.strip()) < 3:
            return original_decode_observe(concepts, obs, curiosity)

        return text

    verbal.decoder.decode_observe = neural_decode_observe
    print("[NeuralLang] Patched VerbalActionController.decode_observe")


def _patch_slot_labeler(sim, nlg) -> None:
    """
    Перехватываем SlotLabeler.process_slots:
    Вместо keyword matching → NeuralConceptProjector.project()
    
    Concept names теперь берутся из SemanticConceptStore (через project()),
    а не из SLOT_PROPERTY_TO_CONCEPTS.
    """
    slot_labeler = getattr(sim, "_slot_labeler", None)
    if slot_labeler is None:
        return

    original_process = slot_labeler.process_slots

    def neural_process_slots(
        vlm_labels,
        slot_positions,
        slot_vectors,
        slot_confidences=None,
        tick=0,
    ):
        # Сначала запускаем оригинальный pipeline (структурные проверки)
        original_result = original_process(
            vlm_labels, slot_positions, slot_vectors,
            slot_confidences, tick
        )

        # Дополняем нейронными проекциями если есть slot vectors
        if not slot_vectors:
            return original_result

        iv = getattr(sim, "_inner_voice", None)
        if iv is None:
            return original_result

        neural_additions: dict[str, float] = {}
        concept_store = iv.concept_store

        for slot_id, vec_list in slot_vectors.items():
            if not vec_list:
                continue
            try:
                slot_k = int(slot_id.split("_")[-1])
            except (ValueError, IndexError):
                continue

            vec = torch.tensor(vec_list, dtype=torch.float32, device=nlg.device)
            projected = nlg.concept_projector.project(vec, top_k=3, threshold=0.4)

            for concept_idx, score in projected:
                concept_name = concept_store.idx_to_name.get(concept_idx)
                if concept_name and not concept_name.startswith("LATENT_"):
                    neural_additions[concept_name] = max(
                        neural_additions.get(concept_name, 0.0), score
                    )
                    # Record co-occurrence for future learning
                    nlg.concept_projector.record_covariance(slot_k, [concept_idx])

        # Merge: neural projected concepts добавляются к result
        merged = dict(original_result)
        for name, score in neural_additions.items():
            existing = merged.get(name, 0.0)
            merged[name] = max(existing, score)

        # Re-sort by score
        return sorted(merged.items(), key=lambda x: -x[1])[:8]

    slot_labeler.process_slots = neural_process_slots
    print("[NeuralLang] Patched SlotLabeler.process_slots with NeuralConceptProjector")


def _patch_tick_step(sim, nlg) -> None:
    """
    Перехватываем шаг симуляции для записи в InterventionalSpatialMemory.
    
    После каждого agent.step(): записываем (slot_before, action, slot_after).
    """
    original_run_step = sim._run_agent_or_skill_step

    def patched_run_agent_or_skill_step(engine_tick: int) -> dict:
        vis_env = getattr(sim, "_visual_env", None)
        slot_before = None
        if vis_env is not None:
            sv = getattr(vis_env, "_last_slot_vecs", None)
            if sv is not None:
                slot_before = sv.mean(dim=0).detach().clone()

        result = original_run_step(engine_tick)

        # Record interventional experience
        if slot_before is not None and not result.get("blocked") and not result.get("skipped"):
            vis_env2 = getattr(sim, "_visual_env", None)
            if vis_env2 is not None:
                sv2 = getattr(vis_env2, "_last_slot_vecs", None)
                if sv2 is not None:
                    slot_after = sv2.mean(dim=0).detach()
                    obs = {}
                    try:
                        obs = dict(sim.agent.env.observe())
                    except Exception:
                        pass
                    nlg.on_agent_step(
                        tick=engine_tick,
                        slot_vec_before=slot_before,
                        slot_vec_after=slot_after,
                        action_var=result.get("variable"),
                        action_val=result.get("value"),
                        body_obs=obs,
                    )

                    # Distill speech decoder если накопилось
                    if sim.tick % 50 == 0 and len(nlg._distill_buf) >= 4:
                        def _distill():
                            nlg.distill_step(batch_size=4)
                        threading.Thread(
                            target=_distill, daemon=True, name="rkk-nlg-distill"
                        ).start()

        return result

    sim._run_agent_or_skill_step = patched_run_agent_or_skill_step
    print("[NeuralLang] Patched _run_agent_or_skill_step for spatial memory writes")


# ─── Snapshot helper ──────────────────────────────────────────────────────────
def get_neural_lang_snapshot(sim) -> dict[str, Any] | None:
    nlg = getattr(sim, "_neural_lang", None)
    if nlg is None:
        return None
    return nlg.snapshot()
