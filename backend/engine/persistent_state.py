"""
persistent_state.py — сохранение состояния обучения между перезапусками сервера.

Проблема: при перезапуске сервера с существующим autosave.rkk всё обнуляется:
  - tick счётчик → 0 (CPG annealing начинается заново)
  - curriculum stage → 0 (stand снова)
  - episodic memory → пустая (история падений потеряна)
  - sleep schedule → потеряна
  - RSSM hidden → сброшен
  - InnerVoice train steps → 0

Решение: autosave.meta.json рядом с autosave.rkk.
Сохраняется при каждом autosave. Восстанавливается при load если
autosave.rkk существует и не был удалён вручную.

Формат autosave.meta.json:
{
  "version": 2,
  "tick": 12453,
  "wall_time_seconds": 7200,
  "curriculum_stage_idx": 3,
  "curriculum_stage_name": "slow_step",
  "total_falls": 147,
  "total_successes": 23,
  "sleep_count": 3,
  "last_sleep_tick": 10000,
  "rssm_upgraded": true,
  "inner_voice_train_steps": 89,
  "cpg_weight": 0.42,
  "mc_dominant": false,
  "reward_total_signals": 12000,
  "constitution_violations": 2,
  "fall_patterns": [...],
  "skill_success_rates": {...}
}

Правила:
  1. Если autosave.rkk НЕ существует → meta игнорируется (свежий старт)
  2. Если autosave.rkk существует → meta восстанавливается полностью
  3. Если meta повреждена → предупреждение, продолжаем без restore
  4. tick восстанавливается в simulation._tick напрямую
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


META_VERSION = 2
META_FILENAME = "autosave.meta.json"


def _meta_path(rkk_path: str) -> Path:
    """Get meta file path adjacent to .rkk file."""
    return Path(rkk_path).parent / META_FILENAME


# ── Meta state dataclass ──────────────────────────────────────────────────────
@dataclass
class PersistentMeta:
    version: int = META_VERSION
    tick: int = 0
    wall_time_seconds: float = 0.0
    saved_at: str = ""

    # Training progress
    curriculum_stage_idx: int = 0
    curriculum_stage_name: str = "static_stance"
    curriculum_total_advances: int = 0
    total_falls: int = 0
    total_successes: int = 0

    # Sleep
    sleep_count: int = 0
    last_sleep_tick: int = -1
    total_sleep_ticks: int = 0

    # Architecture state
    rssm_upgraded: bool = False
    rssm_upgrade_tick: int = -1
    cpg_weight: float = 1.0
    mc_dominant: bool = False
    motor_cortex_train_steps: int = 0

    # Inner voice
    inner_voice_train_steps: int = 0
    llm_teacher_calls: int = 0

    # Reward
    reward_total_signals: int = 0
    constitution_violations: int = 0

    # Patterns (serializable summary)
    fall_patterns: list[dict] = field(default_factory=list)
    skill_success_rates: dict[str, float] = field(default_factory=dict)

    # Physical curriculum
    mastered_skills: list[str] = field(default_factory=list)
    failed_skills: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["saved_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PersistentMeta":
        # Filter only known fields
        known = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in known}
        return cls(**filtered)

    def is_fresh_start(self) -> bool:
        return self.tick == 0

    def summary_str(self) -> str:
        return (
            f"tick={self.tick}, curriculum={self.curriculum_stage_name}({self.curriculum_stage_idx}), "
            f"falls={self.total_falls}, sleeps={self.sleep_count}, "
            f"rssm={'ON' if self.rssm_upgraded else 'OFF'}, "
            f"cpg_w={self.cpg_weight:.3f}"
        )


# ── Persistence manager ────────────────────────────────────────────────────────
class PersistenceManager:
    """
    Manages autosave.meta.json alongside autosave.rkk.

    Интеграция в simulation.py:
      self._persist = PersistenceManager(rkk_path)

    При старте:
      meta = self._persist.try_load(rkk_path)
      if meta is not None:
          self._restore_from_meta(meta)

    При каждом autosave:
      self._persist.save(self._collect_meta())
    """

    def __init__(self, rkk_path: str):
        self._rkk_path = rkk_path
        self._meta_path = _meta_path(rkk_path)
        self._start_wall_time = time.time()
        self._last_save_time = 0.0
        self._loaded_meta: PersistentMeta | None = None

    def rkk_exists(self) -> bool:
        return Path(self._rkk_path).exists()

    def meta_exists(self) -> bool:
        return self._meta_path.exists()

    def try_load(self) -> PersistentMeta | None:
        """
        Load meta if both .rkk and .meta.json exist.
        Returns None if fresh start or meta damaged.
        """
        if not self.rkk_exists():
            print("[Persist] No autosave.rkk found — fresh start")
            return None
        if not self.meta_exists():
            print("[Persist] autosave.rkk found but no meta — partial restore (tick=0)")
            return None

        try:
            with open(self._meta_path, "r", encoding="utf-8") as f:
                d = json.load(f)
        except Exception as e:
            print(f"[Persist] meta.json damaged ({e}) — starting fresh tick count")
            return None

        if d.get("version", 0) < 1:
            print("[Persist] meta.json version too old — ignoring")
            return None

        try:
            meta = PersistentMeta.from_dict(d)
            self._loaded_meta = meta
            print(f"[Persist] ✅ Restored: {meta.summary_str()}")
            return meta
        except Exception as e:
            print(f"[Persist] meta.json parse error ({e}) — fresh tick count")
            return None

    def save(self, meta: PersistentMeta) -> bool:
        """Save meta to disk. Returns True on success."""
        meta.wall_time_seconds += time.time() - self._start_wall_time
        self._start_wall_time = time.time()
        try:
            tmp = str(self._meta_path) + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(meta.to_dict(), f, indent=2)
            os.replace(tmp, self._meta_path)
            self._last_save_time = time.time()
            return True
        except Exception as e:
            print(f"[Persist] save failed: {e}")
            return False

    def delete(self) -> None:
        """Delete meta file (when user manually resets / deletes .rkk)."""
        if self._meta_path.exists():
            self._meta_path.unlink()
            print("[Persist] meta.json deleted")

    def snapshot(self) -> dict[str, Any]:
        return {
            "rkk_path": self._rkk_path,
            "meta_path": str(self._meta_path),
            "rkk_exists": self.rkk_exists(),
            "meta_exists": self.meta_exists(),
            "last_save_time": self._last_save_time,
        }


# ── Simulation integration helpers ────────────────────────────────────────────
def collect_meta_from_simulation(sim) -> PersistentMeta:
    """
    Collect PersistentMeta from simulation instance.
    Call before each autosave.
    """
    meta = PersistentMeta()
    meta.tick = getattr(sim, "tick", 0)

    # Curriculum
    if sim._curriculum is not None:
        meta.curriculum_stage_idx = sim._curriculum._current_idx
        meta.curriculum_stage_name = sim._curriculum.current_stage.name
        meta.curriculum_total_advances = sim._curriculum.total_advances

    # Episodic memory
    if sim._episodic_memory is not None:
        meta.total_falls = sim._episodic_memory.total_falls_recorded
        meta.total_successes = sim._episodic_memory.total_successes_recorded
        meta.fall_patterns = [
            {"desc": p.description[:100], "conf": round(p.confidence, 3)}
            for p in sim._episodic_memory._patterns[:5]
        ]

    # Sleep
    if sim._sleep_ctrl is not None:
        meta.sleep_count = sim._sleep_ctrl.sleep_count
        meta.last_sleep_tick = sim._sleep_ctrl.last_sleep_tick
        meta.total_sleep_ticks = sim._sleep_ctrl.total_sleep_ticks

    # RSSM
    meta.rssm_upgraded = getattr(sim, "_rssm_upgraded", False)
    meta.rssm_upgrade_tick = getattr(sim, "_rssm_upgrade_tick", -1)

    # CPG / Motor Cortex
    lc = getattr(sim, "_locomotion_controller", None)
    if lc is not None:
        meta.cpg_weight = float(getattr(lc, "cpg_weight", 1.0))
    mc = getattr(sim, "_motor_cortex", None)
    if mc is not None:
        meta.mc_dominant = float(getattr(mc, "cpg_weight", 1.0)) < 0.4
        meta.motor_cortex_train_steps = getattr(mc, "train_steps", 0)

    # Inner voice
    if sim._inner_voice is not None:
        meta.inner_voice_train_steps = sim._inner_voice.train_steps
    if sim._llm_teacher is not None:
        meta.llm_teacher_calls = sim._llm_teacher.total_calls

    # Intrinsic objective steps (раньше reward coordinator)
    intr = getattr(sim, "_intrinsic", None)
    if intr is not None:
        meta.reward_total_signals = int(getattr(intr, "total_steps", 0))
    meta.constitution_violations = 0

    # Physical curriculum mastery
    if sim._physical_curriculum is not None:
        meta.mastered_skills = list(sim._physical_curriculum.mastered)
        meta.failed_skills = list(sim._physical_curriculum.failed)

    return meta


def restore_meta_to_simulation(sim, meta: PersistentMeta) -> None:
    """
    Restore PersistentMeta into simulation instance after .rkk load.
    """
    if meta is None or meta.is_fresh_start():
        return

    print(f"[Persist] Restoring simulation state: {meta.summary_str()}")

    # Tick counter — most important
    sim.tick = meta.tick
    sim._tick = meta.tick

    # Curriculum stage
    if sim._curriculum is not None and meta.curriculum_stage_idx > 0:
        sim._curriculum._current_idx = min(
            meta.curriculum_stage_idx,
            len(sim._curriculum._stages) - 1,
        )
        stage = sim._curriculum.current_stage
        stage.entered_tick = meta.tick
        stage.ticks_in_stage = 0
        sim._curriculum.total_advances = meta.curriculum_total_advances
        print(f"[Persist] Curriculum restored: stage {meta.curriculum_stage_idx} '{meta.curriculum_stage_name}'")

    # Episodic memory counters
    if sim._episodic_memory is not None:
        sim._episodic_memory.total_falls_recorded = meta.total_falls
        sim._episodic_memory.total_successes_recorded = meta.total_successes

    # Sleep
    if sim._sleep_ctrl is not None:
        sim._sleep_ctrl.sleep_count = meta.sleep_count
        sim._sleep_ctrl.last_sleep_tick = meta.last_sleep_tick
        sim._sleep_ctrl.total_sleep_ticks = meta.total_sleep_ticks

    # RSSM state
    if meta.rssm_upgraded and not getattr(sim, "_rssm_upgraded", False):
        print(f"[Persist] RSSM was active at tick {meta.rssm_upgrade_tick} — will re-upgrade")
        sim._rssm_upgrade_tick_override = meta.rssm_upgrade_tick

    # CPG weight restore
    lc = getattr(sim, "_locomotion_controller", None)
    if lc is not None and meta.cpg_weight < 1.0:
        if hasattr(lc, "cpg_weight"):
            lc.cpg_weight = meta.cpg_weight
            print(f"[Persist] CPG weight restored: {meta.cpg_weight:.3f}")

    # Physical curriculum
    if sim._physical_curriculum is not None:
        sim._physical_curriculum.mastered = set(meta.mastered_skills)
        sim._physical_curriculum.failed = set(meta.failed_skills)

    print(f"[Persist] ✅ Simulation tick restored to {meta.tick}")
