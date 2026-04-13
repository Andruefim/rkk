"""Simulation mixin: pose, embodied reward async."""
from __future__ import annotations

from engine.features.simulation.mixin_imports import *


class SimulationPoseEmbodiedMixin:
    def _skill_snapshot(self) -> dict | None:
        if not self._skill_library_enabled():
            return None
        lib = self._skill_library
        out: dict = {"enabled": True, "active": None}
        if lib is not None:
            out.update(lib.snapshot())
        else:
            out["n_skills"] = 0
            out["skills"] = []
            out["history_len"] = 0
        if self._skill_exec is not None:
            sk = self._skill_exec["skill"]
            out["active"] = {
                "name": sk.name,
                "step": self._skill_exec["index"],
                "total": len(sk.action_sequence),
            }
        return out

    def _pose_snapshot(self) -> "PoseSnapshot | None":
        """Level 1-A: Construct PoseSnapshot from current environment state."""
        if not _EMBODIED_REWARD_AVAILABLE:
            return None
        try:
            env = self.agent.env
            obs = dict(env.observe())
            nodes = dict(self.agent.graph.nodes)
            posture_window = (
                list(self._mc_posture_window)
                if hasattr(self, "_mc_posture_window")
                else []
            )
            fallen_window = (
                list(self._mc_fallen_count_window)
                if hasattr(self, "_mc_fallen_count_window")
                else []
            )
            recent_fall_rate = float(np.mean(fallen_window)) if fallen_window else 0.0
            mean_posture = float(np.mean(posture_window)) if posture_window else 0.5
            mc = self._ensure_motor_cortex() if hasattr(self, "_ensure_motor_cortex") else None
            cpg_w = mc.cpg_weight if mc is not None else 1.0
            mc_q = mc._quality_ema if mc is not None else 0.0
            return PoseSnapshot.from_obs_and_graph(
                obs=obs,
                graph_nodes=nodes,
                tick=self.tick,
                fallen=self._fall_count > 0,
                fall_count=self._fall_count,
                cpg_weight=cpg_w,
                mc_quality_ema=mc_q,
                recent_fall_rate=recent_fall_rate,
                mean_posture_recent=mean_posture,
            )
        except Exception as e:
            print(f"[Simulation] _pose_snapshot error: {e}")
            return None

    async def _run_embodied_reward_async(self) -> None:
        """Level 1-A: Run embodied LLM reward shaping (async, called as task)."""
        if not _EMBODIED_REWARD_AVAILABLE or self._embodied_reward_ctrl is None:
            return
        pose = self._pose_snapshot()
        if pose is None:
            return
        try:
            mc = self._ensure_motor_cortex() if hasattr(self, "_ensure_motor_cortex") else None
            result = await self._embodied_reward_ctrl.run(
                pose=pose,
                agent=self.agent,
                locomotion_ctrl=self._locomotion_controller,
                motor_cortex=mc,
                llm_url=get_ollama_generate_url(),
                llm_model=get_ollama_model(),
            )
            if result.ok and (result.verbal or result.priority_issue):
                issue_str = f" [{result.priority_issue}]" if result.priority_issue else ""
                try:
                    _r = float(result.combined_reward)
                    _ps = float(result.posture_score)
                    _gq = float(result.gait_quality)
                except (TypeError, ValueError):
                    _r, _ps, _gq = 0.0, 0.0, 0.0
                self._add_event(
                    f"🧠 EmbodiedLLM: r={_r:+.2f}"
                    f" pos={_ps:.2f} gait={_gq:.2f}"
                    f"{issue_str} +{len(result.seeds)}seeds",
                    "#ff99ff",
                    "phase",
                )
        except Exception as e:
            print(f"[Simulation] embodied reward error: {e}")
