# Implementation Plan - Humanoid 'Roll and Freeze' Recovery Override Fix

This plan addresses the humanoid's "roll and freeze" behavior during the `RECOVER_POSTURE` override. Instead of standing up fully, the humanoid rolls onto its side or stomach, halts, and remains frozen until the maximum tick threshold is reached.

## User Review Required

> [!IMPORTANT]
> This fix will adjust the low-level motor intent residual application rate and update the LLM prompt schema. These changes will significantly improve the smoothness and success of physical standing up recoveries without breaking any of the existing 66 unit tests.

## Root Cause Analysis

Three compounding issues cause this behavior:
1. **LLM Step Duration Bug (`ticks` Confusion)**:
   - The LLM interprets `ticks` as a step sequence index (e.g. step 1, 2, 3...) rather than the duration of the step in simulation frames (e.g. 10 to 80).
   - The entire LLM recovery plan executes in just ~15 ticks (~60 milliseconds), giving the physics engine zero time to move the limbs.
2. **Residual Accumulation Saturation**:
   - Step intent deltas are applied directly to the motor state *every single simulation tick*. 
   - Because they are added per tick without dividing by the step's duration, intents like knee flexion saturate instantly to their limits in 3-4 ticks, locking the joints and bypassing physical transitions.
3. **No-Replan Loop while Lying Down**:
   - When the humanoid rolls onto its side/stomach, its Base Z coordinate rises slightly above `0.24` (the fallen threshold).
   - This makes `is_fallen()` return `False`, so the active override tick receives `fallen=False`.
   - The posture exit gate correctly keeps the override session active because the humanoid is not fully upright (`posture_ok=False`).
   - However, the replanning check `if self._s2_override_active and fallen:` prevents a replan because `fallen` is `False`.
   - As a result, the humanoid is stuck in the override state forever, holding the saturated final step of the old recovery plan without ever request a new one.

---

## Proposed Changes

### Slow Control Engine

#### [MODIFY] [controller.py](file:///c:/Users/Andrey/Desktop/agi/rkk/backend/engine/system2/controller.py)
- **Divide Extra Residuals by Duration**: In `_recovery_extra_residuals`, divide the step's deltas by its `ticks` duration so the target delta is accumulated smoothly over the course of the step.
- **Divide Baseline Residuals by Plan Interval**: In `_apply_recover_bundle_no_candidate`, divide the baseline residuals from the posture bundle by `_plan_every_ticks()` (48) so they are scaled smoothly over the planning cycle rather than accumulating instantly.
- **Enable Replanning while Lying Down**: Modify the replanning check in `_maybe_tick_fallen_override` to dispatch `_maybe_dispatch_recovery_llm_replan` whenever `self._s2_override_active` is active, regardless of `fallen`. This allows the robot to ask for a new plan once the current one is exhausted if it is still lying on the floor.

#### [MODIFY] [teacher.py](file:///c:/Users/Andrey/Desktop/agi/rkk/backend/engine/system2/teacher.py)
- **Clarify LLM Prompt Schema**: Update the instruction for `"ticks"` in the recovery prompt to make it extremely clear that `ticks` represents step duration in simulation ticks (10-80 frames), not a step sequence index.

---

## Verification Plan

### Automated Tests
- Run `pytest backend/tests` to verify that all 66 unit tests continue to pass perfectly without regressions.

### Manual Verification
- Run a simulation and verify that LLM recovery steps are generated with proper durations (e.g. 20-40 ticks) and executed smoothly by the humanoid.
