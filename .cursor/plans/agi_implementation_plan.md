# [RKK Architecture Roadmap & Implementation Plan]

This document outlines the detailed implementation steps to refactor the RKK architecture from its current state (approximations and monolithic structures) to a true Causal, Bayesian, and Modular AGI framework, following the strict priority sequence: 2 → 3 → 1 → 4 → 7 → 6 → 5.

## User Review Required

> [!IMPORTANT]
> Please review this phased roadmap. Each phase introduces foundational shifts (e.g., modular predictors, ensemble graphs). Once approved, we will break Phase 1 into actionable tasks and begin execution.

## Open Questions

> [!WARNING]
> 1. **Phase 1 (Modular World Model)**: How should we manage the memory overhead of having an independent MLP per node? Should we use small (e.g., 2-layer, 32-dim) MLPs?
> 2. **Phase 2 (Structure Learning)**: For the ensemble of graphs, what is a reasonable N (e.g., 4-8) to balance computational cost in the simulation loop while maintaining multiple hypotheses?
> 3. **Phase 3 (Genome Priors)**: For the low-rank factorization, do we want to implement this as an offline script that processes logged adjacency matrices from multiple environments, or as an online constraint during training?

## Proposed Changes

---

### [Phase 1: Modular Causal World Model (Current Priority 2)]

**Goal**: Replace the monolithic GRU/GNN shared-weight structure with true independent mechanisms.

#### [MODIFY] `backend/engine/causal_gnn.py`
- Remove the shared `msg_fn` and `out_dec` MLPs.
- Implement an `nn.ModuleDict` or list of independent small MLPs (e.g., `MechanismMLP`), one for each active variable in the graph.
- Update the `forward` pass so that `node_i`'s next state is predicted exclusively by `MechanismMLP_i(parents_of_i)`.
- Update intervention logic: When an intervention is applied to node X (`do(X=x)`), only the gradients for `MechanismMLP_X` are masked/ignored, and only descendants of X are re-evaluated.

#### [MODIFY] `backend/engine/temporal_world_model.py`
- Deprecate the single GRU (`TemporalWorldModel`).
- Integrate the new Modular Causal GNN to handle multi-step rollouts, preserving state locally within each mechanism if recurrence is needed, or dropping recurrence in favor of purely causal Markovian states.

---

### [Phase 2: Bayesian Structure Learning (Current Priority 3)]

**Goal**: Move from a single point-estimate graph with CMI heuristics to a Bayesian posterior (weighted ensemble) with orientation rules.

#### [MODIFY] `backend/engine/causal_graph.py`
- Add v-structure detection (collider logic: A → C ← B) to the existing CMI-based learning (`_structural_learning_step`).
- Implement orientation rules (e.g., PC algorithm rules) to direct edges found via conditional dependence.
- Replace the single adjacency matrix `W` with a `WeightedGraphEnsemble` class containing N matrices (`W_1 ... W_N`), each with an associated probability/weight.

#### [NEW] `backend/engine/hypothesis_testing.py`
- Implement an active hypothesis testing module.
- Given an ensemble of graphs, calculate the Expected Information Gain (EIG) or Jensen-Shannon divergence between the predictions of different graphs for a candidate action.
- Expose this EIG to the intrinsic motivation system.

---

### [Phase 3: Genome Priors via Compression (Current Priority 1)]

**Goal**: Create an evolutionary bottleneck constraint via low-rank factorization, avoiding LLM "wordiness".

#### [NEW] `backend/engine/genome/compressor.py`
- Implement a script to collect trained adjacency matrices (ensembles) from agents surviving in multiple diverse environments (e.g., flat, slope, stairs).
- Apply Low-Rank Factorization (SVD or Autoencoder bottleneck) to the aggregated matrices to find the minimal rank-k representation.
- Reconstruct the sparse adjacency matrix from the low-rank bottleneck.

#### [MODIFY] `backend/engine/genome/priors.py`
- Update the initialization script to load the factored genome (the "surviving" prior) as the initial `W` matrices for the ensemble.
- Introduce "molecular tags" (attributes in `CausalNode`) that enforce strict routing constraints (e.g., sensor nodes cannot have incoming edges, motor nodes cannot be parents of other motor nodes).

---

### [Phase 4: Hierarchical Active Inference (Current Priority 4)]

**Goal**: Upgrade the multi-timescale skeleton to proper Variational Inference.

#### [MODIFY] `backend/engine/hierarchical_active_inference.py`
- Replace hand-tuned PE-PID loops with proper ELBO optimization.
- At each level (sensorimotor, cognitive, executive), implement a Generative Model `p(o, s)` and a variational posterior `q(s)`.
- Implement message passing: top-down empirical priors (expectations) and bottom-up precision-weighted prediction errors.

#### [MODIFY] `backend/engine/multiscale_time.py`
- Replace fixed tick constants (5, 20) with learned temporal abstractions (e.g., using boundary detection or event segmentation where high prediction error signals a new temporal macro-step).

---

### [Phase 5: Intrinsic Motivation & Directed Exploration (Current Priority 7)]

**Goal**: Drive exploration to disambiguate causal structures rather than just maximizing simple prediction error.

#### [MODIFY] `backend/engine/intristic_objective.py`
- Deprecate the simple empowerment heuristic.
- Implement proper Channel Capacity estimation (Mutual Information between actions and future states) using the ensemble world model.
- Integrate the EIG from `hypothesis_testing.py` as the primary curiosity reward: the agent should seek states/actions where the ensemble's graphs disagree the most.

---

### [Phase 6: Embodiment Loop & Continual Learning (Current Priority 6)]

**Goal**: Ensure the loop runs continuously without catastrophic forgetting.

#### [MODIFY] `backend/engine/agent.py`
- Connect the full pipeline: Ensemble WM Predicts → Active Inference (ELBO) selects action → Environment steps → Causal Ensemble updates via Bayes rule → Genome prior slowly adapts (EMA).
- Implement replay buffer prioritization based on causal surprise to maintain rare but structurally important transitions, preventing forgetting of core physics.

---

### [Phase 7: System 1 Event-Driven SNN (Current Priority 5)]

**Goal**: Finalize the architecture for hardware/efficiency via spiking dynamics.

#### [MODIFY] `backend/engine/system1.py`
- Convert the synchronous MLPs to Leaky Integrate-and-Fire (LIF) neurons.
- Transition the `tick()` based loop to an event queue (processing only when $\Delta V > threshold$).
- Replace gradient descent with STDP (Spike-Timing-Dependent Plasticity) for online reflex adaptation.
- *Note: This phase is isolated and strictly deferred until Phases 1-6 are stable.*

## Verification Plan

### Automated Tests
- Create unit tests for independent mechanisms: Verifying that `do(X)` strictly leaves non-descendants unaffected without computing gradients for the whole graph.
- Create tests for v-structure detection in the CMI learner.
- Validate that the Ensemble variance decreases when an intervention definitively refutes one of the hypotheses.

### Manual Verification
- Run the simulation with Phase 1+2. The agent should demonstrate the ability to discard a false graph hypothesis when presented with a novel intervention.
- Visualize the low-rank compressed genome to visually confirm that essential locomotion priors (like bilateral symmetry) survive the bottleneck.
