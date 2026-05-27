# ADR: Modular Causal GNN World Model

## Status

Accepted — executive path uses `CausalGNNCore` (`RKK_WM_RSSM=0`).

## Context

Pearl-style modular SCMs assign each variable an independent structural equation. RKK must stay compatible with NOTEARS-style `W`, DAG penalties, RSI resize, and `causal_graph.train_step`.

## Decision

**Modular (per-node):** `MechanismMLP` per variable — parent aggregation + local decoder (`out_1`, multiscale heads, `latent_predictor`).

**Shared (global):**

- Adjacency `W` (d×d) and DAG constraint
- `node_enc`, `action_enc`, `target_enc` (JEPA)
- `sz_head_z` when `RKK_SZ_SPLIT` is enabled

This is *partial* modularity: mechanisms are local; structure and encoders are global.

## Intervention semantics

`do(X=x)` under `RKK_DO_DESCENDANT_ONLY=1`:

1. Clamp intervened node to `x` in latent/state.
2. Recompute **only strict descendants** in topological order.
3. Non-descendants return observed `X` (frozen one-step).

`intervention_loss` uses `forward_dynamics_under_do`; gradients on the intervened mechanism are masked.

## Configuration

| Key | Default | Role |
|-----|---------|------|
| `RKK_MECHANISM_HIDDEN` | 24 | MLP hidden per mechanism |
| `RKK_DO_DESCENDANT_ONLY` | 1 | Descendant-only do-forward |
| `RKK_MECHANISM_BATCHED` | 0 | Reserved for stacked mechanisms |

## Consequences

- RSI / `resize_to` must migrate `mechanisms[]`, not legacy `msg_fn`/`out_dec`.
- Ensemble structure learning shares executive `W` for speed; hypotheses live in `WeightedGraphEnsemble`.
- Full batched mechanisms optional later (`RKK_MECHANISM_BATCHED`).
