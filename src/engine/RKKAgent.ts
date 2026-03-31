import { CausalGraph } from "./CausalGraph";
import { Environment } from "./Environment";
import type { InterventionResult, EpistemicScore, AgentSnapshot } from "./types";

// ─── RKK Agent ────────────────────────────────────────────────────────────────
// Implements:
//   - Causal hypothesis building from interventions
//   - Intrinsic drive: max E[d|G|/dt]
//   - Alpha trust: text-prior edges decay if unsupported by sandbox
//   - Epistemic scoring: System 1 proxy for E[IG]

export class RKKAgent {
  id:   number;
  name: string;
  graph: CausalGraph;
  env:  Environment;

  private _history: InterventionResult[] = [];
  private _compressionGainHistory: number[] = [];
  private _totalInterventions = 0;
  private _phi = 0.5; // autonomy proxy: ratio of self-determined vs demon-disrupted

  // Text priors: spurious correlations seeded from "language" (will be annealed)
  private _textPriors: Array<{ from: string; to: string; weight: number }> = [];

  constructor(id: number, name: string, env: Environment) {
    this.id    = id;
    this.name  = name;
    this.env   = env;
    this.graph = new CausalGraph();
    this._bootstrap();
  }

  // ── Bootstrap: inject text priors (contaminated starting point) ──
  // These include some true edges and some spurious correlations.
  // Epistemic Annealing will burn away the spurious ones.
  private _bootstrap() {
    const vars = this.env.variableIds;

    // Initialize nodes
    vars.forEach(id => {
      const envVar = this.env.variables.get(id)!;
      this.graph.setNode(id, envVar.value, 0.08);
    });

    // Seed text priors — mix of correct and spurious edges with low alpha
    // Correct (will be strengthened): first 3 ground truth edges (roughly)
    const gt = this.env.getGroundTruthEdges();
    gt.slice(0, 2).forEach(e => {
      this.graph.setEdge(e.from, e.to, e.weight * 0.3 + (Math.random() - 0.5) * 0.4, 0.06);
    });

    // Spurious (will be annealed out): cross-connections that look plausible
    if (vars.length >= 4) {
      this.graph.setEdge(vars[1], vars[3], 0.35, 0.05); // spurious: looks correlated
      this.graph.setEdge(vars[2], vars[0], 0.20, 0.04); // spurious: wrong direction
    }

    this._textPriors = this.graph.edges.map(e => ({ ...e }));
  }

  // ── Epistemic scoring: which intervention has highest expected info gain? ──
  // System 1 proxy: score by current edge uncertainty × variable connectivity
  scoreInterventions(): EpistemicScore[] {
    const vars = this.env.variableIds;
    const scores: EpistemicScore[] = [];

    for (const varId of vars) {
      for (const targetId of vars) {
        if (varId === targetId) continue;

        const uncertainty = this.graph.edgeUncertainty(varId, targetId);
        // Value of intervening: proportional to how uncertain the causal link is
        const connectivity = this.graph.edges.filter(
          e => e.from === varId || e.to === varId
        ).length;

        // E[IG] ≈ uncertainty * (1 + log(1 + connectivity))
        const expectedIG = uncertainty * (1 + Math.log1p(connectivity));

        // Sample a test value: extremes are more informative than midpoints
        const testValue = Math.random() > 0.5 ? 0.9 : 0.1;

        scores.push({
          variable: varId,
          value: testValue,
          expectedInfoGain: expectedIG,
          uncertaintyReduction: uncertainty,
        });
      }
    }

    // Sort by expected info gain descending
    return scores.sort((a, b) => b.expectedInfoGain - a.expectedInfoGain);
  }

  // ── Execute one intervention step ──
  step(): InterventionResult {
    const scores  = this.scoreInterventions();
    const best    = scores[0]; // greedy: take highest E[IG] action
    if (!best) throw new Error("No interventions available");

    const sizeBefore = this.graph.size;

    // Predict what our current graph says should happen
    const predicted = this.graph.propagate(best.variable, best.value);

    // Get ground truth response from environment
    const observed = this.env.intervene(best.variable, best.value);

    // Compute prediction error per variable
    const errors: Record<string, number> = {};
    let   totalError = 0;
    for (const [id, obs] of Object.entries(observed)) {
      const err = Math.abs((predicted[id] ?? 0) - obs);
      errors[id] = err;
      totalError += err;
    }
    const mse = totalError / Object.keys(observed).length;

    // ── Update causal graph based on observed interventional outcomes ──
    const updatedEdges: string[] = [];

    for (const [id, obs] of Object.entries(observed)) {
      if (id === best.variable) continue;

      const delta       = obs - (this.graph.nodes.get(id)?.value ?? obs);
      const inputDelta  = best.value - (this.graph.nodes.get(best.variable)?.value ?? best.value);

      if (Math.abs(inputDelta) < 0.001) continue;

      const empiricalWeight = Math.tanh(delta / (inputDelta + 0.001));

      // If effect is significant → this is a real causal link
      if (Math.abs(empiricalWeight) > 0.08) {
        const existingEdge = this.graph.edges.find(
          e => e.from === best.variable && e.to === id
        );
        const prevAlpha = existingEdge?.alphaTrust ?? 0;
        // Alpha increases with each confirmation
        const newAlpha = Math.min(0.98, prevAlpha + 0.12 * (1 - prevAlpha));

        this.graph.setEdge(best.variable, id, empiricalWeight, newAlpha);
        updatedEdges.push(`${best.variable}→${id}`);
      } else {
        // Effect near zero → prune if it was a spurious prior
        const existingEdge = this.graph.edges.find(
          e => e.from === best.variable && e.to === id
        );
        if (existingEdge && existingEdge.alphaTrust < 0.3) {
          // Epistemic Annealing: burn low-alpha edges that don't survive do()
          existingEdge.alphaTrust = Math.max(0, existingEdge.alphaTrust - 0.08);
          if (existingEdge.alphaTrust < 0.02) {
            this.graph.removeEdge(best.variable, id);
            updatedEdges.push(`PRUNED: ${best.variable}→${id}`);
          }
        }
      }

      // Update node value in graph
      this.graph.nodes.forEach((n, nid) => {
        if (nid === id) n.value = obs;
      });
    }

    // Update the intervened node's value
    const node = this.graph.nodes.get(best.variable);
    if (node) node.value = best.value;

    const sizeAfter      = this.graph.size;
    const compressionDelta = sizeBefore - sizeAfter; // positive = graph got more efficient

    this._compressionGainHistory.push(compressionDelta);
    if (this._compressionGainHistory.length > 20) this._compressionGainHistory.shift();

    this._totalInterventions++;

    // Update phi: autonomy = proportion of steps that increased compression
    const positiveSteps = this._compressionGainHistory.filter(x => x >= 0).length;
    this._phi = positiveSteps / this._compressionGainHistory.length;

    const result: InterventionResult = {
      intervention: { variable: best.variable, value: best.value },
      predicted,
      observed,
      predictionError: mse,
      compressionDelta,
      updatedEdges,
    };

    this._history.push(result);
    if (this._history.length > 50) this._history.shift();

    return result;
  }

  // ── Demon disrupts: reduces phi, injects noise into alpha ──
  demonDisrupt(): void {
    this._phi = Math.max(0.05, this._phi - 0.15);
    // Corrupt a random low-alpha edge
    const vulnerable = this.graph.edges.filter(e => e.alphaTrust < 0.5);
    if (vulnerable.length > 0) {
      const edge = vulnerable[Math.floor(Math.random() * vulnerable.length)];
      edge.alphaTrust = Math.max(0.02, edge.alphaTrust - 0.12);
      edge.weight    += (Math.random() - 0.5) * 0.2;
    }
  }

  // ── Get discovery rate: how well does graph match ground truth? ──
  get discoveryRate(): number {
    return this.env.discoveryRate(this.graph.edges.map(e => ({
      from: e.from,
      to: e.to,
      weight: e.weight,
    })));
  }

  // ── Running compression gain: E[d|G|/dt] over last N steps ──
  get compressionGain(): number {
    if (this._compressionGainHistory.length === 0) return 0;
    const sum = this._compressionGainHistory.reduce((s, x) => s + x, 0);
    return sum / this._compressionGainHistory.length;
  }

  // ── Snapshot for UI ──
  snapshot(): AgentSnapshot {
    return {
      id:                 this.id,
      name:               this.name,
      graphSize:          this.graph.size,
      compressionGain:    this.compressionGain,
      alphaMean:          this.graph.alphaMean,
      nodeCount:          this.graph.nodes.size,
      edgeCount:          this.graph.edges.length,
      phi:                this._phi,
      totalInterventions: this._totalInterventions,
      lastAction:         this._history.length > 0
        ? `do(${this._history[this._history.length - 1].intervention.variable}=${
            this._history[this._history.length - 1].intervention.value.toFixed(2)
          })`
        : "—",
    };
  }

  get lastResult(): InterventionResult | null {
    return this._history.length > 0 ? this._history[this._history.length - 1] : null;
  }
}
