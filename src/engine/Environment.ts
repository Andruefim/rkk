// ─── Interventional Environment ───────────────────────────────────────────────
// The "real world" — has a hidden causal structure the agent must discover.
// Agent can only see observations, not the ground truth graph.
// This is Phase I (Causal Crib): pure logical/physical variables.

export interface EnvVariable {
  id: string;
  value: number;
  min: number;
  max: number;
  label: string;
}

export interface GroundTruthEdge {
  from: string;
  to: string;
  weight: number;       // true causal weight
  noiseStd: number;     // noise added to output (interventional hardness)
  nonlinear?: boolean;  // some edges are nonlinear (harder to discover)
}

export class Environment {
  variables: Map<string, EnvVariable>;
  private _groundTruth: GroundTruthEdge[];
  private _noiseLevel: number;
  interventionCount = 0;

  constructor(preset: "physics" | "chemistry" | "logic" = "physics") {
    this.variables    = new Map();
    this._groundTruth = [];
    this._noiseLevel  = 0.05;
    this._initPreset(preset);
  }

  // ── Presets (Phase I environments) ──────────────────────────────────────────

  private _initPreset(preset: string) {
    if (preset === "physics") {
      // Ground truth: simple thermodynamic system
      // Temperature → Pressure → Volume → StateChange
      //                         ↗
      // Temperature → Energy ──┘
      const vars = [
        { id: "Temp",        value: 0.5, min: 0, max: 1, label: "Temperature" },
        { id: "Pressure",    value: 0.5, min: 0, max: 1, label: "Pressure" },
        { id: "Volume",      value: 0.5, min: 0, max: 1, label: "Volume" },
        { id: "Energy",      value: 0.3, min: 0, max: 1, label: "Energy" },
        { id: "StateChange", value: 0.1, min: 0, max: 1, label: "State Change" },
        { id: "Entropy",     value: 0.2, min: 0, max: 1, label: "Entropy" },
      ];
      vars.forEach(v => this.variables.set(v.id, v));

      this._groundTruth = [
        { from: "Temp",     to: "Pressure",    weight:  0.80, noiseStd: 0.04 },
        { from: "Temp",     to: "Energy",      weight:  0.70, noiseStd: 0.05 },
        { from: "Pressure", to: "Volume",      weight: -0.60, noiseStd: 0.06, nonlinear: true },
        { from: "Energy",   to: "StateChange", weight:  0.75, noiseStd: 0.05 },
        { from: "Volume",   to: "StateChange", weight:  0.40, noiseStd: 0.08 },
        { from: "Energy",   to: "Entropy",     weight:  0.55, noiseStd: 0.06 },
        // Spurious: Pressure and Energy are correlated but Pressure doesn't cause Energy
        // The agent needs to discover this via do(Pressure) → Energy doesn't change
      ];
    }

    if (preset === "chemistry") {
      const vars = [
        { id: "Reactant_A", value: 0.8, min: 0, max: 1, label: "Reactant A" },
        { id: "Reactant_B", value: 0.6, min: 0, max: 1, label: "Reactant B" },
        { id: "Catalyst",   value: 0.3, min: 0, max: 1, label: "Catalyst" },
        { id: "Temp",       value: 0.5, min: 0, max: 1, label: "Temperature" },
        { id: "Rate",       value: 0.4, min: 0, max: 1, label: "Reaction Rate" },
        { id: "Product",    value: 0.2, min: 0, max: 1, label: "Product Yield" },
      ];
      vars.forEach(v => this.variables.set(v.id, v));

      this._groundTruth = [
        { from: "Reactant_A", to: "Rate",    weight: 0.65, noiseStd: 0.05 },
        { from: "Reactant_B", to: "Rate",    weight: 0.55, noiseStd: 0.05 },
        { from: "Catalyst",   to: "Rate",    weight: 0.80, noiseStd: 0.03 },
        { from: "Temp",       to: "Rate",    weight: 0.70, noiseStd: 0.06, nonlinear: true },
        { from: "Rate",       to: "Product", weight: 0.90, noiseStd: 0.04 },
        { from: "Temp",       to: "Product", weight: 0.20, noiseStd: 0.08 }, // weak direct
      ];
    }

    if (preset === "logic") {
      // Logical system: program variables (closest to code sandbox)
      const vars = [
        { id: "Input",    value: 1.0, min: 0, max: 1, label: "Input" },
        { id: "Condition",value: 0.5, min: 0, max: 1, label: "Condition" },
        { id: "Branch_A", value: 0.0, min: 0, max: 1, label: "Branch A" },
        { id: "Branch_B", value: 1.0, min: 0, max: 1, label: "Branch B" },
        { id: "Output",   value: 0.5, min: 0, max: 1, label: "Output" },
        { id: "Error",    value: 0.0, min: 0, max: 1, label: "Error" },
      ];
      vars.forEach(v => this.variables.set(v.id, v));

      this._groundTruth = [
        { from: "Input",     to: "Condition", weight:  0.90, noiseStd: 0.01 }, // near-deterministic
        { from: "Condition", to: "Branch_A",  weight:  0.95, noiseStd: 0.01 },
        { from: "Condition", to: "Branch_B",  weight: -0.95, noiseStd: 0.01 }, // exclusive
        { from: "Branch_A",  to: "Output",    weight:  0.80, noiseStd: 0.02 },
        { from: "Branch_B",  to: "Output",    weight:  0.60, noiseStd: 0.02 },
        { from: "Input",     to: "Error",     weight:  0.10, noiseStd: 0.05 }, // rare
      ];
    }
  }

  // ── Observe current state ──
  observe(): Record<string, number> {
    const obs: Record<string, number> = {};
    this.variables.forEach((v, id) => { obs[id] = v.value; });
    return obs;
  }

  // ── do(variable = value) → ground-truth response ──
  // This is the "hard" interventional feedback that cannot be hallucinated.
  intervene(variable: string, value: number): Record<string, number> {
    this.interventionCount++;

    // 1. Set the intervened variable
    const varNode = this.variables.get(variable);
    if (!varNode) return this.observe();

    const prevValues: Record<string, number> = {};
    this.variables.forEach((v, id) => { prevValues[id] = v.value; });

    // 2. Propagate through GROUND TRUTH (not agent's model)
    const newValues = { ...prevValues };
    newValues[variable] = value;

    // BFS over ground truth edges
    const visited = new Set<string>([variable]);
    const queue   = [variable];

    while (queue.length > 0) {
      const current = queue.shift()!;
      const outgoing = this._groundTruth.filter(e => e.from === current);

      for (const edge of outgoing) {
        const upstreamDelta = newValues[current] - prevValues[current];
        const causalEffect  = edge.weight * upstreamDelta;
        const nonlinear     = edge.nonlinear ? Math.tanh(causalEffect * 2) : causalEffect;
        const noise         = (Math.random() - 0.5) * 2 * edge.noiseStd;

        newValues[edge.to] = Math.max(
          this.variables.get(edge.to)!.min,
          Math.min(
            this.variables.get(edge.to)!.max,
            (newValues[edge.to] ?? prevValues[edge.to]) + nonlinear + noise
          )
        );

        if (!visited.has(edge.to)) {
          visited.add(edge.to);
          queue.push(edge.to);
        }
      }
    }

    // 3. Update env state
    this.variables.forEach((v, id) => {
      if (id in newValues) v.value = newValues[id];
    });

    return this.observe();
  }

  // ── How many ground truth edges has agent discovered? (for scoring) ──
  discoveryRate(agentEdges: Array<{ from: string; to: string; weight: number }>): number {
    let hits = 0;
    for (const gt of this._groundTruth) {
      const found = agentEdges.find(
        e => e.from === gt.from && e.to === gt.to && Math.abs(e.weight - gt.weight) < 0.3
      );
      if (found) hits++;
    }
    return hits / this._groundTruth.length;
  }

  get variableIds(): string[] {
    return Array.from(this.variables.keys());
  }

  get groundTruthSize(): number {
    return this._groundTruth.length;
  }

  // Return ground truth for debug/visualization (would not exist in real system)
  getGroundTruthEdges() {
    return this._groundTruth.map(e => ({ ...e }));
  }
}
