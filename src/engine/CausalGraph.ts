import type { CausalNode, CausalEdge, InterventionResult } from "./types";

// ─── Minimal Description Length proxy ────────────────────────────────────────
// |G| ≈ Σ edges * (1 + |weight| * entropy_factor)
// Smaller = more compressed = more understood
function mdlSize(edges: CausalEdge[]): number {
  if (edges.length === 0) return 0;
  return edges.reduce((sum, e) => {
    const entropyFactor = 1 - Math.abs(e.weight); // uncertain edges cost more
    const trustFactor   = 1 - e.alphaTrust;        // unconfirmed edges cost more
    return sum + 1 + entropyFactor + trustFactor;
  }, 0);
}

// ─── CausalGraph ─────────────────────────────────────────────────────────────
export class CausalGraph {
  nodes: Map<string, CausalNode> = new Map();
  edges: CausalEdge[]            = [];

  private _mdlCache: number | null = null;

  // ── Add / update node ──
  setNode(id: string, value = 0, alphaTrust = 0.1): void {
    this._mdlCache = null;
    const existing = this.nodes.get(id);
    if (existing) {
      existing.value = value;
    } else {
      this.nodes.set(id, { id, value, alphaTrust });
    }
  }

  // ── Add / update edge ──
  setEdge(from: string, to: string, weight: number, alphaTrust = 0.05): void {
    this._mdlCache = null;
    const existing = this.edges.find(e => e.from === from && e.to === to);
    if (existing) {
      existing.weight        = weight;
      existing.alphaTrust    = Math.min(1, alphaTrust);
      existing.interventionCount++;
    } else {
      this.edges.push({ from, to, weight, alphaTrust, interventionCount: 1 });
    }
  }

  // ── Remove edge (spurious correlation detected) ──
  removeEdge(from: string, to: string): void {
    this._mdlCache = null;
    this.edges = this.edges.filter(e => !(e.from === from && e.to === to));
  }

  // ── Get MDL size (cached) ──
  get size(): number {
    if (this._mdlCache === null) this._mdlCache = mdlSize(this.edges);
    return this._mdlCache;
  }

  // ── Forward propagation through graph ──
  // Returns predicted values for all nodes after setting variable=value
  propagate(variable: string, value: number): Record<string, number> {
    const state: Record<string, number> = {};
    this.nodes.forEach((n, id) => { state[id] = n.value; });
    state[variable] = value;

    // Topological propagation (BFS from intervention point)
    const visited = new Set<string>([variable]);
    const queue   = [variable];

    while (queue.length > 0) {
      const current = queue.shift()!;
      const outgoing = this.edges.filter(e => e.from === current);

      for (const edge of outgoing) {
        const prevValue = state[edge.to] ?? 0;
        // Causal update: downstream node shifts by edge weight * upstream delta
        const upstream_delta = state[current] - (this.nodes.get(current)?.value ?? 0);
        state[edge.to] = prevValue + edge.weight * upstream_delta * edge.alphaTrust;

        if (!visited.has(edge.to)) {
          visited.add(edge.to);
          queue.push(edge.to);
        }
      }
    }

    return state;
  }

  // ── Mean edge alpha trust ──
  get alphaMean(): number {
    if (this.edges.length === 0) return 0.05;
    return this.edges.reduce((s, e) => s + e.alphaTrust, 0) / this.edges.length;
  }

  // ── Uncertainty of an edge (for epistemic scoring) ──
  edgeUncertainty(from: string, to: string): number {
    const e = this.edges.find(e => e.from === from && e.to === to);
    if (!e) return 1.0; // completely unknown = max uncertainty
    return (1 - Math.abs(e.weight)) * (1 - e.alphaTrust);
  }

  // ── Shallow clone ──
  clone(): CausalGraph {
    const g    = new CausalGraph();
    this.nodes.forEach((n, id) => g.nodes.set(id, { ...n }));
    g.edges    = this.edges.map(e => ({ ...e }));
    return g;
  }

  toJSON() {
    return {
      nodes: Array.from(this.nodes.values()),
      edges: this.edges,
      size:  this.size,
    };
  }
}
