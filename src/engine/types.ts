// ─── RKK v5 Core Types ────────────────────────────────────────────────────────

/** A node in the causal graph. alphaTrust: 0 = text prior, 1 = sandbox-confirmed */
export interface CausalNode {
  id: string;
  value: number;
  alphaTrust: number;
}

/** A directed causal edge. weight = causal strength [-1, 1] */
export interface CausalEdge {
  from: string;
  to: string;
  weight: number;
  alphaTrust: number;
  interventionCount: number; // how many do() tests confirmed/updated this edge
}

/** Result of a do() intervention */
export interface InterventionResult {
  intervention: { variable: string; value: number };
  predicted: Record<string, number>;
  observed: Record<string, number>;
  predictionError: number;      // MSE between predicted and observed
  compressionDelta: number;     // |G_before| - |G_after|, positive = learning happened
  updatedEdges: string[];       // which edges changed
}

/** Intrinsic drive score for an action */
export interface EpistemicScore {
  variable: string;
  value: number;
  expectedInfoGain: number;     // E[IG] — how surprising this intervention might be
  uncertaintyReduction: number;
}

/** Agent state snapshot for UI */
export interface AgentSnapshot {
  id: number;
  name: string;
  graphSize: number;       // MDL approximation of |G|
  compressionGain: number; // running d|G|/dt
  alphaMean: number;       // mean trust across all edges
  nodeCount: number;
  edgeCount: number;
  phi: number;             // autonomy proxy
  totalInterventions: number;
  lastAction: string;
}

/** Simulation event for the event log */
export interface SimEvent {
  text: string;
  color: string;
  timestamp: number;
  type: "discovery" | "phase" | "demon" | "tom" | "value";
}
