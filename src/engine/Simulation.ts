import { RKKAgent } from "./RKKAgent";
import { Environment } from "./Environment";
import type { SimEvent, AgentSnapshot } from "./types";

// ─── Phase thresholds (based on mean discovery rate across agents) ──────────
const PHASE_THRESHOLDS = [0, 0.20, 0.40, 0.62, 0.80, 0.95];

export interface SimState {
  agents:    AgentSnapshot[];
  phase:     number;
  entropy:   number;   // 1 - mean discovery rate
  events:    SimEvent[];
  tick:      number;
  tomLinks:  Array<{ a: number; b: number; strength: number }>;
}

export class Simulation {
  agents:     RKKAgent[];
  envs:       Environment[];
  events:     SimEvent[] = [];
  phase       = 1;
  tick        = 0;

  // Demon state
  demonCooldown = 0;
  demonEnergy   = 1.0;
  demonTarget   = 0;

  // ToM: agent i models agent j if they share discovered edges
  tomMatrix: number[][] = [[0,0,0],[0,0,0],[0,0,0]];

  constructor() {
    // Three agents, three different environment presets (Seed Diversity)
    const presets: Array<"physics" | "chemistry" | "logic"> = ["physics", "chemistry", "logic"];
    this.envs   = presets.map(p => new Environment(p));
    this.agents = this.envs.map((env, i) => new RKKAgent(i, ["Nova","Aether","Lyra"][i], env));
  }

  // ── Single simulation tick ────────────────────────────────────────────────
  tick_step(): SimState {
    this.tick++;

    // 1. Each agent performs one intervention step
    for (const agent of this.agents) {
      try {
        const result = agent.step();

        if (result.updatedEdges.length > 0) {
          this._addEvent(
            `${agent.name}: do(${result.intervention.variable}=${result.intervention.value.toFixed(2)}) → CG ${result.compressionDelta >= 0 ? "+" : ""}${result.compressionDelta.toFixed(2)}`,
            ["#00ff99","#0099ff","#ff9900"][agent.id],
            "discovery"
          );
        }
      } catch {
        // Agent has no valid interventions this tick
      }
    }

    // 2. Demon attacks (every ~80 ticks, with energy cost)
    this.demonCooldown = Math.max(0, this.demonCooldown - 1);
    if (this.demonCooldown === 0 && this.demonEnergy > 0.15) {
      this.demonEnergy    -= 0.2;
      this.demonCooldown   = 80 + Math.floor(Math.random() * 40);
      this.demonTarget     = (this.demonTarget + 1) % 3;

      const target = this.agents[this.demonTarget];
      target.demonDisrupt();

      this._addEvent(
        `⚠ Demon disrupts ${target.name} — Φ ↓, α corruption`,
        "#ff2244",
        "demon"
      );
    }
    this.demonEnergy = Math.min(1, this.demonEnergy + 0.002);

    // 3. Phase IV: other agents protect disrupted agent (ΔAutonomy ≥ 0)
    if (this.phase >= 4 && this.demonCooldown > 70) {
      for (let i = 0; i < this.agents.length; i++) {
        if (i !== this.demonTarget) {
          // Protective action: reinforce target's best edges
          const target    = this.agents[this.demonTarget];
          const strongest = [...target.graph.edges]
            .sort((a, b) => b.alphaTrust - a.alphaTrust)
            .slice(0, 2);
          strongest.forEach(e => {
            e.alphaTrust = Math.min(0.98, e.alphaTrust + 0.04);
          });
        }
      }
    }

    // 4. Theory of Mind: agents share compressed knowledge when close in discovery
    if (this.tick % 30 === 0 && this.phase >= 3) {
      this._updateToM();
    }

    // 5. Phase progression
    const meanDiscovery = this.agents.reduce((s, a) => s + a.discoveryRate, 0) / this.agents.length;
    const entropy       = Math.round((1 - meanDiscovery) * 100);
    const newPhase      = PHASE_THRESHOLDS.findLastIndex(t => meanDiscovery >= t) + 1;

    if (newPhase > this.phase && newPhase <= 5) {
      this.phase = newPhase;
      const labels = ["","Causal Crib","Robotic Explorer","Social Sandbox","Value Lock","Open Reality"];
      this._addEvent(
        `⬆ Phase ${newPhase}: ${labels[newPhase]} unlocked`,
        "#ffcc00",
        "phase"
      );
    }

    // 6. Trim events
    if (this.events.length > 12) this.events = this.events.slice(0, 12);

    return {
      agents:   this.agents.map(a => a.snapshot()),
      phase:    this.phase,
      entropy,
      events:   [...this.events],
      tick:     this.tick,
      tomLinks: this._getToMLinks(),
    };
  }

  private _updateToM() {
    // Theory of Mind: if two agents have discovered the same edges, they model each other
    for (let i = 0; i < this.agents.length; i++) {
      for (let j = i + 1; j < this.agents.length; j++) {
        // Shared discovery rate (both using same env type for simplicity)
        const phiI     = this.agents[i].snapshot().phi;
        const phiJ     = this.agents[j].snapshot().phi;
        const strength = (phiI + phiJ) / 2;

        this.tomMatrix[i][j] = strength;
        this.tomMatrix[j][i] = strength;

        if (strength > 0.6 && Math.random() < 0.3) {
          this._addEvent(
            `ToM: ${this.agents[i].name} ↔ ${this.agents[j].name} (Φ-sync ${(strength * 100).toFixed(0)}%)`,
            "#004488",
            "tom"
          );
        }
      }
    }
  }

  private _getToMLinks() {
    const links = [];
    for (let i = 0; i < 3; i++) {
      for (let j = i + 1; j < 3; j++) {
        if (this.tomMatrix[i][j] > 0.3) {
          links.push({ a: i, b: j, strength: this.tomMatrix[i][j] });
        }
      }
    }
    return links;
  }

  private _addEvent(text: string, color: string, type: SimEvent["type"]) {
    this.events.unshift({ text, color, timestamp: this.tick, type });
  }
}
