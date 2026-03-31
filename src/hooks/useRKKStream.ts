// src/hooks/useRKKStream.ts
// Улучшение 1: WebSocket вместо fetch — получаем дельты графа в реальном времени

import { useState, useEffect, useRef, useCallback } from "react";

// ── Types (matching Python schemas) ──────────────────────────────────────────
export interface EdgeData {
  from_:              string;
  to:                 string;
  weight:             number;
  alpha_trust:        number;
  intervention_count: number;
}

export interface AgentData {
  id:                  number;
  name:                string;
  env_type:            string;
  activation:          string;
  graph_mdl:           number;
  compression_gain:    number;
  alpha_mean:          number;
  phi:                 number;
  node_count:          number;
  edge_count:          number;
  total_interventions: number;
  last_do:             string;
  discovery_rate:      number;
  edges:               EdgeData[];
}

export interface DemonData {
  energy:                  number;
  cooldown:                number;
  last_target:             number;
  last_action_complexity:  number;
}

export interface ToMLink {
  a: number; b: number; strength: number;
}

export interface SimEventData {
  tick:  number;
  text:  string;
  color: string;
  type:  string;
}

export interface StreamFrame {
  tick:          number;
  phase:         number;
  entropy:       number;
  agents:        AgentData[];
  demon:         DemonData;
  tom_links:     ToMLink[];
  events:        SimEventData[];
  graph_deltas:  Record<number, EdgeData[]>;
}

// ── Default state ─────────────────────────────────────────────────────────────
const DEFAULT_FRAME: StreamFrame = {
  tick: 0, phase: 1, entropy: 100,
  agents: [0, 1, 2].map(i => ({
    id: i, name: ["Nova","Aether","Lyra"][i],
    env_type: ["physics","chemistry","logic"][i],
    activation: ["relu","gelu","tanh"][i],
    graph_mdl: 0, compression_gain: 0, alpha_mean: 0.05,
    phi: 0.1, node_count: 6, edge_count: 0,
    total_interventions: 0, last_do: "—",
    discovery_rate: 0, edges: [],
  })),
  demon: { energy: 1, cooldown: 0, last_target: 0, last_action_complexity: 0 },
  tom_links: [], events: [], graph_deltas: {},
};

// ── Hook ──────────────────────────────────────────────────────────────────────
export function useRKKStream(wsUrl = "ws://localhost:8000/ws/causal-stream") {
  const [frame,     setFrame]     = useState<StreamFrame>(DEFAULT_FRAME);
  const [connected, setConnected] = useState(false);
  const [speed,     setSpeedState] = useState(1);
  const wsRef = useRef<WebSocket | null>(null);

  const setSpeed = useCallback((s: number) => {
    setSpeedState(s);
    wsRef.current?.send(JSON.stringify({ cmd: "set_speed", value: s }));
  }, []);

  const reset = useCallback(() => {
    wsRef.current?.send(JSON.stringify({ cmd: "reset" }));
  }, []);

  useEffect(() => {
    let ws: WebSocket;
    let reconnectTimer: ReturnType<typeof setTimeout>;

    function connect() {
      ws = new WebSocket(wsUrl);
      wsRef.current = ws;

      ws.onopen = () => {
        setConnected(true);
        console.log("[RKK] WebSocket connected");
      };

      ws.onmessage = (ev) => {
        try {
          const data = JSON.parse(ev.data) as StreamFrame;
          setFrame(prev => {
            // Merge graph deltas into agent edges
            const agents = data.agents.map((a, i) => {
              if (data.graph_deltas[i]) {
                return { ...a, edges: data.graph_deltas[i] };
              }
              // No delta → keep previous edges (bandwidth saving)
              return { ...a, edges: prev.agents[i]?.edges ?? a.edges };
            });
            return { ...data, agents };
          });
        } catch (e) {
          console.error("[RKK] Parse error", e);
        }
      };

      ws.onclose = () => {
        setConnected(false);
        console.log("[RKK] WebSocket closed, reconnecting in 2s…");
        reconnectTimer = setTimeout(connect, 2000);
      };

      ws.onerror = (e) => {
        console.error("[RKK] WebSocket error", e);
        ws.close();
      };
    }

    connect();

    return () => {
      clearTimeout(reconnectTimer);
      ws?.close();
    };
  }, [wsUrl]);

  return { frame, connected, speed, setSpeed, reset };
}
