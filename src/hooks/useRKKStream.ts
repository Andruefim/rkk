// src/hooks/useRKKStream.ts
import { useState, useEffect, useRef, useCallback } from "react";

export interface EdgeData {
  from_:              string;
  to:                 string;
  weight:             number;
  alpha_trust:        number;
  intervention_count: number;
}

export interface NOTEARSInfo {
  steps: number;
  loss:  number;
  h_W:   number;
  l_int: number;
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
  peak_discovery_rate: number;
  h_W:                 number;   // DAG constraint: 0 = perfect DAG
  notears:             NOTEARSInfo | null;
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

const DEFAULT_FRAME: StreamFrame = {
  tick: 0, phase: 1, entropy: 100,
  agents: [{
    id: 0, name: "Nova",
    env_type: "humanoid",
    activation: "relu",
    graph_mdl: 0, compression_gain: 0, alpha_mean: 0.05,
    phi: 0.1, node_count: 6, edge_count: 0,
    total_interventions: 0, last_do: "—",
    discovery_rate: 0, peak_discovery_rate: 0,
    h_W: 0, notears: null, edges: [],
  }],
  demon: { energy: 1, cooldown: 0, last_target: 0, last_action_complexity: 0 },
  tom_links: [], events: [], graph_deltas: {},
};

export function useRKKStream(wsUrl = "ws://localhost:8000/ws/causal-stream") {
  const [frame,      setFrame]      = useState<StreamFrame>(DEFAULT_FRAME);
  const [connected,  setConnected]  = useState(false);
  const [speed,      setSpeedState] = useState(1);
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

      ws.onopen  = () => { setConnected(true);  console.log("[RKK] WS connected"); };
      ws.onclose = () => {
        setConnected(false);
        console.log("[RKK] WS closed, reconnecting…");
        reconnectTimer = setTimeout(connect, 2000);
      };
      ws.onerror = () => ws.close();

      ws.onmessage = (ev) => {
        try {
          const data = JSON.parse(ev.data) as StreamFrame;
          if (!data || typeof data !== "object") return;
          setFrame(prev => {
            const raw = Array.isArray(data.agents) ? data.agents : prev.agents;
            const gd = data.graph_deltas ?? {};
            const agents = raw.map((a, i) => ({
              ...a,
              edges: gd[i as keyof typeof gd] ?? prev.agents[i]?.edges ?? a.edges,
            }));
            return { ...data, agents };
          });
        } catch (e) {
          console.error("[RKK] Parse error", e);
        }
      };
    }

    connect();
    return () => { clearTimeout(reconnectTimer); ws?.close(); };
  }, [wsUrl]);

  return { frame, connected, speed, setSpeed, reset };
}