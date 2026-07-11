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

export interface StreamScene {
  skeleton?: Array<{ x?: number; y?: number; z?: number }>;
  static_geometry?: unknown[];
  ankleQuats?: Array<{ x?: number; y?: number; z?: number; w?: number }>;
  cubes?: Array<{ x?: number; y?: number; z?: number }>;
  fallen?: boolean;
  [key: string]: unknown;
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
  scene?:        StreamScene;
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

type WsMetaFrame = StreamFrame & { _ws_hello?: boolean; _ws_recovery?: boolean };

const FRAME_THROTTLE_MS = 100;

function mergeStreamFrame(prev: StreamFrame, data: WsMetaFrame): StreamFrame {
  if (data._ws_hello || data._ws_recovery) {
    return {
      ...prev,
      tick: data.tick ?? prev.tick,
      phase: data.phase ?? prev.phase,
      entropy: data.entropy ?? prev.entropy,
      events: data.events?.length ? data.events : prev.events,
    };
  }
  const raw = Array.isArray(data.agents) ? data.agents : prev.agents;
  const gd = data.graph_deltas ?? {};
  const agents = raw.map((a, i) => ({
    ...a,
    edges: gd[i as keyof typeof gd] ?? prev.agents[i]?.edges ?? a.edges,
  }));
  const prevScene = prev.scene ?? {};
  const nextScene = data.scene ?? {};
  const nextSk = nextScene.skeleton;
  const scene = {
    ...prevScene,
    ...nextScene,
    static_geometry:
      nextScene.static_geometry ?? prevScene.static_geometry,
    skeleton:
      nextSk && nextSk.length >= 3 ? nextSk : prevScene.skeleton,
  };
  return { ...data, agents, scene };
}

export function useRKKStream(wsUrl = "ws://localhost:8000/ws/causal-stream") {
  const [frame,      setFrame]      = useState<StreamFrame>(DEFAULT_FRAME);
  const [connected,  setConnected]  = useState(false);
  const [speed,      setSpeedState] = useState(1);
  const wsRef = useRef<WebSocket | null>(null);
  /** Latest merged WS frame — updated every onmessage for 3D (bypasses React throttle). */
  const rawFrameRef = useRef<StreamFrame>(DEFAULT_FRAME);
  const lastSetFrameMsRef = useRef(0);
  const pendingFrameRef = useRef<StreamFrame | null>(null);
  const throttleTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const scheduleThrottledSetFrame = useCallback((merged: StreamFrame) => {
    const now = Date.now();
    const elapsed = now - lastSetFrameMsRef.current;
    if (elapsed >= FRAME_THROTTLE_MS) {
      lastSetFrameMsRef.current = now;
      setFrame(merged);
      pendingFrameRef.current = null;
      if (throttleTimerRef.current !== null) {
        clearTimeout(throttleTimerRef.current);
        throttleTimerRef.current = null;
      }
      return;
    }
    pendingFrameRef.current = merged;
    if (throttleTimerRef.current !== null) return;
    throttleTimerRef.current = setTimeout(() => {
      throttleTimerRef.current = null;
      const pending = pendingFrameRef.current;
      if (!pending) return;
      pendingFrameRef.current = null;
      lastSetFrameMsRef.current = Date.now();
      setFrame(pending);
    }, FRAME_THROTTLE_MS - elapsed);
  }, []);

  const setSpeed = useCallback((s: number) => {
    setSpeedState(s);
    wsRef.current?.send(JSON.stringify({ cmd: "set_speed", value: s }));
  }, []);

  const reset = useCallback(() => {
    wsRef.current?.send(JSON.stringify({ cmd: "reset" }));
  }, []);

  useEffect(() => {
    let cancelled = false;
    let ws: WebSocket | null = null;
    let reconnectTimer: ReturnType<typeof setTimeout> | undefined;
    let attempt = 0;

    function connect() {
      if (cancelled) return;
      ws = new WebSocket(wsUrl);
      wsRef.current = ws;

      ws.onopen = () => {
        attempt = 0;
        if (!cancelled) {
          setConnected(true);
          console.log("[RKK] WS connected");
        }
      };
      ws.onclose = () => {
        setConnected(false);
        if (cancelled) return;
        attempt += 1;
        const delay = Math.min(8000, 1000 + attempt * 800);
        console.log(`[RKK] WS closed, reconnect in ${delay}ms`);
        reconnectTimer = setTimeout(connect, delay);
      };
      ws.onerror = () => {
        console.warn("[RKK] WS error (waiting for close/reconnect)");
      };

      ws.onmessage = (ev) => {
        try {
          const data = JSON.parse(ev.data) as WsMetaFrame;
          if (!data || typeof data !== "object") return;
          const merged = mergeStreamFrame(rawFrameRef.current, data);
          rawFrameRef.current = merged;
          scheduleThrottledSetFrame(merged);
        } catch (e) {
          console.error("[RKK] Parse error", e);
        }
      };
    }

    connect();
    return () => {
      cancelled = true;
      clearTimeout(reconnectTimer);
      if (throttleTimerRef.current !== null) {
        clearTimeout(throttleTimerRef.current);
        throttleTimerRef.current = null;
      }
      ws?.close();
      wsRef.current = null;
    };
  }, [wsUrl, scheduleThrottledSetFrame]);

  return { frame, rawFrameRef, connected, speed, setSpeed, reset };
}
