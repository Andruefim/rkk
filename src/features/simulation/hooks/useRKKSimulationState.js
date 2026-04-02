import { useState, useEffect, useRef, useCallback } from "react";
import { useRKKStream } from "../../../hooks/useRKKStream";
import { normalizeFrame } from "../normalize.js";

const API = "http://localhost:8000";

export function useRKKSimulationState() {
  const { frame: wsFrame, connected, setSpeed: wsSetSpeed } = useRKKStream();
  const frameRef = useRef(wsFrame);
  frameRef.current = wsFrame;

  const [speed, setSpeedLocal] = useState(1);
  const [ui, setUI] = useState(() => normalizeFrame(wsFrame));
  const [activePanel, setActivePanel] = useState(null);
  const [seedText, setSeedText] = useState(
    '[\n  {"from_": "Temp", "to": "Pressure", "weight": 0.8}\n]'
  );
  const [seedAgent, setSeedAgent] = useState(0);
  const [seedStatus, setSeedStatus] = useState("");
  const [ragLoading, setRagLoading] = useState(false);
  const [ragResults, setRagResults] = useState([]);
  const [demonStats, setDemonStats] = useState(null);

  const setSpeed = useCallback(
    (s) => {
      setSpeedLocal(s);
      if (connected) wsSetSpeed(s);
    },
    [connected, wsSetSpeed]
  );

  useEffect(() => {
    setUI(normalizeFrame(wsFrame));
  }, [wsFrame]);

  useEffect(() => {
    if (!connected) return;
    const iv = setInterval(async () => {
      try {
        const d = await fetch(`${API}/demon/stats`).then((r) => r.json());
        setDemonStats(d);
      } catch {
        /* offline */
      }
    }, 3000);
    return () => clearInterval(iv);
  }, [connected]);

  const injectSeeds = useCallback(async () => {
    try {
      const edges = JSON.parse(seedText);
      const res = await fetch(`${API}/inject-seeds`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ agent_id: seedAgent, edges, source: "manual" }),
      });
      const d = await res.json();
      setSeedStatus(`✓ ${d.injected} edges → ${d.agent}`);
    } catch (e) {
      setSeedStatus(`✗ ${e.message}`);
    }
  }, [seedText, seedAgent]);

  const ragAutoSeed = useCallback(async () => {
    setRagLoading(true);
    setSeedStatus("⏳ Auto-seeding…");
    try {
      const res = await fetch(`${API}/rag/auto-seed-all`, { method: "POST" });
      const d = await res.json();
      setRagResults(d.results ?? []);
      setSeedStatus(
        `✓ ${d.results?.reduce((s, r) => s + r.injected, 0) ?? 0} edges total`
      );
    } catch (e) {
      setSeedStatus(`✗ ${e.message}`);
    } finally {
      setRagLoading(false);
    }
  }, []);

  return {
    ui,
    connected,
    speed,
    setSpeed,
    activePanel,
    setActivePanel,
    seedText,
    setSeedText,
    seedAgent,
    setSeedAgent,
    seedStatus,
    ragLoading,
    ragResults,
    demonStats,
    injectSeeds,
    ragAutoSeed,
    frameRef,
  };
}
