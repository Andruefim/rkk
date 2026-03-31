import { useState, useEffect, useRef } from "react";
import * as THREE from "three";
import { useRKKStream } from "../../hooks/useRKKStream";
import { Simulation }   from "../../engine/Simulation";

// ─── CONSTANTS ────────────────────────────────────────────────────────────────
const WORLD            = 14;
const AGENT_COLORS_HEX = [0x00ff99, 0x0099ff, 0xff9900];
const AGENT_COLORS_CSS = ["#00ff99", "#0099ff", "#ff9900"];
const AGENT_NAMES      = ["Nova", "Aether", "Lyra"];
const DEMON_COLOR      = 0xff2244;
const PHASE_NAMES      = ["", "Causal Crib", "Robotic Explorer", "Social Sandbox", "Value Lock", "Open Reality"];
const GRAPH_LINES      = 14;

// ─── Adapter: normalize backend frame OR local snapshot to common shape ───────
function normalizeFrame(raw) {
  // Backend frame already has agent.phi, agent.alpha_mean etc. (snake_case)
  // Local TS snapshot has agent.alphaMean, agent.phi etc. (camelCase)
  const agents = (raw.agents ?? []).map(a => ({
    id:                  a.id                  ?? 0,
    name:                a.name                ?? "?",
    envType:             a.env_type            ?? a.envType ?? "—",
    activation:          a.activation          ?? "relu",
    graphMdl:            a.graph_mdl           ?? a.graphSize ?? 0,
    compressionGain:     a.compression_gain    ?? a.compressionGain ?? 0,
    alphaMean:           a.alpha_mean          ?? a.alphaMean ?? 0.05,
    phi:                 a.phi                 ?? 0.1,
    nodeCount:           a.node_count          ?? a.nodeCount ?? 0,
    edgeCount:           a.edge_count          ?? a.edgeCount ?? 0,
    totalInterventions:  a.total_interventions ?? a.totalInterventions ?? 0,
    lastDo:              a.last_do             ?? a.lastAction ?? "—",
    discoveryRate:       a.discovery_rate      ?? 0,
    edges:               a.edges               ?? [],
  }));

  return {
    tick:     raw.tick     ?? 0,
    phase:    raw.phase    ?? 1,
    entropy:  raw.entropy  ?? 100,
    agents,
    demon:    raw.demon    ?? { energy: 1, cooldown: 0, last_action_complexity: 0 },
    tomLinks: raw.tomLinks ?? raw.tom_links ?? [],
    events:   raw.events   ?? [],
  };
}

// ─── Local fallback simulation (TypeScript engine) ────────────────────────────
function useLocalSim(active) {
  const simRef   = useRef(null);
  const stateRef = useRef(null);

  if (active && !simRef.current) {
    simRef.current   = new Simulation();
    stateRef.current = simRef.current.tick_step();
  }

  return { simRef, stateRef };
}

// ─── MAIN COMPONENT ───────────────────────────────────────────────────────────
export default function RKKv5() {
  const mountRef = useRef(null);
  const rafRef   = useRef(null);

  // Try WebSocket backend first
  const { frame: wsFrame, connected, speed: wsSpeed, setSpeed: wsSetSpeed } = useRKKStream();

  // Local TS fallback (used when WS not connected)
  const { simRef, stateRef } = useLocalSim(!connected);

  const [speed,   setSpeedLocal] = useState(1);
  const [ui,      setUI]         = useState(() => normalizeFrame(wsFrame));
  const [backend, setBackend]    = useState(false);

  // Speed control: routes to WS or local
  const setSpeed = (s) => {
    setSpeedLocal(s);
    if (connected) wsSetSpeed(s);
  };

  // Update UI from WS when connected
  useEffect(() => {
    if (connected) {
      setBackend(true);
      setUI(normalizeFrame(wsFrame));
    }
  }, [wsFrame, connected]);

  // ── Three.js setup ──────────────────────────────────────────────────────────
  useEffect(() => {
    const mount = mountRef.current;
    if (!mount) return;

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(mount.clientWidth, mount.clientHeight);
    mount.appendChild(renderer.domElement);

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x010810);
    scene.fog = new THREE.FogExp2(0x010810, 0.022);

    const camera = new THREE.PerspectiveCamera(55, mount.clientWidth / mount.clientHeight, 0.1, 200);
    camera.position.set(0, 15, 28);
    camera.lookAt(0, 1, 0);

    scene.add(new THREE.AmbientLight(0x112233, 1.8));
    const sun = new THREE.DirectionalLight(0x334488, 1.2);
    sun.position.set(5, 10, 5);
    scene.add(sun);

    scene.add(new THREE.GridHelper(WORLD * 2 + 4, 30, 0x0a1e38, 0x050f1e));

    // ── Agents ──
    const agentGroups = AGENT_COLORS_HEX.map((col, i) => {
      const g    = new THREE.Group();
      const body = new THREE.Mesh(
        new THREE.SphereGeometry(0.6, 22, 22),
        new THREE.MeshPhongMaterial({ color: col, emissive: col, emissiveIntensity: 0.35, transparent: true, opacity: 0.92 })
      );
      const ring = new THREE.Mesh(
        new THREE.TorusGeometry(0.95, 0.05, 8, 44),
        new THREE.MeshBasicMaterial({ color: col, transparent: true, opacity: 0.4 })
      );
      ring.rotation.x = Math.PI / 2;
      g.add(body); g.add(ring);
      g.add(new THREE.PointLight(col, 0.7, 5));
      g.position.set((i - 1) * 6 + (Math.random() - 0.5), 1, (Math.random() - 0.5) * 3);
      g.userData = { body, ring, vel: new THREE.Vector3((Math.random() - 0.5) * 0.04, 0, (Math.random() - 0.5) * 0.04) };
      scene.add(g);
      return g;
    });

    // ── Demon ──
    const demon = new THREE.Mesh(
      new THREE.OctahedronGeometry(0.8, 0),
      new THREE.MeshPhongMaterial({ color: DEMON_COLOR, emissive: 0xff0022, emissiveIntensity: 0.7, transparent: true, opacity: 0.9 })
    );
    demon.add(new THREE.PointLight(DEMON_COLOR, 0.9, 6));
    demon.position.set(WORLD - 2, 1.2, WORLD - 2);
    demon.userData = { vel: new THREE.Vector3(-0.035, 0, -0.025) };
    scene.add(demon);

    // ── Causal graph orbit nodes ──
    const orbitNodes = agentGroups.map((_, ai) =>
      Array.from({ length: GRAPH_LINES }, () => {
        const m = new THREE.Mesh(
          new THREE.SphereGeometry(0.1, 6, 6),
          new THREE.MeshBasicMaterial({ color: AGENT_COLORS_HEX[ai], transparent: true, opacity: 0 })
        );
        scene.add(m);
        return m;
      })
    );

    // ── Causal edge lines ──
    const causalLines = agentGroups.map((_, ai) =>
      Array.from({ length: GRAPH_LINES }, () => {
        const geom = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), new THREE.Vector3()]);
        const line = new THREE.Line(geom, new THREE.LineBasicMaterial({ color: AGENT_COLORS_HEX[ai], transparent: true, opacity: 0 }));
        scene.add(line);
        return line;
      })
    );

    // ── Weight indicator lines (signed weight → color) ──
    // Positive edges: agent color, Negative: red
    const weightLines = agentGroups.map((_, ai) =>
      Array.from({ length: GRAPH_LINES }, () => {
        const geom = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), new THREE.Vector3()]);
        const line = new THREE.Line(geom, new THREE.LineBasicMaterial({ color: 0x884400, transparent: true, opacity: 0 }));
        scene.add(line);
        return line;
      })
    );

    // ── ToM lines ──
    const tomLines = [];
    for (let a = 0; a < 3; a++) {
      for (let b = a + 1; b < 3; b++) {
        const geom = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), new THREE.Vector3()]);
        const line = new THREE.Line(geom, new THREE.LineBasicMaterial({ color: 0x003366, transparent: true, opacity: 0 }));
        line.userData = { a, b };
        scene.add(line);
        tomLines.push(line);
      }
    }

    // ── Particles ──
    const pCount = 400;
    const pPos   = new Float32Array(pCount * 3);
    for (let i = 0; i < pCount; i++) {
      pPos[i * 3]     = (Math.random() - 0.5) * 44;
      pPos[i * 3 + 1] = Math.random() * 14;
      pPos[i * 3 + 2] = (Math.random() - 0.5) * 44;
    }
    const pGeom = new THREE.BufferGeometry();
    pGeom.setAttribute("position", new THREE.BufferAttribute(pPos, 3));
    scene.add(new THREE.Points(pGeom, new THREE.PointsMaterial({ color: 0x003399, size: 0.07, transparent: true, opacity: 0.35 })));

    // ── Animation loop ──
    let frame    = 0;
    let camAngle = 0;
    let uiTick   = 0;

    // currentState держит последнее нормализованное состояние
    // обновляется либо из WS (через setUI), либо из локального движка
    let localSpeedRef = { val: 1 };

    function loop() {
      rafRef.current = requestAnimationFrame(loop);
      frame++;
      uiTick++;

      // ── Если нет WS — тикаем локально ──
      let state = null;
      if (!connected && simRef.current) {
        if (frame % Math.max(1, Math.round(8 / localSpeedRef.val)) === 0) {
          stateRef.current = simRef.current.tick_step();
        }
        state = normalizeFrame(stateRef.current);
      }

      // ── Camera orbit ──
      camAngle += 0.0012;
      camera.position.x = Math.sin(camAngle) * 28;
      camera.position.z = Math.cos(camAngle) * 28;
      camera.lookAt(0, 2, 0);

      // ── Pull latest UI state ──
      // В WS-режиме state приходит через useEffect → setUI
      // В локальном режиме берём только что вычисленный state
      const uiData = state ?? null;
      if (!uiData && !connected) return;

      const displayState = uiData;

      if (displayState) {
        agentGroups.forEach((g, i) => {
          const snap = displayState.agents[i];
          if (!snap) return;

          const vel = g.userData.vel;
          g.position.add(vel);
          if (g.position.x > WORLD || g.position.x < -WORLD) vel.x *= -1;
          if (g.position.z > WORLD || g.position.z < -WORLD) vel.z *= -1;
          g.position.y = 1 + Math.sin(frame * 0.05 + i * 2.1) * 0.15;

          if (frame % (120 + i * 30) === 0) {
            g.userData.vel = new THREE.Vector3(
              (Math.random() - 0.5) * 0.05, 0, (Math.random() - 0.5) * 0.05
            );
          }

          // Ring speed reflects alpha trust
          g.userData.ring.rotation.z += 0.015 + snap.alphaMean * 0.03;
          g.userData.ring.rotation.y += 0.008;

          // Body glow reflects compression gain
          const cgGlow = Math.max(0, snap.compressionGain);
          g.userData.body.material.emissiveIntensity = 0.2 + cgGlow * 0.12 + Math.sin(frame * 0.07 + i) * 0.07;

          // Orbit nodes: count = real edge count from graph
          const visCount = Math.min(snap.edgeCount + 2, GRAPH_LINES);
          orbitNodes[i].forEach((node, k) => {
            if (k < visCount) {
              const angle = (k / visCount) * Math.PI * 2 + frame * 0.016;
              const r     = 1.6 + (k % 2) * 0.5;
              node.position.set(
                g.position.x + Math.cos(angle) * r,
                g.position.y + Math.sin(frame * 0.04 + k * 0.8) * 0.35 + 0.2,
                g.position.z + Math.sin(angle) * r
              );
              node.material.opacity = 0.4 + snap.alphaMean * 0.5;
            } else {
              node.material.opacity = 0;
            }
          });

          // Causal lines with weight-based color
          const edges = snap.edges ?? [];
          causalLines[i].forEach((line, k) => {
            if (k < visCount - 1) {
              const na = orbitNodes[i][k].position;
              const nb = orbitNodes[i][(k + 1) % visCount].position;
              const pa = line.geometry.attributes.position;
              pa.setXYZ(0, na.x, na.y, na.z);
              pa.setXYZ(1, nb.x, nb.y, nb.z);
              pa.needsUpdate = true;

              // Color by edge weight sign (from actual edges if available)
              const edge       = edges[k];
              const isNegative = edge && edge.weight < 0;
              line.material.color.set(isNegative ? 0xff4422 : AGENT_COLORS_HEX[i]);
              line.material.opacity = 0.3 + snap.alphaMean * 0.45;
            } else {
              line.material.opacity = 0;
            }
          });
        });

        // ── Demon ──
        const target = agentGroups[displayState.tick % 3];
        const chase  = target.position.clone().sub(demon.position).normalize().multiplyScalar(0.032);
        demon.userData.vel.lerp(chase, 0.05);
        demon.position.add(demon.userData.vel);
        if (demon.position.x >  WORLD || demon.position.x < -WORLD) demon.userData.vel.x *= -1;
        if (demon.position.z >  WORLD || demon.position.z < -WORLD) demon.userData.vel.z *= -1;
        demon.position.y = 1.2 + Math.sin(frame * 0.08) * 0.3;
        demon.rotation.y += 0.045;
        demon.rotation.x += 0.028;

        // ── ToM lines ──
        tomLines.forEach(line => {
          const link = displayState.tomLinks.find(l => l.a === line.userData.a && l.b === line.userData.b);
          const pa   = agentGroups[line.userData.a].position;
          const pb   = agentGroups[line.userData.b].position;
          const p    = line.geometry.attributes.position;
          p.setXYZ(0, pa.x, pa.y, pa.z);
          p.setXYZ(1, pb.x, pb.y, pb.z);
          p.needsUpdate = true;
          line.material.opacity = link ? link.strength * 0.65 : 0;
        });

        // ── UI update every 24 frames (local mode only) ──
        if (!connected && uiTick >= 24) {
          uiTick = 0;
          setUI(displayState);
        }
      }

      renderer.render(scene, camera);
    }

    // Expose speed ref so loop can read it without closure staleness
    const speedProxy = { val: 1 };
    Object.assign(localSpeedRef, speedProxy);

    loop();

    function onResize() {
      if (!mount) return;
      camera.aspect = mount.clientWidth / mount.clientHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(mount.clientWidth, mount.clientHeight);
    }
    window.addEventListener("resize", onResize);

    return () => {
      cancelAnimationFrame(rafRef.current);
      window.removeEventListener("resize", onResize);
      if (mount.contains(renderer.domElement)) mount.removeChild(renderer.domElement);
      renderer.dispose();
    };
  }, []); // setup once — state flows through refs/WS

  // ─── UI ───────────────────────────────────────────────────────────────────
  const mono = { fontFamily: "'Courier New', monospace" };

  return (
    <div style={{ position: "relative", width: "100%", height: "100vh", background: "#010810", overflow: "hidden", ...mono }}>
      <div ref={mountRef} style={{ position: "absolute", inset: 0 }} />

      {/* ── Connection status badge ── */}
      <div style={{
        position: "absolute", top: 14, right: 14,
        background: connected ? "rgba(0,40,10,0.9)" : "rgba(30,10,0,0.9)",
        border: `1px solid ${connected ? "#00aa44" : "#aa4400"}`,
        padding: "4px 10px", borderRadius: 2, fontSize: 9,
        color: connected ? "#00ff88" : "#ff8844",
      }}>
        {connected ? "● PYTHON BACKEND" : "○ LOCAL ENGINE"}
      </div>

      {/* ── Header ── */}
      <div style={{
        position: "absolute", top: 14, left: "50%", transform: "translateX(-50%)",
        background: "rgba(0,12,28,0.9)", border: "1px solid #0a2a44",
        padding: "8px 22px", textAlign: "center", borderRadius: 2,
        boxShadow: "0 0 24px #00224455", whiteSpace: "nowrap",
      }}>
        <div style={{ color: "#00ff99", fontSize: 13, fontWeight: "bold", letterSpacing: "0.18em" }}>
          RKK v5 — {backend ? "HIP/GPU ENGINE" : "LIVE CAUSAL ENGINE"}
        </div>
        <div style={{ color: "#115577", fontSize: 10, marginTop: 3, letterSpacing: "0.1em" }}>
          PHASE {ui.phase}/5 › {PHASE_NAMES[ui.phase]}
          &nbsp;│&nbsp;
          ENTROPY: <span style={{ color: ui.entropy < 30 ? "#00ff99" : "#aaccdd" }}>{ui.entropy}%</span>
          &nbsp;│&nbsp;
          TICK: <span style={{ color: "#334455" }}>{ui.tick}</span>
          &nbsp;│&nbsp;
          <span style={{ color: "#ff2244" }}>◆ DEMON</span>
          {backend && ui.demon && (
            <span style={{ color: "#443300", marginLeft: 6 }}>
              E:{(ui.demon.energy * 100).toFixed(0)}% C:{ui.demon.last_action_complexity?.toFixed(2)}
            </span>
          )}
        </div>
      </div>

      {/* ── Controls ── */}
      <div style={{
        position: "absolute", top: 82, left: "50%", transform: "translateX(-50%)",
        background: "rgba(0,8,20,0.85)", border: "1px solid #081e30",
        padding: "6px 14px", borderRadius: 2, display: "flex", gap: 8, alignItems: "center", fontSize: 9,
      }}>
        <span style={{ color: "#224455" }}>SPEED</span>
        {[1, 2, 4, 8].map(s => (
          <button key={s} onClick={() => setSpeed(s)} style={{
            padding: "2px 8px", borderRadius: 2, fontSize: 9, cursor: "pointer",
            background: speed === s ? "#001e38" : "transparent",
            border: `1px solid ${speed === s ? "#00aaff" : "#081e30"}`,
            color: speed === s ? "#00aaff" : "#334455",
          }}>{s}×</button>
        ))}
        <span style={{ color: "#334455", marginLeft: 6 }}>│</span>
        {PHASE_NAMES.slice(1).map((name, i) => {
          const ph = i + 1;
          return (
            <div key={ph} style={{
              padding: "2px 7px", borderRadius: 2, fontSize: 9,
              background: ui.phase === ph ? "#001e38" : "transparent",
              border: `1px solid ${ui.phase === ph ? "#00aaff" : ui.phase > ph ? "#003322" : "#081e30"}`,
              color: ui.phase === ph ? "#00aaff" : ui.phase > ph ? "#00aa55" : "#223344",
            }}>
              {ui.phase > ph ? "✓" : ph} {name}
            </div>
          );
        })}
      </div>

      {/* ── Agent panels ── */}
      <div style={{ position: "absolute", top: 120, left: 14, display: "flex", flexDirection: "column", gap: 8 }}>
        {ui.agents.map((a, i) => {
          const col     = AGENT_COLORS_CSS[i];
          const alpW    = `${Math.round(a.alphaMean * 100)}%`;
          const cgColor = a.compressionGain > 0 ? "#00ff99" : a.compressionGain < -0.05 ? "#ff4433" : "#aaccdd";
          return (
            <div key={i} style={{
              background: "rgba(0,8,20,0.9)",
              border: `1px solid ${col}33`,
              borderLeft: `3px solid ${col}`,
              padding: "9px 13px", minWidth: 215, borderRadius: 2,
              boxShadow: `0 0 14px ${col}15`,
            }}>
              <div style={{ color: col, fontSize: 11, fontWeight: "bold", marginBottom: 7, letterSpacing: "0.1em" }}>
                ◈ {a.name}
                <span style={{ color: "#1a3344", marginLeft: 8, fontSize: 9 }}>
                  {a.envType} · {a.activation ?? "—"}
                </span>
              </div>
              <table style={{ borderCollapse: "collapse", width: "100%", fontSize: 10 }}>
                <tbody>
                  {[
                    ["Φ autonomy",       (a.phi * 100).toFixed(1) + "%"],
                    ["CG (d|G|/dt)",     <span style={{ color: cgColor }}>{a.compressionGain >= 0 ? "+" : ""}{a.compressionGain.toFixed(4)}</span>],
                    ["α trust",          alpW],
                    ["G* nodes/edges",   `${a.nodeCount} / ${a.edgeCount}`],
                    ["MDL |G|",          (a.graphMdl ?? 0).toFixed(2)],
                    ["Discovery",        `${((a.discoveryRate ?? 0) * 100).toFixed(0)}%`],
                    ["Interventions",    a.totalInterventions],
                    ["Last do()",        <span style={{ color: "#334466", fontSize: 9 }}>{a.lastDo}</span>],
                  ].map(([label, val], k) => (
                    <tr key={k}>
                      <td style={{ color: "#335566", paddingRight: 8, paddingBottom: 2 }}>{label}</td>
                      <td style={{ color: "#aad4ee", textAlign: "right" }}>{val}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              {/* Alpha bar */}
              <div style={{ marginTop: 7, height: 3, background: "#061422", borderRadius: 2, overflow: "hidden" }}>
                <div style={{ width: alpW, height: "100%", background: `linear-gradient(90deg, #002266, ${col})`, transition: "width 0.8s ease" }} />
              </div>
              {/* Discovery bar */}
              <div style={{ marginTop: 3, height: 2, background: "#061422", borderRadius: 2, overflow: "hidden" }}>
                <div style={{
                  width: `${((a.discoveryRate ?? 0) * 100).toFixed(0)}%`,
                  height: "100%",
                  background: `linear-gradient(90deg, #003322, #00aa55)`,
                  transition: "width 1s ease",
                }} />
              </div>
              <div style={{ color: "#1a3344", fontSize: 9, marginTop: 2, display: "flex", justifyContent: "space-between" }}>
                <span>α: text→sandbox</span>
                <span style={{ color: "#1a4433" }}>discovery: {((a.discoveryRate ?? 0) * 100).toFixed(0)}%</span>
              </div>
            </div>
          );
        })}
      </div>

      {/* ── Legend / Architecture ── */}
      <div style={{
        position: "absolute", top: 120, right: 14,
        background: "rgba(0,8,20,0.9)", border: "1px solid #081e30",
        padding: "10px 14px", borderRadius: 2, fontSize: 10, maxWidth: 180,
      }}>
        <div style={{ color: "#114466", letterSpacing: "0.1em", marginBottom: 8, fontSize: 9 }}>ARCHITECTURE</div>
        {AGENT_NAMES.map((name, i) => (
          <div key={i} style={{ color: "#335566", marginBottom: 3 }}>
            <span style={{ color: AGENT_COLORS_CSS[i] }}>●</span> {name} ({["Physics","Chem","Logic"][i]})
          </div>
        ))}
        <div style={{ color: "#335566", marginBottom: 3 }}><span style={{ color: "#ff2244" }}>◆</span> Demon (L0 penalty)</div>
        <div style={{ color: "#335566", marginBottom: 3 }}><span style={{ color: "#ff4422" }}>─</span> Negative edge</div>
        <div style={{ color: "#335566", marginBottom: 3 }}><span style={{ color: "#003366" }}>─</span> ToM link</div>
        <div style={{ marginTop: 10, borderTop: "1px solid #081e30", paddingTop: 8, color: "#113344", fontSize: 9, lineHeight: 1.9 }}>
          <div style={{ color: "#115533", marginBottom: 3 }}>LIVE LOGIC:</div>
          <div>do() → interventional hardness</div>
          <div>MDL = Σ(1 + uncertainty)</div>
          <div>α: exp.annealing spurious</div>
          <div>CG = running d|G|/dt</div>
          <div>ToM: Φ-sync between agents</div>
          <div>Phase IV: ΔΦ ≥ 0 protection</div>
        </div>
        {backend && (
          <div style={{ marginTop: 8, borderTop: "1px solid #081e30", paddingTop: 8, color: "#332200", fontSize: 9, lineHeight: 1.9 }}>
            <div style={{ color: "#aa5500" }}>PYTHON BACKEND:</div>
            <div>System 1: MLP epistemic scorer</div>
            <div>Demon policy: NN + L0 penalty</div>
            <div>Seed diversity: ReLU/GELU/Tanh</div>
            <div>Pinned memory (HIP)</div>
          </div>
        )}
      </div>

      {/* ── Event log ── */}
      <div style={{
        position: "absolute", bottom: 14, left: 14, right: 14,
        background: "rgba(0,8,20,0.9)", border: "1px solid #081e30",
        padding: "10px 14px", borderRadius: 2, maxHeight: 140, overflow: "hidden",
      }}>
        <div style={{ color: "#113344", fontSize: 9, letterSpacing: "0.15em", marginBottom: 6 }}>
          CAUSAL EVENT STREAM {connected ? "— PYTHON/HIP" : "— LOCAL ENGINE"}
        </div>
        {ui.events.length === 0 && (
          <div style={{ color: "#1a3344", fontSize: 10 }}>Bootstrapping agents from text priors…</div>
        )}
        {ui.events.map((ev, i) => (
          <div key={i} style={{ color: ev.color, fontSize: 10, marginBottom: 3, opacity: Math.max(0.15, 1 - i * 0.09) }}>
            [{String(ev.tick ?? 0).padStart(4, "0")}] › {ev.text}
          </div>
        ))}
      </div>
    </div>
  );
}