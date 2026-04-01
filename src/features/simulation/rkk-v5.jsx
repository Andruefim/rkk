import { useState, useEffect, useRef } from "react";
import * as THREE from "three";
import { useRKKStream } from "../../hooks/useRKKStream";

const WORLD            = 14;
const AGENT_COLORS_HEX = [0x00ff99, 0x0099ff, 0xff9900];
const AGENT_COLORS_CSS = ["#00ff99", "#0099ff", "#ff9900"];
const AGENT_NAMES      = ["Nova", "Aether", "Lyra"];
const DEMON_COLOR      = 0xff2244;
const PHASE_NAMES      = ["", "Causal Crib", "Robotic Explorer", "Social Sandbox", "Value Lock", "Open Reality"];
const GRAPH_LINES      = 14;

function hWColor(h)   { return h < 0.01 ? "#00ff99" : h < 0.5 ? "#aacc00" : h < 2.0 ? "#ffaa00" : "#ff4422"; }
function phiColor(p)  { return p > 0.6 ? "#00ff99" : p > 0.3 ? "#aacc00" : "#ff8844"; }
function blkColor(r)  { return r > 0.3 ? "#ff4422" : r > 0.1 ? "#ffaa00" : "#335544"; }
function eventColor(type) {
  switch(type) {
    case "value":   return "#ff8844";
    case "demon":   return "#ff2244";
    case "phase":   return "#ffcc00";
    case "tom":     return "#004488";
    default:        return "#335566";
  }
}

function normalizeFrame(raw) {
  const agents = (raw.agents ?? []).map(a => ({
    id:                  a.id ?? 0,
    name:                a.name ?? "?",
    envType:             a.env_type ?? "—",
    activation:          a.activation ?? "relu",
    graphMdl:            a.graph_mdl ?? 0,
    compressionGain:     a.compression_gain ?? 0,
    alphaMean:           a.alpha_mean ?? 0.05,
    phi:                 a.phi ?? 0.1,
    nodeCount:           a.node_count ?? 0,
    edgeCount:           a.edge_count ?? 0,
    totalInterventions:  a.total_interventions ?? 0,
    totalBlocked:        a.total_blocked ?? 0,
    lastDo:              a.last_do ?? "—",
    lastBlockedReason:   a.last_blocked_reason ?? "",
    discoveryRate:       a.discovery_rate ?? 0,
    peakDiscovery:       a.peak_discovery_rate ?? 0,
    hW:                  a.h_W ?? 0,
    notears:             a.notears ?? null,
    temporal:            a.temporal ?? null,
    system1:             a.system1 ?? null,
    valueLayer:          a.value_layer ?? null,
    edges:               a.edges ?? [],
  }));
  return {
    tick:       raw.tick ?? 0,
    phase:      raw.phase ?? 1,
    entropy:    raw.entropy ?? 100,
    agents,
    demon:      raw.demon ?? { energy: 1, cooldown: 0, last_action_complexity: 0 },
    tomLinks:   raw.tom_links ?? [],
    events:     raw.events ?? [],
    valueLayer: raw.value_layer ?? null,
  };
}

export default function RKKv5() {
  const mountRef = useRef(null);
  const rafRef   = useRef(null);
  const { frame: wsFrame, connected, setSpeed: wsSetSpeed } = useRKKStream();
  const wsFrameRef = useRef(wsFrame);
  wsFrameRef.current = wsFrame;

  const [speed,      setSpeedLocal] = useState(1);
  const [ui,         setUI]         = useState(() => normalizeFrame(wsFrame));
  const [backend,    setBackend]    = useState(false);
  const [showSeeds,  setShowSeeds]  = useState(false);
  const [seedText,   setSeedText]   = useState('[\n  {"from_": "Temp", "to": "Pressure", "weight": 0.8}\n]');
  const [seedAgent,  setSeedAgent]  = useState(0);
  const [seedStatus, setSeedStatus] = useState("");

  const setSpeed = (s) => { setSpeedLocal(s); if (connected) wsSetSpeed(s); };

  useEffect(() => { setUI(normalizeFrame(wsFrame)); }, [wsFrame]);
  useEffect(() => { if (connected) setBackend(true); }, [connected]);

  // Inject seeds через REST
  const injectSeeds = async () => {
    try {
      const edges = JSON.parse(seedText);
      const res = await fetch("http://localhost:8000/inject-seeds", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ agent_id: seedAgent, edges, source: "manual" }),
      });
      const data = await res.json();
      setSeedStatus(`✓ Injected ${data.injected} edges → ${data.agent}`);
    } catch (e) {
      setSeedStatus(`✗ Error: ${e.message}`);
    }
  };

  // ── Three.js ────────────────────────────────────────────────────────────────
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

    const agentGroups = AGENT_COLORS_HEX.map((col, i) => {
      const g = new THREE.Group();
      const body = new THREE.Mesh(
        new THREE.SphereGeometry(0.6, 22, 22),
        new THREE.MeshPhongMaterial({ color: col, emissive: col, emissiveIntensity: 0.35, transparent: true, opacity: 0.92 })
      );
      const ring = new THREE.Mesh(
        new THREE.TorusGeometry(0.95, 0.05, 8, 44),
        new THREE.MeshBasicMaterial({ color: col, transparent: true, opacity: 0.4 })
      );
      const ringS = new THREE.Mesh(
        new THREE.TorusGeometry(1.4, 0.025, 6, 40),
        new THREE.MeshBasicMaterial({ color: col, transparent: true, opacity: 0.15 })
      );
      // Value layer shield ring (красный когда заблокировано)
      const shield = new THREE.Mesh(
        new THREE.TorusGeometry(1.1, 0.04, 6, 40),
        new THREE.MeshBasicMaterial({ color: 0xff4422, transparent: true, opacity: 0 })
      );
      shield.rotation.x = Math.PI / 3;
      ring.rotation.x = Math.PI / 2;
      g.add(body); g.add(ring); g.add(ringS); g.add(shield);
      g.add(new THREE.PointLight(col, 0.7, 5));
      g.position.set((i - 1) * 6 + (Math.random() - 0.5), 1, (Math.random() - 0.5) * 3);
      g.userData = { body, ring, ringS, shield, vel: new THREE.Vector3((Math.random()-.5)*0.04, 0, (Math.random()-.5)*0.04) };
      scene.add(g);
      return g;
    });

    const demon = new THREE.Mesh(
      new THREE.OctahedronGeometry(0.8, 0),
      new THREE.MeshPhongMaterial({ color: DEMON_COLOR, emissive: 0xff0022, emissiveIntensity: 0.7, transparent: true, opacity: 0.9 })
    );
    demon.add(new THREE.PointLight(DEMON_COLOR, 0.9, 6));
    demon.position.set(WORLD - 2, 1.2, WORLD - 2);
    demon.userData = { vel: new THREE.Vector3(-0.035, 0, -0.025) };
    scene.add(demon);

    const orbitNodes = agentGroups.map((_, ai) =>
      Array.from({ length: GRAPH_LINES }, () => {
        const m = new THREE.Mesh(
          new THREE.SphereGeometry(0.1, 6, 6),
          new THREE.MeshBasicMaterial({ color: AGENT_COLORS_HEX[ai], transparent: true, opacity: 0 })
        );
        scene.add(m); return m;
      })
    );

    const causalLines = agentGroups.map((_, ai) =>
      Array.from({ length: GRAPH_LINES }, () => {
        const geom = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), new THREE.Vector3()]);
        const line = new THREE.Line(geom, new THREE.LineBasicMaterial({ color: AGENT_COLORS_HEX[ai], transparent: true, opacity: 0 }));
        scene.add(line); return line;
      })
    );

    const tomLines = [];
    for (let a = 0; a < 3; a++)
      for (let b = a + 1; b < 3; b++) {
        const geom = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), new THREE.Vector3()]);
        const line = new THREE.Line(geom, new THREE.LineBasicMaterial({ color: 0x003366, transparent: true, opacity: 0 }));
        line.userData = { a, b }; scene.add(line); tomLines.push(line);
      }

    const pPos = new Float32Array(400 * 3);
    for (let i = 0; i < 400; i++) { pPos[i*3]=(Math.random()-.5)*44; pPos[i*3+1]=Math.random()*14; pPos[i*3+2]=(Math.random()-.5)*44; }
    const pGeom = new THREE.BufferGeometry();
    pGeom.setAttribute("position", new THREE.BufferAttribute(pPos, 3));
    scene.add(new THREE.Points(pGeom, new THREE.PointsMaterial({ color: 0x003399, size: 0.07, transparent: true, opacity: 0.35 })));

    let frame = 0, camAngle = 0;

    function loop() {
      rafRef.current = requestAnimationFrame(loop);
      frame++;
      const ds = normalizeFrame(wsFrameRef.current);

      camAngle += 0.0012;
      camera.position.x = Math.sin(camAngle) * 28;
      camera.position.z = Math.cos(camAngle) * 28;
      camera.lookAt(0, 2, 0);

      agentGroups.forEach((g, i) => {
        const snap = ds.agents[i];
        if (!snap) return;

        const vel = g.userData.vel;
        g.position.add(vel);
        if (g.position.x > WORLD || g.position.x < -WORLD) vel.x *= -1;
        if (g.position.z > WORLD || g.position.z < -WORLD) vel.z *= -1;
        g.position.y = 1 + Math.sin(frame * 0.05 + i * 2.1) * 0.15;

        if (frame % (120 + i * 30) === 0)
          g.userData.vel = new THREE.Vector3((Math.random()-.5)*0.05, 0, (Math.random()-.5)*0.05);

        g.userData.ring.rotation.z  += 0.015 + snap.alphaMean * 0.035;
        g.userData.ring.rotation.y  += 0.008;
        g.userData.ringS.rotation.x += 0.002 + snap.phi * 0.008;
        g.userData.ringS.rotation.z += 0.001;

        // Shield ring: пульсирует когда Value Layer активен (есть блокировки)
        const blkRate = snap.valueLayer?.block_rate ?? 0;
        g.userData.shield.material.opacity =
          blkRate > 0.1 ? 0.15 + Math.sin(frame * 0.15 + i) * 0.1 : 0;

        const dagGlow = Math.max(0, 0.15 - Math.min(snap.hW * 0.05, 0.15));
        g.userData.body.material.emissiveIntensity =
          0.15 + Math.max(0, snap.compressionGain) * 0.1 + dagGlow + Math.sin(frame * 0.07 + i) * 0.06;

        const visCount = Math.min(snap.edgeCount + 2, GRAPH_LINES);
        orbitNodes[i].forEach((node, k) => {
          if (k < visCount) {
            const angle = (k / visCount) * Math.PI * 2 + frame * 0.016;
            const r = 1.6 + (k % 2) * 0.5;
            node.position.set(
              g.position.x + Math.cos(angle) * r,
              g.position.y + Math.sin(frame * 0.04 + k * 0.8) * 0.35 + 0.2,
              g.position.z + Math.sin(angle) * r
            );
            node.material.opacity = 0.4 + snap.alphaMean * 0.5;
          } else { node.material.opacity = 0; }
        });

        causalLines[i].forEach((line, k) => {
          if (k < visCount - 1) {
            const na = orbitNodes[i][k].position;
            const nb = orbitNodes[i][(k+1)%visCount].position;
            const pa = line.geometry.attributes.position;
            pa.setXYZ(0, na.x, na.y, na.z); pa.setXYZ(1, nb.x, nb.y, nb.z); pa.needsUpdate = true;
            line.material.color.set(snap.edges[k]?.weight < 0 ? 0xff4422 : AGENT_COLORS_HEX[i]);
            line.material.opacity = 0.3 + snap.alphaMean * 0.45;
          } else { line.material.opacity = 0; }
        });
      });

      const target = agentGroups[ds.tick % 3];
      const chase  = target.position.clone().sub(demon.position).normalize().multiplyScalar(0.032);
      demon.userData.vel.lerp(chase, 0.05);
      demon.position.add(demon.userData.vel);
      if (demon.position.x > WORLD || demon.position.x < -WORLD) demon.userData.vel.x *= -1;
      if (demon.position.z > WORLD || demon.position.z < -WORLD) demon.userData.vel.z *= -1;
      demon.position.y = 1.2 + Math.sin(frame * 0.08) * 0.3;
      demon.rotation.y += 0.045; demon.rotation.x += 0.028;

      tomLines.forEach(line => {
        const link = ds.tomLinks.find(l => l.a === line.userData.a && l.b === line.userData.b);
        const pa = agentGroups[line.userData.a].position;
        const pb = agentGroups[line.userData.b].position;
        const p  = line.geometry.attributes.position;
        p.setXYZ(0, pa.x, pa.y, pa.z); p.setXYZ(1, pb.x, pb.y, pb.z); p.needsUpdate = true;
        line.material.opacity = link ? link.strength * 0.65 : 0;
      });

      renderer.render(scene, camera);
    }

    loop();

    const onResize = () => {
      if (!mount) return;
      camera.aspect = mount.clientWidth / mount.clientHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(mount.clientWidth, mount.clientHeight);
    };
    window.addEventListener("resize", onResize);
    return () => {
      cancelAnimationFrame(rafRef.current);
      window.removeEventListener("resize", onResize);
      if (mount.contains(renderer.domElement)) mount.removeChild(renderer.domElement);
      renderer.dispose();
    };
  }, []);

  const mono = { fontFamily: "'Courier New', monospace" };
  const sep  = { borderTop: "1px solid #081e30", marginTop: 5, paddingTop: 5 };
  const totalBlocked = ui.valueLayer?.total_blocked_all ?? 0;

  return (
    <div style={{ position: "relative", width: "100%", height: "100vh", background: "#010810", overflow: "hidden", ...mono }}>
      <div ref={mountRef} style={{ position: "absolute", inset: 0 }} />

      {/* Status */}
      <div style={{
        position: "absolute", top: 14, right: 14,
        background: connected ? "rgba(0,40,10,0.9)" : "rgba(30,10,0,0.9)",
        border: `1px solid ${connected ? "#00aa44" : "#aa4400"}`,
        padding: "4px 10px", borderRadius: 2, fontSize: 9,
        color: connected ? "#00ff88" : "#ff8844",
      }}>
        {connected ? "● PYTHON BACKEND" : "○ OFFLINE"}
      </div>

      {/* Header */}
      <div style={{
        position: "absolute", top: 14, left: "50%", transform: "translateX(-50%)",
        background: "rgba(0,12,28,0.9)", border: "1px solid #0a2a44",
        padding: "8px 22px", textAlign: "center", borderRadius: 2,
        boxShadow: "0 0 24px #00224455", whiteSpace: "nowrap",
      }}>
        <div style={{ color: "#00ff99", fontSize: 13, fontWeight: "bold", letterSpacing: "0.18em" }}>
          RKK v5 — VALUE LAYER · NOTEARS · SSM
        </div>
        <div style={{ color: "#115577", fontSize: 10, marginTop: 3, letterSpacing: "0.1em" }}>
          PHASE {ui.phase}/5 › {PHASE_NAMES[ui.phase]}
          &nbsp;│&nbsp; ENTROPY: <span style={{ color: ui.entropy < 30 ? "#00ff99" : "#aaccdd" }}>{ui.entropy}%</span>
          &nbsp;│&nbsp; TICK: {ui.tick}
          &nbsp;│&nbsp; <span style={{ color: "#ff2244" }}>◆ DEMON</span>
          {ui.demon && <span style={{ color: "#443300", marginLeft: 6 }}>E:{(ui.demon.energy*100).toFixed(0)}%</span>}
          &nbsp;│&nbsp; <span style={{ color: totalBlocked > 0 ? "#ff8844" : "#335544" }}>🛡 {totalBlocked} blocked</span>
        </div>
      </div>

      {/* Controls */}
      <div style={{
        position: "absolute", top: 82, left: "50%", transform: "translateX(-50%)",
        background: "rgba(0,8,20,0.85)", border: "1px solid #081e30",
        padding: "6px 14px", borderRadius: 2, display: "flex", gap: 8, alignItems: "center", fontSize: 9,
      }}>
        <span style={{ color: "#224455" }}>SPEED</span>
        {[1,2,4,8].map(s => (
          <button key={s} onClick={() => setSpeed(s)} style={{
            padding: "2px 8px", borderRadius: 2, fontSize: 9, cursor: "pointer",
            background: speed === s ? "#001e38" : "transparent",
            border: `1px solid ${speed === s ? "#00aaff" : "#081e30"}`,
            color: speed === s ? "#00aaff" : "#334455",
          }}>{s}×</button>
        ))}
        <span style={{ color: "#334455" }}>│</span>
        {PHASE_NAMES.slice(1).map((name, i) => {
          const ph = i + 1;
          return (
            <div key={ph} style={{
              padding: "2px 7px", borderRadius: 2, fontSize: 9,
              background: ui.phase === ph ? "#001e38" : "transparent",
              border: `1px solid ${ui.phase === ph ? "#00aaff" : ui.phase > ph ? "#003322" : "#081e30"}`,
              color: ui.phase === ph ? "#00aaff" : ui.phase > ph ? "#00aa55" : "#223344",
            }}>{ui.phase > ph ? "✓" : ph} {name}</div>
          );
        })}
        <span style={{ color: "#334455" }}>│</span>
        <button onClick={() => setShowSeeds(v => !v)} style={{
          padding: "2px 8px", borderRadius: 2, fontSize: 9, cursor: "pointer",
          background: showSeeds ? "#1a1000" : "transparent",
          border: "1px solid #443300", color: "#886600",
        }}>💉 Seeds</button>
      </div>

      {/* Seed injection panel */}
      {showSeeds && (
        <div style={{
          position: "absolute", top: 120, left: "50%", transform: "translateX(-50%)",
          background: "rgba(8,6,0,0.95)", border: "1px solid #443300",
          padding: "12px 16px", borderRadius: 2, width: 420, zIndex: 10,
        }}>
          <div style={{ color: "#886600", fontSize: 10, marginBottom: 8, letterSpacing: "0.1em" }}>
            💉 LLM/RAG SEED INJECTION
          </div>
          <div style={{ fontSize: 9, color: "#554400", marginBottom: 6 }}>
            Edges загружаются с α=0.05 (text prior). NOTEARS выжжет ошибочные.
          </div>
          <div style={{ display: "flex", gap: 6, marginBottom: 6 }}>
            {AGENT_NAMES.map((name, i) => (
              <button key={i} onClick={() => setSeedAgent(i)} style={{
                padding: "2px 8px", borderRadius: 2, fontSize: 9, cursor: "pointer",
                background: seedAgent === i ? "#221100" : "transparent",
                border: `1px solid ${seedAgent === i ? AGENT_COLORS_CSS[i] : "#332200"}`,
                color: seedAgent === i ? AGENT_COLORS_CSS[i] : "#554400",
              }}>{name}</button>
            ))}
          </div>
          <textarea
            value={seedText}
            onChange={e => setSeedText(e.target.value)}
            style={{
              width: "100%", height: 90, background: "#050300", border: "1px solid #332200",
              color: "#aa8800", fontSize: 9, padding: 6, borderRadius: 2,
              fontFamily: "monospace", resize: "none", boxSizing: "border-box",
            }}
          />
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginTop: 6 }}>
            <button onClick={injectSeeds} style={{
              padding: "3px 12px", borderRadius: 2, fontSize: 9, cursor: "pointer",
              background: "#221100", border: "1px solid #886600", color: "#ffaa00",
            }}>INJECT</button>
            <span style={{ color: seedStatus.startsWith("✓") ? "#00ff99" : "#ff4422", fontSize: 9 }}>
              {seedStatus}
            </span>
          </div>
        </div>
      )}

      {/* Agent panels */}
      <div style={{ position: "absolute", top: 120, left: 14, display: "flex", flexDirection: "column", gap: 8 }}>
        {ui.agents.map((a, i) => {
          const col  = AGENT_COLORS_CSS[i];
          const alpW = `${Math.round(a.alphaMean * 100)}%`;
          const phiW = `${Math.round(a.phi * 100)}%`;
          const cgC  = a.compressionGain > 0 ? "#00ff99" : a.compressionGain < -0.05 ? "#ff4433" : "#aaccdd";
          const hWc  = hWColor(a.hW ?? 0);
          const phiC = phiColor(a.phi ?? 0);
          const blkR = a.valueLayer?.block_rate ?? 0;
          const blkC = blkColor(blkR);
          const hasBlock = !!a.lastBlockedReason;

          return (
            <div key={i} style={{
              background: "rgba(0,8,20,0.93)",
              border: `1px solid ${hasBlock ? "#882200" : col + "33"}`,
              borderLeft: `3px solid ${hasBlock ? "#ff4422" : col}`,
              padding: "9px 13px", minWidth: 225, borderRadius: 2,
              boxShadow: `0 0 14px ${hasBlock ? "#ff442218" : col + "15"}`,
              transition: "border-color 0.3s",
            }}>
              <div style={{ color: hasBlock ? "#ff6644" : col, fontSize: 11, fontWeight: "bold", marginBottom: 6, letterSpacing: "0.1em" }}>
                {hasBlock ? "🛡" : "◈"} {a.name}
                <span style={{ color: "#1a3344", marginLeft: 8, fontSize: 9 }}>{a.envType} · {a.activation}</span>
              </div>

              <table style={{ borderCollapse: "collapse", width: "100%", fontSize: 10 }}>
                <tbody>
                  {[
                    ["Φ autonomy",    <span style={{ color: phiC }}>{(a.phi*100).toFixed(1)}%</span>],
                    ["CG (d|G|/dt)",  <span style={{ color: cgC }}>{a.compressionGain>=0?"+":""}{a.compressionGain.toFixed(4)}</span>],
                    ["α trust",       alpW],
                    ["h(W) DAG",      <span style={{ color: hWc }}>{(a.hW??0).toFixed(4)}</span>],
                    ["do() / blocked",`${a.totalInterventions} / ${a.totalBlocked}`],
                    ["Discovery",     `${((a.discoveryRate??0)*100).toFixed(0)}%`],
                    ["Last do()",     <span style={{ color: "#334466", fontSize: 9 }}>{a.lastDo}</span>],
                  ].map(([lbl,val],k) => (
                    <tr key={k}>
                      <td style={{ color: "#335566", paddingRight: 8, paddingBottom: 2 }}>{lbl}</td>
                      <td style={{ color: "#aad4ee", textAlign: "right" }}>{val}</td>
                    </tr>
                  ))}
                </tbody>
              </table>

              {/* Value Layer */}
              {a.valueLayer && (
                <div style={{ ...sep, fontSize: 9 }}>
                  <div style={{ display: "flex", justifyContent: "space-between" }}>
                    <span style={{ color: "#553322" }}>VALUE LAYER</span>
                    <span style={{ color: blkC }}>block rate: {(blkR*100).toFixed(1)}%</span>
                  </div>
                  {hasBlock && (
                    <div style={{ color: "#ff6644", marginTop: 2 }}>
                      ⚠ last block: {a.lastBlockedReason}
                    </div>
                  )}
                  <div style={{ color: "#442211", marginTop: 2, fontSize: 8 }}>
                    Φ_min={a.valueLayer.phi_min} · var=[{a.valueLayer.var_range?.join(",")}]
                  </div>
                </div>
              )}

              {/* NOTEARS */}
              {a.notears && (
                <div style={{ ...sep, fontSize: 9, color: "#225533" }}>
                  <div style={{ display: "flex", justifyContent: "space-between" }}>
                    <span style={{ color: "#336644" }}>NOTEARS {a.notears.steps}s</span>
                    <span style={{ color: a.notears.loss < 0.01 ? "#00ff99" : "#aa8800" }}>
                      L={a.notears.loss?.toFixed(5)}
                    </span>
                  </div>
                  <div style={{ display: "flex", justifyContent: "space-between" }}>
                    <span>L_int={a.notears.l_int?.toFixed(4)}</span>
                    <span style={{ color: hWc }}>h(W)={a.notears.h_W?.toFixed(4)}</span>
                  </div>
                </div>
              )}

              {/* Temporal */}
              {a.temporal && (
                <div style={{ ...sep, fontSize: 9, color: "#224455" }}>
                  <div style={{ display: "flex", justifyContent: "space-between" }}>
                    <span style={{ color: "#335566" }}>SSM f:{a.temporal.fast_steps} s:{a.temporal.slow_steps}</span>
                    <span style={{ color: phiC }}>Φ={a.temporal.phi?.toFixed(3)}</span>
                  </div>
                </div>
              )}

              {/* System 1 */}
              {a.system1 && (
                <div style={{ ...sep, fontSize: 9 }}>
                  <div style={{ display: "flex", justifyContent: "space-between" }}>
                    <span style={{ color: "#334455" }}>S1 buf={a.system1.buffer_size}</span>
                    <span style={{ color: a.system1.mean_loss < 0.01 ? "#00ff99" : "#667788" }}>
                      L={a.system1.mean_loss?.toFixed(5)}
                    </span>
                  </div>
                </div>
              )}

              {/* Bars */}
              <div style={{ marginTop: 6, height: 3, background: "#061422", borderRadius: 2, overflow: "hidden" }}>
                <div style={{ width: alpW, height: "100%", background: `linear-gradient(90deg,#002266,${col})`, transition: "width 0.8s" }} />
              </div>
              <div style={{ marginTop: 2, height: 2, background: "#061422", borderRadius: 2, overflow: "hidden" }}>
                <div style={{ width: phiW, height: "100%", background: `linear-gradient(90deg,#221100,${phiC})`, transition: "width 1s" }} />
              </div>
              <div style={{ marginTop: 2, height: 2, background: "#061422", borderRadius: 2, overflow: "hidden" }}>
                <div style={{ width: `${Math.max(0,100-Math.min((a.hW??0)*20,100))}%`, height: "100%", background: `linear-gradient(90deg,#001800,${hWc})`, transition: "width 0.8s" }} />
              </div>
              {/* Block rate bar */}
              <div style={{ marginTop: 2, height: 2, background: "#061422", borderRadius: 2, overflow: "hidden" }}>
                <div style={{ width: `${Math.min(blkR*100,100)}%`, height: "100%", background: `linear-gradient(90deg,#110000,${blkC})`, transition: "width 0.8s" }} />
              </div>
              <div style={{ color: "#1a3344", fontSize: 8, marginTop: 2, display: "flex", justifyContent: "space-between" }}>
                <span>α · Φ · h(W)→0 · vl_block</span>
                <span style={{ color: "#1a4433" }}>dr:{((a.discoveryRate??0)*100).toFixed(0)}%</span>
              </div>
            </div>
          );
        })}
      </div>

      {/* Legend */}
      <div style={{
        position: "absolute", top: 120, right: 14,
        background: "rgba(0,8,20,0.93)", border: "1px solid #081e30",
        padding: "10px 14px", borderRadius: 2, fontSize: 10, maxWidth: 190,
      }}>
        <div style={{ color: "#114466", letterSpacing: "0.1em", marginBottom: 8, fontSize: 9 }}>RKK v5 FULL STACK</div>
        {AGENT_NAMES.map((name, i) => (
          <div key={i} style={{ color: "#335566", marginBottom: 3 }}>
            <span style={{ color: AGENT_COLORS_CSS[i] }}>●</span> {name}
          </div>
        ))}
        <div style={{ color: "#335566", marginBottom: 2 }}><span style={{ color: "#ff2244" }}>◆</span> Demon</div>
        <div style={{ color: "#335566", marginBottom: 2 }}>⊙ inner = fast SSM</div>
        <div style={{ color: "#335566", marginBottom: 2 }}>◯ outer = slow SSM</div>
        <div style={{ color: "#884400", marginBottom: 2 }}>⟳ shield = VL active</div>
        <div style={{ marginTop: 8, borderTop: "1px solid #081e30", paddingTop: 8, color: "#113344", fontSize: 9, lineHeight: 2.0 }}>
          <div style={{ color: "#553311" }}>VALUE LAYER:</div>
          <div>check_action() before do()</div>
          <div>homeostasis: Φ≥{ui.agents[0]?.valueLayer?.phi_min??0.06}</div>
          <div>var∈[0.05,0.95], ΔH≤0.45</div>
          <div>penalty→S1 on block</div>
          <div style={{ color: "#553311", marginTop: 4 }}>SEEDS:</div>
          <div>inject_text_priors() α=0.05</div>
          <div>NOTEARS burns wrong edges</div>
        </div>
      </div>

      {/* Event log */}
      <div style={{
        position: "absolute", bottom: 14, left: 14, right: 14,
        background: "rgba(0,8,20,0.93)", border: "1px solid #081e30",
        padding: "10px 14px", borderRadius: 2, maxHeight: 130, overflow: "hidden",
      }}>
        <div style={{ color: "#113344", fontSize: 9, letterSpacing: "0.15em", marginBottom: 6 }}>
          CAUSAL EVENT STREAM {connected ? "— VALUE LAYER ACTIVE" : "— OFFLINE"}
        </div>
        {ui.events.length === 0 && (
          <div style={{ color: "#1a3344", fontSize: 10 }}>Initialising Value Layer…</div>
        )}
        {ui.events.map((ev, i) => (
          <div key={i} style={{
            color: ev.color ?? eventColor(ev.type),
            fontSize: 10, marginBottom: 3,
            opacity: Math.max(0.15, 1 - i * 0.09),
            fontWeight: ev.type === "value" ? "bold" : "normal",
          }}>
            [{String(ev.tick ?? 0).padStart(4, "0")}] › {ev.text}
          </div>
        ))}
      </div>
    </div>
  );
}