import { useState, useEffect, useRef } from "react";
import * as THREE from "three";

// ─── CONSTANTS ────────────────────────────────────────────────────────────────
const WORLD = 14;
const AGENT_COLORS_HEX = [0x00ff99, 0x0099ff, 0xff9900];
const AGENT_COLORS_CSS = ["#00ff99", "#0099ff", "#ff9900"];
const AGENT_NAMES      = ["Nova", "Aether", "Lyra"];
const DEMON_COLOR      = 0xff2244;
const PHASE_NAMES      = ["", "Causal Crib", "Robotic Explorer", "Social Sandbox", "Value Lock", "Open Reality"];
const ENV_NODES        = 22;
const GRAPH_LINES      = 12;

function mkAgent(i) {
  return {
    id: i,
    name: AGENT_NAMES[i],
    phi: 20 + Math.random() * 20,
    cg: 0,
    alpha: 0.08,
    nodes: 2,
    edges: 1,
    totalCG: 0,
  };
}

// ─── MAIN COMPONENT ───────────────────────────────────────────────────────────
export default function RKKv5() {
  const mountRef   = useRef(null);
  const simRef     = useRef({
    agents: [mkAgent(0), mkAgent(1), mkAgent(2)],
    phase: 1,
    entropy: 100,
    events: [],
    frame: 0,
  });
  const rafRef     = useRef(null);

  const [ui, setUI] = useState({
    agents: simRef.current.agents.map(a => ({ ...a })),
    phase: 1,
    entropy: 100,
    events: [],
  });

  useEffect(() => {
    const mount = mountRef.current;
    if (!mount) return;

    // ── Renderer ──
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(mount.clientWidth, mount.clientHeight);
    renderer.shadowMap.enabled = false;
    mount.appendChild(renderer.domElement);

    // ── Scene ──
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x010810);
    scene.fog = new THREE.FogExp2(0x010810, 0.025);

    // ── Camera ──
    const camera = new THREE.PerspectiveCamera(55, mount.clientWidth / mount.clientHeight, 0.1, 200);
    camera.position.set(0, 14, 26);
    camera.lookAt(0, 1, 0);

    // ── Lights ──
    scene.add(new THREE.AmbientLight(0x112233, 1.5));
    const sun = new THREE.DirectionalLight(0x334488, 1);
    sun.position.set(5, 10, 5);
    scene.add(sun);

    // ── Grid floor ──
    const grid = new THREE.GridHelper(WORLD * 2 + 4, 28, 0x0a1e38, 0x050f1e);
    scene.add(grid);

    // ── Discovery nodes (env) ──
    const envMeshes = [];
    const nodeGeom  = new THREE.SphereGeometry(0.22, 10, 10);
    for (let i = 0; i < ENV_NODES; i++) {
      const hue  = Math.random();
      const mat  = new THREE.MeshBasicMaterial({
        color: new THREE.Color().setHSL(hue, 0.9, 0.55),
        transparent: true, opacity: 0.85,
      });
      const mesh = new THREE.Mesh(nodeGeom, mat);
      mesh.position.set(
        (Math.random() - 0.5) * (WORLD * 2 - 2),
        0.4 + Math.random() * 2.5,
        (Math.random() - 0.5) * (WORLD * 2 - 2)
      );
      mesh.userData = { discovered: false, value: 4 + Math.random() * 10 };
      scene.add(mesh);
      envMeshes.push(mesh);
    }

    // ── Agent groups ──
    const agentGroups = AGENT_COLORS_HEX.map((col, i) => {
      const g = new THREE.Group();

      // Body
      const body = new THREE.Mesh(
        new THREE.SphereGeometry(0.55, 20, 20),
        new THREE.MeshPhongMaterial({ color: col, emissive: col, emissiveIntensity: 0.35, transparent: true, opacity: 0.92 })
      );
      g.add(body);

      // Ring
      const ring = new THREE.Mesh(
        new THREE.TorusGeometry(0.88, 0.045, 8, 40),
        new THREE.MeshBasicMaterial({ color: col, transparent: true, opacity: 0.45 })
      );
      ring.rotation.x = Math.PI / 2;
      g.add(ring);

      // Point light
      const pt = new THREE.PointLight(col, 0.6, 4);
      g.add(pt);

      g.position.set((i - 1) * 5.5 + (Math.random() - 0.5), 0.9, (Math.random() - 0.5) * 3);
      g.userData = {
        vel: new THREE.Vector3((Math.random() - 0.5) * 0.04, 0, (Math.random() - 0.5) * 0.04),
        target: null,
        body,
        ring,
        index: i,
      };
      scene.add(g);
      return g;
    });

    // ── Demon ──
    const demonMesh = new THREE.Mesh(
      new THREE.OctahedronGeometry(0.75, 0),
      new THREE.MeshPhongMaterial({ color: DEMON_COLOR, emissive: 0xff0022, emissiveIntensity: 0.6, transparent: true, opacity: 0.88 })
    );
    const demonLight = new THREE.PointLight(DEMON_COLOR, 0.7, 5);
    demonMesh.add(demonLight);
    demonMesh.position.set(WORLD - 1, 1, WORLD - 1);
    demonMesh.userData = {
      vel: new THREE.Vector3(-0.03, 0, -0.02),
      energy: 100,
      cooldown: 0,
    };
    scene.add(demonMesh);

    // ── ToM connection lines ──
    const tomLines = [];
    for (let a = 0; a < 3; a++) {
      for (let b = a + 1; b < 3; b++) {
        const pts  = [new THREE.Vector3(), new THREE.Vector3()];
        const geom = new THREE.BufferGeometry().setFromPoints(pts);
        const line = new THREE.Line(geom, new THREE.LineBasicMaterial({ color: 0x004488, transparent: true, opacity: 0 }));
        line.userData = { a, b };
        scene.add(line);
        tomLines.push(line);
      }
    }

    // ── Causal graph lines per agent ──
    const causalLines = agentGroups.map((_, agentIdx) => {
      return Array.from({ length: GRAPH_LINES }, () => {
        const pts  = [new THREE.Vector3(), new THREE.Vector3()];
        const geom = new THREE.BufferGeometry().setFromPoints(pts);
        const line = new THREE.Line(
          geom,
          new THREE.LineBasicMaterial({ color: AGENT_COLORS_HEX[agentIdx], transparent: true, opacity: 0 })
        );
        scene.add(line);
        return line;
      });
    });

    // ── Causal graph orbit nodes ──
    const orbitNodes = agentGroups.map((_, agentIdx) => {
      return Array.from({ length: GRAPH_LINES }, () => {
        const m = new THREE.Mesh(
          new THREE.SphereGeometry(0.09, 6, 6),
          new THREE.MeshBasicMaterial({ color: AGENT_COLORS_HEX[agentIdx], transparent: true, opacity: 0 })
        );
        scene.add(m);
        return m;
      });
    });

    // ── Ambient particles ──
    const pCount = 300;
    const pPos   = new Float32Array(pCount * 3);
    for (let i = 0; i < pCount; i++) {
      pPos[i * 3]     = (Math.random() - 0.5) * 40;
      pPos[i * 3 + 1] = Math.random() * 12;
      pPos[i * 3 + 2] = (Math.random() - 0.5) * 40;
    }
    const pGeom = new THREE.BufferGeometry();
    pGeom.setAttribute("position", new THREE.BufferAttribute(pPos, 3));
    const pMat  = new THREE.PointsMaterial({ color: 0x0033aa, size: 0.08, transparent: true, opacity: 0.4 });
    scene.add(new THREE.Points(pGeom, pMat));

    // ── Animation loop ──
    const clock    = new THREE.Clock();
    let   camAngle = 0;
    let   uiTick   = 0;

    function loop() {
      rafRef.current = requestAnimationFrame(loop);
      const sim = simRef.current;
      sim.frame++;
      uiTick++;

      // Camera orbit
      camAngle += 0.0015;
      camera.position.x = Math.sin(camAngle) * 26;
      camera.position.z = Math.cos(camAngle) * 26;
      camera.lookAt(0, 2, 0);

      // ── Update agents ──
      agentGroups.forEach((g, i) => {
        const pos = g.position;
        const vel = g.userData.vel;
        const agentD = sim.agents[i];

        // Pick target if none
        if (!g.userData.target) {
          let best = null, bestD = Infinity;
          envMeshes.forEach(n => {
            if (!n.userData.discovered) {
              const d = pos.distanceTo(n.position);
              if (d < bestD) { bestD = d; best = n; }
            }
          });
          if (best) {
            g.userData.target = best;
            const dir = best.position.clone().sub(pos).normalize();
            g.userData.vel = dir.multiplyScalar(0.028 + Math.random() * 0.018);
          } else {
            // Wander
            if (sim.frame % 180 === i * 40) {
              g.userData.vel = new THREE.Vector3((Math.random() - 0.5) * 0.05, 0, (Math.random() - 0.5) * 0.05);
            }
          }
        }

        // Arrive at target?
        if (g.userData.target) {
          const dist = pos.distanceTo(g.userData.target.position);
          if (dist < 1.3) {
            const n = g.userData.target;
            if (!n.userData.discovered) {
              n.userData.discovered = true;
              n.material.opacity = 0.18;
              n.material.color.multiplyScalar(0.5);

              agentD.nodes  += 1;
              agentD.edges  += Math.ceil(Math.random() * 2);
              agentD.totalCG += n.userData.value;
              agentD.cg      = n.userData.value;
              agentD.alpha   = Math.min(1, agentD.alpha + 0.04);
              agentD.phi     = Math.min(100, agentD.phi + 1.5);

              const totalCG = sim.agents.reduce((s, a) => s + a.totalCG, 0);
              sim.entropy   = Math.max(0, Math.round(100 - totalCG * 0.42));
              if (totalCG > 45  && sim.phase < 2) { sim.phase = 2; sim.events.unshift({ text: "⬆ Phase II: Robotic Explorer unlocked", color: "#ffcc00" }); }
              if (totalCG > 90  && sim.phase < 3) { sim.phase = 3; sim.events.unshift({ text: "⬆ Phase III: Social Sandbox online", color: "#ffcc00" }); }
              if (totalCG > 145 && sim.phase < 4) { sim.phase = 4; sim.events.unshift({ text: "⬆ Phase IV: Value Lock engaged", color: "#ffcc00" }); }
              if (totalCG > 200 && sim.phase < 5) { sim.phase = 5; sim.events.unshift({ text: "★ Phase V: Open Reality", color: "#00ff99" }); }

              sim.events.unshift({ text: `${agentD.name} → causal node +${n.userData.value.toFixed(1)} CG`, color: AGENT_COLORS_CSS[i] });
              if (sim.events.length > 9) sim.events.length = 9;
            }
            g.userData.target = null;
            g.userData.vel    = new THREE.Vector3((Math.random() - 0.5) * 0.04, 0, (Math.random() - 0.5) * 0.04);
          }
        }

        // Move
        pos.add(vel);
        if (pos.x >  WORLD || pos.x < -WORLD) { vel.x *= -1; pos.x = Math.sign(pos.x) * WORLD; }
        if (pos.z >  WORLD || pos.z < -WORLD) { vel.z *= -1; pos.z = Math.sign(pos.z) * WORLD; }
        pos.y = 0.9 + Math.sin(sim.frame * 0.05 + i * 2.1) * 0.14;

        // Visuals
        g.userData.ring.rotation.z += 0.018 + i * 0.004;
        g.userData.ring.rotation.y += 0.009;
        g.userData.body.material.emissiveIntensity = 0.2 + Math.sin(sim.frame * 0.08 + i) * 0.12;

        // Orbit graph nodes
        const visCount = Math.min(agentD.nodes, GRAPH_LINES);
        orbitNodes[i].forEach((node, k) => {
          if (k < visCount) {
            const angle = (k / visCount) * Math.PI * 2 + sim.frame * 0.018;
            const r     = 1.5 + (k % 2) * 0.4;
            node.position.set(
              pos.x + Math.cos(angle) * r,
              pos.y + Math.sin(sim.frame * 0.04 + k * 0.9) * 0.3 + 0.2,
              pos.z + Math.sin(angle) * r
            );
            node.material.opacity = 0.75;
          } else {
            node.material.opacity = 0;
          }
        });

        // Causal graph edges (orbit node to node)
        causalLines[i].forEach((line, k) => {
          if (k < visCount - 1) {
            const na = orbitNodes[i][k].position;
            const nb = orbitNodes[i][(k + 1) % visCount].position;
            const pa = line.geometry.attributes.position;
            pa.setXYZ(0, na.x, na.y, na.z);
            pa.setXYZ(1, nb.x, nb.y, nb.z);
            pa.needsUpdate = true;
            line.material.opacity = 0.5;
          } else {
            line.material.opacity = 0;
          }
        });
      });

      // ── Demon ──
      const D   = demonMesh;
      const dUD = D.userData;
      if (dUD.cooldown > 0) dUD.cooldown--;

      // Chase nearest agent
      let nearestAgent = agentGroups[0];
      let nearestDist  = D.position.distanceTo(agentGroups[0].position);
      agentGroups.forEach(a => {
        const d = D.position.distanceTo(a.position);
        if (d < nearestDist) { nearestDist = d; nearestAgent = a; }
      });
      const chaseDir = nearestAgent.position.clone().sub(D.position).normalize();
      dUD.vel.lerp(chaseDir.multiplyScalar(0.038), 0.04);

      // Disrupt?
      if (nearestDist < 2.0 && dUD.cooldown === 0 && dUD.energy > 15) {
        dUD.energy   -= 18;
        dUD.cooldown  = 140;
        const idx    = nearestAgent.userData.index;
        const agentD = simRef.current.agents[idx];
        agentD.phi   = Math.max(4, agentD.phi - 6);
        sim.events.unshift({ text: `⚠ Demon disrupts ${agentD.name} — Φ ↓`, color: "#ff2244" });
        if (sim.events.length > 9) sim.events.length = 9;

        // Value layer: other agents converge to protect (visible via ToM lines)
        agentGroups.forEach((a, i) => {
          if (i !== idx && sim.phase >= 4) {
            const protect = nearestAgent.position.clone().sub(a.position).normalize().multiplyScalar(0.06);
            a.userData.vel.add(protect).clampLength(0, 0.07);
            sim.agents[i].phi = Math.min(100, sim.agents[i].phi + 1);
          }
        });
      }
      if (dUD.cooldown === 0) dUD.energy = Math.min(100, dUD.energy + 0.08);

      D.position.add(dUD.vel);
      if (D.position.x >  WORLD || D.position.x < -WORLD) { dUD.vel.x *= -1; D.position.x = Math.sign(D.position.x) * WORLD; }
      if (D.position.z >  WORLD || D.position.z < -WORLD) { dUD.vel.z *= -1; D.position.z = Math.sign(D.position.z) * WORLD; }
      D.position.y = 1 + Math.sin(sim.frame * 0.07) * 0.25;
      D.rotation.y += 0.04;
      D.rotation.x += 0.025;

      // ── ToM lines ──
      tomLines.forEach(line => {
        const pa = agentGroups[line.userData.a].position;
        const pb = agentGroups[line.userData.b].position;
        const d  = pa.distanceTo(pb);
        const p  = line.geometry.attributes.position;
        p.setXYZ(0, pa.x, pa.y, pa.z);
        p.setXYZ(1, pb.x, pb.y, pb.z);
        p.needsUpdate = true;
        line.material.opacity = d < 9 ? Math.max(0, 0.45 - d * 0.045) : 0;
      });

      // ── UI update every 20 frames ──
      if (uiTick >= 20) {
        uiTick = 0;
        const s = simRef.current;
        setUI({
          agents:  s.agents.map(a => ({ ...a })),
          phase:   s.phase,
          entropy: s.entropy,
          events:  [...s.events],
        });
      }

      renderer.render(scene, camera);
    }

    loop();

    // Resize
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
  }, []);

  // ── UI ──────────────────────────────────────────────────────────────────────
  const mono = { fontFamily: "'Courier New', monospace" };

  return (
    <div style={{ position: "relative", width: "100%", height: "100vh", background: "#010810", overflow: "hidden", ...mono }}>
      {/* Three.js mount */}
      <div ref={mountRef} style={{ position: "absolute", inset: 0 }} />

      {/* ── Header ── */}
      <div style={{
        position: "absolute", top: 14, left: "50%", transform: "translateX(-50%)",
        background: "rgba(0,12,28,0.88)", border: "1px solid #0a2a44",
        padding: "8px 22px", textAlign: "center", borderRadius: 2,
        boxShadow: "0 0 20px #00224444",
      }}>
        <div style={{ color: "#00ff99", fontSize: 13, fontWeight: "bold", letterSpacing: "0.18em" }}>
          RKK v5 — RECURSIVE CAUSAL COMPRESSION
        </div>
        <div style={{ color: "#115577", fontSize: 10, marginTop: 3, letterSpacing: "0.12em" }}>
          PHASE&nbsp;{ui.phase}&nbsp;/&nbsp;5 &nbsp;›&nbsp; {PHASE_NAMES[ui.phase]}
          &nbsp;&nbsp;│&nbsp;&nbsp;
          ENV ENTROPY:&nbsp;<span style={{ color: ui.entropy < 30 ? "#00ff99" : "#aaccdd" }}>{ui.entropy}%</span>
          &nbsp;&nbsp;│&nbsp;&nbsp;
          <span style={{ color: "#ff2244" }}>◆ DEMON ACTIVE</span>
        </div>
      </div>

      {/* ── Agent panels ── */}
      <div style={{ position: "absolute", top: 82, left: 14, display: "flex", flexDirection: "column", gap: 8 }}>
        {ui.agents.map((a, i) => {
          const col = AGENT_COLORS_CSS[i];
          const alphaW = `${Math.round(a.alpha * 100)}%`;
          return (
            <div key={i} style={{
              background: "rgba(0,8,20,0.88)",
              border: `1px solid ${col}33`,
              borderLeft: `3px solid ${col}`,
              padding: "9px 13px", minWidth: 195, borderRadius: 2,
              boxShadow: `0 0 14px ${col}18`,
            }}>
              <div style={{ color: col, fontSize: 11, fontWeight: "bold", marginBottom: 6, letterSpacing: "0.1em" }}>
                ◈ {a.name}
              </div>
              <table style={{ borderCollapse: "collapse", width: "100%", fontSize: 10 }}>
                <tbody>
                  {[
                    ["Φ autonomy", a.phi.toFixed(1)],
                    ["CG last",    `+${typeof a.cg === "number" ? a.cg.toFixed(1) : a.cg}`],
                    ["α trust",    alphaW],
                    ["G nodes/edges", `${a.nodes} / ${a.edges}`],
                  ].map(([label, val]) => (
                    <tr key={label}>
                      <td style={{ color: "#335566", paddingRight: 8, paddingBottom: 2 }}>{label}</td>
                      <td style={{ color: "#aad4ee", textAlign: "right" }}>{val}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              {/* Alpha trust bar */}
              <div style={{ marginTop: 7, height: 3, background: "#061422", borderRadius: 2, overflow: "hidden" }}>
                <div style={{
                  width: alphaW, height: "100%",
                  background: `linear-gradient(90deg, #002266, ${col})`,
                  transition: "width 0.6s ease",
                }} />
              </div>
              <div style={{ color: "#225544", fontSize: 9, marginTop: 3 }}>α: text→sandbox trust</div>
            </div>
          );
        })}
      </div>

      {/* ── Legend ── */}
      <div style={{
        position: "absolute", top: 82, right: 14,
        background: "rgba(0,8,20,0.88)", border: "1px solid #081e30",
        padding: "10px 14px", borderRadius: 2, fontSize: 10,
      }}>
        <div style={{ color: "#114466", letterSpacing: "0.12em", marginBottom: 8, fontSize: 9 }}>LEGEND</div>
        {AGENT_NAMES.map((name, i) => (
          <div key={i} style={{ color: "#335566", marginBottom: 4 }}>
            <span style={{ color: AGENT_COLORS_CSS[i] }}>●</span>&nbsp;{name} — RKK Agent
          </div>
        ))}
        <div style={{ color: "#335566", marginBottom: 4 }}><span style={{ color: "#ff2244" }}>◆</span>&nbsp;Adversarial Demon</div>
        <div style={{ color: "#335566", marginBottom: 4 }}><span style={{ color: "#00ff99" }}>─○─</span>&nbsp;Causal graph (G*)</div>
        <div style={{ color: "#335566", marginBottom: 4 }}><span style={{ color: "#004488" }}>─</span>&nbsp;ToM link (Theory of Mind)</div>
        <div style={{ color: "#335566" }}><span style={{ color: "#335588" }}>·</span>&nbsp;Env discovery nodes</div>
        <div style={{ marginTop: 10, borderTop: "1px solid #081e30", paddingTop: 8, color: "#113344", fontSize: 9, lineHeight: 1.7 }}>
          <div>max E[d|G|/dt] — Intrinsic Drive</div>
          <div>ΔΦ(Gᵢ) ≥ 0 — Value Constraint</div>
          <div>α = sandbox trust (0→1)</div>
        </div>
      </div>

      {/* ── Phase bar ── */}
      <div style={{
        position: "absolute", top: 82, left: "50%", transform: "translateX(-50%)",
        background: "rgba(0,8,20,0.82)", border: "1px solid #081e30",
        padding: "8px 16px", borderRadius: 2, display: "flex", gap: 6, fontSize: 9,
      }}>
        {PHASE_NAMES.slice(1).map((name, i) => {
          const ph = i + 1;
          const active = ui.phase === ph;
          const done   = ui.phase > ph;
          return (
            <div key={ph} style={{
              padding: "3px 8px", borderRadius: 2,
              background: active ? "#001e38" : "transparent",
              border: `1px solid ${active ? "#00aaff" : done ? "#003322" : "#081e30"}`,
              color: active ? "#00aaff" : done ? "#00aa55" : "#223344",
              transition: "all 0.5s",
            }}>
              {done ? "✓" : ph}&nbsp;{name}
            </div>
          );
        })}
      </div>

      {/* ── Event log ── */}
      <div style={{
        position: "absolute", bottom: 14, left: 14, right: 14,
        background: "rgba(0,8,20,0.88)", border: "1px solid #081e30",
        padding: "10px 14px", borderRadius: 2, maxHeight: 148, overflow: "hidden",
      }}>
        <div style={{ color: "#113344", fontSize: 9, letterSpacing: "0.15em", marginBottom: 6 }}>
          CAUSAL EVENT STREAM
        </div>
        {ui.events.length === 0 && (
          <div style={{ color: "#1a3344", fontSize: 10 }}>Initialising discovery loop...</div>
        )}
        {ui.events.map((ev, i) => (
          <div key={i} style={{
            color: ev.color || "#445566", fontSize: 10, marginBottom: 3,
            opacity: Math.max(0.15, 1 - i * 0.11),
          }}>
            › {ev.text}
          </div>
        ))}
      </div>
    </div>
  );
}
