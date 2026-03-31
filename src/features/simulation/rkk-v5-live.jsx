import { useEffect, useRef, useMemo } from "react";
import * as THREE from "three";
import { useRKKStream } from "../../hooks/useRKKStream";

// ─── CONSTANTS ────────────────────────────────────────────────────────────────
const WORLD            = 14;
const AGENT_COLORS_HEX = [0x00ff99, 0x0099ff, 0xff9900];
const AGENT_COLORS_CSS = ["#00ff99", "#0099ff", "#ff9900"];
const AGENT_NAMES      = ["Nova", "Aether", "Lyra"];
const DEMON_COLOR      = 0xff2244;
const PHASE_NAMES      = ["", "Causal Crib", "Robotic Explorer", "Social Sandbox", "Value Lock", "Open Reality"];
const GRAPH_LINES      = 14;

export default function RKKv5() {
  const mountRef = useRef(null);
  const sceneRef = useRef(null);
  const rendererRef = useRef(null);
  
  // 5. Подключаем стрим (убираем Simulation)
  const { frame, connected, speed, setSpeed } = useRKKStream();

  // Ссылки на 3D объекты для прямого манипулирования (без ререндера React)
  const objs = useRef({
    agents: [],
    orbitNodes: [],
    causalLines: [],
    tomLines: [],
    demon: null
  });

  // ── INIT SCENE ─────────────────────────────────────────────────────────────
  useEffect(() => {
    const mount = mountRef.current;
    if (!mount) return;

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(mount.clientWidth, mount.clientHeight);
    mount.appendChild(renderer.domElement);
    rendererRef.current = renderer;

    const scene = new THREE.Scene();
    sceneRef.current = scene;
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

    // Создание агентов
    objs.current.agents = AGENT_COLORS_HEX.map((col, i) => {
      const g = new THREE.Group();
      const body = new THREE.Mesh(
        new THREE.SphereGeometry(0.6, 22, 22),
        new THREE.MeshPhongMaterial({ color: col, emissive: col, emissiveIntensity: 0.35, transparent: true, opacity: 0.92 })
      );
      const ring = new THREE.Mesh(
        new THREE.TorusGeometry(0.95, 0.05, 8, 44),
        new THREE.MeshBasicMaterial({ color: col, transparent: true, opacity: 0.4 })
      );
      ring.rotation.x = Math.PI / 2;
      g.add(body, ring, new THREE.PointLight(col, 0.7, 5));
      g.position.set((i - 1) * 8, 1, 0); // Начальная расстановка
      scene.add(g);
      return { group: g, ring, body };
    });

    // Создание графов (узлы и линии)
    objs.current.orbitNodes = objs.current.agents.map((_, ai) => 
      Array.from({ length: GRAPH_LINES }, () => {
        const m = new THREE.Mesh(new THREE.SphereGeometry(0.1, 6, 6), new THREE.MeshBasicMaterial({ color: AGENT_COLORS_HEX[ai], transparent: true, opacity: 0 }));
        scene.add(m);
        return m;
      })
    );

    objs.current.causalLines = objs.current.agents.map((_, ai) => 
      Array.from({ length: GRAPH_LINES }, () => {
        const l = new THREE.Line(
          new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), new THREE.Vector3()]), 
          new THREE.LineBasicMaterial({ color: AGENT_COLORS_HEX[ai], transparent: true, opacity: 0 })
        );
        scene.add(l);
        return l;
      })
    );

    // Demon
    const demon = new THREE.Mesh(
      new THREE.OctahedronGeometry(0.8, 0),
      new THREE.MeshPhongMaterial({ color: DEMON_COLOR, emissive: 0xff0022, emissiveIntensity: 0.7, transparent: true, opacity: 0.9 })
    );
    scene.add(demon);
    objs.current.demon = demon;

    // ToM Lines
    objs.current.tomLines = [];
    for (let a = 0; a < 3; a++) {
      for (let b = a + 1; b < 3; b++) {
        const line = new THREE.Line(
          new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), new THREE.Vector3()]), 
          new THREE.LineBasicMaterial({ color: 0x003366, transparent: true, opacity: 0 })
        );
        line.userData = { a, b };
        scene.add(line);
        objs.current.tomLines.push(line);
      }
    }

    // Render Loop
    let camAngle = 0;
    function animate() {
      camAngle += 0.002;
      camera.position.x = Math.sin(camAngle) * 30;
      camera.position.z = Math.cos(camAngle) * 30;
      camera.lookAt(0, 2, 0);
      renderer.render(scene, camera);
      requestAnimationFrame(animate);
    }
    animate();

    const onResize = () => {
      camera.aspect = mount.clientWidth / mount.clientHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(mount.clientWidth, mount.clientHeight);
    };
    window.addEventListener("resize", onResize);

    return () => {
      window.removeEventListener("resize", onResize);
      renderer.dispose();
      if (mount.contains(renderer.domElement)) mount.removeChild(renderer.domElement);
    };
  }, []);

  // ── SYNC DATA TO 3D ────────────────────────────────────────────────────────
  useEffect(() => {
    if (!frame || !sceneRef.current) return;
    const { agents, orbitNodes, causalLines, tomLines, demon } = objs.current;
    const t = frame.tick;

    // Агенты
    frame.agents.forEach((data, i) => {
      const a = agents[i];
      if (!a) return;

      // Левитация
      a.group.position.y = 1 + Math.sin(t * 0.1 + i) * 0.2;
      a.ring.rotation.z += 0.05;
      a.body.material.emissiveIntensity = 0.2 + data.alpha_mean;

      // Граф (узлы)
      const nodesCount = Math.min(data.edge_count + 2, GRAPH_LINES);
      orbitNodes[i].forEach((node, k) => {
        if (k < nodesCount) {
          const angle = (k / nodesCount) * Math.PI * 2 + t * 0.05;
          node.position.set(
            a.group.position.x + Math.cos(angle) * 2,
            a.group.position.y + Math.sin(t * 0.1 + k) * 0.3,
            a.group.position.z + Math.sin(angle) * 2
          );
          node.material.opacity = 0.5 + data.alpha_mean * 0.5;
        } else {
          node.material.opacity = 0;
        }
      });

      // Линии графа
      causalLines[i].forEach((line, k) => {
        if (k < nodesCount - 1) {
          line.geometry.setFromPoints([orbitNodes[i][k].position, orbitNodes[i][k+1].position]);
          line.material.opacity = 0.3;
        } else {
          line.material.opacity = 0;
        }
      });
    });

    // ToM
    tomLines.forEach(line => {
      const link = frame.tom_links?.find(l => l.a === line.userData.a && l.b === line.userData.b);
      if (link) {
        line.geometry.setFromPoints([agents[line.userData.a].group.position, agents[line.userData.b].group.position]);
        line.material.opacity = link.strength;
      } else {
        line.material.opacity = 0;
      }
    });

    // Demon
    if (demon) {
      demon.position.set(Math.sin(t * 0.05) * 10, 2, Math.cos(t * 0.05) * 10);
      demon.rotation.x += 0.1;
    }

  }, [frame]);

  // ── UI ──────────────────────────────────────────────────────────────────────
  const mono = { fontFamily: "'Courier New', monospace" };

  return (
    <div style={{ position: "relative", width: "100%", height: "100vh", background: "#010810", overflow: "hidden", ...mono }}>
      <div ref={mountRef} style={{ position: "absolute", inset: 0 }} />

      {/* Header */}
      <div style={{
        position: "absolute", top: 14, left: "50%", transform: "translateX(-50%)",
        background: "rgba(0,12,28,0.9)", border: "1px solid #0a2a44",
        padding: "8px 22px", textAlign: "center", borderRadius: 2, zIndex: 10
      }}>
        <div style={{ color: connected ? "#00ff99" : "#ff2244", fontSize: 13, fontWeight: "bold", letterSpacing: "0.18em" }}>
          RKK v5 — {connected ? "LIVE CAUSAL STREAM" : "DISCONNECTED"}
        </div>
        <div style={{ color: "#115577", fontSize: 10, marginTop: 3 }}>
          PHASE {frame.phase}/5 › {PHASE_NAMES[frame.phase]} &nbsp;│&nbsp; TICK: {frame.tick}
        </div>
      </div>

      {/* Speed control */}
      <div style={{ position: "absolute", top: 80, left: "50%", transform: "translateX(-50%)", display: "flex", gap: 10, zIndex: 10 }}>
        {[1, 2, 4, 8].map(s => (
          <button key={s} onClick={() => setSpeed(s)} style={{
            background: speed === s ? "#00aaff22" : "#000",
            color: speed === s ? "#00aaff" : "#335566",
            border: `1px solid ${speed === s ? "#00aaff" : "#112233"}`,
            padding: "4px 10px", fontSize: 10, cursor: "pointer"
          }}>
            {s}x
          </button>
        ))}
      </div>

      {/* Agent Panels */}
      <div style={{ position: "absolute", top: 120, left: 14, display: "flex", flexDirection: "column", gap: 8, zIndex: 10 }}>
        {frame.agents.map((a, i) => (
          <div key={i} style={{
            background: "rgba(0,8,20,0.9)", borderLeft: `3px solid ${AGENT_COLORS_CSS[i]}`,
            padding: "10px", minWidth: 200, fontSize: 10
          }}>
            <div style={{ color: AGENT_COLORS_CSS[i], fontWeight: "bold", marginBottom: 5 }}>◈ {a.name}</div>
            <div style={{ color: "#aad4ee" }}>Φ: {(a.phi * 100).toFixed(1)}%</div>
            <div style={{ color: "#aad4ee" }}>α: {Math.round(a.alpha_mean * 100)}%</div>
            <div style={{ color: "#aad4ee" }}>CG: {a.compression_gain.toFixed(3)}</div>
            <div style={{ color: "#335566", fontSize: 9, marginTop: 4 }}>Last: {a.last_do}</div>
          </div>
        ))}
      </div>

      {/* Event Stream */}
      <div style={{
        position: "absolute", bottom: 14, left: 14, right: 14,
        background: "rgba(0,8,20,0.9)", border: "1px solid #081e30",
        padding: "10px", maxHeight: 100, overflow: "hidden", zIndex: 10
      }}>
        {frame.events.slice(-4).reverse().map((ev, i) => (
          <div key={i} style={{ color: ev.color, fontSize: 10, marginBottom: 2 }}>
            [{ev.tick}] › {ev.text}
          </div>
        ))}
      </div>
    </div>
  );
}