import { useState, useEffect, useRef, useCallback } from "react";
import * as THREE from "three";
import { useRKKStream } from "../../hooks/useRKKStream";

const API = "http://localhost:8000";

const WORLD_COLORS = {
  humanoid:  "#cc44ff",
  robot:     "#aa22dd",
  pybullet:  "#ff44aa",
  physics:   "#00ff99",
  chemistry: "#0099ff",
  logic:     "#ff9900",
};
const WORLD_LABELS = {
  humanoid:  "Humanoid",
  robot:     "Robot Arm",
  pybullet:  "3D Physics",
  physics:   "Thermodynamics",
  chemistry: "Chemical Kinetics",
  logic:     "Logic Gates",
};

const phiColor = p => p>0.6?"#00ff99":p>0.3?"#aacc00":"#ff8844";
const hWColor  = h => h<0.01?"#00ff99":h<0.5?"#aacc00":h<2?"#ffaa00":"#ff4422";
const blkColor = r => r>0.3?"#ff4422":r>0.1?"#ffaa00":"#335544";
const cgColor  = c => c>0?"#00ff99":c<-0.05?"#ff4433":"#aaccdd";

function norm(a) {
  return {
    id:a.id??0, name:a.name??"Nova", envType:a.env_type??"—",
    compressionGain:a.compression_gain??0, alphaMean:a.alpha_mean??0.05,
    phi:a.phi??0.1, nodeCount:a.node_count??0, edgeCount:a.edge_count??0,
    totalInterventions:a.total_interventions??0, totalBlocked:a.total_blocked??0,
    lastDo:a.last_do??"—", lastBlockedReason:a.last_blocked_reason??"",
    discoveryRate:a.discovery_rate??0, hW:a.h_W??0,
    notears:a.notears??null, valueLayer:a.value_layer??null, edges:a.edges??[],
    fallen:a.fallen??false, fallCount:a.fall_count??0,
  };
}

function normFrame(raw) {
  const agent = norm((raw.agents??[])[0]??{});
  return {
    tick:raw.tick??0, phase:raw.phase??1, entropy:raw.entropy??100,
    agent, singleton:true,
    currentWorld:raw.current_world??"humanoid",
    worldLabel:raw.world_label??"",
    worldColor:raw.world_color??"#cc44ff",
    worlds:raw.worlds??{},
    switchHistory:raw.switch_history??[],
    gnnD:raw.gnn_d??0,
    demon:raw.demon??{energy:1,cooldown:0,mode:"probe",success_rate:0},
    events:raw.events??[],
    valueLayer:raw.value_layer??null,
    fallen:raw.fallen??false,
    fallCount:raw.fall_count??0,
    scene:raw.scene??{skeleton:[],cubes:[],target:{x:0,y:0,z:0.9},fallen:false},
  };
}

const PHASE_NAMES = ["","Causal Crib","Robotic Explorer","Social Sandbox","Value Lock","Open Reality"];
const sep = {borderTop:"1px solid #0a1a2e",marginTop:4,paddingTop:4};
const mono = {fontFamily:"'Courier New',monospace"};

// ── Скелет: 15 точек с бэка — без отдельного шара «грудь» для рук ────────────
// 0 голова, 1 шея (отсюда ключицы → плечи), 2 таз/root (ноги). Руки от шеи, не от лишнего узла под ней.
const SKELETON_BONES = [
  [0,1],[1,2],              // голова — шея — таз (ствол)
  [1,3],[3,5],[5,7],        // левая рука от шеи
  [1,4],[4,6],[6,8],        // правая рука от шеи
  [2,9],[9,11],[11,13],     // левая нога от таза
  [2,10],[10,12],[12,14],   // правая нога от таза
];

export default function RKKHumanoid() {
  const mountRef   = useRef(null);
  const rafRef     = useRef(null);
  const { frame: wsFrame, connected, setSpeed: wsSetSpeed } = useRKKStream();
  const wsFrameRef = useRef(wsFrame);
  wsFrameRef.current = wsFrame;

  const [speed,       setSpeedLocal] = useState(1);
  const [ui,          setUI]         = useState(() => normFrame(wsFrame));
  const [activePanel, setPanel]      = useState(null);
  const [seedText,    setSeedText]   = useState('[\n  {"from_": "lhip", "to": "com_z", "weight": 0.6}\n]');
  const [status,      setStatus]     = useState("");
  const [camView,     setCamView]    = useState("diag");
  const [camFrame,    setCamFrame]   = useState(null);
  const [showCam,     setShowCam]    = useState(false);
  const [ragLoading,  setRagLoading] = useState(false);
  const [showCubes,   setShowCubes]  = useState(false);

  const showCubesRef = useRef(showCubes);
  showCubesRef.current = showCubes;

  const setSpeed = s => { setSpeedLocal(s); if (connected) wsSetSpeed(s); };
  useEffect(() => { setUI(normFrame(wsFrame)); }, [wsFrame]);

  // Camera polling
  useEffect(() => {
    if (!connected || !showCam) return;
    const iv = setInterval(async () => {
      try {
        const d = await fetch(`${API}/camera/frame?view=${camView}`).then(r=>r.json());
        if (d.available) setCamFrame(d.frame);
      } catch {}
    }, 400);
    return () => clearInterval(iv);
  }, [connected, showCam, camView]);

  const switchWorld = async w => {
    try {
      const d = await fetch(`${API}/world/switch`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({world:w})}).then(r=>r.json());
      if (d.switched) setStatus(`✓ ${w} (+${d.new_nodes?.length??0} nodes)`);
    } catch(e) { setStatus(`✗ ${e.message}`); }
  };

  const injectSeeds = async () => {
    try {
      const edges = JSON.parse(seedText);
      const d = await fetch(`${API}/inject-seeds`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({agent_id:0,edges,source:"manual"})}).then(r=>r.json());
      setStatus(`✓ ${d.injected} edges`);
    } catch(e) { setStatus(`✗ ${e.message}`); }
  };

  const bootstrapHumanoid = async () => {
    try {
      const d = await fetch(`${API}/bootstrap/humanoid`,{method:"POST"}).then(r=>r.json());
      setStatus(`✓ Humanoid bootstrap: ${d.injected} edges`);
    } catch(e) { setStatus(`✗ ${e.message}`); }
  };

  const ragSeed = async () => {
    setRagLoading(true); setStatus("⏳ LLM generating…");
    try {
      const d = await fetch(`${API}/rag/auto-seed-all`,{method:"POST"}).then(r=>r.json());
      setStatus(`✓ RAG: ${d.results?.[0]?.injected??0} edges (${d.results?.[0]?.source})`);
    } catch(e) { setStatus(`✗ ${e.message}`); }
    finally { setRagLoading(false); }
  };

  // ── Three.js: humanoid-centric world ────────────────────────────────────────
  useEffect(() => {
    const mount = mountRef.current;
    if (!mount) return;

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type    = THREE.PCFSoftShadowMap;
    renderer.setSize(mount.clientWidth, mount.clientHeight);
    mount.appendChild(renderer.domElement);

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x030912);
    scene.fog = new THREE.FogExp2(0x030912, 0.025);

    const camera = new THREE.PerspectiveCamera(55, mount.clientWidth/mount.clientHeight, 0.1, 100);
    camera.position.set(0, 1.2, 5.5);
    camera.lookAt(0, 1.0, 0);

    // Освещение
    scene.add(new THREE.AmbientLight(0x0a1020, 3.0));
    const key = new THREE.DirectionalLight(0x8899ff, 2.5);
    key.position.set(3, 6, 4); key.castShadow = true;
    key.shadow.mapSize.set(1024, 1024);
    scene.add(key);
    const fill = new THREE.DirectionalLight(0x334455, 1.0);
    fill.position.set(-3, 2, -3); scene.add(fill);
    const rim  = new THREE.PointLight(0x6644ff, 1.5, 10);
    rim.position.set(0, 4, -2); scene.add(rim);

    // Пол с grid
    const floor = new THREE.Mesh(
      new THREE.PlaneGeometry(20, 20),
      new THREE.MeshStandardMaterial({
        color: 0x0a0f1a, roughness: 0.9, metalness: 0.1
      })
    );
    floor.rotation.x = -Math.PI/2; floor.receiveShadow = true;
    scene.add(floor);
    scene.add(new THREE.GridHelper(20, 40, 0x0d1a2e, 0x071220));

    // Рампа
    const rampGeo = new THREE.BoxGeometry(3, 0.1, 2);
    const rampMat = new THREE.MeshStandardMaterial({color:0x1a1510,roughness:0.8});
    const ramp = new THREE.Mesh(rampGeo, rampMat);
    ramp.position.set(3, 0.38, 0); ramp.rotation.x = -0.26; ramp.receiveShadow = true;
    scene.add(ramp);

    // ── Humanoid skeleton ─────────────────────────────────────────────────────
    // Суставы (spheres)
    const JOINT_COUNT = 15;
    const jointMeshes = Array.from({length:JOINT_COUNT}, (_, i) => {
      const isTorso = i <= 2;
      const m = new THREE.Mesh(
        new THREE.SphereGeometry(isTorso?0.065:0.05, 8, 8),
        new THREE.MeshStandardMaterial({
          color: isTorso?0xcc88ff:0x8844cc,
          emissive: isTorso?0x6622aa:0x441188,
          emissiveIntensity: 0.4,
          roughness: 0.3,
        })
      );
      m.castShadow = true;
      scene.add(m); return m;
    });

    // Кости (cylinders) — по одному на каждое ребро SKELETON_BONES
    const boneMeshes = SKELETON_BONES.map(([,]) => {
      const m = new THREE.Mesh(
        new THREE.CylinderGeometry(0.025, 0.025, 1, 6),
        new THREE.MeshStandardMaterial({
          color: 0x6633aa, emissive: 0x221144,
          emissiveIntensity: 0.2, roughness: 0.6,
        })
      );
      m.castShadow = true;
      scene.add(m); return m;
    });

    // Кубы (интерактивные объекты)
    const CUBE_COLORS = [0xff6622, 0x22aaff, 0x44ff88];
    const cubeMeshes = CUBE_COLORS.map((col, i) => {
      const size = 0.22 + i * 0.06;
      const m = new THREE.Mesh(
        new THREE.BoxGeometry(size, size, size),
        new THREE.MeshStandardMaterial({
          color: col, emissive: col,
          emissiveIntensity: 0.15, roughness: 0.4,
        })
      );
      m.castShadow = true; m.receiveShadow = true;
      m.position.set(1.5 + i*0.8, size/2, 0.5 - i*0.4);
      scene.add(m); return m;
    });

    // Целевая позиция (стойка прямо)
    const targetMesh = new THREE.Mesh(
      new THREE.RingGeometry(0.12, 0.18, 16),
      new THREE.MeshBasicMaterial({color:0x66ffaa, side:THREE.DoubleSide, transparent:true, opacity:0.4})
    );
    targetMesh.rotation.x = -Math.PI/2; targetMesh.position.y = 0.01;
    scene.add(targetMesh);

    // Fallen indicator: красное свечение под ногами
    const fallenLight = new THREE.PointLight(0xff0022, 0, 3);
    fallenLight.position.set(0, 0.1, 0); scene.add(fallenLight);

    // Particles
    const pPos = new Float32Array(600*3);
    for(let i=0;i<600;i++){
      pPos[i*3]=(Math.random()-.5)*30;
      pPos[i*3+1]=Math.random()*12;
      pPos[i*3+2]=(Math.random()-.5)*30;
    }
    const pGeom = new THREE.BufferGeometry();
    pGeom.setAttribute("position",new THREE.BufferAttribute(pPos,3));
    scene.add(new THREE.Points(pGeom,new THREE.PointsMaterial({color:0x220044,size:0.06,transparent:true,opacity:0.4})));

    let frame = 0;
    // Метры из API (FK skeleton + кубы) — без доп. масштаба; 3.5 ломал позы и тянул кости.
    const WORLD_SCALE = 1;
    const camTarget = new THREE.Vector3(0, 1, 0);
    // Орбита: азимут вокруг Y, угол от горизонтали, расстояние (как раньше ~5.5 м)
    let camAzim = 0;
    let camElev = Math.asin(0.3 / 5.5);
    let camRadius = 5.5;
    let camDrag = false;
    let camPtrX = 0;
    let camPtrY = 0;
    const CAM_ROT = 0.005;
    const CAM_ELEV_MIN = 0.08;
    const CAM_ELEV_MAX = Math.PI / 2 - 0.06;
    const CAM_R_MIN = 2;
    const CAM_R_MAX = 24;

    function updateBone(boneMesh, posA, posB) {
      const mid = new THREE.Vector3().addVectors(posA, posB).multiplyScalar(0.5);
      boneMesh.position.copy(mid);
      const dir = new THREE.Vector3().subVectors(posB, posA);
      const len = dir.length();
      boneMesh.scale.y = Math.max(0.01, len);
      if (len > 0.001) {
        boneMesh.lookAt(posB);
        boneMesh.rotateX(Math.PI/2);
      }
    }

    function loop() {
      rafRef.current = requestAnimationFrame(loop);
      frame++;
      const ds = normFrame(wsFrameRef.current);
      const ag = ds.agent;
      const wCol = parseInt((ds.worldColor||"#cc44ff").replace("#",""), 16);
      const fallen = ds.fallen || ds.scene?.fallen;

      // Центр орбиты — среднее по скелетону (PyBullet x,z → xz сцены)
      let comX = 0;
      let comZ = 0;
      const sk = ds.scene?.skeleton;
      if (sk && sk.length >= 3) {
        let sx = 0;
        let sz = 0;
        let n = 0;
        for (let j = 0; j < Math.min(sk.length, JOINT_COUNT); j++) {
          const pt = sk[j];
          if (!pt) continue;
          sx += (pt.x ?? 0) * WORLD_SCALE;
          sz += (pt.y ?? 0) * WORLD_SCALE;
          n++;
        }
        if (n > 0) { comX = sx / n; comZ = sz / n; }
      }
      camTarget.lerp(new THREE.Vector3(comX, 1.05, comZ), 0.06);
      const ch = Math.cos(camElev);
      camera.position.set(
        camTarget.x + camRadius * ch * Math.sin(camAzim),
        camTarget.y + camRadius * Math.sin(camElev),
        camTarget.z + camRadius * ch * Math.cos(camAzim)
      );
      camera.lookAt(camTarget);

      // Fallen indicator
      fallenLight.intensity = fallen ? 2.0 + Math.sin(frame*0.2)*0.5 : 0;

      // ── Skeleton ────────────────────────────────────────────────────────────
      const skeleton = ds.scene?.skeleton;
      const jointPositions = [];  // THREE.Vector3[]
      const showCubesNow = showCubesRef.current;

      if (skeleton && skeleton.length >= 3) {
        // Позиции суставов из backend (PyBullet world coords → Three.js)
        skeleton.slice(0, JOINT_COUNT).forEach((pt, i) => {
          const v = new THREE.Vector3(
            (pt.x ?? 0) * WORLD_SCALE,
            (pt.z ?? 0) * WORLD_SCALE,   // z PyBullet = вертикаль → y Three.js
            (pt.y ?? 0) * WORLD_SCALE
          );
          jointPositions.push(v);
          if (i < jointMeshes.length) {
            jointMeshes[i].position.copy(v);
            jointMeshes[i].visible = true;
            // Цветовая индикация: торс пульсирует по phi
            if (i === 0) {
              jointMeshes[i].material.emissiveIntensity = 0.3 + (ag.phi??0.1)*0.4 + Math.sin(frame*0.08)*0.1;
              jointMeshes[i].material.emissive.setHex(fallen?0xff2200:0x6622aa);
            }
          }
        });
        // Дополняем если суставов меньше
        for (let i=jointPositions.length; i<JOINT_COUNT; i++) {
          jointPositions.push(jointMeshes[Math.max(0,i-1)]?.position?.clone() ?? new THREE.Vector3());
        }
      } else {
        // Fallback: анимированный т-позовый гуманоид
        const t   = frame * 0.025;
        const phi = ag.phi ?? 0.1;
        const comH = 1.2 + (fallen ? -0.8 : 0) + Math.sin(t*0.5)*0.03;
        const poses = [
          [0, comH+0.26, 0],           // 0 голова
          [0, comH+0.13, 0],           // 1 шея — хаб плеч
          [0, comH-0.10, 0],           // 2 таз — хаб ног
          [-0.26, comH+0.11, 0],       // 3 левое плечо (у шеи по высоте)
          [ 0.26, comH+0.11, 0],       // 4 правое плечо
          [-0.42, comH+0.05+Math.sin(t+1)*0.06, 0], // 5 локоть
          [ 0.42, comH+0.05+Math.sin(t+2)*0.06, 0],
          [-0.50, comH-0.18, 0],       // 7 кисть
          [ 0.50, comH-0.18, 0],
          [-0.11, comH-0.22, 0],       // 9 бедро
          [ 0.11, comH-0.22, 0],
          [-0.11, comH-0.56+Math.sin(t)*0.05, 0], // 11 колено
          [ 0.11, comH-0.56+Math.sin(t+Math.PI)*0.05, 0],
          [-0.11+Math.sin(t)*0.04, comH-0.86, 0.05], // 13 стопа
          [ 0.11+Math.sin(t+Math.PI)*0.04, comH-0.86, 0.05],
        ];
        poses.forEach(([x,y,z],i) => {
          const v = new THREE.Vector3(x,y,z);
          jointPositions.push(v);
          if (i < jointMeshes.length) {
            jointMeshes[i].position.copy(v);
            jointMeshes[i].visible = true;
          }
        });
      }

      // Рисуем кости
      SKELETON_BONES.forEach(([a,b], k) => {
        if (a < jointPositions.length && b < jointPositions.length && k < boneMeshes.length) {
          updateBone(boneMeshes[k], jointPositions[a], jointPositions[b]);
          boneMeshes[k].visible = true;
          boneMeshes[k].material.color.setHex(fallen ? 0x881100 : wCol);
          boneMeshes[k].material.emissiveIntensity = 0.15 + Math.sin(frame*0.05+k)*0.05;
        }
      });

      // ── Кубы (симуляция) — по умолчанию скрыты, чтобы не перекрывать гуманоида
      const cubes = ds.scene?.cubes;
      cubeMeshes.forEach((cm, i) => {
        const c = cubes?.[i];
        const has = c && typeof c.x === "number";
        cm.visible = showCubesNow && has;
        if (has) {
          cm.position.set(c.x * WORLD_SCALE, Math.max(0, c.z??0) * WORLD_SCALE + cm.geometry.parameters.height/2, (c.y??0) * WORLD_SCALE);
        }
        if (showCubesNow) {
          cm.rotation.y += 0.005;
          cm.material.emissiveIntensity = 0.1 + Math.sin(frame*0.04+i)*0.05;
        }
      });

      renderer.render(scene, camera);
    }
    loop();

    const el = renderer.domElement;
    el.style.cursor = "grab";
    el.style.touchAction = "none";

    const onPointerDown = e => {
      if (e.button !== 0) return;
      camDrag = true;
      camPtrX = e.clientX;
      camPtrY = e.clientY;
      el.setPointerCapture(e.pointerId);
      el.style.cursor = "grabbing";
    };
    const onPointerMove = e => {
      if (!camDrag) return;
      const dx = e.clientX - camPtrX;
      const dy = e.clientY - camPtrY;
      camPtrX = e.clientX;
      camPtrY = e.clientY;
      camAzim -= dx * CAM_ROT;
      camElev -= dy * CAM_ROT;
      camElev = Math.max(CAM_ELEV_MIN, Math.min(CAM_ELEV_MAX, camElev));
    };
    const endDrag = e => {
      if (!camDrag) return;
      camDrag = false;
      try { el.releasePointerCapture(e.pointerId); } catch { /* already released */ }
      el.style.cursor = "grab";
    };
    const onWheel = e => {
      e.preventDefault();
      if (e.deltaY > 0) camRadius *= 1.1;
      else camRadius /= 1.1;
      camRadius = Math.max(CAM_R_MIN, Math.min(CAM_R_MAX, camRadius));
    };

    el.addEventListener("pointerdown", onPointerDown);
    el.addEventListener("pointermove", onPointerMove);
    el.addEventListener("pointerup", endDrag);
    el.addEventListener("pointercancel", endDrag);
    el.addEventListener("wheel", onWheel, { passive: false });

    const onResize = () => {
      if (!mount) return;
      camera.aspect = mount.clientWidth/mount.clientHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(mount.clientWidth, mount.clientHeight);
    };
    window.addEventListener("resize", onResize);
    return () => {
      cancelAnimationFrame(rafRef.current);
      el.removeEventListener("pointerdown", onPointerDown);
      el.removeEventListener("pointermove", onPointerMove);
      el.removeEventListener("pointerup", endDrag);
      el.removeEventListener("pointercancel", endDrag);
      el.removeEventListener("wheel", onWheel);
      window.removeEventListener("resize", onResize);
      if (mount.contains(renderer.domElement)) mount.removeChild(renderer.domElement);
      renderer.dispose();
    };
  }, []);

  const a    = ui.agent;
  const wCol = ui.worldColor;
  const phiC = phiColor(a.phi??0);
  const hWc  = hWColor(a.hW??0);
  const cgC  = cgColor(a.compressionGain??0);
  const blkR = a.valueLayer?.block_rate??0;
  const blkC = blkColor(blkR);
  const hasBlk = !!a.lastBlockedReason;
  const fallen = ui.fallen || a.fallen;

  return(
    <div style={{position:"relative",width:"100%",height:"100vh",background:"#030912",overflow:"hidden",...mono}}>
      <div ref={mountRef} style={{position:"absolute",inset:0}}/>

      {/* Camera overlay */}
      {showCam && camFrame && (
        <div style={{position:"absolute",bottom:120,right:14,border:`1px solid ${wCol}55`,borderRadius:3,overflow:"hidden",width:280,boxShadow:"0 0 20px #00000088"}}>
          <div style={{display:"flex",gap:4,padding:"3px 6px",background:"rgba(0,0,0,0.7)",fontSize:8}}>
            {["diag","side","front","top"].map(v=>(
              <button key={v} onClick={()=>setCamView(v)} style={{padding:"1px 5px",borderRadius:2,fontSize:7,cursor:"pointer",background:camView===v?"#441166":"transparent",border:`1px solid ${camView===v?wCol:"#332255"}`,color:camView===v?wCol:"#554477"}}>{v}</button>
            ))}
            <span style={{color:wCol,fontSize:7,marginLeft:"auto"}}>PyBullet 📷</span>
          </div>
          <img src={`data:image/jpeg;base64,${camFrame}`} style={{width:"100%",display:"block"}} alt="pybullet"/>
        </div>
      )}

      {/* Status */}
      <div style={{position:"absolute",top:14,right:14,background:connected?"rgba(0,30,10,0.9)":"rgba(20,0,0,0.9)",border:`1px solid ${connected?"#00aa44":"#aa4400"}`,padding:"4px 10px",borderRadius:2,fontSize:9,color:connected?"#00ff88":"#ff8844"}}>
        {connected?"● ONLINE":"○ OFFLINE"} · d={ui.gnnD} · {ui.currentWorld}
      </div>

      {/* Header */}
      <div style={{position:"absolute",top:14,left:"50%",transform:"translateX(-50%)",background:"rgba(2,6,16,0.92)",border:`1px solid ${wCol}44`,padding:"7px 22px",textAlign:"center",borderRadius:3,boxShadow:`0 0 30px ${wCol}22`,whiteSpace:"nowrap"}}>
        <div style={{color:fallen?"#ff4422":wCol,fontSize:12,fontWeight:"bold",letterSpacing:"0.18em",transition:"color 0.5s"}}>
          {fallen?"⚠ FALLEN — ":""}AGI NOVA — HUMANOID · GNN d={ui.gnnD}
        </div>
        <div style={{color:"#1a3355",fontSize:10,marginTop:2,letterSpacing:"0.06em"}}>
          PHASE {ui.phase}/5 › {PHASE_NAMES[ui.phase]}
          &nbsp;│&nbsp;ENT:<span style={{color:ui.entropy<30?"#00ff99":"#8899bb"}}>{ui.entropy}%</span>
          &nbsp;│&nbsp;T:{ui.tick}
          &nbsp;│&nbsp;<span style={{color:wCol}}>🌍 {ui.worldLabel||ui.currentWorld}</span>
          &nbsp;│&nbsp;<span style={{color:fallen?"#ff2244":"#335544"}}>
            {fallen?`💀×${ui.fallCount}` : "✓ standing"}
          </span>
        </div>
      </div>

      {/* Controls */}
      <div style={{position:"absolute",top:80,left:"50%",transform:"translateX(-50%)",background:"rgba(2,6,16,0.88)",border:"1px solid #0a1a2e",padding:"5px 12px",borderRadius:3,display:"flex",gap:5,alignItems:"center",fontSize:9,flexWrap:"wrap",justifyContent:"center"}}>
        <span style={{color:"#1a3355"}}>SPEED</span>
        {[1,2,4,8].map(s=>(
          <button key={s} onClick={()=>setSpeed(s)} style={{padding:"2px 7px",borderRadius:2,fontSize:9,cursor:"pointer",background:speed===s?"#080f22":"transparent",border:`1px solid ${speed===s?"#4455ff":"#0a1a2e"}`,color:speed===s?"#8899ff":"#334466"}}>{s}×</button>
        ))}
        <span style={{color:"#0a1a2e"}}>│</span>
        {Object.entries(WORLD_COLORS).map(([w,c])=>(
          <button key={w} onClick={()=>switchWorld(w)} style={{padding:"2px 7px",borderRadius:2,fontSize:9,cursor:"pointer",background:ui.currentWorld===w?"#0a0520":"transparent",border:`1px solid ${ui.currentWorld===w?c:"#111"}`,color:ui.currentWorld===w?c:"#334455"}}>
            {w}
          </button>
        ))}
        <span style={{color:"#0a1a2e"}}>│</span>
        {[["💉","seeds"],["🌐","rag"]].map(([icon,panel])=>(
          <button key={panel} onClick={()=>setPanel(v=>v===panel?null:panel)} style={{padding:"2px 7px",borderRadius:2,fontSize:9,cursor:"pointer",background:activePanel===panel?"#180a30":"transparent",border:`1px solid ${activePanel===panel?wCol:"#111"}`,color:activePanel===panel?wCol:"#334455"}}>{icon}</button>
        ))}
        <button onClick={()=>setShowCam(v=>!v)} style={{padding:"2px 7px",borderRadius:2,fontSize:9,cursor:"pointer",background:showCam?"#0a0520":"transparent",border:`1px solid ${showCam?wCol:"#111"}`,color:showCam?wCol:"#334455"}}>📷</button>
      </div>

      {/* Seeds panel */}
      {activePanel==="seeds"&&(
        <div style={{position:"absolute",top:118,left:"50%",transform:"translateX(-50%)",background:"rgba(5,3,12,0.97)",border:`1px solid ${wCol}44`,padding:"12px 16px",borderRadius:3,width:430,zIndex:10}}>
          <div style={{color:wCol,fontSize:10,marginBottom:6}}>💉 SEED INJECTION — {ui.currentWorld}</div>
          <div style={{fontSize:8,color:"#332255",marginBottom:4}}>d={ui.gnnD} vars · Seeds с α=0.05, NOTEARS/GNN выжжет ошибочные</div>
          <textarea value={seedText} onChange={e=>setSeedText(e.target.value)} style={{width:"100%",height:80,background:"#040210",border:`1px solid ${wCol}33`,color:"#aa88cc",fontSize:9,padding:6,borderRadius:2,fontFamily:"monospace",resize:"none",boxSizing:"border-box"}}/>
          <div style={{display:"flex",gap:6,marginTop:6,alignItems:"center",flexWrap:"wrap"}}>
            <button onClick={injectSeeds} style={{padding:"3px 10px",borderRadius:2,fontSize:9,cursor:"pointer",background:"#160830",border:`1px solid ${wCol}`,color:wCol}}>INJECT</button>
            <button onClick={bootstrapHumanoid} style={{padding:"3px 10px",borderRadius:2,fontSize:9,cursor:"pointer",background:"#160830",border:`1px solid #cc44ff`,color:"#cc44ff"}}>🤖 HUMANOID SEEDS</button>
            <span style={{color:status.startsWith("✓")?"#00ff99":"#ff4422",fontSize:9}}>{status}</span>
          </div>
        </div>
      )}

      {/* RAG panel */}
      {activePanel==="rag"&&(
        <div style={{position:"absolute",top:118,left:"50%",transform:"translateX(-50%)",background:"rgba(0,4,10,0.97)",border:"1px solid #003322",padding:"12px 16px",borderRadius:3,width:400,zIndex:10}}>
          <div style={{color:"#00aa66",fontSize:10,marginBottom:6}}>🌐 LLM BOOTSTRAP — культурная память AGI</div>
          <div style={{fontSize:8,color:"#1a3322",marginBottom:6,lineHeight:1.7}}>
            LLM читает биомеханику → гипотезы рёбер графа.<br/>
            Агент стартует с человеческим опытом, а не с нуля.
          </div>
          <button onClick={ragSeed} disabled={ragLoading} style={{width:"100%",padding:"5px",borderRadius:2,fontSize:9,cursor:"pointer",background:ragLoading?"#001a0a":"#002211",border:"1px solid #006633",color:ragLoading?"#225544":"#00ff88",marginBottom:6}}>
            {ragLoading?"⏳ LLM генерирует…":"🌐 AUTO-SEED (Wikipedia + LLM)"}
          </button>
          {ui.switchHistory.length>0&&<div style={{...sep}}>
            <div style={{fontSize:8,color:"#1a3322",marginBottom:2}}>WORLD HISTORY:</div>
            {ui.switchHistory.slice(-4).map((h,i)=>(
              <div key={i} style={{fontSize:8,color:"#336655"}}>
                {h.from_world} → {h.to_world} (+{h.new_nodes?.length??0} vars, d={h.gnn_d})
              </div>
            ))}
          </div>}
          <div style={{marginTop:4,fontSize:9,color:status.startsWith("✓")?"#00ff99":"#ff4422"}}>{status}</div>
        </div>
      )}

      {/* Left HUD: AGI Core metrics */}
      <div style={{position:"absolute",top:118,left:14}}>
        <div style={{background:"rgba(2,5,14,0.92)",border:`1px solid ${fallen?"#660011":wCol+"33"}`,borderLeft:`3px solid ${fallen?"#ff2244":wCol}`,padding:"10px 14px",minWidth:235,borderRadius:3,boxShadow:`0 0 20px ${fallen?"#ff224422":wCol+"18"}`,transition:"border-color 0.5s"}}>
          {/* Header */}
          <div style={{color:fallen?"#ff4422":wCol,fontSize:11,fontWeight:"bold",marginBottom:5,letterSpacing:"0.08em"}}>
            {fallen?"💀":"◈"} NOVA — Singleton AGI
            <span style={{color:"#1a2244",marginLeft:6,fontSize:8}}>GNN · {ui.currentWorld}</span>
          </div>

          {/* Fallen indicator */}
          {fallen&&<div style={{background:"rgba(255,0,34,0.08)",border:"1px solid #ff224433",borderRadius:2,padding:"3px 7px",marginBottom:4,fontSize:8,color:"#ff4422"}}>
            ⚠ FALLEN × {ui.fallCount} — Value Layer штрафует
          </div>}

          {/* Metrics table */}
          <table style={{borderCollapse:"collapse",width:"100%",fontSize:9}}><tbody>
            {[
              ["Φ autonomy",    <span style={{color:phiC}}>{((a.phi??0)*100).toFixed(1)}%</span>],
              ["CG d|G|/dt",    <span style={{color:cgC}}>{(a.compressionGain??0)>=0?"+":""}{(a.compressionGain??0).toFixed(4)}</span>],
              ["α trust",       `${Math.round((a.alphaMean??0)*100)}%`],
              ["h(W) DAG",      <span style={{color:hWc}}>{(a.hW??0).toFixed(4)}</span>],
              ["GNN d",         <span style={{color:wCol}}>{ui.gnnD}</span>],
              ["Edges",         a.edgeCount??0],
              ["do() / blk",    `${a.totalInterventions??0} / ${a.totalBlocked??0}`],
              ["Discovery",     <span style={{color:(a.discoveryRate??0)>0.2?"#00ff99":"#aabbcc"}}>{((a.discoveryRate??0)*100).toFixed(0)}%</span>],
            ].map(([l,v],k)=>(
              <tr key={k}><td style={{color:"#2a4466",paddingRight:8,paddingBottom:2}}>{l}</td><td style={{color:"#aabbcc",textAlign:"right"}}>{v}</td></tr>
            ))}
          </tbody></table>

          {/* VL status */}
          {a.valueLayer&&<div style={{...sep,fontSize:8}}>
            <div style={{display:"flex",justifyContent:"space-between"}}>
              <span style={{color:"#441133"}}>VALUE LAYER {a.valueLayer.vl_phase}</span>
              <span style={{color:blkC}}>blk:{(blkR*100).toFixed(1)}%</span>
            </div>
            {hasBlk&&<div style={{color:"#ff6644",fontSize:7,marginTop:1}}>⚠ {a.lastBlockedReason}</div>}
          </div>}

          {/* GNN / NOTEARS */}
          {a.notears&&<div style={{...sep,fontSize:8}}>
            <div style={{display:"flex",justifyContent:"space-between"}}>
              <span style={{color:"#2a4433"}}>GNN steps={a.notears.steps}</span>
              <span style={{color:a.notears.loss<0.01?"#00ff99":"#aa8800"}}>L={a.notears.loss?.toFixed(5)}</span>
            </div>
          </div>}

          {/* Progress bars */}
          {[
            {w:`${Math.round((a.alphaMean??0)*100)}%`,f:"#110022",t:wCol},
            {w:`${Math.round((a.phi??0)*100)}%`,f:"#220011",t:phiC},
            {w:`${Math.max(0,100-Math.min((a.hW??0)*20,100))}%`,f:"#001811",t:hWc},
            {w:`${Math.min(blkR*100,100)}%`,f:"#110000",t:blkC},
          ].map((b,k)=>(
            <div key={k} style={{marginTop:k===0?5:2,height:k===0?3:2,background:"#050a18",borderRadius:2,overflow:"hidden"}}>
              <div style={{width:b.w,height:"100%",background:`linear-gradient(90deg,${b.f},${b.t})`,transition:"width 0.8s"}}/>
            </div>
          ))}
          <div style={{fontSize:7,marginTop:2,display:"flex",justifyContent:"space-between",color:"#1a2244"}}>
            <span>α·Φ·h·vl</span>
            <span style={{color:"#1a3333"}}>dr:{((a.discoveryRate??0)*100).toFixed(0)}%</span>
          </div>
        </div>
      </div>

      {/* Right: Legend + roadmap */}
      <div style={{position:"absolute",top:118,right:14,background:"rgba(2,5,14,0.92)",border:"1px solid #0a1a2e",padding:"10px 14px",borderRadius:3,fontSize:9,maxWidth:190}}>
        <div style={{color:"#1a3355",marginBottom:6,fontSize:9}}>SINGLETON AGI — HUMANOID</div>
        <div style={{color:"#336655",marginBottom:2,fontSize:8}}>▲ скелетон = joint positions</div>
        <div style={{color:"#442255",marginBottom:2,fontSize:8}}>■ кубы = объекты сцены (PyBullet)</div>
        <label style={{display:"flex",alignItems:"center",gap:6,cursor:"pointer",color:"#556688",fontSize:8,marginBottom:6}}>
          <input type="checkbox" checked={showCubes} onChange={e=>setShowCubes(e.target.checked)} style={{accentColor:wCol}} />
          показать кубы
        </label>
        <div style={{color:"#442244",marginBottom:2,fontSize:8}}>📷 кнопка = PyBullet camera</div>

        <div style={{...sep}}>
          <div style={{color:"#1a2244",fontSize:8,marginBottom:3}}>ACTIVE WORLD:</div>
          {Object.entries(WORLD_COLORS).map(([w,c])=>(
            <div key={w} onClick={()=>switchWorld(w)} style={{fontSize:8,marginBottom:2,cursor:"pointer",color:ui.currentWorld===w?c:"#223344",paddingLeft:ui.currentWorld===w?4:0,transition:"all 0.2s"}}>
              {ui.currentWorld===w?"▶ ":"  "}{w}
            </div>
          ))}
        </div>

        <div style={{...sep,lineHeight:1.9}}>
          <div style={{color:"#1a2244",fontSize:8}}>ROADMAP:</div>
          <div style={{color:wCol,fontSize:8}}>✓ 11. Humanoid Singleton</div>
          <div style={{color:"#224433",fontSize:8}}>→ 12. Causal Vision</div>
          <div style={{color:"#224433",fontSize:8}}>→ 13. N-step Imagination</div>
          <div style={{color:"#224433",fontSize:8}}>→ 14. Cross-Domain</div>
          <div style={{color:"#224433",fontSize:8}}>→ 15. Reality Bridge</div>
        </div>
      </div>

      {/* Event log */}
      <div style={{position:"absolute",bottom:14,left:14,right:14,background:"rgba(2,5,14,0.92)",border:"1px solid #0a1a2e",padding:"8px 14px",borderRadius:3,maxHeight:100,overflow:"hidden"}}>
        <div style={{color:"#0a1a2e",fontSize:9,letterSpacing:"0.1em",marginBottom:4}}>
          CAUSAL EVENT STREAM · Nova Singleton {connected?"● ONLINE":"○ OFFLINE"}
        </div>
        {ui.events.map((ev,i)=>(
          <div key={i} style={{color:ev.color??"#334455",fontSize:9,marginBottom:2,opacity:Math.max(0.15,1-i*0.12),fontWeight:ev.type==="value"?"bold":"normal"}}>
            [{String(ev.tick??0).padStart(5,"0")}] › {ev.text}
          </div>
        ))}
      </div>
    </div>
  );
}