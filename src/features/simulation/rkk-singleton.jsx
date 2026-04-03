import { useState, useEffect, useRef, useCallback } from "react";
import * as THREE from "three";
import { useRKKStream } from "../../hooks/useRKKStream";

const API = "http://localhost:8000";

// ── Константы ─────────────────────────────────────────────────────────────────
const WORLD_COLORS = {
  physics:   "#00ff99",
  chemistry: "#0099ff",
  logic:     "#ff9900",
  robot:     "#cc44ff",
  pybullet:  "#ff44aa",
};

const WORLD_LABELS = {
  physics:   "Thermodynamics",
  chemistry: "Chemical Kinetics",
  logic:     "Logic Gates",
  robot:     "Robot Arm",
  pybullet:  "3D Physics",
};

const hWColor  = h => h<0.01?"#00ff99":h<0.5?"#aacc00":h<2?"#ffaa00":"#ff4422";
const phiColor = p => p>0.6?"#00ff99":p>0.3?"#aacc00":"#ff8844";
const blkColor = r => r>0.3?"#ff4422":r>0.1?"#ffaa00":"#335544";

function normalizeAgent(a) {
  return {
    id:a.id??0, name:a.name??"Nova", envType:a.env_type??"—",
    graphMdl:a.graph_mdl??0, compressionGain:a.compression_gain??0,
    alphaMean:a.alpha_mean??0.05, phi:a.phi??0.1,
    nodeCount:a.node_count??0, edgeCount:a.edge_count??0,
    totalInterventions:a.total_interventions??0, totalBlocked:a.total_blocked??0,
    lastDo:a.last_do??"—", lastBlockedReason:a.last_blocked_reason??"",
    discoveryRate:a.discovery_rate??0, hW:a.h_W??0,
    notears:a.notears??null, valueLayer:a.value_layer??null, edges:a.edges??[],
  };
}

function normalizeFrame(raw) {
  const agent = normalizeAgent((raw.agents??[])[0]??{});
  return {
    tick:raw.tick??0, phase:raw.phase??1, entropy:raw.entropy??100,
    agent, singleton:raw.singleton??true,
    currentWorld:raw.current_world??"robot",
    worldLabel:raw.world_label??"",
    worldColor:raw.world_color??"#cc44ff",
    worlds:raw.worlds??{},
    switchHistory:raw.switch_history??[],
    gnnD:raw.gnn_d??0,
    demon:raw.demon??{energy:1,cooldown:0,mode:"probe",success_rate:0},
    events:raw.events??[],
    valueLayer:raw.value_layer??null,
    robotSkeleton:raw.robot_skeleton??null,
    robotTarget:raw.robot_target??null,
  };
}

const PHASE_NAMES = ["","Causal Crib","Robotic Explorer","Social Sandbox","Value Lock","Open Reality"];
const sep = {borderTop:"1px solid #081e30",marginTop:4,paddingTop:4};
const mono = {fontFamily:"'Courier New',monospace"};

export default function RKKSingleton() {
  const mountRef  = useRef(null);
  const rafRef    = useRef(null);
  const { frame: wsFrame, connected, setSpeed: wsSetSpeed } = useRKKStream();
  const wsFrameRef = useRef(wsFrame);
  wsFrameRef.current = wsFrame;

  const [speed,       setSpeedLocal]  = useState(1);
  const [ui,          setUI]          = useState(() => normalizeFrame(wsFrame));
  const [activePanel, setActivePanel] = useState(null);
  const [seedText,    setSeedText]    = useState('[\n  {"from_": "j0_vel", "to": "j0_pos", "weight": 0.8}\n]');
  const [seedStatus,  setSeedStatus]  = useState("");
  const [cameraFrame, setCameraFrame] = useState(null);
  const [showCamera,  setShowCamera]  = useState(false);
  const [demonStats,  setDemonStats]  = useState(null);
  const [ragLoading,  setRagLoading]  = useState(false);

  const setSpeed = s => { setSpeedLocal(s); if (connected) wsSetSpeed(s); };
  useEffect(() => { setUI(normalizeFrame(wsFrame)); }, [wsFrame]);

  // Camera polling
  useEffect(() => {
    if (!connected || !showCamera) return;
    const iv = setInterval(async () => {
      try {
        const d = await fetch(`${API}/camera/frame`).then(r=>r.json());
        if (d.available) setCameraFrame(d.frame);
      } catch {}
    }, 500);
    return () => clearInterval(iv);
  }, [connected, showCamera]);

  // Demon stats polling
  useEffect(() => {
    if (!connected) return;
    const iv = setInterval(async () => {
      try { setDemonStats(await fetch(`${API}/demon/stats`).then(r=>r.json())); } catch {}
    }, 3000);
    return () => clearInterval(iv);
  }, [connected]);

  const switchWorld = async (world) => {
    try {
      const r = await fetch(`${API}/world/switch`, {
        method:"POST", headers:{"Content-Type":"application/json"},
        body:JSON.stringify({world}),
      });
      const d = await r.json();
      if (d.switched) setSeedStatus(`✓ Switched to ${world} (+${d.new_nodes?.length??0} nodes)`);
    } catch(e) { setSeedStatus(`✗ ${e.message}`); }
  };

  const injectSeeds = async () => {
    try {
      const edges = JSON.parse(seedText);
      const d = await fetch(`${API}/inject-seeds`, {
        method:"POST", headers:{"Content-Type":"application/json"},
        body:JSON.stringify({agent_id:0, edges, source:"manual"}),
      }).then(r=>r.json());
      setSeedStatus(`✓ ${d.injected} edges → Nova`);
    } catch(e) { setSeedStatus(`✗ ${e.message}`); }
  };

  const bootstrapRobot = async () => {
    try {
      const d = await fetch(`${API}/bootstrap/robot`,{method:"POST"}).then(r=>r.json());
      setSeedStatus(`✓ Robot bootstrap: ${d.injected} edges`);
    } catch(e) { setSeedStatus(`✗ ${e.message}`); }
  };

  const ragAutoSeed = async () => {
    setRagLoading(true); setSeedStatus("⏳ LLM seeding…");
    try {
      const d = await fetch(`${API}/rag/auto-seed-all`,{method:"POST"}).then(r=>r.json());
      setSeedStatus(`✓ RAG: ${d.results?.[0]?.injected??0} edges`);
    } catch(e) { setSeedStatus(`✗ ${e.message}`); }
    finally { setRagLoading(false); }
  };

  // ── Three.js scene ──────────────────────────────────────────────────────────
  useEffect(() => {
    const mount = mountRef.current;
    if (!mount) return;

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(mount.clientWidth, mount.clientHeight);
    mount.appendChild(renderer.domElement);

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x010810);
    scene.fog = new THREE.FogExp2(0x010810, 0.015);
    const camera = new THREE.PerspectiveCamera(50, mount.clientWidth/mount.clientHeight, 0.1, 200);
    camera.position.set(0, 12, 22); camera.lookAt(0, 0, 0);
    scene.add(new THREE.AmbientLight(0x112233, 2.0));
    const sun = new THREE.DirectionalLight(0x334488, 1.5); sun.position.set(5,10,5); scene.add(sun);
    scene.add(new THREE.GridHelper(30, 30, 0x0a1e38, 0x050f1e));

    // ── AGI Core sphere ─────────────────────────────────────────────────────
    const agiGroup = new THREE.Group();
    const agiBody = new THREE.Mesh(
      new THREE.SphereGeometry(1.0, 32, 32),
      new THREE.MeshPhongMaterial({color:0x00ff99,emissive:0x00ff99,emissiveIntensity:0.5,transparent:true,opacity:0.92})
    );
    // Rings — one per GNN layer conceptually
    const rings = [1.4, 1.9, 2.5].map((r, i) => {
      const ring = new THREE.Mesh(
        new THREE.TorusGeometry(r, 0.03 - i*0.005, 6, 40),
        new THREE.MeshBasicMaterial({color:0x00ff99, transparent:true, opacity:0.4-i*0.1})
      );
      ring.rotation.x = Math.PI / (2 + i);
      return ring;
    });
    const agiLight = new THREE.PointLight(0x00ff99, 1.2, 8);
    agiGroup.add(agiBody, ...rings, agiLight);
    agiGroup.position.set(0, 1.5, 0);
    scene.add(agiGroup);

    // ── Robot skeleton ──────────────────────────────────────────────────────
    const robotGroup = new THREE.Group();
    robotGroup.position.set(5, 0, 0);
    scene.add(robotGroup);

    const jointMeshes = [];
    const boneMeshes  = [];

    // Создаём N+1 суставов и N костей
    for (let i = 0; i < 5; i++) {
      const jMesh = new THREE.Mesh(
        new THREE.SphereGeometry(0.12, 8, 8),
        new THREE.MeshPhongMaterial({color:0xcc44ff, emissive:0xaa22cc, emissiveIntensity:0.3})
      );
      robotGroup.add(jMesh);
      jointMeshes.push(jMesh);

      if (i < 4) {
        const boneMesh = new THREE.Mesh(
          new THREE.CylinderGeometry(0.04, 0.04, 1, 6),
          new THREE.MeshPhongMaterial({color:0x883399, transparent:true, opacity:0.8})
        );
        robotGroup.add(boneMesh);
        boneMeshes.push(boneMesh);
      }
    }

    // Target sphere
    const targetMesh = new THREE.Mesh(
      new THREE.SphereGeometry(0.15, 10, 10),
      new THREE.MeshPhongMaterial({color:0xff4422, emissive:0xff2200, emissiveIntensity:0.6, transparent:true, opacity:0.8})
    );
    targetMesh.add(new THREE.PointLight(0xff4422, 0.5, 2));
    scene.add(targetMesh);

    // ── Orbit nodes (causal graph visualization) ─────────────────────────────
    const N_ORBIT = 18;
    const orbitNodes = Array.from({length:N_ORBIT}, () => {
      const m = new THREE.Mesh(
        new THREE.SphereGeometry(0.08, 6, 6),
        new THREE.MeshBasicMaterial({color:0x00ff99, transparent:true, opacity:0})
      );
      scene.add(m); return m;
    });
    const orbitLines = Array.from({length:N_ORBIT}, () => {
      const geom = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), new THREE.Vector3()]);
      const l = new THREE.Line(geom, new THREE.LineBasicMaterial({color:0x00ff99, transparent:true, opacity:0}));
      scene.add(l); return l;
    });

    // ── Demon ────────────────────────────────────────────────────────────────
    const demon = new THREE.Mesh(
      new THREE.OctahedronGeometry(0.6, 0),
      new THREE.MeshPhongMaterial({color:0xff2244,emissive:0xff0022,emissiveIntensity:0.8,transparent:true,opacity:0.9})
    );
    demon.add(new THREE.PointLight(0xff2244, 0.7, 5));
    demon.position.set(10, 1, 8);
    demon.userData = {vel: new THREE.Vector3(-0.02,0,-0.015)};
    scene.add(demon);

    // Particles
    const pPos = new Float32Array(400*3);
    for(let i=0;i<400;i++){pPos[i*3]=(Math.random()-.5)*40;pPos[i*3+1]=Math.random()*12;pPos[i*3+2]=(Math.random()-.5)*40;}
    const pGeom = new THREE.BufferGeometry();
    pGeom.setAttribute("position",new THREE.BufferAttribute(pPos,3));
    scene.add(new THREE.Points(pGeom, new THREE.PointsMaterial({color:0x003399,size:0.07,transparent:true,opacity:0.3})));

    let frame=0, camAngle=0;
    const ROBOT_SCALE = 3.0;   // масштаб скелетона в Three.js

    function loop(){
      rafRef.current = requestAnimationFrame(loop);
      frame++;
      const ds = normalizeFrame(wsFrameRef.current);
      const ag = ds.agent;
      const wCol = parseInt((ds.worldColor||"#00ff99").replace("#",""), 16);

      camAngle += 0.0007;
      camera.position.x = Math.sin(camAngle) * 22;
      camera.position.z = Math.cos(camAngle) * 22;
      camera.lookAt(0, 1, 0);

      // AGI Core animation
      const phi = ag.phi??0.1;
      const cg  = ag.compressionGain??0;
      agiBody.material.color.setHex(wCol);
      agiBody.material.emissive.setHex(wCol);
      agiBody.material.emissiveIntensity = 0.2 + Math.max(0,cg)*0.2 + Math.sin(frame*0.07)*0.08 + phi*0.15;
      agiGroup.position.y = 1.5 + Math.sin(frame*0.04)*0.2;
      rings.forEach((r,i) => {
        r.rotation.z += 0.012 + i*0.005 + (ag.alphaMean??0.05)*0.04;
        r.rotation.x += 0.005 + i*0.003;
        r.material.color.setHex(wCol);
        r.material.opacity = 0.3 - i*0.08 + (phi*0.15);
      });
      agiLight.color.setHex(wCol);
      agiLight.intensity = 0.8 + phi*0.5;

      // Orbit nodes (represent GNN d nodes)
      const d = Math.max(ds.gnnD??0, ag.nodeCount??6);
      const visN = Math.min(d, N_ORBIT);
      const tickPhase = ds.tick * 0.01;
      orbitNodes.forEach((node, k) => {
        if (k < visN) {
          const a2 = (k/visN)*Math.PI*2 + tickPhase + frame*0.008;
          const r  = 3.0 + (k%3)*0.8;
          node.position.set(
            agiGroup.position.x + Math.cos(a2)*r,
            agiGroup.position.y + Math.sin(frame*0.03+k*0.5)*0.4,
            agiGroup.position.z + Math.sin(a2)*r
          );
          node.material.color.setHex(wCol);
          node.material.opacity = 0.3 + (ag.alphaMean??0.05)*0.5;
        } else node.material.opacity = 0;
      });
      orbitLines.forEach((l,k) => {
        if (k < visN-1) {
          const pa = orbitNodes[k].position, pb = orbitNodes[(k+1)%visN].position;
          const p  = l.geometry.attributes.position;
          p.setXYZ(0,pa.x,pa.y,pa.z); p.setXYZ(1,pb.x,pb.y,pb.z); p.needsUpdate=true;
          l.material.color.setHex((ag.edges[k]?.weight??0)<0?0xff4422:wCol);
          l.material.opacity = 0.2 + (ag.alphaMean??0.05)*0.4;
        } else l.material.opacity = 0;
      });

      // Robot skeleton from backend data
      const skeleton = ds.robotSkeleton;
      if (skeleton && skeleton.length >= 2) {
        skeleton.forEach((pt, i) => {
          if (i < jointMeshes.length) {
            jointMeshes[i].position.set(
              robotGroup.position.x + pt.x * ROBOT_SCALE,
              pt.z * ROBOT_SCALE + 0.5,
              pt.y * ROBOT_SCALE
            );
            jointMeshes[i].visible = true;
          }
        });
        // Обновляем кости между суставами
        boneMeshes.forEach((bone, k) => {
          if (k+1 < skeleton.length && k < jointMeshes.length-1) {
            const a = jointMeshes[k].position;
            const b = jointMeshes[k+1].position;
            const mid = new THREE.Vector3().addVectors(a, b).multiplyScalar(0.5);
            bone.position.copy(mid);
            const dir = new THREE.Vector3().subVectors(b, a);
            const len = dir.length();
            bone.scale.y = len;
            bone.lookAt(b);
            bone.rotateX(Math.PI/2);
            bone.visible = true;
          } else bone.visible = false;
        });
      } else {
        // Анимация-заглушка
        const t = frame * 0.02;
        for (let i=0;i<jointMeshes.length;i++){
          const angle = t + i * 0.8;
          const r = 0.3 * i;
          jointMeshes[i].position.set(
            robotGroup.position.x + Math.cos(angle)*r,
            0.5 + i * 0.35,
            Math.sin(angle)*r * 0.5
          );
        }
      }

      // Target position
      if (ds.robotTarget && typeof ds.robotTarget.x === "number") {
        targetMesh.position.set(
          robotGroup.position.x + ds.robotTarget.x * ROBOT_SCALE,
          (ds.robotTarget.z??0) * ROBOT_SCALE + 0.5,
          (ds.robotTarget.y??0) * ROBOT_SCALE
        );
        targetMesh.scale.setScalar(0.9 + Math.sin(frame*0.1)*0.1);
      }

      // Demon chasing AGI
      const demonDir = agiGroup.position.clone().sub(demon.position).normalize().multiplyScalar(0.022);
      demon.userData.vel.lerp(demonDir, 0.06);
      demon.position.add(demon.userData.vel);
      demon.position.y = 1.0 + (ds.demon?.energy??1)*0.4;
      demon.rotation.y += 0.04; demon.rotation.x += 0.02;
      const dMode = ds.demon?.mode??"probe";
      demon.material.emissiveIntensity = dMode==="siege"?1.0:0.6;

      renderer.render(scene, camera);
    }
    loop();

    const onResize = () => {
      if (!mount) return;
      camera.aspect = mount.clientWidth/mount.clientHeight;
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

  const a = ui.agent;
  const phiC = phiColor(a.phi??0);
  const hWc  = hWColor(a.hW??0);
  const cgC  = (a.compressionGain??0)>0?"#00ff99":(a.compressionGain??0)<-0.05?"#ff4433":"#aaccdd";
  const blkR = a.valueLayer?.block_rate??0;
  const blkC = blkColor(blkR);
  const hasBlk = !!a.lastBlockedReason;
  const wCol = ui.worldColor;
  const totalBlocked = ui.valueLayer?.total_blocked_all??a.totalBlocked??0;

  return(
    <div style={{position:"relative",width:"100%",height:"100vh",background:"#010810",overflow:"hidden",...mono}}>
      <div ref={mountRef} style={{position:"absolute",inset:0}}/>

      {/* Status */}
      <div style={{position:"absolute",top:14,right:14,background:connected?"rgba(0,40,10,0.9)":"rgba(30,10,0,0.9)",border:`1px solid ${connected?"#00aa44":"#aa4400"}`,padding:"4px 10px",borderRadius:2,fontSize:9,color:connected?"#00ff88":"#ff8844"}}>
        {connected?"● PYTHON":"○ OFFLINE"} · AGI · d={ui.gnnD}
      </div>

      {/* Camera overlay */}
      {showCamera && cameraFrame && (
        <div style={{position:"absolute",top:14,right:130,border:"1px solid #441188",borderRadius:2,overflow:"hidden",width:200}}>
          <div style={{fontSize:8,color:"#441166",background:"rgba(0,0,0,0.8)",padding:"2px 6px"}}>📷 PyBullet Camera</div>
          <img src={`data:image/png;base64,${cameraFrame}`} style={{width:"100%",display:"block"}} alt="robot cam"/>
        </div>
      )}

      {/* Header */}
      <div style={{position:"absolute",top:14,left:"50%",transform:"translateX(-50%)",background:"rgba(0,12,28,0.9)",border:"1px solid #0a2a44",padding:"7px 22px",textAlign:"center",borderRadius:2,boxShadow:"0 0 24px #00224455",whiteSpace:"nowrap"}}>
        <div style={{color:wCol,fontSize:13,fontWeight:"bold",letterSpacing:"0.18em"}}>
          AGI NOVA — SINGLETON · GNN · {ui.worldLabel||ui.currentWorld.toUpperCase()}
        </div>
        <div style={{color:"#115577",fontSize:10,marginTop:2,letterSpacing:"0.08em"}}>
          PHASE {ui.phase}/5 › {PHASE_NAMES[ui.phase]}
          &nbsp;│&nbsp;ENT:<span style={{color:ui.entropy<30?"#00ff99":"#aaccdd"}}>{ui.entropy}%</span>
          &nbsp;│&nbsp;T:{ui.tick}
          &nbsp;│&nbsp;<span style={{color:wCol}}>🌍 {ui.currentWorld}</span>
          &nbsp;│&nbsp;GNN d={ui.gnnD}
          &nbsp;│&nbsp;<span style={{color:totalBlocked>0?"#ff8844":"#335544"}}>🛡{totalBlocked}</span>
          &nbsp;│&nbsp;<span style={{color:"#ff2244"}}>◆{ui.demon?.mode}</span>
        </div>
      </div>

      {/* Controls */}
      <div style={{position:"absolute",top:80,left:"50%",transform:"translateX(-50%)",background:"rgba(0,8,20,0.85)",border:"1px solid #081e30",padding:"5px 12px",borderRadius:2,display:"flex",gap:6,alignItems:"center",fontSize:9}}>
        <span style={{color:"#224455"}}>SPEED</span>
        {[1,2,4,8].map(s=><button key={s} onClick={()=>setSpeed(s)} style={{padding:"2px 7px",borderRadius:2,fontSize:9,cursor:"pointer",background:speed===s?"#001e38":"transparent",border:`1px solid ${speed===s?"#00aaff":"#081e30"}`,color:speed===s?"#00aaff":"#334455"}}>{s}×</button>)}
        <span style={{color:"#334455"}}>│</span>
        {/* World switcher */}
        {Object.entries(WORLD_COLORS).map(([world, col])=>(
          <button key={world} onClick={()=>switchWorld(world)} style={{padding:"2px 7px",borderRadius:2,fontSize:9,cursor:"pointer",background:ui.currentWorld===world?"#0a0020":"transparent",border:`1px solid ${ui.currentWorld===world?col:"#222233"}`,color:ui.currentWorld===world?col:"#445566"}}>
            {world}
          </button>
        ))}
        <span style={{color:"#334455"}}>│</span>
        {[["💉","seeds"],["🌐","rag"],["◆","demon"]].map(([icon,panel])=>(
          <button key={panel} onClick={()=>setActivePanel(v=>v===panel?null:panel)} style={{padding:"2px 7px",borderRadius:2,fontSize:9,cursor:"pointer",background:activePanel===panel?"#1a0010":"transparent",border:`1px solid ${activePanel===panel?"#884400":"#333"}`,color:activePanel===panel?wCol:"#445555"}}>{icon}</button>
        ))}
        <button onClick={()=>setShowCamera(v=>!v)} style={{padding:"2px 7px",borderRadius:2,fontSize:9,cursor:"pointer",background:showCamera?"#0a0020":"transparent",border:"1px solid #441188",color:showCamera?"#cc44ff":"#334455"}}>📷</button>
      </div>

      {/* Seeds panel */}
      {activePanel==="seeds"&&(
        <div style={{position:"absolute",top:118,left:"50%",transform:"translateX(-50%)",background:"rgba(8,6,0,0.96)",border:"1px solid #443300",padding:"12px 16px",borderRadius:2,width:420,zIndex:10}}>
          <div style={{color:"#886600",fontSize:10,marginBottom:6}}>💉 SEED INJECTION — {ui.currentWorld}</div>
          <div style={{fontSize:8,color:"#554400",marginBottom:6}}>Current vars: {ui.gnnD} nodes in GNN</div>
          <textarea value={seedText} onChange={e=>setSeedText(e.target.value)} style={{width:"100%",height:80,background:"#050300",border:"1px solid #332200",color:"#aa8800",fontSize:9,padding:6,borderRadius:2,fontFamily:"monospace",resize:"none",boxSizing:"border-box"}}/>
          <div style={{display:"flex",gap:6,marginTop:6,alignItems:"center"}}>
            <button onClick={injectSeeds} style={{padding:"3px 10px",borderRadius:2,fontSize:9,cursor:"pointer",background:"#221100",border:"1px solid #886600",color:"#ffaa00"}}>INJECT</button>
            <button onClick={bootstrapRobot} style={{padding:"3px 10px",borderRadius:2,fontSize:9,cursor:"pointer",background:"#1a0020",border:`1px solid ${wCol}`,color:wCol}}>🤖 ROBOT SEEDS</button>
            <span style={{color:seedStatus.startsWith("✓")?"#00ff99":"#ff4422",fontSize:9}}>{seedStatus}</span>
          </div>
        </div>
      )}

      {/* RAG panel */}
      {activePanel==="rag"&&(
        <div style={{position:"absolute",top:118,left:"50%",transform:"translateX(-50%)",background:"rgba(0,6,14,0.96)",border:"1px solid #004433",padding:"12px 16px",borderRadius:2,width:400,zIndex:10}}>
          <div style={{color:"#00aa66",fontSize:10,marginBottom:6}}>🌐 LLM/RAG BOOTSTRAP</div>
          <div style={{fontSize:8,color:"#225544",marginBottom:8,lineHeight:1.7}}>
            LLM читает физику/химию → генерирует гипотезы рёбер.<br/>
            'Cultural memory' — агент не начинает с нуля.
          </div>
          <button onClick={ragAutoSeed} disabled={ragLoading} style={{width:"100%",padding:"5px",borderRadius:2,fontSize:9,cursor:"pointer",background:ragLoading?"#001a0a":"#002211",border:"1px solid #006633",color:ragLoading?"#225544":"#00ff88",marginBottom:6}}>
            {ragLoading?"⏳ LLM generating…":"🌐 AUTO-SEED (Wikipedia + LLM)"}
          </button>
          {/* Switch history */}
          {ui.switchHistory.length>0&&<div style={{...sep}}>
            <div style={{fontSize:8,color:"#225544",marginBottom:3}}>WORLD HISTORY:</div>
            {ui.switchHistory.slice(-3).map((h,i)=>(
              <div key={i} style={{fontSize:8,color:"#336655",marginBottom:1}}>
                {h.from_world} → {h.to_world} (+{h.new_nodes?.length??0} nodes)
              </div>
            ))}
          </div>}
          <div style={{marginTop:4,fontSize:9,color:seedStatus.startsWith("✓")?"#00ff99":"#ff4422"}}>{seedStatus}</div>
        </div>
      )}

      {/* Demon panel */}
      {activePanel==="demon"&&demonStats&&(
        <div style={{position:"absolute",top:118,left:"50%",transform:"translateX(-50%)",background:"rgba(14,0,0,0.96)",border:"1px solid #440000",padding:"12px 16px",borderRadius:2,width:300,zIndex:10}}>
          <div style={{color:"#ff4422",fontSize:10,marginBottom:6}}>◆ DEMON — PPO-lite</div>
          <table style={{width:"100%",borderCollapse:"collapse",fontSize:9}}><tbody>
            {[["Mode",demonStats.mode?.toUpperCase()],["Energy",`${(demonStats.energy*100).toFixed(0)}%`],["Success",`${((demonStats.success_rate??0)*100).toFixed(1)}%`]].map(([l,v],k)=>(
              <tr key={k}><td style={{color:"#553333",paddingRight:10,paddingBottom:3}}>{l}</td><td style={{color:"#cc6644",textAlign:"right"}}>{v}</td></tr>
            ))}
          </tbody></table>
        </div>
      )}

      {/* Left panel: AGI Core stats */}
      <div style={{position:"absolute",top:118,left:14}}>
        <div style={{background:"rgba(0,8,20,0.93)",border:`1px solid ${hasBlk?"#882200":wCol+"33"}`,borderLeft:`3px solid ${hasBlk?"#ff4422":wCol}`,padding:"10px 14px",minWidth:240,borderRadius:2,boxShadow:`0 0 18px ${wCol}18`}}>
          <div style={{color:wCol,fontSize:12,fontWeight:"bold",marginBottom:6,letterSpacing:"0.1em"}}>
            ◈ NOVA — Singleton AGI
            <span style={{color:"#1a3344",marginLeft:8,fontSize:8}}>{ui.currentWorld} · GNN</span>
          </div>
          <table style={{borderCollapse:"collapse",width:"100%",fontSize:10}}><tbody>
            {[
              ["Φ autonomy",    <span style={{color:phiC}}>{((a.phi??0)*100).toFixed(1)}%</span>],
              ["CG (d|G|/dt)",  <span style={{color:cgC}}>{(a.compressionGain??0)>=0?"+":""}{(a.compressionGain??0).toFixed(4)}</span>],
              ["α trust mean",  `${Math.round((a.alphaMean??0)*100)}%`],
              ["h(W) DAG",      <span style={{color:hWc}}>{(a.hW??0).toFixed(4)}</span>],
              ["GNN nodes (d)", <span style={{color:wCol}}>{ui.gnnD}</span>],
              ["Edges active",  a.edgeCount??0],
              ["do() / blk",   `${a.totalInterventions??0} / ${a.totalBlocked??0}`],
              ["Discovery",     <span style={{color:(a.discoveryRate??0)>0.3?"#00ff99":"#aaccdd"}}>{((a.discoveryRate??0)*100).toFixed(0)}%</span>],
              ["Last do()",     <span style={{color:"#334466",fontSize:8}}>{a.lastDo}</span>],
            ].map(([l,v],k)=>(
              <tr key={k}><td style={{color:"#335566",paddingRight:8,paddingBottom:2}}>{l}</td><td style={{color:"#aad4ee",textAlign:"right"}}>{v}</td></tr>
            ))}
          </tbody></table>

          {/* Value Layer */}
          {a.valueLayer&&<div style={{...sep,fontSize:8}}>
            <div style={{display:"flex",justifyContent:"space-between"}}>
              <span style={{color:"#553322"}}>VALUE LAYER {a.valueLayer.vl_phase}</span>
              <span style={{color:blkC}}>blk:{(blkR*100).toFixed(1)}%</span>
            </div>
            {hasBlk&&<div style={{color:"#ff6644",fontSize:8}}>⚠ {a.lastBlockedReason}</div>}
          </div>}

          {/* NOTEARS/GNN */}
          {a.notears&&<div style={{...sep,fontSize:8}}>
            <div style={{display:"flex",justifyContent:"space-between"}}>
              <span style={{color:"#336644"}}>GNN {a.notears.steps}s</span>
              <span style={{color:a.notears.loss<0.01?"#00ff99":"#aa8800"}}>L={a.notears.loss?.toFixed(5)}</span>
            </div>
          </div>}

          {/* Bars */}
          {[
            {w:`${Math.round((a.alphaMean??0)*100)}%`,f:"#002266",t:wCol},
            {w:`${Math.round((a.phi??0)*100)}%`,f:"#221100",t:phiC},
            {w:`${Math.max(0,100-Math.min((a.hW??0)*20,100))}%`,f:"#001800",t:hWc},
            {w:`${Math.min(blkR*100,100)}%`,f:"#110000",t:blkC},
          ].map((b,k)=>(
            <div key={k} style={{marginTop:k===0?6:2,height:k===0?3:2,background:"#061422",borderRadius:2,overflow:"hidden"}}>
              <div style={{width:b.w,height:"100%",background:`linear-gradient(90deg,${b.f},${b.t})`,transition:"width 0.8s"}}/>
            </div>
          ))}
          <div style={{color:"#1a3344",fontSize:7,marginTop:2,display:"flex",justifyContent:"space-between"}}>
            <span>α·Φ·h·vl</span><span style={{color:"#1a4433"}}>dr:{((a.discoveryRate??0)*100).toFixed(0)}%</span>
          </div>
        </div>
      </div>

      {/* Right panel: Legend + roadmap */}
      <div style={{position:"absolute",top:118,right:14,background:"rgba(0,8,20,0.93)",border:"1px solid #081e30",padding:"10px 14px",borderRadius:2,fontSize:9,maxWidth:185}}>
        <div style={{color:"#114466",marginBottom:6,fontSize:9}}>SINGLETON AGI v5</div>
        <div style={{color:"#335566",marginBottom:4}}>
          <span style={{color:wCol}}>●</span> Nova — Singleton Core
        </div>
        <div style={{color:"#335566",marginBottom:2}}>GNN d={ui.gnnD} → resize_to()</div>
        <div style={{color:"#335566",marginBottom:2}}>No Byzantine (no peers)</div>
        <div style={{color:"#335566",marginBottom:2}}>WorldSwitcher preserves W</div>

        {/* World list */}
        <div style={{...sep,marginBottom:4}}>
          <div style={{color:"#003355",fontSize:8,marginBottom:3}}>WORLDS:</div>
          {Object.entries(WORLD_COLORS).map(([w,c])=>(
            <div key={w} style={{fontSize:8,marginBottom:2,cursor:"pointer",color:ui.currentWorld===w?c:"#334455"}}
              onClick={()=>switchWorld(w)}>
              {ui.currentWorld===w?"▶":""} {w} — {WORLD_LABELS[w]}
            </div>
          ))}
        </div>

        {/* Roadmap */}
        <div style={{...sep,lineHeight:1.8}}>
          <div style={{color:"#003344",fontSize:8}}>ROADMAP:</div>
          <div style={{color:"#00ff99",fontSize:8}}>✓ 11. Singleton + Robot</div>
          <div style={{color:"#334455",fontSize:8}}>→ 12. Causal Vision (Slots)</div>
          <div style={{color:"#334455",fontSize:8}}>→ 13. Imagination N-step</div>
          <div style={{color:"#334455",fontSize:8}}>→ 14. Cross-Domain Transfer</div>
          <div style={{color:"#334455",fontSize:8}}>→ 15. Reality Bridge</div>
        </div>
      </div>

      {/* Event log */}
      <div style={{position:"absolute",bottom:14,left:14,right:14,background:"rgba(0,8,20,0.93)",border:"1px solid #081e30",padding:"8px 14px",borderRadius:2,maxHeight:110,overflow:"hidden"}}>
        <div style={{color:"#113344",fontSize:9,letterSpacing:"0.1em",marginBottom:4}}>
          CAUSAL EVENT STREAM — Singleton AGI {connected?"● ONLINE":"○ OFFLINE"}
        </div>
        {ui.events.map((ev,i)=>(
          <div key={i} style={{color:ev.color??"#335566",fontSize:10,marginBottom:2,opacity:Math.max(0.15,1-i*0.1),fontWeight:ev.type==="value"?"bold":"normal"}}>
            [{String(ev.tick??0).padStart(4,"0")}] › {ev.text}
          </div>
        ))}
      </div>
    </div>
  );
}
