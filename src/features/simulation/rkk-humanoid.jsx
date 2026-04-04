import { useState, useEffect, useRef } from "react";
import * as THREE from "three";
import { useRKKStream } from "../../hooks/useRKKStream";

const API = "http://localhost:8000";
/** Опрос PyBullet-превью (не связан с тиками симуляции) */
const CAMERA_PREVIEW_MS = 500;
/** query `view` для /camera/frame (бэкенд гуманоида пока игнорирует — всегда first-person) */
const CAM_VIEW = "fp";

const WORLD_COLORS = {
  humanoid:"#cc44ff", robot:"#aa22dd", pybullet:"#ff44aa",
  physics:"#00ff99", chemistry:"#0099ff", logic:"#ff9900",
};
const WORLD_LABELS = {
  humanoid:"Humanoid", robot:"Robot Arm", pybullet:"3D Physics",
  physics:"Thermodynamics", chemistry:"Chemical Kinetics", logic:"Logic Gates",
};

// Цвета слотов (те же что на бэке)
const SLOT_COLORS = [
  "#ff5050","#50c8ff","#50ff64","#ffc850",
  "#c850ff","#ff8c50","#50ffdc","#b4b4ff",
];

/** Красный «демон» в Three.js-сцене (AdversarialDemon только визуализация) */
const SHOW_SCENE_DEMON = false;

const phiC  = p => p>0.6?"#00ff99":p>0.3?"#aacc00":"#ff8844";
const hWC   = h => h<0.01?"#00ff99":h<0.5?"#aacc00":h<2?"#ffaa00":"#ff4422";
const blkC  = r => r>0.3?"#ff4422":r>0.1?"#ffaa00":"#335544";
const cgC   = c => c>0?"#00ff99":c<-0.05?"#ff4433":"#aaccdd";

function normAgent(a) {
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
  const agent = normAgent((raw.agents??[])[0]??{});
  return {
    tick:raw.tick??0, phase:raw.phase??1, entropy:raw.entropy??100,
    agent, singleton:true,
    currentWorld:raw.current_world??"humanoid",
    worldLabel:raw.world_label??"", worldColor:raw.world_color??"#cc44ff",
    worlds:raw.worlds??{}, switchHistory:raw.switch_history??[],
    gnnD:raw.gnn_d??0,
    demon:raw.demon??{energy:1,cooldown:0,mode:"probe",success_rate:0},
    events:raw.events??[], valueLayer:raw.value_layer??null,
    fallen:raw.fallen??false, fallCount:raw.fall_count??0,
    scene:raw.scene??{skeleton:[],cubes:[],target:{x:0,y:0,z:0.9},fallen:false},
    // Фаза 12
    visualMode:raw.visual_mode??false,
    visionTicks:raw.vision_ticks??0,
    vision:raw.vision??null,
  };
}

const PHASE_NAMES = ["","Causal Crib","Robotic Explorer","Social Sandbox","Value Lock","Open Reality"];
const sep  = {borderTop:"1px solid #0a1a2e",marginTop:4,paddingTop:4};
const mono = {fontFamily:"'Courier New',monospace"};

const SKELETON_BONES = [
  [0,1],[1,2],[2,3],
  [1,4],[4,6],[6,8], [1,5],[5,7],[7,9],
  [3,10],[10,12],[12,14], [3,11],[11,13],[13,15],
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
  const [camFrame,    setCamFrame]   = useState(null);
  const [showCam,     setShowCam]    = useState(false);
  const [showCubes,   setShowCubes]  = useState(false);
  const [ragLoading,  setRagLoading] = useState(false);

  // Фаза 12
  const [visionData,   setVisionData]   = useState(null);
  const [visionLoading,setVisionLoading]= useState(false);
  const [vlmLoading,    setVlmLoading]    = useState(false);
  const [vlmWeakEdges,  setVlmWeakEdges]  = useState(false);
  const [selectedSlot, setSelectedSlot] = useState(0);
  const [attnFrame,    setAttnFrame]    = useState(null);
  const [visionEnabled,setVisionEnabled]= useState(false);

  const showCubesRef = useRef(showCubes);
  showCubesRef.current = showCubes;

  const setSpeed = s => { setSpeedLocal(s); if (connected) wsSetSpeed(s); };
  useEffect(() => {
    const f = normFrame(wsFrame);
    setUI(f);
    setVisionEnabled(f.visualMode);
  }, [wsFrame]);

  // Camera polling
  useEffect(() => {
    if (!connected || !showCam) return;
    const iv = setInterval(async () => {
      try {
        const d = await fetch(`${API}/camera/frame?view=${CAM_VIEW}`).then(r=>r.json());
        if (d.available) setCamFrame(d.frame);
      } catch {}
    }, CAMERA_PREVIEW_MS);
    return () => clearInterval(iv);
  }, [connected, showCam]);

  // Vision slots polling (только когда visual mode и панель открыта)
  useEffect(() => {
    if (!connected || !visionEnabled || activePanel !== "vision") return;
    const iv = setInterval(async () => {
      try {
        const d = await fetch(`${API}/vision/slots`).then(r=>r.json());
        if (d.visual_mode !== false) setVisionData(d);
      } catch {}
    }, 800);
    return () => clearInterval(iv);
  }, [connected, visionEnabled, activePanel]);

  // Attn frame для выбранного слота
  useEffect(() => {
    if (!visionEnabled || activePanel !== "vision") return;
    const fetchAttn = async () => {
      try {
        const d = await fetch(`${API}/vision/attn_frame?slot_idx=${selectedSlot}`).then(r=>r.json());
        if (d.available) setAttnFrame(d.frame);
      } catch {}
    };
    fetchAttn();
  }, [selectedSlot, visionEnabled, activePanel]);

  const switchWorld = async w => {
    try {
      const d = await fetch(`${API}/world/switch`,{method:"POST",
        headers:{"Content-Type":"application/json"},body:JSON.stringify({world:w})}).then(r=>r.json());
      if (d.switched) setStatus(`✓ ${w} (+${d.new_nodes?.length??0} nodes)`);
    } catch(e) { setStatus(`✗ ${e.message}`); }
  };

  const injectSeeds = async () => {
    try {
      const edges = JSON.parse(seedText);
      const d = await fetch(`${API}/inject-seeds`,{method:"POST",
        headers:{"Content-Type":"application/json"},
        body:JSON.stringify({agent_id:0,edges,source:"manual"})}).then(r=>r.json());
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

  // ── Фаза 12 действия ──────────────────────────────────────────────────────
  const enableVision = async (nSlots=8) => {
    setVisionLoading(true);
    try {
      const d = await fetch(`${API}/vision/enable`,{method:"POST",
        headers:{"Content-Type":"application/json"},
        body:JSON.stringify({n_slots:nSlots,mode:"hybrid"})}).then(r=>r.json());
      if (d.visual) {
        setVisionEnabled(true);
        setStatus(`👁 Vision ON: ${d.n_slots} slots, d=${d.gnn_d}`);
        setPanel("vision");
      } else {
        setStatus(`✗ ${d.error||"failed"}`);
      }
    } catch(e) { setStatus(`✗ ${e.message}`); }
    finally { setVisionLoading(false); }
  };

  const runVlmSlotLabels = async () => {
    setVlmLoading(true);
    try {
      const d = await fetch(`${API}/vision/vlm-label`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          llm_url: "http://localhost:11434/api/generate",
          llm_model: "gemma4:e4b",
          max_mask_images: 4,
          text_only: false,
          inject_weak_edges: vlmWeakEdges,
        }),
      }).then((r) => r.json());
      if (d.ok) {
        const w = d.weak_edges_injected ? ` · +${d.weak_edges_injected} weak edges` : "";
        const warn = d.warning ? ` (${d.warning})` : "";
        setStatus(`🔬 VLM ${d.mode}: ${d.n_slots_labeled} labels${w}${warn}`);
        try {
          const refresh = await fetch(`${API}/vision/slots`).then((r) => r.json());
          if (refresh.visual_mode !== false) setVisionData(refresh);
        } catch {}
      } else {
        setStatus(`✗ VLM: ${d.error || "failed"}`);
      }
    } catch (e) {
      setStatus(`✗ ${e.message}`);
    } finally {
      setVlmLoading(false);
    }
  };

  const disableVision = async () => {
    try {
      await fetch(`${API}/vision/disable`,{method:"POST"});
      setVisionEnabled(false);
      setVisionData(null);
      setAttnFrame(null);
      setStatus("👁 Vision OFF — ручные переменные");
    } catch(e) { setStatus(`✗ ${e.message}`); }
  };

  // ── Three.js ───────────────────────────────────────────────────────────────
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
    scene.fog = new THREE.FogExp2(0x030912, 0.022);

    const camera = new THREE.PerspectiveCamera(55, mount.clientWidth/mount.clientHeight, 0.1, 100);
    camera.position.set(0, 1.2, 5.5);

    scene.add(new THREE.AmbientLight(0x0a1020, 3.0));
    const key = new THREE.DirectionalLight(0x8899ff, 2.5);
    key.position.set(3,6,4); key.castShadow=true; scene.add(key);
    scene.add(new THREE.DirectionalLight(0x334455, 1.0).position.set(-3,2,-3) && new THREE.DirectionalLight(0x334455, 1.0));
    const rim = new THREE.PointLight(0x6644ff, 1.5, 10);
    rim.position.set(0,4,-2); scene.add(rim);

    const floor = new THREE.Mesh(
      new THREE.PlaneGeometry(20,20),
      new THREE.MeshStandardMaterial({color:0x0a0f1a,roughness:0.9})
    );
    floor.rotation.x=-Math.PI/2; floor.receiveShadow=true; scene.add(floor);
    scene.add(new THREE.GridHelper(20,40,0x0d1a2e,0x071220));

    const ramp = new THREE.Mesh(
      new THREE.BoxGeometry(3,0.1,2),
      new THREE.MeshStandardMaterial({color:0x1a1510,roughness:0.8})
    );
    ramp.position.set(3,0.38,0); ramp.rotation.x=-0.26; scene.add(ramp);

    // Гуманоид
    const JOINT_COUNT = 18;
    const jointMeshes = Array.from({length:JOINT_COUNT},(_,i)=>{
      if (i >= 16) {
        const o = new THREE.Object3D();
        o.visible = false;
        scene.add(o);
        return o;
      }
      const isTorso = i <= 2;
      const geom = new THREE.SphereGeometry(isTorso ? 0.065 : 0.05, 8, 8);
      const mat = new THREE.MeshStandardMaterial({
        color: isTorso ? 0xcc88ff : 0x8844cc,
        emissive: isTorso ? 0x6622aa : 0x441188,
        emissiveIntensity: 0.4,
        roughness: 0.3,
      });
      const m = new THREE.Mesh(geom, mat);
      m.castShadow = true;
      scene.add(m);
      return m;
    });

    const boneMeshes = SKELETON_BONES.map(()=>{
      const m = new THREE.Mesh(
        new THREE.CylinderGeometry(0.025,0.025,1,6),
        new THREE.MeshStandardMaterial({color:0x6633aa,emissive:0x221144,emissiveIntensity:0.2,roughness:0.6})
      );
      m.castShadow=true; scene.add(m); return m;
    });

    // Кубы
    const CUBE_COLORS_HEX = [0xff6622,0x22aaff,0x44ff88];
    const cubeMeshes = CUBE_COLORS_HEX.map((col,i)=>{
      const size=0.22+i*0.06;
      const m = new THREE.Mesh(
        new THREE.BoxGeometry(size,size,size),
        new THREE.MeshStandardMaterial({color:col,emissive:col,emissiveIntensity:0.15,roughness:0.4})
      );
      m.castShadow=true; m.receiveShadow=true;
      m.position.set(1.5+i*0.8,size/2,0.5-i*0.4); scene.add(m); return m;
    });

    // Target ring
    const targetMesh = new THREE.Mesh(
      new THREE.RingGeometry(0.12,0.18,16),
      new THREE.MeshBasicMaterial({color:0x66ffaa,side:THREE.DoubleSide,transparent:true,opacity:0.4})
    );
    targetMesh.rotation.x=-Math.PI/2; targetMesh.position.y=0.01; scene.add(targetMesh);

    const fallenLight = new THREE.PointLight(0xff0022,0,3);
    fallenLight.position.set(0,0.1,0); scene.add(fallenLight);

    // Фаза 12: слот-сферы (K=8 маленьких орбитальных шаров, цветных по слоту)
    const slotSpheres = Array.from({length:8},(_, k)=>{
      const col = parseInt(SLOT_COLORS[k].replace("#",""), 16);
      const m = new THREE.Mesh(
        new THREE.SphereGeometry(0.05,8,8),
        new THREE.MeshStandardMaterial({
          color:col, emissive:col, emissiveIntensity:0.5,
          transparent:true, opacity:0.0,
        })
      );
      scene.add(m); return m;
    });

    // Visual mode glow ring вокруг персонажа
    const visionRing = new THREE.Mesh(
      new THREE.TorusGeometry(0.6,0.02,8,32),
      new THREE.MeshBasicMaterial({color:0x44ffcc,transparent:true,opacity:0.0})
    );
    scene.add(visionRing);

    let demon = null;
    if (SHOW_SCENE_DEMON) {
      demon = new THREE.Mesh(
        new THREE.OctahedronGeometry(0.35,0),
        new THREE.MeshStandardMaterial({color:0xff2244,emissive:0xff0022,emissiveIntensity:0.9,roughness:0.2})
      );
      demon.add(new THREE.PointLight(0xff2244,0.8,4));
      demon.position.set(4,0.8,2);
      demon.userData={vel:new THREE.Vector3(-0.015,0,-0.01)};
      scene.add(demon);
    }

    // Particles
    const pPos = new Float32Array(600*3);
    for(let i=0;i<600;i++){pPos[i*3]=(Math.random()-.5)*30;pPos[i*3+1]=Math.random()*12;pPos[i*3+2]=(Math.random()-.5)*30;}
    const pGeom = new THREE.BufferGeometry();
    pGeom.setAttribute("position",new THREE.BufferAttribute(pPos,3));
    scene.add(new THREE.Points(pGeom,new THREE.PointsMaterial({color:0x220044,size:0.06,transparent:true,opacity:0.4})));

    let frame=0;
    const camTarget = new THREE.Vector3(0,1,0);
    let camAzim=0, camElev=0.2, camRadius=5.5;
    let camDrag=false, camPtrX=0, camPtrY=0;

    function updateBone(b,a,bp){
      const mid=new THREE.Vector3().addVectors(a,bp).multiplyScalar(0.5);
      b.position.copy(mid);
      const dir=new THREE.Vector3().subVectors(bp,a);
      const len=dir.length();
      b.scale.y=Math.max(0.01,len);
      if(len>0.001){b.lookAt(bp);b.rotateX(Math.PI/2);}
    }

    function loop(){
      rafRef.current=requestAnimationFrame(loop);
      frame++;
      const ds=normFrame(wsFrameRef.current);
      const ag=ds.agent;
      const wCol=parseInt((ds.worldColor||"#cc44ff").replace("#",""),16);
      const fallen=ds.fallen||ds.scene?.fallen;
      const isVis=ds.visualMode;

      // Камера
      let comX=0,comZ=0;
      const sk=ds.scene?.skeleton;
      if(sk&&sk.length>=3){
        let sx=0,sz=0,n=0;
        for(let j=0;j<Math.min(sk.length,JOINT_COUNT);j++){
          const pt=sk[j]; if(!pt)continue;
          sx+=pt.x??0; sz+=pt.y??0; n++;
        }
        if(n>0){comX=sx/n;comZ=sz/n;}
      }
      camTarget.lerp(new THREE.Vector3(comX,1.05,comZ),0.05);
      const ch=Math.cos(camElev);
      camera.position.set(
        camTarget.x+camRadius*ch*Math.sin(camAzim),
        camTarget.y+camRadius*Math.sin(camElev),
        camTarget.z+camRadius*ch*Math.cos(camAzim)
      );
      camera.lookAt(camTarget);

      fallenLight.intensity=fallen?2.0+Math.sin(frame*0.2)*0.5:0;

      // Skeleton
      const jointPositions=[];
      if(sk&&sk.length>=3){
        sk.slice(0,JOINT_COUNT).forEach((pt,i)=>{
          const v=new THREE.Vector3(pt.x??0,pt.z??0,pt.y??0);
          jointPositions.push(v);
          if(i<jointMeshes.length){
            jointMeshes[i].position.copy(v);
            if(i<16) jointMeshes[i].visible=true;
            if(i===0){
              jointMeshes[i].material.emissiveIntensity=0.3+(ag.phi??0.1)*0.4+Math.sin(frame*0.08)*0.1;
              jointMeshes[i].material.emissive.setHex(fallen?0xff2200:isVis?0x22ccaa:0x6622aa);
              jointMeshes[i].material.color.setHex(isVis?0x44ffcc:0xcc88ff);
            }
          }
        });
        for(let i=jointPositions.length;i<JOINT_COUNT;i++)
          jointPositions.push(jointMeshes[Math.max(0,i-1)]?.position?.clone()??new THREE.Vector3());
      } else {
        const t=frame*0.025;
        const comH=1.2+(fallen?-0.8:0);
        const poses=[
          [0,comH+0.26,0],[0,comH+0.13,0],[0,comH+0.02,0],[0,comH-0.10,0],
          [-0.26,comH+0.11,0],[0.26,comH+0.11,0],
          [-0.42,comH+0.05+Math.sin(t+1)*0.06,0],[0.42,comH+0.05+Math.sin(t+2)*0.06,0],
          [-0.50,comH-0.18,0],[0.50,comH-0.18,0],
          [-0.11,comH-0.22,0],[0.11,comH-0.22,0],
          [-0.11,comH-0.56+Math.sin(t)*0.05,0],[0.11,comH-0.56+Math.sin(t+Math.PI)*0.05,0],
          [-0.11+Math.sin(t)*0.04,comH-0.86,0.05],[0.11+Math.sin(t+Math.PI)*0.04,comH-0.86,0.05],
          [-0.15+Math.sin(t)*0.04,comH-0.90,0.05],[0.15+Math.sin(t+Math.PI)*0.04,comH-0.90,0.05],
        ];
        poses.forEach(([x,y,z],i)=>{
          const v=new THREE.Vector3(x,y,z);
          jointPositions.push(v);
          if(i<jointMeshes.length){
            jointMeshes[i].position.copy(v);
            if(i<16) jointMeshes[i].visible=true;
          }
        });
      }

      SKELETON_BONES.forEach(([a,b],k)=>{
        if(a<jointPositions.length&&b<jointPositions.length&&k<boneMeshes.length){
          updateBone(boneMeshes[k],jointPositions[a],jointPositions[b]);
          boneMeshes[k].visible=true;
          const boneCol=fallen?0x881100:isVis?parseInt(SLOT_COLORS[k%8].replace("#",""),16):wCol;
          boneMeshes[k].material.color.setHex(boneCol);
          boneMeshes[k].material.emissiveIntensity=0.15+Math.sin(frame*0.05+k)*0.05;
        }
      });

      // Кубы
      const cubes=ds.scene?.cubes;
      cubeMeshes.forEach((cm,i)=>{
        const c=cubes?.[i];
        const has=c&&typeof c.x==="number";
        cm.visible=showCubesRef.current&&has;
        if(has){cm.position.set(c.x,Math.max(0,c.z??0)+cm.geometry.parameters.height/2,c.y??0);}
        if(showCubesRef.current){cm.rotation.y+=0.005;}
      });

      // Фаза 12: слот сферы — орбитируют вокруг головы
      const head = jointPositions[0] ?? new THREE.Vector3(0,1.5,0);
      const visionS = ds.vision;
      slotSpheres.forEach((sm, k) => {
        const active = isVis && visionS?.variability?.[k] !== undefined;
        const varVal  = active ? (visionS.variability[k] ?? 0) : 0;
        const slotVal = active ? (visionS.slot_values?.[k] ?? 0.5) : 0.5;

        sm.visible = active && varVal > 0.01;
        if (!active) return;

        const angle = (k / 8) * Math.PI * 2 + frame * 0.015;
        const r = 0.8 + varVal * 0.4;
        sm.position.set(
          head.x + Math.cos(angle) * r,
          head.y + 0.3 + Math.sin(frame * 0.03 + k) * 0.15,
          head.z + Math.sin(angle) * r
        );
        sm.material.opacity = 0.3 + varVal * 0.6;
        sm.material.emissiveIntensity = 0.3 + slotVal * 0.6;
        sm.scale.setScalar(0.8 + slotVal * 0.5);
      });

      // Vision ring
      const showRing = isVis;
      visionRing.position.copy(head);
      visionRing.position.y = head.y - 0.05;
      visionRing.rotation.x = -Math.PI / 2 + Math.sin(frame * 0.02) * 0.05;
      visionRing.rotation.z = frame * 0.01;
      visionRing.material.opacity = showRing ? 0.5 + Math.sin(frame * 0.06) * 0.2 : 0;
      visionRing.material.color.setHex(0x44ffcc);

      if (demon) {
        const comPos=jointPositions[2]??new THREE.Vector3(0,1,0);
        const tmpV=new THREE.Vector3().copy(comPos).sub(demon.position).normalize().multiplyScalar(0.018);
        demon.userData.vel.lerp(tmpV,0.06);
        demon.position.add(demon.userData.vel);
        demon.position.y=Math.max(0.3,demon.position.y);
        demon.rotation.y+=0.05; demon.rotation.x+=0.025;
      }

      renderer.render(scene,camera);
    }
    loop();

    const el=renderer.domElement;
    el.style.cursor="grab"; el.style.touchAction="none";
    const onPD=e=>{if(e.button!==0)return;camDrag=true;camPtrX=e.clientX;camPtrY=e.clientY;el.setPointerCapture(e.pointerId);el.style.cursor="grabbing";};
    const onPM=e=>{if(!camDrag)return;camAzim-=(e.clientX-camPtrX)*0.005;camElev-=(e.clientY-camPtrY)*0.005;camElev=Math.max(0.08,Math.min(Math.PI/2-0.06,camElev));camPtrX=e.clientX;camPtrY=e.clientY;};
    const onPU=e=>{camDrag=false;try{el.releasePointerCapture(e.pointerId);}catch{}el.style.cursor="grab";};
    const onW=e=>{e.preventDefault();camRadius=Math.max(2,Math.min(24,camRadius*(e.deltaY>0?1.1:1/1.1)));};
    el.addEventListener("pointerdown",onPD);
    el.addEventListener("pointermove",onPM);
    el.addEventListener("pointerup",onPU);
    el.addEventListener("pointercancel",onPU);
    el.addEventListener("wheel",onW,{passive:false});
    const onR=()=>{if(!mount)return;camera.aspect=mount.clientWidth/mount.clientHeight;camera.updateProjectionMatrix();renderer.setSize(mount.clientWidth,mount.clientHeight);};
    window.addEventListener("resize",onR);
    return ()=>{
      cancelAnimationFrame(rafRef.current);
      el.removeEventListener("pointerdown",onPD);el.removeEventListener("pointermove",onPM);
      el.removeEventListener("pointerup",onPU);el.removeEventListener("pointercancel",onPU);
      el.removeEventListener("wheel",onW);window.removeEventListener("resize",onR);
      if(mount.contains(renderer.domElement))mount.removeChild(renderer.domElement);
      renderer.dispose();
    };
  }, []);

  const a=ui.agent, wCol=ui.worldColor;
  const blkR=a.valueLayer?.block_rate??0;
  const fallen=ui.fallen||a.fallen;
  const isVis=ui.visualMode;
  const visColor="#44ffcc";

  return(
    <div style={{position:"relative",width:"100%",height:"100vh",background:"#030912",overflow:"hidden",...mono}}>
      <div ref={mountRef} style={{position:"absolute",inset:0}}/>

      {/* Camera overlay */}
      {showCam&&camFrame&&(
        <div style={{position:"absolute",bottom:120,right:14,border:`1px solid ${wCol}55`,borderRadius:3,overflow:"hidden",width:280}}>
          <div style={{display:"flex",gap:4,padding:"3px 6px",background:"rgba(0,0,0,0.7)",fontSize:8}}>
            <span style={{color:wCol,fontSize:7}}>вид из головы (FP)</span>
          </div>
          <img src={`data:image/jpeg;base64,${camFrame}`} style={{width:"100%",display:"block"}} alt="cam"/>
        </div>
      )}

      {/* Status badge */}
      <div style={{position:"absolute",top:14,right:14,display:"flex",gap:6,alignItems:"center"}}>
        {isVis&&<div style={{background:"rgba(0,40,30,0.9)",border:"1px solid #44ffcc",padding:"4px 10px",borderRadius:2,fontSize:9,color:"#44ffcc"}}>
          👁 VISUAL CORTEX · {ui.vision?.n_slots||8} slots
        </div>}
        <div style={{background:connected?"rgba(0,30,10,0.9)":"rgba(20,0,0,0.9)",border:`1px solid ${connected?"#00aa44":"#aa4400"}`,padding:"4px 10px",borderRadius:2,fontSize:9,color:connected?"#00ff88":"#ff8844"}}>
          {connected?"● ONLINE":"○ OFFLINE"} · d={ui.gnnD}
        </div>
      </div>

      {/* Header */}
      <div style={{position:"absolute",top:14,left:"50%",transform:"translateX(-50%)",background:"rgba(2,6,16,0.92)",border:`1px solid ${isVis?visColor:wCol}44`,padding:"7px 22px",textAlign:"center",borderRadius:3,whiteSpace:"nowrap",boxShadow:`0 0 30px ${isVis?visColor:wCol}22`}}>
        <div style={{color:fallen?"#ff4422":isVis?visColor:wCol,fontSize:12,fontWeight:"bold",letterSpacing:"0.18em"}}>
          {fallen?"⚠ FALLEN — ":isVis?"👁 VISION — ":""}AGI NOVA · {isVis?"CAUSAL VISUAL CORTEX":"HUMANOID"} · d={ui.gnnD}
        </div>
        <div style={{color:"#1a3355",fontSize:10,marginTop:2,letterSpacing:"0.06em"}}>
          PHASE {ui.phase}/5 › {PHASE_NAMES[ui.phase]}
          &nbsp;│&nbsp;ENT:<span style={{color:ui.entropy<30?"#00ff99":"#8899bb"}}>{ui.entropy}%</span>
          &nbsp;│&nbsp;T:{ui.tick}
          &nbsp;│&nbsp;<span style={{color:isVis?visColor:wCol}}>🌍 {ui.worldLabel||ui.currentWorld}</span>
          {isVis&&<>&nbsp;│&nbsp;<span style={{color:visColor}}>🔬 {ui.vision?.active_slots??0}/{ui.vision?.n_slots??8} active</span></>}
          &nbsp;│&nbsp;<span style={{color:fallen?"#ff2244":"#335544"}}>{fallen?`💀×${ui.fallCount}`:"✓"}</span>
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
          <button key={w} onClick={()=>switchWorld(w)} style={{padding:"2px 7px",borderRadius:2,fontSize:9,cursor:"pointer",background:ui.currentWorld===w?"#0a0520":"transparent",border:`1px solid ${ui.currentWorld===w?c:"#111"}`,color:ui.currentWorld===w?c:"#334455"}}>{w}</button>
        ))}
        <span style={{color:"#0a1a2e"}}>│</span>
        {[["💉","seeds"],["🌐","rag"],["👁","vision"]].map(([icon,panel])=>(
          <button key={panel} onClick={()=>setPanel(v=>v===panel?null:panel)} style={{padding:"2px 7px",borderRadius:2,fontSize:9,cursor:"pointer",background:activePanel===panel?"#180a30":"transparent",border:`1px solid ${activePanel===panel?(panel==="vision"?visColor:wCol):"#111"}`,color:activePanel===panel?(panel==="vision"?visColor:wCol):"#334455"}}>{icon}</button>
        ))}
        <button onClick={()=>setShowCam(v=>!v)} style={{padding:"2px 7px",borderRadius:2,fontSize:9,cursor:"pointer",background:showCam?"#0a0520":"transparent",border:`1px solid ${showCam?wCol:"#111"}`,color:showCam?wCol:"#334455"}}>📷</button>
      </div>

      {/* Seeds panel */}
      {activePanel==="seeds"&&(
        <div style={{position:"absolute",top:118,left:"50%",transform:"translateX(-50%)",background:"rgba(5,3,12,0.97)",border:`1px solid ${wCol}44`,padding:"12px 16px",borderRadius:3,width:430,zIndex:10}}>
          <div style={{color:wCol,fontSize:10,marginBottom:6}}>💉 SEED INJECTION</div>
          <textarea value={seedText} onChange={e=>setSeedText(e.target.value)} style={{width:"100%",height:80,background:"#040210",border:`1px solid ${wCol}33`,color:"#aa88cc",fontSize:9,padding:6,borderRadius:2,fontFamily:"monospace",resize:"none",boxSizing:"border-box"}}/>
          <div style={{display:"flex",gap:6,marginTop:6,alignItems:"center",flexWrap:"wrap"}}>
            <button onClick={injectSeeds} style={{padding:"3px 10px",borderRadius:2,fontSize:9,cursor:"pointer",background:"#160830",border:`1px solid ${wCol}`,color:wCol}}>INJECT</button>
            <button onClick={bootstrapHumanoid} style={{padding:"3px 10px",borderRadius:2,fontSize:9,cursor:"pointer",background:"#160830",border:"1px solid #cc44ff",color:"#cc44ff"}}>🤖 HUMANOID</button>
            <span style={{color:status.startsWith("✓")?"#00ff99":"#ff4422",fontSize:9}}>{status}</span>
          </div>
        </div>
      )}

      {/* RAG panel */}
      {activePanel==="rag"&&(
        <div style={{position:"absolute",top:118,left:"50%",transform:"translateX(-50%)",background:"rgba(0,4,10,0.97)",border:"1px solid #003322",padding:"12px 16px",borderRadius:3,width:380,zIndex:10}}>
          <div style={{color:"#00aa66",fontSize:10,marginBottom:6}}>🌐 LLM BOOTSTRAP</div>
          <button onClick={ragSeed} disabled={ragLoading} style={{width:"100%",padding:"5px",borderRadius:2,fontSize:9,cursor:"pointer",background:ragLoading?"#001a0a":"#002211",border:"1px solid #006633",color:ragLoading?"#225544":"#00ff88",marginBottom:6}}>
            {ragLoading?"⏳ LLM генерирует…":"🌐 AUTO-SEED (Wikipedia + LLM)"}
          </button>
          <div style={{marginTop:4,fontSize:9,color:status.startsWith("✓")?"#00ff99":"#ff4422"}}>{status}</div>
        </div>
      )}

      {/* ═══════════════════════════════════════════════════════════════════ */}
      {/* ФАЗА 12: VISUAL CORTEX PANEL                                       */}
      {/* ═══════════════════════════════════════════════════════════════════ */}
      {activePanel==="vision"&&(
        <div style={{position:"absolute",top:118,left:"50%",transform:"translateX(-50%)",background:"rgba(0,8,10,0.97)",border:`1px solid ${visColor}44`,padding:"12px 16px",borderRadius:3,width:520,zIndex:10,maxHeight:"70vh",overflowY:"auto"}}>
          <div style={{color:visColor,fontSize:11,fontWeight:"bold",marginBottom:8,letterSpacing:"0.1em"}}>
            👁 CAUSAL VISUAL CORTEX — Фаза 12
          </div>

          {/* Enable/Disable */}
          <div style={{display:"flex",gap:8,marginBottom:10,alignItems:"center"}}>
            {!isVis?(
              <>
                <button onClick={()=>enableVision(8)} disabled={visionLoading}
                  style={{padding:"4px 12px",borderRadius:2,fontSize:9,cursor:"pointer",
                    background:"#002218",border:`1px solid ${visColor}`,color:visColor}}>
                  {visionLoading?"⏳ loading…":"👁 ENABLE (8 slots)"}
                </button>
                <button onClick={()=>enableVision(12)} disabled={visionLoading}
                  style={{padding:"4px 10px",borderRadius:2,fontSize:9,cursor:"pointer",
                    background:"#001810",border:"1px solid #22cc88",color:"#22cc88"}}>
                  12 slots
                </button>
              </>
            ):(
              <button onClick={disableVision}
                style={{padding:"4px 12px",borderRadius:2,fontSize:9,cursor:"pointer",
                  background:"#180010",border:"1px solid #ff4422",color:"#ff4422"}}>
                ✕ DISABLE
              </button>
            )}
            <div style={{fontSize:8,color:"#224433",flexGrow:1}}>
              {isVis?`visual mode · ${ui.visionTicks} ticks · d=${ui.gnnD}`:"ручные переменные"}
            </div>
          </div>

          {isVis&&(
            <div style={{display:"flex",flexWrap:"wrap",gap:8,alignItems:"center",marginBottom:10}}>
              <button
                type="button"
                onClick={runVlmSlotLabels}
                disabled={vlmLoading}
                style={{
                  padding: "4px 10px",
                  borderRadius: 2,
                  fontSize: 9,
                  cursor: vlmLoading ? "wait" : "pointer",
                  background: "#001828",
                  border: `1px solid ${visColor}`,
                  color: visColor,
                }}
              >
                {vlmLoading ? "⏳ VLM…" : "🔬 VLM label slots (Фаза 2)"}
              </button>
              <label style={{ fontSize: 8, color: "#336655", display: "flex", alignItems: "center", gap: 4, cursor: "pointer" }}>
                <input
                  type="checkbox"
                  checked={vlmWeakEdges}
                  onChange={(e) => setVlmWeakEdges(e.target.checked)}
                />
                weak slot→phys edges
              </label>
              <span style={{ fontSize: 7, color: "#224433" }}>
                Ollama /api/chat + images; иначе текстовый fallback
              </span>
            </div>
          )}

          {isVis&&visionData&&(
            <>
              {/* Slot Values & Variability */}
              <div style={{marginBottom:10}}>
                <div style={{fontSize:9,color:"#336655",marginBottom:4}}>
                  SLOT ACTIVATION — {visionData.active_slots??0}/{visionData.n_slots??8} active
                </div>
                <div style={{display:"grid",gridTemplateColumns:"repeat(4,1fr)",gap:4}}>
                  {Array.from({length:visionData.n_slots??8},(_,k)=>{
                    const val  = visionData.slot_values?.[k]??0;
                    const varV = visionData.variability?.[k]??0;
                    const active = varV > 0.02;
                    const col = SLOT_COLORS[k%SLOT_COLORS.length];
                    const sl = visionData.slot_labels?.[k];
                    const lbl = sl?.label;
                    return(
                      <div key={k} onClick={()=>setSelectedSlot(k)}
                        style={{background:selectedSlot===k?"rgba(68,255,204,0.08)":"rgba(0,10,12,0.8)",
                          border:`1px solid ${selectedSlot===k?col:active?"#1a3322":"#0a1a1a"}`,
                          borderRadius:3,padding:"5px 6px",cursor:"pointer"}}>
                        <div style={{display:"flex",justifyContent:"space-between",marginBottom:2}}>
                          <span style={{color:col,fontSize:8,fontWeight:"bold"}}>S{k}</span>
                          <span style={{color:active?"#44ffcc":"#334444",fontSize:7}}>{active?"●":"○"}</span>
                        </div>
                        {lbl ? (
                          <div style={{ fontSize: 6, color: "#66bbaa", marginBottom: 2, lineHeight: 1.2, minHeight: 14 }} title={sl?.likely_phys?.join(", ") || ""}>
                            {lbl}
                            {typeof sl?.confidence === "number" ? ` · ${(sl.confidence * 100).toFixed(0)}%` : ""}
                          </div>
                        ) : (
                          <div style={{ fontSize: 6, color: "#223333", marginBottom: 2, minHeight: 14 }}>—</div>
                        )}
                        {/* Value bar */}
                        <div style={{height:3,background:"#0a1a1a",borderRadius:2,overflow:"hidden",marginBottom:2}}>
                          <div style={{width:`${val*100}%`,height:"100%",background:col,transition:"width 0.5s"}}/>
                        </div>
                        {/* Variability */}
                        <div style={{height:2,background:"#050d0d",borderRadius:2,overflow:"hidden"}}>
                          <div style={{width:`${Math.min(varV*100*5,100)}%`,height:"100%",background:`${col}88`}}/>
                        </div>
                        <div style={{fontSize:6,color:"#225544",marginTop:1,display:"flex",justifyContent:"space-between"}}>
                          <span>{val.toFixed(2)}</span>
                          <span>σ={varV.toFixed(3)}</span>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>

              {/* Attention frame для выбранного слота */}
              <div style={{...sep,paddingTop:8}}>
                <div style={{fontSize:9,color:"#336655",marginBottom:4,display:"flex",justifyContent:"space-between"}}>
                  <span>ATTENTION MASK — <span style={{color:SLOT_COLORS[selectedSlot%8]}}>slot_{selectedSlot}</span>
                    {visionData.slot_labels?.[selectedSlot]?.label ? (
                      <span style={{ color: "#66bbaa", marginLeft: 6 }}>
                        ({visionData.slot_labels[selectedSlot].label})
                      </span>
                    ) : null}
                  </span>
                  <span style={{color:"#224433",fontSize:8}}>что видит этот слот</span>
                </div>
                {attnFrame?(
                  <img src={`data:image/jpeg;base64,${attnFrame}`}
                    style={{width:"100%",borderRadius:3,display:"block",border:`1px solid ${SLOT_COLORS[selectedSlot%8]}44`}}
                    alt={`slot ${selectedSlot} attention`}/>
                ):(
                  <div style={{height:80,background:"#050d10",borderRadius:3,display:"flex",alignItems:"center",justifyContent:"center",color:"#224433",fontSize:9}}>
                    {isVis?"loading…":"enable vision first"}
                  </div>
                )}
                {/* Переключалки слотов */}
                <div style={{display:"flex",gap:3,marginTop:4,flexWrap:"wrap"}}>
                  {Array.from({length:visionData.n_slots??8},(_,k)=>(
                    <button key={k} onClick={()=>setSelectedSlot(k)}
                      style={{padding:"2px 6px",borderRadius:2,fontSize:8,cursor:"pointer",
                        background:selectedSlot===k?"rgba(0,20,15,0.9)":"transparent",
                        border:`1px solid ${selectedSlot===k?SLOT_COLORS[k%8]:"#0a1a1a"}`,
                        color:selectedSlot===k?SLOT_COLORS[k%8]:"#334444"}}>
                      S{k}
                    </button>
                  ))}
                </div>
              </div>

              {/* Cortex stats */}
              {visionData.cortex&&<div style={{...sep,paddingTop:6}}>
                <div style={{fontSize:8,color:"#224433",display:"flex",gap:10}}>
                  <span>encodes: <span style={{color:"#44ffcc"}}>{visionData.cortex.n_encode}</span></span>
                  <span>trains: <span style={{color:"#44ffcc"}}>{visionData.cortex.n_train}</span></span>
                  <span>loss: <span style={{color:visionData.cortex.mean_loss<0.05?"#00ff99":"#ffaa00"}}>{visionData.cortex.mean_loss?.toFixed(5)}</span></span>
                </div>
              </div>}
            </>
          )}

          {isVis&&!visionData&&(
            <div style={{color:"#224433",fontSize:9,marginTop:8,padding:"10px",background:"rgba(0,10,8,0.6)",borderRadius:3}}>
              ⏳ Загрузка данных visual cortex…
            </div>
          )}

          {/* Установка зависимостей */}
          <div style={{...sep,paddingTop:6}}>
            <div style={{fontSize:7,color:"#1a3322"}}>
              Требования Фазы 12: <code style={{color:"#336644",fontSize:7}}>pip install opencv-python scipy Pillow</code>
            </div>
          </div>
        </div>
      )}

      {/* Left HUD */}
      <div style={{position:"absolute",top:118,left:14}}>
        <div style={{background:"rgba(2,5,14,0.92)",border:`1px solid ${fallen?"#660011":isVis?visColor+"33":wCol+"33"}`,borderLeft:`3px solid ${fallen?"#ff2244":isVis?visColor:wCol}`,padding:"10px 14px",minWidth:235,borderRadius:3,transition:"border-color 0.5s"}}>
          <div style={{color:fallen?"#ff4422":isVis?visColor:wCol,fontSize:11,fontWeight:"bold",marginBottom:5,letterSpacing:"0.08em"}}>
            {fallen?"💀":isVis?"👁":"◈"} NOVA — {isVis?"Visual AGI":"Singleton AGI"}
            <span style={{color:"#1a2244",marginLeft:6,fontSize:8}}>GNN·{ui.currentWorld}</span>
          </div>

          {fallen&&<div style={{background:"rgba(255,0,34,0.08)",border:"1px solid #ff224433",borderRadius:2,padding:"3px 7px",marginBottom:4,fontSize:8,color:"#ff4422"}}>
            ⚠ FALLEN × {ui.fallCount}
          </div>}

          {isVis&&<div style={{background:"rgba(0,255,200,0.05)",border:"1px solid #44ffcc33",borderRadius:2,padding:"3px 7px",marginBottom:4,fontSize:8,color:"#44ffcc"}}>
            👁 visual cortex · {ui.vision?.active_slots??0} slots active · {ui.visionTicks} ticks
          </div>}

          <table style={{borderCollapse:"collapse",width:"100%",fontSize:9}}><tbody>
            {[
              ["Φ autonomy",  <span style={{color:phiC(a.phi??0)}}>{((a.phi??0)*100).toFixed(1)}%</span>],
              ["CG d|G|/dt",  <span style={{color:cgC(a.compressionGain??0)}}>{(a.compressionGain??0)>=0?"+":""}{(a.compressionGain??0).toFixed(4)}</span>],
              ["α trust",    `${Math.round((a.alphaMean??0)*100)}%`],
              ["h(W) DAG",   <span style={{color:hWC(a.hW??0)}}>{(a.hW??0).toFixed(4)}</span>],
              ["GNN d",      <span style={{color:isVis?visColor:wCol}}>{ui.gnnD}</span>],
              ["Edges",      a.edgeCount??0],
              ["do()/blk",   `${a.totalInterventions??0}/${a.totalBlocked??0}`],
              ["Discovery",  <span style={{color:(a.discoveryRate??0)>0.2?"#00ff99":"#aabbcc"}}>{((a.discoveryRate??0)*100).toFixed(0)}%</span>],
            ].map(([l,v],k)=>(
              <tr key={k}><td style={{color:"#2a4466",paddingRight:8,paddingBottom:2}}>{l}</td><td style={{color:"#aabbcc",textAlign:"right"}}>{v}</td></tr>
            ))}
          </tbody></table>

          {a.valueLayer&&<div style={{...sep,fontSize:8}}>
            <div style={{display:"flex",justifyContent:"space-between"}}>
              <span style={{color:"#441133"}}>VALUE LAYER {a.valueLayer.vl_phase}</span>
              <span style={{color:blkC(blkR)}}>{(blkR*100).toFixed(1)}%</span>
            </div>
            {(a.valueLayer.imagination_horizon??0)>0&&<div style={{color:"#553366",marginTop:3,fontSize:7}}>
              imagination N={a.valueLayer.imagination_horizon} · checks {a.valueLayer.imagination_checks??0} · blk {a.valueLayer.imagination_blocks??0}
            </div>}
          </div>}

          {[
            {w:`${Math.round((a.alphaMean??0)*100)}%`,f:"#110022",t:wCol},
            {w:`${Math.round((a.phi??0)*100)}%`,f:"#220011",t:phiC(a.phi??0)},
            {w:`${Math.max(0,100-Math.min((a.hW??0)*20,100))}%`,f:"#001811",t:hWC(a.hW??0)},
          ].map((b,k)=>(
            <div key={k} style={{marginTop:k===0?5:2,height:k===0?3:2,background:"#050a18",borderRadius:2,overflow:"hidden"}}>
              <div style={{width:b.w,height:"100%",background:`linear-gradient(90deg,${b.f},${b.t})`,transition:"width 0.8s"}}/>
            </div>
          ))}
        </div>
      </div>

      {/* Right panel */}
      <div style={{position:"absolute",top:118,right:14,background:"rgba(2,5,14,0.92)",border:"1px solid #0a1a2e",padding:"10px 14px",borderRadius:3,fontSize:9,maxWidth:200}}>
        <div style={{color:"#1a3355",marginBottom:6,fontSize:9}}>AGI NOVA — PHASE 12</div>

        <label style={{display:"flex",alignItems:"center",gap:6,cursor:"pointer",color:"#556688",fontSize:8,marginBottom:4}}>
          <input type="checkbox" checked={showCubes} onChange={e=>setShowCubes(e.target.checked)} style={{accentColor:wCol}}/>
          показать кубы
        </label>

        <div style={{...sep}}>
          {Object.entries(WORLD_COLORS).map(([w,c])=>(
            <div key={w} onClick={()=>switchWorld(w)} style={{fontSize:8,marginBottom:2,cursor:"pointer",color:ui.currentWorld===w?c:"#223344"}}>
              {ui.currentWorld===w?"▶ ":"  "}{w}
            </div>
          ))}
        </div>

        <div style={{...sep,lineHeight:1.9}}>
          <div style={{color:"#1a2244",fontSize:8}}>ROADMAP:</div>
          <div style={{color:"#224433",fontSize:8}}>✓ 11. Humanoid Singleton</div>
          <div style={{color:isVis?visColor:"#00cc88",fontSize:8,fontWeight:isVis?"bold":"normal"}}>
            {isVis?"● ":"→ "}12. Causal Vision{isVis?` · ${ui.visionTicks}t`:""}
          </div>
          <div style={{color:(a.valueLayer?.imagination_horizon??0)>0?"#00cc88":"#224433",fontSize:8,fontWeight:(a.valueLayer?.imagination_horizon??0)>0?"bold":"normal"}}>
            {(a.valueLayer?.imagination_horizon??0)>0?"● ":"→ "}13. N-step Imagination{(a.valueLayer?.imagination_horizon??0)>0?` · N=${a.valueLayer.imagination_horizon}`:""}
          </div>
          <div style={{color:"#224433",fontSize:8}}>→ 14. Cross-Domain</div>
          <div style={{color:"#224433",fontSize:8}}>→ 15. Reality Bridge</div>
        </div>

        {isVis&&ui.vision&&<div style={{...sep}}>
          <div style={{fontSize:8,color:visColor,marginBottom:3}}>👁 VISION STATS</div>
          {(ui.vision.variability||[]).map((v,k)=>(
            <div key={k} style={{display:"flex",gap:4,alignItems:"center",marginBottom:2}}>
              <span style={{color:SLOT_COLORS[k%8],fontSize:7,width:16}}>S{k}</span>
              <div style={{flexGrow:1,height:3,background:"#0a1a1a",borderRadius:1,overflow:"hidden"}}>
                <div style={{width:`${Math.min(v*100*5,100)}%`,height:"100%",background:SLOT_COLORS[k%8]}}/>
              </div>
              <span style={{color:"#224433",fontSize:6}}>{v.toFixed(3)}</span>
            </div>
          ))}
        </div>}
      </div>

      {/* Event log */}
      <div style={{position:"absolute",bottom:14,left:14,right:14,background:"rgba(2,5,14,0.92)",border:"1px solid #0a1a2e",padding:"8px 14px",borderRadius:3,maxHeight:100,overflow:"hidden"}}>
        <div style={{color:"#0a1a2e",fontSize:9,letterSpacing:"0.1em",marginBottom:4}}>
          CAUSAL EVENT STREAM {connected?"● ONLINE":"○ OFFLINE"} {isVis?"· 👁 VISUAL MODE":""}
        </div>
        {ui.events.map((ev,i)=>(
          <div key={i} style={{color:ev.color??"#334455",fontSize:9,marginBottom:2,opacity:Math.max(0.15,1-i*0.12)}}>
            [{String(ev.tick??0).padStart(5,"0")}] › {ev.text}
          </div>
        ))}
      </div>
    </div>
  );
}