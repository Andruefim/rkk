import { useState, useEffect, useRef } from "react";
import * as THREE from "three";
import { useRKKStream } from "../../hooks/useRKKStream";

const API = "http://localhost:8000";
const CAMERA_PREVIEW_MS = 500;
const CAM_VIEW = "fp";

const WORLD_COLORS = {
  humanoid:"#cc44ff", robot:"#aa22dd", pybullet:"#ff44aa",
  physics:"#00ff99", chemistry:"#0099ff", logic:"#ff9900",
};

const SLOT_COLORS = [
  "#ff5050","#50c8ff","#50ff64","#ffc850",
  "#c850ff","#ff8c50","#50ffdc","#b4b4ff",
];

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
    visualMode:raw.visual_mode??false,
    visionTicks:raw.vision_ticks??0,
    vision:raw.vision??null,
    fixedRoot:raw.fixed_root??false,  // НОВОЕ
  };
}

const PHASE_NAMES = ["","Causal Crib","Robotic Explorer","Social Sandbox","Value Lock","Open Reality"];
const sep  = {borderTop:"1px solid #0a1a2e",marginTop:4,paddingTop:4};
const mono = {fontFamily:"'Courier New',monospace"};

const SKELETON_BONES = [
  [0,1],[1,2],[2,3],
  [1,4],[4,6],[6,8],[1,5],[5,7],[7,9],
  [3,10],[10,12],[12,14],[3,11],[11,13],[13,15],
];

export default function RKKHumanoid() {
  const mountRef   = useRef(null);
  const rafRef     = useRef(null);
  const { frame: wsFrame, connected, setSpeed: wsSetSpeed } = useRKKStream();
  const wsFrameRef = useRef(wsFrame);
  wsFrameRef.current = wsFrame;

  const [speed,        setSpeedLocal]    = useState(1);
  const [ui,           setUI]            = useState(() => normFrame(wsFrame));
  const [activePanel,  setPanel]         = useState(null);
  const [seedText,     setSeedText]      = useState('[\n  {"from_": "lshoulder", "to": "cube0_x", "weight": 0.6}\n]');
  const [status,       setStatus]        = useState("");
  const [camFrame,     setCamFrame]      = useState(null);
  const [showCam,      setShowCam]       = useState(false);
  const [showCubes,    setShowCubes]     = useState(true);
  const [ragLoading,   setRagLoading]    = useState(false);

  // Vision
  const [visionData,    setVisionData]    = useState(null);
  const [visionLoading, setVisionLoading] = useState(false);
  const [vlmLoading,    setVlmLoading]    = useState(false);
  const [vlmWeakEdges,  setVlmWeakEdges]  = useState(false);
  const [selectedSlot,  setSelectedSlot]  = useState(0);
  const [attnFrame,     setAttnFrame]     = useState(null);
  const [visionEnabled, setVisionEnabled] = useState(false);

  // Fixed root — НОВОЕ
  const [fixedRoot,        setFixedRoot]        = useState(false);
  const [fixedRootLoading, setFixedRootLoading] = useState(false);
  const fixedRootRef = useRef(false);

  const showCubesRef = useRef(showCubes);
  showCubesRef.current = showCubes;

  const setSpeed = s => { setSpeedLocal(s); if (connected) wsSetSpeed(s); };

  useEffect(() => {
    const f = normFrame(wsFrame);
    setUI(f);
    setVisionEnabled(f.visualMode);
    // Синхронизируем fixed_root из WS stream
    setFixedRoot(f.fixedRoot);
    fixedRootRef.current = f.fixedRoot;
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

  // Vision polling
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

  useEffect(() => {
    if (!visionEnabled || activePanel !== "vision") return;
    (async () => {
      try {
        const d = await fetch(`${API}/vision/attn_frame?slot_idx=${selectedSlot}`).then(r=>r.json());
        if (d.available) setAttnFrame(d.frame);
      } catch {}
    })();
  }, [selectedSlot, visionEnabled, activePanel]);

  // ── Actions ────────────────────────────────────────────────────────────────
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

  const enableVision = async (nSlots=8) => {
    setVisionLoading(true);
    try {
      const d = await fetch(`${API}/vision/enable`,{method:"POST",
        headers:{"Content-Type":"application/json"},
        body:JSON.stringify({n_slots:nSlots,mode:"hybrid"})}).then(r=>r.json());
      if (d.visual) { setVisionEnabled(true); setStatus(`👁 Vision ON: ${d.n_slots} slots, d=${d.gnn_d}`); setPanel("vision"); }
      else setStatus(`✗ ${d.error||"failed"}`);
    } catch(e) { setStatus(`✗ ${e.message}`); }
    finally { setVisionLoading(false); }
  };

  const runVlmSlotLabels = async () => {
    setVlmLoading(true);
    try {
      const d = await fetch(`${API}/vision/vlm-label`,{method:"POST",
        headers:{"Content-Type":"application/json"},
        body:JSON.stringify({max_mask_images:4,text_only:false,
          inject_weak_edges:vlmWeakEdges})}).then(r=>r.json());
      if (d.ok) {
        setStatus(`🔬 VLM ${d.mode}: ${d.n_slots_labeled} labels${d.warning?` (${d.warning})`:""}`);
        try { const r2=await fetch(`${API}/vision/slots`).then(r=>r.json()); if(r2.visual_mode!==false) setVisionData(r2); } catch{}
      } else setStatus(`✗ VLM: ${d.error||"failed"}`);
    } catch(e) { setStatus(`✗ ${e.message}`); }
    finally { setVlmLoading(false); }
  };

  const disableVision = async () => {
    try {
      await fetch(`${API}/vision/disable`,{method:"POST"});
      setVisionEnabled(false); setVisionData(null); setAttnFrame(null);
      setStatus("👁 Vision OFF");
    } catch(e) { setStatus(`✗ ${e.message}`); }
  };

  // ── Fixed root toggle ──────────────────────────────────────────────────────
  const toggleFixedRoot = async () => {
    setFixedRootLoading(true);
    try {
      const endpoint = fixedRoot
        ? `${API}/fixed-root/disable`
        : `${API}/fixed-root/enable`;
      const d = await fetch(endpoint,{method:"POST"}).then(r=>r.json());
      if (d.error) {
        setStatus(`✗ Fixed root: ${d.error}`);
      } else {
        const on = d.fixed_root ?? !fixedRoot;
        setFixedRoot(on);
        fixedRootRef.current = on;
        setStatus(`📌 Fixed root ${on?"ON":"OFF"}: d=${d.gnn_d}, ${d.new_vars?.length??0} vars, +${d.seeds_injected??0} seeds`);
      }
    } catch(e) { setStatus(`✗ ${e.message}`); }
    finally { setFixedRootLoading(false); }
  };

  // ── Three.js ───────────────────────────────────────────────────────────────
  useEffect(() => {
    const mount = mountRef.current;
    if (!mount) return;

    const renderer = new THREE.WebGLRenderer({antialias:true});
    renderer.setPixelRatio(Math.min(window.devicePixelRatio,2));
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type    = THREE.PCFSoftShadowMap;
    renderer.setSize(mount.clientWidth,mount.clientHeight);
    mount.appendChild(renderer.domElement);

    const scene  = new THREE.Scene();
    scene.background = new THREE.Color(0x030912);
    scene.fog = new THREE.FogExp2(0x030912,0.022);

    const camera = new THREE.PerspectiveCamera(55,mount.clientWidth/mount.clientHeight,0.1,100);
    camera.position.set(0,0.8,4.0);

    scene.add(new THREE.AmbientLight(0x0a1020,3.0));
    const key = new THREE.DirectionalLight(0x8899ff,2.5);
    key.position.set(3,6,4); key.castShadow=true; scene.add(key);
    const rim = new THREE.PointLight(0x6644ff,1.5,10);
    rim.position.set(0,4,-2); scene.add(rim);

    const floor = new THREE.Mesh(
      new THREE.PlaneGeometry(20,20),
      new THREE.MeshStandardMaterial({color:0x0a0f1a,roughness:0.9})
    );
    floor.rotation.x=-Math.PI/2; floor.receiveShadow=true; scene.add(floor);
    scene.add(new THREE.GridHelper(20,40,0x0d1a2e,0x071220));

    const vS = 0.62;

    const ramp = new THREE.Mesh(
      new THREE.BoxGeometry(3.75,0.12,2.5),
      new THREE.MeshStandardMaterial({color:0x1a1510,roughness:0.8})
    );
    ramp.position.set(3.5,0.48,0); ramp.rotation.x=-0.26; scene.add(ramp);

    const shelf = new THREE.Mesh(
      new THREE.BoxGeometry(0.6,0.06,0.6),
      new THREE.MeshStandardMaterial({color:0x1a2030,roughness:0.6})
    );
    shelf.position.set(1.2,0.22,0.0); scene.add(shelf);

    // Гуманоид — Figure 03 aesthetic: smooth matte-white body with dark accents
    const JOINT_COUNT = 18;
    const bodyMat = new THREE.MeshStandardMaterial({color:0xe8e0d8,roughness:0.65,metalness:0.05});
    const accentMat = new THREE.MeshStandardMaterial({color:0x2a2a2e,roughness:0.5,metalness:0.2});
    const jointMat = new THREE.MeshStandardMaterial({color:0x1a1a1e,roughness:0.4,metalness:0.3});
    const headMat = new THREE.MeshStandardMaterial({color:0xd0c8c0,roughness:0.55,metalness:0.1,
      emissive:0x111115,emissiveIntensity:0.15});

    const JOINT_RADII = [
      0.065, 0.055, 0.05,  // 0=head, 1=neck, 2=chest
      0.06,                // 3=upper torso
      0.04, 0.04,          // 4,5=shoulders
      0.035, 0.035,        // 6,7=elbows
      0.028, 0.028,        // 8,9=wrists/hands
      0.045, 0.045,        // 10,11=hips
      0.038, 0.038,        // 12,13=knees
      0.042, 0.042,        // 14,15=ankles/feet
    ];
    const jointMeshes = Array.from({length:JOINT_COUNT},(_,i)=>{
      if (i>=16){const o=new THREE.Object3D();o.visible=false;scene.add(o);return o;}
      const r = (JOINT_RADII[i] || 0.04) * vS;
      const isHead = i===0;
      const isHand = i===8||i===9;
      const isFoot = i===14||i===15;
      let geo, mat;
      if (isHead) {
        geo = new THREE.SphereGeometry(r, 16, 16);
        mat = headMat;
      } else if (isFoot) {
        geo = new THREE.BoxGeometry(r*2.2, r*0.7, r*1.6);
        mat = accentMat;
      } else if (isHand) {
        geo = new THREE.SphereGeometry(r, 10, 10);
        mat = accentMat;
      } else {
        geo = new THREE.SphereGeometry(r, 10, 10);
        mat = jointMat;
      }
      const m = new THREE.Mesh(geo, mat);
      m.castShadow = true;
      scene.add(m);
      return m;
    });

    const BONE_RADII = [
      0.032, 0.028, 0.025,       // spine segments
      0.030, 0.025, 0.022,       // L arm: shoulder→elbow, elbow→wrist
      0.030, 0.025, 0.022,       // R arm
      0.035, 0.030, 0.028,       // L leg: hip→knee, knee→ankle
      0.035, 0.030, 0.028,       // R leg
    ];
    const boneMeshes = SKELETON_BONES.map((_,bi)=>{
      const r = (BONE_RADII[bi] || 0.025) * vS;
      const isLeg = bi >= 9;
      const m = new THREE.Mesh(
        new THREE.CylinderGeometry(r, r * 0.85, 1, 8),
        isLeg ? bodyMat.clone() : bodyMat
      );
      m.castShadow = true;
      scene.add(m);
      return m;
    });

    // Кубы
    // cube0: box оранжевый, cube1: СФЕРА синяя, cube2: box зелёный
    const cubeMeshes = [];

    // cube0 — box
    const cm0=new THREE.Mesh(
      new THREE.BoxGeometry(0.25,0.25,0.25),
      new THREE.MeshStandardMaterial({color:0xff6622,emissive:0xff4400,emissiveIntensity:0.15,roughness:0.4})
    );
    cm0.castShadow=true; cm0.receiveShadow=true;
    cm0.position.set(0.56,0.125,0.12); scene.add(cm0); cubeMeshes.push(cm0);

    // cube1 — sphere
    const cm1=new THREE.Mesh(
      new THREE.SphereGeometry(0.11,12,12),
      new THREE.MeshStandardMaterial({color:0x22aaff,emissive:0x0066ff,emissiveIntensity:0.2,roughness:0.2,metalness:0.1})
    );
    cm1.castShadow=true; cm1.receiveShadow=true;
    cm1.position.set(-0.44,0.19,0.50); scene.add(cm1); cubeMeshes.push(cm1);

    // cube2 — heavy box
    const cm2=new THREE.Mesh(
      new THREE.BoxGeometry(0.35,0.35,0.35),
      new THREE.MeshStandardMaterial({color:0x44ff88,emissive:0x22aa44,emissiveIntensity:0.15,roughness:0.5})
    );
    cm2.castShadow=true; cm2.receiveShadow=true;
    cm2.position.set(0.25,0.25,-0.75); scene.add(cm2); cubeMeshes.push(cm2);

    // Рычаг
    const leverBase = new THREE.Mesh(
      new THREE.BoxGeometry(0.08,0.08,0.08),
      new THREE.MeshStandardMaterial({color:0x444455,roughness:0.7})
    );
    leverBase.position.set(0.50,0.04,0.45); scene.add(leverBase);

    const leverArm = new THREE.Mesh(
      new THREE.BoxGeometry(0.56,0.06,0.04),
      new THREE.MeshStandardMaterial({color:0xddaa22,emissive:0x886600,emissiveIntensity:0.3,roughness:0.4})
    );
    leverArm.position.set(0.50,0.08,0.45); scene.add(leverArm);

    // Fixed root indicator ring (вокруг таза, жёлтый)
    const fixedRootRing = new THREE.Mesh(
      new THREE.TorusGeometry(0.35*vS,0.015*vS,8,32),
      new THREE.MeshBasicMaterial({color:0xffcc00,transparent:true,opacity:0.})
    );
    scene.add(fixedRootRing);

    // Vision ring
    const visionRing = new THREE.Mesh(
      new THREE.TorusGeometry(0.6*vS,0.02*vS,8,32),
      new THREE.MeshBasicMaterial({color:0x44ffcc,transparent:true,opacity:0.})
    );
    scene.add(visionRing);

    // Slot spheres
    const slotSpheres = Array.from({length:8},(_,k)=>{
      const col=parseInt(SLOT_COLORS[k].replace("#",""),16);
      const m=new THREE.Mesh(
        new THREE.SphereGeometry(0.05*vS,8,8),
        new THREE.MeshStandardMaterial({color:col,emissive:col,emissiveIntensity:0.5,transparent:true,opacity:0.})
      );
      scene.add(m); return m;
    });

    // Fallen light
    const fallenLight = new THREE.PointLight(0xff0022,0,3);
    fallenLight.position.set(0,0.1,0); scene.add(fallenLight);

    // Particles
    const pPos=new Float32Array(600*3);
    for(let i=0;i<600;i++){pPos[i*3]=(Math.random()-.5)*30;pPos[i*3+1]=Math.random()*12;pPos[i*3+2]=(Math.random()-.5)*30;}
    const pGeom=new THREE.BufferGeometry();
    pGeom.setAttribute("position",new THREE.BufferAttribute(pPos,3));
    scene.add(new THREE.Points(pGeom,new THREE.PointsMaterial({color:0x220044,size:0.06,transparent:true,opacity:0.4})));

    let frame=0;
    const camTarget=new THREE.Vector3(0,0.5,0);
    let camAzim=0,camElev=0.25,camRadius=3.5;
    let camDrag=false,camPtrX=0,camPtrY=0;

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
      const isFR=fixedRootRef.current;

      // Камера
      let comX=0,comZ=0;
      const sk=ds.scene?.skeleton;
      if(sk&&sk.length>=3){
        let sx=0,sz=0,n=0;
        for(let j=0;j<Math.min(sk.length,JOINT_COUNT);j++){
          const pt=sk[j]; if(!pt)continue; sx+=pt.x??0;sz+=pt.y??0;n++;
        }
        if(n>0){comX=sx/n;comZ=sz/n;}
      }
      camTarget.lerp(new THREE.Vector3(comX,0.7,comZ),0.05);
      const ch=Math.cos(camElev);
      camera.position.set(
        camTarget.x+camRadius*ch*Math.sin(camAzim),
        camTarget.y+camRadius*Math.sin(camElev),
        camTarget.z+camRadius*ch*Math.cos(camAzim)
      );
      camera.lookAt(camTarget);

      fallenLight.intensity=fallen?2.+Math.sin(frame*.2)*.5:0.;

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
              jointMeshes[i].material.emissiveIntensity=0.1+(ag.phi??0.1)*0.2+Math.sin(frame*.08)*.05;
              jointMeshes[i].material.emissive.setHex(fallen?0x660800:isFR?0x443300:isVis?0x114433:0x111115);
              jointMeshes[i].material.color.setHex(fallen?0xff4422:isFR?0xe8d0a0:isVis?0xb0e8d8:0xd0c8c0);
            }
          }
        });
        for(let i=jointPositions.length;i<JOINT_COUNT;i++)
          jointPositions.push(jointMeshes[Math.max(0,i-1)]?.position?.clone()??new THREE.Vector3());
      } else {
        const t=frame*.025;
        const comH=0.7+(fallen?-.5:0);
        const poses=[
          [0,comH+.26,0],[0,comH+.13,0],[0,comH+.02,0],[0,comH-.10,0],
          [-.26,comH+.11,0],[.26,comH+.11,0],
          [-.42,comH+.05+Math.sin(t+1)*.06,0],[.42,comH+.05+Math.sin(t+2)*.06,0],
          [-.50,comH-.18,0],[.50,comH-.18,0],
          [-.11,comH-.22,0],[.11,comH-.22,0],
          [-.11,comH-.56+Math.sin(t)*.05,0],[.11,comH-.56+Math.sin(t+Math.PI)*.05,0],
          [-.11+Math.sin(t)*.04,comH-.86,.05],[.11+Math.sin(t+Math.PI)*.04,comH-.86,.05],
          [-.15+Math.sin(t)*.04,comH-.90,.05],[.15+Math.sin(t+Math.PI)*.04,comH-.90,.05],
        ];
        poses.forEach(([x,y,z],i)=>{
          const v=new THREE.Vector3(x,y,z);
          jointPositions.push(v);
          if(i<jointMeshes.length){jointMeshes[i].position.copy(v);if(i<16)jointMeshes[i].visible=true;}
        });
      }

      SKELETON_BONES.forEach(([a,b],k)=>{
        if(a<jointPositions.length&&b<jointPositions.length&&k<boneMeshes.length){
          updateBone(boneMeshes[k],jointPositions[a],jointPositions[b]);
          boneMeshes[k].visible=true;
          const boneCol=fallen?0x993322:isFR?0xd0c0a0:isVis?parseInt(SLOT_COLORS[k%8].replace("#",""),16):0xddd5cd;
          boneMeshes[k].material.color.setHex(boneCol);
          boneMeshes[k].material.emissiveIntensity=fallen?0.1:0.02;
        }
      });

      // Кубы из scene data
      const cubes=ds.scene?.cubes;
      cubeMeshes.forEach((cm,i)=>{
        const c=cubes?.[i];
        const has=c&&typeof c.x==="number";
        cm.visible=showCubesRef.current&&has;
        if(has){
          cm.position.set(c.x,Math.max(0,c.z??0)+cm.geometry.parameters?.height/2||0.12,c.y??0);
        }
        if(showCubesRef.current){
          // Шар (cube1) вращается быстрее
          cm.rotation.y+=i===1?0.03:0.005;
        }
      });

      // Рычаг
      const lever=ds.scene?.lever;
      if(lever){
        leverBase.position.set(lever.x,lever.z,lever.y);
        leverArm.position.set(lever.x,lever.z+0.07,lever.y);
      }
      // Наклон рычага из nodes (lever_angle)
      const la=ag.edges?.find?.(e=>e.from_==="lever_angle")?.weight??0;
      leverArm.rotation.z=la*0.5;
      // Подсветка рычага если активен
      const leverGlow=Math.abs(la)>0.1;
      leverArm.material.emissiveIntensity=leverGlow?0.7+Math.sin(frame*.1)*.3:0.3;
      leverArm.material.emissive.setHex(leverGlow?0xff8800:0x886600);

      // Fixed root ring
      const pelvisPos=jointPositions[3]??new THREE.Vector3(0,0.9,0);
      fixedRootRing.position.copy(pelvisPos);
      fixedRootRing.rotation.x=Math.PI/2;
      fixedRootRing.material.opacity=isFR?0.6+Math.sin(frame*.08)*.2:0.;
      fixedRootRing.material.color.setHex(0xffcc00);

      // Vision ring
      const head=jointPositions[0]??new THREE.Vector3(0,1.5,0);
      const visionS=ds.vision;
      slotSpheres.forEach((sm,k)=>{
        const active=isVis&&visionS?.variability?.[k]!==undefined;
        const varVal=active?(visionS.variability[k]??0):0;
        const slotVal=active?(visionS.slot_values?.[k]??0.5):0.5;
        sm.visible=active&&varVal>0.01;
        if(!active)return;
        const angle=(k/8)*Math.PI*2+frame*.015;
        const r=0.8+varVal*.4;
        sm.position.set(head.x+Math.cos(angle)*r,head.y+.3+Math.sin(frame*.03+k)*.15,head.z+Math.sin(angle)*r);
        sm.material.opacity=0.3+varVal*.6;
        sm.material.emissiveIntensity=0.3+slotVal*.6;
        sm.scale.setScalar(0.8+slotVal*.5);
      });
      visionRing.position.copy(head);
      visionRing.position.y=head.y-.05;
      visionRing.rotation.x=-Math.PI/2+Math.sin(frame*.02)*.05;
      visionRing.rotation.z=frame*.01;
      visionRing.material.opacity=isVis?0.5+Math.sin(frame*.06)*.2:0.;

      renderer.render(scene,camera);
    }
    loop();

    const el=renderer.domElement;
    el.style.cursor="grab"; el.style.touchAction="none";
    const onPD=e=>{if(e.button!==0)return;camDrag=true;camPtrX=e.clientX;camPtrY=e.clientY;el.setPointerCapture(e.pointerId);el.style.cursor="grabbing";};
    const onPM=e=>{if(!camDrag)return;camAzim-=(e.clientX-camPtrX)*.005;camElev-=(e.clientY-camPtrY)*.005;camElev=Math.max(.08,Math.min(Math.PI/2-.06,camElev));camPtrX=e.clientX;camPtrY=e.clientY;};
    const onPU=e=>{camDrag=false;try{el.releasePointerCapture(e.pointerId);}catch{}el.style.cursor="grab";};
    const onW=e=>{e.preventDefault();camRadius=Math.max(2,Math.min(24,camRadius*(e.deltaY>0?1.1:1/1.1)));};
    el.addEventListener("pointerdown",onPD);el.addEventListener("pointermove",onPM);
    el.addEventListener("pointerup",onPU);el.addEventListener("pointercancel",onPU);
    el.addEventListener("wheel",onW,{passive:false});
    const onR=()=>{if(!mount)return;camera.aspect=mount.clientWidth/mount.clientHeight;camera.updateProjectionMatrix();renderer.setSize(mount.clientWidth,mount.clientHeight);};
    window.addEventListener("resize",onR);
    return()=>{
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
  const isFR=fixedRoot;
  const visColor="#44ffcc";
  const frColor="#ffcc44";

  return(
    <div style={{position:"relative",width:"100%",height:"100vh",background:"#030912",overflow:"hidden",...mono}}>
      <div ref={mountRef} style={{position:"absolute",inset:0}}/>

      {/* Camera overlay */}
      {showCam&&camFrame&&(
        <div style={{position:"absolute",bottom:120,right:14,border:`1px solid ${wCol}55`,borderRadius:3,overflow:"hidden",width:280}}>
          <div style={{display:"flex",gap:4,padding:"3px 6px",background:"rgba(0,0,0,0.7)",fontSize:8}}>
            <span style={{color:wCol,fontSize:7}}>FP view</span>
            {isFR&&<span style={{color:frColor,fontSize:7}}>📌 fixed</span>}
          </div>
          <img src={`data:image/jpeg;base64,${camFrame}`} style={{width:"100%",display:"block"}} alt="cam"/>
        </div>
      )}

      {/* Status badges */}
      <div style={{position:"absolute",top:14,right:14,display:"flex",gap:6,alignItems:"center"}}>
        {isFR&&<div style={{background:"rgba(40,30,0,0.9)",border:`1px solid ${frColor}`,padding:"4px 10px",borderRadius:2,fontSize:9,color:frColor}}>
          📌 FIXED BASE · d={ui.gnnD}
        </div>}
        {isVis&&<div style={{background:"rgba(0,40,30,0.9)",border:"1px solid #44ffcc",padding:"4px 10px",borderRadius:2,fontSize:9,color:"#44ffcc"}}>
          👁 VISUAL · {ui.vision?.active_slots||0} active
        </div>}
        <div style={{background:connected?"rgba(0,30,10,0.9)":"rgba(20,0,0,0.9)",border:`1px solid ${connected?"#00aa44":"#aa4400"}`,padding:"4px 10px",borderRadius:2,fontSize:9,color:connected?"#00ff88":"#ff8844"}}>
          {connected?"● ONLINE":"○ OFFLINE"} · d={ui.gnnD}
        </div>
      </div>

      {/* Header */}
      <div style={{position:"absolute",top:14,left:"50%",transform:"translateX(-50%)",background:"rgba(2,6,16,0.92)",border:`1px solid ${isFR?frColor:isVis?visColor:wCol}44`,padding:"7px 22px",textAlign:"center",borderRadius:3,whiteSpace:"nowrap",boxShadow:`0 0 30px ${isFR?frColor:isVis?visColor:wCol}22`}}>
        <div style={{color:fallen?"#ff4422":isFR?frColor:isVis?visColor:wCol,fontSize:12,fontWeight:"bold",letterSpacing:"0.18em"}}>
          {fallen?"⚠ FALLEN — ":isFR?"📌 FIXED BASE — ":isVis?"👁 VISION — ":""}AGI NOVA · d={ui.gnnD}
        </div>
        <div style={{color:"#1a3355",fontSize:10,marginTop:2,letterSpacing:"0.06em"}}>
          PHASE {ui.phase}/5 › {PHASE_NAMES[ui.phase]}
          &nbsp;│&nbsp;ENT:<span style={{color:ui.entropy<30?"#00ff99":"#8899bb"}}>{ui.entropy}%</span>
          &nbsp;│&nbsp;T:{ui.tick}
          &nbsp;│&nbsp;<span style={{color:isFR?frColor:isVis?visColor:wCol}}>🌍 {ui.worldLabel||ui.currentWorld}</span>
          {isFR&&<>&nbsp;│&nbsp;<span style={{color:frColor}}>arms+spine+cubes</span></>}
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
        {/* Fixed root toggle */}
        <button onClick={toggleFixedRoot} disabled={fixedRootLoading}
          style={{padding:"2px 8px",borderRadius:2,fontSize:9,cursor:fixedRootLoading?"wait":"pointer",
            background:isFR?"rgba(255,204,0,0.12)":"transparent",
            border:`1px solid ${isFR?frColor:"#444"}`,
            color:isFR?frColor:"#556677",fontWeight:isFR?"bold":"normal"}}>
          {fixedRootLoading?"⏳":isFR?"📌 FIXED":"📌"}
        </button>
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
            {ragLoading?"⏳ LLM…":"🌐 AUTO-SEED (Wikipedia + LLM)"}
          </button>
          <div style={{fontSize:9,color:status.startsWith("✓")?"#00ff99":"#ff4422"}}>{status}</div>
        </div>
      )}

      {/* Vision panel */}
      {activePanel==="vision"&&(
        <div style={{position:"absolute",top:118,left:"50%",transform:"translateX(-50%)",background:"rgba(0,8,10,0.97)",border:`1px solid ${visColor}44`,padding:"12px 16px",borderRadius:3,width:520,zIndex:10,maxHeight:"70vh",overflowY:"auto"}}>
          <div style={{color:visColor,fontSize:11,fontWeight:"bold",marginBottom:8}}>👁 CAUSAL VISUAL CORTEX</div>
          <div style={{display:"flex",gap:8,marginBottom:10,alignItems:"center"}}>
            {!isVis?(
              <>
                <button onClick={()=>enableVision(8)} disabled={visionLoading} style={{padding:"4px 12px",borderRadius:2,fontSize:9,cursor:"pointer",background:"#002218",border:`1px solid ${visColor}`,color:visColor}}>
                  {visionLoading?"⏳…":"👁 ENABLE (8 slots)"}
                </button>
                <button onClick={()=>enableVision(12)} disabled={visionLoading} style={{padding:"4px 10px",borderRadius:2,fontSize:9,cursor:"pointer",background:"#001810",border:"1px solid #22cc88",color:"#22cc88"}}>12 slots</button>
              </>
            ):(
              <button onClick={disableVision} style={{padding:"4px 12px",borderRadius:2,fontSize:9,cursor:"pointer",background:"#180010",border:"1px solid #ff4422",color:"#ff4422"}}>✕ DISABLE</button>
            )}
            {isFR&&<div style={{fontSize:8,color:frColor}}>📌 fixed_root active — phys keys ограничены</div>}
          </div>

          {isVis&&<div style={{display:"flex",flexWrap:"wrap",gap:8,alignItems:"center",marginBottom:10}}>
            <button type="button" onClick={runVlmSlotLabels} disabled={vlmLoading}
              style={{padding:"4px 10px",borderRadius:2,fontSize:9,cursor:vlmLoading?"wait":"pointer",
                background:"#001828",border:`1px solid ${visColor}`,color:visColor}}>
              {vlmLoading?"⏳ VLM…":"🔬 VLM label slots"}
            </button>
            <label style={{fontSize:8,color:"#336655",display:"flex",alignItems:"center",gap:4,cursor:"pointer"}}>
              <input type="checkbox" checked={vlmWeakEdges} onChange={e=>setVlmWeakEdges(e.target.checked)}/>
              weak edges
            </label>
          </div>}

          {isVis&&visionData&&(
            <>
              <div style={{marginBottom:10}}>
                <div style={{fontSize:9,color:"#336655",marginBottom:4}}>
                  SLOTS — {visionData.active_slots??0}/{visionData.n_slots??8} active
                </div>
                <div style={{display:"grid",gridTemplateColumns:"repeat(4,1fr)",gap:4}}>
                  {Array.from({length:visionData.n_slots??8},(_,k)=>{
                    const val=visionData.slot_values?.[k]??0;
                    const varV=visionData.variability?.[k]??0;
                    const active=varV>0.02;
                    const col=SLOT_COLORS[k%SLOT_COLORS.length];
                    const sl=visionData.slot_labels?.[k];
                    return(
                      <div key={k} onClick={()=>setSelectedSlot(k)}
                        style={{background:selectedSlot===k?"rgba(68,255,204,0.08)":"rgba(0,10,12,0.8)",
                          border:`1px solid ${selectedSlot===k?col:active?"#1a3322":"#0a1a1a"}`,
                          borderRadius:3,padding:"5px 6px",cursor:"pointer"}}>
                        <div style={{display:"flex",justifyContent:"space-between",marginBottom:2}}>
                          <span style={{color:col,fontSize:8,fontWeight:"bold"}}>S{k}</span>
                          <span style={{color:active?"#44ffcc":"#334444",fontSize:7}}>{active?"●":"○"}</span>
                        </div>
                        {sl?.label&&<div style={{fontSize:6,color:"#66bbaa",marginBottom:2,minHeight:14}}>{sl.label} {typeof sl.confidence==="number"?`${(sl.confidence*100).toFixed(0)}%`:""}</div>}
                        <div style={{height:3,background:"#0a1a1a",borderRadius:2,overflow:"hidden",marginBottom:2}}>
                          <div style={{width:`${val*100}%`,height:"100%",background:col,transition:"width 0.5s"}}/>
                        </div>
                        <div style={{height:2,background:"#050d0d",borderRadius:2,overflow:"hidden"}}>
                          <div style={{width:`${Math.min(varV*100*5,100)}%`,height:"100%",background:`${col}88`}}/>
                        </div>
                        <div style={{fontSize:6,color:"#225544",marginTop:1,display:"flex",justifyContent:"space-between"}}>
                          <span>{val.toFixed(2)}</span><span>σ={varV.toFixed(3)}</span>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
              {attnFrame&&<div style={{...sep,paddingTop:8}}>
                <div style={{fontSize:9,color:"#336655",marginBottom:4}}>
                  MASK — <span style={{color:SLOT_COLORS[selectedSlot%8]}}>slot_{selectedSlot}</span>
                  {visionData.slot_labels?.[selectedSlot]?.label&&
                    <span style={{color:"#66bbaa",marginLeft:6}}>({visionData.slot_labels[selectedSlot].label})</span>}
                </div>
                <img src={`data:image/jpeg;base64,${attnFrame}`}
                  style={{width:"100%",borderRadius:3,border:`1px solid ${SLOT_COLORS[selectedSlot%8]}44`}}/>
                <div style={{display:"flex",gap:3,marginTop:4,flexWrap:"wrap"}}>
                  {Array.from({length:visionData.n_slots??8},(_,k)=>(
                    <button key={k} onClick={()=>setSelectedSlot(k)}
                      style={{padding:"2px 6px",borderRadius:2,fontSize:8,cursor:"pointer",
                        background:selectedSlot===k?"rgba(0,20,15,0.9)":"transparent",
                        border:`1px solid ${selectedSlot===k?SLOT_COLORS[k%8]:"#0a1a1a"}`,
                        color:selectedSlot===k?SLOT_COLORS[k%8]:"#334444"}}>S{k}</button>
                  ))}
                </div>
              </div>}
            </>
          )}
        </div>
      )}

      {/* Left HUD */}
      <div style={{position:"absolute",top:118,left:14}}>
        <div style={{background:"rgba(2,5,14,0.92)",border:`1px solid ${fallen?"#660011":isFR?frColor+"33":isVis?visColor+"33":wCol+"33"}`,borderLeft:`3px solid ${fallen?"#ff2244":isFR?frColor:isVis?visColor:wCol}`,padding:"10px 14px",minWidth:235,borderRadius:3}}>
          <div style={{color:fallen?"#ff4422":isFR?frColor:isVis?visColor:wCol,fontSize:11,fontWeight:"bold",marginBottom:5}}>
            {fallen?"💀":isFR?"📌":isVis?"👁":"◈"} NOVA {isFR?"Fixed Base":isVis?"Visual":"Singleton"}
            <span style={{color:"#1a2244",marginLeft:6,fontSize:8}}>d={ui.gnnD}</span>
          </div>

          {fallen&&<div style={{background:"rgba(255,0,34,0.08)",border:"1px solid #ff224433",borderRadius:2,padding:"3px 7px",marginBottom:4,fontSize:8,color:"#ff4422"}}>
            ⚠ FALLEN × {ui.fallCount}
          </div>}

          {isFR&&<div style={{background:"rgba(255,204,0,0.06)",border:"1px solid #ffcc0033",borderRadius:2,padding:"3px 7px",marginBottom:4,fontSize:8,color:frColor}}>
            📌 FIXED BASE · arm+spine+cubes · no falling
          </div>}

          {isVis&&!isFR&&<div style={{background:"rgba(0,255,200,0.05)",border:"1px solid #44ffcc33",borderRadius:2,padding:"3px 7px",marginBottom:4,fontSize:8,color:"#44ffcc"}}>
            👁 visual cortex · {ui.vision?.active_slots??0} active · {ui.visionTicks} ticks
          </div>}

          <table style={{borderCollapse:"collapse",width:"100%",fontSize:9}}><tbody>
            {[
              ["Φ autonomy",  <span style={{color:phiC(a.phi??0)}}>{((a.phi??0)*100).toFixed(1)}%</span>],
              ["CG d|G|/dt",  <span style={{color:cgC(a.compressionGain??0)}}>{a.compressionGain>=0?"+":""}{(a.compressionGain??0).toFixed(4)}</span>],
              ["α trust",    `${Math.round((a.alphaMean??0)*100)}%`],
              ["h(W) DAG",   <span style={{color:hWC(a.hW??0)}}>{(a.hW??0).toFixed(4)}</span>],
              ["GNN d",      <span style={{color:isFR?frColor:isVis?visColor:wCol}}>{ui.gnnD}</span>],
              ["Edges",      a.edgeCount??0],
              ["do()/blk",   `${a.totalInterventions??0}/${a.totalBlocked??0}`],
              ["block%",     <span style={{color:blkC(blkR)}}>{(blkR*100).toFixed(1)}%</span>],
              ["Discovery",  <span style={{color:(a.discoveryRate??0)>0.2?"#00ff99":"#aabbcc"}}>{((a.discoveryRate??0)*100).toFixed(0)}%</span>],
            ].map(([l,v],k)=>(
              <tr key={k}><td style={{color:"#2a4466",paddingRight:8,paddingBottom:2}}>{l}</td><td style={{color:"#aabbcc",textAlign:"right"}}>{v}</td></tr>
            ))}
          </tbody></table>

          {a.valueLayer&&<div style={{...sep,fontSize:8}}>
            <div style={{display:"flex",justifyContent:"space-between"}}>
              <span style={{color:"#441133"}}>VL · {a.valueLayer.vl_phase}{a.valueLayer.fixed_root_mode?" · 📌fixed":""}</span>
              <span style={{color:blkC(blkR)}}>{(blkR*100).toFixed(1)}%</span>
            </div>
          </div>}
        </div>
      </div>

      {/* Right HUD */}
      <div style={{position:"absolute",top:118,right:14,background:"rgba(2,5,14,0.92)",border:"1px solid #0a1a2e",padding:"10px 14px",borderRadius:3,fontSize:9,maxWidth:200}}>
        <div style={{color:"#1a3355",marginBottom:6,fontSize:9}}>NOVA · CURRICULUM</div>

        <label style={{display:"flex",alignItems:"center",gap:6,cursor:"pointer",color:"#556688",fontSize:8,marginBottom:4}}>
          <input type="checkbox" checked={showCubes} onChange={e=>setShowCubes(e.target.checked)} style={{accentColor:wCol}}/>
          show objects
        </label>

        <div style={{...sep,marginBottom:6}}>
          <div style={{fontSize:8,color:"#333344",marginBottom:4}}>CURRICULUM PATH:</div>
          <div style={{fontSize:8,color:isFR?frColor:"#445566",fontWeight:isFR?"bold":"normal",marginBottom:2}}>
            {isFR?"● ":"→ "}Step 1: Fixed Base (arms only)
            {isFR&&<span style={{color:"#665500"}}> · active</span>}
          </div>
          <div style={{fontSize:8,color:!isFR&&ui.currentWorld==="humanoid"?"#00cc88":"#224433",marginBottom:2}}>
            → Step 2: Balance assist
          </div>
          <div style={{fontSize:8,color:"#224433",marginBottom:2}}>
            → Step 3: Free locomotion
          </div>
        </div>

        <div style={{...sep}}>
          {Object.entries(WORLD_COLORS).map(([w,c])=>(
            <div key={w} onClick={()=>switchWorld(w)} style={{fontSize:8,marginBottom:2,cursor:"pointer",color:ui.currentWorld===w?c:"#223344"}}>
              {ui.currentWorld===w?"▶ ":"  "}{w}
            </div>
          ))}
        </div>

        {isVis&&ui.vision&&<div style={{...sep}}>
          <div style={{fontSize:8,color:visColor,marginBottom:3}}>SLOTS</div>
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

        {status&&<div style={{...sep,fontSize:8,color:status.startsWith("✓")?"#00ff99":"#ff4422",wordBreak:"break-all"}}>
          {status}
        </div>}
      </div>

      {/* Event log */}
      <div style={{position:"absolute",bottom:14,left:14,right:14,background:"rgba(2,5,14,0.92)",border:"1px solid #0a1a2e",padding:"8px 14px",borderRadius:3,maxHeight:220, maxWidth: "500px", margin: "0 auto",overflowY:"auto",overflowX:"hidden"}}>
        <div style={{color:"#0a1a2e",fontSize:9,letterSpacing:"0.1em",marginBottom:4}}>
          CAUSAL STREAM {connected?"● ONLINE":"○ OFFLINE"}
          {isFR&&<span style={{color:frColor}}> · 📌 FIXED BASE</span>}
          {isVis&&<span style={{color:visColor}}> · 👁 VISUAL</span>}
        </div>
        {ui.events.map((ev,i)=>(
          <div key={i} style={{color:ev.color??"#334455",fontSize:9,marginBottom:2,opacity:Math.max(0.15,1-i*.12),wordBreak:"break-word",whiteSpace:"pre-wrap"}}>
            [{String(ev.tick??0).padStart(5,"0")}] › {ev.text}
          </div>
        ))}
      </div>
    </div>
  );
}