import { useState, useEffect, useRef } from "react";
import * as THREE from "three";
import { useRKKStream } from "../../hooks/useRKKStream";

const WORLD = 16;
const AGENT_COLORS_HEX = [0x00ff99, 0x0099ff, 0xff9900, 0xcc44ff];
const AGENT_COLORS_CSS = ["#00ff99", "#0099ff", "#ff9900", "#cc44ff"];
const AGENT_NAMES      = ["Nova",    "Aether",   "Lyra",    "Ignis"];
const AGENT_ENVS       = ["physics", "chemistry","logic",   "pybullet"];
const DEMON_COLOR      = 0xff2244;
const GRAPH_LINES      = 14;
const PHASE_NAMES      = ["","Causal Crib","Robotic Explorer","Social Sandbox","Value Lock","Open Reality"];
const ATTACK_MODES     = { probe:"#335566", targeted:"#ff8844", siege:"#ff2244", anti_byz:"#aa00ff" };

const hWColor  = h => h<0.01?"#00ff99":h<0.5?"#aacc00":h<2?"#ffaa00":"#ff4422";
const phiColor = p => p>0.6?"#00ff99":p>0.3?"#aacc00":"#ff8844";
const blkColor = r => r>0.3?"#ff4422":r>0.1?"#ffaa00":"#335544";
const modeColor= m => ATTACK_MODES[m]??"#335566";
const devColor = d => d<0.05?"#00ff99":d<0.15?"#aacc00":"#ff8844";

function normalizeAgent(a) {
  return {
    id:a.id??0, name:a.name??"?", envType:a.env_type??"—",
    activation:a.activation??"relu",
    graphMdl:a.graph_mdl??0, compressionGain:a.compression_gain??0,
    alphaMean:a.alpha_mean??0.05, phi:a.phi??0.1,
    nodeCount:a.node_count??0, edgeCount:a.edge_count??0,
    totalInterventions:a.total_interventions??0, totalBlocked:a.total_blocked??0,
    lastDo:a.last_do??"—", lastBlockedReason:a.last_blocked_reason??"",
    discoveryRate:a.discovery_rate??0,
    hW:a.h_W??0, notears:a.notears??null, temporal:a.temporal??null,
    system1:a.system1??null, valueLayer:a.value_layer??null, edges:a.edges??[],
  };
}

function normalizeFrame(raw) {
  const nAgents = raw.n_agents ?? 4;
  const agents  = (raw.agents ?? []).slice(0, nAgents).map(normalizeAgent);
  // Дополняем до nAgents если вдруг меньше пришло
  while (agents.length < nAgents) agents.push(normalizeAgent({ id: agents.length }));
  return {
    tick:raw.tick??0, phase:raw.phase??1, entropy:raw.entropy??100,
    agents, nAgents,
    demon:raw.demon??{energy:1,cooldown:0,mode:"probe",success_rate:0,last_target:0},
    tomLinks:raw.tom_links??[], events:raw.events??[],
    valueLayer:raw.value_layer??null,
    byzantine:raw.byzantine??null,
    motif:raw.motif??null,
    pybullet:raw.pybullet??null,
    multiprocess:raw.multiprocess??false,
  };
}

export default function RKKv5() {
  const mountRef   = useRef(null);
  const rafRef     = useRef(null);
  const { frame: wsFrame, connected, setSpeed: wsSetSpeed } = useRKKStream();
  const wsFrameRef = useRef(wsFrame);
  wsFrameRef.current = wsFrame;

  const [speed,       setSpeedLocal]  = useState(1);
  const [ui,          setUI]          = useState(() => normalizeFrame(wsFrame));
  const [activePanel, setActivePanel] = useState(null);
  const [seedText,    setSeedText]    = useState('[\n  {"from_": "Temp", "to": "Pressure", "weight": 0.8}\n]');
  const [seedAgent,   setSeedAgent]   = useState(0);
  const [seedStatus,  setSeedStatus]  = useState("");
  const [ragLoading,  setRagLoading]  = useState(false);
  const [ragResults,  setRagResults]  = useState([]);
  const [demonStats,  setDemonStats]  = useState(null);

  const setSpeed = s => { setSpeedLocal(s); if (connected) wsSetSpeed(s); };
  useEffect(() => { setUI(normalizeFrame(wsFrame)); }, [wsFrame]);

  useEffect(() => {
    if (!connected) return;
    const iv = setInterval(async () => {
      try {
        const d = await fetch("http://localhost:8000/demon/stats").then(r=>r.json());
        setDemonStats(d);
      } catch {}
    }, 3000);
    return () => clearInterval(iv);
  }, [connected]);

  const injectSeeds = async () => {
    try {
      const edges = JSON.parse(seedText);
      const res = await fetch("http://localhost:8000/inject-seeds", {
        method:"POST", headers:{"Content-Type":"application/json"},
        body:JSON.stringify({agent_id:seedAgent, edges, source:"manual"}),
      });
      const d = await res.json();
      setSeedStatus(`✓ ${d.injected} edges → ${d.agent}`);
    } catch(e) { setSeedStatus(`✗ ${e.message}`); }
  };

  const ragAutoSeed = async () => {
    setRagLoading(true); setSeedStatus("⏳ Auto-seeding…");
    try {
      const res = await fetch("http://localhost:8000/rag/auto-seed-all",{method:"POST"});
      const d = await res.json();
      setRagResults(d.results??[]);
      setSeedStatus(`✓ ${d.results?.reduce((s,r)=>s+r.injected,0)??0} edges total`);
    } catch(e) { setSeedStatus(`✗ ${e.message}`); }
    finally { setRagLoading(false); }
  };

  // ── Three.js ──────────────────────────────────────────────────────────────
  useEffect(() => {
    const mount = mountRef.current;
    if (!mount) return;

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(mount.clientWidth, mount.clientHeight);
    mount.appendChild(renderer.domElement);

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x010810);
    scene.fog = new THREE.FogExp2(0x010810, 0.018);
    const camera = new THREE.PerspectiveCamera(52, mount.clientWidth/mount.clientHeight, 0.1, 220);
    camera.position.set(0, 16, 32); camera.lookAt(0, 1, 0);
    scene.add(new THREE.AmbientLight(0x112233, 1.8));
    const sun = new THREE.DirectionalLight(0x334488, 1.2); sun.position.set(5,10,5); scene.add(sun);
    scene.add(new THREE.GridHelper(WORLD*2+4, 32, 0x0a1e38, 0x050f1e));

    // 4 агента — позиции: треугольник + центр, или ромб
    const AGENT_POSITIONS = [
      [-6, 1, -2],  // Nova
      [ 2, 1,  3],  // Aether
      [-2, 1,  3],  // Lyra
      [ 6, 1, -2],  // Ignis (PyBullet) — правее всех
    ];

    const agentGroups = AGENT_COLORS_HEX.map((col, i) => {
      const g = new THREE.Group();

      // PyBullet агент (Ignis) — другая форма: тетраэдр вместо сферы
      const isPybullet = i === 3;
      const bodyGeo = isPybullet
        ? new THREE.TetrahedronGeometry(0.65, 0)
        : new THREE.SphereGeometry(0.6, 22, 22);
      const body  = new THREE.Mesh(bodyGeo, new THREE.MeshPhongMaterial({color:col,emissive:col,emissiveIntensity:0.4,transparent:true,opacity:0.92}));
      const ring  = new THREE.Mesh(new THREE.TorusGeometry(0.95,0.05,8,44), new THREE.MeshBasicMaterial({color:col,transparent:true,opacity:0.4}));
      const ringS = new THREE.Mesh(new THREE.TorusGeometry(1.4,0.025,6,40), new THREE.MeshBasicMaterial({color:col,transparent:true,opacity:0.15}));
      const byz   = new THREE.Mesh(new THREE.TorusGeometry(1.7,0.02,5,36), new THREE.MeshBasicMaterial({color:0x004488,transparent:true,opacity:0}));
      const ragRing = new THREE.Mesh(new THREE.TorusGeometry(1.9,0.015,4,32), new THREE.MeshBasicMaterial({color:0xaa8800,transparent:true,opacity:0}));
      const shield= new THREE.Mesh(new THREE.TorusGeometry(1.1,0.04,6,40), new THREE.MeshBasicMaterial({color:0xff4422,transparent:true,opacity:0}));
      // PyBullet — дополнительное кольцо орбит объектов
      const physRing = isPybullet
        ? new THREE.Mesh(new THREE.TorusGeometry(2.2,0.01,4,28), new THREE.MeshBasicMaterial({color:0xcc44ff,transparent:true,opacity:0.1}))
        : null;

      shield.rotation.x=Math.PI/3; ring.rotation.x=Math.PI/2;
      g.add(body,ring,ringS,byz,ragRing,shield);
      if (physRing) g.add(physRing);
      g.add(new THREE.PointLight(col,0.7,5));
      g.position.set(...AGENT_POSITIONS[i]);
      g.userData = {
        body,ring,ringS,byz,ragRing,shield,physRing,
        vel: new THREE.Vector3((Math.random()-.5)*0.025, 0, (Math.random()-.5)*0.025),
        ragPulse: 0,
        isPybullet,
      };
      scene.add(g); return g;
    });

    // PyBullet объекты (3 маленьких куба вокруг Ignis)
    const physObjects = [0,1,2].map(i => {
      const geo = new THREE.BoxGeometry(0.2,0.2,0.2);
      const mat = new THREE.MeshPhongMaterial({color:0xcc44ff,emissive:0xaa22cc,emissiveIntensity:0.3,transparent:true,opacity:0.7});
      const m = new THREE.Mesh(geo,mat);
      scene.add(m); return m;
    });

    const demon = new THREE.Mesh(new THREE.OctahedronGeometry(0.8,0), new THREE.MeshPhongMaterial({color:DEMON_COLOR,emissive:0xff0022,emissiveIntensity:0.7,transparent:true,opacity:0.9}));
    const demonSiege = new THREE.Mesh(new THREE.OctahedronGeometry(1.1,0), new THREE.MeshBasicMaterial({color:0xff4422,transparent:true,opacity:0,wireframe:true}));
    demon.add(new THREE.PointLight(DEMON_COLOR,0.9,6)); demon.add(demonSiege);
    demon.position.set(WORLD-2,1.2,WORLD-2);
    demon.userData={vel:new THREE.Vector3(-0.035,0,-0.025)};
    scene.add(demon);

    const orbitNodes = agentGroups.map((_,ai) => Array.from({length:GRAPH_LINES},()=>{
      const m=new THREE.Mesh(new THREE.SphereGeometry(0.1,6,6), new THREE.MeshBasicMaterial({color:AGENT_COLORS_HEX[ai],transparent:true,opacity:0}));
      scene.add(m); return m;
    }));
    const causalLines = agentGroups.map((_,ai) => Array.from({length:GRAPH_LINES},()=>{
      const geom=new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(),new THREE.Vector3()]);
      const l=new THREE.Line(geom,new THREE.LineBasicMaterial({color:AGENT_COLORS_HEX[ai],transparent:true,opacity:0}));
      scene.add(l); return l;
    }));

    // Byzantine + ToM lines между агентами (все пары)
    const byzLines=[], tomLines=[];
    for(let a=0;a<4;a++) for(let b=a+1;b<4;b++) {
      [byzLines,tomLines].forEach((arr,idx)=>{
        const geom=new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(),new THREE.Vector3()]);
        const l=new THREE.Line(geom,new THREE.LineBasicMaterial({color:idx?0x003366:0x002255,transparent:true,opacity:0}));
        l.userData={a,b}; scene.add(l); arr.push(l);
      });
    }

    const pPos=new Float32Array(500*3);
    for(let i=0;i<500;i++){pPos[i*3]=(Math.random()-.5)*50;pPos[i*3+1]=Math.random()*16;pPos[i*3+2]=(Math.random()-.5)*50;}
    const pGeom=new THREE.BufferGeometry();
    pGeom.setAttribute("position",new THREE.BufferAttribute(pPos,3));
    scene.add(new THREE.Points(pGeom,new THREE.PointsMaterial({color:0x003399,size:0.07,transparent:true,opacity:0.3})));

    let frame=0, camAngle=0, byzPulse=0;

    function loop(){
      rafRef.current=requestAnimationFrame(loop);
      frame++;byzPulse++;
      const ds=normalizeFrame(wsFrameRef.current);
      camAngle+=0.001;
      camera.position.x=Math.sin(camAngle)*32; camera.position.z=Math.cos(camAngle)*32;
      camera.lookAt(0,2,0);

      const byzActive=ds.byzantine&&ds.byzantine.round>0;
      const byzOp=byzActive?Math.max(0,Math.sin(byzPulse*0.08)*0.2):0;
      const demonMode=ds.demon?.mode??"probe";
      const isSiege=demonMode==="siege";
      demonSiege.material.opacity=isSiege?0.2+Math.sin(frame*0.12)*0.1:0;

      // PyBullet объекты — орбитируют вокруг Ignis
      const ignisPos = agentGroups[3].position;
      const pbStats  = ds.pybullet;
      physObjects.forEach((obj,k)=>{
        const angle=(k/3)*Math.PI*2+frame*0.02*(1+k*0.3);
        const r=2.0+Math.sin(frame*0.03+k)*0.3;
        const vy=pbStats?.phi??0;
        obj.position.set(
          ignisPos.x+Math.cos(angle)*r,
          ignisPos.y+0.3+Math.sin(frame*0.05+k*1.2)*0.5*(1+vy),
          ignisPos.z+Math.sin(angle)*r,
        );
        obj.rotation.x+=0.04*(k+1); obj.rotation.y+=0.03*(k+1);
        obj.material.opacity=0.4+vy*0.4;
      });

      agentGroups.forEach((g,i)=>{
        const snap=ds.agents[i]; if(!snap) return;
        const vel=g.userData.vel;
        // PyBullet агент дрейфует медленнее
        const driftScale = g.userData.isPybullet ? 0.6 : 1.0;
        g.position.x+=vel.x*driftScale; g.position.z+=vel.z*driftScale;
        if(Math.abs(g.position.x)>WORLD)vel.x*=-1;
        if(Math.abs(g.position.z)>WORLD)vel.z*=-1;
        g.position.y=1+Math.sin(frame*0.05+i*2.1)*0.15;
        if(frame%(150+i*30)===0)g.userData.vel=new THREE.Vector3((Math.random()-.5)*0.025,0,(Math.random()-.5)*0.025);

        g.userData.ring.rotation.z+=0.015+snap.alphaMean*0.03;
        g.userData.ring.rotation.y+=0.008;
        g.userData.ringS.rotation.x+=0.002+snap.phi*0.008;
        g.userData.byz.rotation.y+=0.003;
        g.userData.byz.material.opacity=byzOp;
        g.userData.ragPulse=Math.max(0,g.userData.ragPulse-1);
        g.userData.ragRing.rotation.z+=0.008;
        g.userData.ragRing.material.opacity=g.userData.ragPulse>0?0.1+Math.sin(frame*0.2)*0.05:0;

        // PhysRing для Ignis — пульсирует по discovery rate
        if(g.userData.physRing){
          g.userData.physRing.rotation.z+=0.005;
          g.userData.physRing.material.opacity=0.05+snap.discoveryRate*0.3;
        }

        const blkRate=snap.valueLayer?.block_rate??0;
        g.userData.shield.material.opacity=blkRate>0.1?0.15+Math.sin(frame*0.15+i)*0.1:0;

        // PyBullet агент (Ignis) — тетраэдр вращается быстрее
        if(g.userData.isPybullet){
          g.userData.body.rotation.x+=0.025;
          g.userData.body.rotation.y+=0.018;
        }

        const dagGlow=Math.max(0,0.15-Math.min((snap.hW??0)*0.05,0.15));
        g.userData.body.material.emissiveIntensity=0.2+Math.max(0,snap.compressionGain)*0.1+dagGlow+Math.sin(frame*0.07+i)*0.06;

        const visCount=Math.min((snap.edgeCount??0)+2,GRAPH_LINES);
        orbitNodes[i].forEach((node,k)=>{
          if(k<visCount){
            const angle=(k/visCount)*Math.PI*2+frame*0.016;
            const r=1.6+(k%2)*0.5;
            node.position.set(g.position.x+Math.cos(angle)*r,g.position.y+Math.sin(frame*0.04+k*0.8)*0.35+0.2,g.position.z+Math.sin(angle)*r);
            node.material.opacity=0.4+snap.alphaMean*0.5;
          } else node.material.opacity=0;
        });
        causalLines[i].forEach((line,k)=>{
          if(k<visCount-1){
            const na=orbitNodes[i][k].position,nb=orbitNodes[i][(k+1)%visCount].position;
            const pa=line.geometry.attributes.position;
            pa.setXYZ(0,na.x,na.y,na.z);pa.setXYZ(1,nb.x,nb.y,nb.z);pa.needsUpdate=true;
            line.material.color.set((snap.edges[k]?.weight??0)<0?0xff4422:AGENT_COLORS_HEX[i]);
            line.material.opacity=0.3+snap.alphaMean*0.45;
          } else line.material.opacity=0;
        });
      });

      const dTarget=agentGroups[isSiege?ds.demon.last_target:ds.tick%4];
      const chase=dTarget.position.clone().sub(demon.position).normalize().multiplyScalar(isSiege?0.045:0.03);
      demon.userData.vel.lerp(chase,isSiege?0.08:0.05);
      demon.position.add(demon.userData.vel);
      if(Math.abs(demon.position.x)>WORLD)demon.userData.vel.x*=-1;
      if(Math.abs(demon.position.z)>WORLD)demon.userData.vel.z*=-1;
      demon.position.y=1.2+Math.sin(frame*0.08)*0.3;
      demon.rotation.y+=isSiege?0.08:0.045; demon.rotation.x+=0.028;

      byzLines.forEach(l=>{
        const pa=agentGroups[l.userData.a].position,pb=agentGroups[l.userData.b].position,p=l.geometry.attributes.position;
        p.setXYZ(0,pa.x,pa.y,pa.z);p.setXYZ(1,pb.x,pb.y,pb.z);p.needsUpdate=true;
        l.material.opacity=byzActive?byzOp*0.6:0;
      });
      tomLines.forEach(l=>{
        const link=ds.tomLinks.find(x=>x.a===l.userData.a&&x.b===l.userData.b);
        const pa=agentGroups[l.userData.a].position,pb=agentGroups[l.userData.b].position,p=l.geometry.attributes.position;
        p.setXYZ(0,pa.x,pa.y,pa.z);p.setXYZ(1,pb.x,pb.y,pb.z);p.needsUpdate=true;
        l.material.opacity=link?link.strength*0.6:0;
      });

      renderer.render(scene,camera);
    }
    loop();

    const onResize=()=>{if(!mount)return;camera.aspect=mount.clientWidth/mount.clientHeight;camera.updateProjectionMatrix();renderer.setSize(mount.clientWidth,mount.clientHeight);};
    window.addEventListener("resize",onResize);
    return()=>{cancelAnimationFrame(rafRef.current);window.removeEventListener("resize",onResize);if(mount.contains(renderer.domElement))mount.removeChild(renderer.domElement);renderer.dispose();};
  },[]);

  const mono={fontFamily:"'Courier New',monospace"};
  const sep={borderTop:"1px solid #081e30",marginTop:4,paddingTop:4};
  const totalBlocked=ui.valueLayer?.total_blocked_all??0;
  const demonMode=ui.demon?.mode??"probe";

  return(
    <div style={{position:"relative",width:"100%",height:"100vh",background:"#010810",overflow:"hidden",...mono}}>
      <div ref={mountRef} style={{position:"absolute",inset:0}}/>

      {/* Status */}
      <div style={{position:"absolute",top:14,right:14,background:connected?"rgba(0,40,10,0.9)":"rgba(30,10,0,0.9)",border:`1px solid ${connected?"#00aa44":"#aa4400"}`,padding:"4px 10px",borderRadius:2,fontSize:9,color:connected?"#00ff88":"#ff8844"}}>
        {connected?"● PYTHON BACKEND":"○ OFFLINE"} · MP · {ui.nAgents}A
      </div>

      {/* Header */}
      <div style={{position:"absolute",top:14,left:"50%",transform:"translateX(-50%)",background:"rgba(0,12,28,0.9)",border:"1px solid #0a2a44",padding:"7px 20px",textAlign:"center",borderRadius:2,boxShadow:"0 0 24px #00224455",whiteSpace:"nowrap"}}>
        <div style={{color:"#00ff99",fontSize:12,fontWeight:"bold",letterSpacing:"0.18em"}}>
          RKK v5 — PYBULLET · BYZANTINE · RAG
        </div>
        <div style={{color:"#115577",fontSize:10,marginTop:2,letterSpacing:"0.08em"}}>
          PHASE {ui.phase}/5 › {PHASE_NAMES[ui.phase]}
          &nbsp;│&nbsp;ENT:<span style={{color:ui.entropy<30?"#00ff99":"#aaccdd"}}>{ui.entropy}%</span>
          &nbsp;│&nbsp;T:{ui.tick}
          &nbsp;│&nbsp;<span style={{color:modeColor(demonMode)}}>◆{demonMode}</span>
          {ui.demon&&<span style={{color:"#443300",marginLeft:4}}>E:{(ui.demon.energy*100).toFixed(0)}%</span>}
          &nbsp;│&nbsp;<span style={{color:totalBlocked>0?"#ff8844":"#335544"}}>🛡{totalBlocked}</span>
          {ui.byzantine&&<span style={{color:"#004466",marginLeft:4}}>🗳R{ui.byzantine.round}</span>}
          {ui.motif?.last_donor!=null&&<span style={{color:"#224466",marginLeft:4}}>🧬{["N","A","L","I"][ui.motif.last_donor]}</span>}
          &nbsp;│&nbsp;<span style={{color:"#6622aa"}}>◆Ignis</span>
          {ui.pybullet&&<span style={{color:"#441166",marginLeft:4}}>Φ={ui.pybullet.phi?.toFixed(2)} DR={((ui.pybullet.discovery_rate??0)*100).toFixed(0)}%</span>}
        </div>
      </div>

      {/* Controls */}
      <div style={{position:"absolute",top:80,left:"50%",transform:"translateX(-50%)",background:"rgba(0,8,20,0.85)",border:"1px solid #081e30",padding:"5px 12px",borderRadius:2,display:"flex",gap:6,alignItems:"center",fontSize:9}}>
        <span style={{color:"#224455"}}>SPEED</span>
        {[1,2,4,8].map(s=><button key={s} onClick={()=>setSpeed(s)} style={{padding:"2px 7px",borderRadius:2,fontSize:9,cursor:"pointer",background:speed===s?"#001e38":"transparent",border:`1px solid ${speed===s?"#00aaff":"#081e30"}`,color:speed===s?"#00aaff":"#334455"}}>{s}×</button>)}
        <span style={{color:"#334455"}}>│</span>
        {PHASE_NAMES.slice(1).map((name,i)=>{const ph=i+1;return(
          <div key={ph} style={{padding:"2px 6px",borderRadius:2,fontSize:9,background:ui.phase===ph?"#001e38":"transparent",border:`1px solid ${ui.phase===ph?"#00aaff":ui.phase>ph?"#003322":"#081e30"}`,color:ui.phase===ph?"#00aaff":ui.phase>ph?"#00aa55":"#223344"}}>{ui.phase>ph?"✓":ph} {name}</div>
        );})}
        <span style={{color:"#334455"}}>│</span>
        {[["💉","seeds"],["🌐","rag"],["◆","demon"],["⚙","pybullet"]].map(([icon,panel])=>(
          <button key={panel} onClick={()=>setActivePanel(v=>v===panel?null:panel)} style={{padding:"2px 7px",borderRadius:2,fontSize:9,cursor:"pointer",background:activePanel===panel?"#1a0820":"transparent",border:`1px solid ${activePanel===panel?"#884400":"#333"}`,color:activePanel===panel?"#cc44ff":"#445555"}}>{icon}</button>
        ))}
      </div>

      {/* Seeds panel */}
      {activePanel==="seeds"&&(
        <div style={{position:"absolute",top:118,left:"50%",transform:"translateX(-50%)",background:"rgba(8,6,0,0.96)",border:"1px solid #443300",padding:"12px 16px",borderRadius:2,width:420,zIndex:10}}>
          <div style={{color:"#886600",fontSize:10,marginBottom:6}}>💉 MANUAL SEED INJECTION</div>
          <div style={{display:"flex",gap:5,marginBottom:6}}>
            {AGENT_NAMES.map((name,i)=>(
              <button key={i} onClick={()=>setSeedAgent(i)} style={{flex:1,padding:"2px 4px",borderRadius:2,fontSize:9,cursor:"pointer",background:seedAgent===i?"#221100":"transparent",border:`1px solid ${seedAgent===i?AGENT_COLORS_CSS[i]:"#332200"}`,color:seedAgent===i?AGENT_COLORS_CSS[i]:"#554400"}}>{name}</button>
            ))}
          </div>
          {seedAgent===3&&<div style={{fontSize:8,color:"#552288",marginBottom:4}}>Ignis vars: obj0_x, obj0_vx, obj1_y, obj1_vy, obj2_vz… (18 vars)</div>}
          <textarea value={seedText} onChange={e=>setSeedText(e.target.value)} style={{width:"100%",height:80,background:"#050300",border:"1px solid #332200",color:"#aa8800",fontSize:9,padding:6,borderRadius:2,fontFamily:"monospace",resize:"none",boxSizing:"border-box"}}/>
          <div style={{display:"flex",justifyContent:"space-between",alignItems:"center",marginTop:6}}>
            <button onClick={injectSeeds} style={{padding:"3px 12px",borderRadius:2,fontSize:9,cursor:"pointer",background:"#221100",border:"1px solid #886600",color:"#ffaa00"}}>INJECT</button>
            <span style={{color:seedStatus.startsWith("✓")?"#00ff99":"#ff4422",fontSize:9}}>{seedStatus}</span>
          </div>
        </div>
      )}

      {/* RAG panel */}
      {activePanel==="rag"&&(
        <div style={{position:"absolute",top:118,left:"50%",transform:"translateX(-50%)",background:"rgba(0,6,14,0.96)",border:"1px solid #004433",padding:"12px 16px",borderRadius:2,width:420,zIndex:10}}>
          <div style={{color:"#00aa66",fontSize:10,marginBottom:6}}>🌐 RAG SEED PIPELINE</div>
          <button onClick={ragAutoSeed} disabled={ragLoading} style={{width:"100%",padding:"5px",borderRadius:2,fontSize:9,cursor:"pointer",background:ragLoading?"#001a0a":"#002211",border:"1px solid #006633",color:ragLoading?"#225544":"#00ff88",marginBottom:8}}>
            {ragLoading?"⏳ Fetching Wikipedia…":"🌐 AUTO-SEED ALL AGENTS"}
          </button>
          {ragResults.length>0&&<div style={{fontSize:9}}>
            {ragResults.map((r,i)=><div key={i} style={{color:AGENT_COLORS_CSS[i]??"#aaa",marginBottom:2,display:"flex",justifyContent:"space-between"}}>
              <span>{r.agent} ({r.preset})</span><span style={{color:"#336655"}}>{r.injected} edges · {r.source}</span>
            </div>)}
          </div>}
          <div style={{...sep,fontSize:8,color:"#1a3322"}}>
            PyBullet seeds: obj0_vx→obj0_x, obj1_vy→obj1_y, etc.<br/>
            NOTEARS анилирует неверные, физика подтверждает верные.
          </div>
          <div style={{marginTop:4,fontSize:9,color:seedStatus.startsWith("✓")?"#00ff99":"#ff4422"}}>{seedStatus}</div>
        </div>
      )}

      {/* Demon panel */}
      {activePanel==="demon"&&demonStats&&(
        <div style={{position:"absolute",top:118,left:"50%",transform:"translateX(-50%)",background:"rgba(14,0,0,0.96)",border:"1px solid #440000",padding:"12px 16px",borderRadius:2,width:360,zIndex:10}}>
          <div style={{color:"#ff4422",fontSize:10,marginBottom:6}}>◆ DEMON v2 — PPO-lite</div>
          <table style={{width:"100%",borderCollapse:"collapse",fontSize:9}}><tbody>
            {[["Mode",<span style={{color:modeColor(demonStats.mode)}}>{demonStats.mode?.toUpperCase()}</span>],["Energy",`${(demonStats.energy*100).toFixed(0)}%`],["Success",`${((demonStats.success_rate??0)*100).toFixed(1)}%`],["Reward",<span style={{color:(demonStats.mean_recent_reward??0)>0?"#ff8844":"#335544"}}>{(demonStats.mean_recent_reward??0).toFixed(3)}</span>]].map(([l,v],k)=>(
              <tr key={k}><td style={{color:"#553333",paddingRight:10,paddingBottom:3}}>{l}</td><td style={{color:"#cc6644",textAlign:"right"}}>{v}</td></tr>
            ))}
          </tbody></table>
          {demonStats.memory&&<div style={{...sep}}>
            {demonStats.memory.map(m=>(
              <div key={m.agent} style={{fontSize:9,display:"flex",justifyContent:"space-between",marginBottom:2}}>
                <span style={{color:AGENT_COLORS_CSS[m.agent]}}>{AGENT_NAMES[m.agent]}</span>
                <span style={{color:"#553333"}}>✓{m.success} ✗{m.fail} ΔΦ={m.phi_drop?.toFixed(4)}</span>
              </div>
            ))}
          </div>}
        </div>
      )}

      {/* PyBullet panel */}
      {activePanel==="pybullet"&&(
        <div style={{position:"absolute",top:118,left:"50%",transform:"translateX(-50%)",background:"rgba(8,0,20,0.96)",border:"1px solid #441188",padding:"14px 18px",borderRadius:2,width:400,zIndex:10}}>
          <div style={{color:"#cc44ff",fontSize:10,marginBottom:8,letterSpacing:"0.1em"}}>◆ IGNIS — PyBullet 3D Physics</div>
          {ui.pybullet?(
            <table style={{width:"100%",borderCollapse:"collapse",fontSize:9}}><tbody>
              {[
                ["Φ autonomy",     <span style={{color:phiColor(ui.pybullet.phi??0)}}>{((ui.pybullet.phi??0)*100).toFixed(1)}%</span>],
                ["Discovery rate", <span style={{color:"#9933cc"}}>{((ui.pybullet.discovery_rate??0)*100).toFixed(0)}%</span>],
                ["Interventions",  ui.pybullet.interventions],
                ["Nodes (vars)",   ui.pybullet.node_count],
                ["Edges",          ui.pybullet.edge_count],
                ["h(W) DAG",       <span style={{color:hWColor(ui.pybullet.h_W??0)}}>{(ui.pybullet.h_W??0).toFixed(4)}</span>],
                ["CG (d|G|/dt)",   <span style={{color:(ui.pybullet.compression_gain??0)>0?"#00ff99":"#ff4433"}}>{(ui.pybullet.compression_gain??0).toFixed(4)}</span>],
              ].map(([l,v],k)=><tr key={k}><td style={{color:"#553377",paddingRight:10,paddingBottom:3}}>{l}</td><td style={{color:"#cc88ff",textAlign:"right"}}>{v}</td></tr>)}
            </tbody></table>
          ):<div style={{color:"#441166",fontSize:9}}>Connecting to Ignis…</div>}
          <div style={{...sep,fontSize:8,color:"#331155",lineHeight:1.8}}>
            <div style={{color:"#663399"}}>ENVIRONMENT:</div>
            <div>3 объекта → 18 vars (objN_x/y/z/vx/vy/vz)</div>
            <div>do(objN_vx=v) → set_velocity() → step(×10)</div>
            <div style={{color:"#663399",marginTop:3}}>GT CAUSAL STRUCTURE:</div>
            <div>vx→x, vy→y, vz→z (интеграция)</div>
            <div>obj_x→obj_vx (столкновения)</div>
            <div style={{color:"#663399",marginTop:3}}>BACKEND:</div>
            <div>PyBullet → Fallback если не установлен</div>
            <div>pip install pybullet</div>
          </div>
        </div>
      )}

      {/* Agent panels — 2 слева (Nova+Aether), 2 справа (Lyra+Ignis) */}
      <div style={{position:"absolute",top:118,left:14,display:"flex",flexDirection:"column",gap:6}}>
        {ui.agents.slice(0,2).map((a,i)=><AgentCard key={i} a={a} i={i} isDemonTarget={ui.demon?.last_target===i&&ui.demon?.mode!=="probe"} sep={sep}/>)}
      </div>
      <div style={{position:"absolute",top:118,right:14,display:"flex",flexDirection:"column",gap:6}}>
        {ui.agents.slice(2,4).map((a,orig_i)=>{const i=orig_i+2;return<AgentCard key={i} a={a} i={i} isDemonTarget={ui.demon?.last_target===i&&ui.demon?.mode!=="probe"} sep={sep} isPybullet={i===3}/>;}).reverse()}
        {/* Legend */}
        <div style={{background:"rgba(0,8,20,0.93)",border:"1px solid #081e30",padding:"8px 12px",borderRadius:2,fontSize:9,maxWidth:185}}>
          <div style={{color:"#114466",marginBottom:6,fontSize:9}}>ARCHITECTURE v5</div>
          {AGENT_NAMES.map((n,i)=><div key={i} style={{color:"#335566",marginBottom:2}}><span style={{color:AGENT_COLORS_CSS[i]}}>●</span> {n} <span style={{color:"#1a3344"}}>({AGENT_ENVS[i]})</span></div>)}
          <div style={{color:"#335566",marginBottom:2}}><span style={{color:modeColor(demonMode)}}>◆</span> Demon [{demonMode}]</div>
          {ui.byzantine&&<div style={{...sep}}>
            <div style={{color:"#003355",marginBottom:2}}>🗳 BYZ R{ui.byzantine.round} dev={devColor(ui.byzantine.mean_deviance??0)&&(ui.byzantine.mean_deviance??0).toFixed(3)}</div>
          </div>}
          {ui.motif?.last_donor!=null&&<div style={{color:"#224466",fontSize:8}}>🧬 donor={AGENT_NAMES[ui.motif.last_donor]}</div>}
          <div style={{...sep,color:"#551188",lineHeight:1.8}}>
            <div>◎ purple ring = PyBullet orbits</div>
            <div>▲ tetrahedron = Ignis</div>
            <div>■ cubes = physics objects</div>
          </div>
        </div>
      </div>

      {/* Event log */}
      <div style={{position:"absolute",bottom:14,left:14,right:14,background:"rgba(0,8,20,0.93)",border:"1px solid #081e30",padding:"8px 14px",borderRadius:2,maxHeight:110,overflow:"hidden"}}>
        <div style={{color:"#113344",fontSize:9,letterSpacing:"0.1em",marginBottom:4}}>
          CAUSAL EVENT STREAM {connected?"— PyBullet·Byzantine·RAG":"— OFFLINE"}
        </div>
        {ui.events.map((ev,i)=>(
          <div key={i} style={{color:ev.color??"#335566",fontSize:10,marginBottom:2,opacity:Math.max(0.15,1-i*0.1),fontWeight:ev.type==="value"||ev.type==="tom"?"bold":"normal"}}>
            [{String(ev.tick??0).padStart(4,"0")}] › {ev.text}
          </div>
        ))}
      </div>
    </div>
  );
}

// ─── AgentCard компонент ──────────────────────────────────────────────────────
function AgentCard({ a, i, isDemonTarget, sep, isPybullet=false }) {
  const col   = AGENT_COLORS_CSS[i];
  const phiC  = phiColor(a.phi??0);
  const hWc   = hWColor(a.hW??0);
  const cgC   = (a.compressionGain??0)>0?"#00ff99":(a.compressionGain??0)<-0.05?"#ff4433":"#aaccdd";
  const blkR  = a.valueLayer?.block_rate??0;
  const blkC  = blkColor(blkR);
  const hasBlk= !!a.lastBlockedReason;
  const bdrCol= isDemonTarget?"#660000":hasBlk?"#882200":col+"33";
  const lBdr  = isDemonTarget?"#ff0000":hasBlk?"#ff4422":col;

  return(
    <div style={{background:"rgba(0,8,20,0.93)",border:`1px solid ${bdrCol}`,borderLeft:`3px solid ${lBdr}`,padding:"8px 11px",minWidth:215,borderRadius:2,transition:"border-color 0.3s",boxShadow:isPybullet?`0 0 18px #6622aa22`:undefined}}>
      <div style={{color:isDemonTarget?"#ff4422":hasBlk?"#ff6644":col,fontSize:10,fontWeight:"bold",marginBottom:4,letterSpacing:"0.08em"}}>
        {isDemonTarget?"⚔":isPybullet?"◆":hasBlk?"🛡":"◈"} {a.name}
        <span style={{color:"#1a3344",marginLeft:6,fontSize:8}}>{a.envType}</span>
        {isPybullet&&<span style={{color:"#441166",marginLeft:4,fontSize:8}}>3D</span>}
      </div>
      <table style={{borderCollapse:"collapse",width:"100%",fontSize:9}}><tbody>
        {[
          ["Φ",        <span style={{color:phiC}}>{((a.phi??0)*100).toFixed(1)}%</span>],
          ["CG",       <span style={{color:cgC}}>{(a.compressionGain??0)>=0?"+":""}{(a.compressionGain??0).toFixed(4)}</span>],
          ["α",        `${Math.round((a.alphaMean??0)*100)}%`],
          ["h(W)",     <span style={{color:hWc}}>{(a.hW??0).toFixed(4)}</span>],
          ["do/blk",   `${a.totalInterventions??0}/${a.totalBlocked??0}`],
          ["DR",       `${((a.discoveryRate??0)*100).toFixed(0)}%`],
          [isPybullet?"nodes":"edges", isPybullet?(a.nodeCount??0):(a.edgeCount??0)],
        ].map(([l,v],k)=><tr key={k}><td style={{color:"#335566",paddingRight:7,paddingBottom:1}}>{l}</td><td style={{color:"#aad4ee",textAlign:"right"}}>{v}</td></tr>)}
      </tbody></table>

      {a.valueLayer&&<div style={{...sep,fontSize:8}}>
        <div style={{display:"flex",justifyContent:"space-between"}}>
          <span style={{color:"#553322"}}>VL {a.valueLayer.vl_phase}</span>
          <span style={{color:blkC}}>blk:{(blkR*100).toFixed(1)}%</span>
        </div>
        {hasBlk&&<div style={{color:"#ff6644",fontSize:8}}>⚠ {a.lastBlockedReason}</div>}
      </div>}

      {a.notears&&<div style={{...sep,fontSize:8}}>
        <div style={{display:"flex",justifyContent:"space-between"}}>
          <span style={{color:"#336644"}}>NT {a.notears.steps}s h={a.notears.h_W?.toFixed(3)}</span>
          <span style={{color:a.notears.loss<0.01?"#00ff99":"#aa8800"}}>L={a.notears.loss?.toFixed(5)}</span>
        </div>
      </div>}

      {/* Progress bars */}
      {[
        {w:`${Math.round((a.alphaMean??0)*100)}%`, f:"#002266",t:col},
        {w:`${Math.round((a.phi??0)*100)}%`, f:"#221100",t:phiC},
        {w:`${Math.max(0,100-Math.min((a.hW??0)*20,100))}%`, f:"#001800",t:hWc},
        {w:`${Math.min(blkR*100,100)}%`, f:"#110000",t:blkC},
      ].map((b,k)=>(
        <div key={k} style={{marginTop:k===0?4:2,height:k===0?3:2,background:"#061422",borderRadius:2,overflow:"hidden"}}>
          <div style={{width:b.w,height:"100%",background:`linear-gradient(90deg,${b.f},${b.t})`,transition:"width 0.8s"}}/>
        </div>
      ))}
      <div style={{color:"#1a3344",fontSize:7,marginTop:1,display:"flex",justifyContent:"space-between"}}>
        <span>α·Φ·h·vl</span><span style={{color:"#1a4433"}}>dr:{((a.discoveryRate??0)*100).toFixed(0)}%</span>
      </div>
    </div>
  );
}