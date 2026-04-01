import { useState, useEffect, useRef } from "react";
import * as THREE from "three";
import { useRKKStream } from "../../hooks/useRKKStream";

const WORLD            = 14;
const AGENT_COLORS_HEX = [0x00ff99, 0x0099ff, 0xff9900];
const AGENT_COLORS_CSS = ["#00ff99", "#0099ff", "#ff9900"];
const AGENT_NAMES      = ["Nova", "Aether", "Lyra"];
const DEMON_COLOR      = 0xff2244;
const PHASE_NAMES      = ["","Causal Crib","Robotic Explorer","Social Sandbox","Value Lock","Open Reality"];
const GRAPH_LINES      = 14;
const ATTACK_MODES     = { probe:"#335566", targeted:"#ff8844", siege:"#ff2244", anti_byz:"#aa00ff" };

const hWColor  = h => h<0.01?"#00ff99":h<0.5?"#aacc00":h<2?"#ffaa00":"#ff4422";
const phiColor = p => p>0.6?"#00ff99":p>0.3?"#aacc00":"#ff8844";
const blkColor = r => r>0.3?"#ff4422":r>0.1?"#ffaa00":"#335544";
const devColor = d => d<0.05?"#00ff99":d<0.15?"#aacc00":"#ff8844";
const modeColor= m => ATTACK_MODES[m] ?? "#335566";

function normalizeFrame(raw) {
  const agents = (raw.agents ?? []).map(a => ({
    id:a.id??0, name:a.name??"?", envType:a.env_type??"—", activation:a.activation??"relu",
    graphMdl:a.graph_mdl??0, compressionGain:a.compression_gain??0,
    alphaMean:a.alpha_mean??0.05, phi:a.phi??0.1,
    nodeCount:a.node_count??0, edgeCount:a.edge_count??0,
    totalInterventions:a.total_interventions??0, totalBlocked:a.total_blocked??0,
    lastDo:a.last_do??"—", lastBlockedReason:a.last_blocked_reason??"",
    discoveryRate:a.discovery_rate??0, peakDiscovery:a.peak_discovery_rate??0,
    hW:a.h_W??0, notears:a.notears??null, temporal:a.temporal??null,
    system1:a.system1??null, valueLayer:a.value_layer??null, edges:a.edges??[],
  }));
  return {
    tick:raw.tick??0, phase:raw.phase??1, entropy:raw.entropy??100,
    agents, demon:raw.demon??{energy:1,cooldown:0,last_action_complexity:0,mode:"probe",success_rate:0},
    tomLinks:raw.tom_links??[], events:raw.events??[],
    valueLayer:raw.value_layer??null,
    byzantine:raw.byzantine??null,
    motif:raw.motif??null,
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
  const [backend,     setBackend]     = useState(false);
  const [activePanel, setActivePanel] = useState(null); // "seeds" | "rag" | "demon"
  const [seedText,    setSeedText]    = useState('[\n  {"from_": "Temp", "to": "Pressure", "weight": 0.8}\n]');
  const [seedAgent,   setSeedAgent]   = useState(0);
  const [seedStatus,  setSeedStatus]  = useState("");
  const [ragStatus,   setRagStatus]   = useState(null);
  const [ragLoading,  setRagLoading]  = useState(false);
  const [demonStats,  setDemonStats]  = useState(null);

  const setSpeed = s => { setSpeedLocal(s); if (connected) wsSetSpeed(s); };
  useEffect(() => { setUI(normalizeFrame(wsFrame)); }, [wsFrame]);
  useEffect(() => { if (connected) setBackend(true); }, [connected]);

  // Периодически обновляем stats панелей
  useEffect(() => {
    if (!connected) return;
    const interval = setInterval(async () => {
      try {
        const [ragRes, demonRes] = await Promise.all([
          fetch("http://localhost:8000/rag/status"),
          fetch("http://localhost:8000/demon/stats"),
        ]);
        setRagStatus(await ragRes.json());
        setDemonStats(await demonRes.json());
      } catch {}
    }, 3000);
    return () => clearInterval(interval);
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

  const ragGenerate = async (agentId) => {
    setRagLoading(true);
    setSeedStatus("⏳ RAG generating…");
    try {
      const res = await fetch("http://localhost:8000/rag/generate", {
        method:"POST", headers:{"Content-Type":"application/json"},
        body:JSON.stringify({agent_id:agentId, max_hypotheses:6}),
      });
      const d = await res.json();
      setSeedStatus(`✓ RAG: ${d.injected} edges (${d.source})`);
    } catch(e) { setSeedStatus(`✗ RAG: ${e.message}`); }
    finally { setRagLoading(false); }
  };

  const ragAutoSeed = async () => {
    setRagLoading(true);
    setSeedStatus("⏳ Auto-seeding all agents…");
    try {
      const res = await fetch("http://localhost:8000/rag/auto-seed-all", { method:"POST" });
      const d = await res.json();
      const total = d.results?.reduce((s,r)=>s+r.injected,0)??0;
      setSeedStatus(`✓ Auto-seed: ${total} edges across all agents`);
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
    scene.fog = new THREE.FogExp2(0x010810, 0.022);
    const camera = new THREE.PerspectiveCamera(55, mount.clientWidth/mount.clientHeight, 0.1, 200);
    camera.position.set(0,15,28); camera.lookAt(0,1,0);
    scene.add(new THREE.AmbientLight(0x112233, 1.8));
    const sun = new THREE.DirectionalLight(0x334488, 1.2); sun.position.set(5,10,5); scene.add(sun);
    scene.add(new THREE.GridHelper(WORLD*2+4, 30, 0x0a1e38, 0x050f1e));

    const agentGroups = AGENT_COLORS_HEX.map((col,i) => {
      const g = new THREE.Group();
      const body  = new THREE.Mesh(new THREE.SphereGeometry(0.6,22,22), new THREE.MeshPhongMaterial({color:col,emissive:col,emissiveIntensity:0.35,transparent:true,opacity:0.92}));
      const ring  = new THREE.Mesh(new THREE.TorusGeometry(0.95,0.05,8,44), new THREE.MeshBasicMaterial({color:col,transparent:true,opacity:0.4}));
      const ringS = new THREE.Mesh(new THREE.TorusGeometry(1.4,0.025,6,40), new THREE.MeshBasicMaterial({color:col,transparent:true,opacity:0.15}));
      const byz   = new THREE.Mesh(new THREE.TorusGeometry(1.7,0.02,5,36), new THREE.MeshBasicMaterial({color:0x004488,transparent:true,opacity:0}));
      // RAG seed ring (золотой, появляется при инъекции)
      const ragRing = new THREE.Mesh(new THREE.TorusGeometry(1.9,0.015,4,32), new THREE.MeshBasicMaterial({color:0xaa8800,transparent:true,opacity:0}));
      const shield= new THREE.Mesh(new THREE.TorusGeometry(1.1,0.04,6,40), new THREE.MeshBasicMaterial({color:0xff4422,transparent:true,opacity:0}));
      shield.rotation.x=Math.PI/3; ring.rotation.x=Math.PI/2;
      g.add(body,ring,ringS,byz,ragRing,shield);
      g.add(new THREE.PointLight(col,0.7,5));
      g.position.set((i-1)*6+(Math.random()-0.5),1,(Math.random()-0.5)*3);
      g.userData={body,ring,ringS,byz,ragRing,shield,vel:new THREE.Vector3((Math.random()-.5)*0.04,0,(Math.random()-.5)*0.04),ragPulse:0};
      scene.add(g); return g;
    });

    const demon = new THREE.Mesh(new THREE.OctahedronGeometry(0.8,0), new THREE.MeshPhongMaterial({color:DEMON_COLOR,emissive:0xff0022,emissiveIntensity:0.7,transparent:true,opacity:0.9}));
    // Siege indicator — второй октаэдр когда demon в siege mode
    const demonSiege = new THREE.Mesh(new THREE.OctahedronGeometry(1.1,0), new THREE.MeshBasicMaterial({color:0xff4422,transparent:true,opacity:0,wireframe:true}));
    demon.add(new THREE.PointLight(DEMON_COLOR,0.9,6));
    demon.position.set(WORLD-2,1.2,WORLD-2);
    demon.userData={vel:new THREE.Vector3(-0.035,0,-0.025)};
    demon.add(demonSiege);
    scene.add(demon);

    const orbitNodes = agentGroups.map((_,ai)=>Array.from({length:GRAPH_LINES},()=>{
      const m=new THREE.Mesh(new THREE.SphereGeometry(0.1,6,6), new THREE.MeshBasicMaterial({color:AGENT_COLORS_HEX[ai],transparent:true,opacity:0}));
      scene.add(m); return m;
    }));
    const causalLines = agentGroups.map((_,ai)=>Array.from({length:GRAPH_LINES},()=>{
      const geom=new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(),new THREE.Vector3()]);
      const l=new THREE.Line(geom,new THREE.LineBasicMaterial({color:AGENT_COLORS_HEX[ai],transparent:true,opacity:0}));
      scene.add(l); return l;
    }));
    const byzLines=[], tomLines=[];
    for(let a=0;a<3;a++) for(let b=a+1;b<3;b++){
      [byzLines,tomLines].forEach((arr,idx)=>{
        const geom=new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(),new THREE.Vector3()]);
        const l=new THREE.Line(geom,new THREE.LineBasicMaterial({color:idx?0x003366:0x002255,transparent:true,opacity:0}));
        l.userData={a,b}; scene.add(l); arr.push(l);
      });
    }

    const pPos=new Float32Array(400*3);
    for(let i=0;i<400;i++){pPos[i*3]=(Math.random()-.5)*44;pPos[i*3+1]=Math.random()*14;pPos[i*3+2]=(Math.random()-.5)*44;}
    const pGeom=new THREE.BufferGeometry();
    pGeom.setAttribute("position",new THREE.BufferAttribute(pPos,3));
    scene.add(new THREE.Points(pGeom,new THREE.PointsMaterial({color:0x003399,size:0.07,transparent:true,opacity:0.35})));

    let frame=0,camAngle=0,byzPulse=0;

    function loop(){
      rafRef.current=requestAnimationFrame(loop);
      frame++;byzPulse++;
      const ds=normalizeFrame(wsFrameRef.current);
      camAngle+=0.0012;
      camera.position.x=Math.sin(camAngle)*28;
      camera.position.z=Math.cos(camAngle)*28;
      camera.lookAt(0,2,0);

      const byzActive=ds.byzantine&&ds.byzantine.round>0;
      const byzOp=byzActive?Math.max(0,Math.sin(byzPulse*0.08)*0.25):0;
      const demonMode=ds.demon?.mode??"probe";
      const isSiege=demonMode==="siege";
      demonSiege.material.opacity=isSiege?0.2+Math.sin(frame*0.12)*0.1:0;

      agentGroups.forEach((g,i)=>{
        const snap=ds.agents[i]; if(!snap) return;
        const vel=g.userData.vel;
        g.position.add(vel);
        if(g.position.x>WORLD||g.position.x<-WORLD)vel.x*=-1;
        if(g.position.z>WORLD||g.position.z<-WORLD)vel.z*=-1;
        g.position.y=1+Math.sin(frame*0.05+i*2.1)*0.15;
        if(frame%(120+i*30)===0)g.userData.vel=new THREE.Vector3((Math.random()-.5)*0.05,0,(Math.random()-.5)*0.05);

        g.userData.ring.rotation.z+=0.015+snap.alphaMean*0.035;
        g.userData.ring.rotation.y+=0.008;
        g.userData.ringS.rotation.x+=0.002+snap.phi*0.008;
        g.userData.ringS.rotation.z+=0.001;
        g.userData.byz.rotation.y+=0.003;
        g.userData.byz.material.opacity=byzOp;
        // RAG ring: пульсирует несколько секунд после инъекции
        g.userData.ragPulse=Math.max(0,g.userData.ragPulse-1);
        g.userData.ragRing.rotation.z+=0.01;
        g.userData.ragRing.material.opacity=g.userData.ragPulse>0?0.1+Math.sin(frame*0.2+i)*0.06:0;

        const blkRate=snap.valueLayer?.block_rate??0;
        g.userData.shield.material.opacity=blkRate>0.1?0.15+Math.sin(frame*0.15+i)*0.1:0;

        const dagGlow=Math.max(0,0.15-Math.min(snap.hW*0.05,0.15));
        g.userData.body.material.emissiveIntensity=0.15+Math.max(0,snap.compressionGain)*0.1+dagGlow+Math.sin(frame*0.07+i)*0.06;

        const visCount=Math.min(snap.edgeCount+2,GRAPH_LINES);
        orbitNodes[i].forEach((node,k)=>{
          if(k<visCount){const angle=(k/visCount)*Math.PI*2+frame*0.016,r=1.6+(k%2)*0.5;node.position.set(g.position.x+Math.cos(angle)*r,g.position.y+Math.sin(frame*0.04+k*0.8)*0.35+0.2,g.position.z+Math.sin(angle)*r);node.material.opacity=0.4+snap.alphaMean*0.5;}
          else node.material.opacity=0;
        });
        causalLines[i].forEach((line,k)=>{
          if(k<visCount-1){const na=orbitNodes[i][k].position,nb=orbitNodes[i][(k+1)%visCount].position,pa=line.geometry.attributes.position;pa.setXYZ(0,na.x,na.y,na.z);pa.setXYZ(1,nb.x,nb.y,nb.z);pa.needsUpdate=true;line.material.color.set(snap.edges[k]?.weight<0?0xff4422:AGENT_COLORS_HEX[i]);line.material.opacity=0.3+snap.alphaMean*0.45;}
          else line.material.opacity=0;
        });
      });

      // Trigger RAG ring pulse when seeds are injected (check via events)
      if(ds.events?.[0]?.text?.includes("Seeds")&&frame%2===0){
        const agentIdx=ds.events[0].color==="#886600"?-1:AGENT_COLORS_CSS.indexOf(ds.events[0].color);
        if(agentIdx>=0&&agentIdx<3)agentGroups[agentIdx].userData.ragPulse=120;
      }

      // Demon: chase siege target
      const demonTarget=isSiege?agentGroups[ds.demon?.last_target??0]:agentGroups[ds.tick%3];
      const chase=demonTarget.position.clone().sub(demon.position).normalize().multiplyScalar(isSiege?0.045:0.032);
      demon.userData.vel.lerp(chase,isSiege?0.08:0.05);
      demon.position.add(demon.userData.vel);
      if(demon.position.x>WORLD||demon.position.x<-WORLD)demon.userData.vel.x*=-1;
      if(demon.position.z>WORLD||demon.position.z<-WORLD)demon.userData.vel.z*=-1;
      demon.position.y=1.2+Math.sin(frame*0.08)*0.3;
      demon.rotation.y+=isSiege?0.08:0.045; demon.rotation.x+=0.028;

      byzLines.forEach(line=>{const pa=agentGroups[line.userData.a].position,pb=agentGroups[line.userData.b].position,p=line.geometry.attributes.position;p.setXYZ(0,pa.x,pa.y,pa.z);p.setXYZ(1,pb.x,pb.y,pb.z);p.needsUpdate=true;line.material.opacity=byzActive?byzOp*0.6:0;});
      tomLines.forEach(line=>{const link=ds.tomLinks.find(l=>l.a===line.userData.a&&l.b===line.userData.b),pa=agentGroups[line.userData.a].position,pb=agentGroups[line.userData.b].position,p=line.geometry.attributes.position;p.setXYZ(0,pa.x,pa.y,pa.z);p.setXYZ(1,pb.x,pb.y,pb.z);p.needsUpdate=true;line.material.opacity=link?link.strength*0.65:0;});
      renderer.render(scene,camera);
    }
    loop();

    const onResize=()=>{if(!mount)return;camera.aspect=mount.clientWidth/mount.clientHeight;camera.updateProjectionMatrix();renderer.setSize(mount.clientWidth,mount.clientHeight);};
    window.addEventListener("resize",onResize);
    return()=>{cancelAnimationFrame(rafRef.current);window.removeEventListener("resize",onResize);if(mount.contains(renderer.domElement))mount.removeChild(renderer.domElement);renderer.dispose();};
  },[]);

  const mono={fontFamily:"'Courier New',monospace"};
  const sep={borderTop:"1px solid #081e30",marginTop:5,paddingTop:5};
  const totalBlocked=ui.valueLayer?.total_blocked_all??0;
  const demonMode=ui.demon?.mode??"probe";

  return(
    <div style={{position:"relative",width:"100%",height:"100vh",background:"#010810",overflow:"hidden",...mono}}>
      <div ref={mountRef} style={{position:"absolute",inset:0}}/>

      {/* Status */}
      <div style={{position:"absolute",top:14,right:14,background:connected?"rgba(0,40,10,0.9)":"rgba(30,10,0,0.9)",border:`1px solid ${connected?"#00aa44":"#aa4400"}`,padding:"4px 10px",borderRadius:2,fontSize:9,color:connected?"#00ff88":"#ff8844"}}>
        {connected?"● PYTHON BACKEND":"○ OFFLINE"}{ui.multiprocess?" · MP":""}
      </div>

      {/* Header */}
      <div style={{position:"absolute",top:14,left:"50%",transform:"translateX(-50%)",background:"rgba(0,12,28,0.9)",border:"1px solid #0a2a44",padding:"8px 22px",textAlign:"center",borderRadius:2,boxShadow:"0 0 24px #00224455",whiteSpace:"nowrap"}}>
        <div style={{color:"#00ff99",fontSize:13,fontWeight:"bold",letterSpacing:"0.18em"}}>
          RKK v5 — RAG · BYZANTINE · VALUE LAYER
        </div>
        <div style={{color:"#115577",fontSize:10,marginTop:3,letterSpacing:"0.1em"}}>
          PHASE {ui.phase}/5 › {PHASE_NAMES[ui.phase]}
          &nbsp;│&nbsp;ENT:<span style={{color:ui.entropy<30?"#00ff99":"#aaccdd"}}>{ui.entropy}%</span>
          &nbsp;│&nbsp;TICK:{ui.tick}
          &nbsp;│&nbsp;<span style={{color:modeColor(demonMode)}}>◆{demonMode}</span>
          {ui.demon&&<span style={{color:"#443300",marginLeft:4}}>E:{(ui.demon.energy*100).toFixed(0)}% sr:{((ui.demon.success_rate??0)*100).toFixed(0)}%</span>}
          &nbsp;│&nbsp;<span style={{color:totalBlocked>0?"#ff8844":"#335544"}}>🛡{totalBlocked}</span>
          {ui.byzantine&&<span style={{color:"#004466",marginLeft:4}}>🗳R{ui.byzantine.round}</span>}
          {ui.motif?.last_donor!=null&&<span style={{color:"#224466",marginLeft:4}}>🧬{["N","A","L"][ui.motif.last_donor]}</span>}
          {ragStatus?.running&&<span style={{color:"#886600",marginLeft:4}}>⏳RAG</span>}
        </div>
      </div>

      {/* Controls */}
      <div style={{position:"absolute",top:82,left:"50%",transform:"translateX(-50%)",background:"rgba(0,8,20,0.85)",border:"1px solid #081e30",padding:"6px 14px",borderRadius:2,display:"flex",gap:6,alignItems:"center",fontSize:9}}>
        <span style={{color:"#224455"}}>SPEED</span>
        {[1,2,4,8].map(s=><button key={s} onClick={()=>setSpeed(s)} style={{padding:"2px 7px",borderRadius:2,fontSize:9,cursor:"pointer",background:speed===s?"#001e38":"transparent",border:`1px solid ${speed===s?"#00aaff":"#081e30"}`,color:speed===s?"#00aaff":"#334455"}}>{s}×</button>)}
        <span style={{color:"#334455"}}>│</span>
        {PHASE_NAMES.slice(1).map((name,i)=>{const ph=i+1;return<div key={ph} style={{padding:"2px 6px",borderRadius:2,fontSize:9,background:ui.phase===ph?"#001e38":"transparent",border:`1px solid ${ui.phase===ph?"#00aaff":ui.phase>ph?"#003322":"#081e30"}`,color:ui.phase===ph?"#00aaff":ui.phase>ph?"#00aa55":"#223344"}}>{ui.phase>ph?"✓":ph} {name}</div>;})}
        <span style={{color:"#334455"}}>│</span>
        {/* Panel toggles */}
        {[["💉","seeds"],["🌐","rag"],["◆","demon"]].map(([icon,panel])=>(
          <button key={panel} onClick={()=>setActivePanel(v=>v===panel?null:panel)} style={{padding:"2px 7px",borderRadius:2,fontSize:9,cursor:"pointer",background:activePanel===panel?"#1a1000":"transparent",border:`1px solid ${activePanel===panel?"#886600":"#333"}`,color:activePanel===panel?"#ffaa00":"#445555"}}>{icon}</button>
        ))}
      </div>

      {/* Seeds panel */}
      {activePanel==="seeds"&&(
        <div style={{position:"absolute",top:120,left:"50%",transform:"translateX(-50%)",background:"rgba(8,6,0,0.95)",border:"1px solid #443300",padding:"12px 16px",borderRadius:2,width:420,zIndex:10}}>
          <div style={{color:"#886600",fontSize:10,marginBottom:6}}>💉 MANUAL SEED INJECTION</div>
          <div style={{display:"flex",gap:6,marginBottom:6}}>
            {AGENT_NAMES.map((name,i)=><button key={i} onClick={()=>setSeedAgent(i)} style={{padding:"2px 8px",borderRadius:2,fontSize:9,cursor:"pointer",background:seedAgent===i?"#221100":"transparent",border:`1px solid ${seedAgent===i?AGENT_COLORS_CSS[i]:"#332200"}`,color:seedAgent===i?AGENT_COLORS_CSS[i]:"#554400"}}>{name}</button>)}
          </div>
          <textarea value={seedText} onChange={e=>setSeedText(e.target.value)} style={{width:"100%",height:80,background:"#050300",border:"1px solid #332200",color:"#aa8800",fontSize:9,padding:6,borderRadius:2,fontFamily:"monospace",resize:"none",boxSizing:"border-box"}}/>
          <div style={{display:"flex",justifyContent:"space-between",alignItems:"center",marginTop:6}}>
            <button onClick={injectSeeds} style={{padding:"3px 12px",borderRadius:2,fontSize:9,cursor:"pointer",background:"#221100",border:"1px solid #886600",color:"#ffaa00"}}>INJECT</button>
            <span style={{color:seedStatus.startsWith("✓")?"#00ff99":"#ff4422",fontSize:9}}>{seedStatus}</span>
          </div>
        </div>
      )}

      {/* RAG panel */}
      {activePanel==="rag"&&(
        <div style={{position:"absolute",top:120,left:"50%",transform:"translateX(-50%)",background:"rgba(0,6,14,0.96)",border:"1px solid #004433",padding:"14px 18px",borderRadius:2,width:440,zIndex:10}}>
          <div style={{color:"#00aa66",fontSize:10,marginBottom:8,letterSpacing:"0.1em"}}>🌐 RAG SEED PIPELINE</div>
          <div style={{fontSize:9,color:"#225544",marginBottom:8,lineHeight:1.7}}>
            Wikipedia → causal extraction → inject α=0.05<br/>
            NOTEARS burns wrong edges. Correct ones grow α→0.9.
          </div>

          {/* Auto-seed all */}
          <button
            onClick={ragAutoSeed}
            disabled={ragLoading}
            style={{width:"100%",padding:"6px",borderRadius:2,fontSize:9,cursor:"pointer",background:ragLoading?"#001a0a":"#002211",border:"1px solid #006633",color:ragLoading?"#225544":"#00ff88",marginBottom:8}}
          >
            {ragLoading?"⏳ Fetching Wikipedia…":"🌐 AUTO-SEED ALL AGENTS (Wikipedia)"}
          </button>

          {/* Per-agent RAG */}
          <div style={{display:"flex",gap:6,marginBottom:8}}>
            {AGENT_NAMES.map((name,i)=>(
              <button key={i} onClick={()=>ragGenerate(i)} disabled={ragLoading} style={{flex:1,padding:"4px",borderRadius:2,fontSize:9,cursor:"pointer",background:"transparent",border:`1px solid ${AGENT_COLORS_CSS[i]}44`,color:AGENT_COLORS_CSS[i]}}>{name}</button>
            ))}
          </div>

          {/* RAG status */}
          {ragStatus?.results?.length>0&&(
            <div style={{...sep}}>
              <div style={{color:"#225544",fontSize:9,marginBottom:4}}>LAST RAG RESULTS:</div>
              {ragStatus.results.slice(-3).map((r,i)=>(
                <div key={i} style={{fontSize:9,color:"#336655",display:"flex",justifyContent:"space-between"}}>
                  <span>{r.agent} ({r.preset})</span>
                  <span>{r.injected} edges · {r.source}</span>
                </div>
              ))}
            </div>
          )}

          <div style={{marginTop:6,fontSize:9,color:seedStatus.startsWith("✓")?"#00ff99":"#ff4422"}}>{seedStatus}</div>

          {/* LLM config note */}
          <div style={{...sep,fontSize:8,color:"#1a3322"}}>
            For LLM: set LLM_URL in server.py (Ollama compatible)<br/>
            qwen2.5:3b recommended · Phase 11+ feature
          </div>
        </div>
      )}

      {/* Demon stats panel */}
      {activePanel==="demon"&&demonStats&&(
        <div style={{position:"absolute",top:120,left:"50%",transform:"translateX(-50%)",background:"rgba(14,0,0,0.96)",border:"1px solid #440000",padding:"14px 18px",borderRadius:2,width:380,zIndex:10}}>
          <div style={{color:"#ff4422",fontSize:10,marginBottom:8,letterSpacing:"0.1em"}}>◆ ADVERSARIAL DEMON v2</div>
          <table style={{width:"100%",borderCollapse:"collapse",fontSize:9}}><tbody>
            {[
              ["Mode",<span style={{color:modeColor(demonStats.mode)}}>{demonStats.mode?.toUpperCase()}</span>],
              ["Energy",`${(demonStats.energy*100).toFixed(0)}%`],
              ["Success rate",`${((demonStats.success_rate??0)*100).toFixed(1)}%`],
              ["Recent reward",<span style={{color:(demonStats.mean_recent_reward??0)>0?"#ff8844":"#335544"}}>{(demonStats.mean_recent_reward??0).toFixed(3)}</span>],
            ].map(([l,v],k)=>(
              <tr key={k}><td style={{color:"#553333",paddingRight:10,paddingBottom:3}}>{l}</td><td style={{color:"#cc6644",textAlign:"right"}}>{v}</td></tr>
            ))}
          </tbody></table>
          {demonStats.memory&&(
            <div style={{...sep}}>
              <div style={{color:"#441111",fontSize:8,marginBottom:4}}>AGENT MEMORY:</div>
              {demonStats.memory.map(m=>(
                <div key={m.agent} style={{fontSize:9,display:"flex",justifyContent:"space-between",marginBottom:2}}>
                  <span style={{color:AGENT_COLORS_CSS[m.agent]}}>{AGENT_NAMES[m.agent]}</span>
                  <span style={{color:"#553333"}}>✓{m.success} ✗{m.fail} ΔΦ={m.phi_drop?.toFixed(4)}</span>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Agent panels */}
      <div style={{position:"absolute",top:120,left:14,display:"flex",flexDirection:"column",gap:7}}>
        {ui.agents.map((a,i)=>{
          const col=AGENT_COLORS_CSS[i];
          const alpW=`${Math.round(a.alphaMean*100)}%`;
          const phiW=`${Math.round(a.phi*100)}%`;
          const cgC=a.compressionGain>0?"#00ff99":a.compressionGain<-0.05?"#ff4433":"#aaccdd";
          const hWc=hWColor(a.hW??0);
          const phiC=phiColor(a.phi??0);
          const blkR=a.valueLayer?.block_rate??0;
          const blkC=blkColor(blkR);
          const hasBlock=!!a.lastBlockedReason;
          const isDemonTarget=ui.demon?.last_target===i&&ui.demon?.mode!=="probe";
          return(
            <div key={i} style={{background:"rgba(0,8,20,0.93)",border:`1px solid ${isDemonTarget?"#660000":hasBlock?"#882200":col+"33"}`,borderLeft:`3px solid ${isDemonTarget?"#ff0000":hasBlock?"#ff4422":col}`,padding:"8px 12px",minWidth:222,borderRadius:2,boxShadow:`0 0 14px ${isDemonTarget?"#ff000018":hasBlock?"#ff442218":col+"15"}`,transition:"border-color 0.3s"}}>
              <div style={{color:isDemonTarget?"#ff4422":hasBlock?"#ff6644":col,fontSize:11,fontWeight:"bold",marginBottom:5,letterSpacing:"0.1em"}}>
                {isDemonTarget?"⚔":hasBlock?"🛡":"◈"} {a.name}
                <span style={{color:"#1a3344",marginLeft:8,fontSize:9}}>{a.envType}·{a.activation}</span>
              </div>
              <table style={{borderCollapse:"collapse",width:"100%",fontSize:10}}><tbody>
                {[
                  ["Φ",<span style={{color:phiC}}>{(a.phi*100).toFixed(1)}%</span>],
                  ["CG",<span style={{color:cgC}}>{a.compressionGain>=0?"+":""}{a.compressionGain.toFixed(4)}</span>],
                  ["α",alpW],["h(W)",<span style={{color:hWc}}>{(a.hW??0).toFixed(4)}</span>],
                  ["do/blk",`${a.totalInterventions}/${a.totalBlocked}`],
                  ["DR",`${((a.discoveryRate??0)*100).toFixed(0)}%`],
                ].map(([l,v],k)=><tr key={k}><td style={{color:"#335566",paddingRight:8,paddingBottom:2}}>{l}</td><td style={{color:"#aad4ee",textAlign:"right"}}>{v}</td></tr>)}
              </tbody></table>

              {a.valueLayer&&<div style={{...sep,fontSize:9}}>
                <div style={{display:"flex",justifyContent:"space-between"}}>
                  <span style={{color:"#553322"}}>VL {a.valueLayer.vl_phase} {a.valueLayer.vl_strictness?.toFixed(2)}</span>
                  <span style={{color:blkC}}>blk:{(blkR*100).toFixed(1)}%</span>
                </div>
                {hasBlock&&<div style={{color:"#ff6644",fontSize:8}}>⚠ {a.lastBlockedReason}</div>}
              </div>}

              {a.notears&&<div style={{...sep,fontSize:9}}>
                <div style={{display:"flex",justifyContent:"space-between"}}>
                  <span style={{color:"#336644"}}>NT {a.notears.steps}s</span>
                  <span style={{color:a.notears.loss<0.01?"#00ff99":"#aa8800"}}>L={a.notears.loss?.toFixed(5)}</span>
                </div>
                <div style={{display:"flex",justifyContent:"space-between"}}>
                  <span style={{color:"#224433"}}>h={a.notears.h_W?.toFixed(4)}</span>
                  <span style={{color:"#224433"}}>L_int={a.notears.l_int?.toFixed(4)}</span>
                </div>
              </div>}

              {(a.temporal||a.system1)&&<div style={{...sep,fontSize:9}}>
                {a.temporal&&<div style={{display:"flex",justifyContent:"space-between",marginBottom:2}}>
                  <span style={{color:"#224455"}}>SSM f:{a.temporal.fast_steps} s:{a.temporal.slow_steps}</span>
                  <span style={{color:phiC}}>Φ={a.temporal.phi?.toFixed(3)}</span>
                </div>}
                {a.system1&&<div style={{display:"flex",justifyContent:"space-between"}}>
                  <span style={{color:"#334455"}}>S1={a.system1.buffer_size}</span>
                  <span style={{color:a.system1.mean_loss<0.01?"#00ff99":"#667788"}}>L={a.system1.mean_loss?.toFixed(5)}</span>
                </div>}
              </div>}

              {[{w:alpW,f:"#002266",t:col},{w:phiW,f:"#221100",t:phiC},{w:`${Math.max(0,100-Math.min((a.hW??0)*20,100))}%`,f:"#001800",t:hWc},{w:`${Math.min(blkR*100,100)}%`,f:"#110000",t:blkC}].map((b,k)=>(
                <div key={k} style={{marginTop:k===0?5:2,height:k===0?3:2,background:"#061422",borderRadius:2,overflow:"hidden"}}>
                  <div style={{width:b.w,height:"100%",background:`linear-gradient(90deg,${b.f},${b.t})`,transition:"width 0.8s"}}/>
                </div>
              ))}
              <div style={{color:"#1a3344",fontSize:8,marginTop:2,display:"flex",justifyContent:"space-between"}}>
                <span>α·Φ·h·vl</span><span style={{color:"#1a4433"}}>dr:{((a.discoveryRate??0)*100).toFixed(0)}%</span>
              </div>
            </div>
          );
        })}
      </div>

      {/* Right panel */}
      <div style={{position:"absolute",top:120,right:14,background:"rgba(0,8,20,0.93)",border:"1px solid #081e30",padding:"10px 14px",borderRadius:2,fontSize:10,maxWidth:190}}>
        <div style={{color:"#114466",marginBottom:8,fontSize:9,letterSpacing:"0.1em"}}>ARCHITECTURE</div>
        {AGENT_NAMES.map((n,i)=><div key={i} style={{color:"#335566",marginBottom:2}}><span style={{color:AGENT_COLORS_CSS[i]}}>●</span> {n}</div>)}
        <div style={{color:"#335566",marginBottom:2}}><span style={{color:modeColor(demonMode)}}>◆</span> Demon [{demonMode}]</div>
        <div style={{color:"#884400",marginBottom:2}}>⟳ shield=VL ◯=slow SSM</div>
        <div style={{color:"#554400",marginBottom:2}}>◎ gold=RAG seeds pulse</div>
        <div style={{color:"#004466",marginBottom:2}}>— Byzantine links</div>

        {ui.byzantine&&<div style={{...sep}}>
          <div style={{color:"#003355",fontSize:9,marginBottom:3}}>🗳 BYZANTINE</div>
          <div style={{fontSize:9,display:"flex",justifyContent:"space-between"}}>
            <span style={{color:"#224455"}}>round</span><span style={{color:"#336677"}}>{ui.byzantine.round}</span>
          </div>
          <div style={{fontSize:9,display:"flex",justifyContent:"space-between"}}>
            <span style={{color:"#224455"}}>deviance</span>
            <span style={{color:devColor(ui.byzantine.mean_deviance??0)}}>{(ui.byzantine.mean_deviance??0).toFixed(4)}</span>
          </div>
        </div>}

        {ui.motif&&<div style={{...sep}}>
          <div style={{color:"#002244",fontSize:9,marginBottom:3}}>🧬 MOTIF EMA=0.15</div>
          {ui.motif.last_donor!=null&&<div style={{fontSize:9,display:"flex",justifyContent:"space-between"}}>
            <span style={{color:"#224455"}}>donor</span>
            <span style={{color:AGENT_COLORS_CSS[ui.motif.last_donor]}}>{AGENT_NAMES[ui.motif.last_donor]}</span>
          </div>}
        </div>}

        <div style={{...sep,color:"#113344",fontSize:9,lineHeight:1.9}}>
          <div style={{color:"#004422"}}>RAG PIPELINE:</div>
          <div>Wikipedia→regex→seeds</div>
          <div>NOTEARS anneals bad edges</div>
          <div style={{color:"#220044",marginTop:3}}>DEMON v2:</div>
          <div>PPO-lite · memory · siege</div>
          <div>anti-byz every {200} ticks</div>
        </div>
      </div>

      {/* Event log */}
      <div style={{position:"absolute",bottom:14,left:14,right:14,background:"rgba(0,8,20,0.93)",border:"1px solid #081e30",padding:"10px 14px",borderRadius:2,maxHeight:120,overflow:"hidden"}}>
        <div style={{color:"#113344",fontSize:9,letterSpacing:"0.15em",marginBottom:5}}>
          CAUSAL EVENT STREAM {connected?"— RAG·BYZANTINE·DEMON":"— OFFLINE"}
        </div>
        {ui.events.length===0&&<div style={{color:"#1a3344",fontSize:10}}>Awaiting events…</div>}
        {ui.events.map((ev,i)=>(
          <div key={i} style={{color:ev.color??"#335566",fontSize:10,marginBottom:2,opacity:Math.max(0.15,1-i*0.1),fontWeight:ev.type==="value"||ev.type==="tom"?"bold":"normal"}}>
            [{String(ev.tick??0).padStart(4,"0")}] › {ev.text}
          </div>
        ))}
      </div>
    </div>
  );
}