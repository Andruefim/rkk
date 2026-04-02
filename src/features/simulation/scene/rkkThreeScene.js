import * as THREE from "three";
import {
  WORLD,
  AGENT_COLORS_HEX,
  DEMON_COLOR,
  GRAPH_LINES,
  AGENT_POSITIONS,
} from "../constants.js";

const PB_SCALE = 0.42;

/**
 * @param {HTMLElement} mount
 * @param {{ frameRef: React.MutableRefObject<object>, normalizeFrame: (raw: object) => object }} opts
 */
export function runRKKScene(mount, { frameRef, normalizeFrame }) {
  if (!mount) return () => {};

  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setSize(mount.clientWidth, mount.clientHeight);
  mount.appendChild(renderer.domElement);

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x010810);
  scene.fog = new THREE.FogExp2(0x010810, 0.018);
  const camera = new THREE.PerspectiveCamera(
    52,
    mount.clientWidth / mount.clientHeight,
    0.1,
    220
  );
  camera.position.set(0, 16, 32);
  camera.lookAt(0, 1, 0);
  scene.add(new THREE.AmbientLight(0x112233, 1.8));
  const sun = new THREE.DirectionalLight(0x334488, 1.2);
  sun.position.set(5, 10, 5);
  scene.add(sun);
  scene.add(new THREE.GridHelper(WORLD * 2 + 4, 32, 0x0a1e38, 0x050f1e));

  const nPillars = AGENT_POSITIONS.length;
  const agentGroups = AGENT_COLORS_HEX.map((col, i) => {
    const g = new THREE.Group();
    const isPybullet = i === 3;
    const bodyGeo = isPybullet
      ? new THREE.TetrahedronGeometry(0.65, 0)
      : new THREE.SphereGeometry(0.6, 22, 22);
    const body = new THREE.Mesh(
      bodyGeo,
      new THREE.MeshPhongMaterial({
        color: col,
        emissive: col,
        emissiveIntensity: 0.4,
        transparent: true,
        opacity: 0.92,
      })
    );
    const ring = new THREE.Mesh(
      new THREE.TorusGeometry(0.95, 0.05, 8, 44),
      new THREE.MeshBasicMaterial({ color: col, transparent: true, opacity: 0.4 })
    );
    const ringS = new THREE.Mesh(
      new THREE.TorusGeometry(1.4, 0.025, 6, 40),
      new THREE.MeshBasicMaterial({ color: col, transparent: true, opacity: 0.15 })
    );
    const byz = new THREE.Mesh(
      new THREE.TorusGeometry(1.7, 0.02, 5, 36),
      new THREE.MeshBasicMaterial({ color: 0x004488, transparent: true, opacity: 0 })
    );
    const ragRing = new THREE.Mesh(
      new THREE.TorusGeometry(1.9, 0.015, 4, 32),
      new THREE.MeshBasicMaterial({ color: 0xaa8800, transparent: true, opacity: 0 })
    );
    const shield = new THREE.Mesh(
      new THREE.TorusGeometry(1.1, 0.04, 6, 40),
      new THREE.MeshBasicMaterial({ color: 0xff4422, transparent: true, opacity: 0 })
    );
    const physRing = isPybullet
      ? new THREE.Mesh(
          new THREE.TorusGeometry(2.2, 0.01, 4, 28),
          new THREE.MeshBasicMaterial({ color: 0xcc44ff, transparent: true, opacity: 0.1 })
        )
      : null;

    shield.rotation.x = Math.PI / 3;
    ring.rotation.x = Math.PI / 2;
    g.add(body, ring, ringS, byz, ragRing, shield);
    if (physRing) g.add(physRing);
    g.add(new THREE.PointLight(col, 0.7, 5));
    g.position.set(...AGENT_POSITIONS[i]);
    g.userData = {
      body,
      ring,
      ringS,
      byz,
      ragRing,
      shield,
      physRing,
      ragPulse: 0,
      isPybullet,
    };
    scene.add(g);
    return g;
  });

  const physObjects = [0, 1, 2].map(() => {
    const geo = new THREE.BoxGeometry(0.2, 0.2, 0.2);
    const mat = new THREE.MeshPhongMaterial({
      color: 0xcc44ff,
      emissive: 0xaa22cc,
      emissiveIntensity: 0.3,
      transparent: true,
      opacity: 0.7,
    });
    const m = new THREE.Mesh(geo, mat);
    scene.add(m);
    return m;
  });

  const demon = new THREE.Mesh(
    new THREE.OctahedronGeometry(0.8, 0),
    new THREE.MeshPhongMaterial({
      color: DEMON_COLOR,
      emissive: 0xff0022,
      emissiveIntensity: 0.7,
      transparent: true,
      opacity: 0.9,
    })
  );
  const demonSiege = new THREE.Mesh(
    new THREE.OctahedronGeometry(1.1, 0),
    new THREE.MeshBasicMaterial({ color: 0xff4422, transparent: true, opacity: 0, wireframe: true })
  );
  demon.add(new THREE.PointLight(DEMON_COLOR, 0.9, 6));
  demon.add(demonSiege);
  demon.position.set(WORLD - 2, 1.2, WORLD - 2);
  demon.userData = { vel: new THREE.Vector3(-0.02, 0, -0.02) };
  scene.add(demon);

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
  const causalLines = agentGroups.map((_, ai) =>
    Array.from({ length: GRAPH_LINES }, () => {
      const geom = new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(),
        new THREE.Vector3(),
      ]);
      const l = new THREE.Line(
        geom,
        new THREE.LineBasicMaterial({ color: AGENT_COLORS_HEX[ai], transparent: true, opacity: 0 })
      );
      scene.add(l);
      return l;
    })
  );

  const byzLines = [];
  const tomLines = [];
  for (let a = 0; a < nPillars; a++)
    for (let b = a + 1; b < nPillars; b++) {
      [byzLines, tomLines].forEach((arr, idx) => {
        const geom = new THREE.BufferGeometry().setFromPoints([
          new THREE.Vector3(),
          new THREE.Vector3(),
        ]);
        const l = new THREE.Line(
          geom,
          new THREE.LineBasicMaterial({
            color: idx ? 0x003366 : 0x002255,
            transparent: true,
            opacity: 0,
          })
        );
        l.userData = { a, b };
        scene.add(l);
        arr.push(l);
      });
    }

  const pPos = new Float32Array(500 * 3);
  for (let i = 0; i < 500; i++) {
    pPos[i * 3] = (Math.random() - 0.5) * 50;
    pPos[i * 3 + 1] = Math.random() * 16;
    pPos[i * 3 + 2] = (Math.random() - 0.5) * 50;
  }
  const pGeom = new THREE.BufferGeometry();
  pGeom.setAttribute("position", new THREE.BufferAttribute(pPos, 3));
  scene.add(
    new THREE.Points(pGeom, new THREE.PointsMaterial({ color: 0x003399, size: 0.07, transparent: true, opacity: 0.3 }))
  );

  let rafId = 0;
  let localFrame = 0;
  let camAngle = 0;
  let byzPulse = 0;
  const tmpCentroid = new THREE.Vector3();
  const tmpTarget = new THREE.Vector3();
  const demonDir = new THREE.Vector3();

  function loop() {
    rafId = requestAnimationFrame(loop);
    localFrame++;
    byzPulse++;
    const ds = normalizeFrame(frameRef.current);
    const simTick = ds.tick;

    tmpCentroid.set(0, 0, 0);
    let na = 0;
    for (let i = 0; i < Math.min(ds.nAgents, agentGroups.length); i++) {
      tmpCentroid.add(agentGroups[i].position);
      na++;
    }
    if (na > 0) tmpCentroid.multiplyScalar(1 / na);
    else tmpCentroid.set(0, 1, 0);

    camAngle += 0.0009;
    camera.position.x = tmpCentroid.x + Math.sin(camAngle) * 30;
    camera.position.z = tmpCentroid.z + Math.cos(camAngle) * 30;
    camera.position.y = 14 + Math.sin(camAngle * 0.7) * 2;
    camera.lookAt(tmpCentroid.x, 2, tmpCentroid.z);

    const byzActive = ds.byzantine && ds.byzantine.round > 0;
    const byzOp = byzActive ? Math.max(0, Math.sin(byzPulse * 0.08) * 0.2) : 0;
    const demonMode = ds.demon?.mode ?? "probe";
    const isSiege = demonMode === "siege";
    demonSiege.material.opacity = isSiege ? 0.2 + Math.sin(localFrame * 0.12) * 0.1 : 0;

    const ignisPos = agentGroups[3].position;
    const pbStats = ds.pybullet;
    const objs = pbStats?.objects;
    const hasRealPb = objs && objs.length > 0;

    physObjects.forEach((obj, k) => {
      const o = objs?.[k];
      if (hasRealPb && o && typeof o.x === "number") {
        obj.position.set(
          ignisPos.x + o.x * PB_SCALE,
          ignisPos.y + 0.25 + o.y * PB_SCALE,
          ignisPos.z + o.z * PB_SCALE
        );
      } else {
        const angle = (k / 3) * Math.PI * 2 + localFrame * 0.015 * (1 + k * 0.2);
        const r = 1.85 + Math.sin(localFrame * 0.02 + k) * 0.2;
        const vy = pbStats?.phi ?? 0;
        obj.position.set(
          ignisPos.x + Math.cos(angle) * r,
          ignisPos.y + 0.35 + Math.sin(localFrame * 0.04 + k * 1.1) * 0.35 * (1 + vy),
          ignisPos.z + Math.sin(angle) * r
        );
      }
      obj.rotation.x += 0.025 * (k + 1);
      obj.rotation.y += 0.02 * (k + 1);
      obj.material.opacity = 0.45 + (pbStats?.phi ?? 0) * 0.45;
    });

    agentGroups.forEach((g, i) => {
      const snap = ds.agents[i];
      if (!snap) return;

      const ax = AGENT_POSITIONS[i][0];
      const ay = AGENT_POSITIONS[i][1];
      const az = AGENT_POSITIONS[i][2];
      const phi = snap.phi ?? 0.1;
      const am = snap.alphaMean ?? 0.05;
      const cg = snap.compressionGain ?? 0;

      tmpTarget.set(
        ax + (phi - 0.1) * 7,
        ay + Math.max(-0.35, Math.min(0.55, cg * 1.2 + (phi - 0.35) * 0.4)),
        az + (am - 0.05) * 55
      );
      const lerp = g.userData.isPybullet ? 0.035 : 0.055;
      g.position.lerp(tmpTarget, lerp);

      g.userData.ring.rotation.z += 0.015 + am * 0.03;
      g.userData.ring.rotation.y += 0.008;
      g.userData.ringS.rotation.x += 0.002 + phi * 0.008;
      g.userData.byz.rotation.y += 0.003;
      g.userData.byz.material.opacity = byzOp;
      g.userData.ragPulse = Math.max(0, g.userData.ragPulse - 1);
      g.userData.ragRing.rotation.z += 0.008;
      g.userData.ragRing.material.opacity =
        g.userData.ragPulse > 0 ? 0.1 + Math.sin(localFrame * 0.2) * 0.05 : 0;

      if (g.userData.physRing) {
        g.userData.physRing.rotation.z += 0.005;
        g.userData.physRing.material.opacity = 0.05 + snap.discoveryRate * 0.3;
      }

      const blkRate = snap.valueLayer?.block_rate ?? 0;
      g.userData.shield.material.opacity =
        blkRate > 0.1 ? 0.15 + Math.sin(localFrame * 0.12 + i) * 0.08 : 0;

      if (g.userData.isPybullet) {
        g.userData.body.rotation.x += 0.02 + phi * 0.02;
        g.userData.body.rotation.y += 0.014 + (snap.discoveryRate ?? 0) * 0.04;
      }

      const dagGlow = Math.max(0, 0.15 - Math.min((snap.hW ?? 0) * 0.05, 0.15));
      g.userData.body.material.emissiveIntensity =
        0.2 + Math.max(0, snap.compressionGain) * 0.1 + dagGlow + phi * 0.08;

      const visCount = Math.min((snap.edgeCount ?? 0) + 2, GRAPH_LINES);
      const tickPhase = simTick * 0.012;
      orbitNodes[i].forEach((node, k) => {
        if (k < visCount) {
          const angle = (k / visCount) * Math.PI * 2 + tickPhase;
          const r = 1.6 + (k % 2) * 0.5;
          node.position.set(
            g.position.x + Math.cos(angle) * r,
            g.position.y + Math.sin(tickPhase * 0.85 + k * 0.7) * 0.28 + 0.2,
            g.position.z + Math.sin(angle) * r
          );
          node.material.opacity = 0.4 + am * 0.5;
        } else node.material.opacity = 0;
      });
      causalLines[i].forEach((line, k) => {
        if (k < visCount - 1) {
          const na = orbitNodes[i][k].position;
          const nb = orbitNodes[i][(k + 1) % visCount].position;
          const pa = line.geometry.attributes.position;
          pa.setXYZ(0, na.x, na.y, na.z);
          pa.setXYZ(1, nb.x, nb.y, nb.z);
          pa.needsUpdate = true;
          line.material.color.set((snap.edges[k]?.weight ?? 0) < 0 ? 0xff4422 : AGENT_COLORS_HEX[i]);
          line.material.opacity = 0.3 + am * 0.45;
        } else line.material.opacity = 0;
      });
    });

    const nAg = Math.max(1, Math.min(ds.nAgents, agentGroups.length));
    const lt = Math.min(Math.max(0, ds.demon?.last_target ?? 0), nAg - 1);
    const dTarget = agentGroups[lt];
    demonDir.copy(dTarget.position).sub(demon.position);
    const chaseSp = isSiege ? 0.052 : demonMode === "anti_byz" ? 0.038 : 0.028;
    if (demonDir.lengthSq() > 1e-6) demonDir.normalize().multiplyScalar(chaseSp);
    else demonDir.set(0, 0, 0);
    demon.userData.vel.lerp(demonDir, isSiege ? 0.12 : 0.07);
    demon.position.add(demon.userData.vel);
    demon.position.y = 1.15 + (ds.demon?.energy ?? 1) * 0.35;
    demon.rotation.y += isSiege ? 0.07 : 0.04;
    demon.rotation.x += 0.022;

    byzLines.forEach((l) => {
      const pa = agentGroups[l.userData.a].position;
      const pb = agentGroups[l.userData.b].position;
      const p = l.geometry.attributes.position;
      p.setXYZ(0, pa.x, pa.y, pa.z);
      p.setXYZ(1, pb.x, pb.y, pb.z);
      p.needsUpdate = true;
      l.material.opacity = byzActive ? byzOp * 0.6 : 0;
    });
    tomLines.forEach((l) => {
      const link = ds.tomLinks.find((x) => x.a === l.userData.a && x.b === l.userData.b);
      const pa = agentGroups[l.userData.a].position;
      const pb = agentGroups[l.userData.b].position;
      const p = l.geometry.attributes.position;
      p.setXYZ(0, pa.x, pa.y, pa.z);
      p.setXYZ(1, pb.x, pb.y, pb.z);
      p.needsUpdate = true;
      l.material.opacity = link ? link.strength * 0.6 : 0;
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
    cancelAnimationFrame(rafId);
    window.removeEventListener("resize", onResize);
    if (mount.contains(renderer.domElement)) mount.removeChild(renderer.domElement);
    renderer.dispose();
  };
}
