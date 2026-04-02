import { useEffect, useRef } from "react";
import { runRKKScene } from "../scene/rkkThreeScene.js";
import { normalizeFrame } from "../normalize.js";

export function SimulationViewport({ frameRef }) {
  const mountRef = useRef(null);

  useEffect(() => {
    const mount = mountRef.current;
    if (!mount) return undefined;
    return runRKKScene(mount, { frameRef, normalizeFrame });
  }, [frameRef]);

  return <div ref={mountRef} style={{ position: "absolute", inset: 0 }} />;
}
