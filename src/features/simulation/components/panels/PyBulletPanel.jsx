import { phiColor, hWColor, sep } from "../../constants.js";

export function PyBulletPanel({ pybullet }) {
  return (
    <div
      style={{
        position: "absolute",
        top: 118,
        left: "50%",
        transform: "translateX(-50%)",
        background: "rgba(8,0,20,0.96)",
        border: "1px solid #441188",
        padding: "14px 18px",
        borderRadius: 2,
        width: 400,
        zIndex: 10,
      }}
    >
      <div style={{ color: "#cc44ff", fontSize: 10, marginBottom: 8, letterSpacing: "0.1em" }}>
        ◆ IGNIS — PyBullet 3D Physics
      </div>
      {pybullet ? (
        <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 9 }}>
          <tbody>
            {[
              [
                "Φ autonomy",
                <span style={{ color: phiColor(pybullet.phi ?? 0) }}>
                  {((pybullet.phi ?? 0) * 100).toFixed(1)}%
                </span>,
              ],
              [
                "Discovery rate",
                <span style={{ color: "#9933cc" }}>
                  {((pybullet.discovery_rate ?? 0) * 100).toFixed(0)}%
                </span>,
              ],
              ["Interventions", pybullet.interventions],
              ["Nodes (vars)", pybullet.node_count],
              ["Edges", pybullet.edge_count],
              [
                "h(W) DAG",
                <span style={{ color: hWColor(pybullet.h_W ?? 0) }}>
                  {(pybullet.h_W ?? 0).toFixed(4)}
                </span>,
              ],
              [
                "CG (d|G|/dt)",
                <span style={{ color: (pybullet.compression_gain ?? 0) > 0 ? "#00ff99" : "#ff4433" }}>
                  {(pybullet.compression_gain ?? 0).toFixed(4)}
                </span>,
              ],
            ].map(([l, v], k) => (
              <tr key={k}>
                <td style={{ color: "#553377", paddingRight: 10, paddingBottom: 3 }}>{l}</td>
                <td style={{ color: "#cc88ff", textAlign: "right" }}>{v}</td>
              </tr>
            ))}
          </tbody>
        </table>
      ) : (
        <div style={{ color: "#441166", fontSize: 9 }}>Connecting to Ignis…</div>
      )}
      <div style={{ ...sep, fontSize: 8, color: "#331155", lineHeight: 1.8 }}>
        <div style={{ color: "#663399" }}>ENVIRONMENT:</div>
        <div>3 объекта → 18 vars (objN_x/y/z/vx/vy/vz)</div>
        <div>do(objN_vx=v) → set_velocity() → step(×10)</div>
        <div style={{ color: "#663399", marginTop: 3 }}>GT CAUSAL STRUCTURE:</div>
        <div>vx→x, vy→y, vz→z (интеграция)</div>
        <div>obj_x→obj_vx (столкновения)</div>
        <div style={{ color: "#663399", marginTop: 3 }}>BACKEND:</div>
        <div>PyBullet → Fallback если не установлен</div>
        <div>pip install pybullet</div>
      </div>
    </div>
  );
}
