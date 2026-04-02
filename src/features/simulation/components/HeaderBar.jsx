import { PHASE_NAMES, modeColor } from "../constants.js";

export function HeaderBar({ ui, totalBlocked, demonMode }) {
  return (
    <div
      style={{
        position: "absolute",
        top: 14,
        left: "50%",
        transform: "translateX(-50%)",
        background: "rgba(0,12,28,0.9)",
        border: "1px solid #0a2a44",
        padding: "7px 20px",
        textAlign: "center",
        borderRadius: 2,
        boxShadow: "0 0 24px #00224455",
        whiteSpace: "nowrap",
      }}
    >
      <div
        style={{
          color: "#00ff99",
          fontSize: 12,
          fontWeight: "bold",
          letterSpacing: "0.18em",
        }}
      >
        RKK v5 — PYBULLET · BYZANTINE · RAG
      </div>
      <div style={{ color: "#115577", fontSize: 10, marginTop: 2, letterSpacing: "0.08em" }}>
        PHASE {ui.phase}/5 › {PHASE_NAMES[ui.phase]}
        &nbsp;│&nbsp;ENT:
        <span style={{ color: ui.entropy < 30 ? "#00ff99" : "#aaccdd" }}>{ui.entropy}%</span>
        &nbsp;│&nbsp;T:{ui.tick}
        &nbsp;│&nbsp;
        <span style={{ color: modeColor(demonMode) }}>◆{demonMode}</span>
        {ui.demon && (
          <span style={{ color: "#443300", marginLeft: 4 }}>
            E:{(ui.demon.energy * 100).toFixed(0)}%
          </span>
        )}
        &nbsp;│&nbsp;
        <span style={{ color: totalBlocked > 0 ? "#ff8844" : "#335544" }}>🛡{totalBlocked}</span>
        {ui.byzantine && (
          <span style={{ color: "#004466", marginLeft: 4 }}>🗳R{ui.byzantine.round}</span>
        )}
        {ui.motif?.last_donor != null && (
          <span style={{ color: "#224466", marginLeft: 4 }}>
            🧬{["N", "A", "L", "I"][ui.motif.last_donor]}
          </span>
        )}
        &nbsp;│&nbsp;<span style={{ color: "#6622aa" }}>◆Ignis</span>
        {ui.pybullet && (
          <span style={{ color: "#441166", marginLeft: 4 }}>
            Φ={ui.pybullet.phi?.toFixed(2)} DR={((ui.pybullet.discovery_rate ?? 0) * 100).toFixed(0)}%
          </span>
        )}
      </div>
    </div>
  );
}
