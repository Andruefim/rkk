import {
  AGENT_NAMES,
  AGENT_ENVS,
  AGENT_COLORS_CSS,
  modeColor,
  devColor,
  sep,
} from "../constants.js";

export function LegendCard({ ui, demonMode }) {
  return (
    <div
      style={{
        background: "rgba(0,8,20,0.93)",
        border: "1px solid #081e30",
        padding: "8px 12px",
        borderRadius: 2,
        fontSize: 9,
        maxWidth: 185,
      }}
    >
      <div style={{ color: "#114466", marginBottom: 6, fontSize: 9 }}>ARCHITECTURE v5</div>
      {AGENT_NAMES.map((n, i) => (
        <div key={n} style={{ color: "#335566", marginBottom: 2 }}>
          <span style={{ color: AGENT_COLORS_CSS[i] }}>●</span> {n}{" "}
          <span style={{ color: "#1a3344" }}>({AGENT_ENVS[i]})</span>
        </div>
      ))}
      <div style={{ color: "#335566", marginBottom: 2 }}>
        <span style={{ color: modeColor(demonMode) }}>◆</span> Demon [{demonMode}]
      </div>
      {ui.byzantine && (
        <div style={{ ...sep }}>
          <div style={{ color: "#003355", marginBottom: 2 }}>
            🗳 BYZ R{ui.byzantine.round} dev=
            <span style={{ color: devColor(ui.byzantine.mean_deviance ?? 0) }}>
              {(ui.byzantine.mean_deviance ?? 0).toFixed(3)}
            </span>
          </div>
        </div>
      )}
      {ui.motif?.last_donor != null && (
        <div style={{ color: "#224466", fontSize: 8 }}>
          🧬 donor={AGENT_NAMES[ui.motif.last_donor]}
        </div>
      )}
      <div style={{ ...sep, color: "#551188", lineHeight: 1.8 }}>
        <div>◎ purple ring = PyBullet orbits</div>
        <div>▲ tetrahedron = Ignis</div>
        <div>■ cubes = physics objects (xyz from sim)</div>
      </div>
    </div>
  );
}
