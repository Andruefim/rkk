import { PHASE_NAMES } from "../constants.js";

const PANELS = [
  ["💉", "seeds"],
  ["🌐", "rag"],
  ["◆", "demon"],
  ["⚙", "pybullet"],
];

export function ControlsBar({ ui, speed, setSpeed, activePanel, setActivePanel }) {
  return (
    <div
      style={{
        position: "absolute",
        top: 80,
        left: "50%",
        transform: "translateX(-50%)",
        background: "rgba(0,8,20,0.85)",
        border: "1px solid #081e30",
        padding: "5px 12px",
        borderRadius: 2,
        display: "flex",
        gap: 6,
        alignItems: "center",
        fontSize: 9,
      }}
    >
      <span style={{ color: "#224455" }}>SPEED</span>
      {[1, 2, 4, 8].map((s) => (
        <button
          key={s}
          type="button"
          onClick={() => setSpeed(s)}
          style={{
            padding: "2px 7px",
            borderRadius: 2,
            fontSize: 9,
            cursor: "pointer",
            background: speed === s ? "#001e38" : "transparent",
            border: `1px solid ${speed === s ? "#00aaff" : "#081e30"}`,
            color: speed === s ? "#00aaff" : "#334455",
          }}
        >
          {s}×
        </button>
      ))}
      <span style={{ color: "#334455" }}>│</span>
      {PHASE_NAMES.slice(1).map((name, i) => {
        const ph = i + 1;
        return (
          <div
            key={ph}
            style={{
              padding: "2px 6px",
              borderRadius: 2,
              fontSize: 9,
              background: ui.phase === ph ? "#001e38" : "transparent",
              border: `1px solid ${
                ui.phase === ph ? "#00aaff" : ui.phase > ph ? "#003322" : "#081e30"
              }`,
              color:
                ui.phase === ph ? "#00aaff" : ui.phase > ph ? "#00aa55" : "#223344",
            }}
          >
            {ui.phase > ph ? "✓" : ph} {name}
          </div>
        );
      })}
      <span style={{ color: "#334455" }}>│</span>
      {PANELS.map(([icon, panel]) => (
        <button
          key={panel}
          type="button"
          onClick={() => setActivePanel((v) => (v === panel ? null : panel))}
          style={{
            padding: "2px 7px",
            borderRadius: 2,
            fontSize: 9,
            cursor: "pointer",
            background: activePanel === panel ? "#1a0820" : "transparent",
            border: `1px solid ${activePanel === panel ? "#884400" : "#333"}`,
            color: activePanel === panel ? "#cc44ff" : "#445555",
          }}
        >
          {icon}
        </button>
      ))}
    </div>
  );
}
