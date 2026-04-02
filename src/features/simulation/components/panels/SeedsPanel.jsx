import { AGENT_NAMES, AGENT_COLORS_CSS } from "../../constants.js";

export function SeedsPanel({
  seedAgent,
  setSeedAgent,
  seedText,
  setSeedText,
  seedStatus,
  injectSeeds,
}) {
  return (
    <div
      style={{
        position: "absolute",
        top: 118,
        left: "50%",
        transform: "translateX(-50%)",
        background: "rgba(8,6,0,0.96)",
        border: "1px solid #443300",
        padding: "12px 16px",
        borderRadius: 2,
        width: 420,
        zIndex: 10,
      }}
    >
      <div style={{ color: "#886600", fontSize: 10, marginBottom: 6 }}>💉 MANUAL SEED INJECTION</div>
      <div style={{ display: "flex", gap: 5, marginBottom: 6 }}>
        {AGENT_NAMES.map((name, i) => (
          <button
            key={name}
            type="button"
            onClick={() => setSeedAgent(i)}
            style={{
              flex: 1,
              padding: "2px 4px",
              borderRadius: 2,
              fontSize: 9,
              cursor: "pointer",
              background: seedAgent === i ? "#221100" : "transparent",
              border: `1px solid ${seedAgent === i ? AGENT_COLORS_CSS[i] : "#332200"}`,
              color: seedAgent === i ? AGENT_COLORS_CSS[i] : "#554400",
            }}
          >
            {name}
          </button>
        ))}
      </div>
      {seedAgent === 3 && (
        <div style={{ fontSize: 8, color: "#552288", marginBottom: 4 }}>
          Ignis vars: obj0_x, obj0_vx, obj1_y, obj1_vy, obj2_vz… (18 vars)
        </div>
      )}
      <textarea
        value={seedText}
        onChange={(e) => setSeedText(e.target.value)}
        style={{
          width: "100%",
          height: 80,
          background: "#050300",
          border: "1px solid #332200",
          color: "#aa8800",
          fontSize: 9,
          padding: 6,
          borderRadius: 2,
          fontFamily: "monospace",
          resize: "none",
          boxSizing: "border-box",
        }}
      />
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          marginTop: 6,
        }}
      >
        <button
          type="button"
          onClick={injectSeeds}
          style={{
            padding: "3px 12px",
            borderRadius: 2,
            fontSize: 9,
            cursor: "pointer",
            background: "#221100",
            border: "1px solid #886600",
            color: "#ffaa00",
          }}
        >
          INJECT
        </button>
        <span style={{ color: seedStatus.startsWith("✓") ? "#00ff99" : "#ff4422", fontSize: 9 }}>
          {seedStatus}
        </span>
      </div>
    </div>
  );
}
