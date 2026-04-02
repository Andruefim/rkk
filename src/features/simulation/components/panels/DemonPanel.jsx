import { AGENT_NAMES, AGENT_COLORS_CSS, modeColor, sep } from "../../constants.js";

export function DemonPanel({ demonStats }) {
  if (!demonStats) return null;
  return (
    <div
      style={{
        position: "absolute",
        top: 118,
        left: "50%",
        transform: "translateX(-50%)",
        background: "rgba(14,0,0,0.96)",
        border: "1px solid #440000",
        padding: "12px 16px",
        borderRadius: 2,
        width: 360,
        zIndex: 10,
      }}
    >
      <div style={{ color: "#ff4422", fontSize: 10, marginBottom: 6 }}>◆ DEMON v2 — PPO-lite</div>
      <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 9 }}>
        <tbody>
          {[
            [
              "Mode",
              <span style={{ color: modeColor(demonStats.mode) }}>
                {demonStats.mode?.toUpperCase()}
              </span>,
            ],
            ["Energy", `${(demonStats.energy * 100).toFixed(0)}%`],
            ["Success", `${((demonStats.success_rate ?? 0) * 100).toFixed(1)}%`],
            [
              "Reward",
              <span
                style={{
                  color: (demonStats.mean_recent_reward ?? 0) > 0 ? "#ff8844" : "#335544",
                }}
              >
                {(demonStats.mean_recent_reward ?? 0).toFixed(3)}
              </span>,
            ],
          ].map(([l, v], k) => (
            <tr key={k}>
              <td style={{ color: "#553333", paddingRight: 10, paddingBottom: 3 }}>{l}</td>
              <td style={{ color: "#cc6644", textAlign: "right" }}>{v}</td>
            </tr>
          ))}
        </tbody>
      </table>
      {demonStats.memory && (
        <div style={{ ...sep }}>
          {demonStats.memory.map((m) => (
            <div
              key={m.agent}
              style={{ fontSize: 9, display: "flex", justifyContent: "space-between", marginBottom: 2 }}
            >
              <span style={{ color: AGENT_COLORS_CSS[m.agent] }}>{AGENT_NAMES[m.agent]}</span>
              <span style={{ color: "#553333" }}>
                ✓{m.success} ✗{m.fail} ΔΦ={m.phi_drop?.toFixed(4)}
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
