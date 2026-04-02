import { AGENT_COLORS_CSS, sep } from "../../constants.js";

export function RagPanel({ ragLoading, ragAutoSeed, ragResults, seedStatus }) {
  return (
    <div
      style={{
        position: "absolute",
        top: 118,
        left: "50%",
        transform: "translateX(-50%)",
        background: "rgba(0,6,14,0.96)",
        border: "1px solid #004433",
        padding: "12px 16px",
        borderRadius: 2,
        width: 420,
        zIndex: 10,
      }}
    >
      <div style={{ color: "#00aa66", fontSize: 10, marginBottom: 6 }}>🌐 RAG SEED PIPELINE</div>
      <button
        type="button"
        onClick={ragAutoSeed}
        disabled={ragLoading}
        style={{
          width: "100%",
          padding: 5,
          borderRadius: 2,
          fontSize: 9,
          cursor: "pointer",
          background: ragLoading ? "#001a0a" : "#002211",
          border: "1px solid #006633",
          color: ragLoading ? "#225544" : "#00ff88",
          marginBottom: 8,
        }}
      >
        {ragLoading ? "⏳ Fetching Wikipedia…" : "🌐 AUTO-SEED ALL AGENTS"}
      </button>
      {ragResults.length > 0 && (
        <div style={{ fontSize: 9 }}>
          {ragResults.map((r, i) => (
            <div
              key={i}
              style={{
                color: AGENT_COLORS_CSS[i] ?? "#aaa",
                marginBottom: 2,
                display: "flex",
                justifyContent: "space-between",
              }}
            >
              <span>
                {r.agent} ({r.preset})
              </span>
              <span style={{ color: "#336655" }}>
                {r.injected} edges · {r.source}
              </span>
            </div>
          ))}
        </div>
      )}
      <div style={{ ...sep, fontSize: 8, color: "#1a3322" }}>
        PyBullet seeds: obj0_vx→obj0_x, obj1_vy→obj1_y, etc.
        <br />
        NOTEARS анилирует неверные, физика подтверждает верные.
      </div>
      <div
        style={{
          marginTop: 4,
          fontSize: 9,
          color: seedStatus.startsWith("✓") ? "#00ff99" : "#ff4422",
        }}
      >
        {seedStatus}
      </div>
    </div>
  );
}
