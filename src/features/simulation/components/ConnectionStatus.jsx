export function ConnectionStatus({ connected, nAgents }) {
  return (
    <div
      style={{
        position: "absolute",
        top: 14,
        right: 14,
        background: connected ? "rgba(0,40,10,0.9)" : "rgba(30,10,0,0.9)",
        border: `1px solid ${connected ? "#00aa44" : "#aa4400"}`,
        padding: "4px 10px",
        borderRadius: 2,
        fontSize: 9,
        color: connected ? "#00ff88" : "#ff8844",
      }}
    >
      {connected ? "● PYTHON BACKEND" : "○ OFFLINE"} · MP · {nAgents}A
    </div>
  );
}
