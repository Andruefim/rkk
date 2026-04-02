export function EventStream({ connected, events }) {
  return (
    <div
      style={{
        position: "absolute",
        bottom: 14,
        left: 14,
        right: 14,
        background: "rgba(0,8,20,0.93)",
        border: "1px solid #081e30",
        padding: "8px 14px",
        borderRadius: 2,
        maxHeight: 110,
        overflow: "hidden",
      }}
    >
      <div style={{ color: "#113344", fontSize: 9, letterSpacing: "0.1em", marginBottom: 4 }}>
        CAUSAL EVENT STREAM {connected ? "— PyBullet·Byzantine·RAG" : "— OFFLINE"}
      </div>
      {events.map((ev, i) => (
        <div
          key={i}
          style={{
            color: ev.color ?? "#335566",
            fontSize: 10,
            marginBottom: 2,
            opacity: Math.max(0.15, 1 - i * 0.1),
            fontWeight: ev.type === "value" || ev.type === "tom" ? "bold" : "normal",
          }}
        >
          [{String(ev.tick ?? 0).padStart(4, "0")}] › {ev.text}
        </div>
      ))}
    </div>
  );
}
