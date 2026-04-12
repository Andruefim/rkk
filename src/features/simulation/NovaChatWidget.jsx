import { useCallback, useEffect, useRef, useState } from "react";

/** Backend origin: Vite dev on :5173 → API on :8000 */
function agentBaseUrl() {
  if (typeof window === "undefined") return "http://localhost:8000";
  const o = window.location.origin;
  return o.replace("5173", "8000").replace("5174", "8000");
}

export default function NovaChatWidget() {
  const AGENT_URL = agentBaseUrl();
  const WS_URL = AGENT_URL.replace(/^http/, "ws") + "/api/ws/chat";

  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [statusLine, setStatusLine] = useState("tick 0 · …");
  const [statusColor, setStatusColor] = useState("#1D9E75");
  const [concepts, setConcepts] = useState([]);
  const pendingAskIdRef = useRef(null);

  const msgsRef = useRef(null);
  const wsRef = useRef(null);
  const reconnectRef = useRef(null);

  const scrollBottom = () => {
    const el = msgsRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  };

  const escHtml = (t) =>
    String(t ?? "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;");

  const typeTag = (type) => {
    const map = {
      OBSERVE: { label: "наблюдение", bg: "#E1F5EE", color: "#0F6E56" },
      ASK: { label: "вопрос", bg: "#E6F1FB", color: "#185FA5" },
      REPORT: { label: "рапорт", bg: "#FAEEDA", color: "#854F0B" },
      HUMAN: { label: "вы", bg: "rgba(20,30,50,0.6)", color: "#aabbcc" },
    };
    const t = map[type] || map.OBSERVE;
    return (
      <span
        style={{
          fontSize: 10,
          padding: "2px 6px",
          borderRadius: 20,
          background: t.bg,
          color: t.color,
          fontWeight: 500,
        }}
      >
        {t.label}
      </span>
    );
  };

  const addAgentMsg = useCallback((data) => {
    const isAsk = data.type === "ASK";
    if (isAsk) pendingAskIdRef.current = data.id;

    const leftAccent =
      { OBSERVE: "#1D9E75", ASK: "#378ADD", REPORT: "#BA7517" }[data.type] ||
      "#1D9E75";

    setMessages((prev) => [
      ...prev,
      {
        kind: "agent",
        data,
        leftAccent,
        isAsk,
        key: data.id || `a-${data.tick}-${prev.length}`,
      },
    ]);
  }, []);

  const addHumanMsg = useCallback((text, reward) => {
    const pid = pendingAskIdRef.current;
    if (pid) pendingAskIdRef.current = null;

    setMessages((prev) => [
      ...prev,
      {
        kind: "human",
        text,
        reward,
        pendingId: pid,
        key: `h-${Date.now()}`,
      },
    ]);
  }, []);

  const loadHistory = useCallback(
    (rows) => {
      setMessages([]);
      if (!rows?.length) return;
      rows.forEach((m) => {
        addAgentMsg(m);
        if (m.human_replied && m.human_reply) addHumanMsg(m.human_reply, m.reward);
      });
    },
    [addAgentMsg, addHumanMsg]
  );

  useEffect(() => {
    scrollBottom();
  }, [messages]);

  const sendReply = async () => {
    const text = input.trim();
    if (!text) return;
    setInput("");
    addHumanMsg(text, 0);

    try {
      const ws = wsRef.current;
      if (ws && ws.readyState === 1) {
        ws.send(JSON.stringify({ type: "reply", text }));
      } else {
        const r = await fetch(AGENT_URL + "/api/agent/reply", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text }),
        });
        const d = await r.json();
        if (d.reward != null) {
          /* badge handled in next paint via human row */
        }
      }
    } catch {
      setMessages((prev) => [
        ...prev,
        { kind: "sys", text: "ошибка отправки", key: `e-${Date.now()}` },
      ]);
    }
  };

  useEffect(() => {
    const fetchStats = async () => {
      try {
        const r = await fetch(AGENT_URL + "/api/snapshot");
        if (!r.ok) return;
        const d = await r.json();
        const tick = d.tick ?? 0;
        setStatusLine(`tick ${tick} · ${d.world ?? d.current_world ?? "humanoid"}`);
        const verbal = d.verbal || {};
        if (verbal.last_message) {
          const m = verbal.last_message;
          setStatusLine(
            `tick ${tick} · ${(m.type || "").toLowerCase()} · cur=${((m.curiosity ?? 0) * 100).toFixed(0)}%`
          );
        }
        const iv = d.inner_voice || {};
        if (iv.active_concepts?.length) {
          setConcepts(iv.active_concepts.slice(0, 4));
        } else setConcepts([]);

        const sleeping = d.sleep?.is_sleeping;
        if (sleeping) {
          setStatusColor("#BA7517");
          setStatusLine(`tick ${tick} · сон (${(d.sleep.current_phase || "").toLowerCase()})`);
        } else {
          setStatusColor("#1D9E75");
        }
      } catch {
        setStatusColor("#E24B4A");
        setStatusLine("нет подключения");
      }
    };

    const connectWS = () => {
      if (wsRef.current) wsRef.current.close();
      setStatusColor("#BA7517");

      const ws = new WebSocket(WS_URL);
      wsRef.current = ws;
      ws.onopen = () => {
        setStatusColor("#1D9E75");
        if (reconnectRef.current) {
          clearTimeout(reconnectRef.current);
          reconnectRef.current = null;
        }
      };
      ws.onmessage = (e) => {
        try {
          const msg = JSON.parse(e.data);
          if (msg.event === "history") loadHistory(msg.data || []);
          else if (msg.event === "agent_message") addAgentMsg(msg.data);
          else if (msg.event === "human_message") {
            /* optional: could append system line */
          }
        } catch {
          /* ignore */
        }
      };
      ws.onclose = () => {
        setStatusColor("#E24B4A");
        reconnectRef.current = setTimeout(connectWS, 3000);
      };
      ws.onerror = () => ws.close();
    };

    connectWS();

    (async () => {
      try {
        const r = await fetch(AGENT_URL + "/api/agent/messages?last_n=30");
        if (!r.ok) {
          setMessages([{ kind: "sys", text: "backend недоступен", key: "s0" }]);
          return;
        }
        const d = await r.json();
        if (d.available === false) {
          setMessages([{ kind: "sys", text: "verbal_action недоступен", key: "s1" }]);
          return;
        }
        loadHistory(d.messages || []);
      } catch {
        setMessages([{ kind: "sys", text: "backend недоступен", key: "s2" }]);
      }
    })();

    const tickIv = setInterval(fetchStats, 2000);
    return () => {
      clearInterval(tickIv);
      if (reconnectRef.current) clearTimeout(reconnectRef.current);
      if (wsRef.current) wsRef.current.close();
    };
  }, [AGENT_URL, WS_URL, addAgentMsg, loadHistory]);

  const bg = "rgba(2,5,14,0.95)";
  const border = "1px solid #0a1a2e";

  return (
    <div
      style={{
        fontFamily: "'Segoe UI', system-ui, sans-serif",
        display: "flex",
        flexDirection: "column",
        height: 340,
        border,
        borderRadius: 4,
        overflow: "hidden",
        background: bg,
        minWidth: 260,
        maxWidth: 320,
      }}
    >
      <div
        style={{
          padding: "8px 10px",
          borderBottom: border,
          display: "flex",
          alignItems: "center",
          gap: 8,
          background: "rgba(5,8,18,0.98)",
        }}
      >
        <div
          style={{
            width: 8,
            height: 8,
            borderRadius: "50%",
            background: statusColor,
            flexShrink: 0,
          }}
        />
        <div style={{ flex: 1, minWidth: 0 }}>
          <div style={{ fontSize: 12, fontWeight: 600, color: "#dde6ff" }}>Nova</div>
          <div
            style={{
              fontSize: 10,
              color: "#7788aa",
              fontFamily: "ui-monospace, monospace",
              whiteSpace: "nowrap",
              overflow: "hidden",
              textOverflow: "ellipsis",
            }}
          >
            {statusLine}
          </div>
        </div>
        <div style={{ display: "flex", gap: 4, flexWrap: "wrap", justifyContent: "flex-end" }}>
          {concepts.map((c, i) => {
            const name = typeof c === "string" ? c : c[0];
            return (
              <span
                key={i}
                style={{
                  fontSize: 9,
                  padding: "2px 6px",
                  borderRadius: 20,
                  background: "#2a2244",
                  color: "#b4aee0",
                  fontFamily: "ui-monospace, monospace",
                }}
              >
                {String(name).toLowerCase().replace(/_/g, " ")}
              </span>
            );
          })}
        </div>
      </div>

      <div
        ref={msgsRef}
        style={{
          flex: 1,
          overflowY: "auto",
          padding: 10,
          display: "flex",
          flexDirection: "column",
          gap: 8,
          background: "rgba(0,4,12,0.5)",
        }}
      >
        {messages.map((m) => {
          if (m.kind === "sys") {
            return (
              <div key={m.key} style={{ textAlign: "center", padding: "6px 0" }}>
                <span style={{ fontSize: 10, color: "#556677" }}>{escHtml(m.text)}</span>
              </div>
            );
          }
          if (m.kind === "human") {
            return (
              <div
                key={m.key}
                style={{
                  display: "flex",
                  justifyContent: "flex-end",
                  gap: 8,
                  alignItems: "flex-start",
                }}
              >
                <div style={{ maxWidth: "78%" }}>
                  <div
                    style={{
                      background: "rgba(15,25,45,0.95)",
                      border: "1px solid #1a2a40",
                      borderRadius: "8px 0 8px 8px",
                      padding: "8px 10px",
                    }}
                  >
                    <div style={{ fontSize: 12, color: "#dde6f0", lineHeight: 1.45 }}>
                      {escHtml(m.text)}
                    </div>
                  </div>
                  <div
                    style={{
                      marginTop: 3,
                      display: "flex",
                      justifyContent: "flex-end",
                    }}
                  >
                    {typeTag("HUMAN")}
                  </div>
                </div>
                <div
                  style={{
                    width: 26,
                    height: 26,
                    borderRadius: "50%",
                    background: "#121a28",
                    border: "1px solid #223",
                    flexShrink: 0,
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    fontSize: 10,
                    color: "#8899aa",
                  }}
                >
                  Вы
                </div>
              </div>
            );
          }
          const data = m.data;
          const isAsk = m.isAsk;
          return (
            <div
              key={m.key}
              style={{ display: "flex", gap: 8, alignItems: "flex-start" }}
            >
              <div
                style={{
                  width: 26,
                  height: 26,
                  borderRadius: "50%",
                  background: "#e8ecff",
                  border: "1px solid #b8c4f0",
                  flexShrink: 0,
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  fontSize: 11,
                  fontWeight: 600,
                  color: "#1a2040",
                }}
              >
                N
              </div>
              <div style={{ maxWidth: "82%", minWidth: 70 }}>
                <div
                  style={{
                    background: "#e8edff",
                    border: "1px solid #c5d0f0",
                    borderLeft: `2px solid ${m.leftAccent}`,
                    borderRadius: "0 6px 6px 6px",
                    padding: "8px 10px",
                  }}
                >
                  <div style={{ fontSize: 12, color: "#1a2040", lineHeight: 1.45 }}>
                    {escHtml(data.text)}
                  </div>
                  {data.concepts?.length > 0 && (
                    <div
                      style={{
                        marginTop: 5,
                        fontSize: 9,
                        color: "#5f5e5a",
                        fontFamily: "ui-monospace, monospace",
                      }}
                    >
                      {data.concepts
                        .slice(0, 3)
                        .map((c) => String(c).toLowerCase().replace(/_/g, " "))
                        .join(" · ")}
                    </div>
                  )}
                </div>
                <div
                  style={{
                    marginTop: 3,
                    display: "flex",
                    gap: 4,
                    alignItems: "center",
                    flexWrap: "wrap",
                  }}
                >
                  {typeTag(data.type)}
                  <span style={{ fontSize: 9, color: "#667788" }}>
                    tick {data.tick ?? 0} · cur={((data.curiosity ?? 0) * 100).toFixed(0)}%
                  </span>
                  {isAsk && (
                    <span style={{ fontSize: 9, color: "#378ADD" }}>ожидает ответа…</span>
                  )}
                </div>
              </div>
            </div>
          );
        })}
      </div>

      <div
        style={{
          padding: "8px 10px",
          borderTop: border,
          background: "rgba(5,8,18,0.98)",
          display: "flex",
          gap: 8,
          alignItems: "flex-end",
        }}
      >
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              sendReply();
            }
          }}
          placeholder="Ответить агенту…"
          rows={1}
          style={{
            flex: 1,
            resize: "none",
            fontFamily: "inherit",
            fontSize: 12,
            padding: "8px 10px",
            border: "1px solid #223344",
            borderRadius: 4,
            background: "rgba(0,8,20,0.6)",
            color: "#dde6f0",
            lineHeight: 1.4,
            maxHeight: 72,
            overflowY: "auto",
          }}
        />
        <button
          type="button"
          onClick={sendReply}
          style={{
            padding: "8px 12px",
            fontSize: 11,
            fontWeight: 600,
            borderRadius: 4,
            background: "#121a28",
            border: "1px solid #334",
            color: "#dde6f0",
            cursor: "pointer",
            height: 34,
            flexShrink: 0,
          }}
        >
          Отпр.
        </button>
      </div>
    </div>
  );
}
