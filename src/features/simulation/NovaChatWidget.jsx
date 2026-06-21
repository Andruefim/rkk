import { useCallback, useEffect, useRef, useState } from "react";

function agentBaseUrl() {
  if (typeof window === "undefined") return "http://localhost:8000";
  const o = window.location.origin;
  return o.replace("5173", "8000").replace("5174", "8000");
}

/** Чат Nova: только реплики агента (нейро-речь) и ваши команды. */
export default function NovaChatWidget() {
  const AGENT_URL = agentBaseUrl();
  const WS_URL = AGENT_URL.replace(/^http/, "ws") + "/api/ws/chat";

  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [online, setOnline] = useState(false);
  const [pendingAsk, setPendingAsk] = useState(false);
  const pendingAskIdRef = useRef(null);

  const msgsRef = useRef(null);
  const wsRef = useRef(null);
  const reconnectRef = useRef(null);

  const escHtml = (t) =>
    String(t ?? "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;");

  const scrollBottom = () => {
    const el = msgsRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  };

  const addAgentMsg = useCallback((data) => {
    if (data.type === "REPORT") return;

    const isAsk = data.type === "ASK";
    if (isAsk) {
      pendingAskIdRef.current = data.id;
      setPendingAsk(true);
    }

    setMessages((prev) => [
      ...prev,
      {
        kind: "agent",
        text: data.text,
        isAsk,
        key: data.id || `a-${data.tick}-${prev.length}`,
      },
    ]);
  }, []);

  const addHumanMsg = useCallback((text) => {
    if (pendingAskIdRef.current) {
      pendingAskIdRef.current = null;
      setPendingAsk(false);
    }
    setMessages((prev) => [
      ...prev,
      { kind: "human", text, key: `h-${Date.now()}` },
    ]);
  }, []);

  const loadHistory = useCallback(
    (rows) => {
      setMessages([]);
      pendingAskIdRef.current = null;
      setPendingAsk(false);
      if (!rows?.length) return;
      rows.forEach((m) => {
        if (m.type === "REPORT") return;
        addAgentMsg(m);
        if (m.human_replied && m.human_reply) addHumanMsg(m.human_reply);
      });
      const last = [...rows].reverse().find((m) => m.type === "ASK" && !m.human_replied);
      if (last) {
        pendingAskIdRef.current = last.id;
        setPendingAsk(true);
      }
    },
    [addAgentMsg, addHumanMsg]
  );

  useEffect(() => {
    scrollBottom();
  }, [messages]);

  const sendCommand = async () => {
    const text = input.trim();
    if (!text) return;
    setInput("");
    addHumanMsg(text);

    try {
      const ws = wsRef.current;
      if (ws && ws.readyState === 1) {
        ws.send(JSON.stringify({ type: "command", text }));
      } else {
        await fetch(AGENT_URL + "/api/agent/command", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text }),
        });
      }
    } catch {
      /* ignore */
    }
  };

  const sendReply = async () => {
    const text = input.trim();
    if (!text) return;
    if (!pendingAskIdRef.current) {
      await sendCommand();
      return;
    }
    setInput("");
    addHumanMsg(text);
    try {
      const ws = wsRef.current;
      if (ws && ws.readyState === 1) {
        ws.send(JSON.stringify({ type: "reply", text }));
      } else {
        await fetch(AGENT_URL + "/api/agent/reply", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text }),
        });
      }
    } catch {
      /* ignore */
    }
  };

  useEffect(() => {
    const connectWS = () => {
      if (wsRef.current) wsRef.current.close();
      setOnline(false);

      const ws = new WebSocket(WS_URL);
      wsRef.current = ws;
      ws.onopen = () => {
        setOnline(true);
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
        } catch {
          /* ignore */
        }
      };
      ws.onclose = () => {
        setOnline(false);
        reconnectRef.current = setTimeout(connectWS, 3000);
      };
      ws.onerror = () => ws.close();
    };

    connectWS();

    (async () => {
      try {
        const r = await fetch(AGENT_URL + "/api/agent/messages?last_n=30");
        if (!r.ok) return;
        const d = await r.json();
        if (d.available === false) return;
        loadHistory(d.messages || []);
      } catch {
        /* ignore */
      }
    })();

    return () => {
      if (reconnectRef.current) clearTimeout(reconnectRef.current);
      if (wsRef.current) wsRef.current.close();
    };
  }, [AGENT_URL, WS_URL, addAgentMsg, loadHistory]);

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
        background: "rgba(2,5,14,0.95)",
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
            background: online ? "#1D9E75" : "#E24B4A",
            flexShrink: 0,
          }}
        />
        <div style={{ fontSize: 12, fontWeight: 600, color: "#dde6ff" }}>Nova</div>
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
        {messages.length === 0 && (
          <div style={{ textAlign: "center", padding: "24px 8px", fontSize: 11, color: "#556677" }}>
            {online ? "Жду речь агента…" : "Нет связи с backend"}
          </div>
        )}
        {messages.map((m) =>
          m.kind === "human" ? (
            <div
              key={m.key}
              style={{ display: "flex", justifyContent: "flex-end" }}
            >
              <div
                style={{
                  maxWidth: "85%",
                  background: "rgba(15,25,45,0.95)",
                  border: "1px solid #1a2a40",
                  borderRadius: "8px 0 8px 8px",
                  padding: "8px 10px",
                  fontSize: 12,
                  color: "#dde6f0",
                  lineHeight: 1.45,
                }}
              >
                {escHtml(m.text)}
              </div>
            </div>
          ) : (
            <div key={m.key} style={{ display: "flex", gap: 8, alignItems: "flex-start" }}>
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
              <div
                style={{
                  maxWidth: "85%",
                  background: "#e8edff",
                  border: "1px solid #c5d0f0",
                  borderRadius: "0 8px 8px 8px",
                  padding: "8px 10px",
                  fontSize: 12,
                  color: "#1a2040",
                  lineHeight: 1.45,
                }}
              >
                {escHtml(m.text)}
              </div>
            </div>
          )
        )}
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
          placeholder={pendingAsk ? "Ответить…" : "Написать агенту…"}
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
