import { memo, useCallback, useEffect, useMemo, useState } from "react";

const BORDER = "1px solid #0a1a2e";

const STATUS = {
  pending:   { color: "#556677", icon: "○", label: "Pending" },
  active:    { color: "#44aaff", icon: "▶", label: "Active" },
  verifying: { color: "#ccaa44", icon: "◎", label: "Verifying" },
  done:      { color: "#1D9E75", icon: "✓", label: "Done" },
  failed:    { color: "#E24B4A", icon: "✕", label: "Failed" },
  cancelled: { color: "#667788", icon: "—", label: "Cancelled" },
};

function normTaskNode(node) {
  if (!node || typeof node !== "object") return null;
  return {
    id: String(node.id ?? ""),
    label: node.label ?? "—",
    status: STATUS[node.status] ? node.status : "pending",
    kind: node.kind ?? "",
    progress: typeof node.progress === "number" ? node.progress : null,
    current: Boolean(node.current),
    meta: node.meta ?? null,
    children: Array.isArray(node.children)
      ? node.children.map(normTaskNode).filter(Boolean)
      : [],
  };
}

export function normTaskTree(raw) {
  if (!raw || typeof raw !== "object") return null;
  return {
    active: Boolean(raw.active),
    sessionId: raw.session_id ?? null,
    tick: raw.tick ?? 0,
    commandText: raw.command_text ?? "",
    rootStatus: STATUS[raw.root_status] ? raw.root_status : "pending",
    currentNodeId: raw.current_node_id != null ? String(raw.current_node_id) : null,
    progress: typeof raw.progress === "number" ? raw.progress : null,
    nodes: Array.isArray(raw.nodes)
      ? raw.nodes.map(normTaskNode).filter(Boolean)
      : [],
    cleared: Boolean(raw.cleared),
  };
}

function collectAncestorIds(nodes, targetId, path = []) {
  if (!targetId || !nodes?.length) return null;
  for (const node of nodes) {
    const next = [...path, node.id];
    if (node.id === targetId) return next;
    const found = collectAncestorIds(node.children, targetId, next);
    if (found) return found;
  }
  return null;
}

const TaskTreeRow = memo(function TaskTreeRow({
  node,
  depth,
  expanded,
  onToggle,
  currentNodeId,
}) {
  const isCurrent =
    node.current || (currentNodeId != null && node.id === currentNodeId);
  const st = STATUS[node.status] ?? STATUS.pending;
  const hasChildren = node.children.length > 0;
  const isOpen = expanded.has(node.id);
  const pct =
    node.progress != null
      ? Math.round(Math.max(0, Math.min(1, node.progress)) * 100)
      : null;

  const handleToggle = useCallback(() => {
    onToggle(node.id);
  }, [node.id, onToggle]);

  const handleKeyDown = useCallback(
    (e) => {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        onToggle(node.id);
      }
    },
    [node.id, onToggle],
  );

  return (
    <div
      role="treeitem"
      aria-expanded={hasChildren ? isOpen : undefined}
      aria-current={isCurrent ? "step" : undefined}
      aria-label={`${node.label}, ${st.label}`}
      style={{ outline: "none" }}
    >
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: 6,
          padding: "3px 0",
          paddingLeft: depth * 14,
          minHeight: 22,
          background: isCurrent ? "rgba(68,170,255,0.08)" : "transparent",
          borderRadius: 2,
        }}
      >
        {hasChildren ? (
          <button
            type="button"
            onClick={handleToggle}
            onKeyDown={handleKeyDown}
            aria-label={isOpen ? `Collapse ${node.label}` : `Expand ${node.label}`}
            style={{
              width: 18,
              height: 18,
              padding: 0,
              flexShrink: 0,
              border: "1px solid #1a2a40",
              borderRadius: 2,
              background: "rgba(0,8,20,0.6)",
              color: "#8899aa",
              fontSize: 9,
              lineHeight: 1,
              cursor: "pointer",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
            }}
          >
            {isOpen ? "▾" : "▸"}
          </button>
        ) : (
          <span style={{ width: 18, flexShrink: 0 }} aria-hidden="true" />
        )}

        <span
          aria-hidden="true"
          style={{
            color: st.color,
            fontSize: 10,
            width: 12,
            flexShrink: 0,
            textAlign: "center",
          }}
        >
          {st.icon}
        </span>

        <span
          style={{
            flex: 1,
            fontSize: 11,
            color: isCurrent ? "#dde6ff" : "#aabbcc",
            fontWeight: isCurrent ? 600 : 400,
            lineHeight: 1.3,
            wordBreak: "break-word",
          }}
        >
          {node.label}
          {node.kind ? (
            <span style={{ color: "#445566", fontSize: 9, marginLeft: 6 }}>
              [{node.kind}]
            </span>
          ) : null}
        </span>

        {pct != null ? (
          <span
            style={{
              fontSize: 9,
              color: st.color,
              fontFamily: "'Courier New',monospace",
              flexShrink: 0,
            }}
          >
            {pct}%
          </span>
        ) : null}
      </div>

      {pct != null ? (
        <div
          style={{
            marginLeft: depth * 14 + 24,
            marginBottom: 2,
            height: 2,
            background: "#0a1a1a",
            borderRadius: 1,
            overflow: "hidden",
          }}
          aria-hidden="true"
        >
          <div
            style={{
              width: `${pct}%`,
              height: "100%",
              background: st.color,
              opacity: 0.75,
            }}
          />
        </div>
      ) : null}

      {hasChildren && isOpen ? (
        <div role="group" aria-label={`${node.label} subtasks`}>
          {node.children.map((child) => (
            <TaskTreeRow
              key={child.id}
              node={child}
              depth={depth + 1}
              expanded={expanded}
              onToggle={onToggle}
              currentNodeId={currentNodeId}
            />
          ))}
        </div>
      ) : null}
    </div>
  );
});

function TaskTreePanel({ taskTree }) {
  const [expanded, setExpanded] = useState(() => new Set());

  const terminal = taskTree?.rootStatus === "done" ||
    taskTree?.rootStatus === "failed" ||
    taskTree?.rootStatus === "cancelled";
  const visible = Boolean(
    (taskTree?.active || terminal || taskTree?.cleared) &&
      (taskTree.nodes?.length > 0 || taskTree.commandText),
  );

  const currentId = taskTree?.currentNodeId ?? null;
  const sessionId = taskTree?.sessionId ?? null;

  useEffect(() => {
    if (taskTree?.cleared) {
      setExpanded(new Set());
    }
  }, [taskTree?.cleared]);

  useEffect(() => {
    setExpanded(new Set());
  }, [sessionId]);

  useEffect(() => {
    if (!visible || !taskTree?.nodes?.length) return;
    setExpanded((prev) => {
      const next = new Set(prev);
      const path = collectAncestorIds(taskTree.nodes, currentId);
      if (path) {
        path.forEach((id) => next.add(id));
      } else {
        taskTree.nodes.forEach((n) => {
          if (n.status === "active" || n.current) next.add(n.id);
        });
      }
      return next;
    });
  }, [visible, sessionId, currentId, taskTree?.tick]);

  const onToggle = useCallback((id) => {
    setExpanded((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }, []);

  const rootSt = STATUS[taskTree?.rootStatus] ?? STATUS.pending;
  const rootPct =
    taskTree?.progress != null
      ? Math.round(Math.max(0, Math.min(1, taskTree.progress)) * 100)
      : null;

  const treeKey = useMemo(
    () => `${sessionId ?? "none"}-${taskTree?.tick ?? 0}`,
    [sessionId, taskTree?.tick],
  );

  if (!visible) return null;

  return (
    <div
      style={{
        fontFamily: "'Segoe UI', system-ui, sans-serif",
        display: "flex",
        flexDirection: "column",
        maxHeight: 280,
        border: BORDER,
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
          borderBottom: BORDER,
          background: "rgba(5,8,18,0.98)",
        }}
      >
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: 8,
            marginBottom: taskTree.commandText ? 4 : 0,
          }}
        >
          <span style={{ color: rootSt.color, fontSize: 11 }} aria-hidden="true">
            {rootSt.icon}
          </span>
          <div style={{ fontSize: 12, fontWeight: 600, color: "#dde6ff" }}>
            Task tree
          </div>
          {rootPct != null ? (
            <span
              style={{
                marginLeft: "auto",
                fontSize: 9,
                color: rootSt.color,
                fontFamily: "'Courier New',monospace",
              }}
            >
              {rootPct}%
            </span>
          ) : null}
        </div>
        {taskTree.commandText ? (
          <div
            style={{
              fontSize: 10,
              color: "#8899aa",
              lineHeight: 1.35,
              wordBreak: "break-word",
            }}
            title={taskTree.commandText}
          >
            {taskTree.commandText}
          </div>
        ) : null}
        {rootPct != null ? (
          <div
            style={{
              marginTop: 6,
              height: 3,
              background: "#0a1a1a",
              borderRadius: 2,
              overflow: "hidden",
            }}
            role="progressbar"
            aria-valuenow={rootPct}
            aria-valuemin={0}
            aria-valuemax={100}
            aria-label="Overall task progress"
          >
            <div
              style={{
                width: `${rootPct}%`,
                height: "100%",
                background: rootSt.color,
                opacity: 0.8,
              }}
            />
          </div>
        ) : null}
      </div>

      <div
        key={treeKey}
        role="tree"
        aria-label="Task steps"
        style={{
          flex: 1,
          overflowY: "auto",
          padding: "6px 10px 8px",
          background: "rgba(0,4,12,0.5)",
        }}
      >
        {taskTree.nodes.map((node) => (
          <TaskTreeRow
            key={node.id}
            node={node}
            depth={0}
            expanded={expanded}
            onToggle={onToggle}
            currentNodeId={currentId}
          />
        ))}
      </div>
    </div>
  );
}

export default memo(TaskTreePanel);
