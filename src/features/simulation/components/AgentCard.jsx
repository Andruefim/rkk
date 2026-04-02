import {
  AGENT_COLORS_CSS,
  phiColor,
  hWColor,
  blkColor,
  sep,
} from "../constants.js";

export function AgentCard({ a, i, isDemonTarget, isPybullet = false }) {
  const col = AGENT_COLORS_CSS[i];
  const phiC = phiColor(a.phi ?? 0);
  const hWc = hWColor(a.hW ?? 0);
  const cgC =
    (a.compressionGain ?? 0) > 0
      ? "#00ff99"
      : (a.compressionGain ?? 0) < -0.05
        ? "#ff4433"
        : "#aaccdd";
  const blkR = a.valueLayer?.block_rate ?? 0;
  const blkC = blkColor(blkR);
  const hasBlk = !!a.lastBlockedReason;
  const bdrCol = isDemonTarget ? "#660000" : hasBlk ? "#882200" : col + "33";
  const lBdr = isDemonTarget ? "#ff0000" : hasBlk ? "#ff4422" : col;

  return (
    <div
      style={{
        background: "rgba(0,8,20,0.93)",
        border: `1px solid ${bdrCol}`,
        borderLeft: `3px solid ${lBdr}`,
        padding: "8px 11px",
        minWidth: 215,
        borderRadius: 2,
        transition: "border-color 0.3s",
        boxShadow: isPybullet ? "0 0 18px #6622aa22" : undefined,
      }}
    >
      <div
        style={{
          color: isDemonTarget ? "#ff4422" : hasBlk ? "#ff6644" : col,
          fontSize: 10,
          fontWeight: "bold",
          marginBottom: 4,
          letterSpacing: "0.08em",
        }}
      >
        {isDemonTarget ? "⚔" : isPybullet ? "◆" : hasBlk ? "🛡" : "◈"} {a.name}
        <span style={{ color: "#1a3344", marginLeft: 6, fontSize: 8 }}>{a.envType}</span>
        {isPybullet && (
          <span style={{ color: "#441166", marginLeft: 4, fontSize: 8 }}>3D</span>
        )}
      </div>
      <table style={{ borderCollapse: "collapse", width: "100%", fontSize: 9 }}>
        <tbody>
          {[
            ["Φ", <span style={{ color: phiC }}>{((a.phi ?? 0) * 100).toFixed(1)}%</span>],
            [
              "CG",
              <span style={{ color: cgC }}>
                {(a.compressionGain ?? 0) >= 0 ? "+" : ""}
                {(a.compressionGain ?? 0).toFixed(4)}
              </span>,
            ],
            ["α", `${Math.round((a.alphaMean ?? 0) * 100)}%`],
            ["h(W)", <span style={{ color: hWc }}>{(a.hW ?? 0).toFixed(4)}</span>],
            ["do/blk", `${a.totalInterventions ?? 0}/${a.totalBlocked ?? 0}`],
            ["DR", `${((a.discoveryRate ?? 0) * 100).toFixed(0)}%`],
            [isPybullet ? "nodes" : "edges", isPybullet ? a.nodeCount ?? 0 : a.edgeCount ?? 0],
          ].map(([l, v], k) => (
            <tr key={k}>
              <td style={{ color: "#335566", paddingRight: 7, paddingBottom: 1 }}>{l}</td>
              <td style={{ color: "#aad4ee", textAlign: "right" }}>{v}</td>
            </tr>
          ))}
        </tbody>
      </table>

      {a.valueLayer && (
        <div style={{ ...sep, fontSize: 8 }}>
          <div style={{ display: "flex", justifyContent: "space-between" }}>
            <span style={{ color: "#553322" }}>VL {a.valueLayer.vl_phase}</span>
            <span style={{ color: blkC }}>blk:{(blkR * 100).toFixed(1)}%</span>
          </div>
          {hasBlk && (
            <div style={{ color: "#ff6644", fontSize: 8 }}>⚠ {a.lastBlockedReason}</div>
          )}
        </div>
      )}

      {a.notears && (
        <div style={{ ...sep, fontSize: 8 }}>
          <div style={{ display: "flex", justifyContent: "space-between" }}>
            <span style={{ color: "#336644" }}>
              NT {a.notears.steps}s h={a.notears.h_W?.toFixed(3)}
            </span>
            <span style={{ color: a.notears.loss < 0.01 ? "#00ff99" : "#aa8800" }}>
              L={a.notears.loss?.toFixed(5)}
            </span>
          </div>
        </div>
      )}

      {[
        { w: `${Math.round((a.alphaMean ?? 0) * 100)}%`, f: "#002266", t: col },
        { w: `${Math.round((a.phi ?? 0) * 100)}%`, f: "#221100", t: phiC },
        { w: `${Math.max(0, 100 - Math.min((a.hW ?? 0) * 20, 100))}%`, f: "#001800", t: hWc },
        { w: `${Math.min(blkR * 100, 100)}%`, f: "#110000", t: blkC },
      ].map((b, k) => (
        <div
          key={k}
          style={{
            marginTop: k === 0 ? 4 : 2,
            height: k === 0 ? 3 : 2,
            background: "#061422",
            borderRadius: 2,
            overflow: "hidden",
          }}
        >
          <div
            style={{
              width: b.w,
              height: "100%",
              background: `linear-gradient(90deg,${b.f},${b.t})`,
              transition: "width 0.8s",
            }}
          />
        </div>
      ))}
      <div
        style={{
          color: "#1a3344",
          fontSize: 7,
          marginTop: 1,
          display: "flex",
          justifyContent: "space-between",
        }}
      >
        <span>α·Φ·h·vl</span>
        <span style={{ color: "#1a4433" }}>
          dr:{((a.discoveryRate ?? 0) * 100).toFixed(0)}%
        </span>
      </div>
    </div>
  );
}
