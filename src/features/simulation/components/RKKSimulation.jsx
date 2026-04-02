import { mono } from "../constants.js";
import { useRKKSimulationState } from "../hooks/useRKKSimulationState.js";
import { SimulationViewport } from "./SimulationViewport.jsx";
import { ConnectionStatus } from "./ConnectionStatus.jsx";
import { HeaderBar } from "./HeaderBar.jsx";
import { ControlsBar } from "./ControlsBar.jsx";
import { SeedsPanel } from "./panels/SeedsPanel.jsx";
import { RagPanel } from "./panels/RagPanel.jsx";
import { DemonPanel } from "./panels/DemonPanel.jsx";
import { PyBulletPanel } from "./panels/PyBulletPanel.jsx";
import { AgentCard } from "./AgentCard.jsx";
import { LegendCard } from "./LegendCard.jsx";
import { EventStream } from "./EventStream.jsx";

export function RKKSimulation() {
  const {
    ui,
    connected,
    speed,
    setSpeed,
    activePanel,
    setActivePanel,
    seedText,
    setSeedText,
    seedAgent,
    setSeedAgent,
    seedStatus,
    ragLoading,
    ragResults,
    demonStats,
    injectSeeds,
    ragAutoSeed,
    frameRef,
  } = useRKKSimulationState();

  const totalBlocked = ui.valueLayer?.total_blocked_all ?? 0;
  const demonMode = ui.demon?.mode ?? "probe";

  return (
    <div
      style={{
        position: "relative",
        width: "100%",
        height: "100vh",
        background: "#010810",
        overflow: "hidden",
        ...mono,
      }}
    >
      <SimulationViewport frameRef={frameRef} />

      <ConnectionStatus connected={connected} nAgents={ui.nAgents} />
      <HeaderBar ui={ui} totalBlocked={totalBlocked} demonMode={demonMode} />
      <ControlsBar
        ui={ui}
        speed={speed}
        setSpeed={setSpeed}
        activePanel={activePanel}
        setActivePanel={setActivePanel}
      />

      {activePanel === "seeds" && (
        <SeedsPanel
          seedAgent={seedAgent}
          setSeedAgent={setSeedAgent}
          seedText={seedText}
          setSeedText={setSeedText}
          seedStatus={seedStatus}
          injectSeeds={injectSeeds}
        />
      )}
      {activePanel === "rag" && (
        <RagPanel
          ragLoading={ragLoading}
          ragAutoSeed={ragAutoSeed}
          ragResults={ragResults}
          seedStatus={seedStatus}
        />
      )}
      {activePanel === "demon" && <DemonPanel demonStats={demonStats} />}
      {activePanel === "pybullet" && <PyBulletPanel pybullet={ui.pybullet} />}

      <div
        style={{
          position: "absolute",
          top: 118,
          left: 14,
          display: "flex",
          flexDirection: "column",
          gap: 6,
        }}
      >
        {ui.agents.slice(0, 2).map((a, i) => (
          <AgentCard
            key={i}
            a={a}
            i={i}
            isDemonTarget={ui.demon?.last_target === i && ui.demon?.mode !== "probe"}
          />
        ))}
      </div>
      <div
        style={{
          position: "absolute",
          top: 118,
          right: 14,
          display: "flex",
          flexDirection: "column",
          gap: 6,
        }}
      >
        {ui.agents
          .slice(2, 4)
          .map((a, orig_i) => {
            const i = orig_i + 2;
            return (
              <AgentCard
                key={i}
                a={a}
                i={i}
                isDemonTarget={ui.demon?.last_target === i && ui.demon?.mode !== "probe"}
                isPybullet={i === 3}
              />
            );
          })
          .reverse()}
        <LegendCard ui={ui} demonMode={demonMode} />
      </div>

      <EventStream connected={connected} events={ui.events} />
    </div>
  );
}
