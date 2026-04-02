export const WORLD = 16;
export const AGENT_COLORS_HEX = [0x00ff99, 0x0099ff, 0xff9900, 0xcc44ff];
export const AGENT_COLORS_CSS = ["#00ff99", "#0099ff", "#ff9900", "#cc44ff"];
export const AGENT_NAMES = ["Nova", "Aether", "Lyra", "Ignis"];
export const AGENT_ENVS = ["physics", "chemistry", "logic", "pybullet"];
export const DEMON_COLOR = 0xff2244;
export const GRAPH_LINES = 14;
export const PHASE_NAMES = [
  "",
  "Causal Crib",
  "Robotic Explorer",
  "Social Sandbox",
  "Value Lock",
  "Open Reality",
];
export const ATTACK_MODES = {
  probe: "#335566",
  targeted: "#ff8844",
  siege: "#ff2244",
  anti_byz: "#aa00ff",
};

/** Базовые якоря агентов в мире Three.js */
export const AGENT_POSITIONS = [
  [-6, 1, -2],
  [2, 1, 3],
  [-2, 1, 3],
  [6, 1, -2],
];

export const hWColor = (h) =>
  h < 0.01 ? "#00ff99" : h < 0.5 ? "#aacc00" : h < 2 ? "#ffaa00" : "#ff4422";
export const phiColor = (p) =>
  p > 0.6 ? "#00ff99" : p > 0.3 ? "#aacc00" : "#ff8844";
export const blkColor = (r) =>
  r > 0.3 ? "#ff4422" : r > 0.1 ? "#ffaa00" : "#335544";
export const modeColor = (m) => ATTACK_MODES[m] ?? "#335566";
export const devColor = (d) =>
  d < 0.05 ? "#00ff99" : d < 0.15 ? "#aacc00" : "#ff8844";

export const mono = { fontFamily: "'Courier New',monospace" };
export const sep = {
  borderTop: "1px solid #081e30",
  marginTop: 4,
  paddingTop: 4,
};
