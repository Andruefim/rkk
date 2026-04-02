export function normalizeAgent(a) {
  return {
    id: a.id ?? 0,
    name: a.name ?? "?",
    envType: a.env_type ?? "—",
    activation: a.activation ?? "relu",
    graphMdl: a.graph_mdl ?? 0,
    compressionGain: a.compression_gain ?? 0,
    alphaMean: a.alpha_mean ?? 0.05,
    phi: a.phi ?? 0.1,
    nodeCount: a.node_count ?? 0,
    edgeCount: a.edge_count ?? 0,
    totalInterventions: a.total_interventions ?? 0,
    totalBlocked: a.total_blocked ?? 0,
    lastDo: a.last_do ?? "—",
    lastBlockedReason: a.last_blocked_reason ?? "",
    discoveryRate: a.discovery_rate ?? 0,
    hW: a.h_W ?? 0,
    notears: a.notears ?? null,
    temporal: a.temporal ?? null,
    system1: a.system1 ?? null,
    valueLayer: a.value_layer ?? null,
    edges: a.edges ?? [],
  };
}

export function normalizeFrame(raw) {
  const nAgents = raw.n_agents ?? 4;
  const agents = (raw.agents ?? []).slice(0, nAgents).map(normalizeAgent);
  while (agents.length < nAgents) agents.push(normalizeAgent({ id: agents.length }));

  const pb = raw.pybullet;
  const pybulletNorm = pb
    ? {
        ...pb,
        objects: Array.isArray(pb.objects) ? pb.objects : [],
      }
    : null;

  return {
    tick: raw.tick ?? 0,
    phase: raw.phase ?? 1,
    entropy: raw.entropy ?? 100,
    agents,
    nAgents,
    demon: raw.demon ?? {
      energy: 1,
      cooldown: 0,
      mode: "probe",
      success_rate: 0,
      last_target: 0,
    },
    tomLinks: raw.tom_links ?? [],
    events: raw.events ?? [],
    valueLayer: raw.value_layer ?? null,
    byzantine: raw.byzantine ?? null,
    motif: raw.motif ?? null,
    pybullet: pybulletNorm,
    multiprocess: raw.multiprocess ?? false,
  };
}
