"""
visual_concepts.py — Phase M: Visual Concepts для InnerVoiceNet.

120+ концептов описывающих МИРОВОЕ ВОСПРИЯТИЕ, а не тело агента.
Дополняет bodily_concepts из concept_store.py.

Домены:
  SPATIAL   — пространственные отношения (близко, далеко, слева)
  OBJECT    — типы объектов (сфера, куб, рампа, платформа)
  SCENE     — состояние сцены (открытое, загромождённое)
  MOTION    — движение объектов мира
  AFFORDANCE— что можно сделать с объектом (можно толкнуть, обойти)
  NOVELTY   — новизна/знакомость (впервые вижу, уже видел)
  GOAL      — цели связанные с миром (дойти до, обойти)
  RELATION  — отношения между объектами
  DANGER    — опасность в мире (обрыв, скользко)

Принципиальное отличие от bodily concepts:
  bodily: "я теряю баланс" — про тело
  visual: "передо мной препятствие" — про мир
"""
from __future__ import annotations

VISUAL_CONCEPT_DEFS: list[tuple[str, str, str]] = [
    # ── SPATIAL ───────────────────────────────────────────────────────────────
    ("OBJECT_VERY_CLOSE",      "SPATIAL", "object within 0.5m, immediate proximity"),
    ("OBJECT_CLOSE",           "SPATIAL", "object within 1-2m, reachable distance"),
    ("OBJECT_MEDIUM",          "SPATIAL", "object 2-4m away"),
    ("OBJECT_FAR",             "SPATIAL", "object 4m+ away, background"),
    ("OBJECT_LEFT",            "SPATIAL", "object predominantly to the left"),
    ("OBJECT_RIGHT",           "SPATIAL", "object predominantly to the right"),
    ("OBJECT_AHEAD",           "SPATIAL", "object directly in forward path"),
    ("OBJECT_BEHIND",          "SPATIAL", "object behind agent"),
    ("OBJECT_ABOVE",           "SPATIAL", "object above agent's center of mass"),
    ("OBJECT_BELOW",           "SPATIAL", "object below agent's foot level"),
    ("MULTIPLE_OBJECTS",       "SPATIAL", "multiple distinct objects in scene"),
    ("SINGLE_OBJECT",          "SPATIAL", "only one salient object visible"),
    ("OBJECT_BLOCKING_PATH",   "SPATIAL", "object in direct movement path"),
    ("CLEAR_PATH_AHEAD",       "SPATIAL", "forward path is unobstructed"),
    ("DENSE_CLUSTER",          "SPATIAL", "many objects clustered together"),
    ("SPREAD_OBJECTS",         "SPATIAL", "objects spread across scene"),
    ("OBJECT_MOVING",          "SPATIAL", "object is in motion"),
    ("OBJECT_STATIC",          "SPATIAL", "object is stationary"),

    # ── OBJECT TYPES ──────────────────────────────────────────────────────────
    ("SPHERE_DETECTED",        "OBJECT", "spherical object visible"),
    ("CUBE_DETECTED",          "OBJECT", "cubic or box-shaped object visible"),
    ("CYLINDER_DETECTED",      "OBJECT", "cylindrical object visible"),
    ("FLAT_SURFACE",           "OBJECT", "flat horizontal surface like a plate or step"),
    ("INCLINED_SURFACE",       "OBJECT", "inclined or ramp-like surface"),
    ("NARROW_OBJECT",          "OBJECT", "narrow object like a beam or rod"),
    ("LARGE_OBJECT",           "OBJECT", "object larger than agent"),
    ("SMALL_OBJECT",           "OBJECT", "object smaller than agent's torso"),
    ("TALL_OBJECT",            "OBJECT", "object taller than agent's head"),
    ("LOW_OBSTACLE",           "OBJECT", "obstacle lower than knee height"),
    ("ELEVATED_SURFACE",       "OBJECT", "surface at step or platform height"),
    ("RAMP_DETECTED",          "OBJECT", "inclined walkable surface detected"),
    ("STAIRS_DETECTED",        "OBJECT", "stepped surface, potential staircase"),
    ("PLATFORM_DETECTED",      "OBJECT", "elevated flat platform"),
    ("BEAM_DETECTED",          "OBJECT", "narrow beam or balance challenge"),
    ("BALL_DETECTED",          "OBJECT", "round ball-like object"),
    ("CONTAINER_DETECTED",     "OBJECT", "box or container that can be moved"),
    ("MARKER_DETECTED",        "OBJECT", "floor marking or sensor plate"),

    # ── OBJECT PROPERTIES ─────────────────────────────────────────────────────
    ("BRIGHT_OBJECT",          "OBJECT", "object with high brightness or emission"),
    ("DARK_OBJECT",            "OBJECT", "object with low brightness"),
    ("WARM_COLOR",             "OBJECT", "object with warm color (red/orange/yellow)"),
    ("COOL_COLOR",             "OBJECT", "object with cool color (blue/cyan/green)"),
    ("GREEN_OBJECT",           "OBJECT", "green colored object"),
    ("BLUE_OBJECT",            "OBJECT", "blue colored object"),
    ("RED_OBJECT",             "OBJECT", "red colored object"),
    ("WHITE_OBJECT",           "OBJECT", "white or light gray object"),
    ("ORANGE_OBJECT",          "OBJECT", "orange colored object"),
    ("GLOWING_OBJECT",         "OBJECT", "object with emissive glow effect"),

    # ── SCENE STATE ───────────────────────────────────────────────────────────
    ("OPEN_SCENE",             "SCENE", "large open space, minimal clutter"),
    ("CLUTTERED_SCENE",        "SCENE", "many objects, complex navigation"),
    ("SYMMETRIC_SCENE",        "SCENE", "scene is roughly symmetric around agent"),
    ("ASYMMETRIC_SCENE",       "SCENE", "more objects on one side than other"),
    ("FAMILIAR_SCENE",         "SCENE", "scene matches previously seen configuration"),
    ("NOVEL_SCENE",            "SCENE", "scene configuration not seen before"),
    ("WALL_NEARBY",            "SCENE", "room wall or boundary in close proximity"),
    ("CORNER_DETECTED",        "SCENE", "room corner visible, navigation constraint"),
    ("CENTER_OPEN",            "SCENE", "center of room is clear"),
    ("BOUNDARY_NEAR",          "SCENE", "approaching room boundary or wall"),
    ("BRIGHT_LIGHTING",        "SCENE", "well-lit environment"),
    ("DARK_AREA",              "SCENE", "poorly lit region in scene"),
    ("FLOOR_MARKING_VISIBLE",  "SCENE", "colored floor marking or grid visible"),
    ("SENSOR_PLATE_VISIBLE",   "SCENE", "force plate or sensor tile visible"),

    # ── MOTION ───────────────────────────────────────────────────────────────
    ("SCENE_STABLE",           "MOTION", "no objects moving, stable scene"),
    ("OBJECT_JUST_MOVED",      "MOTION", "object changed position recently"),
    ("ROLLING_OBJECT",         "MOTION", "spherical object rolling"),
    ("FALLING_OBJECT",         "MOTION", "object falling or tipping"),
    ("I_MOVED_OBJECT",         "MOTION", "agent action caused object to move"),
    ("SLOT_APPEARED",          "MOTION", "new object appeared in visual field"),
    ("SLOT_DISAPPEARED",       "MOTION", "object left visual field"),
    ("SLOT_CHANGED",           "MOTION", "slot representation changed significantly"),

    # ── AFFORDANCE ────────────────────────────────────────────────────────────
    ("CAN_WALK_THROUGH",       "AFFORD", "space can be traversed on foot"),
    ("CAN_STEP_OVER",          "AFFORD", "obstacle small enough to step over"),
    ("MUST_GO_AROUND",         "AFFORD", "obstacle too large to step over"),
    ("CAN_CLIMB",              "AFFORD", "surface climbable if approached correctly"),
    ("CAN_PUSH",               "AFFORD", "object light enough to push"),
    ("CAN_REACH",              "AFFORD", "object within arm's reach"),
    ("SURFACE_WALKABLE",       "AFFORD", "surface is flat and stable for walking"),
    ("SURFACE_CHALLENGING",    "AFFORD", "surface is inclined or narrow"),
    ("SURFACE_DANGEROUS",      "AFFORD", "surface likely to cause fall"),
    ("SPACE_TOO_NARROW",       "AFFORD", "gap too narrow to pass through"),

    # ── NOVELTY / FAMILIARITY ─────────────────────────────────────────────────
    ("FIRST_TIME_SEEING",      "NOVELTY", "this object configuration never seen before"),
    ("SEEN_BEFORE",            "NOVELTY", "similar scene configuration recalled"),
    ("UNEXPECTED_OBJECT",      "NOVELTY", "object in unexpected position"),
    ("EXPECTED_LAYOUT",        "NOVELTY", "scene matches expectation"),
    ("CHANGED_SINCE_LAST",     "NOVELTY", "something is different from last observation"),
    ("SAME_AS_LAST",           "NOVELTY", "scene unchanged from previous look"),
    ("HIGH_VISUAL_SURPRISE",   "NOVELTY", "prediction error high for visual observation"),
    ("LOW_VISUAL_SURPRISE",    "NOVELTY", "visual scene matches internal model well"),

    # ── GOAL / ATTENTION ──────────────────────────────────────────────────────
    ("ATTENDING_LEFT",         "GOAL", "attention focused on left side"),
    ("ATTENDING_RIGHT",        "GOAL", "attention focused on right side"),
    ("ATTENDING_CENTER",       "GOAL", "attention focused on center"),
    ("GOAL_OBJECT_VISIBLE",    "GOAL", "salient goal-like object visible"),
    ("NAVIGATING_AROUND",      "GOAL", "currently avoiding obstacle"),
    ("APPROACHING_OBJECT",     "GOAL", "moving toward detected object"),
    ("RETREATING",             "GOAL", "moving away from object"),
    ("EXPLORING",              "GOAL", "no specific target, exploring"),
    ("INVESTIGATING",          "GOAL", "examining specific object closely"),

    # ── DANGER / WARNING ─────────────────────────────────────────────────────
    ("EDGE_DETECTED",          "DANGER", "drop or edge visible ahead"),
    ("SLIPPERY_SURFACE",       "DANGER", "surface with low friction visible"),
    ("COLLISION_RISK",         "DANGER", "high risk of collision with object"),
    ("NARROW_PASSAGE",         "DANGER", "tight passage with little margin"),
    ("UNSTABLE_OBJECT",        "DANGER", "object looks unstable, may fall"),

    # ── RELATIONS ─────────────────────────────────────────────────────────────
    ("OBJECTS_ADJACENT",       "RELAT", "two objects touching or very close"),
    ("OBJECT_ON_SURFACE",      "RELAT", "object resting on another surface"),
    ("AGENT_BETWEEN_OBJECTS",  "RELAT", "agent positioned between two objects"),
    ("SYMMETRIC_PAIR",         "RELAT", "two similar objects mirrored in scene"),
    ("STACKED_OBJECTS",        "RELAT", "objects arranged vertically"),
]

# Total visual concepts count
N_VISUAL_CONCEPTS = len(VISUAL_CONCEPT_DEFS)

# Build lookup sets by domain
VISUAL_BY_DOMAIN: dict[str, list[str]] = {}
for name, domain, _ in VISUAL_CONCEPT_DEFS:
    VISUAL_BY_DOMAIN.setdefault(domain, []).append(name)

VISUAL_CONCEPT_NAMES: set[str] = {name for name, _, _ in VISUAL_CONCEPT_DEFS}
