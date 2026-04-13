"""Общие импорты для миксинов Simulation (без циклического import пакета `features.simulation`)."""
from __future__ import annotations

import asyncio
import copy
import os
import queue
import threading
import time
from typing import Any

import numpy as np
import torch
from collections import deque

from engine.hierarchical_graph import HierarchicalGraph, hierarchical_graph_enabled
from engine.ollama_env import get_ollama_generate_url, get_ollama_model
from engine.value_layer import HomeostaticBounds
from engine.wm_neural_ode import integrate_world_model_step

from engine.core import (
    PHASE_HOLD_TICKS,
    PHASE_NAMES,
    PHASE_THRESHOLDS,
    VISION_GNN_FEED_EVERY,
    MotorState,
    WORLDS,
)
from engine.core.constants import (
    agent_loop_hz_from_env as _agent_loop_hz_from_env,
    cpg_loop_hz_from_env as _cpg_loop_hz_from_env,
    l3_loop_hz_from_env as _l3_loop_hz_from_env,
    l4_worker_enabled as _l4_worker_enabled,
)
from engine.features.simulation.imports import *

# Как в `imports.py`: иначе `import *` не подтягивает `_FOO_AVAILABLE` в миксины.
__all__ = [n for n in list(globals()) if not n.startswith("__")]
