from __future__ import annotations
import torch
import numpy as np
import uuid
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from engine.causal_graph import CausalGraph
    from engine.agent import RKKAgent

class NeurogenesisEngine:
    """
    Structural ASI: Динамическое выращивание абстрактных латентных узлов.
    Вместо усреднения, мы создаём настоящую скрытую переменную в GNN,
    которая берёт на себя моделирование сложных нелинейных взаимодействий.
    """
    def __init__(self, min_interventions: int = 1500, error_threshold: float = 0.25):
        self.min_interventions = min_interventions
        self.error_threshold = error_threshold
        self._last_growth_tick = 0
        self._cooldown = 1000

    def scan_and_grow(self, agent: RKKAgent, tick: int) -> dict | None:
        """
        Анализирует граф на наличие 'бутылочных горлышек' репрезентации
        и инициирует рост новой архитектуры.
        """
        if agent._total_interventions < self.min_interventions:
            return None
        if tick - self._last_growth_tick < self._cooldown:
            return None
        
        graph = agent.graph
        if graph._core is None:
            return None

        # 1. Ищем узлы с высокой неопределенностью и градиентом (где GNN "страдает")
        with torch.no_grad():
            W_grad = graph._core.W.grad
            if W_grad is None:
                return None
            
            grad_norm = W_grad.abs().cpu().numpy()
            alpha_trust = graph._core.alpha_trust_matrix().cpu().numpy()
        
        # Ищем пару узлов, между которыми высокая неопределенность (1 - alpha)
        # и GNN отчаянно пытается менять веса (высокий градиент).
        uncertainty = 1.0 - alpha_trust
        stress_matrix = grad_norm * uncertainty
        
        # Находим самый проблемный кластер (например, рука взаимодействует с кубом)
        i, j = np.unravel_index(np.argmax(stress_matrix), stress_matrix.shape)
        max_stress = stress_matrix[i, j]

        if max_stress < self.error_threshold:
            return None # Граф пока справляется сам

        node_from = graph._node_ids[i]
        node_to = graph._node_ids[j]

        # Защита от разрастания на базовой локомоции (оставляем её на CPG/низком уровне)
        if "leg" in node_from or "leg" in node_to or "hip" in node_from:
            return None

        # 2. Инициируем нейрогенез
        return self._execute_neurogenesis(agent, node_from, node_to, tick)

    def _execute_neurogenesis(self, agent: RKKAgent, src_node: str, dst_node: str, tick: int) -> dict:
        graph = agent.graph
        
        # Создаём ID нового абстрактного узла (латентного концепта)
        latent_id = f"latent_{src_node.split('_')[0]}_to_{dst_node.split('_')[0]}_{str(uuid.uuid4())[:4]}"
        
        # Получаем текущие состояния и добавляем новый узел с нейтральным значением
        current_ids = list(graph._node_ids)
        new_ids = current_ids + [latent_id]
        
        values = {nid: float(graph.nodes.get(nid, 0.5)) for nid in current_ids}
        values[latent_id] = 0.5 # Нейтральная активация
        
        # Rebind переменных расширит матрицу W внутри GNN (d = d + 1)
        # preserve_state=True использует resize_to в CausalGNNCore, сохраняя выученные веса!
        graph.rebind_variables(new_ids, values, preserve_state=True)
        
        # 3. Каузальное связывание (Wiring)
        # Создаём обходной путь через новый латентный узел
        # src -> latent -> dst
        graph.set_edge(src_node, latent_id, weight=0.4, alpha=0.1)
        graph.set_edge(latent_id, dst_node, weight=0.4, alpha=0.1)
        
        # Штрафуем прямую связь, заставляя сеть использовать новый концепт
        graph.remove_edge(src_node, dst_node)
        
        self._last_growth_tick = tick
        
        print(f"[Neurogenesis] Created abstract node '{latent_id}' to mediate {src_node} -> {dst_node}")
        
        return {
            "type": "structural_asi_growth",
            "new_node": latent_id,
            "mediated_path": f"{src_node} -> {dst_node}",
            "gnn_d_new": graph._d
        }