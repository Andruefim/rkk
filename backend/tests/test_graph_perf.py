from engine.graph_perf import (
    is_large_graph,
    large_graph_node_cap,
    snapshot_edges_max_for_graph,
)


class _G:
    def __init__(self, n: int):
        self._node_ids = [f"n{i}" for i in range(n)]


def test_large_graph_cap():
    assert large_graph_node_cap() >= 48
    assert not is_large_graph(_G(72))
    assert is_large_graph(_G(100))


def test_snapshot_edges_zero_when_large():
    assert snapshot_edges_max_for_graph(_G(40)) == 512
    assert snapshot_edges_max_for_graph(_G(120)) == 0
