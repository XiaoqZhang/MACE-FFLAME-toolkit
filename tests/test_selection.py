# tests/test_selection.py
import pytest
from ase.build import molecule
from fflame.sample.selection import selector, build_graph, mst_dfs_tsp

def test_selector_runs():
    frames = [molecule("H2O"), molecule("CH4"), molecule("NH3")]
    selected_frames, indices = selector(frames, number=2, sort=False)
    assert len(selected_frames) == 2
    assert all(isinstance(idx, int) for idx in indices)

def test_graph_and_mst():
    import numpy as np
    points = np.random.rand(5, 3)
    G, dist = build_graph(points)
    assert G.number_of_nodes() == 5
    order = mst_dfs_tsp(points)
    assert set(order) == set(range(5))
