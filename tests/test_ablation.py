"""The frozen-random arm of the growth ablation must really be frozen.

The ablation is only interpretable if the control arm differs from the grown arm
in one thing: whether the topology changes. These tests pin that down -- a graph
that grows back, or a birth counter that moves, would silently turn the control
into a second treatment.
"""

from __future__ import annotations

import torch

from mycelia.ablation import freeze_random_graph
from mycelia.brain import Brain


def test_frozen_graph_is_fixed_and_wired() -> None:
    torch.manual_seed(0)
    brain = Brain(n_nodes=32, max_hops=3, max_edges=256)
    frozen = freeze_random_graph(brain, edges=120, seed=7)

    assert frozen["edges"] == brain.ne == 120
    assert brain.max_births == 0
    assert brain.max_node_births == 0
    assert not brain.allow_node_growth
    assert brain.births == 0
    # The seed path must survive, or the door and the reader are disconnected.
    assert all(brain.has_edge(source, target) for source, target in brain.seed_pairs)
    # Every node has an incoming synapse: an orphan would measure a mute brain.
    in_degree = torch.bincount(brain.edge_index[1, : brain.ne], minlength=brain.n_nodes)
    assert int((in_degree == 0).sum()) == 0
    # No self-loop, which is a definition of the wiring and not a rule of growth.
    assert int((brain.edge_index[0, : brain.ne] == brain.edge_index[1, : brain.ne]).sum()) == 0


def test_frozen_brain_does_not_grow() -> None:
    torch.manual_seed(0)
    brain = Brain(n_nodes=32, max_hops=3, max_edges=256)
    freeze_random_graph(brain, edges=120, seed=7)
    before = brain.ne
    x = torch.randint(2, 99, (2, 6))
    for _ in range(5):
        brain.train()
        brain(x, grow=True)
    assert brain.ne == before
    assert brain.node_births == 0
    assert brain.n_nodes == 32
