"""Le nom : déterministe, stable après réorganisation, figé après écriture."""

from __future__ import annotations

import shutil

import torch

from mycelia.brain import Brain
from mycelia.identity import UNNAMED, Identity
from mycelia.naming import name_at_first_sleep, name_for_brain


def _brain(seed: int = 0, edges: int = 40) -> Brain:
    brain = Brain(n_nodes=32, dim=16, vocab=10, max_hops=3, max_births=0,
                  max_node_births=0, max_edges=200)
    generator = torch.Generator().manual_seed(seed)
    brain.edge_index[0, :edges] = torch.randint(0, 32, (edges,), generator=generator)
    brain.edge_index[1, :edges] = torch.randint(0, 32, (edges,), generator=generator)
    brain.ne = edges
    brain._rebuild_edge_keys()
    return brain


def test_name_is_deterministic() -> None:
    brain = _brain()
    assert name_for_brain(brain) == name_for_brain(brain)


def test_name_ignores_edge_order() -> None:
    brain = _brain()
    first = name_for_brain(brain)
    permutation = torch.randperm(brain.ne, generator=torch.Generator().manual_seed(3))
    src = brain.edge_index[0, : brain.ne].clone()
    tgt = brain.edge_index[1, : brain.ne].clone()
    brain.edge_index[0, : brain.ne] = src[permutation]
    brain.edge_index[1, : brain.ne] = tgt[permutation]
    assert name_for_brain(brain) == first


def test_name_frozen_after_first_sleep(tmp_path) -> None:
    identity = Identity.birth(tmp_path)
    assert identity.name == UNNAMED
    brain = _brain()
    chosen = name_at_first_sleep(identity, brain, tmp_path / "self.json")
    assert chosen != UNNAMED
    # L'ingestion change le graphe : le nom ne doit plus bouger.
    other = _brain(seed=9, edges=80)
    assert name_at_first_sleep(identity, other, tmp_path / "self.json") == chosen
    assert Identity.load(tmp_path / "self.json").name == chosen


def test_name_survives_compaction(tmp_path) -> None:
    identity = Identity.birth(tmp_path)
    chosen = name_at_first_sleep(identity, _brain(), tmp_path / "self.json")
    clone = tmp_path / "compact"
    shutil.copytree(tmp_path, clone)
    assert Identity.load(clone / "self.json").name == chosen
