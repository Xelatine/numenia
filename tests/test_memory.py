"""La mémoire à deux poids : l'anti-piège LRU, testé avant d'être cru.

Un souvenir rare mais vital (salience 0.95, jamais rappelé) doit survivre à
1 000 pas de decay alors que son énergie est à zéro. C'est exactement ce qu'un
LRU tuerait.
"""

from __future__ import annotations

import numpy as np

from mycelia.memory import EpisodicMemory


def _vec(seed: int, dim: int = 32) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal(dim).astype(np.float32)


def test_vital_memory_survives_1000_decays() -> None:
    memory = EpisodicMemory(dim=32)
    memory.add(_vec(1), salience=0.95, text="allergie à la pénicilline")
    memory.energy[0] = 0.0  # plus jamais rappelé
    memory.decay(1000)
    removed = memory.prune()
    assert removed == 0, "un souvenir vital a été purgé : piège LRU"
    assert len(memory) == 1
    assert memory.cemetery == 0
    assert memory.salience[0] > 0.9 * 0.95  # décroissance 10x plus lente


def test_banal_memory_is_pruned() -> None:
    memory = EpisodicMemory(dim=32)
    memory.add(_vec(2), salience=0.01, text="il a plu mardi")
    memory.energy[0] = 0.0
    memory.decay(1000)
    removed = memory.prune()
    assert removed == 1
    assert len(memory) == 0
    assert memory.cemetery == 1


def test_salience_alone_does_not_filter_retrieval() -> None:
    """La salience protège de la purge, jamais de la recherche."""
    memory = EpisodicMemory(dim=32)
    target = _vec(3)
    memory.add(target, salience=0.01, text="fait sans importance apparente")
    hits = memory.retrieve(target, top_k=1)
    assert hits and hits[0][0] == 0


def test_energy_floor_blocks_retrieval() -> None:
    memory = EpisodicMemory(dim=32)
    target = _vec(4)
    memory.add(target, salience=0.9, text="souvenir éteint")
    memory.energy[0] = 0.0
    assert memory.retrieve(target, top_k=1, min_energy=0.1) == []


def test_salience_decays_ten_times_slower() -> None:
    """On compare les TAUX, pas les valeurs restantes (deux exponentielles)."""
    import math

    memory = EpisodicMemory(dim=32, salience_decay=0.0001, energy_decay=0.001)
    memory.add(_vec(5), salience=1.0, text="x")
    memory.decay(100)
    # log(0.9999)/log(0.999) ≈ 0.1 : le taux de salience est 10x plus petit.
    rate_ratio = math.log(float(memory.salience[0])) / math.log(float(memory.energy[0]))
    assert 0.095 < rate_ratio < 0.105


def test_retrieval_reinforces_energy() -> None:
    memory = EpisodicMemory(dim=32)
    target = _vec(6)
    memory.add(target, salience=0.2, text="rappelé")
    memory.energy[0] = 0.2
    memory.retrieve(target, top_k=1)
    assert memory.energy[0] == 0.4


def test_save_roundtrip(tmp_path) -> None:
    memory = EpisodicMemory(dim=32)
    memory.add(_vec(7), salience=0.7, text="ligne mémorisée")
    path = tmp_path / "mem.npz"
    md5 = memory.save(path)
    assert len(md5) == 32
    loaded = EpisodicMemory.load(path)
    assert loaded.text == ["ligne mémorisée"]
    assert loaded.salience[0] == memory.salience[0]
