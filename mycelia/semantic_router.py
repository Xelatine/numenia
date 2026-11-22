"""Routage sémantique : une question choisit les nœuds par son embedding.

Piste C de la greffe : Qwen (ou EmbeddingGemma) ne perçoit pas à la place des
nœuds, il **guide** le routage. Une clé aléatoire par nœud (dimension externe,
ici 768) transforme l'embedding de la question en scores ; les k nœuds les plus
proches sont éveillés. Des questions proches doivent réveiller des ensembles
proches — c'est la propriété qu'on mesure, pas celle qu'on suppose.

La clé est gelée : aucun poids de NUMENIA n'est modifié, et le routeur reste
déterministe pour une question donnée.
"""

from __future__ import annotations

import numpy as np


class SemanticRouter:
    def __init__(
        self,
        n_nodes: int,
        dim: int = 768,
        fraction: float = 0.10,
        seed: int = 20261115,
    ) -> None:
        if not 0.0 < float(fraction) <= 1.0:
            raise ValueError("fraction doit être dans (0, 1]")
        self.n_nodes = int(n_nodes)
        self.dim = int(dim)
        self.fraction = float(fraction)
        generator = np.random.default_rng(int(seed))
        keys = generator.standard_normal((self.n_nodes, self.dim)).astype(np.float32)
        keys /= np.linalg.norm(keys, axis=1, keepdims=True) + 1e-9
        self.keys = keys

    @property
    def k(self) -> int:
        return max(2, int(round(self.fraction * self.n_nodes)))

    def select(self, embedding) -> np.ndarray:
        vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
        if vector.shape[0] != self.dim:
            raise ValueError(f"embedding de dimension {vector.shape[0]}, attendu {self.dim}")
        vector = vector / (np.linalg.norm(vector) + 1e-9)
        scores = self.keys @ vector
        return np.argsort(-scores)[: self.k]

    def overlap(self, first, second) -> float:
        """Jaccard des deux ensembles de nœuds éveillés."""
        a = set(int(i) for i in first)
        b = set(int(i) for i in second)
        union = a | b
        return len(a & b) / max(1, len(union))
