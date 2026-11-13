"""Mémoire épisodique : chaque souvenir porte DEUX poids indépendants.

``weight_energy``   décroît selon la forgetting curve, remonte à l'usage.
``weight_salience`` décroît 10x moins vite, fixé à l'encodage
                    (surprise, like/dislike, modulateurs).

Purge si ET SEULEMENT SI les deux tombent sous seuil : l'allergie à la
pénicilline survit six mois sans rappel. Et le retrieval ne filtre JAMAIS par
salience — elle ne sert qu'à la purge. Un souvenir rare mais vital reste
trouvable ; il survit au temps par sa salience, pas par sa fréquence.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np


class EpisodicMemory:
    def __init__(self, dim: int = 32, salience_decay: float = 0.0001, energy_decay: float = 0.001):
        self.dim = int(dim)
        self.salience_decay = float(salience_decay)  # 10x plus lent
        self.energy_decay = float(energy_decay)
        self.emb = np.zeros((0, self.dim), dtype=np.float32)
        self.energy = np.zeros(0, dtype=np.float32)
        self.salience = np.zeros(0, dtype=np.float32)
        self.t_created = np.zeros(0, dtype=np.int64)
        self.text: list[str] = []
        self.cemetery = 0

    def __len__(self) -> int:
        return len(self.text)

    def add(self, embedding, salience, text, t: int | None = None) -> int:
        e = np.asarray(embedding, dtype=np.float32).reshape(1, -1)
        if e.shape[1] != self.dim:
            raise ValueError(f"embedding de dimension {e.shape[1]}, attendu {self.dim}")
        self.emb = np.vstack([self.emb, e])
        # float32 explicitement : np.append(…, 1.0) promeut en float64 et le
        # round-trip NPZ perdait alors la valeur exacte (test_save_roundtrip).
        self.energy = np.append(self.energy, np.float32(1.0))
        self.salience = np.append(self.salience, np.float32(np.clip(salience, 0.0, 1.0)))
        self.t_created = np.append(
            self.t_created, np.int64(t if t is not None else int(time.time()))
        )
        self.text.append(str(text))
        return len(self.text) - 1

    def decay(self, steps: int = 1) -> None:
        """Un pas de forgetting curve (ou ``steps``). Salience 10x plus lente."""
        for _ in range(int(steps)):
            self.energy *= 1.0 - self.energy_decay
            self.salience *= 1.0 - self.salience_decay

    def reinforce(self, idx, boost: float = 0.2) -> None:
        """Rappelé → consolidé à l'usage (replay biologique)."""
        idx = np.atleast_1d(np.asarray(idx, dtype=np.int64))
        self.energy[idx] = np.minimum(self.energy[idx] + boost, 1.0)

    def prune(self, e_thr: float = 0.05, s_thr: float = 0.05) -> int:
        """Purge UNIQUEMENT si energy ET salience sous seuils."""
        dead = (self.energy < e_thr) & (self.salience < s_thr)
        alive = ~dead
        self.cemetery += int(dead.sum())
        self.emb = self.emb[alive]
        self.energy = self.energy[alive]
        self.salience = self.salience[alive]
        self.t_created = self.t_created[alive]
        self.text = [t for t, a in zip(self.text, alive) if a]
        return int(dead.sum())

    def retrieve(self, query_emb, top_k: int = 5, min_energy: float = 0.1):
        """Similarité cosinus, filtrée par énergie seulement.

        Renvoie [(index, similarité), …] et consolide les souvenirs touchés.
        """
        if not self.text:
            return []
        q = np.asarray(query_emb, dtype=np.float32).reshape(-1)
        if q.shape[0] != self.dim:
            raise ValueError(f"requête de dimension {q.shape[0]}, attendu {self.dim}")
        cands = np.where(self.energy >= min_energy)[0]
        if cands.size == 0:
            return []
        norms = np.linalg.norm(self.emb[cands], axis=1) * np.linalg.norm(q) + 1e-9
        sims = (self.emb[cands] @ q) / norms
        order = np.argsort(-sims)[: int(top_k)]
        hits = [(int(cands[i]), float(sims[i])) for i in order]
        self.reinforce([h[0] for h in hits])
        return hits

    def save(self, path: str | Path) -> str:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            emb=self.emb,
            energy=self.energy,
            salience=self.salience,
            t=self.t_created,
            text=np.array(self.text, dtype=object),
        )
        return hashlib.md5(path.read_bytes()).hexdigest()

    @classmethod
    def load(cls, path: str | Path) -> "EpisodicMemory":
        blob = np.load(Path(path), allow_pickle=True)
        memory = cls(dim=blob["emb"].shape[1] if blob["emb"].size else 32)
        memory.emb = blob["emb"].astype(np.float32)
        memory.energy = blob["energy"].astype(np.float32)
        memory.salience = blob["salience"].astype(np.float32)
        memory.t_created = blob["t"].astype(np.int64)
        memory.text = [str(t) for t in blob["text"].tolist()]
        return memory

    def stats(self) -> dict:
        return {
            "n": len(self.text),
            "cemetery": int(self.cemetery),
            "energy_mean": float(self.energy.mean()) if len(self.text) else 0.0,
            "salience_mean": float(self.salience.mean()) if len(self.text) else 0.0,
        }


def save_manifest(path: str | Path, **fields) -> str:
    """Écrit un petit JSON de traçabilité et renvoie son md5."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(fields, indent=2, ensure_ascii=False), encoding="utf-8")
    return hashlib.md5(path.read_bytes()).hexdigest()
