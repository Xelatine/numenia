"""TEST DE RETRIEVAL — à faire AVANT d'ingérer le moindre souvenir.

Un souvenir présent mais introuvable SE MANIFESTE comme un souvenir absent :
c'est le problème n°1. On stocke N souvenirs au contenu connu, on interroge par
paraphrase, on exige hit@1.

Trois encodeurs, trois preuves différentes — aucune ne doit être confondue :

* ``hash``    : vecteurs aléatoires déterministes. Ne prouve QUE le mécanisme
                (cosinus, top-k, consolidation). hit@1 attendu ≈ hasard.
* ``lexical`` : sac-de-mots haché. Prouve le retrieval quand l'embedding porte
                un signal, mais c'est du LEXIQUE, pas de la sémantique.
* ``graph``   : moyenne des embeddings de tokens du cerveau texte. C'est
                l'encodeur réel visé ; il n'est sémantique qu'APRÈS la phase 2,
                donc rouge avant entraînement — et c'est un résultat, pas une
                gêne.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mycelia.memory import EpisodicMemory  # noqa: E402

GOLDEN = [
    ("Je m'appelle Théo, j'ai une allergie à la pénicilline.",
     "quelle est l'allergie de Théo ?", "pénicilline"),
    ("Ma couleur préférée est le bleu cobalt.",
     "quelle couleur aime-t-il ?", "bleu"),
    ("Je travaille au laboratoire de biologie du campus.",
     "où travaille-t-il ?", "laboratoire"),
    ("Mon chat s'appelle Fermion et il est noir.",
     "comment s'appelle le chat ?", "Fermion"),
    ("Je déteste qu'on me coupe la parole.",
     "qu'est-ce qui l'énerve ?", "parole"),
]

DIM = 32


def hash_embed(text: str, dim: int = DIM) -> np.ndarray:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
    return rng.standard_normal(dim).astype(np.float32)


def lexical_embed(text: str, dim: int = DIM) -> np.ndarray:
    """Sac-de-mots haché, L2-normalisé. Lexical, jamais présenté comme sémantique."""
    vector = np.zeros(dim, dtype=np.float32)
    for word in re.findall(r"[a-zà-ÿ0-9]+", text.lower()):
        slot = int.from_bytes(hashlib.md5(word.encode("utf-8")).digest()[:4], "little") % dim
        vector[slot] += 1.0
    norm = np.linalg.norm(vector)
    return vector / norm if norm else vector


class GraphEncoder:
    """Moyenne des embeddings de tokens du cerveau — l'encodeur réel visé."""

    def __init__(self, dim: int = DIM, vocab: int = 10_000):
        import torch
        from mycelia.brain import Brain

        self.torch = torch
        self.tokenizer = None
        from mycelia.data import load_tokenizer

        try:
            self.tokenizer = load_tokenizer()
        except FileNotFoundError:
            self.tokenizer = None
        torch.manual_seed(0)
        self.brain = Brain(n_nodes=64, dim=dim, vocab=vocab, max_hops=2,
                           max_births=0, max_edges=256)
        self.brain.eval()

    def __call__(self, text: str) -> np.ndarray:
        if self.tokenizer is None:
            return hash_embed(text)
        ids = self.tokenizer.encode(text).ids[:64] or [0]
        with self.torch.no_grad():
            rows = self.brain.embedding(self.torch.tensor(ids, dtype=self.torch.long))
        return rows.mean(dim=0).numpy().astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", choices=("lexical", "hash", "graph"), default="lexical")
    args = parser.parse_args()
    encode = {"hash": hash_embed, "lexical": lexical_embed, "graph": GraphEncoder()}[args.encoder]

    memory = EpisodicMemory(dim=DIM)
    for content, _query, _expect in GOLDEN:
        memory.add(encode(content), salience=0.9, text=content)

    hits = 0
    for content, query, expect in GOLDEN:
        result = memory.retrieve(encode(query), top_k=1)
        ok = bool(result) and expect.lower() in memory.text[result[0][0]].lower()
        hits += int(ok)
        shown = memory.text[result[0][0]][:44] if result else "AUCUN"
        print(f"{'OK ' if ok else 'KO '} [{expect}] → {shown}")
    ratio = hits / len(GOLDEN)
    print(f"HIT@1 = {hits}/{len(GOLDEN)} (encodeur {args.encoder})")
    if args.encoder == "graph":
        print("  rappel : embeddings non entraînés → ce chiffre est la BASE de la phase 2")
    if args.encoder == "lexical":
        print("  rappel : preuve LEXICALE du pipeline, pas sémantique")
    md5 = memory.save(ROOT / "data" / "mem_golden_test.npz")
    print(f"md5 {md5}")
    assert hits == len(GOLDEN), "RETRIEVAL EN ÉCHEC — ne pas construire dessus"
    print("RETRIEVAL OK (mécanisme vérifié pour cet encodeur)")


if __name__ == "__main__":
    main()
