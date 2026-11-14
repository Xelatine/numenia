"""EmbeddingGemma-300M : l'encodeur dédié du retrieval (organe externe).

Le GGUF Q8_0 est téléchargé sur BIO-1, md5 consigné à côté du fichier. Le modèle
est un encodeur de phrase (pas un LM causal) : pooling moyen, vecteur 768-d
L2-normalisé, préfixes de tâche officiels pour séparer requête et document.

Honnêteté : le sens est emprunté à un modèle externe ; les souvenirs restent les
nœuds de MYCELIA. Gemma est la rétine, pas le cortex.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / "data" / "models" / "embeddinggemma-300m-q8_0.gguf"
MD5_FILE = MODEL.with_suffix(".md5")

# Mesuré le 14/11 : le préfixe de requête officiel fait tomber le golden à 4/5
# (la paraphrase « parole » sort 2e). Requête brute + document préfixé = 5/5,
# confirmé sur 5 paires inédites. On garde la config mesurée, pas la théorique.
QUERY_PREFIX = ""
DOC_PREFIX = "title: none | text: "
# Seuil d'ignorance calibré sur Gemma : connus min 0,322 / inconnus max 0,260.
CONFIDENCE_THRESHOLD = 0.30


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class GemmaEncoder:
    """Encodeur sémantique : un vecteur 768-d normalisé par texte."""

    def __init__(
        self,
        model_path: Path | str = MODEL,
        n_ctx: int = 512,
        verify_md5: bool = True,
        pooling_type: int = 1,
    ):
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"GGUF absent : {path}")
        if verify_md5:
            if not MD5_FILE.exists():
                raise FileNotFoundError(f"md5 de référence absent : {MD5_FILE}")
            expected = MD5_FILE.read_text(encoding="utf-8").split()[0].strip()
            found = _md5(path)
            if found != expected:
                raise RuntimeError(f"md5 du GGUF {found} != {expected}")
        from llama_cpp import Llama

        options = dict(
            model_path=str(path),
            embedding=True,
            n_ctx=int(n_ctx),
            n_threads=4,
            verbose=False,
        )
        try:
            # LLAMA_POOLING_TYPE_MEAN : EmbeddingGemma est entraîné en moyenne.
            self._llm = Llama(**options, pooling_type=int(pooling_type))
        except TypeError:
            self._llm = Llama(**options)
        self.dim = int(self._llm.n_embd())
        self.pooling_type = int(pooling_type)

    def __call__(self, text: str, kind: str = "doc") -> np.ndarray:
        prefix = QUERY_PREFIX if kind == "query" else DOC_PREFIX
        raw = np.asarray(
            self._llm.create_embedding(prefix + str(text))["data"][0]["embedding"],
            dtype=np.float32,
        )
        if raw.ndim == 1 and raw.size > self.dim:
            raw = raw.reshape(-1, self.dim).mean(axis=0)
        norm = float(np.linalg.norm(raw))
        return raw / norm if norm else raw


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--texts", nargs="*", default=[])
    args = parser.parse_args()
    encoder = GemmaEncoder()
    print(f"dim={encoder.dim}")
    texts = args.texts or ["Je m'appelle Théo, j'ai une allergie à la pénicilline."]
    for text in texts:
        vector = encoder(text)
        print(f"{text[:52]!r} → dim {vector.shape[0]}, norme {float(np.linalg.norm(vector)):.4f}")


if __name__ == "__main__":
    main()
