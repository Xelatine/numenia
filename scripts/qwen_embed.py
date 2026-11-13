"""Qwen2.5-0.5B comme organe des sens du retrieval (encodeur externe).

Le GGUF est déjà sur BIO-1 (md5 vérifié en session O) : `llama-cpp-python`
tourne en mode embedding, un forward par texte, vecteur 896-d L2-normalisé.

Honnêteté : le *sens* de la recherche est emprunté à un modèle externe. Les
souvenirs restent les nœuds de MYCELIA ; Qwen est la rétine, pas le cortex.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / "data" / "models" / "qwen2.5-0.5b-instruct-q8_0.gguf"
EXPECTED_MD5 = "a49598f88d6157e900ae2aac730873c9"


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class QwenEncoder:
    """Encodeur sémantique : un vecteur 896-d normalisé par texte."""

    def __init__(
        self,
        model_path: Path | str = MODEL,
        n_ctx: int = 512,
        verify_md5: bool = True,
        pooling_type: int = 3,
    ):
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"GGUF absent : {path}")
        if verify_md5:
            found = _md5(path)
            if found != EXPECTED_MD5:
                raise RuntimeError(f"md5 du GGUF {found} != {EXPECTED_MD5}")
        from llama_cpp import Llama

        options = dict(
            model_path=str(path),
            embedding=True,
            n_ctx=int(n_ctx),
            n_threads=4,
            verbose=False,
        )
        try:
            # LLAMA_POOLING_TYPE_MEAN : un vecteur par texte, pas un par token.
            self._llm = Llama(**options, pooling_type=int(pooling_type))
        except TypeError:  # version sans pooling exposé : on moyenne nous-mêmes
            self._llm = Llama(**options)
        self.dim = int(self._llm.n_embd())
        # 1 = mean, 2 = CLS, 3 = last token (standard pour un LM causal).
        self.pooling_type = int(pooling_type)

    def __call__(self, text: str) -> np.ndarray:
        raw = np.asarray(
            self._llm.create_embedding(str(text))["data"][0]["embedding"],
            dtype=np.float32,
        )
        # Sans pooling, llama.cpp concatène les vecteurs par token : (n_tok × dim).
        if raw.ndim == 1 and raw.size > self.dim:
            rows = raw.reshape(-1, self.dim)
            if self.pooling_type == 3:
                vector = rows[-1]
            elif self.pooling_type == 2:
                vector = rows[0]
            else:
                vector = rows.mean(axis=0)
        else:
            vector = raw
        norm = float(np.linalg.norm(vector))
        return vector / norm if norm else vector


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--texts", nargs="*", default=[])
    args = parser.parse_args()
    encoder = QwenEncoder()
    print(f"dim={encoder.dim}")
    texts = args.texts or [
        "Je m'appelle Théo, j'ai une allergie à la pénicilline.",
        "quelle est l'allergie de Théo ?",
        "Mon chat s'appelle Fermion et il est noir.",
    ]
    vectors = [encoder(t) for t in texts]
    for i, text in enumerate(texts):
        print(f"[{i}] {text[:52]!r} → norme {float(np.linalg.norm(vectors[i])):.4f}")
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            print(f"cos({i},{j}) = {float(vectors[i] @ vectors[j]):+.4f}")


if __name__ == "__main__":
    main()
