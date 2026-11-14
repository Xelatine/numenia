"""Greffe 1 — pilote mesuré : Qwen 896-d → MYCELIA 32-d.

Seuils CONSIGNÉS AVANT le calcul (dans ce fichier, puis dans le JSON) :

  S1 — fidélité de voisinage : Spearman des similarités cosinus par paires
       (20 phrases, 190 paires) entre l'espace Qwen et l'espace greffé ≥ 0,70.
  S2 — erreur cosinus moyenne ≤ 0,15.
  S3 — retrieval : hit@1 sur 5 paires question/réponse projetées ≥ 4/5,
       et pas plus d'un point sous le hit@1 Qwen.

Deux cartographies comparées, même protocole :
  * projection aléatoire (Johnson-Lindenstrauss, graine fixe) ;
  * PCA ajustée sur 4 096 tokens du vocabulaire Qwen (hors phrases de test).

Aucun poids de MYCELIA n'est modifié ici : c'est la preuve de la projection
avant toute greffe réelle.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mycelia.gguf_reader import read_embedding_rows, read_inventory  # noqa: E402

MODEL = ROOT / "data" / "models" / "qwen2.5-0.5b-instruct-q8_0.gguf"
OUT = ROOT / "data" / "mem_report" / "greffe_pilot.json"
DIM = 32
SEED = 20261114
THRESHOLDS = {"spearman_min": 0.70, "cos_error_max": 0.15, "hit_at_1_min": 4}

SENTENCES = [
    "Je m'appelle Théo et je suis allergique à la pénicilline.",
    "Ma couleur préférée est le bleu cobalt.",
    "Je travaille au laboratoire de biologie du campus.",
    "Mon chat s'appelle Fermion et il est noir.",
    "Je déteste qu'on me coupe la parole.",
    "Le train pour Lyon part à huit heures du matin.",
    "La bibliothèque ferme exceptionnellement à dix-neuf heures.",
    "J'ai perdu mon portefeuille dans le métro hier soir.",
    "Mon frère habite à Montréal depuis trois ans.",
    "Le musée expose des peintures de la Renaissance.",
    "Il pleut depuis deux jours sans interruption.",
    "Le café du coin sert un excellent chocolat chaud.",
    "Je dois rendre mon rapport avant vendredi.",
    "Les enfants jouent dans le parc après l'école.",
    "Mon ordinateur portable a besoin d'une nouvelle batterie.",
    "La réunion est reportée à la semaine prochaine.",
    "J'apprends le piano depuis six mois.",
    "Le marché du samedi vend des légumes de saison.",
    "Cette ville est connue pour ses ponts anciens.",
    "Je préfère lire le soir avant de dormir.",
]

PAIRS = [
    ("Je suis allergique à la pénicilline.", "quelle est mon allergie ?", "pénicilline"),
    ("Ma couleur préférée est le bleu.", "quelle couleur j'aime ?", "bleu"),
    ("Je travaille dans un laboratoire.", "où est-ce que je travaille ?", "laboratoire"),
    ("Mon chat s'appelle Fermion.", "comment s'appelle mon chat ?", "Fermion"),
    ("Je déteste qu'on me coupe la parole.", "qu'est-ce qui m'énerve ?", "parole"),
]


def ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    result = np.empty_like(order, dtype=np.float64)
    result[order] = np.arange(len(values))
    return result


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra, rb = ranks(a), ranks(b)
    ra -= ra.mean()
    rb -= rb.mean()
    return float((ra @ rb) / (np.linalg.norm(ra) * np.linalg.norm(rb) + 1e-12))


def pairwise_cos(vectors: np.ndarray) -> np.ndarray:
    vectors = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-9)
    matrix = vectors @ vectors.T
    upper = np.triu_indices(len(vectors), k=1)
    return matrix[upper]


def main() -> int:
    global DIM
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, default=DIM)
    args = parser.parse_args()
    DIM = int(args.dim)

    from llama_cpp import Llama

    llm = Llama(model_path=str(MODEL), n_ctx=256, n_threads=2, verbose=False)
    metadata, tensors = read_inventory(MODEL)
    named = {t.name: t for t in tensors}
    emb = named["token_embd.weight"]
    n_vocab = int(emb.shape[1])

    rng = np.random.default_rng(SEED)
    sample = rng.choice(n_vocab, size=4096, replace=False).tolist()
    sample_vectors = read_embedding_rows(MODEL, emb, sample)
    mean = sample_vectors.mean(axis=0)
    centered = sample_vectors - mean
    _u, _s, vt = np.linalg.svd(centered, full_matrices=False)
    pca = vt[:DIM]  # [DIM, 896]
    random_projection = rng.normal(
        0.0, 1.0 / np.sqrt(emb.shape[0]), (emb.shape[0], DIM)
    ).astype(np.float32)

    def qwen_embeddings(texts: list[str]) -> np.ndarray:
        rows = []
        for text in texts:
            ids = llm.tokenize(text.encode("utf-8"), add_bos=False)[:128]
            rows.append(ids)
        flat = sorted({i for ids in rows for i in ids})
        table = dict(zip(flat, read_embedding_rows(MODEL, emb, flat)))
        vectors = []
        for ids in rows:
            matrix = np.stack([table[i] for i in ids]) if ids else np.zeros((1, emb.shape[0]))
            vectors.append(matrix.mean(axis=0))
        return np.asarray(vectors, dtype=np.float32)

    base = qwen_embeddings(SENTENCES)
    pca_vectors = (base - mean) @ pca.T
    rand_vectors = base @ random_projection

    # Troisième cartographie : PCA ajustée sur des PHRASES du corpus, pas sur
    # des tokens isolés. La structure sémantique des moyennes de phrases n'est
    # pas celle du vocabulaire ; on la mesure au lieu de la supposer.
    corpus_path = ROOT / "data" / "corpus_fr.txt"
    corpus_sentences: list[str] = []
    if corpus_path.exists():
        import re
        raw = corpus_path.read_text(encoding="utf-8", errors="replace")
        for chunk in re.split(r"(?<=[.!?])\s+", raw):
            text = " ".join(chunk.split())
            if 20 <= len(text) <= 280:
                corpus_sentences.append(text)
            if len(corpus_sentences) >= 800:
                break
    corpus_vectors = qwen_embeddings(corpus_sentences) if corpus_sentences else base
    sentence_mean = corpus_vectors.mean(axis=0)
    _su, _ss, sentence_vt = np.linalg.svd(
        corpus_vectors - sentence_mean, full_matrices=False
    )
    sentence_pca = sentence_vt[:DIM]
    sentence_vectors = (base - sentence_mean) @ sentence_pca.T

    base_cos = pairwise_cos(base)
    result = {
        "thresholds": THRESHOLDS,
        "dim_source": int(emb.shape[0]),
        "dim_target": DIM,
        "sentences": len(SENTENCES),
        "pca_fit_tokens": len(sample),
        "mappings": {},
    }
    for name, vectors in (
        ("pca", pca_vectors),
        ("random", rand_vectors),
        ("pca_sentences", sentence_vectors),
    ):
        mapped_cos = pairwise_cos(vectors)
        rho = spearman(base_cos, mapped_cos)
        error = float(np.abs(base_cos - mapped_cos).mean())
        result["mappings"][name] = {
            "spearman": round(rho, 4),
            "cos_error_mean": round(error, 4),
            "pass_s1": bool(rho >= THRESHOLDS["spearman_min"]),
            "pass_s2": bool(error <= THRESHOLDS["cos_error_max"]),
        }

    def retrieval(project) -> int:
        facts = project(qwen_embeddings([p[0] for p in PAIRS]))
        queries = project(qwen_embeddings([p[1] for p in PAIRS]))
        facts = facts / (np.linalg.norm(facts, axis=1, keepdims=True) + 1e-9)
        queries = queries / (np.linalg.norm(queries, axis=1, keepdims=True) + 1e-9)
        hits = 0
        for index, (_fact, _query, expected) in enumerate(PAIRS):
            scores = facts @ queries[index]
            best = int(np.argmax(scores))
            hits += int(expected.lower() in PAIRS[best][0].lower())
        return hits

    base_hits = retrieval(lambda x: x)
    pca_hits = retrieval(lambda x: (x - mean) @ pca.T)
    rand_hits = retrieval(lambda x: x @ random_projection)
    sent_hits = retrieval(lambda x: (x - sentence_mean) @ sentence_pca.T)
    result["retrieval"] = {
        "qwen": base_hits,
        "pca": pca_hits,
        "random": rand_hits,
        "pca_sentences": sent_hits,
        "pass_pca": bool(
            pca_hits >= THRESHOLDS["hit_at_1_min"] and pca_hits >= base_hits - 1
        ),
    }
    result["verdict"] = (
        "VERT" if (
            result["mappings"]["pca"]["pass_s1"]
            and result["mappings"]["pca"]["pass_s2"]
            and result["retrieval"]["pass_pca"]
        ) else "ROUGE"
    )
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print("écrit", OUT)
    return 0 if result["verdict"] == "VERT" else 1


if __name__ == "__main__":
    raise SystemExit(main())
