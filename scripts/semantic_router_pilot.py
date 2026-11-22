"""Piste C — le routage sémantique réveille-t-il les mêmes nœuds pour des
questions proches ?

Seuils CONSIGNÉS AVANT :
  P1 — Jaccard moyen des paires de paraphrases ≥ 0,50.
  P2 — ce Jaccard dépasse celui de questions sans rapport d'au moins 0,30.
  P3 — les 10 questions couvrent ensemble ≥ 30 % des nœuds (pas de routage
       qui envoie tout aux mêmes 100 favoris).

Aucun poids de NUMENIA n'est modifié : les clés du routeur sont gelées.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from mycelia.semantic_router import SemanticRouter  # noqa: E402

OUT = ROOT / "data" / "mem_report" / "semantic_router.json"
THRESHOLDS = {"paraphrase_jaccard_min": 0.50, "gap_over_unrelated_min": 0.30, "coverage_min": 0.30}

PARAPHRASES = [
    ("quelle est mon allergie ?", "à quoi suis-je allergique ?"),
    ("où est-ce que je travaille ?", "quel est mon lieu de travail ?"),
    ("comment s'appelle mon chat ?", "quel est le nom de mon chat ?"),
    ("quelle est ma couleur préférée ?", "quelle couleur j'aime ?"),
    ("qu'est-ce qui m'énerve ?", "quelle chose me déplaît le plus ?"),
]
UNRELATED = [
    "quelle heure est-il à Tokyo ?",
    "combien de pattes a une araignée ?",
    "qui a écrit Les Misérables ?",
    "comment fonctionne un moteur à explosion ?",
    "quel est le prix du bitcoin ?",
]


def main() -> int:
    from gemma_embed import GemmaEncoder

    encoder = GemmaEncoder()
    router = SemanticRouter(n_nodes=10_020, dim=encoder.dim, fraction=0.10)

    selections = []
    for first, second in PARAPHRASES:
        a = router.select(encoder(first, kind="query"))
        b = router.select(encoder(second, kind="query"))
        selections.append((a, b))
        print(f"paraphrase Jaccard {router.overlap(a, b):.3f} | {first[:32]!r}")

    para = [router.overlap(a, b) for a, b in selections]
    para_mean = float(np.mean(para))

    unrelated_pairs = []
    for index, question in enumerate(UNRELATED):
        a = router.select(encoder(question, kind="query"))
        b = router.select(encoder(PARAPHRASES[index % len(PARAPHRASES)][0], kind="query"))
        unrelated_pairs.append(router.overlap(a, b))
    unrelated_mean = float(np.mean(unrelated_pairs))

    seen = np.zeros(router.n_nodes, dtype=bool)
    for first, _second in PARAPHRASES:
        seen[router.select(encoder(first, kind="query"))] = True
    for question in UNRELATED:
        seen[router.select(encoder(question, kind="query"))] = True
    coverage = float(seen.mean())

    result = {
        "thresholds": THRESHOLDS,
        "n_nodes": router.n_nodes,
        "k": router.k,
        "paraphrase_jaccard": round(para_mean, 4),
        "unrelated_jaccard": round(unrelated_mean, 4),
        "gap": round(para_mean - unrelated_mean, 4),
        "coverage": round(coverage, 4),
        "pass_p1": bool(para_mean >= THRESHOLDS["paraphrase_jaccard_min"]),
        "pass_p2": bool(para_mean - unrelated_mean >= THRESHOLDS["gap_over_unrelated_min"]),
        "pass_p3": bool(coverage >= THRESHOLDS["coverage_min"]),
    }
    result["verdict"] = "VERT" if all((result["pass_p1"], result["pass_p2"], result["pass_p3"])) else "ROUGE"
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print("écrit", OUT)
    return 0 if result["verdict"] == "VERT" else 1


if __name__ == "__main__":
    raise SystemExit(main())
