"""Grounding minimal : chaque phrase générée est confrontée aux souvenirs.

Un claim non attesté est SIGNALÉ (incertitude), jamais censé. Contrairement à
un LLM pur, NUMENIA possède ses sources sous forme de nœuds vérifiables.

Honnêteté structurelle : ce module ne devine pas d'embedding. Si aucun
``encode`` n'est fourni, ``check_grounded`` renvoie un ratio ``None`` et des
détails ``grounded: None`` — jamais un faux vert. Le jour où l'encodeur réel du
graphe est branché, le seuil se calibre sur les golden tests ; il n'est pas
décrété ici.
"""

from __future__ import annotations

import numpy as np


def split_claims(response: str, min_chars: int = 10) -> list[str]:
    """Découpage naïf par phrases — suffisant pour v1."""
    normalized = response.replace("!", ".").replace("?", ".")
    return [chunk.strip() for chunk in normalized.split(".") if len(chunk.strip()) > min_chars]


def check_grounded(response, memory, query_emb, threshold: float = 0.55, encode=None):
    """Retourne (ratio, détails).

    ``encode`` : callable texte → vecteur, LE MÊME que celui de l'ingestion.
    Sans lui, rien n'est attesté : le ratio vaut None.
    """
    hits = memory.retrieve(query_emb, top_k=8, min_energy=0.0)
    details = []
    grounded_count = 0
    for claim in split_claims(response):
        if encode is None or not hits:
            details.append({"claim": claim[:80], "grounded": None, "sim": None})
            continue
        claim_vec = np.asarray(encode(claim), dtype=np.float32).reshape(-1)
        claim_vec = claim_vec / (np.linalg.norm(claim_vec) + 1e-9)
        sims = []
        for index, _ in hits:
            ref = memory.emb[index].astype(np.float32)
            ref = ref / (np.linalg.norm(ref) + 1e-9)
            sims.append(float(ref @ claim_vec))
        best = max(sims) if sims else 0.0
        grounded = best >= threshold
        grounded_count += int(grounded)
        details.append({"claim": claim[:80], "grounded": bool(grounded), "sim": round(best, 4)})
    if encode is None or not details:
        return None, details
    return grounded_count / len(details), details
