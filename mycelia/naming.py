"""Le nom : choisi UNE fois, déterministe, jamais aléatoire, stable ensuite.

Chemin B de la session U : au premier Sommeil, l'organisme se nomme à partir de
sa propre structure — pas d'un tirage. La règle doit être :

* déterministe : même graphe → même nom, à chaque exécution ;
* indépendante de l'ordre des arêtes : le nom ne doit pas changer si le fichier
  est réorganisé ou compacté ;
* figée ensuite : une fois écrit dans self.json, le nom n'est jamais recalculé
  (l'ingestion change le graphe ; une identité qui dérive n'est pas une
  personnalité).

Le mécanisme retenu : signature structurelle = (nœuds vivants, arêtes, suite des
degrés triée). Un hachage stable la projette sur une table de syllabes fixe.
"""

from __future__ import annotations

import hashlib

from mycelia.identity import UNNAMED, Identity

ONSETS = ("b", "c", "d", "f", "g", "l", "m", "n", "p", "r", "s", "t", "v", "z",
          "br", "cl", "dr", "fl", "gr", "pl", "pr", "tr", "vr", "ch")
VOWELS = ("a", "e", "i", "o", "u", "an", "on", "in", "eu", "ou")
CODAS = ("", "", "", "l", "n", "r", "s", "t", "x", "m", "v", "d")


def structural_signature(n_alive: int, degrees: list[int] | tuple[int, ...]) -> str:
    """Signature indépendante de l'ordre des arêtes et des identifiants."""
    ordered = ",".join(str(int(d)) for d in sorted(degrees, reverse=True))
    payload = f"{int(n_alive)}|{len(degrees)}|{ordered}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def name_from_signature(signature: str) -> str:
    value = int(signature[:16], 16)
    onset = ONSETS[value % len(ONSETS)]
    value //= len(ONSETS)
    vowel = VOWELS[value % len(VOWELS)]
    value //= len(VOWELS)
    coda = CODAS[value % len(CODAS)]
    name = (onset + vowel + coda).capitalize()
    return name if len(name) >= 3 else name + "a"


def name_for_brain(brain) -> str:
    alive = int(brain.node_alive.sum())
    degrees = [0] * brain.n_nodes
    for index in range(int(brain.ne)):
        degrees[int(brain.edge_index[0, index])] += 1
    signature = structural_signature(alive, degrees)
    return name_from_signature(signature)


def name_at_first_sleep(identity: Identity, brain, path) -> str:
    """Nomme l'organisme s'il ne l'est pas encore ; sinon ne touche à rien."""
    if identity.name != UNNAMED:
        return identity.name
    identity.name = name_for_brain(brain)
    identity.save(path)
    return identity.name
