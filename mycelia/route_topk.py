"""Top-k STRICT sur le routeur : activer 5 % du graphe, le reste dort.

Le principe évolutif — traiter moins de choses plus pertinentes — appliqué à la
RAM : seuls les k nœuds les plus énergiques propagent. Le graphe devient
insensible à sa taille totale pour la RAM *active*, comme checkpoint_turns
l'est pour le pic intra-pas.

MESURE OBLIGATOIRE avant de s'en réjouir : le % de nœuds déjà activés par pas.
Si c'est ≤ 5 %, le gain est nul et ce module reste un garde-fou.
"""

from __future__ import annotations

import torch


def route_topk(energies: torch.Tensor, k_fraction: float = 0.05, min_k: int = 8):
    """Renvoie (masque booléen [n], k). Les k plus énergiques seulement."""
    if energies.dim() != 1:
        raise ValueError("energies doit être un tenseur 1-D")
    n = int(energies.numel())
    k = max(int(min_k), int(n * float(k_fraction)))
    k = min(k, n)
    mask = torch.zeros_like(energies, dtype=torch.bool)
    if k > 0:
        mask[torch.topk(energies, k).indices] = True
    return mask, k


def active_fraction(brain, hops: int | None = None) -> dict:
    """Mesure le régime actuel : nœuds ayant émis / nœuds vivants.

    À appeler après un forward (la télémétrie ``firing_nodes`` est vidée à
    chaque pas, donc la mesure doit être prise dans le même pas).
    """
    alive = int(brain.node_alive.sum())
    fired = len(brain.last_firing_nodes)
    return {
        "alive": alive,
        "fired": fired,
        "fraction": (fired / alive) if alive else 0.0,
        "hops": int(brain.max_hops if hops is None else hops),
    }
