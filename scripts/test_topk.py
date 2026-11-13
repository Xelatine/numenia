"""Top-k strict : mesurer ce qu'il coûte AVANT d'en faire un défaut.

Deux mesures sur le graphe réellement entraîné (run 3 000, tâche jouet) :

1. perte d'entropie croisée sur des batchs fixes, à top-k décroissant ;
2. activité par hop à la fraction retenue (profil).

Le graphe jouet est utilisé parce que sa perte est basse (0,001) : une
dégradation y est visible immédiatement, alors que sur un cerveau texte neuf
(perte 9,2) vingt pas ne distinguent rien.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mycelia.checkpoint import brain_from_genome, load_genome  # noqa: E402
from mycelia.data import get_batch  # noqa: E402

OUT = ROOT / "data" / "mem_report" / "topk.json"
FRACTIONS = [None, 0.50, 0.25, 0.10, 0.05, 0.01]


def main() -> None:
    brain = brain_from_genome(load_genome(ROOT / "data" / "genome_run3000.pt"))
    brain.eval()
    generator = torch.Generator().manual_seed(7)
    batches = [get_batch(16, 32, generator=generator) for _ in range(4)]

    rows = []
    baseline = None
    for fraction in FRACTIONS:
        brain.topk_fraction = fraction
        total = 0.0
        with torch.no_grad():
            for x, y in batches:
                logits, _ = brain(x, grow=False)
                logits = logits[:, 1:]
                targets = y[:, 1:].reshape(-1)
                total += float(F.cross_entropy(logits.reshape(-1, brain.vocab), targets))
        loss = total / len(batches)
        baseline = baseline if baseline is not None else loss
        rows.append({
            "fraction": fraction,
            "loss": round(loss, 6),
            "relative": round((loss - baseline) / baseline, 5),
        })
        print(f"top-k {fraction}: perte {loss:.6f} | relatif {rows[-1]['relative']:+.4f}")

    # Profil par hop à 5 % (le routeur voit tout, la propagation est restreinte).
    brain.topk_fraction = 0.05
    brain.profile_active = True
    brain.active_profile = []
    with torch.no_grad():
        brain(batches[0][0], grow=False)
    by_hop: dict[int, list[int]] = {}
    for _position, hop, count in brain.active_profile:
        by_hop.setdefault(hop, []).append(count)
    profile = {hop: round(sum(v) / len(v), 1) for hop, v in sorted(by_hop.items())}

    OUT.write_text(json.dumps({"fractions": rows, "profile_topk5": profile}, indent=2), encoding="utf-8")
    print("profil top-k 5 % (nœuds/hop):", profile)
    print("écrit", OUT)


if __name__ == "__main__":
    main()
