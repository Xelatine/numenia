"""Preuve du format genome.myc/ et profil d'activité par hop/position.

Deux mesures, un seul cerveau (le génome réel du run 3 000, 1 129 nœuds) :

1. équité : forward .pt vs forward .myc, mêmes tokens → mêmes logits (atol 1e-3) ;
2. activité : pour chaque position et chaque hop, combien de nœuds émettent.
   La session Q n'avait que la fraction cumulée (0,999) ; celle-ci est la
   mesure par hop qui dira si le top-k à 5 % a réellement un levier.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mycelia.checkpoint import brain_from_genome, load_genome  # noqa: E402
from mycelia.genome_io import equity, load_myc, save_myc  # noqa: E402

CHECKPOINT = ROOT / "data" / "genome_run3000.pt"
OUT = ROOT / "data" / "mem_report" / "genome_format.json"
PROFILE = ROOT / "data" / "mem_report" / "active_fraction.json"


def main() -> None:
    blob = load_genome(CHECKPOINT)
    brain = brain_from_genome(blob)
    tokens = torch.tensor([[2, 5, 9, 14, 3, 7, 11, 4, 8, 6, 12, 15, 2, 9, 5, 3]] * 4)

    import numpy as np

    formats = {}
    for dtype, label in ((np.float16, "fp16"), (np.float32, "fp32")):
        target = ROOT / "data" / ("genome_myc" if label == "fp32" else "genome_myc_fp16")
        info = save_myc(target, brain, source=CHECKPOINT, weight_dtype=dtype)
        reloaded, meta = load_myc(target)
        formats[label] = {"info": info, "equity": equity(brain, reloaded, tokens)}
    info = formats["fp32"]["info"]
    result = formats["fp32"]["equity"]

    # Profil par hop/position sur le cerveau original.
    brain.profile_active = True
    brain.active_profile = []
    with torch.no_grad():
        _ = brain(tokens, grow=False)
    rows = brain.active_profile
    by_hop: dict[int, list[float]] = {}
    for _position, hop, count in rows:
        by_hop.setdefault(int(hop), []).append(count / max(1, brain.n_nodes))
    summary = {
        hop: {
            "positions": len(values),
            "mean_fraction": round(sum(values) / len(values), 4),
            "max_fraction": round(max(values), 4),
            "min_fraction": round(min(values), 4),
        }
        for hop, values in sorted(by_hop.items())
    }
    PROFILE.write_text(json.dumps({
        "genome": str(CHECKPOINT),
        "n_nodes": int(brain.n_nodes),
        "ne": int(brain.ne),
        "positions": int(tokens.shape[1]),
        "hops_per_position": int(brain.max_hops),
        "by_hop": summary,
        "note": "fraction = noeuds emetteurs / noeuds du pool, par hop et par position",
    }, indent=2), encoding="utf-8")

    payload = {
        "genome": str(CHECKPOINT),
        "myc": info,
        "equity": result,
        "formats": formats,
        "profile": summary,
    }
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"format {meta['format']} | nœuds {meta['n_nodes']} | arêtes {meta['ne']}")
    print(f"hash global {info['hash_global']} | taille {info['bytes']} o")
    print(f"EQUITE max_abs={result['max_abs']:.3e} mean={result['mean_abs']:.3e} ok={result['ok']}")
    for hop, stats in summary.items():
        print(f"hop {hop}: moyenne {stats['mean_fraction']:.3f} | max {stats['max_fraction']:.3f}")
    print(f"écrit {OUT} et {PROFILE}")
    if not result["ok"]:
        raise SystemExit("FORMAT REJETÉ : l'équité forward n'est pas respectée")


if __name__ == "__main__":
    main()
