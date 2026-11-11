"""Mesure la dérive mémoire pas à pas, et rien d'autre.

Pourquoi ce fichier : ``bench_step.py`` donne une dérive moyenne par pas, ce qui
ne distingue pas trois choses très différentes :

1. une vraie fuite (croissance linéaire qui ne s'arrête jamais) ;
2. un échauffement d'allocateur (la première dizaine de pas réserve, puis plus
   rien) ;
3. une croissance par paliers (checkpoints, historiques, caches périodiques).

La sonde relève la mémoire toutes les ``--every`` étapes et imprime la série,
pour que la forme de la courbe réponde à la question au lieu de la moyenne.
C'est une mesure seule : elle ne modifie rien et n'écrit aucun fichier.

Usage :

    python scripts/drift_probe.py --scale 4 --batch 2 --seq 16 --steps 60
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mycelia.config import load_config  # noqa: E402
from mycelia.train import Trainer, process_memory  # noqa: E402


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="memory drift, step by step")
    parser.add_argument("--scale", type=int, default=4, choices=[0, 1, 2, 3, 4])
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--seq", type=int, default=16)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--every", type=int, default=10)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args(argv)

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

    config = load_config()
    trainer = Trainer(
        config=config,
        data="toy",
        device=args.device,
        scale=args.scale,
        batch_size=args.batch,
        seq_len=args.seq,
    )
    # Loin de toute période de croissance/validation : on mesure le régime.
    trainer.step = 10_007

    start = process_memory()
    print(
        f"scale {args.scale} | nœuds {trainer.brain.n_nodes} | "
        f"batch {args.batch} | seq {args.seq} | device {trainer.device}"
    )
    print(f"depart: RSS {start['working_set']:.0f} Mo | commit {start['commit']:.0f} Mo")
    print("pas | RSS | pic | commit | delta commit vs depart")

    previous = start
    for step in range(1, args.steps + 1):
        trainer.step_once()
        if step % args.every == 0 or step == 1:
            now = process_memory()
            print(
                f"{step:4d} | {now['working_set']:6.0f} | {now['peak']:6.0f} | "
                f"{now['commit']:7.0f} | {now['commit'] - start['commit']:+7.0f}"
            )
            previous = now
    _ = previous


if __name__ == "__main__":
    main()
