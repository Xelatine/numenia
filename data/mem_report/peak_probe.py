"""Pic intra-pas : quel levier le réduit vraiment ?

Reprend le checkpoint du pas 500 et fait 20 pas, en imprimant VmHWM (record
monotone) et RSS de fin de pas. Deux modes :
  plain  : tel quel
  ckpt   : checkpoint_turns=True (recalcul des activations au backward)

Aucune sauvegarde : le checkpoint n'est pas modifié.
"""

from __future__ import annotations

import sys

sys.path.insert(0, "/home/utilisateur/mycelia")

from mycelia.config import load_config  # noqa: E402
from mycelia.train import Trainer, process_memory  # noqa: E402


def hwm_mb() -> float:
    with open("/proc/self/status", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("VmHWM"):
                return int(line.split()[1]) / 1024.0
    return 0.0


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "plain"
    trainer = Trainer.resume(
        "data/genome_run3000.pt",
        config=load_config(),
        device="cpu",
        batch_size=2,
        seq_len=16,
    )
    # Comme la CLI : le drapeau appartient au run, pas au génome.
    if mode == "ckpt":
        trainer.brain.checkpoint_turns = True
    print(
        f"mode {mode} | reprise step {trainer.step} | arêtes {trainer.brain.ne}"
        f" | checkpoint_turns {trainer.brain.checkpoint_turns}",
        flush=True,
    )
    for index in range(20):
        trainer.train_step()
        if (index + 1) % 5 == 0:
            print(
                f"  +{index + 1:2d} | RSS {process_memory()['working_set']:6.0f} Mo"
                f" | HWM {hwm_mb():6.0f} Mo | arêtes {trainer.brain.ne:5d}",
                flush=True,
            )


if __name__ == "__main__":
    main()
