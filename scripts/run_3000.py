"""Run 3 000 pas — validation du patch malloc_trim à l'échelle longue.

Réutilise Trainer.run() (mêmes pas, mêmes checkpoints de Trainer) et journalise,
tous les 100 pas, le RSS et l'état du tas glibc (uordblks / fordblks / arena).
Checkpoint du génome tous les 500 pas dans un fichier dédié, pour ne pas
écraser `data/genome.pt`.

Usage (serveur, sous screen) :

    .venv/bin/python scripts/run_3000.py
"""

from __future__ import annotations

import ctypes
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mycelia.config import load_config  # noqa: E402
from mycelia.train import Trainer, process_memory  # noqa: E402

CHECKPOINT = ROOT / "data" / "genome_run3000.pt"
STEPS = 3000


class MallInfo2(ctypes.Structure):
    _fields_ = [
        ("arena", ctypes.c_size_t),
        ("ordblks", ctypes.c_size_t),
        ("smblks", ctypes.c_size_t),
        ("hblks", ctypes.c_size_t),
        ("hblkhd", ctypes.c_size_t),
        ("usmblks", ctypes.c_size_t),
        ("fsmblks", ctypes.c_size_t),
        ("uordblks", ctypes.c_size_t),
        ("fordblks", ctypes.c_size_t),
        ("keepcost", ctypes.c_size_t),
    ]


def mall() -> MallInfo2:
    libc = ctypes.CDLL("libc.so.6")
    libc.mallinfo2.restype = MallInfo2
    return libc.mallinfo2()


def main() -> None:
    trainer = Trainer(
        config=load_config(),
        data="toy",
        device="cpu",
        scale=4,
        batch_size=2,
        seq_len=16,
        seed=42,
    )
    trainer.report_every = 100
    brain = trainer.brain
    print(
        f"RUN 3000 | scale 4 ({brain.n_nodes} nœuds) | batch 2 | seq 16 | toy"
        f" | max_edges {brain.max_edges} | patch malloc_trim actif",
        flush=True,
    )

    def callback(t: Trainer, metrics: dict) -> None:
        if t.step % 100 == 0:
            m = mall()
            rss = process_memory()["working_set"]
            print(
                f"TRACE {t.step:5d} | RSS {rss:6.0f} Mo"
                f" | uordblks {m.uordblks / 1048576:7.1f}"
                f" | fordblks {m.fordblks / 1048576:7.1f}"
                f" | arena {m.arena / 1048576:7.1f}"
                f" | arêtes {t.brain.ne:5d} | cimetière {len(t.brain.cemetery):4d}"
                f" | perte {metrics['loss']:.4f}",
                flush=True,
            )
        if t.step % 500 == 0:
            t.save(CHECKPOINT)
            print(f"CHECKPOINT step {t.step} -> {CHECKPOINT}", flush=True)

    trainer.run(STEPS, callback=callback)
    trainer.save(CHECKPOINT)
    print(f"FIN step {trainer.step} | {CHECKPOINT}", flush=True)


if __name__ == "__main__":
    main()
