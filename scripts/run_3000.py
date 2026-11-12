"""Run 3 000 pas — reprise au checkpoint 500, pic maîtrisé par checkpoint_turns.

Décision du 12 novembre (session O) : le pic intra-pas scalait avec les arêtes
(+0,66 Mo/arête, VmHWM 3,70 Go à 3 101 arêtes) et la marge tombait sous 2 Go.
`checkpoint_turns=True` ramène ce pic à 633 Mo, plat sur 20 pas. On reprend le
checkpoint du pas 500 avec ce drapeau (même config : scale 4, batch 2, seq 16).

Journalise RSS, VmHWM et mallinfo2 tous les 100 pas ; checkpoint tous les 500.
"""

from __future__ import annotations

import ctypes
import sys
import time
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


def hwm_mb() -> float:
    with open("/proc/self/status", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("VmHWM"):
                return int(line.split()[1]) / 1024.0
    return 0.0


def main() -> None:
    started = time.perf_counter()
    last = {"step": None, "time": started}
    config = load_config()
    if CHECKPOINT.exists():
        trainer = Trainer.resume(
            CHECKPOINT, config=config, device="cpu", batch_size=2, seq_len=16
        )
    else:
        trainer = Trainer(
            config=config, data="toy", device="cpu", scale=4,
            batch_size=2, seq_len=16, seed=42,
        )
    # Le drapeau appartient au run, pas au génome (même geste que la CLI).
    trainer.brain.checkpoint_turns = True
    trainer.brain.checkpoint_selective = False
    trainer.report_every = 100
    remaining = max(0, STEPS - trainer.step)
    print(
        f"RUN 3000 | reprise step {trainer.step} | reste {remaining} pas"
        f" | scale 4 ({trainer.brain.n_nodes} nœuds) | batch 2 | seq 16"
        f" | max_edges {trainer.brain.max_edges} | checkpoint_turns ON",
        flush=True,
    )

    def callback(t: Trainer, metrics: dict) -> None:
        if t.step % 20 == 0:
            now = time.perf_counter()
            anchor_step = last["step"] if last["step"] is not None else t.step - 20
            anchor_time = last["time"]
            pace = 1000.0 * (now - anchor_time) / max(1, t.step - anchor_step)
            print(
                f"PACE {t.step:5d} | {pace:6.0f} ms/pas"
                f" | RSS {process_memory()['working_set']:6.0f} Mo"
                f" | HWM {hwm_mb():6.0f} Mo | arêtes {t.brain.ne:5d}",
                flush=True,
            )
            last["step"], last["time"] = t.step, now
        if t.step % 100 == 0:
            m = mall()
            print(
                f"TRACE {t.step:5d} | RSS {process_memory()['working_set']:6.0f} Mo"
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

    trainer.run(remaining, callback=callback)
    trainer.save(CHECKPOINT)
    print(f"FIN step {trainer.step} | {CHECKPOINT}", flush=True)


if __name__ == "__main__":
    main()
