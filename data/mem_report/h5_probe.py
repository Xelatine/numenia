"""H5 — vérification post-patch sur 100 pas (régime réel, prune à 10100).

Critère du protocole : dérive ~0 ou plafonnée, avec uordblks plat.
"""

from __future__ import annotations

import ctypes
import sys

sys.path.insert(0, "/home/dieux/mycelia")

from mycelia.config import load_config  # noqa: E402
from mycelia.train import Trainer, process_memory  # noqa: E402


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
        config=load_config(), data="toy", device="cpu", scale=4, batch_size=2, seq_len=16
    )
    trainer.step = 10_007
    base = None
    for step in range(1, 101):
        trainer.step_once()
        if step == 10:
            base = process_memory()["working_set"]
        if step % 10 == 0 or step == 1:
            m = mall()
            print(
                f"pas {step:3d} | RSS {process_memory()['working_set']:6.0f} Mo"
                f" | uordblks {m.uordblks / 1048576:7.1f}"
                f" | fordblks {m.fordblks / 1048576:7.1f}"
                f" | arena {m.arena / 1048576:7.1f}"
                f" | ne {trainer.brain.ne:5d}"
            )
    if base is not None:
        rss = process_memory()["working_set"]
        print(f"pente RSS pas10->100 : {(rss - base) / 90:.2f} Mo/pas")


if __name__ == "__main__":
    main()
