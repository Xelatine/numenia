"""H4b — tester les deux leviers anti-fragmentation, mesurés.

Modes (env) :
  TRIM=1        -> libc.malloc_trim(0) après chaque pas
  (externe)     -> MALLOC_MMAP_THRESHOLD_=131072 avant lancement

Imprime RSS, uordblks/fordblks/arena, et le temps moyen par pas.
"""

from __future__ import annotations

import ctypes
import os
import sys
import time

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
    trim = os.environ.get("TRIM") == "1"
    libc = ctypes.CDLL("libc.so.6")
    print(
        f"mode trim={trim} | MALLOC_MMAP_THRESHOLD_={os.environ.get('MALLOC_MMAP_THRESHOLD_', 'defaut')}"
        f" | MALLOC_ARENA_MAX={os.environ.get('MALLOC_ARENA_MAX', 'defaut')}"
    )
    trainer = Trainer(
        config=load_config(), data="toy", device="cpu", scale=4, batch_size=2, seq_len=16
    )
    trainer.step = 10_007
    base = None
    elapsed = []
    for step in range(1, 21):
        tick = time.perf_counter()
        trainer.step_once()
        if trim:
            libc.malloc_trim(0)
        elapsed.append(time.perf_counter() - tick)
        if step == 5:
            base = process_memory()["working_set"]
        if step in (5, 10, 15, 20):
            m = mall()
            print(
                f"pas {step:2d} | RSS {process_memory()['working_set']:6.0f} Mo"
                f" | uordblks {m.uordblks / 1048576:7.1f}"
                f" | fordblks {m.fordblks / 1048576:7.1f}"
                f" | arena {m.arena / 1048576:7.1f}"
                f" | pas {1000 * sum(elapsed[-5:]) / 5:6.0f} ms"
            )
    if base is not None:
        rss = process_memory()["working_set"]
        print(f"pente RSS pas5->20 : {(rss - base) / 15:.2f} Mo/pas")


if __name__ == "__main__":
    main()
