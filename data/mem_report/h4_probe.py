"""H4 — où vit la croissance native ? (exécuté par stdin SSH, ne modifie rien)

1. census des tenseurs vivants : nombre, storages uniques, octets totaux,
   top formes — pour distinguer rétention de tenseurs vs fragmentation ;
2. mallinfo2 : uordblks (en usage) vs fordblks (libre non rendu) ;
3. RSS/commit par pas.
"""

from __future__ import annotations

import ctypes
import gc
import sys

sys.path.insert(0, "/home/utilisateur/mycelia")

import torch  # noqa: E402

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


def tensor_census() -> tuple[int, int, float, list[tuple[str, float]]]:
    tensors = 0
    storages: dict[int, int] = {}
    shapes: dict[str, float] = {}
    for obj in gc.get_objects():
        if not isinstance(obj, torch.Tensor):
            continue
        tensors += 1
        try:
            storage = obj.untyped_storage()
        except Exception:
            continue
        storages[id(storage)] = storage.nbytes()
        key = f"{tuple(obj.shape)} {obj.dtype}"
        shapes[key] = shapes.get(key, 0.0) + obj.element_size() * obj.nelement()
    top = sorted(shapes.items(), key=lambda kv: kv[1], reverse=True)[:5]
    return tensors, len(storages), sum(storages.values()) / 1048576.0, top


def main() -> None:
    threads = torch.get_num_threads()
    print(f"torch {torch.__version__} | threads {threads} | interop {torch.get_num_interop_threads()}")
    trainer = Trainer(
        config=load_config(), data="toy", device="cpu", scale=4, batch_size=2, seq_len=16
    )
    trainer.step = 10_007
    brain = trainer.brain
    print(f"brain: n_nodes {brain.n_nodes} | max_edges {brain.max_edges} | max_pop {brain.max_population}")

    start = None
    for step in range(1, 21):
        trainer.step_once()
        if step == 5:
            gc.collect()
            start = process_memory()["working_set"]
        if step in (1, 5, 10, 15, 20):
            gc.collect()
            m = mall()
            count, unique, total_mb, top = tensor_census()
            print(
                f"pas {step:2d} | RSS {process_memory()['working_set']:6.0f} Mo"
                f" | uordblks {m.uordblks / 1048576:7.1f} Mo"
                f" | fordblks {m.fordblks / 1048576:7.1f} Mo"
                f" | arena {m.arena / 1048576:7.1f} Mo"
                f" | tenseurs {count:6d} / storages {unique:5d}"
                f" | tenseurs_vivants {total_mb:7.1f} Mo"
                f" | ne {brain.ne:5d} | edge_keys {len(brain._edge_keys):5d}"
            )
            for shape, size in top:
                print(f"        {shape}: {size / 1048576:.2f} Mo")
    if start is not None:
        print(f"\npente RSS pas5->20 : {(process_memory()['working_set'] - start) / 15:.2f} Mo/pas")


if __name__ == "__main__":
    main()
