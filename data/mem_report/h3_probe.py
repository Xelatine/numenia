"""H3 — sonde de rétention, exécutée sur le serveur par stdin SSH.

Ne modifie rien, n'écrit rien. Mesure, à chaque pas :
  - RSS et mémoire Python tracée (tracemalloc) ;
  - longueur des structures candidates (set d'arêtes, listes de télémétrie,
    cimetière, historique) ;
  - diff tracemalloc avec compteurs et ligne d'allocation exacte.
"""

from __future__ import annotations

import gc
import sys
import tracemalloc

sys.path.insert(0, "/home/utilisateur/mycelia")

import torch  # noqa: E402

from mycelia.config import load_config  # noqa: E402
from mycelia.train import Trainer, process_memory  # noqa: E402


def mb(value: int) -> float:
    return value / 1048576.0


def main() -> None:
    tracemalloc.start(25)
    trainer = Trainer(
        config=load_config(),
        data="toy",
        device="cpu",
        scale=4,
        batch_size=2,
        seq_len=16,
    )
    trainer.step = 10_007
    brain = trainer.brain
    print(
        "brain: n_nodes", brain.n_nodes,
        "| max_edges", brain.max_edges,
        "| max_population", brain.max_population,
        "| max_births", brain.max_births,
    )

    snap1 = None
    for step in range(1, 21):
        trainer.step_once()
        if step == 5:
            gc.collect()
            snap1 = tracemalloc.take_snapshot()
            print("--- snapshot 1 pris au pas 5 ---")
        if step in (1, 2, 5, 10, 15, 20):
            current, peak = tracemalloc.get_traced_memory()
            print(
                f"pas {step:2d} | RSS {process_memory()['working_set']:6.0f} Mo"
                f" | py_cur {mb(current):6.2f} Mo | py_peak {mb(peak):6.2f} Mo"
                f" | edge_keys {len(brain._edge_keys):6d} | ne {brain.ne:6d}"
                f" | firing {len(brain.last_firing_nodes):4d}"
                f" | node_io {len(brain.last_node_io):2d}"
                f" | trace {len(brain.last_trace):2d}"
                f" | cemetery {len(brain.cemetery):4d}"
                f" | telemetry {len(brain.telemetry):2d}"
                f" | hist {len(trainer.history):3d}"
                f" | gc_objs {len(gc.get_objects())}"
            )

    gc.collect()
    snap2 = tracemalloc.take_snapshot()
    print("\n=== DIFF tracemalloc pas5 -> pas20 (rétention) ===")
    stats = snap2.compare_to(snap1, "lineno")
    for index, stat in enumerate(stats[:15]):
        frame = stat.traceback[0]
        print(
            f"{index:2d} | {mb(stat.size_diff):+8.2f} Mo"
            f" | {stat.count_diff:+8d} objets"
            f" | {frame.filename}:{frame.lineno}"
        )
        print(f"     now={mb(stat.size):.2f} Mo, {stat.count} objets")

    # Les gros postes par filename, pour voir si la croissance est hors brain.py
    print("\n=== TOP 10 par fichier ===")
    by_file = snap2.compare_to(snap1, "filename")
    for index, stat in enumerate(by_file[:10]):
        print(f"{index:2d} | {mb(stat.size_diff):+8.2f} Mo | {stat.count_diff:+8d} objets | {stat.traceback[0].filename}")


if __name__ == "__main__":
    main()
