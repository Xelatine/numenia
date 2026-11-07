"""Degree distribution of the wiring, read from the checkpoints.

The routing rule (F2) makes an address a key computed from the membrane, and a
new synapse is opened where the router *wanted* one. If the rich get richer,
in-degree should concentrate: a few nodes receive almost everything. If the
medium equilibrates, the distribution stays close to Poisson with mean 2E/N.

    python scripts/degree_stats.py data/genome_comparaison.pt data/genome.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from mycelia.checkpoint import brain_from_genome, load_genome  # noqa: E402


def gini(values: torch.Tensor) -> float:
    """Gini coefficient: 0 = every node equal, 1 = one node has everything."""
    values = values.double().flatten().sort().values
    n = values.numel()
    if n == 0 or float(values.sum()) == 0.0:
        return 0.0
    index = torch.arange(1, n + 1, dtype=torch.double)
    return float((2 * (index * values).sum() / (n * values.sum())) - (n + 1) / n)


def describe(path: Path) -> dict:
    blob = load_genome(path)
    brain = brain_from_genome(blob)
    edges = brain.edge_index[:, : brain.ne]
    alive = brain.node_alive
    in_degree = torch.bincount(edges[1], minlength=brain.n_nodes).double()
    out_degree = torch.bincount(edges[0], minlength=brain.n_nodes).double()
    in_alive = in_degree[alive]
    top = max(1, int(0.01 * in_alive.numel()))
    top_share = float(in_alive.sort(descending=True).values[:top].sum() / in_alive.sum())
    return {
        "file": path.name,
        "step": int(blob.get("step", 0)),
        "nodes": int(alive.sum()),
        "pool": brain.n_nodes,
        "edges": brain.ne,
        "mean_in": round(float(in_alive.mean()), 2),
        "max_in": int(in_alive.max()),
        "max_out": int(out_degree[alive].max()),
        "orphans_in": int((in_degree[alive] == 0).sum()),
        "gini_in": round(gini(in_alive), 4),
        "top1pct_share": round(top_share, 4),
        "cv_in": round(float(in_alive.std() / in_alive.mean().clamp_min(1e-9)), 3),
        "histogram": {
            str(bucket): int(((in_alive >= bucket) & (in_alive < bucket + 5)).sum())
            for bucket in (0, 5, 10, 15, 20, 30, 50)
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="degree distribution of the colony")
    parser.add_argument("checkpoints", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, default=ROOT / "degree_stats.json")
    args = parser.parse_args()
    rows = [describe(path) for path in args.checkpoints]
    args.out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    for row in rows:
        print(
            f"{row['file']} (pas {row['step']}) | {row['edges']} arêtes sur {row['nodes']} nœuds | "
            f"degré entrant moyen {row['mean_in']}, max {row['max_in']} | "
            f"orphelins {row['orphans_in']} | Gini {row['gini_in']} | "
            f"1 % les plus riches {row['top1pct_share']:.1%} | CV {row['cv_in']}",
            flush=True,
        )
    print(f"écrit dans {args.out}")


if __name__ == "__main__":
    main()
