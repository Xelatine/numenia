"""Diagnostic session S : activation, complexité du forward, delta temporel.

Aucun changement de modèle ici. On mesure ce qui existe :

1. les fonctions d'activation réellement appliquées ;
2. la fraction de sorties EXACTEMENT nulles par hop (le gate à 0,1 en produit) ;
3. la complexité du forward : temps vs nombre d'arêtes, à N constant ;
4. la sparsité temporelle : |état(t) - état(t-1)| / |état(t-1)| par nœud.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mycelia.brain import Brain  # noqa: E402
from mycelia.checkpoint import brain_from_genome, load_genome  # noqa: E402

OUT = ROOT / "data" / "mem_report" / "sparsity_diagnosis.json"
TOKENS = torch.tensor([[2, 5, 9, 14, 3, 7, 11, 4, 8, 6, 12, 15, 2, 9, 5, 3]] * 4)


def activation_inventory() -> dict:
    return {
        "gate": "sigmoid(gate_w·x + b), seuil dur GATE_THRESHOLD=0.1 -> zéros exacts",
        "body": "GELU (lisse, jamais exactement zéro)",
        "router_score": "sigmoid(q·k/sqrt(dim))",
        "readout": "LayerNorm + Linear",
        "conclusion": "des zéros exacts existent déjà (gate), mais la couche body reste lisse",
    }


def zero_fraction(brain: Brain) -> dict:
    """Fraction de sorties nulles et de gates fermés, par hop."""
    stats: list[tuple[int, float, float]] = []
    original = brain.bank.forward_rows

    def wrapped(x, ids):
        out, gate = original(x, ids)
        stats.append((len(ids), float((out == 0).float().mean()), float((gate < 0.1).float().mean())))
        return out, gate

    brain.bank.forward_rows = wrapped
    with torch.no_grad():
        brain(TOKENS, grow=False)
    brain.bank.forward_rows = original
    by_hop: dict[int, list[tuple[float, float]]] = {}
    # 16 positions x 8 hops : on agrège par hop en supposant l'ordre du forward.
    per_position = brain.max_hops
    for index, (_n, zeros, closed) in enumerate(stats):
        hop = index % per_position
        by_hop.setdefault(hop, []).append((zeros, closed))
    return {
        hop: {
            "sorties_nulles": round(sum(z for z, _ in vals) / len(vals), 4),
            "gates_fermes": round(sum(c for _, c in vals) / len(vals), 4),
        }
        for hop, vals in sorted(by_hop.items())
    }


def random_brain(n: int, edges: int, seed: int = 0) -> Brain:
    brain = Brain(n_nodes=n, dim=32, vocab=100, max_hops=8, max_edges=max(20_000, edges * 2),
                  max_births=0, max_node_births=0)
    generator = torch.Generator().manual_seed(seed)
    src = torch.randint(0, n, (edges,), generator=generator)
    tgt = torch.randint(0, n, (edges,), generator=generator)
    brain.edge_index[0, :edges] = src
    brain.edge_index[1, :edges] = tgt
    brain.ne = edges
    brain._rebuild_edge_keys()
    return brain


def complexity() -> list[dict]:
    rows = []
    for edges in (3_123, 6_246, 12_492, 24_984):
        brain = random_brain(1_129, edges)
        brain.eval()
        with torch.no_grad():
            brain(TOKENS[:, :8], grow=False)  # chauffe
            started = time.perf_counter()
            brain(TOKENS[:, :8], grow=False)
            elapsed = time.perf_counter() - started
        rows.append({"noeuds": 1_129, "aretes": edges, "forward_s": round(elapsed, 4)})
        print(f"complexité: N=1129 arêtes={edges} -> {elapsed:.4f} s")
    base = rows[1]["forward_s"]
    for row in rows:
        row["ratio_vs_6246"] = round(row["forward_s"] / base, 3)
    return rows


def temporal_delta(brain: Brain) -> dict:
    """État par position, puis delta relatif par nœud."""
    brain.eval()
    state = None
    states = []
    with torch.no_grad():
        for position in range(TOKENS.shape[1]):
            _logits, state, _trace = brain(TOKENS[:, position:position + 1], state=state,
                                           grow=False, return_state=True)
            states.append(state.detach().clone())
    results = {}
    previous = states[0]
    for epsilon in (1e-3, 1e-2):
        fractions = []
        deltas_all = []
        for current in states[1:]:
            delta = (current - previous).abs().sum(dim=(1, 2)) / (
                previous.abs().sum(dim=(1, 2)) + 1e-8
            )
            deltas_all.append(delta)
            fractions.append(float((delta > epsilon).float().mean()))
            previous = current
        results[str(epsilon)] = {
            "fraction_moyenne_changeante": round(sum(fractions) / len(fractions), 4),
            "delta_median": round(float(torch.cat(deltas_all).median()), 5),
        }
    return results


def main() -> None:
    brain = brain_from_genome(load_genome(ROOT / "data" / "genome_run3000.pt"))
    payload = {
        "activation": activation_inventory(),
        "zeros_par_hop": zero_fraction(brain),
        "complexite": complexity(),
        "delta_temporel": temporal_delta(brain),
        "genome": "data/genome_run3000.pt",
        "note": "delta calcule sur l'etat complet [N, batch, dim], par position",
    }
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("zéros par hop:", payload["zeros_par_hop"])
    print("delta temporel:", payload["delta_temporel"])
    print("écrit", OUT)


if __name__ == "__main__":
    main()
