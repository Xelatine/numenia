"""Smoke 100 pas : un cerveau né sparse subit-il le pruning comme un mort ?

La question de la session T : les 48 % de zéros d'activation peuvent faire
passer des nœuds DORMANTS pour des nœuds MORTS. On mesure, à chaque pas, les
trois populations séparément — vivants, dormants (sortie exactement nulle sur un
forward de contrôle), enterrés (cimetière) — et on refuse de confondre.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mycelia.config import load_config  # noqa: E402
from mycelia.data import get_batch  # noqa: E402
from mycelia.train import Trainer, process_memory  # noqa: E402

OUT = ROOT / "data" / "mem_report" / "smoke_sparse_100.json"
STEPS = 100


def dormant_count(brain) -> dict:
    brain.eval()
    x, _ = get_batch(4, 16, generator=torch.Generator().manual_seed(3))
    stats = []
    original = brain.bank.forward_rows

    def wrapped(wave, ids):
        out, gate = original(wave, ids)
        stats.append((len(ids), float((out == 0).float().mean())))
        return out, gate

    brain.bank.forward_rows = wrapped
    with torch.no_grad():
        brain(x, grow=False)
    brain.bank.forward_rows = original
    brain.train()
    zeros = sum(z for _, z in stats) / max(1, len(stats))
    return {"sorties_nulles": round(zeros, 4)}


def main() -> None:
    config = load_config()
    trainer = Trainer(
        config=config, data="toy", device="cpu", scale=4,
        batch_size=2, seq_len=16, seed=42,
        activation="relu", output_sparse=True, checkpoint_turns=True,
    )
    trainer.brain.checkpoint_turns = True
    rows = []

    def callback(t: Trainer, metrics: dict) -> None:
        if t.step % 20 == 0:
            alive = int(t.brain.node_alive.sum())
            dorm = dormant_count(t.brain)
            row = {
                "step": t.step,
                "loss": round(float(metrics["loss"]), 4),
                "edges": int(t.brain.ne),
                "alive": alive,
                "cemetery": len(t.brain.cemetery),
                "edge_deaths": int(t.brain.edge_deaths),
                "node_deaths": int(t.brain.node_deaths),
                "zeros": dorm["sorties_nulles"],
                "rss_mb": round(process_memory()["working_set"], 1),
            }
            rows.append(row)
            print(row, flush=True)

    trainer.run(STEPS, callback=callback)
    final = rows[-1] if rows else {}
    verdict = "compatible"
    reasons = []
    if final.get("cemetery", 0) > 50:
        verdict = "conflit probable"
        reasons.append("cimetière anormalement haut")
    if final.get("alive", 0) < 900:
        verdict = "conflit probable"
        reasons.append("population vivante trop basse")
    if final.get("edges", 0) < 50:
        verdict = "conflit probable"
        reasons.append("croissance arrêtée")
    payload = {"steps": STEPS, "rows": rows, "verdict": verdict, "reasons": reasons,
               "activation": "relu", "output_sparse": True}
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("VERDICT", verdict, reasons, "| écrit", OUT, flush=True)


if __name__ == "__main__":
    main()
