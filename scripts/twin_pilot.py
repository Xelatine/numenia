"""Jumeau court dense vs sparse : pilote, pas les 3 000 pas de la session S.

Deux cerveaux neufs, même tâche jouet, même durée (50 pas), même seed :
- dense  : GELU (défaut historique)
- sparse : ReLU + ReLU de sortie (zéros exacts dès la naissance)

Puis la même campagne top-k que la session R est rejouée sur les deux. La
durée est un PILOTE : elle montre si la sparsité émerge et si le top-k résiste
mieux, pas si un cerveau de 3 000 pas converge.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mycelia.config import load_config  # noqa: E402
from mycelia.data import get_batch  # noqa: E402
from mycelia.train import Trainer  # noqa: E402

OUT = ROOT / "data" / "mem_report" / "twin_pilot.json"
STEPS = 50
FRACTIONS = [None, 0.5, 0.25, 0.1, 0.05]


def zeros_and_delta(brain) -> dict:
    from scripts.diagnose_sparsity import TOKENS, temporal_delta, zero_fraction

    return {"zeros": zero_fraction(brain), "delta": temporal_delta(brain)}


def topk_campaign(brain) -> list[dict]:
    brain.eval()
    generator = torch.Generator().manual_seed(7)
    batches = [get_batch(16, 32, generator=generator) for _ in range(4)]
    rows = []
    baseline = None
    for fraction in FRACTIONS:
        brain.topk_fraction = fraction
        total = 0.0
        with torch.no_grad():
            for x, y in batches:
                logits, _ = brain(x, grow=False)
                total += float(F.cross_entropy(logits[:, 1:].reshape(-1, brain.vocab), y[:, 1:].reshape(-1)))
        loss = total / len(batches)
        baseline = baseline if baseline is not None else loss
        rows.append({"fraction": fraction, "loss": round(loss, 4),
                     "factor": round(loss / baseline, 3)})
    brain.topk_fraction = None
    return rows


def arm(label: str, config, **overrides) -> dict:
    trainer = Trainer(config=config, data="toy", device="cpu", scale=4,
                      batch_size=2, seq_len=16, seed=42, **overrides)
    trainer.run(STEPS)
    path = ROOT / "data" / f"genome_pilot_{label}.pt"
    trainer.save(path)
    brain = trainer.brain
    stats = zeros_and_delta(brain)
    mean_zero = sum(v["sorties_nulles"] for v in stats["zeros"].values()) / len(stats["zeros"])
    return {
        "label": label,
        "activation": brain.bank.activation,
        "output_sparse": brain.bank.output_sparse,
        "steps": STEPS,
        "final_loss": round(float(trainer.history[-1]["loss"] if trainer.history else 0.0), 4),
        "edges": int(brain.ne),
        "mean_zero_fraction": round(mean_zero, 4),
        "delta": stats["delta"],
        "topk": topk_campaign(brain),
        "checkpoint": str(path),
    }


def main() -> None:
    config = load_config()
    dense = arm("dense", config)
    print("dense:", dense["final_loss"], "zéros", dense["mean_zero_fraction"], flush=True)
    sparse = arm("sparse", config, activation="relu", output_sparse=True)
    print("sparse:", sparse["final_loss"], "zéros", sparse["mean_zero_fraction"], flush=True)
    payload = {"steps": STEPS, "arms": [dense, sparse],
               "note": "PILOTE 50 pas ; le jumeau 3000 pas n'a pas été exécuté"}
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    for arm_data in payload["arms"]:
        print(arm_data["label"], "top-k:", arm_data["topk"], flush=True)
    print("écrit", OUT, flush=True)


if __name__ == "__main__":
    main()
