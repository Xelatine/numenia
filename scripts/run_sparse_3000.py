"""Jumeau sparse — 3 000 pas, protocole à deux étages (session T).

Étage 1 (admissibilité) : perte finale ≤ 2 × la référence dense 0,002342.
Étage 2 (seulement si vert) : campagne top-k {50,25,10,5} % sur le cerveau
entraîné, mêmes batchs fixes que la session R.

Run détaché : journal `data/mem_report/sparse_3000.log`, checkpoint tous les
500 pas, JSON final `data/mem_report/sparse_3000.json`.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mycelia.config import load_config  # noqa: E402
from mycelia.data import get_batch  # noqa: E402
from mycelia.train import Trainer, process_memory  # noqa: E402

STEPS = 3000
REF_LOSS = 0.002342
CEILING = 2 * REF_LOSS
CHECKPOINT = ROOT / "data" / "genome_sparse_3000.pt"
OUT = ROOT / "data" / "mem_report" / "sparse_3000.json"
LOG = ROOT / "data" / "mem_report" / "sparse_3000.log"


def hwm_mb() -> float:
    with open("/proc/self/status", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("VmHWM"):
                return int(line.split()[1]) / 1024.0
    return 0.0


def topk_campaign(brain) -> list[dict]:
    brain.eval()
    generator = torch.Generator().manual_seed(7)
    batches = [get_batch(16, 32, generator=generator) for _ in range(4)]
    rows = []
    baseline = None
    for fraction in (None, 0.50, 0.25, 0.10, 0.05):
        brain.topk_fraction = fraction
        total = 0.0
        with torch.no_grad():
            for x, y in batches:
                logits, _ = brain(x, grow=False)
                total += float(F.cross_entropy(
                    logits[:, 1:].reshape(-1, brain.vocab), y[:, 1:].reshape(-1)
                ))
        loss = total / len(batches)
        baseline = baseline if baseline is not None else loss
        rows.append({"fraction": fraction, "loss": round(loss, 6),
                     "factor": round(loss / baseline, 6)})
    brain.topk_fraction = None
    return rows


def main() -> None:
    started = time.perf_counter()
    config = load_config()
    trainer = Trainer(
        config=config, data="toy", device="cpu", scale=4,
        batch_size=2, seq_len=16, seed=42,
        activation="relu", output_sparse=True, checkpoint_turns=True,
    )
    trainer.brain.checkpoint_turns = True
    trainer.report_every = 100
    print(
        f"JUMEAU SPARSE | 3000 pas | activation=relu output_sparse=True"
        f" | seuil étage 1 : {CEILING:.6f} | checkpoint {CHECKPOINT}",
        flush=True,
    )

    last = {"step": 0, "time": started}

    def callback(t: Trainer, metrics: dict) -> None:
        if t.step % 20 == 0:
            now = time.perf_counter()
            pace = (now - last["time"]) / max(1, t.step - last["step"])
            print(
                f"PACE {t.step:5d} | {pace:5.2f} s/pas | RSS {process_memory()['working_set']:6.0f} Mo"
                f" | HWM {hwm_mb():6.0f} Mo | arêtes {t.brain.ne:5d} | perte {metrics['loss']:.4f}",
                flush=True,
            )
            last["step"], last["time"] = t.step, now
        if t.step % 100 == 0:
            print(
                f"TRACE {t.step:5d} | perte {metrics['loss']:.6f} | arêtes {t.brain.ne:5d}"
                f" | nœuds {int(t.brain.node_alive.sum()):5d} | cimetière {len(t.brain.cemetery):4d}"
                f" | HWM {hwm_mb():6.0f} Mo",
                flush=True,
            )
        if t.step % 500 == 0:
            t.save(CHECKPOINT)
            print(f"CHECKPOINT step {t.step} -> {CHECKPOINT}", flush=True)

    trainer.run(STEPS, callback=callback)
    trainer.save(CHECKPOINT)
    final_loss = float(trainer.history[-1]["loss"]) if trainer.history else float("nan")
    admissible = final_loss <= CEILING
    payload = {
        "steps": STEPS,
        "activation": "relu",
        "output_sparse": True,
        "final_loss": final_loss,
        "reference_dense": REF_LOSS,
        "ceiling": CEILING,
        "stage1_admissible": bool(admissible),
        "edges": int(trainer.brain.ne),
        "nodes_alive": int(trainer.brain.node_alive.sum()),
        "cemetery": len(trainer.brain.cemetery),
        "hwm_mb": round(hwm_mb(), 1),
        "seconds": round(time.perf_counter() - started, 1),
    }
    if admissible:
        payload["topk"] = topk_campaign(trainer.brain)
    else:
        payload["topk"] = "non exécuté : étage 1 échoué (plan B)"
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("ETAGE1", "VERT" if admissible else "ECHEC", f"perte {final_loss:.6f}", flush=True)
    print("écrit", OUT, flush=True)


if __name__ == "__main__":
    main()
