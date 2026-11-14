"""Jumeau B1 — 2 004 nœuds, la moitié éveillée PAR ENTRÉE (session W).

Architecture : 2× la capacité du dense 1 002, mais un routeur appris choisit
~50 % des nœuds éligibles pour chaque token. Les autres sont dormants, pas
morts : le token suivant en réveille un autre sous-ensemble. Le coût de calcul
reste celui du dense (mêmes tokens, moitié des nœuds actifs), la capacité totale
double.

Protocole à deux étages, seuil consigné AVANT :
  ÉTAGE 1 : 3 000 pas, perte finale ≤ 0,002342 (référence dense 1 002).
  ÉTAGE 2 (si vert) : fractions {100, 50, 25, 10} %, dégradation constatée.

Smoke 100 pas : couverture (chaque nœud éveillé au moins une fois), fraction
réelle, HWM. Aucun réglage déguisé : si l'étage 1 échoue, c'est un échec de B1.

Usage :
    .venv/bin/python scripts/run_twin_2004.py --mode smoke
    setsid .venv/bin/python -u scripts/run_twin_2004.py --mode run
"""

from __future__ import annotations

import argparse
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

NODES = 2004
REF_LOSS = 0.002342
CEILING = REF_LOSS  # le VRAI test : aussi bien que le dense, pas 2× pire
FRACTION = 0.50
REPORT = ROOT / "data" / "mem_report"


def hwm_mb() -> float:
    with open("/proc/self/status", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("VmHWM"):
                return int(line.split()[1]) / 1024.0
    return 0.0


def build_trainer(seed: int = 42) -> Trainer:
    trainer = Trainer(
        config=load_config(), data="toy", device="cpu", scale=4,
        batch_size=1, seq_len=32, seed=seed,
        # 2× la capacité du dense (scale 4 = 1 000), nœuds fixes : la capacité
        # annoncée ne doit pas dériver pendant le run.
        n_nodes=NODES, max_node_births=0,
        route_fraction=FRACTION,
        checkpoint_turns=True,
    )
    trainer.brain.checkpoint_turns = True
    trainer.report_every = 100
    return trainer


def stage2(brain) -> list[dict]:
    """Où la dégradation devient visible : mêmes batchs, fractions décroissantes."""
    brain.eval()
    generator = torch.Generator().manual_seed(7)
    batches = [get_batch(4, 32, generator=generator) for _ in range(4)]
    rows = []
    baseline = None
    for fraction in (1.0, 0.50, 0.25, 0.10):
        brain.route_fraction = fraction
        total = 0.0
        awake = []
        with torch.no_grad():
            for x, y in batches:
                logits, _ = brain(x, grow=False)
                total += float(F.cross_entropy(
                    logits[:, 1:].reshape(-1, brain.vocab), y[:, 1:].reshape(-1)
                ))
                awake.append(brain.last_awake_nodes)
        loss = total / len(batches)
        baseline = baseline if baseline is not None else loss
        rows.append({
            "fraction": fraction,
            "loss": round(loss, 6),
            "factor": round(loss / baseline, 6),
            "awake_mean": round(sum(awake) / len(awake), 1),
        })
    brain.route_fraction = FRACTION
    brain.train()
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "run"), default="run")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    steps = args.steps or (100 if args.mode == "smoke" else 3000)
    checkpoint = ROOT / "data" / f"genome_twin2004_{args.mode}.pt"
    report_path = REPORT / ("smoke_twin2004.json" if args.mode == "smoke" else "twin2004.json")
    log_path = REPORT / ("smoke_twin2004.log" if args.mode == "smoke" else "twin2004.log")
    started = time.perf_counter()
    trainer = build_trainer(seed=args.seed)
    print(
        f"JUMEAU 2004 | mode {args.mode} | {NODES} nœuds | route {FRACTION:.0%}"
        f" | batch 1 | seq 32 | seuil {CEILING:.6f} | checkpoint {checkpoint}",
        flush=True,
    )

    last = {"step": 0, "time": started}
    rows: list[dict] = []

    def callback(t: Trainer, metrics: dict) -> None:
        if t.step % 20 == 0:
            now = time.perf_counter()
            pace = (now - last["time"]) / max(1, t.step - last["step"])
            seen = int(t.brain.awake_seen.sum())
            print(
                f"PACE {t.step:5d} | {pace:5.2f} s/pas | RSS {process_memory()['working_set']:6.0f} Mo"
                f" | HWM {hwm_mb():6.0f} Mo | arêtes {t.brain.ne:5d}"
                f" | éveillés {t.brain.last_awake_nodes:4d} | couverts {seen:4d}/{NODES}"
                f" | perte {metrics['loss']:.4f}",
                flush=True,
            )
            last["step"], last["time"] = t.step, now
        if t.step % 100 == 0:
            print(
                f"TRACE {t.step:5d} | perte {metrics['loss']:.6f} | arêtes {t.brain.ne:5d}"
                f" | nœuds {int(t.brain.node_alive.sum()):5d} | cimetière {len(t.brain.cemetery):4d}"
                f" | éveillés {t.brain.last_awake_nodes:4d} | HWM {hwm_mb():6.0f} Mo",
                flush=True,
            )
            rows.append({
                "step": t.step,
                "loss": round(float(metrics["loss"]), 6),
                "edges": int(t.brain.ne),
                "awake": int(t.brain.last_awake_nodes),
                "covered": int(t.brain.awake_seen.sum()),
                "hwm_mb": round(hwm_mb(), 1),
            })
        if t.step % 500 == 0:
            t.save(checkpoint)
            print(f"CHECKPOINT step {t.step} -> {checkpoint}", flush=True)

    trainer.run(steps, callback=callback)
    trainer.save(checkpoint)

    final_loss = float(trainer.history[-1]["loss"]) if trainer.history else float("nan")
    covered = int(trainer.brain.awake_seen.sum())
    never = [i for i in range(NODES) if not bool(trainer.brain.awake_seen[i])]
    payload = {
        "mode": args.mode,
        "steps": steps,
        "n_nodes": NODES,
        "route_fraction": FRACTION,
        "batch": 1,
        "seq_len": 32,
        "reference_dense": REF_LOSS,
        "ceiling": CEILING,
        "final_loss": final_loss,
        "awake_last": int(trainer.brain.last_awake_nodes),
        "covered_nodes": covered,
        "never_awake": len(never),
        "never_awake_ids": never[:20],
        "edges": int(trainer.brain.ne),
        "cemetery": len(trainer.brain.cemetery),
        "hwm_mb": round(hwm_mb(), 1),
        "seconds": round(time.perf_counter() - started, 1),
        "rows": rows,
    }
    if args.mode == "run":
        payload["stage1_admissible"] = bool(final_loss <= CEILING)
        payload["stage2"] = (
            stage2(trainer.brain) if payload["stage1_admissible"]
            else "non exécuté : étage 1 échoué (piste B1)"
        )
    else:
        payload["routing_ok"] = covered == NODES
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if args.mode == "run":
        print(
            "ETAGE1", "VERT" if payload["stage1_admissible"] else "ECHEC",
            f"perte {final_loss:.6f} | HWM {payload['hwm_mb']:.0f} Mo"
            f" | couverture {covered}/{NODES}", flush=True,
        )
    else:
        print(
            f"SMOKE couverture {covered}/{NODES} | jamais éveillés {len(never)}"
            f" | HWM {payload['hwm_mb']:.0f} Mo | écrit {report_path}", flush=True,
        )
    print("écrit", report_path, flush=True)
    return 0 if covered == NODES else 1


if __name__ == "__main__":
    sys.exit(main())
