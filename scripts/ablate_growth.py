"""Ablation: grown colony, random schedule, frozen graph, same budget, same seed.

Three arms, one question asked in two steps. ``--arm grown`` is the comparison
run (births and pruning on, identities chosen by the traffic); ``--arm frozen``
starts from a random graph of the same size and never changes it; ``--arm
schedule`` keeps the grown arm's calendar (8 births per step) but draws the
identities at random, so it separates *routing* from *capacity*.

    python scripts/ablate_growth.py --arm grown    --steps 500
    python scripts/ablate_growth.py --arm schedule --steps 500
    python scripts/ablate_growth.py --arm frozen   --steps 500

Each arm writes its curve to ``logs_ablation_<arm>.txt``, its per-step journal to
``data/per_step_<arm>.csv``, its final genome to ``data/genome_<arm>.pt`` and its
final validation perplexity to ``ablation_<arm>.txt``. Nothing is timed for a
verdict: a loss curve does not care about the clock, only about the arithmetic.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from mycelia.checkpoint import save_genome  # noqa: E402
from mycelia.config import load_config  # noqa: E402
from mycelia.train import PER_STEP_KEYS, Trainer, per_step_line  # noqa: E402


def main() -> None:
    # Redirected stdout uses cp1252 on Windows: a single arrow in a growth
    # message would kill a 25-minute arm with UnicodeEncodeError.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass
    parser = argparse.ArgumentParser(description="grown vs frozen-random ablation")
    parser.add_argument("--arm", choices=["grown", "schedule", "frozen"], required=True)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--edges", type=int, default=3997, help="size of the frozen graph")
    parser.add_argument("--nodes", type=int, default=532, help="pool of the frozen graph")
    parser.add_argument("--device", default=None)
    parser.add_argument("--report-every", type=int, default=50)
    args = parser.parse_args()

    config = load_config()
    config["growth"]["mode"] = {
        "frozen": "frozen_random",
        "schedule": "random_schedule",
    }.get(args.arm, "grown")
    config["growth"]["frozen_edges"] = args.edges

    overrides = {}
    if args.arm == "frozen":
        # Same pool as the grown run's final state (532 slots, 514 alive).
        overrides["n_nodes"] = args.nodes

    trainer = Trainer(
        config=config,
        data="text",
        device=args.device,
        seed=args.seed,
        scale=0,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        enable_growth=args.arm != "frozen",
        enable_lifecycle=args.arm != "frozen",
        **overrides,
    )
    brain = trainer.brain
    # One journal per arm: a resumed arm must never overwrite the other's series,
    # and each arm is a fresh run, so truncation is correct here.
    per_step_path = ROOT / "data" / f"per_step_{args.arm}.csv"
    per_step_path.parent.mkdir(parents=True, exist_ok=True)
    per_step = per_step_path.open("w", encoding="utf-8", buffering=1)
    per_step.write(",".join(PER_STEP_KEYS) + "\n")

    print(
        f"arm {args.arm} | device {trainer.device} | pool {brain.n_nodes} | "
        f"arêtes {brain.ne} | hops {brain.max_hops} | seq {trainer.seq_len} | "
        f"batch {trainer.batch_size} | naissances {brain.max_births}",
        flush=True,
    )

    log_path = ROOT / f"logs_ablation_{args.arm}.txt"
    curve: list[dict] = []
    started = time.time()
    first = None
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write(f"# arm {args.arm} | steps {args.steps} | seed {args.seed}\n")
        for _ in range(args.steps):
            metrics = trainer.train_step()
            per_step.write(per_step_line(metrics))
            first = metrics["loss"] if first is None else first
            if trainer.step % args.report_every == 0 or trainer.step == 1:
                line = (
                    f"step {trainer.step:5d} | perte {metrics['loss']:.4f} | "
                    f"ppl {metrics['ppl']:8.1f} | arêtes {metrics['edges']:5d} | "
                    f"nœuds {metrics['nodes']:4d} | temps {time.time() - started:6.0f} s"
                )
                print(line, flush=True)
                handle.write(line + "\n")
                handle.flush()
                curve.append(
                    {
                        "step": trainer.step,
                        "loss": round(metrics["loss"], 4),
                        "edges": metrics["edges"],
                        "nodes": metrics["nodes"],
                    }
                )

    per_step.close()
    val_ppl = trainer.validate()
    venv = {
        "arm": args.arm,
        "steps": args.steps,
        "first_loss": round(float(first), 4),
        "final_loss": round(float(curve[-1]["loss"]), 4) if curve else None,
        "val_ppl": round(float(val_ppl), 3),
        "edges": brain.ne,
        "nodes": int(brain.node_alive.sum()),
        "pool": brain.n_nodes,
        "curve": curve,
        # Wall clock, for the cost line only: no verdict reads it.
        "seconds": round(time.time() - started),
    }
    # The genome is the evidence, not a by-product: the 8 November arm saved
    # none, and its degree distribution had to be borrowed from a twin run.
    genome_path = save_genome(
        ROOT / "data" / f"genome_{args.arm}.pt",
        brain,
        step=trainer.step,
        optimizer=trainer.optimizer,
        growth=trainer.growth,
        history=trainer.history,
        extra={"arm": args.arm, "steps": args.steps, "seed": args.seed},
    )
    venv["genome"] = str(genome_path.relative_to(ROOT))
    (ROOT / f"ablation_{args.arm}.json").write_text(
        json.dumps(venv, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (ROOT / f"ablation_{args.arm}.txt").write_text(
        f"ablation {args.arm} | perte {venv['first_loss']:.4f} -> "
        f"{venv['final_loss']:.4f} | validation perplexity {val_ppl:.3f} | "
        f"{brain.ne} arêtes | {venv['nodes']} nœuds\n",
        encoding="utf-8",
    )
    print(
        f"FIN {args.arm} | perte {venv['first_loss']:.4f} -> {venv['final_loss']:.4f} | "
        f"val ppl {val_ppl:.3f} | arêtes {brain.ne} | nœuds {venv['nodes']} | "
        f"génome {venv['genome']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
