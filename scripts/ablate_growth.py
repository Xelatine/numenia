"""Ablation: grown colony vs frozen random graph, same budget, same seed.

Two arms, one question. ``--arm grown`` is the comparison run (births and
pruning on); ``--arm frozen`` starts from a random graph of the same size and
never changes it. The only difference between the two is therefore whether the
topology is *found* or *drawn*.

    python scripts/ablate_growth.py --arm grown  --steps 500
    python scripts/ablate_growth.py --arm frozen --steps 500

Each arm writes its curve to ``logs_ablation_<arm>.txt`` and its final
validation perplexity to ``ablation_<arm>.txt``. Nothing is timed for a
verdict: the card may be shared, and a loss curve does not care.
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

from mycelia.config import load_config  # noqa: E402
from mycelia.train import Trainer  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="grown vs frozen-random ablation")
    parser.add_argument("--arm", choices=["grown", "frozen"], required=True)
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
    config["growth"]["mode"] = "frozen_random" if args.arm == "frozen" else "grown"
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
        enable_growth=args.arm == "grown",
        enable_lifecycle=args.arm == "grown",
        **overrides,
    )
    brain = trainer.brain
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

    val_ppl = trainer.validate()
    summary = {
        "arm": args.arm,
        "steps": args.steps,
        "first_loss": round(float(first), 4),
        "final_loss": round(float(curve[-1]["loss"]), 4) if curve else None,
        "val_ppl": round(float(val_ppl), 3),
        "edges": brain.ne,
        "nodes": int(brain.node_alive.sum()),
        "pool": brain.n_nodes,
        "curve": curve,
    }
    (ROOT / f"ablation_{args.arm}.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (ROOT / f"ablation_{args.arm}.txt").write_text(
        f"ablation {args.arm} | perte {summary['first_loss']:.4f} -> "
        f"{summary['final_loss']:.4f} | validation perplexity {val_ppl:.3f} | "
        f"{brain.ne} arêtes | {summary['nodes']} nœuds\n",
        encoding="utf-8",
    )
    print(
        f"FIN {args.arm} | perte {summary['first_loss']:.4f} -> {summary['final_loss']:.4f} | "
        f"val ppl {val_ppl:.3f} | {math.exp(0):.0f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
