"""The memory constant: delayed copy at 5, 10, 25, 50 and 100 tokens.

The toy task is a copy delayed by one token. Here the delay is the variable:
``y[t] = x[t - d]``. The colony has no positional encoding, so the only thing
that can carry a token forward is the membrane, which cools by ``state_decay``
per hop (~0.96 per token for 8 hops). The delay at which accuracy collapses
*is* the memory constant -- measured, not assumed.

Reduced regime by default (32 nodes, 200 random synapses, 6 hops, CPU) so the
whole curve runs in minutes on a machine that is short of memory: same laws,
smaller colony, and the decay constant is unchanged. The script prints accuracy
on *unseen* batches only -- the labels are rebuilt from the very tokens fed in,
so a pass beyond 25 tokens would have to be explained before it is believed.

    python scripts/memory_delay.py --delays 5 10 25 50 100 --steps 80
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from mycelia.ablation import freeze_random_graph  # noqa: E402
from mycelia.brain import Brain  # noqa: E402
from mycelia.data import FIRST_TOKEN, LAST_TOKEN, VOCAB  # noqa: E402

IGNORE = -100


def delayed_batch(batch: int, seq: int, delay: int, generator: torch.Generator):
    x = torch.randint(FIRST_TOKEN, LAST_TOKEN + 1, (batch, seq), generator=generator)
    y = torch.full_like(x, IGNORE)
    if delay < seq:
        y[:, delay:] = x[:, : seq - delay]
    return x, y


def accuracy(logits: torch.Tensor, x: torch.Tensor, delay: int) -> float:
    """"Did the colony say the token from `delay` steps ago?" -- on positions t >= d."""
    if delay >= logits.shape[1]:
        return float("nan")
    guess = logits[:, delay:].argmax(dim=-1)
    truth = x[:, : logits.shape[1] - delay]
    return float((guess == truth).float().mean())


def run_delay(args, delay: int) -> dict:
    torch.manual_seed(args.seed)
    brain = Brain(
        n_nodes=args.nodes,
        dim=args.dim,
        vocab=VOCAB,
        max_hops=args.hops,
        max_edges=max(256, args.edges * 2),
        max_births=0,
        max_node_births=0,
    ).to(args.device)
    freeze_random_graph(brain, edges=args.edges, seed=args.seed)
    optimiser = torch.optim.AdamW(brain.parameters(), lr=args.lr)
    generator = torch.Generator().manual_seed(args.seed)
    started = time.time()
    first = None
    loss_value = float("nan")
    for _ in range(args.steps):
        x, y = delayed_batch(args.batch_size, args.seq_len, delay, generator)
        logits, _ = brain(x.to(args.device), grow=False)
        loss = F.cross_entropy(
            logits[:, delay:].reshape(-1, VOCAB), y[:, delay:].reshape(-1)
        )
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(brain.parameters(), 1.0)
        optimiser.step()
        if first is None:
            first = float(loss.detach())
        loss_value = float(loss.detach())

    brain.eval()
    probe = torch.Generator().manual_seed(args.seed + 999)  # unseen batches only
    hits = []
    with torch.no_grad():
        for _ in range(5):
            x, _y = delayed_batch(args.batch_size, args.seq_len, delay, probe)
            logits, _ = brain(x.to(args.device), grow=False)
            hits.append(accuracy(logits, x, delay))
    return {
        "delay": delay,
        "steps": args.steps,
        "first_loss": round(first, 4),
        "last_loss": round(loss_value, 4),
        "accuracy": round(sum(hits) / len(hits), 4),
        "seconds": round(time.time() - started, 1),
        "edges": brain.ne,
        "nodes": brain.n_nodes,
        "dim": brain.dim,
        "hops": brain.max_hops,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="delayed copy: the memory constant")
    parser.add_argument("--delays", type=int, nargs="+", default=[5, 10, 25, 50, 100])
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--seq-len", type=int, default=110)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--nodes", type=int, default=32)
    parser.add_argument("--dim", type=int, default=16)
    parser.add_argument("--hops", type=int, default=6)
    parser.add_argument("--edges", type=int, default=200)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", type=Path, default=ROOT / "memory_delay.json")
    args = parser.parse_args()

    torch.set_num_threads(int(os.environ.get("MYCELIA_THREADS", "2")))
    print(
        f"régime réduit : {args.nodes} nœuds, dim {args.dim}, {args.hops} sauts, "
        f"{args.edges} arêtes | seq {args.seq_len} | batch {args.batch_size} | "
        f"{args.steps} pas | threads {torch.get_num_threads()}",
        flush=True,
    )
    results = []
    for delay in args.delays:
        result = run_delay(args, delay)
        results.append(result)
        print(
            f"délai {delay:3d} | perte {result['first_loss']:.3f} -> {result['last_loss']:.3f} "
            f"| exactitude {result['accuracy']:.4f} | {result['seconds']:.0f} s",
            flush=True,
        )
        args.out.write_text(
            json.dumps({"results": results, "regime": vars(args) | {"out": str(args.out)}}, indent=2),
            encoding="utf-8",
        )
    chance = 1.0 / (LAST_TOKEN - FIRST_TOKEN + 1)
    print(f"hasard = {chance:.4f} | écrit dans {args.out}")


if __name__ == "__main__":
    main()
