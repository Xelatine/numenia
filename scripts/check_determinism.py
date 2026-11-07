"""Is a resumed run the same run on GPU? The 1e-6 promise, measured.

``test_resume_continuity`` runs on CPU and is the golden test of the logic.
This script replays the same protocol on CPU and on CUDA, then reports the
largest deviation between the uninterrupted and the resumed losses. On GPU the
answer may be worse: ``index_add_`` accumulates through atomics, whose order is
not fixed. The number printed here is the tolerance a real run should be judged
against -- an unexplained gap above it is a bug, a gap inside it is arithmetic.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from mycelia.train import Trainer  # noqa: E402

KWARGS = dict(
    data="toy",
    seed=42,
    n_nodes=16,
    max_hops=3,
    max_births=4,
    max_edges=256,
    batch_size=8,
    seq_len=10,
    enable_growth=False,
    enable_lifecycle=False,
)


def measure(device: str, tmp: Path, half: int) -> dict:
    trainer = Trainer(device=device, **KWARGS)
    trainer.brain.allow_node_growth = False
    if device == "cuda":
        torch.cuda.synchronize()
    losses = [trainer.train_step()["loss"] for _ in range(half)]
    path = trainer.save(tmp / f"genome_{device}.pt", latest=False)
    losses += [trainer.train_step()["loss"] for _ in range(half)]
    resumed = Trainer.resume(path, config=trainer.config, device=device, **KWARGS)
    after = [resumed.train_step()["loss"] for _ in range(half)]
    deviation = max(abs(a - b) for a, b in zip(losses[half:], after))
    return {
        "device": device,
        "max_deviation": deviation,
        "first_expected": losses[half],
        "first_got": after[0],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="resume continuity on cpu and gpu")
    parser.add_argument("--steps", type=int, default=50)
    args = parser.parse_args()
    tmp = ROOT / "data"
    results = []
    for device in ("cpu", "cuda"):
        if device == "cuda" and not torch.cuda.is_available():
            print("cuda indisponible : ignoré")
            continue
        result = measure(device, tmp, args.steps)
        results.append(result)
        verdict = "OK 1e-6" if result["max_deviation"] < 1e-6 else "AU-DESSUS DE 1e-6"
        print(
            f"{device}: écart max {result['max_deviation']:.3e} entre run continu et reprise "
            f"({args.steps} pas) -> {verdict}",
            flush=True,
        )
        if device == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
