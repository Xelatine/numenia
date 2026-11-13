"""Mesure la config texte (batch 16, seq 128) sous trim_heap + checkpoint_turns.

RÈGLE : VmHWM ≤ 2 500 Mo, sinon réduire le batch AVANT toute ingestion.

Ce que ce script ne fait pas : il ne simule pas le forward. Il instancie le
vrai `Trainer(data="text")` — même cerveau, même TBPTT (8 blocs de 16), même
trim_heap en fin de pas — et lit la mémoire que le run consommera vraiment.
Le `checkpoint_sequential` d'un prototype antérieur était inopérant : la
protection vit dans `Brain.checkpoint_turns`, pas dans un wrapper externe.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mycelia.config import load_config  # noqa: E402
from mycelia.data import TextDataset  # noqa: E402
from mycelia.train import Trainer, process_memory  # noqa: E402

CORPUS = ROOT / "data" / "corpus_gutenberg.txt"
OUT = ROOT / "data" / "mem_report" / "text_config.json"
CEILING_MB = 2500.0


def hwm_mb() -> float:
    with open("/proc/self/status", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("VmHWM"):
                return int(line.split()[1]) / 1024.0
    return 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--seq", type=int, default=128)
    parser.add_argument("--checkpoint-turns", action="store_true", default=True)
    parser.add_argument("--no-checkpoint-turns", dest="checkpoint_turns", action="store_false")
    args = parser.parse_args()

    if not CORPUS.exists():
        raise SystemExit(f"corpus absent : {CORPUS}")
    config = load_config()
    dataset = TextDataset(corpus_path=CORPUS, seq_len=args.seq)
    trainer = Trainer(
        config=config,
        data="text",
        device="cpu",
        scale=4,
        batch_size=args.batch,
        seq_len=args.seq,
        dataset=dataset,
        seed=42,
        checkpoint_turns=bool(args.checkpoint_turns),
    )
    trainer.brain.checkpoint_turns = bool(args.checkpoint_turns)
    print(
        f"[0] nœuds={trainer.brain.n_nodes} vocab={trainer.brain.vocab}"
        f" seq={args.seq} batch={args.batch} tbptt={trainer.tbptt}"
        f" ckpt_turns={trainer.brain.checkpoint_turns} trim=on"
        f" | dataset {dataset.train.numel()} tokens train",
        flush=True,
    )

    started = time.perf_counter()
    rows: list[dict] = []

    def callback(t: Trainer, metrics: dict) -> None:
        now = time.perf_counter()
        elapsed = now - started
        rss = process_memory()["working_set"]
        hwm = hwm_mb()
        rows.append(
            {
                "step": t.step,
                "loss": float(metrics["loss"]),
                "rss_mb": round(rss, 1),
                "hwm_mb": round(hwm, 1),
                "edges": int(t.brain.ne),
                "seconds": round(elapsed, 2),
            }
        )
        print(
            f"[{t.step:3d}] perte={metrics['loss']:.4f} RSS={rss:6.0f} Mo"
            f" HWM={hwm:6.0f} Mo arêtes={t.brain.ne:5d}"
            f" | {elapsed / max(1, t.step):5.1f} s/pas cumulé",
            flush=True,
        )

    trainer.run(args.steps, callback=callback)
    hwm = hwm_mb()
    verdict = "AUTORISÉ (batch %d)" % args.batch if hwm <= CEILING_MB else "RÉDUIRE LE BATCH"
    summary = {
        "batch": args.batch,
        "seq": args.seq,
        "tbptt": trainer.tbptt,
        "checkpoint_turns": bool(trainer.brain.checkpoint_turns),
        "steps": args.steps,
        "vmhwm_mb": round(hwm, 1),
        "seuil_mb": CEILING_MB,
        "verdict": verdict,
        "nodes": int(trainer.brain.n_nodes),
        "vocab": int(trainer.brain.vocab),
        "rows": rows,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"FIN HWM={hwm:.0f} Mo — seuil {CEILING_MB:.0f} — {verdict}")
    print(f"écrit {OUT}")
    if hwm > CEILING_MB:
        sys.exit(2)


if __name__ == "__main__":
    main()
