"""Jumeau B2 — 10 020 nœuds, 10 % éveillés PAR ENTRÉE (session W clarifiée).

Vision : la taille ne coûte plus cher. Dix fois la capacité du dense 1 002,
mais le routeur n'éveille que ~10 % des nœuds pour chaque token (≈ 1 002), donc
le même effort de calcul qu'aujourd'hui. Les autres sont dormants, pas morts.

Seuils consignés AVANT :
  ÉTAGE 1 : 3 000 pas, perte finale ≤ 0,002342 (référence dense 1 002).
  ÉTAGE 2 (si vert) : inférence à 10 / 5 / 2 % éveillé, dégradation constatée.

Mesures obligatoires :
  * HWM mesuré, et empreinte paramètres calculée (pas lue sur une fiche) ;
  * couverture : chaque nœud éveillé au moins une fois au smoke 100 pas ;
  * stabilité : concentration du routage (part des 200 nœuds les plus choisis).

Aucun anneau 30 %→10 % : la couverture est mesurée suffisante sans lui, et un
anneau changerait le coût moyen par pas — donc le protocole. Si l'étage 1
échoue, c'est un échec de la piste B2, pas un prétexte à réglage.

Usage :
    .venv/bin/python scripts/run_twin_10020.py --mode smoke
    setsid .venv/bin/python -u scripts/run_twin_10020.py --mode run
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

NODES = 10_020
REF_LOSS = 0.002342
CEILING = REF_LOSS
FRACTION = 0.10
REPORT = ROOT / "data" / "mem_report"


def hwm_mb() -> float:
    with open("/proc/self/status", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("VmHWM"):
                return int(line.split()[1]) / 1024.0
    return 0.0


def param_mb(module) -> float:
    return sum(p.numel() for p in module.parameters()) * 4 / 1048576.0


def build_trainer(seed: int = 42) -> Trainer:
    trainer = Trainer(
        config=load_config(), data="toy", device="cpu", scale=4,
        batch_size=1, seq_len=32, seed=seed,
        n_nodes=NODES, max_node_births=0,
        route_fraction=FRACTION,
        checkpoint_turns=True,
    )
    trainer.brain.checkpoint_turns = True
    trainer.report_every = 100
    return trainer


def resume_trainer(path: str | Path) -> Trainer:
    """Reprend un run interrompu depuis son checkpoint.

    Jamais de zéro : ``Trainer.resume`` recharge le cerveau, l'optimiseur, la
    croissance, l'historique et les RNG, et loggue « reprise à step X » (cf.
    mycelia/train.py). Le checkpoint de run est écrit tous les 500 pas.
    """
    trainer = Trainer.resume(
        path,
        config=load_config(),
        device="cpu",
        batch_size=1,
        seq_len=32,
        route_fraction=FRACTION,
        checkpoint_turns=True,
    )
    trainer.brain.checkpoint_turns = True
    trainer.report_every = 100
    return trainer


def routing_concentration(brain) -> dict:
    """Le piège MoE : tout envoyer aux mêmes favoris. On le mesure."""
    seen = brain.awake_seen
    top = torch.topk(seen.to(torch.int64), min(200, NODES)).values
    return {
        "nodes_ever_awake": int(seen.sum()),
        "top200_share_of_awake_events": round(float(top.sum()) / max(1, int(seen.sum())), 4),
        "never_awake": int((~seen).sum()),
    }


def stage2(brain) -> list[dict]:
    brain.eval()
    generator = torch.Generator().manual_seed(7)
    batches = [get_batch(4, 32, generator=generator) for _ in range(4)]
    rows = []
    baseline = None
    for fraction in (0.10, 0.05, 0.02):
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
    parser.add_argument(
        "--resume", default=None,
        help="checkpoint de run à reprendre (jamais de zéro si présent)",
    )
    args = parser.parse_args(argv)

    steps = args.steps or (100 if args.mode == "smoke" else 3000)
    checkpoint = ROOT / "data" / f"genome_twin10020_{args.mode}.pt"
    report_path = REPORT / ("smoke_twin10020.json" if args.mode == "smoke" else "twin10020.json")
    started = time.perf_counter()
    trainer = resume_trainer(args.resume) if args.resume else build_trainer(seed=args.seed)
    if args.resume:
        print(f"REPRISE | {args.resume} | step {trainer.step} -> {steps}", flush=True)
    print(
        f"JUMEAU 10020 | mode {args.mode} | {NODES} nœuds | route {FRACTION:.0%}"
        f" | batch 1 | seq 32 | seuil {CEILING:.6f} | checkpoint {checkpoint}",
        flush=True,
    )
    print(
        f"PARAMS | {param_mb(trainer.brain):.1f} Mo fp32"
        f" | avant run RSS {process_memory()['working_set']:.0f} Mo",
        flush=True,
    )

    last = {"step": 0, "time": started}
    rows: list[dict] = []

    def callback(t: Trainer, metrics: dict) -> None:
        if t.step % 20 == 0:
            now = time.perf_counter()
            pace = (now - last["time"]) / max(1, t.step - last["step"])
            print(
                f"PACE {t.step:5d} | {pace:5.2f} s/pas | RSS {process_memory()['working_set']:6.0f} Mo"
                f" | HWM {hwm_mb():6.0f} Mo | arêtes {t.brain.ne:5d}"
                f" | éveillés {t.brain.last_awake_nodes:4d}"
                f" | couverts {int(t.brain.awake_seen.sum()):5d}/{NODES}"
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
                "rss_mb": round(process_memory()["working_set"], 1),
                "hwm_mb": round(hwm_mb(), 1),
            })
        if t.step % 500 == 0:
            t.save(checkpoint)
            print(f"CHECKPOINT step {t.step} -> {checkpoint}", flush=True)

    remaining = max(0, steps - trainer.step)
    trainer.run(remaining, callback=callback)
    trainer.save(checkpoint)

    final_loss = float(trainer.history[-1]["loss"]) if trainer.history else float("nan")
    routing = routing_concentration(trainer.brain)
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
        "param_mb_fp32": round(param_mb(trainer.brain), 1),
        "routing": routing,
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
            else "non exécuté : étage 1 échoué (piste B2)"
        )
    else:
        payload["routing_ok"] = routing["never_awake"] == 0
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if args.mode == "run":
        print(
            "ETAGE1", "VERT" if payload["stage1_admissible"] else "ECHEC",
            f"perte {final_loss:.6f} | HWM {payload['hwm_mb']:.0f} Mo"
            f" | couverture {routing['nodes_ever_awake']}/{NODES}", flush=True,
        )
    else:
        print(
            f"SMOKE couverture {routing['nodes_ever_awake']}/{NODES}"
            f" | jamais éveillés {routing['never_awake']}"
            f" | top200 {routing['top200_share_of_awake_events']:.3f}"
            f" | HWM {payload['hwm_mb']:.0f} Mo | écrit {report_path}", flush=True,
        )
    print("écrit", report_path, flush=True)
    return 0 if routing["never_awake"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
