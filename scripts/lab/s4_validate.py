"""S4 R3 — le protocole de validation de la lecture reparee (par bras).

Meme corpus que S3-BIS (1000 phrases) **et** le corpus 10k deja genere, trois
semences, deux variantes de tete :

* ``pleine``        — l'agregat des 8 membranes + le noeud 1 (voie lente seule) ;
* ``deux_vitesses`` — idem + la voie rapide (embedding du token precedent).

Toutes les mesures sont donnees deux fois :

* ``honnete`` — la cible du cerveau (le token qu'il predit), comparable a la
  tache du dense et aux planchers n-grammes ;
* ``decalee`` — la metrique publiee de S2/S3-BIS (token suivant), pour la
  continuite des chiffres 5,1509 / 3,7921.

Le point de comparaison est ecrit ici une fois pour toutes : bigramme 3,0466,
dense 3,7921, ancien NUMENIA 5,1509 (mesure decalee ; la mesure honnete de
l'ancien est 5,1484, donc la conclusion de S3-BIS ne bouge pas).

Le script tourne **par bras** (un corpus + un mode = 3 semences, ~7 min sur CPU)
et ecrit un fragment ; ``--merge`` assemble les fragments et rend le verdict.
C'est le decoupage qui garde chaque experience en circuit court.

    python scripts/lab/s4_validate.py --arm 1000:deux_vitesses
    python scripts/lab/s4_validate.py --merge
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from lab_common import LAB, ToyCorpus, ensure_lab, read_json, write_json
from repaired_mycelia import (
    SEQ,
    build_repaired,
    readout_accounting,
    save_brain,
    train_repaired,
)
from twins import MYCELIA_BATCH, MYCELIA_LR, MYCELIA_OVERRIDES, MYCELIA_WD

STEPS = 600
SEEDS = (1, 2, 3)
MODES = ("pleine", "deux_vitesses")
CORPORA = {
    "1000": {"path": None, "val": 58},
    "10k": {"path": "corpus_toy10k.txt", "val": 100},
}
REFERENCE = {
    "bigramme": 3.0466,
    "dense": 3.7921,
    "ancien_mycelia": 5.1509,
    "ancien_mycelia_honnete": 5.1484,
    "unigramme": 5.0829,
}


def load_corpus(name: str) -> ToyCorpus:
    entry = CORPORA[name]
    path = None if entry["path"] is None else LAB / entry["path"]
    return ToyCorpus(seq_len=SEQ, path=path)


def fragment_path(corpus: str, mode: str) -> Path:
    return LAB / f"s4_val_{corpus}_{mode}.json"


def run_arm(corpus_name: str, mode: str, seeds=SEEDS) -> dict:
    corpus = load_corpus(corpus_name)
    val = corpus.fixed(CORPORA[corpus_name]["val"], "val")
    rows = []
    path = fragment_path(corpus_name, mode)
    if path.exists():
        # Reprise : on garde les semences deja mesurees, on ne refait que les
        # autres (chaque bras doit tenir dans un appel de quelques minutes).
        rows = [r for r in read_json(path)["lignes"] if int(r["semence"]) not in set(seeds)]
    for seed in seeds:
        brain = build_repaired(
            corpus.vocab_size, seed, two_speeds=mode == "deux_vitesses",
            overrides=MYCELIA_OVERRIDES,
        )
        account = readout_accounting(brain)
        started = time.time()
        result = train_repaired(
            brain, corpus, STEPS, lr=MYCELIA_LR, batch=MYCELIA_BATCH, seed=seed,
            log_every=50, weight_decay=MYCELIA_WD, val_batches=val,
        )
        brain = result["brain"]
        save_brain(LAB / f"s4_{corpus_name}_{mode}_s{seed}.pt", brain,
                   step=result["best_step"], extra=account)
        row = {
            "corpus": corpus_name,
            "mode": mode,
            "semence": seed,
            "honnete": result["best_val"],
            "decalee": result["best_val_s3bis"],
            "derniere": result["final_val"],
            "pas": result["best_step"],
            "positions": result["per_position"],
            "parametres_ajoutes": account["ajoutes"],
            "parametres_total": account["total_modele"],
            "aretes": result["edges"],
            "secondes": round(time.time() - started, 1),
            "courbe": [{"step": r["step"], "val": r["val"]} for r in result["curve"]],
        }
        rows.append(row)
        print(
            f"{corpus_name:4s} {mode:13s} semence {seed} : honnete {row['honnete']:.4f} "
            f"(pas {row['pas']}), decalee {row['decalee']:.4f} — {row['secondes']}s",
            flush=True,
        )
    payload = {
        "corpus": corpus_name,
        "mode": mode,
        "fenetres_validation": len(val),
        "protocole": {
            "pas": STEPS, "semences": list(SEEDS), "lr": MYCELIA_LR,
            "batch": MYCELIA_BATCH, "weight_decay": MYCELIA_WD,
            "cible": "token courant (aligned_loss)",
        },
        "lignes": sorted(rows, key=lambda r: int(r["semence"])),
    }
    write_json(path, payload)
    return payload


def merge() -> None:
    payload: dict = {
        "protocole": {
            "pas": STEPS, "semences": list(SEEDS), "modes": list(MODES),
            "corpus": list(CORPORA), "lr": MYCELIA_LR, "batch": MYCELIA_BATCH,
            "weight_decay": MYCELIA_WD, "cible": "token courant (aligned_loss)",
            "reference": REFERENCE,
        },
        "runs": {},
    }
    for name in CORPORA:
        lines = []
        for mode in MODES:
            path = fragment_path(name, mode)
            if path.exists():
                lines.extend(read_json(path)["lignes"])
        if not lines:
            continue
        block = {"fenetres_validation": CORPORA[name]["val"], "lignes": lines}
        summary = {}
        for mode in MODES:
            rows = [r for r in lines if r["mode"] == mode]
            if not rows:
                continue
            values = [float(r["honnete"]) for r in rows]
            mean = sum(values) / len(values)
            spread = 0.0 if len(values) < 2 else (
                sum((v - mean) ** 2 for v in values) / (len(values) - 1)
            ) ** 0.5
            summary[mode] = {
                "moyenne": round(mean, 4),
                "ecart_type": round(spread, 4),
                "valeurs": [round(v, 4) for v in values],
                "decalee_moyenne": round(
                    sum(float(r["decalee"]) for r in rows) / len(rows), 4
                ),
            }
        block["resume"] = summary
        payload["runs"][name] = block

    main_rows = [
        r for r in payload.get("runs", {}).get("1000", {}).get("lignes", [])
        if r["mode"] == "deux_vitesses"
    ]
    if not main_rows:
        raise SystemExit("aucun fragment '1000/deux_vitesses' : lancer le bras d'abord")
    mean_1000 = sum(float(r["honnete"]) for r in main_rows) / len(main_rows)
    if mean_1000 < REFERENCE["bigramme"]:
        verdict, phrase = "🥇", "le NUMENIA réparé bat le bigramme : l'architecture dépasse son professeur dense"
    elif mean_1000 < REFERENCE["dense"]:
        verdict, phrase = "🥈", "le NUMENIA réparé bat le dense : la distillation peut commencer"
    else:
        verdict, phrase = "🥉", "le NUMENIA réparé ne bat pas le seuil publié du dense : le goulot restant doit être diagnostiqué"
    payload["verdict"] = {
        "code": verdict,
        "phrase": phrase,
        "mycelia_repere_1000": round(mean_1000, 4),
        "seuil_bigramme": REFERENCE["bigramme"],
        "seuil_dense": REFERENCE["dense"],
    }
    write_json(LAB / "s4_validation.json", payload)
    print(f"verdict {verdict} — NUMENIA réparé (1000 phrases, moyenne) : {mean_1000:.4f}")
    for name, block in payload["runs"].items():
        for mode, row in block["resume"].items():
            print(f"  {name:4s} {mode:13s} {row['moyenne']:.4f} ± {row['ecart_type']:.4f}")
    print("ecrit : data/lab/s4_validation.json")


def main() -> None:
    ensure_lab()
    parser = argparse.ArgumentParser(description="validation S4 de la lecture reparee")
    parser.add_argument(
        "--arm", action="append",
        help="corpus:mode:semence (repetable), par exemple 1000:deux_vitesses:1",
    )
    parser.add_argument("--merge", action="store_true")
    args = parser.parse_args()
    for arm in args.arm or []:
        parts = arm.split(":")
        name, mode = parts[0], parts[1]
        seeds = (int(parts[2]),) if len(parts) > 2 else SEEDS
        run_arm(name, mode, seeds)
    if args.merge:
        merge()


if __name__ == "__main__":
    main()
