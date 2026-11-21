"""S3-bis D2' — corriger l'agregation, re-tester (le « et si ? » du verdict BUG).

Le diagnostic D2 : la seule memoire du cerveau est une somme a decroissance
exponentielle des messages recents, sans mecanisme qui designe le **dernier**
token ; meme a 128 dims l'etat ne retrouve pas l'ordre (4,91 contre 3,05 pour le
bigramme). Si c'est bien la cause du plafond, une seule vis change tout : la
vitesse de refroidissement de la membrane (``state_decay``). Plus elle est basse,
plus le dernier token domine l'etat — au prix de la memoire longue. C'est le
compromis exact que la colonie paie aujourd'hui par defaut.

Trois valeurs, meme budget court (300 pas, semence 1, memes lots) :

* ``0.95`` — la valeur de S2 (controle) ;
* ``0.80`` — memoire d'environ 2-3 tokens ;
* ``0.50`` — memoire d'environ 1 token : le dernier mot domine.

Si la perte tombe vers 4,5 ou moins, la cause est confirmee et la S4-CORRIGE sait
quoi corriger (lecture a deux vitesses, ou attention sur la fenetre recente).
Si rien ne bouge, l'agregation n'explique pas tout et il faut chercher ailleurs.

Ecrit ``data/lab/s3bis_recency_<valeur>.json``.
"""

from __future__ import annotations

import argparse
import time

from lab_common import LAB, ToyCorpus, ensure_lab, read_json, set_seed, write_json
from micro_mycelia import n_parameters, train_mycelia
from twins import MYCELIA_BATCH, MYCELIA_LR, MYCELIA_WD, MYCELIA_OVERRIDES, SEQ, mycelia_twin

STEPS = 300
LOG_EVERY = 10
VAL_WINDOWS = 48
SEED = 1
DECAYS = (0.95, 0.80, 0.50)


def main() -> None:
    parser = argparse.ArgumentParser(description="lecture et recence (S3-bis)")
    parser.add_argument("--decay", type=float, required=True, choices=DECAYS)
    args = parser.parse_args()

    ensure_lab()
    corpus = ToyCorpus(seq_len=SEQ)
    val = corpus.fixed(VAL_WINDOWS, "val")
    overrides = dict(MYCELIA_OVERRIDES)
    overrides["state_decay"] = float(args.decay)
    set_seed(SEED)
    brain = mycelia_twin(corpus.vocab_size, SEED)
    # mycelia_twin applique MYCELIA_OVERRIDES ; on reconstruit avec la decroissance
    from micro_mycelia import build_mycelia

    brain = build_mycelia(corpus.vocab_size, SEED, overrides)
    started = time.time()
    result = train_mycelia(
        brain, corpus, STEPS, lr=MYCELIA_LR, batch=MYCELIA_BATCH, seed=SEED,
        log_every=LOG_EVERY, val_batches=val, weight_decay=MYCELIA_WD,
        patience=0, restore_best=True,
    )
    floors = read_json(LAB / "s3bis_floors.json")
    payload = {
        "state_decay": float(args.decay),
        "pas": STEPS,
        "params": int(n_parameters(result["brain"])),
        "best_val": result["best_val"],
        "best_step": result["best_step"],
        "secondes": round(time.time() - started, 1),
        "aretes": int(result["edges"]),
        "courbe": [{"step": r["step"], "val": r["val"], "loss": r["loss"]} for r in result["curve"]],
        "planchers": {
            "unigramme": floors["unigramme"]["entropie_lissee"],
            "bigramme": floors["bigramme"]["entropie_lissee"],
        },
    }
    write_json(LAB / f"s3bis_recency_{args.decay:.2f}.json", payload)
    print(f"state_decay {args.decay} : {result['best_val']:.4f} au pas {result['best_step']} "
          f"({payload['secondes']} s) — unigramme {payload['planchers']['unigramme']:.4f}")
    print(f"ecrit : data/lab/s3bis_recency_{args.decay:.2f}.json")


if __name__ == "__main__":
    main()
