"""S3-bis D5 — le jumeau equitable : parametres ACTIFS egaux.

Egaliser le nombre total de parametres (S2, 23 748 des deux cotes) ne compare pas
le meme travail : le dense consulte **tous** ses parametres a chaque token, alors
que MYCELIA n'en touche qu'une partie. Le vrai benchmark du sparse, c'est un
dense dimensionne sur ce que MYCELIA utilise reellement.

Estimation, documentee et mesuree (pas supposee) :

    actifs(MYCELIA) = dim                          (une ligne d'embedding lue)
                    + dim * vocab                  (la tete de lecture, entiere)
                    + noeuds_actifs * 1 121        (gate + corps + LayerNorm)
                    + noeuds_actifs * 544          (requete + cle, par noeud)
                    + aretes                       (un coefficient par synapse)

Les 1 121 et 544 se deduisent de ``bank_shapes(dim=16)`` : gate 17, corps 512+32+
512+16 = 1 072, LayerNorm 32 -> 1 121 ; requete 272 + cle 272 -> 544. Les 8
noeuds etaient tous eveilles dans S2, mais on **mesure** la frequence reelle sur
les lots de validation plutot que de la supposer. Les aretes sont comptees
entieres (chaque synapse est lue a chaque saut) ; c'est une majoration.

Le dense a la meme embedding, la meme tete, la meme dimension et deux blocs ; son
unique degre de liberte est la largeur MLP :

    params_dense = 4 992 (embed) + 128 (position) + 32 (ln_f) + 5 304 (tete)
                 + 2 * (1 184 + 32 * mlp)

On resout ``mlp`` pour que ``params_dense`` egale ``actifs(MYCELIA)``, puis on
entraine ce dense au meme budget que la courbe longue (500 pas, memes lots).

Ecrit ``data/lab/s3bis_fair.json``.
"""

from __future__ import annotations

import math
import time

import torch

from lab_common import LAB, ToyCorpus, ensure_lab, read_json, set_seed, write_json
from micro_dense import DenseTransformer, n_parameters, train_dense
from micro_mycelia import load_mycelia
from twins import DENSE_BATCH, DENSE_LR, SEQ, dense_twin

STEPS = 500
LOG_EVERY = 10
VAL_WINDOWS = 48
SEED = 1
PER_NODE = 1121
PER_ADDRESS = 544
FIXED_DENSE = 4992 + 128 + 32 + 5304
BLOCK_FIXED = 1184


def active_estimate(brain, corpus: ToyCorpus, batches: int = 24) -> dict:
    """Parametres reellement consultes par token, mesures sur la validation.

    On compte les **executions de noeud** (``forward_rows`` appele avec ``ids``),
    pas la telemetrie ``last_firing_nodes`` : celle-ci est l'union des noeuds
    allumes sur toute la fenetre, et la diviser par 8 tokens sous-estimait
    l'activite d'un facteur 8 (mesure : 0,875 noeud/token, impossible quand le
    hop 0 fait deja tourner le noeud 0). Chaque saut qui fait tourner un noeud
    paie sa banque complete : c'est cela, le cout par token.
    """
    original = brain.bank.forward_rows
    counter = {"visits": 0}

    def counted(x, ids):
        counter["visits"] += int(ids.numel())
        return original(x, ids)

    # Un token a la fois, etat neuf : c'est la definition « banque consultee par
    # token ». ``last_firing_nodes`` est l'union des noeuds qui ont parle pendant
    # ce forward, donc exactement les noeuds distincts allumes pour ce token.
    distinct = 0
    tokens = 0
    brain.bank.forward_rows = counted
    try:
        for x, _y in corpus.fixed(batches, "val"):
            for position in range(int(x.shape[1])):
                brain(x[:, position : position + 1], grow=False, trace=False)
                distinct += len(brain.last_firing_nodes)
                tokens += 1
    finally:
        brain.bank.forward_rows = original
    per_token = distinct / max(1, tokens)
    visits_per_token = counter["visits"] / max(1, tokens)
    dim, vocab = int(brain.dim), int(brain.vocab)
    embedding = dim
    head = dim * vocab
    nodes = per_token * PER_NODE
    address = per_token * PER_ADDRESS
    edges = float(brain.ne)
    total = embedding + head + nodes + address + edges
    return {
        "noeuds_actifs_par_token": round(per_token, 3),
        "executions_noeud_par_token": round(visits_per_token, 3),
        "executions_noeud": int(counter["visits"]),
        "tokens_mesures": int(tokens),
        "embedding": int(embedding),
        "tete": int(head),
        "noeuds": round(nodes, 1),
        "adresses": round(address, 1),
        "aretes": int(edges),
        "total": round(total, 1),
    }


def mlp_for(target: float) -> int:
    """Largeur MLP qui amene le dense exactement au budget actif vise."""
    mlp = (float(target) - FIXED_DENSE) / 64.0 - BLOCK_FIXED / 32.0
    return max(8, int(round(mlp)))


def main() -> None:
    ensure_lab()
    corpus = ToyCorpus(seq_len=SEQ)
    val = corpus.fixed(VAL_WINDOWS, "val")
    brain = load_mycelia(str(LAB / "s2_mycelia_s1.pt"))
    estimate = active_estimate(brain, corpus)
    mlp = mlp_for(estimate["total"])
    set_seed(SEED)
    model = DenseTransformer(corpus.vocab_size, dim=16, layers=2, heads=2, mlp=mlp, seq=SEQ)
    params = int(n_parameters(model))
    print(f"MYCELIA actifs/token : {estimate['total']:.0f} ({estimate['noeuds_actifs_par_token']} noeuds)")
    print(f"dense equitable      : MLP {mlp} -> {params} parametres total")

    started = time.time()
    result = train_dense(
        model, corpus, STEPS, lr=DENSE_LR, batch=DENSE_BATCH, seed=SEED,
        log_every=LOG_EVERY, val_batches=val, patience=0,
    )
    long_run = read_json(LAB / "s3bis_long.json") if (LAB / "s3bis_long.json").exists() else {}
    mycelia_best = None
    if long_run:
        mycelia_best = float(long_run["mycelia"]["best_val"])

    payload = {
        "estimation": estimate,
        "dense_equitable": {
            "mlp": int(mlp),
            "params": params,
            "parametres_actifs_mycelia": round(estimate["total"], 1),
            "best_val": result["final_val"],
            "best_step": result["best_step"],
            "secondes": round(time.time() - started, 1),
            "courbe": [{"step": r["step"], "val": r["val"]} for r in result["curve"]],
        },
        "mycelia_meme_budget": mycelia_best,
        "dense_jumeau_23748": 4.58,
        "ecart": None if mycelia_best is None else round(float(mycelia_best) - result["final_val"], 4),
    }
    write_json(LAB / "s3bis_fair.json", payload)
    print(f"dense equitable : {result['final_val']:.4f} au pas {result['best_step']} "
          f"en {result['seconds']} s")
    if mycelia_best is not None:
        print(f"MYCELIA (500 pas) : {mycelia_best:.4f} — ecart {payload['ecart']:+.4f}")
    print("ecrit : data/lab/s3bis_fair.json")


if __name__ == "__main__":
    main()
