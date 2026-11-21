"""S3-bis D4 — ablations du routage : quel composant est le goulot ?

Trois variantes du MYCELIA jouet, **meme budget que la courbe longue** (500 pas,
semence 1, memes lots de validation) :

* ``controle``            — la config de S2 (8 noeuds, affinite par arete,
  naissances d'aretes payees par le trafic, cycle de vie tous les 100 pas) ;
* ``i_routage_dense``     — plus de selection par affinite : chaque arete
  transmet a pleine puissance (``route_affinity = False``). Les 8 noeuds etaient
  deja tous eveilles (aucun top-k dans le labo), donc c'est la seule maniere
  d'aller « vers le dense » sans changer le graphe ;
* ``ii_metabolisme_gele`` — les naissances s'arretent au pas 25 (le cablage y
  est deja a ses 48 aretes, comme le controle en fin de course) et le cycle de
  vie est coupe. Meme graphe, mais plus aucune synapse ne nait ;
* ``iii_graphe_aleatoire`` — 48 aretes tirees uniformement au depart (memes
  interdits structurels), figees. Meme taille, memes affinites, mais un cablage
  que le trafic n'a pas choisi.

Diagnostic du routage sur le controle, apres entrainement :

* **equilibre de charge** — frequence d'emission par noeud (un noeud qui parle
  a chaque pas prend tous les gradients : l'effet « riches toujours plus
  riches ») et son indice de Gini ;
* **norme du gradient par noeud** — somme des normes de gradient de la banque,
  l'autre face de la meme inegalite.

Ecrit ``data/lab/s3bis_ablations.json``.
"""

from __future__ import annotations

import argparse
import math
import time

import numpy as np
import torch
import torch.nn.functional as F

from lab_common import LAB, ToyCorpus, ensure_lab, read_json, set_seed, write_json
from micro_mycelia import build_mycelia, n_parameters, train_mycelia
from mycelia.node import COMPUTE_PARAMS
from twins import MYCELIA_BATCH, MYCELIA_LR, MYCELIA_WD, MYCELIA_OVERRIDES, SEQ

STEPS = 500
LOG_EVERY = 10
VAL_WINDOWS = 48
SEED = 1
FREEZE_AT = 25
RANDOM_EDGES = 48

ARMS = ("controle", "i_routage_dense", "ii_metabolisme_gele", "iii_graphe_aleatoire")


def random_graph(brain, edges: int = RANDOM_EDGES, seed: int = SEED) -> dict:
    """Cablage aleatoire fixe de ``edges`` aretes, memes interdits structurels.

    Les regles sont celles du cerveau : le lecteur (1) ne parle jamais, aucun
    lien n'arrive sur une porte (0 ou 1), pas de boucle, pas de doublon.
    """
    generator = torch.Generator().manual_seed(int(seed) + 99)
    attempts = 0
    while brain.ne < int(edges) and attempts < 20000:
        attempts += 1
        source = int(torch.randint(0, brain.n_nodes, (1,), generator=generator))
        target = int(torch.randint(0, brain.n_nodes, (1,), generator=generator))
        if source == target or source == 1 or target in (0, 1):
            continue
        if brain.has_edge(source, target):
            continue
        brain.add_edge(source, target, count=False)
    brain.max_edges = int(brain.ne)  # plus aucune naissance possible
    brain.births = 0
    return {"aretes": int(brain.ne), "tentatives": attempts}


def run_arm(name: str, corpus: ToyCorpus, val) -> dict:
    set_seed(SEED)
    brain = build_mycelia(corpus.vocab_size, SEED, dict(MYCELIA_OVERRIDES))
    kwargs = {
        "lr": MYCELIA_LR,
        "batch": MYCELIA_BATCH,
        "seed": SEED,
        "log_every": LOG_EVERY,
        "val_batches": val,
        "weight_decay": MYCELIA_WD,
        "patience": 0,
        "restore_best": True,
    }
    note = "config S2 (référence)"
    if name == "i_routage_dense":
        brain.route_affinity = False
        note = "affinité par arête coupée : transmission pleine"
    elif name == "ii_metabolisme_gele":
        kwargs["stop_births_at"] = FREEZE_AT
        kwargs["lifecycle_every"] = 0
        note = f"naissances figées au pas {FREEZE_AT}, cycle de vie coupé"
    elif name == "iii_graphe_aleatoire":
        info = random_graph(brain)
        brain.max_births = 0
        kwargs["lifecycle_every"] = 0
        note = f"arêtes aléatoires figées ({info['tentatives']} tirages)"
    started = time.time()
    result = train_mycelia(brain, corpus, STEPS, **kwargs)
    trained = result["brain"]
    return {
        "variante": name,
        "note": note,
        "params": int(n_parameters(trained)),
        "best_val": result["best_val"],
        "best_step": result["best_step"],
        "aretes": int(result["edges"]),
        "pool": int(result["pool"]),
        "secondes": round(time.time() - started, 1),
        "courbe": [{"step": r["step"], "val": r["val"]} for r in result["curve"]],
        "_brain": trained,
    }


def gini(values: np.ndarray) -> float:
    """Inegalite de la charge : 0 = parfaitement egal, 1 = un seul noeud."""
    x = np.sort(np.asarray(values, dtype=np.float64))
    n = len(x)
    if n == 0 or x.sum() <= 0:
        return 0.0
    cumulative = np.cumsum(x)
    return float((n + 1 - 2 * (cumulative / cumulative[-1]).sum()) / n)


def route_diagnostics(brain, corpus: ToyCorpus, batches: int = 24) -> dict:
    """Frequence d'emission et gradient par noeud, sur les lots de validation.

    Le gradient est mesure sur la validation *uniquement comme instrument* : on
    ne fait aucun pas d'optimiseur, on lit qui recoit le signal d'apprentissage.
    """
    brain.eval()
    n = int(brain.n_nodes)
    fires = np.zeros(n)
    grad = np.zeros(n)
    steps = 0
    for index, (x, _y) in enumerate(corpus.fixed(batches, "val")):
        brain.zero_grad(set_to_none=True)
        logits = brain(x, grow=False, trace=False)[0]
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), x.reshape(-1))
        loss.backward()
        for node in brain.last_firing_nodes:
            fires[node] += 1
        for name in COMPUTE_PARAMS:
            value = getattr(brain.bank, name).grad
            if value is not None:
                grad += value.detach().flatten(1).norm(dim=1).numpy() ** 2
        steps += 1
    brain.zero_grad(set_to_none=True)
    fires /= max(1, steps)
    grad = np.sqrt(grad / max(1, steps))
    total = fires.sum()
    # Le nœud 0 est la porte : il reçoit le token, jamais un message. Son zéro
    # est une définition, pas une inégalité de charge — il sort du Gini.
    recipients = fires[1:] if len(fires) > 1 else fires
    return {
        # « allumage » = le nœud a été chaud (il a reçu un message) dans ce lot.
        "frequence_allumage": [round(float(v), 4) for v in fires],
        "frequence_relue": [round(float(v / total), 4) for v in fires] if total else [],
        "gini_allumage": round(gini(recipients), 4),
        "noeuds_destinataires": [int(i) for i in range(1, len(fires))],
        "grad_norm_par_noeud": [round(float(v), 4) for v in grad],
        "gini_gradient": round(gini(grad), 4),
        "noeuds_sans_allumage": [int(i) for i in np.nonzero(fires == 0)[0]],
        "noeuds_sans_gradient": [int(i) for i in np.nonzero(grad < 1e-6)[0]],
    }


def arm_path(name: str):
    return LAB / f"s3bis_ab_{name}.json"


def run_and_save(name: str, corpus: ToyCorpus, val) -> None:
    """Un bras, dans son propre processus : les 4 bras tournent en parallele."""
    row = run_arm(name, corpus, val)
    # Le cerveau sort du dict AVANT la copie dans le payload : sinon il part dans
    # le JSON (mesure : TypeError Brain is not JSON serializable, apres 3 min
    # d'entrainement perdues).
    trained = row.pop("_brain")
    payload = {"variante": row.pop("variante"), **row}
    if name == "controle":
        payload["diagnostic_routage"] = route_diagnostics(trained, corpus)
    write_json(arm_path(name), payload)
    print(f"{name}: {payload['best_val']:.4f} au pas {payload['best_step']} "
          f"({payload['secondes']} s, {payload['aretes']} aretes)")


def assemble(rows: dict[str, dict], diagnostics: dict) -> dict:
    control = float(rows["controle"]["best_val"])
    for name in ARMS:
        rows[name]["ecart_au_controle"] = round(float(rows[name]["best_val"]) - control, 4)
    dense_reference = 4.58
    arm_i = float(rows["i_routage_dense"]["best_val"])
    if arm_i <= dense_reference + 0.15:
        verdict = "ROUTAGE"
        phrase = (
            "sans sélection par affinité, MYCELIA rejoint le dense : le goulot "
            "est le routage, pas la colonie"
        )
    else:
        verdict = "PAS_LE_ROUTAGE"
        phrase = (
            "couper l'affinité ne rapproche pas du dense : le routage n'est pas "
            "le goulot à ce dimensionnement"
        )
    return {
        "protocole": {
            "pas": STEPS,
            "log_every": LOG_EVERY,
            "validation_fenetres": VAL_WINDOWS,
            "semence": SEED,
            "budget": "identique a la courbe longue D3",
        },
        "variantes": rows,
        "diagnostic_routage": diagnostics,
        "dense_reference": dense_reference,
        "verdict": verdict,
        "phrase": phrase,
    }


def merge_arms() -> None:
    """Assemble les 4 fichiers ecrits en parallele, puis le rapport les lit."""
    rows: dict[str, dict] = {}
    diagnostics: dict = {}
    for name in ARMS:
        blob = read_json(arm_path(name))
        if name == "controle":
            diagnostics = blob.pop("diagnostic_routage", {})
        rows[name] = blob
    payload = assemble(rows, diagnostics)
    write_json(LAB / "s3bis_ablations.json", payload)
    print("\nvariante                  perte   ecart   aretes")
    for name in ARMS:
        row = rows[name]
        print(f"{name:24s} {row['best_val']:.4f}  {row['ecart_au_controle']:+.4f}  {row['aretes']}")
    print(f"allumage par noeud : {diagnostics.get('frequence_allumage')}")
    print(f"gradient par noeud : {diagnostics.get('grad_norm_par_noeud')}")
    print(f"VERDICT D4 : {payload['verdict']} — {payload['phrase']}")
    print("ecrit : data/lab/s3bis_ablations.json")


def main() -> None:
    parser = argparse.ArgumentParser(description="ablations du routage (S3-bis D4)")
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--merge", action="store_true")
    args = parser.parse_args()
    ensure_lab()
    corpus = ToyCorpus(seq_len=SEQ)
    val = corpus.fixed(VAL_WINDOWS, "val")
    if args.arm:
        run_and_save(args.arm, corpus, val)
        return
    if args.merge:
        merge_arms()
        return
    rows: dict[str, dict] = {}
    for name in ARMS:
        print(f"ablation {name} ...")
        row = run_arm(name, corpus, val)
        print(
            f"  {row['best_val']:.4f} (pas {row['best_step']}, {row['aretes']} aretes, "
            f"{row['secondes']} s) — {row['note']}"
        )
        rows[name] = row

    diagnostics = route_diagnostics(rows["controle"]["_brain"], corpus)
    control = float(rows["controle"]["best_val"])
    for name in ARMS:
        rows[name]["ecart_au_controle"] = round(float(rows[name]["best_val"]) - control, 4)
        rows[name].pop("_brain", None)

    dense_reference = 4.58
    arm_i = float(rows["i_routage_dense"]["best_val"])
    if arm_i <= dense_reference + 0.15:
        verdict = "ROUTAGE"
        phrase = (
            "sans sélection par affinité, MYCELIA rejoint le dense : le goulot "
            "est le routage, pas la colonie"
        )
    else:
        verdict = "PAS_LE_ROUTAGE"
        phrase = (
            "couper l'affinité ne rapproche pas du dense : le routage n'est pas "
            "le goulot à ce dimensionnement"
        )

    payload = {
        "protocole": {
            "pas": STEPS,
            "log_every": LOG_EVERY,
            "validation_fenetres": VAL_WINDOWS,
            "semence": SEED,
            "budget": "identique a la courbe longue D3",
        },
        "variantes": rows,
        "diagnostic_routage": diagnostics,
        "dense_reference": dense_reference,
        "verdict": verdict,
        "phrase": phrase,
    }
    write_json(LAB / "s3bis_ablations.json", payload)

    print("\nvariante                  perte   ecart   aretes")
    for name in ARMS:
        row = rows[name]
        print(f"{name:24s} {row['best_val']:.4f}  {row['ecart_au_controle']:+.4f}  {row['aretes']}")
    print(f"\nemission par noeud : {diagnostics['frequence_emission']}")
    print(f"gradient par noeud : {diagnostics['grad_norm_par_noeud']}")
    print(f"Gini emission {diagnostics['gini_emission']} / gradient {diagnostics['gini_gradient']}")
    print(f"VERDICT D4 : {verdict} — {phrase}")
    print("ecrit : data/lab/s3bis_ablations.json")


if __name__ == "__main__":
    main()
