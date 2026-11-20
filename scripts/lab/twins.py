"""S2 — les jumeaux paramétriquement egaux.

La Mission S comparait un dense de 17 016 parametres a un MYCELIA de 167 638 :
la taille etait la variable cachee, et aucun resultat de conversion ne pouvait
dire si l'echec venait de l'architecture ou du budget. Ici les deux cerveaux
ont **exactement** le meme nombre de parametres, le meme vocabulaire (312 mots),
la meme dimension (16) et la meme tache (mot suivant). La seule variable qui
reste est l'architecture.

Pourquoi 23 748 et pas 17 016 : le vrai ``Brain`` refuse moins de 8 noeuds (ses
12 aretes de depart occupent les noeuds 0..7), et a ``dim=16`` avec 312 mots,
l'embedding (4 992) plus la tete de lecture (5 304) pesent deja 10 296. Huit
noeuds a 1 665 parametres chacun mettent le plancher a 23 664. La cible
« ~17k » est donc hors d'atteinte pour l'architecture MYCELIA ; on egalise vers
le haut, ce qui satisfait la contrainte qui compte (egalite stricte) :

    dense  : embed 4 992 + pos 128 + 2 blocs a MLP 166 + ln_f 32 = 23 748
    MYCELIA: embed 4 992 + tete 5 304 + ln 32 + 8 noeuds (13 320) + 100 aretes = 23 748

Metabolisme : la population est **fixee** a 8 noeuds (sinon chaque naissance
ajouterait 1 665 parametres et l'egalite tomberait en cours d'entrainement),
mais le cablage vit : naissances d'aretes payees par le trafic, coupe des
synapses faminees, enterrement des aretes isolees. ``max_edges=100`` laisse 88
slots libres apres les 12 aretes de depart.

Meme savoir : les deux s'entrainent jusqu'a la meme perte de validation, pas au
meme nombre de pas. Si 4,0 n'est pas atteignable par les deux, le script prend
la meilleure perte commune et l'ecrit noir sur blanc dans le JSON.
"""

from __future__ import annotations

import argparse
import time

import torch

from lab_common import LAB, ToyCorpus, ensure_lab, set_seed, write_json
from micro_dense import DenseTransformer, n_parameters, train_dense
from micro_mycelia import build_mycelia, n_parameters as mycelia_parameters, train_mycelia
from mycelia.checkpoint import save_genome

DIM = 16
LAYERS = 2
HEADS = 2
MLP = 166
SEQ = 8
POOL = 8
EDGES = 100
TARGET = 4.0

MYCELIA_OVERRIDES = {
    "n_nodes": POOL,
    "max_edges": EDGES,
    "max_population": POOL,
    "max_node_births": 0,
}

DENSE_LR = 3e-3
DENSE_BATCH = 32
MYCELIA_LR = 2e-3
MYCELIA_BATCH = 32
MYCELIA_WD = 0.01


def dense_twin(vocab: int, seed: int) -> DenseTransformer:
    torch.manual_seed(int(seed))
    return DenseTransformer(vocab, dim=DIM, layers=LAYERS, heads=HEADS, mlp=MLP, seq=SEQ)


def mycelia_twin(vocab: int, seed: int):
    return build_mycelia(vocab, int(seed), dict(MYCELIA_OVERRIDES))


def dense_path(seed: int):
    return LAB / f"s2_dense_s{int(seed)}.pt"


def mycelia_path(seed: int):
    return LAB / f"s2_mycelia_s{int(seed)}.pt"


def save_dense(model: DenseTransformer, seed: int, step: int, curve: list[dict]) -> None:
    torch.save(
        {
            "state_dict": model.state_dict(),
            "vocab": list(ToyCorpus(seq_len=SEQ).vocab),
            "config": {
                "dim": DIM,
                "layers": LAYERS,
                "heads": HEADS,
                "mlp": MLP,
                "seq": SEQ,
            },
            "seed": int(seed),
            "step": int(step),
            "curve": curve,
        },
        dense_path(seed),
    )


def load_dense_twin(seed: int) -> DenseTransformer:
    blob = torch.load(dense_path(seed), weights_only=False)
    model = DenseTransformer(
        len(blob["vocab"]),
        dim=blob["config"]["dim"],
        layers=blob["config"]["layers"],
        heads=blob["config"]["heads"],
        mlp=blob["config"]["mlp"],
        seq=blob["config"]["seq"],
    )
    model.load_state_dict(blob["state_dict"])
    model.eval()
    return model


def load_mycelia_twin(seed: int):
    from micro_mycelia import load_mycelia

    return load_mycelia(str(mycelia_path(seed)))


def train_one_seed(corpus: ToyCorpus, seed: int, verbose: bool = True) -> dict:
    """Deux passes : converger, puis rejouer jusqu'au savoir commun ``L``.

    La deuxieme passe est possible parce que tout est deterministe (memes lots,
    meme graine) : elle redonne exactement la meme trajectoire et s'arrete au
    premier point de validation sous ``L``. C'est ce point-la qui devient le
    jumeau — un modele arrete a 5,08 et un autre arrete a 5,08, pas un modele
    de 300 pas et un autre de 150.
    """
    set_seed(seed)
    started = time.time()

    # ---- passe 1 : convergence de chacun, a son rythme -------------------
    dense = dense_twin(corpus.vocab_size, seed)
    mycelia = mycelia_twin(corpus.vocab_size, seed)
    if n_parameters(dense) != mycelia_parameters(mycelia):
        raise SystemExit(
            f"jumeaux non egaux : dense {n_parameters(dense)} != "
            f"mycelia {mycelia_parameters(mycelia)}"
        )
    params = int(n_parameters(dense))

    first_dense = train_dense(
        dense, corpus, 600, lr=DENSE_LR, batch=DENSE_BATCH, seed=seed,
        log_every=50, patience=6,
    )
    first_mycelia = train_mycelia(
        mycelia, corpus, 600, lr=MYCELIA_LR, batch=MYCELIA_BATCH, seed=seed,
        log_every=50, patience=5, weight_decay=MYCELIA_WD,
    )

    best_dense = float(first_dense["final_val"])
    best_mycelia = float(first_mycelia["final_val"])
    if best_dense <= TARGET and best_mycelia <= TARGET:
        common = TARGET
        motive = "les deux jumeaux atteignent la cible 4,0"
    else:
        common = round(max(best_dense, best_mycelia), 4)
        motive = (
            "cible 4,0 non atteinte par les deux : la perte commune est la "
            "meilleure perte atteignable par le plus faible"
        )

    # ---- passe 2 : rejouer jusqu'au savoir commun ------------------------
    set_seed(seed)
    dense2 = dense_twin(corpus.vocab_size, seed)
    second_dense = train_dense(
        dense2, corpus, 600, lr=DENSE_LR, batch=DENSE_BATCH, seed=seed,
        log_every=25, patience=0, stop_at=common,
    )
    set_seed(seed)
    mycelia2 = mycelia_twin(corpus.vocab_size, seed)
    second_mycelia = train_mycelia(
        mycelia2, corpus, 600, lr=MYCELIA_LR, batch=MYCELIA_BATCH, seed=seed,
        log_every=25, patience=0, weight_decay=MYCELIA_WD, stop_at=common,
    )
    brain = second_mycelia.pop("brain")

    save_dense(dense2, seed, int(second_dense["best_step"]), second_dense["curve"])
    save_genome(
        mycelia_path(seed), brain, step=int(second_mycelia["best_step"]),
        history=second_mycelia["curve"],
    )

    row = {
        "semence": int(seed),
        "parametres": params,
        "egaux": True,
        "cible": TARGET,
        "cible_atteinte": bool(best_dense <= TARGET and best_mycelia <= TARGET),
        "perte_commune": round(float(common), 4),
        "motive": motive,
        "dense": {
            "params": int(n_parameters(dense2)),
            "convergence_val": best_dense,
            "convergence_step": int(first_dense["best_step"]),
            "arret_val": float(second_dense["final_val"]),
            "arret_step": int(second_dense["best_step"]),
            "secondes": first_dense["seconds"],
            "courbe_convergence": first_dense["curve"],
            "courbe_arret": second_dense["curve"],
        },
        "mycelia": {
            "params": int(mycelia_parameters(brain)),
            "convergence_val": best_mycelia,
            "convergence_step": int(first_mycelia["best_step"]),
            "arret_val": float(second_mycelia["final_val"]),
            "arret_step": int(second_mycelia["best_step"]),
            "secondes": first_mycelia["seconds"],
            "pool": int(brain.n_nodes),
            "aretes": int(brain.ne),
            "naissances": int(brain.births),
            "aretes_mortes": int(brain.edge_deaths),
            "courbe_convergence": first_mycelia["curve"],
            "courbe_arret": second_mycelia["curve"],
        },
        "secondes": round(time.time() - started, 1),
    }
    if verbose:
        print(
            f"semence {seed} : {params} parametres — dense {best_dense:.4f} / "
            f"MYCELIA {best_mycelia:.4f} -> commun {common:.4f} ; "
            f"arret dense pas {row['dense']['arret_step']}, "
            f"MYCELIA pas {row['mycelia']['arret_step']} "
            f"({row['mycelia']['aretes']} aretes, {row['mycelia']['secondes']} s)"
        )
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="jumeaux S2 : memes parametres, memes savoirs")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    args = parser.parse_args()

    ensure_lab()
    corpus = ToyCorpus(seq_len=SEQ)
    rows = [train_one_seed(corpus, int(seed)) for seed in args.seeds]
    payload = {
        "architecture": {
            "dim": DIM,
            "layers": LAYERS,
            "heads": HEADS,
            "mlp": MLP,
            "seq": SEQ,
            "pool": POOL,
            "aretes_max": EDGES,
            "vocab": corpus.vocab_size,
        },
        "cible": TARGET,
        "semences": rows,
        "parametres": rows[0]["parametres"],
        "egalite": all(row["parametres"] == rows[0]["parametres"] for row in rows),
    }
    write_json(LAB / "s2_twins.json", payload)
    print(
        f"ecrit : data/lab/s2_twins.json — {len(rows)} paires, "
        f"{rows[0]['parametres']} parametres chacune"
    )


if __name__ == "__main__":
    main()
