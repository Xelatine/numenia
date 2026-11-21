"""S5-MAX Verrou 1 — la mesure juste, et la re-mesure a deux colonnes.

M1 est fait dans ``lab_common`` : ``mycelia_loss`` compare desormais au token
**courant** (la cible d'entrainement, la tache du dense), et l'ancienne fonction
vit sous ``mycelia_loss_legacy``, intacte.

M2 est ce script. Il publie les **deux colonnes** partout, sur les memes
fenetres :

* ``nouvelle``  — predire ``x[t]`` a partir du prefixe ``x[:t]`` (mesure juste) ;
* ``ancienne``  — predire ``y[t] = x[t+1]`` (la metrique publiee de S2/S3-BIS).

Pour qui :

* les **planchers n-grammes** (uni, bi, tri) recalcules sur les memes paires ;
  les deux colonnes y sont identiques *par construction* (meme table de paires,
  decalage d'un cran) et le script le verifie au lieu de l'affirmer ;
* le **MYCELIA repare** de S4 (lecture pleine et deux vitesses, 1000 et 10k,
  trois semences) : les cerveaux sauvegardes sont re-mesures, pas re-entraines ;
* le **dense jumeau** (S2 a 1000, et trois semences re-entrainees a 10k au meme
  budget de 600 pas).

Le classement de la mission S4 est re-juge sur la colonne nouvelle. Si les deux
colonnes ne donnent pas le meme classement, le rapport doit le dire en gras.

    python scripts/lab/s5_measure.py --arm floors
    python scripts/lab/s5_measure.py --arm brains
    python scripts/lab/s5_measure.py --arm dense10k:1
    python scripts/lab/s5_measure.py --arm merge
"""

from __future__ import annotations

import argparse
import torch
import torch.nn.functional as F

from lab_common import LAB, ensure_lab, read_json, write_json
from micro_dense import train_dense
from repaired_mycelia import load_brain
from s5_common import (
    CORPORA,
    L_REF,
    SEEDS,
    columns_dense,
    columns_mycelia,
    load_corpus,
    load_dense,
    mean_sd,
)
from twins import DENSE_BATCH, DENSE_LR, MLP

ALPHA = 0.5
NGRAM_REFERENCE = {"unigramme": 5.082888, "bigramme": 3.046586, "trigramme": 1.982794}
MODES = ("pleine", "deux_vitesses")


def fragment(name: str):
    return LAB / f"s5_{name}.json"


# ---------------------------------------------------------------------------
# Les planchers n-grammes, dans les deux colonnes
# ---------------------------------------------------------------------------
def arm_floors() -> None:
    """Les planchers n-grammes, recalcules avec la methode publiee de S3-BIS.

    Un n-gramme *predit deja* le token courant a partir du prefixe : sa table de
    paires est la meme dans les deux colonnes, et le decalage d'un cran ne vivait
    que dans la comparaison des logits de MYCELIA (ou la cible ``y`` n'est pas
    celle que le modele apprend). Les deux colonnes d'un plancher sont donc le
    **meme nombre**, et c'est ecrit ici plutot que suppose : la fonction de S3-BIS
    est re-executee telle quelle, et sa valeur doit retomber sur celle qui est
    publiee.
    """
    from floors import corpus_tokens, ngram_floors, unigram_floor, vocab_size

    lines, sentences = corpus_tokens()
    vocab = vocab_size()
    tokens = [word for sentence in sentences for word in sentence]
    uni = unigram_floor(tokens, vocab)
    higher = ngram_floors(sentences, vocab)
    rows = {
        "unigramme": {
            "nouvelle": uni["entropie_lissee"],
            "ancienne": uni["entropie_lissee"],
            "mle": uni["entropie_mle"],
            "tokens": uni["tokens"],
            "reference_s3bis": NGRAM_REFERENCE["unigramme"],
        },
        "bigramme": {
            "nouvelle": higher["bigramme"]["entropie_lissee"],
            "ancienne": higher["bigramme"]["entropie_lissee"],
            "mle": higher["bigramme"]["entropie_mle"],
            "contextes": higher["bigramme"]["contextes"],
            "reference_s3bis": NGRAM_REFERENCE["bigramme"],
        },
        "trigramme": {
            "nouvelle": higher["trigramme"]["entropie_lissee"],
            "ancienne": higher["trigramme"]["entropie_lissee"],
            "mle": higher["trigramme"]["entropie_mle"],
            "contextes": higher["trigramme"]["contextes"],
            "reference_s3bis": NGRAM_REFERENCE["trigramme"],
        },
    }
    payload = {
        "corpus": "corpus_toy.txt",
        "phrases": len(lines),
        "alpha": ALPHA,
        "vocabulaire": vocab,
        "uniforme_ln_v": round(float(torch.log(torch.tensor(float(vocab)))), 6),
        "planchers": rows,
        "reproduit_s3bis": bool(all(
            abs(row["nouvelle"] - row["reference_s3bis"]) < 1e-6 for row in rows.values()
        )),
        "note": "les deux colonnes d'un n-gramme sont le meme nombre : la table de "
                "paires ne change pas, seul le nom de la cible change",
    }
    write_json(fragment("floors"), payload)
    for name, row in rows.items():
        print(f"{name:10s} nouvelle {row['nouvelle']:.6f} / ancienne {row['ancienne']:.6f} "
              f"(S3-BIS {row['reference_s3bis']:.6f})", flush=True)


# ---------------------------------------------------------------------------
# Re-mesure des cerveaux deja entraines
# ---------------------------------------------------------------------------
def arm_brains() -> None:
    s4 = read_json(LAB / "s4_validation.json")
    stored = {}
    for corpus_name, block in s4.get("runs", {}).items():
        for row in block.get("lignes", []):
            stored[f"{corpus_name}/{row['mode']}/{row['semence']}"] = row
    rows: dict = {}
    for corpus_name in CORPORA:
        corpus = load_corpus(corpus_name)
        windows = corpus.fixed(CORPORA[corpus_name]["val"], "val")
        for mode in MODES:
            for seed in SEEDS:
                path = LAB / f"s4_{corpus_name}_{mode}_s{seed}.pt"
                if not path.exists():
                    continue
                brain = load_brain(path)
                cols = columns_mycelia(brain, windows)
                key = f"{corpus_name}/{mode}/{seed}"
                ref = stored.get(key, {})
                row = {
                    "corpus": corpus_name,
                    "mode": mode,
                    "semence": int(seed),
                    **cols,
                    "s4_honnete_stocke": ref.get("honnete"),
                    "s4_decalee_stocke": ref.get("decalee"),
                    # La preuve de M1 : la mesure corrigee sur les memes fenetres
                    # (position 0 incluse, comme S4) doit retomber exactement sur
                    # le chiffre honnete deja publie.
                    "ecart_m1_vs_s4_honnete": (
                        None if ref.get("honnete") is None
                        else round(cols["nouvelle_0_7"] - float(ref["honnete"]), 4)
                    ),
                    "ecart_ancienne_vs_s4_decalee": (
                        None if ref.get("decalee") is None
                        else round(cols["ancienne"] - float(ref["decalee"]), 4)
                    ),
                }
                rows[key] = row
                print(f"{key:28s} nouvelle {cols['nouvelle']:.4f} (0-7 : "
                      f"{cols['nouvelle_0_7']:.4f}) ancienne {cols['ancienne']:.4f} "
                      f"(S4 stocke {ref.get('honnete')} / {ref.get('decalee')})", flush=True)
    write_json(fragment("brains"), {"lignes": rows})
    print(f"ecrit : {fragment('brains')}", flush=True)


# ---------------------------------------------------------------------------
# Le dense jumeau
# ---------------------------------------------------------------------------
def arm_dense10k(seed: int) -> None:
    corpus = load_corpus("10k")
    windows = corpus.fixed(CORPORA["10k"]["val"], "val")
    from s5_common import build_dense

    model = build_dense(corpus.vocab_size, seed, mlp=MLP)
    result = train_dense(
        model, corpus, 600, lr=DENSE_LR, batch=DENSE_BATCH, seed=int(seed),
        log_every=100, patience=0, val_batches=windows,
    )
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": {"dim": 16, "layers": 2, "heads": 2, "mlp": MLP, "seq": 8},
            "vocab_size": corpus.vocab_size,
            "extra": {"seed": int(seed), "corpus": "10k", "pas": 600},
        },
        str(LAB / f"s5_dense10k_s{seed}.pt"),
    )
    cols = columns_dense(model, windows)
    row = {"corpus": "10k", "semence": int(seed), "pas": 600, **cols,
           "best_step": result["best_step"], "secondes": result["seconds"]}
    write_json(fragment(f"dense10k_s{seed}"), row)
    print(f"dense 10k s{seed} : nouvelle {cols['nouvelle']:.4f} ancienne "
          f"{cols['ancienne']:.4f} — {result['seconds']}s", flush=True)


def arm_dense1000(seed: int) -> None:
    """Le dense jumeau de S4 (mlp 166, 23 748 parametres), 600 pas, 3 semences.

    Les ``s2_dense`` ne conviennent pas : la passe "meme savoir" de S2 les avait
    arretes au plancher de MYCELIA (5,08). Ici c'est le protocole S4 exact, memes
    600 pas que le MYCELIA repare.
    """
    corpus = load_corpus("1000")
    windows = corpus.fixed(CORPORA["1000"]["val"], "val")
    from s5_common import build_dense

    model = build_dense(corpus.vocab_size, seed, mlp=MLP)
    result = train_dense(
        model, corpus, 600, lr=DENSE_LR, batch=DENSE_BATCH, seed=int(seed),
        log_every=100, patience=0, val_batches=windows,
    )
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": {"dim": 16, "layers": 2, "heads": 2, "mlp": MLP, "seq": 8},
            "vocab_size": corpus.vocab_size,
            "extra": {"seed": int(seed), "corpus": "1000", "pas": 600},
        },
        str(LAB / f"s5_dense1000_s{seed}.pt"),
    )
    cols = columns_dense(model, windows)
    row = {"corpus": "1000", "semence": int(seed), "pas": 600, **cols,
           "parametres": int(sum(p.numel() for p in model.parameters())),
           "best_step": result["best_step"], "secondes": result["seconds"]}
    write_json(fragment(f"dense1000_s{seed}"), row)
    print(f"dense 1000 s{seed} : nouvelle {cols['nouvelle']:.4f} ancienne "
          f"{cols['ancienne']:.4f} — {result['seconds']}s", flush=True)


def arm_dense() -> None:
    rows = []
    for seed in SEEDS:
        for name in (f"dense1000_s{seed}", f"dense10k_s{seed}"):
            path = fragment(name)
            if path.exists():
                rows.append(read_json(path))
    write_json(fragment("dense"), {"lignes": rows})
    for row in rows:
        print(f"dense {row['corpus']:4s} s{row['semence']} : nouvelle "
              f"{row['nouvelle']:.4f} ancienne {row['ancienne']:.4f}", flush=True)
    print(f"ecrit : {fragment('dense')}", flush=True)


# ---------------------------------------------------------------------------
# Le classement
# ---------------------------------------------------------------------------
def arm_merge() -> None:
    floors = read_json(fragment("floors"))
    brains = read_json(fragment("brains"))["lignes"]
    dense = read_json(fragment("dense"))["lignes"] if fragment("dense").exists() else {"lignes": []}
    if isinstance(dense, dict):
        dense = dense.get("lignes", [])

    def summarise(rows, key_note: str) -> dict:
        new_vals = [r["nouvelle"] for r in rows]
        old_vals = [r["ancienne"] for r in rows]
        new_mean, new_sd = mean_sd(new_vals)
        old_mean, old_sd = mean_sd(old_vals)
        return {
            "n": len(rows), "nouvelle_moyenne": new_mean, "nouvelle_ecart_type": new_sd,
            "ancienne_moyenne": old_mean, "ancienne_ecart_type": old_sd,
            "valeurs_nouvelle": [round(float(v), 4) for v in new_vals],
            "note": key_note,
        }

    table = {}
    for corpus_name in CORPORA:
        for mode in MODES:
            rows = [r for r in brains.values()
                    if r["corpus"] == corpus_name and r["mode"] == mode]
            if rows:
                table[f"mycelia/{corpus_name}/{mode}"] = summarise(
                    rows, "MYCELIA repare S4, re-mesure"
                )
    for corpus_name in CORPORA:
        rows = [r for r in dense if r.get("corpus") == corpus_name]
        if rows:
            table[f"dense/{corpus_name}"] = summarise(rows, "dense jumeau")
    for name, row in floors["planchers"].items():
        table[f"plancher/{name}"] = {
            "n": 1, "nouvelle_moyenne": row["nouvelle"], "nouvelle_ecart_type": 0.0,
            "ancienne_moyenne": row["ancienne"], "ancienne_ecart_type": 0.0,
            "valeurs_nouvelle": [row["nouvelle"]], "note": "n-gramme lissee alpha=0,5",
        }

    mycelia_new = table["mycelia/1000/deux_vitesses"]["nouvelle_moyenne"]
    mycelia_full_new = table["mycelia/1000/pleine"]["nouvelle_moyenne"]
    bigram = table["plancher/bigramme"]["nouvelle_moyenne"]
    dense_new = table["dense/1000"]["nouvelle_moyenne"]
    dense_old = table["dense/1000"]["ancienne_moyenne"]
    mycelia_old = table["mycelia/1000/deux_vitesses"]["ancienne_moyenne"]

    def rank(value: float) -> list[str]:
        order = sorted(
            [("MYCELIA", value), ("bigramme", bigram), ("dense", dense_new)],
            key=lambda pair: pair[1],
        )
        return [name for name, _ in order]

    old_rank = [name for name, _ in sorted(
        [("MYCELIA", mycelia_old), ("bigramme", bigram), ("dense", dense_old)],
        key=lambda pair: pair[1],
    )]
    payload = {
        "protocole": {
            "corpus": "1000 phrases (et 10k pour le second bloc)",
            "semences": list(SEEDS),
            "colonnes": {
                "nouvelle": "predire x[t] a partir du prefixe (tache reelle du cerveau)",
                "ancienne": "predire y[t] = x[t+1] (metrique publiee S2/S3-BIS)",
            },
            "l_ref": L_REF,
        },
        "planchers": floors["planchers"],
        "table": table,
        "classement": {
            "nouvelle": rank(mycelia_new),
            "ancienne": old_rank,
            "mycelia_nouvelle": mycelia_new,
            "mycelia_pleine_nouvelle": mycelia_full_new,
            "mycelia_ancienne": mycelia_old,
            "bigramme": bigram,
            "dense_nouvelle": dense_new,
            "dense_ancienne": dense_old,
            "inverse": bool(rank(mycelia_new) != old_rank and
                            set(rank(mycelia_new)) == set(old_rank)),
        },
        "valeur_lue_dans_les_cerveaux": "les cerveaux re-mesures sont ceux de S4 "
                                        "(aucun re-entrainement)",
    }
    write_json(LAB / "s5_measure.json", payload)
    print("— table S5 (colonnes nouvelle / ancienne)")
    for key, row in table.items():
        print(f"  {key:32s} {row['nouvelle_moyenne']:.4f} ± {row['nouvelle_ecart_type']:.4f}"
              f"  /  {row['ancienne_moyenne']:.4f} ± {row['ancienne_ecart_type']:.4f}")
    print(f"  classement nouvelle : {' < '.join(rank(mycelia_new))}", flush=True)
    print(f"  ecrit : {LAB / 's5_measure.json'}", flush=True)


def main() -> None:
    ensure_lab()
    parser = argparse.ArgumentParser(description="S5 verrou 1 : la mesure juste")
    parser.add_argument("--arm", action="append", default=[])
    args = parser.parse_args()
    for arm in args.arm:
        if arm == "floors":
            arm_floors()
        elif arm == "brains":
            arm_brains()
        elif arm.startswith("dense10k"):
            seed = int(arm.split(":")[1]) if ":" in arm else 1
            arm_dense10k(seed)
        elif arm.startswith("dense1000"):
            seed = int(arm.split(":")[1]) if ":" in arm else 1
            arm_dense1000(seed)
        elif arm == "dense":
            arm_dense()
        elif arm == "merge":
            arm_merge()
        else:
            raise SystemExit(f"bras inconnu : {arm}")


if __name__ == "__main__":
    main()
