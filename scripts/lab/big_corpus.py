"""S3-bis D6 — le corpus 10x : capacite ou surapprentissage ?

Le corpus de S2 fait 1000 phrases (4 664 tokens). MYCELIA y atteint sa meilleure
validation au pas 50 puis la validation **remonte** jusqu'a 7,4 pendant que la
perte d'entrainement descend a 3,2 : c'est du surapprentissage franc. La question
que ce script tranche : le plafond de 5,1 tient-il quand il n'y a plus rien a
memoriser ?

* 10 000 phrases, **meme grammaire** que ``corpus_toy.txt`` (meme generateur de
  ``make_lab_data.build_sentences``, graine differente), meme vocabulaire de 312
  mots — sinon les embeddings des deux cerveaux ne correspondraient plus ;
* planchers n-grammes recalcules sur le gros corpus (les bornes bougent : plus
  de phrases, donc moins de contextes rares) ;
* MYCELIA et le dense jumeau reentraines au **meme budget court** (300 pas) sur
  ce corpus, memes lots de validation.

Si le plafond tient alors que le surapprentissage a disparu, c'est la capacite
(l'estimation) ; s'il tombe, c'est la taille du corpus qui parlait.

Ecrit ``data/lab/corpus_toy10k.txt`` et ``data/lab/s3bis_bigcorpus.json``.
"""

from __future__ import annotations

import time

from floors import ngram_floors, unigram_floor
from lab_common import LAB, ToyCorpus, ensure_lab, read_json, set_seed, write_json
from make_lab_data import build_sentences
from micro_dense import n_parameters as dense_parameters, train_dense
from micro_mycelia import n_parameters as mycelia_parameters, train_mycelia
from twins import (
    DENSE_BATCH, DENSE_LR, MYCELIA_BATCH, MYCELIA_LR, MYCELIA_WD, SEQ,
    dense_twin, mycelia_twin,
)

COUNT = 10000
STEPS = 300
LOG_EVERY = 10
VAL_WINDOWS = 48
SEED = 1
PATH = LAB / "corpus_toy10k.txt"


def main() -> None:
    ensure_lab()
    meta = read_json(LAB / "vocab.json")
    vocab = set(meta["vocab"])
    # On genere un peu plus que demande et on filtre : un mot hors vocabulaire
    # serait lu <unk>, et deux cerveaux qui ne lisent que des trous ne
    # s'entrainent pas sur la meme chose. Le filtre garde le vocabulaire de S2
    # exactement (embedding inchange), c'est la quantite de phrases qui change.
    generated = build_sentences(COUNT + 2000, seed=20261121)
    lines = [line for line in generated if all(word in vocab for word in line.split())][:COUNT]
    if len(lines) < COUNT:
        raise SystemExit(f"corpus filtre trop court : {len(lines)}/{COUNT}")
    PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    unknown = sorted({word for line in lines for word in line.split()} - vocab)
    corpus = ToyCorpus(seq_len=SEQ, path=PATH)
    sentences = [line.split() for line in lines]
    tokens = [word for sentence in sentences for word in sentence]
    uni = unigram_floor(tokens, len(meta["vocab"]))
    higher = ngram_floors(sentences, len(meta["vocab"]))
    print(f"corpus 10k : {len(lines)} phrases, {len(tokens)} tokens, mots inconnus {len(unknown)}")
    print(f"planchers  : uni {uni['entropie_lissee']:.4f} / bi {higher['bigramme']['entropie_lissee']:.4f}"
          f" / tri {higher['trigramme']['entropie_lissee']:.4f}")

    val = corpus.fixed(VAL_WINDOWS, "val")

    set_seed(SEED)
    brain = mycelia_twin(corpus.vocab_size, SEED)
    started = time.time()
    mycelia = train_mycelia(
        brain, corpus, STEPS, lr=MYCELIA_LR, batch=MYCELIA_BATCH, seed=SEED,
        log_every=LOG_EVERY, val_batches=val, weight_decay=MYCELIA_WD,
        patience=0, restore_best=True,
    )
    mycelia_row = {
        "params": int(mycelia_parameters(mycelia["brain"])),
        "best_val": mycelia["best_val"],
        "best_step": mycelia["best_step"],
        "secondes": round(time.time() - started, 1),
        "aretes": int(mycelia["edges"]),
        "courbe": [{"step": r["step"], "val": r["val"], "loss": r["loss"]} for r in mycelia["curve"]],
    }
    print(f"MYCELIA  : {mycelia_row['best_val']:.4f} au pas {mycelia_row['best_step']} "
          f"({mycelia_row['secondes']} s)")

    set_seed(SEED)
    model = dense_twin(corpus.vocab_size, SEED)
    started = time.time()
    dense = train_dense(
        model, corpus, STEPS, lr=DENSE_LR, batch=DENSE_BATCH, seed=SEED,
        log_every=LOG_EVERY, val_batches=val, patience=0,
    )
    dense_row = {
        "params": int(dense_parameters(model)),
        "best_val": dense["final_val"],
        "best_step": dense["best_step"],
        "secondes": round(time.time() - started, 1),
        "courbe": [{"step": r["step"], "val": r["val"], "loss": r["loss"]} for r in dense["curve"]],
    }
    print(f"dense    : {dense_row['best_val']:.4f} au pas {dense_row['best_step']} "
          f"({dense_row['secondes']} s)")

    mycelia_hold = float(mycelia_row["best_val"]) >= float(uni["entropie_lissee"]) - 0.15
    payload = {
        "corpus": {
            "phrases": len(lines),
            "tokens": len(tokens),
            "mots_inconnus": len(unknown),
            "fichier": str(PATH.name),
        },
        "planchers": {
            "unigramme": uni["entropie_lissee"],
            "bigramme": higher["bigramme"]["entropie_lissee"],
            "trigramme": higher["trigramme"]["entropie_lissee"],
        },
        "budget": {"pas": STEPS, "log_every": LOG_EVERY, "validation_fenetres": VAL_WINDOWS},
        "mycelia": mycelia_row,
        "dense": dense_row,
        "ecart": round(float(mycelia_row["best_val"]) - float(dense_row["best_val"]), 4),
        "plafond_tient": bool(mycelia_hold),
    }
    write_json(LAB / "s3bis_bigcorpus.json", payload)
    print(f"ecart MYCELIA - dense : {payload['ecart']:+.4f} — plafond tient : {mycelia_hold}")
    print("ecrit : data/lab/s3bis_bigcorpus.json")


if __name__ == "__main__":
    main()
