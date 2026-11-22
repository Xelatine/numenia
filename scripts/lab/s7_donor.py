"""S7 — le professeur dense (donneur) : entrainement sur corpus 10k, 3 semences.

Config gefree et exportée dans data/lab/donor_s7.pt (poids + config).
C'est le meilleur de la lignée S5 (3.88-3.92) : meme architecture que S5
(2 couches, 16 dim, 2 têtes, MLP 768), meme corpus 10k, 3 semences.

    python scripts/lab/s7_donor.py
"""

from __future__ import annotations

import time
from pathlib import Path

import torch
import torch.nn.functional as F

from lab_common import LAB, ToyCorpus, ensure_lab, read_json, set_seed, write_json
from micro_dense import DenseTransformer, train_dense
from s5_common import (
    DIM,
    HEADS,
    LAYERS,
    MLP_TWIN,
    SEEDS,
    build_dense,
    dense_parameters,
    load_corpus,
    save_dense,
)

OUT = LAB / "donor_s7.pt"
LOG = LAB / "s7_donor.json"


def main() -> None:
    ensure_lab()
    corpus = load_corpus("10k")
    val = corpus.fixed(100, "val")
    results = []

    print(f"== S7 DONOR : professeur dense sur corpus 10k ==")
    print(f"   architecture : {DIM}d, {LAYERS} couches, {HEADS} tetes, MLP {MLP_TWIN}")
    print(f"   parametres : {dense_parameters(MLP_TWIN)}")
    print(f"   corpus : {corpus.vocab_size} mots, {corpus.train.numel()} tokens")
    print(f"   validation : {len(val)} fenetres")

    for seed in SEEDS:
        started = time.time()
        set_seed(seed)
        teacher = build_dense(corpus.vocab_size, seed, mlp=MLP_TWIN)
        result = train_dense(
            teacher,
            corpus,
            600,
            lr=3e-3,
            batch=32,
            seed=seed,
            log_every=100,
            patience=6,
            val_batches=val,
        )
        save_dense(teacher, LAB / f"s7_teacher_s{seed}.pt", {"val": result["final_val"], "seed": seed})
        row = {
            "semence": int(seed),
            "val_finale": round(float(result["final_val"]), 4),
            "best_step": int(result["best_step"]),
            "secondes": round(time.time() - started, 1),
            "parametres": dense_parameters(MLP_TWIN),
        }
        results.append(row)
        print(f"   semence {seed} : val {result['final_val']:.4f} au pas {result['best_step']} ({result['seconds']}s)")

    # Exporter le meilleur comme donneur officiel
    best = min(results, key=lambda r: r["val_finale"])
    best_teacher = build_dense(corpus.vocab_size, best["semence"], mlp=MLP_TWIN)
    blob = torch.load(str(LAB / f"s7_teacher_s{best['semence']}.pt"), weights_only=False)
    best_teacher.load_state_dict(blob["state_dict"])
    best_teacher.eval()

    save_dense(
        best_teacher,
        OUT,
        {
            "val": best["val_finale"],
            "seed": best["semence"],
            "architecture": {"dim": DIM, "layers": LAYERS, "heads": HEADS, "mlp": MLP_TWIN},
            "corpus": "10k",
            "val_windows": 100,
            "patience": 6,
        },
    )

    payload = {
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "architecture": {"dim": DIM, "layers": LAYERS, "heads": HEADS, "mlp": MLP_TWIN},
        "parametres": dense_parameters(MLP_TWIN),
        "corpus": "10k (data/lab/corpus_toy10k.txt)",
        "semences": results,
        "meilleur": best,
        "donneur_export": str(OUT),
    }
    write_json(LOG, payload)
    print(f"\nDonneur exporté : {OUT} (semence {best['semence']}, val {best['val_finale']})")
    print(f"Rapport : {LOG}")


if __name__ == "__main__":
    main()
