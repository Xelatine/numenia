#!/usr/bin/env python3
"""
S8 — Banc multi-donneurs : entraînement de 4 donneurs divergents
================================================================

D1. 4 DONNEURS DIVERGENTS sur le corpus grammaire 10k :
    · DONNEUR-A : dense 63k (lignée S5, référence) — seed 1
    · DONNEUR-B : dense même taille, graine + architecture différente — seed 2, couche 3
    · DONNEUR-C : dense avec ordre des tokens mélangé pendant l'entraînement (contexte différent)
    · DONNEUR-D : NUMENIA pool 24 entraînée directement (test symbolique)

D2. Mesurer la divergence : matrice KL croisée 4×4 + top-3 identique
D3. Export : data/lab/s8/donors/ avec config + poids + carte d'identité
"""

import argparse
import json
import sys
import time
from pathlib import Path

# Ajouter les chemins nécessaires
sys.path.insert(0, str(Path(__file__).parents[1]))
sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn.functional as F

from lab_common import LAB, ToyCorpus, set_seed, write_json
from micro_dense import DenseTransformer, train_dense, n_parameters
from repaired_mycelia import build_repaired, train_repaired, save_brain, load_brain, aligned_loss
from s5_common import columns_dense, columns_mycelia, kl_direction, topk_overlap

# Configuration
CORPUS_PATH = LAB / "corpus_toy.txt"  # corpus 10k (grammaire)
SEQ_LEN = 8
VAL_WINDOWS = 100
TRAIN_STEPS = 600
BATCH_SIZE = 32
DENSE_LR = 3e-3
NUMENIA_LR = 2e-3
NUMENIA_WD = 0.01
DONORS_DIR = LAB / "s8" / "donors"
DONORS_DIR.mkdir(parents=True, exist_ok=True)

# Identité des donneurs
DONOR_CONFIGS = {
    "A": {
        "type": "dense",
        "label": "dense (lignée S5)",
        "seed": 1,
        "mlp": 768,  # 63k paramètres
        "layers": 2,
        "heads": 2,
        "dim": 16,
    },
    "B": {
        "type": "dense",
        "label": "dense variant (graine+arch)",
        "seed": 2,
        "mlp": 768,
        "layers": 3,  # architecture différente (plus profond)
        "heads": 2,
        "dim": 16,
    },
    "C": {
        "type": "dense",
        "label": "dense ordre mélange (contextualité)",
        "seed": 3,
        "mlp": 768,
        "layers": 2,
        "heads": 2,
        "dim": 16,
        "permute_tokens": True,  # ordre mélangé pendant l'entraînement
    },
    "D": {
        "type": "brain",
        "label": "NUMENIA sparse entraînée (test symbolique)",
        "seed": 4,
        "pool": 24,
        "max_edges": 220,
        "two_speeds": True,
    },
}


def load_corpus():
    """Charger le corpus 10k."""
    corpus = ToyCorpus(seq_len=SEQ_LEN, path=CORPUS_PATH)
    val = corpus.fixed(VAL_WINDOWS, "val")
    return corpus, val


def train_donor_A(corpus, val):
    """DONNEUR-A : dense 63k lignée S5, seed 1."""
    config = DONOR_CONFIGS["A"]
    set_seed(config["seed"])
    model = DenseTransformer(
        corpus.vocab_size,
        dim=config["dim"],
        layers=config["layers"],
        heads=config["heads"],
        mlp=config["mlp"],
        seq=SEQ_LEN,
    )
    result = train_dense(
        model, corpus, TRAIN_STEPS,
        lr=DENSE_LR, batch=BATCH_SIZE, seed=config["seed"],
        log_every=100, patience=6, val_batches=val,
    )
    save_dense(model, "A", result)
    print(f"[A] dense done: val={result['final_val']:.4f}, params={n_parameters(model)}")
    return "A", model, result


def train_donor_B(corpus, val):
    """DONNEUR-B : dense variant, seed 2, 3 couches."""
    config = DONOR_CONFIGS["B"]
    set_seed(config["seed"])
    model = DenseTransformer(
        corpus.vocab_size,
        dim=config["dim"],
        layers=config["layers"],
        heads=config["heads"],
        mlp=config["mlp"],
        seq=SEQ_LEN,
    )
    result = train_dense(
        model, corpus, TRAIN_STEPS,
        lr=DENSE_LR, batch=BATCH_SIZE, seed=config["seed"],
        log_every=100, patience=6, val_batches=val,
    )
    save_dense(model, "B", result)
    print(f"[B] dense variant done: val={result['final_val']:.4f}, params={n_parameters(model)}")
    return "B", model, result


def train_donor_C(corpus, val):
    """
    DONNEUR-C : dense avec ordre des tokens mélangé pendant l'entraînement.

    L'idée est que le contexte d'apprentissage est différent, donc le modèle
    apprend une représentation différente même si l'architecture est identique.
    On mélange les tokens du batch avant l'entraînement.
    """
    config = DONOR_CONFIGS["C"]
    set_seed(config["seed"])
    model = DenseTransformer(
        corpus.vocab_size,
        dim=config["dim"],
        layers=config["layers"],
        heads=config["heads"],
        mlp=config["mlp"],
        seq=SEQ_LEN,
    )

    # Créer une permutation des tokens pour l'entraînement
    perm = torch.randperm(corpus.vocab_size, generator=torch.Generator().manual_seed(77))

    # Entraînement avec tokens mélangés (on modifie le batch)
    opt = torch.optim.AdamW(model.parameters(), lr=DENSE_LR, weight_decay=0.01)
    best = {"val": float("inf"), "step": 0, "state": None}
    stale = 0
    generator = torch.Generator().manual_seed(config["seed"])
    started = time.time()

    model.train()
    for step in range(1, TRAIN_STEPS + 1):
        x, y = corpus.batch(BATCH_SIZE, generator=generator)

        # Mélanger les tokens dans chaque séquence (contexte différent)
        x_permuted = perm[x]

        logits = model(x_permuted)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % 100 == 0 or step == 1:
            model.eval()
            with torch.no_grad():
                total = 0.0
                for vx, vy in val:
                    out = model(vx)
                    total += float(F.cross_entropy(out.reshape(-1, out.shape[-1]), vy.reshape(-1)))
            model.train()
            value = total / max(1, len(val))
            if value < best["val"]:
                best = {"val": value, "step": step, "state": {k: v.detach().clone() for k, v in model.state_dict().items()}}
                stale = 0
            else:
                stale += 1

            if stale >= 6:
                break

    if best["state"] is not None:
        model.load_state_dict(best["state"])

    result = {
        "curve": [],
        "seconds": round(time.time() - started, 1),
        "final_val": round(float(best["val"]), 4),
        "best_step": int(best["step"]),
    }
    save_dense(model, "C", result)
    print(f"[C] dense mélangé done: val={result['final_val']:.4f}, params={n_parameters(model)}")
    return "C", model, result


def train_donor_D(corpus, val):
    """
    DONNEUR-D : NUMENIA pool 24 entraînée directement.

    C'est le test symbolique : un cerveau sparse comme donneur.
    On utilise build_repaired + train_repaired de repaired_mycelia.
    """
    config = DONOR_CONFIGS["D"]
    set_seed(config["seed"])

    brain = build_repaired(
        corpus.vocab_size,
        seed=config["seed"],
        two_speeds=True,
        overrides={
            "n_nodes": config["pool"],
            "max_edges": config["max_edges"],
            "max_population": config["pool"],
            "max_node_births": 0,
        },
    )

    result = train_repaired(
        brain, corpus, TRAIN_STEPS,
        lr=NUMENIA_LR, batch=BATCH_SIZE, seed=config["seed"],
        log_every=100, weight_decay=NUMENIA_WD, val_batches=val,
        lifecycle_every=100,
        restore_best=True,
    )

    brain = result["brain"]
    save_brain(DONORS_DIR / "D.pt", brain, step=result["best_step"], extra={
        "type": "brain",
        "seed": config["seed"],
        "pool": config["pool"],
        "max_edges": config["max_edges"],
        "two_speeds": True,
        "val": result["best_val"],
    })

    cols = columns_mycelia(brain, val)
    print(f"[D] NUMENIA pool24 done: val={cols['nouvelle']:.4f}, params={sum(p.numel() for p in brain.parameters())}")
    return "D", brain, result


def save_dense(model, tag, result):
    """Sauvegarder un dense avec sa config."""
    torch.save({
        "state_dict": model.state_dict(),
        "config": {
            "type": "dense",
            "dim": model.dim,
            "layers": model.blocks.__len__(),
            "heads": model.blocks[0].heads,
            "mlp": model.blocks[0].fc1.out_features,
            "seq": model.seq,
        },
        "vocab_size": int(model.vocab),
        "extra": {
            "val": result["final_val"],
            "seed": DONOR_CONFIGS[tag]["seed"],
            "label": DONOR_CONFIGS[tag]["label"],
        },
    }, DONORS_DIR / f"{tag}.pt")


def compute_divergence_matrix(donors_dict, corpus, val):
    """
    D2. Matrice KL croisée 4×4 sur 5000 contextes + top-3 identique.
    """
    tags = ["A", "B", "C", "D"]
    n_contextes = 5000
    batch_size = 64

    # Pré-calculer les logits pour chaque donneur
    print("[D2] Calcul des logits pour chaque donneur...")
    all_logits = {}
    for tag in tags:
        path = DONORS_DIR / f"{tag}.pt"
        blob = torch.load(str(path), weights_only=False)
        config = blob["config"]

        if config["type"] == "dense":
            from micro_dense import DenseTransformer
            model = DenseTransformer(
                int(blob["vocab_size"]),
                dim=config["dim"],
                layers=config["layers"],
                heads=config["heads"],
                mlp=config["mlp"],
                seq=config["seq"],
            )
            model.load_state_dict(blob["state_dict"])
            model.eval()
        else:  # brain
            model = load_brain(path)
            model.eval()

        # Calculer les logits sur le corpus de validation
        logits_list = []
        for x, _ in val:
            with torch.no_grad():
                if config["type"] == "dense":
                    logits = model(x)
                else:
                    logits, _ = model(x, grow=False, trace=False)
                logits_list.append(logits)
        all_logits[tag] = torch.cat(logits_list, dim=0)
        print(f"  [{tag}] logits shape: {all_logits[tag].shape}")

    # Matrice KL
    print("[D2] Calcul de la matrice KL...")
    kl_matrix = {}
    top3_matrix = {}
    for i, tag_a in enumerate(tags):
        for j, tag_b in enumerate(tags):
            if i == j:
                continue
            key = f"{tag_a}_vers_{tag_b}"
            kl = kl_direction(all_logits[tag_a], all_logits[tag_b])
            kl_matrix[key] = round(float(kl), 4)

            # Top-3 overlap
            flat_a = all_logits[tag_a].reshape(-1, all_logits[tag_a].shape[-1])
            flat_b = all_logits[tag_b].reshape(-1, all_logits[tag_b].shape[-1])
            overlap = topk_overlap(flat_a, flat_b)
            top3_matrix[key] = {
                "identiques": round(float(overlap["identiques"]), 4),
                "recouvrement": round(float(overlap["recouvrement"]), 4),
            }

    print(f"[D2] Matrice KL: {json.dumps(kl_matrix, indent=2)}")

    return {
        "donneurs": tags,
        "kl_matrix": kl_matrix,
        "top3_matrix": top3_matrix,
        "vocab_size": int(corpus.vocab_size),
        "contextes": n_contextes,
    }


def main():
    parser = argparse.ArgumentParser(description="S8 : entraînement des 4 donneurs divergents")
    parser.add_argument("--donors", nargs="+", default=["A", "B", "C", "D"], help="Quels donneurs entraîner")
    parser.add_argument("--divergence", action="store_true", help="Calculer la matrice de divergence après entraînement")
    parser.add_argument("--all", action="store_true", help="Tout faire : entraînement + divergence + export")
    args = parser.parse_args()

    print("=" * 60)
    print("S8 — Banc multi-donneurs : entraînement donneurs")
    print("=" * 60)

    corpus, val = load_corpus()
    print(f"Corpus: {len(corpus.lines)} phrases, vocab={corpus.vocab_size}, val={len(val)} fenêtres")
    print(f"Donors dir: {DONORS_DIR}")

    tags_to_train = args.donors

    if "all" in args.donors or args.all:
        tags_to_train = ["A", "B", "C", "D"]

    # Entraînement des donneurs
    if tags_to_train:
        print(f"\nEntraînement des donneurs: {tags_to_train}")
        for tag in tags_to_train:
            if tag == "A":
                train_donor_A(corpus, val)
            elif tag == "B":
                train_donor_B(corpus, val)
            elif tag == "C":
                train_donor_C(corpus, val)
            elif tag == "D":
                train_donor_D(corpus, val)
            else:
                print(f"[!] Tag inconnu: {tag}")

    # Calcul de la divergence
    if args.divergence or args.all:
        print("\n" + "=" * 60)
        print("D2 — Mesure de la divergence")
        print("=" * 60)
        donors_dict = {}
        for tag in ["A", "B", "C", "D"]:
            path = DONORS_DIR / f"{tag}.pt"
            if path.exists():
                blob = torch.load(str(path), weights_only=False)
                donors_dict[tag] = blob

        if len(donors_dict) >= 2:
            divergence = compute_divergence_matrix(donors_dict, corpus, val)
            write_json(LAB / "s8" / "divergence.json", divergence)
            print(f"\nMatrice de divergence écrite dans data/lab/s8/divergence.json")

            # Carte d'identité de chaque donneur
            print("\nCarte d'identité des donneurs:")
            for tag in ["A", "B", "C", "D"]:
                path = DONORS_DIR / f"{tag}.pt"
                if path.exists():
                    blob = torch.load(str(path), weights_only=False)
                    config = blob.get("config", {})
                    extra = blob.get("extra", {})
                    n_params = sum(p.numel() for p in (blob.get("state_dict", {}).values()))

                    # Calculer la perte sur la validation
                    model_type = config.get("type", "unknown")
                    if model_type == "dense":
                        from micro_dense import DenseTransformer
                        model = DenseTransformer(
                            int(blob["vocab_size"]),
                            dim=config["dim"],
                            layers=config["layers"],
                            heads=config["heads"],
                            mlp=config["mlp"],
                            seq=config["seq"],
                        )
                        model.load_state_dict(blob["state_dict"])
                        model.eval()
                        perte_nouvelle = columns_dense(model, val)["nouvelle"]
                        perte_ancienne = columns_dense(model, val)["ancienne"]
                    else:
                        model = load_brain(path)
                        model.eval()
                        perte_nouvelle = columns_mycelia(model, val)["nouvelle"]
                        perte_ancienne = columns_mycelia(model, val)["ancienne"]

                    identity = {
                        "tag": tag,
                        "label": DONOR_CONFIGS[tag]["label"],
                        "type": model_type,
                        "config": config,
                        "perte_nouvelle": round(perte_nouvelle, 4),
                        "perte_ancienne": round(perte_ancienne, 4),
                        "n_parameters": n_params,
                        "vocab_size": int(blob.get("vocab_size", corpus.vocab_size)),
                        "epoch": extra.get("val", "N/A"),
                    }
                    write_json(LAB / "s8" / f"identity_{tag}.json", identity)
                    print(f"  [{tag}] {DONOR_CONFIGS[tag]['label']}: perte_nouvelle={perte_nouvelle:.4f}, perte_ancienne={perte_ancienne:.4f}, params={n_params}")

                    # Sauvegarder aussi la config détaillée
                    if "config" not in blob:
                        blob["config"] = config
                    torch.save(blob, DONORS_DIR / f"{tag}.pt")
        else:
            print("[!] Pas assez de donateurs entraînés pour calculer la divergence")

    print("\n" + "=" * 60)
    print("S8 — Entraînement des donneurs terminé")
    print("=" * 60)
    print(f"Donors sauvegardés dans: {DONORS_DIR}")
    print("Fichiers:")
    for f in DONORS_DIR.iterdir():
        if f.suffix == ".pt":
            print(f"  - {f.name} ({f.stat().st_size} octets)")


if __name__ == "__main__":
    main()
