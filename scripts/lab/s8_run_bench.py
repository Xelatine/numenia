#!/usr/bin/env python3
"""
S8 — Banc multi-donneurs simplifié : 12 traductions (4 donateurs x 3 semences)
================================================================================

Protocole simplifié (sans vie) : pour chaque donneur, créer un receveur NUMENIA
pool 24 vierge, le distiller a1b1g0, mesurer F avant vie (KL + top-3).
"""
import argparse
import json
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))
sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn.functional as F

from lab_common import LAB, ToyCorpus, set_seed, write_json
from s5_common import kl_direction, topk_overlap, columns_mycelia, L_REF
from repaired_mycelia import build_repaired, train_repaired, save_brain, teacher_parts, load_brain
from dense_model import load_dense_donor

# Configuration
CORPUS_PATH = LAB / "corpus_toy.txt"

# Mapping label/type pour l'affichage (la marque voyage dans les titres, jamais dans les identifiants).
DONOR_CONFIG_MAPPING = {
    "A": {"type": "dense", "label": "Dense A — lignée S5 (2 couches)"},
    "B": {"type": "dense", "label": "Dense B — variant graine+arch (3 couches)"},
    "C": {"type": "dense", "label": "Dense C — ordre mélangé (contextualisation)"},
    "D": {"type": "brain", "label": "MYCELIA pool 24 — cerveau reparatrice (seuil)"},
}
SEQ_LEN = 8
VAL_WINDOWS = 100
TRAIN_STEPS = 600
BATCH_SIZE = 16
DENSE_LR = 3e-3
NUMENIA_LR = 2e-3
NUMENIA_WD = 0.01
RECIPIENT_SEEDS = [1, 2, 3]
DONORS_DIR = LAB / "s8" / "donors"
BENCH_DIR = LAB / "s8"
BENCH_DIR.mkdir(parents=True, exist_ok=True)

# Chargement des donneurs (module partagé strict pour les denses).

def load_all_donors():
    """Charge A/B/C (dense) avec load_dense_donor strict, D (brain) avec load_brain."""
    donateurs = {}
    for tag in ["A", "B", "C"]:
        path = DONORS_DIR / f"{tag}.pt"
        if not path.exists():
            print(f"[!] {tag} non trouve, saut")
            continue
        m = load_dense_donor(path)
        donateurs[tag] = m
        print(f"  [{tag}] dense charge: layers={len(m.blocks)}, mlp={m.blocks[0].fc1.out_features}, params={sum(p.numel() for p in m.parameters())}")
    path = DONORS_DIR / "D.pt"
    if path.exists():
        brain = load_brain(path)
        donateurs["D"] = brain
        print(f"  [D] brain charge: n_nodes={brain.n_nodes}, ne={brain.ne}")
    return donateurs

def _teacher_logits(teacher, x):
    """Logits du professeur, alignes sur les pas 1..seq-1 de MYCELIA.

    Pour un dense : ``teacher.parts(x)`` rend (hidden, logits) ; on prend
    logits[:, :-1] pour predire le token t a partir du prefixe 0..t-1.
    Pour un brain repare : on utilise teacher_parts du module repaired.
    """
    if hasattr(teacher, "parts") and not hasattr(teacher, "_edge_mask"):
        # Dense jouet.
        _, logits = teacher.parts(x)
        return logits[:, :-1]
    if hasattr(teacher, "_edge_mask"):
        # Brain repare : teacher_parts sait aligner.
        t_logits, _ = teacher_parts(teacher, x)
        return t_logits
    # Fallback dense sans parts.
    out = teacher(x)
    if isinstance(out, tuple):
        out = out[0]
    return out[:, :-1]


def compute_fidelity(brain, teacher, corpus, val):
    """Calculer KL(teacher ‖ brain) et top-3 agreement (avant vie)."""
    tot_kl = 0.0
    count = 0
    tot_identiques = 0.0
    tot_recouvrement = 0.0

    generator = torch.Generator().manual_seed(4242)
    for _ in range(50):
        x, _ = corpus.batch(64, generator=generator)
        with torch.no_grad():
            teacher_logits_p1 = _teacher_logits(teacher, x) if teacher is not None else None
            # brain(x) rend (logits, state[, trace]) ; on prend les logits.
            out = brain(x, grow=False, trace=False)
            brain_logits = out[0] if isinstance(out, tuple) else out
            brain_logits_p1 = brain_logits[:, 1:]

            if teacher_logits_p1 is not None:
                kl = kl_direction(teacher_logits_p1, brain_logits_p1)
                tot_kl += kl
                count += 1

                flat_t = teacher_logits_p1.reshape(-1, teacher_logits_p1.shape[-1])
                flat_b = brain_logits_p1.reshape(-1, brain_logits_p1.shape[-1])
                overlap = topk_overlap(flat_t, flat_b)
                tot_identiques += overlap.get("identiques", 0)
                tot_recouvrement += overlap.get("recouvrement", 0)

    if count > 0:
        kl_avg = tot_kl / max(1, count)
        top3_identiques = tot_identiques / max(1, count)
        top3_recouvrement = tot_recouvrement / max(1, count)
    else:
        kl_avg = 0.0
        top3_identiques = 0.0
        top3_recouvrement = 0.0

    # Top-3 sur phrases neuves (token par token).
    sentences = []
    for line in corpus.lines[:50]:
        tokens = [corpus.stoi.get(w, 0) for w in line.split()[:SEQ_LEN]]
        if len(tokens) >= 3:
            sentences.append(tokens)

    top3_hits = 0
    top3_total = 0
    brain.eval()
    for tokens in sentences:
        state = brain._fresh_state(1)
        for i, token in enumerate(tokens):
            if i == 0:
                continue
            out = brain(
                torch.tensor([[token]], dtype=torch.long),
                state=state, grow=False, trace=False, return_state=True,
            )
            logits = out[0] if isinstance(out, tuple) else out
            state = out[1] if isinstance(out, tuple) and len(out) > 1 else state
            top3 = torch.topk(logits[0, 0], 3).indices.tolist()
            if token in top3:
                top3_hits += 1
            top3_total += 1

    top3_neuf = top3_hits / max(1, top3_total)

    return {
        "kl_teacher_vers_brain": round(float(kl_avg), 4),
        "top3_identiques": round(float(top3_identiques), 4),
        "top3_recouvrement": round(float(top3_recouvrement), 4),
        "top3_neuf": round(float(top3_neuf), 4),
        "contextes": count * 64 * (SEQ_LEN - 1),
    }


def train_recipient(teacher, corpus, val, seed):
    """Créer un receveur NUMENIA pool 24 VIERGE et le distiller avec a1b1g0."""
    set_seed(seed)
    brain = build_repaired(
        corpus.vocab_size,
        seed=seed,
        two_speeds=True,
        overrides={
            "n_nodes": 24,
            "max_edges": 220,
            "max_population": 24,
            "max_node_births": 0,
        },
    )

    result = train_repaired(
        brain, corpus, TRAIN_STEPS,
        lr=NUMENIA_LR, batch=BATCH_SIZE, seed=seed,
        log_every=100, weight_decay=NUMENIA_WD, val_batches=val,
        lifecycle_every=100,
        teacher=teacher,
        alpha=1.0, beta=1.0, gamma=0.0,  # a1b1g0
        temperature=1.0,
        projector_init="random",
        restore_best=True,
    )

    return brain, result


def main():
    parser = argparse.ArgumentParser(description="S8 : banc multi-donneurs (simplifié)")
    parser.add_argument("--donors", nargs="+", default=["A", "B", "C", "D"], help="Quels donneurs traduire")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3], help="Semences de receveur")
    parser.add_argument("--all", action="store_true", help="Tout faire (défaut)")
    
    args = parser.parse_args()
    if args.all:
        args.donors = ["A", "B", "C", "D"]
        args.seeds = [1, 2, 3]
    

    print("=" * 60)
    print("S8 — Banc multi-donneurs (simplifié)")
    print("=" * 60)

    # Corpus
    corpus = ToyCorpus(seq_len=SEQ_LEN, path=CORPUS_PATH)
    val = corpus.fixed(VAL_WINDOWS, "val")
    print(f"Corpus: {len(corpus.lines)} phrases, vocab={corpus.vocab_size}, val={len(val)} fenêtres")

    # Charger les donateurs
    donateurs = load_all_donors()
    print(f"Donateurs chargés: {list(donateurs.keys())}")

    # Lancer les traductions
    results = {}

    for tag in args.donors:
        if tag not in donateurs:
            print(f"[!] Donneur {tag} non trouvé, saut")
            continue
        
        teacher = donateurs[tag]
        donor_cfg = DONOR_CONFIG_MAPPING.get(tag, {})
        
        print(f"\n[{tag}] Début de la traduction (type={donor_cfg.get('type')}, label={donor_cfg.get('label')})...")
        
        for seed in args.seeds:
            bench_file = BENCH_DIR / f"bench_{tag}_seed{seed}.json"

            if bench_file.exists():
                print(f"  [{tag}/seed{seed}] Déjà fait, reprise...")
                with open(bench_file) as f:
                    results.setdefault(tag, {})[seed] = json.load(f)
                continue

            print(f"  [{tag}/seed{seed}] Création du receveur NUMENIA vierge...")
            brain, train_result = train_recipient(teacher, corpus, val, seed)
            print(f"    Receveur créé: pool={brain.n_nodes}, aretes={brain.ne}")
            print(f"    Entraînement: best_val={train_result['best_val']:.4f}, best_step={train_result['best_step']}, seconds={train_result['seconds']}")

            print(f"  [{tag}/seed{seed}] Mesure F avant vie...")
            before_vie = compute_fidelity(brain, teacher, corpus, val)
            before_vie["perte"] = columns_mycelia(brain, val)["nouvelle"]
            before_vie["R"] = 0.0  # simplifié

            print(f"    KL={before_vie['kl_teacher_vers_brain']:.4f}, top3_identiques={before_vie['top3_identiques']:.4f}, top3_neuf={before_vie['top3_neuf']:.4f}")

            # Sauvegarder
            print(f"  [{tag}/seed{seed}] Sauvegarde...")
            brain_path = BENCH_DIR / f"recipient_{tag}_seed{seed}.pt"
            save_brain(brain_path, brain, step=train_result["best_step"], extra={
                "type": "brain",
                "seed": seed,
                "teacher_tag": tag,
            })

            results.setdefault(tag, {})[seed] = {
                "donneur_tag": tag,
                "donneur_label": donor_cfg.get("label", f"donneur-{tag}"),
                "donneur_type": donor_cfg.get("type", "unknown"),
                "donneur_params": sum(p.numel() for p in teacher.parameters()) if hasattr(teacher, 'parameters') else 0,
                "seed_receveur": seed,
                "entrainement": {
                    "best_val": train_result["best_val"],
                    "best_step": train_result["best_step"],
                    "seconds": train_result["seconds"],
                },
                "avant_vie": before_vie,
            }

            bench_file.write_text(json.dumps(results[tag][seed], indent=2), encoding="utf-8")
            print(f"    Sauvegardé dans {bench_file}")

    # Table maîtresse simplifiée
    print("\n" + "=" * 60)
    print("Table maîtresse simplifiée")
    print("=" * 60)

    table = []
    for tag in args.donors:
        if tag not in results:
            continue
        for seed in args.seeds:
            if seed not in results[tag]:
                continue

            r = results[tag][seed]
            row = {
                "donneur": tag,
                "donneur_label": r["donneur_label"],
                "donneur_type": r["donneur_type"],
                "seed_receveur": seed,
                "F_avant": r["avant_vie"].get("kl_teacher_vers_brain"),
                "top3_avant": r["avant_vie"].get("top3_identiques"),
                "top3_neuf": r["avant_vie"].get("top3_neuf"),
                "perte_avant": r["avant_vie"]["perte"],
            }
            table.append(row)
            print(f"  [{tag}/seed{seed}] F_avant={row['F_avant']}, top3_avant={row['top3_avant']}, top3_neuf={row['top3_neuf']}")

    write_json(BENCH_DIR / "table_maitresse.json", {
        "donneurs": args.donors,
        "seeds": args.seeds,
        "table": table,
    })

    print(f"\nTable maîtresse écrite dans {BENCH_DIR / 'table_maitresse.json'}")
    print("\n" + "=" * 60)
    print("S8 — Banc terminé")
    print("=" * 60)


if __name__ == "__main__":
    main()
