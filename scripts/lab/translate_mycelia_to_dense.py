"""S4 — la traduction inverse : micro-MYCELIA -> micro-transformer dense.

Un vrai traducteur va dans les deux sens. Meme protocole que S3, avec deux
differences imposees par le receveur :

* le dense n'a pas de « naissance de noeud » : il a une **largeur cachee**. Un
  neurone ajoute a ``fc1``/``fc2`` recoit le concept dans sa ligne d'entree et
  ecrit zero dans sa colonne de sortie, donc la fonction du modele est
  exactement preservee jusqu'au premier pas d'affinage ;
* les noeuds du numenia qui portent le concept sont adresses par ``k_b``, et le
  dense ecoute une direction de son espace residuel (ligne de ``fc1``). La
  projection ridge relie les deux espaces, comme dans l'autre sens.

Verdict : perte AVANT (apres amorcage du receveur), APRÈS 100 pas avec greffe,
et le meme budget sans greffe, sur les memes lots. Puis les 20 phrases neuves.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from lab_common import (
    LAB,
    ToyCorpus,
    align,
    dense_loss,
    ensure_lab,
    kmeans,
    profiles,
    set_seed,
    write_json,
)
from micro_dense import build_model, n_parameters, train_dense
from micro_mycelia import load_mycelia
from translate_dense_to_mycelia import family_profiles, load_dense, shared_stimuli
from translate_lib import (
    MLP,
    arm_dense,
    batches_loss_dense,
    concept_direction,
    dense_clone,
    dense_directions,
    heldout_sentences,
    make_batches,
    mycelia_directions,
    projection,
    sandbox_texts,
    scan_dense,
    scan_mycelia,
    summary,
    test_dense,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="conversion numenia -> dense (S4)")
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--seed", type=int, default=99)
    parser.add_argument("--seuil", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--amorce", type=int, default=50, help="pas d'amorcage du dense")
    args = parser.parse_args()

    ensure_lab()
    set_seed()
    corpus = ToyCorpus(seq_len=8)
    brain = load_mycelia()
    refined = load_dense()
    texts = sandbox_texts(corpus, 500)

    # amorcage du receveur dense : meme raison qu'en S3
    model = build_model(corpus.vocab_size, 7)
    warm = train_dense(model, corpus, args.amorce, batch=32, seed=7, log_every=max(1, args.amorce), patience=0)
    print(
        f"amorcage receveur : {args.amorce} pas, val {warm['final_val']:.4f} "
        f"(reference convergee : data/lab/micro_dense.pt, val 4,4869)"
    )

    # 1. cartographie source : familles de noeuds du numenia
    scan_source = scan_mycelia(brain, corpus, texts)
    alive = brain.node_alive.cpu().numpy()
    nodes = np.nonzero(alive)[0]
    block = scan_source[:, nodes]
    nodes = nodes[block.max(axis=0) > 0]
    block = block[:, block.max(axis=0) > 0]
    labels_s, centres_s, inertia_s = kmeans(profiles(block), args.k, seed=11)
    directions_s = mycelia_directions(brain)[nodes]
    prof_s = family_profiles(block, labels_s, args.k)
    members_s = {
        family: nodes[np.nonzero(labels_s == family)[0]] for family in range(int(args.k))
    }

    # 2. cartographie receveur : neurones caches du dense
    scan_receiver = scan_dense(model, corpus, texts)[:, : 2 * MLP]
    labels_r, centres_r, inertia_r = kmeans(profiles(scan_receiver), args.k, seed=11)
    directions_r = dense_directions(model)[: 2 * MLP]
    prof_r = family_profiles(scan_receiver, labels_r, args.k)

    # 3. alignement mutuel
    matching = align(prof_s, prof_r, threshold=args.seuil)
    pairs = matching["pairs"]
    matched_s = [i for i, _, _ in pairs]
    matched_r = [j for _, j, _ in pairs]

    src_dirs = (
        np.stack([concept_direction(directions_s, np.nonzero(labels_s == i)[0]) for i in matched_s])
        if len(matched_s) >= 2
        else np.zeros((0, directions_s.shape[1]))
    )
    dst_dirs = (
        np.stack([concept_direction(directions_r, np.nonzero(labels_r == j)[0]) for j in matched_r])
        if len(matched_r) >= 2
        else np.zeros((0, directions_r.shape[1]))
    )
    W = (
        projection(src_dirs, dst_dirs)
        if src_dirs.shape[0] >= 2
        else np.eye(directions_s.shape[1], directions_r.shape[1], dtype=np.float32)
    )

    table = []
    for i, j, score in pairs:
        table.append(
            {
                "famille_source": int(i),
                "concept_receveur": int(j),
                "similarite": round(float(score), 4),
                "noeuds_source": int(members_s[i].size),
                "stimuli_communs": shared_stimuli(prof_s[i], prof_r[j], texts),
            }
        )

    # 4. transplant : un neurone par famille sans equivalent
    co = prof_s @ prof_s.T
    grafted = dense_clone(model)
    implanted = []
    for family in matching["source_unmatched"]:
        vec = W @ concept_direction(directions_s, np.nonzero(labels_s == family)[0])
        before_width = int(grafted.blocks[0].fc1.out_features)
        grafted.grow_mlp(0, torch.as_tensor(vec))
        implanted.append(
            {
                "famille": int(family),
                "neurone": before_width,
                "largeur_avant": before_width,
                "largeur_apres": int(grafted.blocks[0].fc1.out_features),
                "norme_projetee": round(float(np.linalg.norm(vec)), 4),
                "familles_coactives": [
                    int(i)
                    for i in range(len(prof_s))
                    if float(co[family, i]) >= 0.5 and i != family
                ],
            }
        )

    # verdict
    val_batches = corpus.fixed(24, "val")
    batches = make_batches(corpus, args.steps, args.batch, args.seed)
    before = {
        "val": round(float(dense_loss(model, val_batches)), 4),
        "train": round(float(batches_loss_dense(model, batches[:10])), 4),
    }
    grafted_arm = arm_dense(grafted, batches, val_batches, lr=args.lr)
    control_arm = arm_dense(dense_clone(model), batches, val_batches, lr=args.lr)

    sentences = heldout_sentences(corpus, 20)
    lang_before = test_dense(model, corpus, sentences)
    lang_grafted = test_dense(grafted, corpus, sentences)
    lang_control = test_dense(control_arm["model"], corpus, sentences)

    verdict = summary(before, grafted_arm, control_arm)
    verdict.update(
        {
            "language_avant": {"perte": lang_before["perte"], "exactitude": lang_before["exactitude"]},
            "language_greffe": {"perte": lang_grafted["perte"], "exactitude": lang_grafted["exactitude"]},
            "language_controle": {"perte": lang_control["perte"], "exactitude": lang_control["exactitude"]},
            "phrases_gagnees_vs_controle": int(
                sum(
                    1
                    for a, b in zip(lang_grafted["exemples"], lang_control["exemples"])
                    if a["perte"] < b["perte"]
                )
            ),
            "phrases": len(sentences),
        }
    )

    write_json(
        LAB / "conversion_mycelia_to_dense.json",
        {
            "sens": "numenia -> dense",
            "meta": {
                "stimuli": len(texts),
                "k": int(args.k),
                "seuil": float(args.seuil),
                "steps": int(args.steps),
                "batch": int(args.batch),
                "amorce": int(args.amorce),
                "inertie_source": round(float(inertia_s), 4),
                "inertie_receveur": round(float(inertia_r), 4),
                "noeuds_source": int(nodes.size),
                "neurones_receveur": int(2 * MLP),
                "apparies": len(pairs),
                "parametres_avant": int(n_parameters(model)),
                "parametres_greffe": int(n_parameters(grafted)),
            },
            "table_alignement": table,
            "familles_sans_equivalent": [int(f) for f in matching["source_unmatched"]],
            "concepts_sans_equivalent": [int(c) for c in matching["receiver_unmatched"]],
            "transplants": implanted,
            "verdict": verdict,
            "courbes": {"greffe": grafted_arm["curve"], "controle": control_arm["curve"]},
            "predictions_greffe": lang_grafted["exemples"],
            "predictions_controle": lang_control["exemples"],
        },
    )

    print(
        f"familles {args.k} / concepts {args.k} : {len(pairs)} appariements "
        f"(seuil {args.seuil}), {len(matching['source_unmatched'])} sans equivalent"
    )
    print(f"transplants : {len(implanted)} neurones, largeur "
          f"{grafted_arm['mlp']} parametres {n_parameters(model)} -> {n_parameters(grafted)}")
    print(f"AVANT  val {before['val']:.4f}  train {before['train']:.4f}")
    print(f"GREFFE val {grafted_arm['val']:.4f}  (courbe {[c['val'] for c in grafted_arm['curve']]})")
    print(f"CONTROLE val {control_arm['val']:.4f}  (courbe {[c['val'] for c in control_arm['curve']]})")
    print(f"verdict : {verdict}")
    print("ecrit : data/lab/conversion_mycelia_to_dense.json")


if __name__ == "__main__":
    main()
