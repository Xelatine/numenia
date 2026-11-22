"""S3 — la premiere conversion : micro-transformer dense -> micro-MYCELIA.

Quatre phases, dans cet ordre :

1. **Cartographie source** : les 128 neurones caches du dense sont presentes aux
   500 phrases du bac ; leur profil d'activation les groupe en concepts.
2. **Cartographie receveur** : le meme scan sur NUMENIA donne des familles de
   noeuds (c'est S1, applique au bac).
3. **Alignement fonctionnel** : chaque concept source est apparie a la famille
   receveuse qui repond aux memes phrases. L'alignement est semantique (meme
   profil sur les memes stimuli), jamais structurel — c'est la regle du
   traducteur universel.
4. **Transplant** : les concepts sans equivalent font naitre un noeud dans
   NUMENIA, avec l'embedding projete (projection lineaire ridge 16 -> 16) et des
   aretes vers les familles co-activees.

Puis le verdict chiffre : perte AVANT, perte APRES 100 pas d'affinage, et le
meme budget pour un controle parti du meme point sans greffe. Les deux bras
lisent exactement les memes lots (le generateur est seme une fois).
"""

from __future__ import annotations

import argparse

import numpy as np

from lab_common import (
    LAB,
    ToyCorpus,
    align,
    ensure_lab,
    kmeans,
    mycelia_loss,
    profiles,
    set_seed,
    write_json,
)
from micro_dense import DenseTransformer
from micro_mycelia import build_mycelia, load_mycelia, train_mycelia
from translate_lib import (
    MLP,
    arm_mycelia,
    batches_loss_mycelia,
    concept_direction,
    dense_directions,
    heldout_sentences,
    implant_node,
    make_batches,
    mycelia_clone,
    mycelia_directions,
    projection,
    sandbox_texts,
    scan_dense,
    scan_mycelia,
    summary,
    test_mycelia,
)


def load_dense(path=None):
    blob = __import__("torch").load(path or str(LAB / "micro_dense.pt"), weights_only=False)
    config = dict(blob["config"])
    model = DenseTransformer(len(blob["vocab"]), **config)
    model.load_state_dict(blob["state_dict"])
    model.eval()
    return model


def family_profiles(matrix: np.ndarray, labels: np.ndarray, k: int) -> np.ndarray:
    """Profil moyen d'une famille sur les 500 stimuli, normalise en L2.

    La famille est un ensemble d'unites : son profil est la moyenne de leurs
    profils (deja normalises), renormalisee pour que la similarite cosinus reste
    une comparaison de formes et pas d'amplitudes.
    """
    normalised = profiles(matrix)  # [unites, stimuli]
    rows = []
    for family in range(int(k)):
        members = np.nonzero(labels == family)[0]
        if members.size == 0:
            rows.append(np.zeros(normalised.shape[1], dtype=np.float32))
            continue
        mean = normalised[members].mean(axis=0)
        norm = np.linalg.norm(mean)
        rows.append(mean / max(1e-8, norm))
    return np.stack(rows).astype(np.float32)


def shared_stimuli(
    profile_s: np.ndarray, profile_r: np.ndarray, texts: list[str], top: int = 3
) -> list[dict]:
    """Les phrases qui activent les deux membres de la paire (le dictionnaire)."""
    both = np.minimum(profile_s, profile_r)
    order = np.argsort(-both)[: int(top)]
    return [
        {"texte": texts[int(i)], "force": round(float(both[int(i)]), 4)}
        for i in order
        if both[int(i)] > 0
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="conversion dense -> numenia (S3)")
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--seed", type=int, default=99)
    parser.add_argument("--seuil", type=float, default=0.5, help="seuil d'appariement")
    # 3e-4 : le point de depart est deja au minimum de validation ; un lr plus
    # haut ferait surapprendre les deux bras et effacerait la comparaison.
    parser.add_argument("--lr", type=float, default=1e-3)
    # 50 pas d'amorcage : converti depuis son optimum (val 5,41), le receveur ne
    # peut que se degrader, et la comparaison ne mesure plus rien. A 50 pas il
    # reste du chemin a parcourir — c'est la que "converger plus vite" a un sens.
    parser.add_argument("--amorce", type=int, default=50)
    args = parser.parse_args()

    ensure_lab()
    set_seed()
    corpus = ToyCorpus(seq_len=8)
    model = load_dense()
    converged = load_mycelia()
    warm = train_mycelia(
        build_mycelia(corpus.vocab_size, 7),
        corpus,
        args.amorce,
        batch=args.batch,
        seed=7,
        log_every=max(1, args.amorce),
        lr=1e-3,
        weight_decay=0.05,
        patience=0,
    )
    brain = warm["brain"]
    print(f"amorcage receveur : {args.amorce} pas, val {warm['final_val']:.4f}, "
          f"pool {brain.n_nodes}, aretes {brain.ne} "
          f"(reference convergee : {converged.n_nodes} noeuds)")
    texts = sandbox_texts(corpus, 500)

    # 1. cartographie source -------------------------------------------------
    scan_source = scan_dense(model, corpus, texts)[:, : 2 * MLP]
    labels_s, centres_s, inertia_s = kmeans(profiles(scan_source), args.k, seed=11)
    directions_s = dense_directions(model)[: 2 * MLP]
    prof_s = family_profiles(scan_source, labels_s, args.k)

    # 2. cartographie receveur -----------------------------------------------
    scan_receiver = scan_mycelia(brain, corpus, texts)
    alive = brain.node_alive.cpu().numpy()
    nodes = np.nonzero(alive)[0]
    block = scan_receiver[:, nodes]
    nodes = nodes[block.max(axis=0) > 0]
    block = block[:, block.max(axis=0) > 0]
    labels_r, centres_r, inertia_r = kmeans(profiles(block), args.k, seed=11)
    directions_r = mycelia_directions(brain)[nodes]
    prof_r = family_profiles(block, labels_r, args.k)
    members_r = {
        family: nodes[np.nonzero(labels_r == family)[0]] for family in range(int(args.k))
    }

    # 3. alignement fonctionnel ----------------------------------------------
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
                "concept_source": int(i),
                "famille_receveuse": int(j),
                "similarite": round(float(score), 4),
                "noeuds_receveurs": int(members_r[j].size),
                "stimuli_communs": shared_stimuli(prof_s[i], prof_r[j], texts),
            }
        )

    # 4. transplant ----------------------------------------------------------
    co = prof_s @ prof_s.T
    grafted = mycelia_clone(brain)
    implanted = []
    for concept in matching["source_unmatched"]:
        vec = W @ concept_direction(directions_s, np.nonzero(labels_s == concept)[0])
        partners: list[int] = []
        for other, (i, j, _score) in enumerate(pairs):
            if float(co[concept, i]) >= 0.5:
                partners.extend(int(n) for n in members_r[j][:2])
        node = implant_node(grafted, vec, partners=partners)
        implanted.append(
            {
                "concept": int(concept),
                "noeud": int(node),
                "partenaires": int(len(partners)),
                "norme_projetee": round(float(np.linalg.norm(vec)), 4),
                "concepts_coactifs": [
                    int(i) for i in range(len(prof_s)) if float(co[concept, i]) >= 0.5 and i != concept
                ],
            }
        )

    # verdict ----------------------------------------------------------------
    val_batches = corpus.fixed(24, "val")
    batches = make_batches(corpus, args.steps, args.batch, args.seed)
    before = {
        "val": round(float(mycelia_loss(brain, val_batches)), 4),
        "train": round(float(batches_loss_mycelia(brain, batches[:10])), 4),
    }
    grafted_arm = arm_mycelia(grafted, batches, val_batches, lr=args.lr)
    control_arm = arm_mycelia(mycelia_clone(brain), batches, val_batches, lr=args.lr)

    sentences = heldout_sentences(corpus, 20)
    lang_before = test_mycelia(brain, corpus, sentences)
    lang_grafted = test_mycelia(grafted, corpus, sentences)
    lang_control = test_mycelia(control_arm["brain"], corpus, sentences)

    verdict = summary(before, grafted_arm, control_arm)
    verdict.update(
        {
            "language_avant": {"perte": lang_before["perte"], "exactitude": lang_before["exactitude"]},
            "language_greffe": {"perte": lang_grafted["perte"], "exactitude": lang_grafted["exactitude"]},
            "language_controle": {"perte": lang_control["perte"], "exactitude": lang_control["exactitude"]},
            "phrases_gagnees_vs_controle": int(
                sum(1 for a, b in zip(lang_grafted["exemples"], lang_control["exemples"]) if a["perte"] < b["perte"])
            ),
            "phrases": len(sentences),
        }
    )

    write_json(
        LAB / "conversion_dense_to_mycelia.json",
        {
            "sens": "dense -> numenia",
            "meta": {
                "stimuli": len(texts),
                "k": int(args.k),
                "seuil": float(args.seuil),
                "steps": int(args.steps),
                "batch": int(args.batch),
                "inertie_source": round(float(inertia_s), 4),
                "inertie_receveur": round(float(inertia_r), 4),
                "neurones_source": int(2 * MLP),
                "noeuds_receveur": int(nodes.size),
                "apparies": len(pairs),
            },
            "table_alignement": table,
            "concepts_sans_equivalent": [int(c) for c in matching["source_unmatched"]],
            "familles_sans_concept": [int(f) for f in matching["receiver_unmatched"]],
            "transplants": implanted,
            "verdict": verdict,
            "courbes": {
                "greffe": grafted_arm["curve"],
                "controle": control_arm["curve"],
            },
            "predictions_greffe": lang_grafted["exemples"],
            "predictions_controle": lang_control["exemples"],
        },
    )

    print(f"concepts {args.k} / familles {args.k} : {len(pairs)} appariements "
          f"(seuil {args.seuil}), {len(matching['source_unmatched'])} concepts sans equivalent")
    print(f"transplants : {len(implanted)} noeuds, pool {brain.n_nodes} -> {grafted.n_nodes}")
    print(f"AVANT  val {before['val']:.4f}  train {before['train']:.4f}")
    print(f"GREFFE val {grafted_arm['val']:.4f}  (courbe {[c['val'] for c in grafted_arm['curve']]})")
    print(f"CONTROLE val {control_arm['val']:.4f}  (courbe {[c['val'] for c in control_arm['curve']]})")
    print(f"verdict : {verdict}")
    print("ecrit : data/lab/conversion_dense_to_mycelia.json")


if __name__ == "__main__":
    main()
