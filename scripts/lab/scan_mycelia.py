"""NUMENIA-SCAN v0 : cartographie du cerveau reel par stimulation.

Le principe est celui d'une electrode : on ne lit pas les poids (32 dimensions
par noeud ne veulent rien dire seuls), on lit **ce qui s'allume quand on parle**.
500 phrases francaises sont presentees une par une ; pour chacune on enregistre
l'activation de tous les noeuds vivants, c'est-a-dire la norme maximale de leur
membrane sur les tokens de la phrase.

Trois lectures sortent de cette matrice :

* un **atlas** : les noeuds sont regroupes en familles fonctionnelles (k-means
  sur leur profil d'activation), et chaque famille recoit ses dix stimuli les
  plus activateurs ;
* un **graphe de co-activation** compare au graphe structurel : les aretes
  cablees mais jamais co-actives, et les noeuds co-actifs sans arete. C'est la
  question scientifique du projet (la fonction suit-elle le cable ?) ;
* un **film** : cinq stimuli, token par token, pour l'onglet du dashboard.

Le cerveau n'est jamais modifie : ``grow=False``, aucun passage d'optimiseur,
et le genome reel n'est ouvert qu'en lecture.
"""

from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np
import torch

from lab_common import (
    LAB,
    STIMULI,
    ensure_lab,
    film_brain,
    kmeans,
    profiles,
    scan_brain,
    set_seed,
    write_json,
)
from mycelia.checkpoint import brain_from_genome, load_genome
from mycelia.data import TOKENIZER_PATH, load_tokenizer


def read_stimuli(path=None) -> list[tuple[str, str]]:
    """Lit ``categorie|texte`` ; une ligne sans barre est une phrase simple."""
    rows = []
    for line in (path or STIMULI).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        if "|" in line:
            category, text = line.split("|", 1)
        else:
            category, text = "phrase", line
        rows.append((category, text))
    return rows


def load_brain(genome: str):
    blob = load_genome(genome)
    brain = brain_from_genome(blob)
    brain.eval()
    return brain, blob


def degree_tables(brain) -> tuple[np.ndarray, np.ndarray]:
    """Degre sortant et entrant de chaque noeud, sur les aretes reellement usees."""
    n = int(brain.n_nodes)
    out_deg = np.zeros(n, dtype=np.int32)
    in_deg = np.zeros(n, dtype=np.int32)
    if int(brain.ne):
        src = brain.edge_index[0, : brain.ne].cpu().numpy()
        tgt = brain.edge_index[1, : brain.ne].cpu().numpy()
        np.add.at(out_deg, src, 1)
        np.add.at(in_deg, tgt, 1)
    return out_deg, in_deg


def coactivation(matrix: np.ndarray, top: int = 64) -> np.ndarray:
    """Compte les co-activations par paire sur les ``top`` noeuds de chaque stimulus.

    Pourquoi un top et pas toutes les paires : 514 noeuds font 132 000 paires,
    mais une phrase n'allume qu'une poignee de noeuds — compter les paires
    faibles reviendrait a compter du bruit. Les ``top`` noeuds les plus actifs
    sont ceux qui portent la reponse.
    """
    n = matrix.shape[1]
    counts = np.zeros((n, n), dtype=np.int32)
    for row in matrix:
        take = min(int(top), n)
        strongest = np.argpartition(-row, take - 1)[:take]
        strongest = strongest[row[strongest] > 0]
        counts[np.ix_(strongest, strongest)] += 1
    return counts


def build_atlas(
    labels: np.ndarray,
    nodes: np.ndarray,
    matrix: np.ndarray,
    stimuli: list[tuple[str, str]],
    out_deg: np.ndarray,
    in_deg: np.ndarray,
    top_stimuli: int = 10,
) -> list[dict]:
    """Une famille = ses noeuds, sa force moyenne, ses dix meilleurs stimuli."""
    families = []
    texts = [text for _, text in stimuli]
    categories = [category for category, _ in stimuli]
    for family in sorted(set(int(v) for v in labels)):
        members = np.array([i for i, v in enumerate(labels) if int(v) == family], dtype=int)
        if members.size == 0:
            continue
        node_ids = nodes[members]
        block = matrix[:, members]
        strength = block.mean(axis=1)
        order = np.argsort(-strength)[: int(top_stimuli)]
        families.append(
            {
                "famille": int(family),
                "noeuds": [int(n) for n in node_ids],
                "taille": int(members.size),
                "activation_moyenne": round(float(block.mean()), 4),
                "activation_max": round(float(block.max()), 4),
                "degre_sortant_moyen": round(float(out_deg[node_ids].mean()), 2),
                "degre_entrant_moyen": round(float(in_deg[node_ids].mean()), 2),
                "cables": int((out_deg[node_ids] + in_deg[node_ids] > 0).sum()),
                "top_stimuli": [
                    {
                        "texte": texts[int(i)],
                        "categorie": categories[int(i)],
                        "activation": round(float(strength[int(i)]), 4),
                    }
                    for i in order
                    if strength[int(i)] > 0
                ],
            }
        )
    families.sort(key=lambda row: -row["activation_moyenne"])
    return families


def divergences(
    counts: np.ndarray,
    nodes: np.ndarray,
    edges: set[tuple[int, int]],
    n_stimuli: int,
    threshold_share: float = 0.1,
    examples: int = 12,
) -> dict:
    """Compare le cable (aretes) et la fonction (co-activation).

    Deux divergences, qui sont deux questions differentes :

    * ``cablees_jamais_coactives`` : une arete existe mais les deux noeuds ne
      s'allument jamais ensemble — le cable est peut-etre mort, ou l'arete
      transporte un signal qui ne change pas l'etat ;
    * ``coactifs_sans_arete`` : deux noeuds s'allument ensemble sans synapse —
      la fonction est portee par un chemin indirect, ou par le pool commun.
    """
    index = {int(node): i for i, node in enumerate(nodes)}
    wired_pairs = [
        (index[int(s)], index[int(t)])
        for s, t in edges
        if int(s) in index and int(t) in index
    ]
    dead_cable = [(i, j) for i, j in wired_pairs if counts[i, j] == 0]

    cut = max(1, int(round(float(threshold_share) * n_stimuli)))
    upper = np.triu(counts, k=1)
    hot_i, hot_j = np.nonzero(upper >= cut)
    free_pairs = [
        (int(i), int(j))
        for i, j in zip(hot_i, hot_j)
        if (int(nodes[i]), int(nodes[j])) not in edges
        and (int(nodes[j]), int(nodes[i])) not in edges
    ]
    free_pairs.sort(key=lambda pair: -int(upper[pair[0], pair[1]]))
    dead_cable.sort(key=lambda pair: -int(counts.max()))
    return {
        "aretes_totales_dans_le_scan": len(wired_pairs),
        "cablees_jamais_coactives": {
            "nombre": len(dead_cable),
            "part": round(len(dead_cable) / max(1, len(wired_pairs)), 4),
            "exemples": [
                {
                    "source": int(nodes[i]),
                    "cible": int(nodes[j]),
                    "coactivations": int(counts[i, j]),
                }
                for i, j in dead_cable[:examples]
            ],
        },
        "coactifs_sans_arete": {
            "seuil_coactivations": cut,
            "nombre": len(free_pairs),
            "exemples": [
                {
                    "a": int(nodes[i]),
                    "b": int(nodes[j]),
                    "coactivations": int(upper[i, j]),
                }
                for i, j in free_pairs[:examples]
            ],
        },
    }


def markdown_atlas(
    families: list[dict],
    meta: dict,
    div: dict,
    stimuli: list[tuple[str, str]],
) -> str:
    """Le rapport humain : lisible sans ouvrir le JSON."""
    lines = [
        "# ATLAS NYRA — NUMENIA-SCAN v0",
        "",
        f"Cerveau : `{meta['genome']}` — {meta['n_nodes']} noeuds, "
        f"{meta['alive']} vivants, {meta['ne']} aretes, step {meta['step']}.",
        f"Stimulation : {len(stimuli)} phrases, {meta['n_tokens']} tokens, "
        f"{meta['dim']} dimensions par noeud.",
        "",
        "Activation d'un noeud = norme L2 maximale de sa membrane sur les tokens "
        "du stimulus. Familles = k-means (k=" + str(meta["k"]) + ") sur le profil "
        "d'activation de chaque noeud.",
        "",
        "## Familles fonctionnelles",
        "",
        "| # | noeuds | taille | activation moy. | degre sortant | cables |",
        "|---|---|---|---|---|---|",
    ]
    for fam in families:
        sample = ", ".join(str(n) for n in fam["noeuds"][:6])
        more = "…" if fam["taille"] > 6 else ""
        lines.append(
            f"| {fam['famille']} | {sample}{more} | {fam['taille']} | "
            f"{fam['activation_moyenne']:.3f} | {fam['degre_sortant_moyen']:.2f} | "
            f"{fam['cables']}/{fam['taille']} |"
        )
    lines += ["", "### Top-10 stimuli par famille", ""]
    for fam in families:
        lines.append(f"**Famille {fam['famille']}** ({fam['taille']} noeuds)")
        lines.append("")
        for row in fam["top_stimuli"]:
            lines.append(f"- `{row['activation']:.3f}` [{row['categorie']}] {row['texte']}")
        lines.append("")
    dead = div["cablees_jamais_coactives"]
    free = div["coactifs_sans_arete"]
    lines += [
        "## Divergences structure / fonction",
        "",
        f"- Aretes observees dans le scan : {div['aretes_totales_dans_le_scan']}",
        f"- **Cablees mais jamais co-actives** : {dead['nombre']} "
        f"({dead['part']:.1%} des aretes).",
        f"- **Co-actives sans arete** (>= {free['seuil_coactivations']} co-activations) : "
        f"{free['nombre']} paires.",
        "",
        "Exemples de cable muet : "
        + ", ".join(
            f"{row['source']}→{row['cible']} ({row['coactivations']})"
            for row in dead["exemples"][:6]
        ),
        "",
        "Exemples de paire co-active sans synapse : "
        + ", ".join(
            f"{row['a']}~{row['b']} ({row['coactivations']})"
            for row in free["exemples"][:6]
        ),
        "",
        "Lecture : une arete qui ne co-active jamais n'est pas forcement morte — "
        "elle peut transporter un signal qui ne fait pas changer d'etat. Une paire "
        "co-active sans arete n'est pas une erreur de cablage — elle peut passer "
        "par deux sauts. Ces deux listes sont les candidates a verifier une par une.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="NUMENIA-SCAN v0 : atlas du cerveau reel")
    parser.add_argument("--genome", default="data/genome.pt")
    parser.add_argument("--k", type=int, default=6, help="nombre de familles")
    parser.add_argument("--top", type=int, default=64, help="noeuds retenus par stimulus")
    parser.add_argument("--film", type=int, default=5, help="stimuli filmes")
    parser.add_argument("--max-tokens", type=int, default=32)
    args = parser.parse_args()

    ensure_lab()
    set_seed()
    brain, blob = load_brain(args.genome)
    tokenizer = load_tokenizer(TOKENIZER_PATH)
    if tokenizer.get_vocab_size() != int(brain.vocab):
        raise SystemExit(
            f"tokenizer {tokenizer.get_vocab_size()} != vocab du cerveau {brain.vocab}"
        )

    stimuli = read_stimuli()
    texts = [text for _, text in stimuli]
    print(f"scan : {len(texts)} stimuli sur {int(brain.n_nodes)} noeuds…")
    matrix = scan_brain(brain, texts, tokenizer, max_tokens=args.max_tokens)
    np.save(LAB / "matrix_stimxnodes.npy", matrix)
    print(f"matrice : {matrix.shape} -> data/lab/matrix_stimxnodes.npy")

    alive = brain.node_alive.cpu().numpy()
    nodes = np.nonzero(alive)[0]
    block = matrix[:, nodes]
    activated = block.max(axis=0) > 0
    nodes = nodes[activated]
    block = block[:, activated]
    print(f"noeuds vivants actives : {len(nodes)} / {int(alive.sum())}")

    labels, centres, inertia = kmeans(profiles(block), args.k, seed=7)
    out_deg, in_deg = degree_tables(brain)
    families = build_atlas(labels, nodes, block, stimuli, out_deg, in_deg)

    counts = coactivation(block, top=args.top)
    edges = {
        (int(s), int(t))
        for s, t in brain.edge_index[:, : int(brain.ne)].t().tolist()
    }
    div = divergences(counts, nodes, edges, len(texts))
    np.save(LAB / "coactivation_counts.npy", counts)

    picks = []
    used_categories: set[str] = set()
    for index, (category, text) in enumerate(stimuli):
        if category in used_categories:
            continue
        used_categories.add(category)
        picks.append((index, text))
        if len(picks) >= args.film:
            break
    film = []
    for index, text in picks:
        shot = film_brain(brain, text, tokenizer, max_tokens=args.max_tokens)
        film.append(
            {
                "index": int(index),
                "texte": shot["texte"],
                "tokens": shot["tokens"],
                "frames": shot["frames"][:, nodes],
                "activation": matrix[index, nodes],
            }
        )
    film_payload = {
        "noeuds": [int(n) for n in nodes],
        "stimuli": [
            {
                "texte": shot["texte"],
                "tokens": shot["tokens"],
                "frames": [[round(float(v), 4) for v in frame] for frame in shot["frames"]],
            }
            for shot in film
        ],
    }

    meta = {
        "genome": args.genome,
        "n_nodes": int(brain.n_nodes),
        "alive": int(alive.sum()),
        "ne": int(brain.ne),
        "step": int(brain.clock),
        "dim": int(brain.dim),
        "n_tokens": int(
            sum(len(tokenizer.encode(text).ids or [0]) for text in texts)
        ),
        "k": int(args.k),
        "inertie": round(float(inertia), 4),
        "top_par_stimulus": int(args.top),
    }
    write_json(
        LAB / "atlas_nyra.json",
        {
            "meta": meta,
            "noeuds": [int(n) for n in nodes],
            "familles": families,
            "divergences": div,
            "degres": {
                "sortant": [int(v) for v in out_deg[nodes]],
                "entrant": [int(v) for v in in_deg[nodes]],
            },
        },
    )
    write_json(LAB / "film_nyra.json", film_payload)
    (LAB / "atlas_nyra.md").write_text(
        markdown_atlas(families, meta, div, stimuli), encoding="utf-8"
    )

    print(f"atlas   : {len(families)} familles, inertie {meta['inertie']}")
    for fam in families:
        top = fam["top_stimuli"][0]["texte"] if fam["top_stimuli"] else "-"
        print(f"  famille {fam['famille']} : {fam['taille']:3d} noeuds, top « {top} »")
    dead = div["cablees_jamais_coactives"]
    free = div["coactifs_sans_arete"]
    print(
        f"divergences : {dead['nombre']} aretes jamais co-actives "
        f"({dead['part']:.1%}), {free['nombre']} paires co-actives sans arete"
    )
    print("ecrit : data/lab/atlas_nyra.json, atlas_nyra.md, film_nyra.json")


if __name__ == "__main__":
    main()
