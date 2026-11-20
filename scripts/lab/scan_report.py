"""C4 — le scan recurrent : MYCELI-SCAN sur le genome actuel, et son historique.

Le scan de S1 a ete fait sur ``data/genome.pt`` au step 56. Aucun entrainement
n'a tourne depuis (Alex a suspendu le 24/7) : reexecuter le scan doit donc
redonner **exactement** la meme carte. C'est la preuve de reproductibilite, et
c'est la seule facon de savoir qu'une difference ulterieure vient du cerveau et
pas du thermometre.

Le script n'ecrit **pas** ``atlas_nyra.json`` (la reference de S1 reste
intacte) : il produit ``scan_s2.json`` et empile une entree dans
``scan_history.json``. Chaque relance de Nyra pourra ajouter son point, ce qui
donne la premiere serie temporelle d'evolution d'un cerveau MYCELIA.

Lecture de l'engagement : noeuds actives par au moins un stimulus / noeuds
vivants. Il doit rester stable (~45 %) tant que rien n'a appris.
"""

from __future__ import annotations

import argparse
import datetime as _dt

import numpy as np

from lab_common import LAB, ensure_lab, kmeans, profiles, read_json, scan_brain, set_seed, write_json
from mycelia.data import TOKENIZER_PATH, load_tokenizer
from scan_mycelia import coactivation, degree_tables, divergences, load_brain, read_stimuli


def main() -> None:
    parser = argparse.ArgumentParser(description="MYCELI-SCAN recurrent + historique")
    parser.add_argument("--genome", default="data/genome.pt")
    parser.add_argument("--k", type=int, default=6)
    parser.add_argument("--top", type=int, default=64)
    parser.add_argument("--stimuli", type=int, default=500, help="memes 500 qu'en S1")
    parser.add_argument("--label", default="S2")
    args = parser.parse_args()

    ensure_lab()
    set_seed()
    brain, _blob = load_brain(args.genome)
    tokenizer = load_tokenizer(TOKENIZER_PATH)
    stimuli = read_stimuli()[: int(args.stimuli)]
    texts = [text for _, text in stimuli]

    matrix = scan_brain(brain, texts, tokenizer, max_tokens=32)
    alive = brain.node_alive.cpu().numpy()
    nodes = np.nonzero(alive)[0]
    block = matrix[:, nodes]
    active = block.max(axis=0) > 0
    nodes = nodes[active]
    block = block[:, active]

    labels, _centres, inertia = kmeans(profiles(block), args.k, seed=7)
    counts = coactivation(block, top=args.top)
    edges = {
        (int(s), int(t)) for s, t in brain.edge_index[:, : int(brain.ne)].t().tolist()
    }
    div = divergences(counts, nodes, edges, len(texts))

    engagement = round(float(nodes.size) / max(1, int(alive.sum())), 4)
    entry = {
        "label": args.label,
        "date": _dt.datetime.now().isoformat(timespec="seconds"),
        "genome": args.genome,
        "step": int(brain.clock),
        "noeuds": int(brain.n_nodes),
        "vivants": int(alive.sum()),
        "actifs": int(nodes.size),
        "engagement": engagement,
        "familles": int(len(set(int(v) for v in labels))),
        "inertie": round(float(inertia), 4),
        "aretes": int(brain.ne),
        "aretes_jamais_coactives": int(div["cablees_jamais_coactives"]["nombre"]),
        "paires_coactives_sans_arete": int(div["coactifs_sans_arete"]["nombre"]),
        "stimuli": len(texts),
        "k": int(args.k),
        "top_par_stimulus": int(args.top),
    }

    # ---- comparaison avec S1 ---------------------------------------------
    reference = read_json(LAB / "atlas_nyra.json")
    meta_ref = reference.get("meta", {})
    s1_active = len(reference.get("noeuds", []))
    s1_alive = int(meta_ref.get("alive", 0))
    s1_engagement = round(s1_active / max(1, s1_alive), 4)
    comparison = {
        "S1": {
            "vivants": s1_alive,
            "actifs": s1_active,
            "engagement": s1_engagement,
            "familles": len(reference.get("familles", [])),
            "inertie": meta_ref.get("inertie"),
            "aretes_jamais_coactives": reference.get("divergences", {})
            .get("cablees_jamais_coactives", {})
            .get("nombre"),
            "paires_coactives_sans_arete": reference.get("divergences", {})
            .get("coactifs_sans_arete", {})
            .get("nombre"),
        },
        "S2": {
            "vivants": entry["vivants"],
            "actifs": entry["actifs"],
            "engagement": engagement,
            "familles": entry["familles"],
            "inertie": entry["inertie"],
            "aretes_jamais_coactives": entry["aretes_jamais_coactives"],
            "paires_coactives_sans_arete": entry["paires_coactives_sans_arete"],
        },
    }

    ecart = None
    old_path = LAB / "matrix_stimxnodes.npy"
    if old_path.exists():
        old = np.load(old_path)
        if old.shape == matrix.shape:
            ecart = round(float(np.abs(old - matrix).max()), 8)
        else:
            ecart = f"formes differentes : {old.shape} vs {matrix.shape}"
    comparison["ecart_max_matrice"] = ecart
    comparison["reproductible"] = bool(
        entry["actifs"] == s1_active and entry["familles"] == len(reference.get("familles", []))
        and (ecart == 0.0 or ecart is None)
    )

    entry["reproductible"] = comparison["reproductible"]
    entry["ecart_max_matrice"] = ecart

    history_path = LAB / "scan_history.json"
    history = read_json(history_path) if history_path.exists() else {"scans": []}
    history.setdefault("scans", []).append(entry)
    write_json(history_path, history)
    write_json(
        LAB / "scan_s2.json",
        {
            "meta": {
                **{k: entry[k] for k in ("label", "date", "genome", "step", "noeuds", "vivants",
                                          "actifs", "engagement", "familles", "inertie", "aretes",
                                          "stimuli", "k", "top_par_stimulus")},
                "dim": int(brain.dim),
            },
            "noeuds": [int(n) for n in nodes],
            "comparaison_S1": comparison,
            "divergences": div,
        },
    )

    print(f"scan {entry['label']} : {entry['actifs']}/{entry['vivants']} noeuds actifs "
          f"({engagement:.1%}), {entry['familles']} familles, inertie {entry['inertie']}")
    print(f"S1                       : {s1_active}/{s1_alive} noeuds actifs "
          f"({s1_engagement:.1%}), {len(reference.get('familles', []))} familles, "
          f"inertie {meta_ref.get('inertie')}")
    print(f"ecart max des matrices   : {ecart}")
    print(f"reproductible            : {comparison['reproductible']}")
    print(f"historique               : {len(history['scans'])} scan(s) dans data/lab/scan_history.json")


if __name__ == "__main__":
    main()
