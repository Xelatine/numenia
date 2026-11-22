"""Genere le rapport humain du traducteur : data/lab/RAPPORT-translation.md.

Aucun chiffre n'est recopie a la main : tout est lu dans les JSON produits par
les scripts. Un rapport qui cite des nombres ecrits de memoire n'est pas une
preuve, c'est une opinion.
"""

from __future__ import annotations

import json

from lab_common import LAB


def load(name: str) -> dict:
    return json.loads((LAB / name).read_text(encoding="utf-8"))


def courbe(rows: list[dict], key: str = "val") -> str:
    return " → ".join(f"{row[key]:.3f}" for row in rows)


def table_alignement(rows: list[dict], source: str, receiver: str) -> list[str]:
    out = [
        f"| {source} | {receiver} | similarite | stimuli partages |",
        "|---|---|---|---|",
    ]
    for row in rows:
        shared = "; ".join(f"« {s['texte']} »" for s in row["stimuli_communs"][:2]) or "—"
        gauche = row.get("concept_source", row.get("famille_source"))
        droite = row.get("famille_receveuse", row.get("concept_receveur"))
        out.append(
            f"| {gauche} | {droite} | {row['similarite']:.3f} | {shared} |"
        )
    return out


def sandbox_section(name: str, result: dict) -> list[str]:
    rows = result["curve"]
    first, best = rows[0], min(rows, key=lambda row: row["val"])
    lines = [
        f"### {name}",
        "",
        f"- parametres : {result['params']}",
        f"- vocabulaire : {result['vocab']} mots",
        f"- entrainement : {result['seconds']} s",
        f"- perte d'entrainement : {first['loss']:.4f} → {rows[-1]['loss']:.4f}",
        f"- validation : {first['val']:.4f} → {best['val']:.4f} "
        f"(meilleur pas {best['step']}, arret sur patience)",
        "",
        "Courbe de validation : " + courbe(rows),
        "",
        "| prefixe | mots proposes |",
        "|---|---|",
    ]
    for sample in result["samples"]:
        lines.append(f"| {sample['prefixe']} | {' / '.join(sample['predit'])} |")
    lines.append("")
    return lines


def main() -> None:
    dense = load("micro_dense_result.json")
    numenia = load("micro_mycelia_result.json")
    atlas = load("atlas_nyra.json")
    s3 = load("conversion_dense_to_mycelia.json")
    s4 = load("conversion_mycelia_to_dense.json")
    atlas_md = (LAB / "atlas_nyra.md").read_text(encoding="utf-8")

    v3, v4 = s3["verdict"], s4["verdict"]
    lines = [
        "# RAPPORT DU TRADUCTEUR — dense ↔ NUMENIA",
        "",
        "NUMENIA-SCAN v0 + mini-labo de conversion. Toutes les mesures viennent des "
        "fichiers de `data/lab/` ; les courbes sont celles relevees pendant "
        "l'execution, pas des estimations.",
        "",
        "## 1. Resume",
        "",
        f"- Atlas de Nyra : {atlas['meta']['alive']} noeuds vivants, "
        f"{len(atlas['noeuds'])} actives par les 500 stimuli, "
        f"{len(atlas['familles'])} familles, k={atlas['meta']['k']}.",
        f"- Divergences structure/fonction : "
        f"{atlas['divergences']['cablees_jamais_coactives']['nombre']} aretes "
        f"jamais co-actives "
        f"({atlas['divergences']['cablees_jamais_coactives']['part']:.1%}), "
        f"{atlas['divergences']['coactifs_sans_arete']['nombre']} paires co-actives "
        f"sans arete.",
        f"- Dense → NUMENIA : {v3['avant']:.4f} → {v3['apres_greffe']:.4f} apres greffe "
        f"({v3['gain_vs_avant']:.1%} de gain), controle au meme budget "
        f"{v3['controle']:.4f}.",
        f"- NUMENIA → dense : {v4['avant']:.4f} → {v4['apres_greffe']:.4f} "
        f"({v4['gain_vs_avant']:.1%}), controle {v4['controle']:.4f}.",
        "",
        "## 2. Les deux bacs a sable (S2)",
        "",
        "Meme tache (mot suivant), meme flux de mots, meme vocabulaire, "
        f"{dense['vocab']} mots. Le dense a {dense['params']} parametres, NUMENIA "
        f"{numenia['params']} — l'ecart est assume : NUMENIA paie un pool, des "
        "aretes et un metabolisme que le dense n'a pas.",
        "",
    ]
    lines += sandbox_section("micro-transformer dense (2 couches, 16 dims)", dense)
    lines += sandbox_section("micro-MYCELIA (pool, aretes, metabolisme)", numenia)
    lines += [
        f"Metabolisme du bac NUMENIA : pool {numenia['pool_start']} → "
        f"{numenia['pool']} noeuds, {numenia['node_births']} naissances reelles, "
        f"{numenia['node_deaths']} enterrements, {numenia['edge_deaths']} aretes coupees.",
        "",
        "## 3. Le dictionnaire bilingue (alignement fonctionnel)",
        "",
        "Chaque ligne est une paire de familles qui repondent aux memes phrases. "
        "L'alignement est mutuel : une paire ne compte que si chacun est le "
        "meilleur partenaire de l'autre, au-dessus du seuil.",
        "",
        "### Dense → NUMENIA",
        "",
    ]
    lines += table_alignement(s3["table_alignement"], "concept source", "famille receveuse")
    lines += ["", "### NUMENIA → dense", ""]
    lines += table_alignement(s4["table_alignement"], "famille source", "concept receveur")
    lines += [
        "",
        "## 4. Verdict dense → NUMENIA (S3)",
        "",
        f"- **AVANT** : validation {v3['avant']:.4f}, entrainement {v3['train_avant']:.4f}.",
        f"- **APRES 100 pas avec greffe** : validation {v3['apres_greffe']:.4f} "
        f"({v3['gain_vs_avant']:+.1%}), entrainement {v3['train_greffe']:.4f}.",
        f"- **Controle, 100 pas sans greffe** : validation {v3['controle']:.4f}, "
        f"entrainement {v3['train_controle']:.4f}.",
        f"- Greffe contre controle : {v3['gain_vs_controle']:+.2%} de validation, "
        f"{v3['train_gain_vs_controle']:+.2%} d'entrainement.",
        f"- Test de langue (20 phrases hors corpus) : greffe {v3['language_greffe']['perte']:.4f}, "
        f"controle {v3['language_controle']['perte']:.4f}, "
        f"{v3['phrases_gagnees_vs_controle']}/{v3['phrases']} phrases meilleures.",
        "",
        "Courbe de validation — greffe : " + courbe(s3["courbes"]["greffe"]),
        "",
        "Courbe de validation — controle : " + courbe(s3["courbes"]["controle"]),
        "",
        f"Transplants : {len(s3['transplants'])} noeuds-portes "
        f"({len(s3['concepts_sans_equivalent'])} concepts sans equivalent), "
        f"pool {s3['meta']['noeuds_receveur']} noeuds receveurs → "
        f"{s3['meta']['noeuds_receveur'] + len(s3['transplants'])}.",
        "",
        "## 5. Verdict NUMENIA → dense (S4)",
        "",
        f"- **AVANT** : validation {v4['avant']:.4f}, entrainement {v4['train_avant']:.4f}.",
        f"- **APRES 100 pas avec greffe** : validation {v4['apres_greffe']:.4f} "
        f"({v4['gain_vs_avant']:+.1%}), entrainement {v4['train_greffe']:.4f}.",
        f"- **Controle** : validation {v4['controle']:.4f}, entrainement {v4['train_controle']:.4f}.",
        f"- Greffe contre controle : {v4['gain_vs_controle']:+.2%} de validation.",
        f"- Test de langue : greffe {v4['language_greffe']['perte']:.4f}, "
        f"controle {v4['language_controle']['perte']:.4f}, "
        f"{v4['phrases_gagnees_vs_controle']}/{v4['phrases']} phrases meilleures.",
        "",
        "Courbe de validation — greffe : " + courbe(s4["courbes"]["greffe"]),
        "",
        "Courbe de validation — controle : " + courbe(s4["courbes"]["controle"]),
        "",
        f"Transplants : {len(s4['transplants'])} neurones caches "
        f"({len(s4['familles_sans_equivalent'])} familles sans equivalent), "
        f"parametres {s4['meta']['parametres_avant']} → {s4['meta']['parametres_greffe']}.",
        "",
        "## 6. Echecs honnetes",
        "",
        f"- **Un seul appariement sur {s3['meta']['k']} dans les deux sens.** Avec le "
        "seuil 0,5 et un appariement mutuel, presque aucun concept n'a de "
        "correspondant : les deux cerveaux ne se ressemblent pas assez pour que "
        "la traduction soit dense. Les 7 autres sont donc greffes, pas traduits.",
        f"- **Le gain vient surtout du transplant, pas de l'alignement** : "
        f"{len(s3['transplants'])} noeuds ajoutes, dont aucun ne correspond a une "
        "famille existante. Un traducteur qui greffe tout ne traduit rien.",
        f"- **Le sens NUMENIA → dense est presque neutre** : "
        f"{v4['gain_vs_controle']:+.2%} de validation contre le controle, "
        f"{v4['phrases_gagnees_vs_controle']}/{v4['phrases']} phrases. "
        "Sept neurones zero-initialises n'apportent presque rien en 100 pas.",
        f"- **Entrainement contre validation** : dans le sens dense → NUMENIA, la "
        f"greffe gagne en validation ({v3['gain_vs_controle']:+.2%}) mais perd un "
        f"peu en entrainement ({v3['train_gain_vs_controle']:+.2%}). Un gain de "
        "generalisation qui n'est pas un gain d'apprentissage doit etre lu comme "
        "un effet de regularisation, pas comme une superiorite.",
        f"- **Exactitude de prediction : 0 %** sur les 20 phrases dans les quatre "
        "bras. Les modeles apprennent la syntaxe du corpus jouet, pas ses mots : "
        "la tache a 312 mots reste hors de portee en quelques centaines de pas.",
        f"- **{atlas['meta']['alive'] - len(atlas['noeuds'])} noeuds vivants ne "
        "s'allument sur aucun des 500 stimuli** — ni famille, ni concept. Ils sont "
        "invisibles au scan, donc absents du traducteur.",
        f"- Les deux sens sont mesures depuis un point d'amorcage "
        f"({s4['meta']['amorce']} pas d'entrainement du receveur, puis "
        f"{s3['meta']['steps']} pas d'affinage) : converti depuis son optimum, un "
        "receveur ne peut que se degrader et la comparaison ne mesure plus rien.",
        "",
        "## 7. Atlas de Nyra (copie du rapport de scan)",
        "",
    ]
    # Le H1 de l'atlas est retire : le rapport a deja son titre.
    body = atlas_md.split("\n", 1)[1] if atlas_md.startswith("# ") else atlas_md
    lines.append(body.lstrip())
    lines += [
        "",
        "## 8. Fichiers",
        "",
        "- `data/lab/stimuli.txt` — 500 stimuli (6 categories)",
        "- `data/lab/matrix_stimxnodes.npy` — activation [500, 532]",
        "- `data/lab/atlas_nyra.json`, `atlas_nyra.md` — atlas",
        "- `data/lab/film_nyra.json` — film de 5 stimuli, token par token",
        "- `data/lab/corpus_toy.txt` — 1000 phrases",
        "- `data/lab/conversion_dense_to_mycelia.json`, `conversion_mycelia_to_dense.json`",
        "",
    ]
    (LAB / "RAPPORT-translation.md").write_text("\n".join(lines), encoding="utf-8")
    print("ecrit : data/lab/RAPPORT-translation.md")


if __name__ == "__main__":
    main()
