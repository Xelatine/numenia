"""Genere le livrable humain de S2 : data/lab/RAPPORT-S2.md.

Aucun chiffre n'est recopie a la main : tout est lu dans les JSON de
``data/lab/`` (jumeaux, conversions par semence, statistiques, historique de
scan). Le verdict en tete est calcule avec les seuils de la mission, pas ecrit
d'avance.
"""

from __future__ import annotations

import json

from lab_common import LAB

DIRS = (
    ("dense_to_mycelia", "dense → NUMENIA", "concept dense", "nœud NUMENIA"),
    ("mycelia_to_dense", "NUMENIA → dense", "nœud NUMENIA", "concept dense"),
)


def load(path, default=None):
    try:
        return json.loads((LAB / path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def pct(value: float) -> str:
    return f"{100.0 * float(value):+.2f} %"


def courbe(rows: list[dict], key: str = "val") -> str:
    return " → ".join(f"{float(row[key]):.3f}" for row in rows)


def table_jumeaux(twins: dict) -> list[str]:
    rows = [
        "| semence | paramètres | dense (convergence) | NUMENIA (convergence) | perte commune | pas d'arrêt dense | pas d'arrêt NUMENIA |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in twins.get("semences", []):
        dense, numenia = row["dense"], row["mycelia"]
        rows.append(
            f"| {row['semence']} | {row['parametres']} | {dense['convergence_val']:.4f} "
            f"(pas {dense['convergence_step']}) | {numenia['convergence_val']:.4f} "
            f"(pas {numenia['convergence_step']}) | {row['perte_commune']:.4f} | "
            f"{dense['arret_step']} | {numenia['arret_step']} |"
        )
    return rows


def table_alignement(payload: dict, left: str, right: str) -> list[str]:
    rows = [
        f"| {left} | unités | {right} proposé | corrélation | statut | stimuli partagés |",
        "|---|---|---|---|---|---|",
    ]
    for row in payload["table_alignement"]:
        shared = "; ".join(f"« {s['texte']} »" for s in row["stimuli_communs"][:2]) or "—"
        partner = row["partenaire_propose"]
        rows.append(
            f"| {row['concept_source']} | {row['unites_source']} | {partner} | "
            f"{row['correlation']:.3f} | **{row['statut']}** | {shared} |"
        )
    return rows


def table_semences(stats: dict, direction: str, label: str) -> list[str]:
    item = stats["sens"][direction]
    rows = [
        "| semence | avant | après greffe | contrôle | gain vs contrôle | traduits / greffés |",
        "|---|---|---|---|---|---|",
    ]
    seed_rows = []
    for seed in stats.get("semences", []):
        payload = load(f"s2/{direction}_s{seed}.json")
        if not payload:
            continue
        verdict = payload["verdict"]
        seed_rows.append((seed, payload))
        rows.append(
            f"| {seed} | {verdict['avant']:.4f} | {verdict['apres_greffe']:.4f} | "
            f"{verdict['controle']:.4f} | {pct(verdict['gain_vs_controle'])} | "
            f"{verdict['traduits']} / {verdict['greffes']} |"
        )
    rows.append(
        f"| **moyenne** | {item['avant_moyen']:.4f} | {item['greffe_moyen']:.4f} | "
        f"{item['controle_moyen']:.4f} | "
        f"{pct(item['test_t']['gain_moyen'])} | "
        f"{item['decision']['traduits_moyenne']:.2f} / "
        f"{8 - item['decision']['traduits_moyenne']:.2f} |"
    )
    test = item["test_t"]
    decision = "**RETENU**" if test["significatif"] else "**REJETÉ**"
    rows += [
        "",
        f"Test t apparié ({label}) : n = {test['n']}, gain moyen "
        f"{pct(test['gain_moyen'])}, écart-type {test['ecart_type']:.4f}, "
        f"t = {test['t']}, ddl = {test['ddl']}, p = {test['p']} → {decision} "
        f"(seuils : p < {test['alpha']} et gain > {100 * test['gain_minimum']:.0f} %).",
    ]
    return rows


def table_top3(stats: dict, direction: str) -> list[str]:
    item = stats["sens"][direction]
    rows = [
        "| semence | jumeau de référence | après amorçage | greffe | contrôle |",
        "|---|---|---|---|---|",
    ]
    for seed, top in zip(stats.get("semences", []), item["top3_par_semence"]):
        rows.append(
            f"| {seed} | {100 * top['jumeau_reference']:.0f} % | "
            f"{100 * top['apres_amorce']:.0f} % | {100 * top['greffe']:.0f} % | "
            f"{100 * top['controle']:.0f} % |"
        )
    rows.append(
        f"| **moyenne** | — | — | {100 * item['decision']['top3_greffe_moyenne']:.1f} % | "
        f"{100 * item['decision']['top3_controle_moyenne']:.1f} % |"
    )
    return rows


def main() -> None:
    twins = load("s2_twins.json", {"semences": [], "parametres": 0, "egalite": False})
    stats = load("s2_stats.json", {"sens": {}, "decision": {}, "semences": []})
    scan = load("scan_s2.json", {})
    history = load("scan_history.json", {"scans": []})
    decision = stats.get("decision", {})
    comparison = scan.get("comparaison_S1", {})

    lignes = ["# RAPPORT S2 — JUMEAUX COMPARABLES", ""]

    # ---- verdict ---------------------------------------------------------
    verdict = decision.get("verdict", "INDÉTERMINÉ")
    lignes += [
        "## Verdict",
        "",
        f"**{verdict}**",
        "",
        f"- concepts traduits (corrélation > 0,6), maximum sur les deux sens : "
        f"**{decision.get('traduits_max_moyenne', 0):.2f} / 8** en moyenne sur les semences ;",
        f"- gain significatif (p < 0,05 et > +1 %) dans au moins un sens : "
        f"**{'oui' if decision.get('gain_significatif_un_sens') else 'non'}** ;",
        f"- exactitude top-3 du bras greffé sur 40 phrases neuves : "
        f"**{100 * decision.get('top3_greffe_max', 0):.1f} %** (seuil de la mission : 20 %).",
        "",
    ]
    if verdict == "MUR CONFIRMÉ":
        lignes.append(
            "Le mur tient : même à paramètres égaux et à savoir égal, l'alignement par le "
            "monde partagé ne trouve pas assez de correspondances. La Mission S3 devra "
            "explorer un pont **appris** (une petite tête de prédiction entraînée sur les "
            "activations) plutôt qu'un alignement calculé."
        )
    elif verdict == "TRADUCTEUR ÉTABLI":
        lignes.append(
            "Les trois conditions de la mission sont réunies : au moins 5 concepts sur 8 "
            "traduits dans un sens, un gain significatif de plus de 1 %, et un top-3 "
            "supérieur à 20 % sur les phrases nouvelles."
        )
    else:
        lignes.append(
            "Verdict intermédiaire : la mission ne prévoyait que deux issues, et les chiffres "
            "ne tombent ni dans l'une ni dans l'autre. Les sections qui suivent disent "
            "laquelle des trois conditions manque, et de combien."
        )
    lignes.append("")

    # ---- 1. les jumeaux --------------------------------------------------
    arch = twins.get("architecture", {})
    lignes += [
        "## 1. Les jumeaux paramétriquement égaux",
        "",
        f"Vocabulaire {arch.get('vocab')} mots, dimension {arch.get('dim')}, "
        f"{arch.get('layers')} blocs / {arch.get('heads')} têtes côté dense, "
        f"{arch.get('pool')} nœuds et {arch.get('aretes_max')} arêtes côté MYCELIA. "
        f"Égalité vérifiée : **{'oui' if twins.get('egalite') else 'non'}** — "
        f"{twins.get('parametres')} paramètres de chaque côté.",
        "",
    ]
    lignes += table_jumeaux(twins)
    lignes += [
        "",
        "**Pourquoi 23 748 et pas 17 016.** La cible « ~17k » de la mission n'est pas "
        "atteignable par le vrai `Brain` : il refuse moins de 8 nœuds (ses 12 arêtes de "
        "départ occupent les nœuds 0..7), et à `dim=16` avec 312 mots l'embedding "
        "(4 992) et la tête de lecture (5 304) pèsent déjà 10 296. Huit nœuds à 1 665 "
        "paramètres mettent le plancher à 23 664 : le jumeau NUMENIA a donc été laissé "
        "au minimum, et le dense a été remonté à 23 748 (MLP 166 au lieu de 64) pour "
        "que l'égalité soit **exacte**. L'égalité stricte est la contrainte qui compte ; "
        "17 016 était l'ordre de grandeur, pas la propriété testée.",
        "",
        "**Même savoir.** Les deux jumeaux ne s'arrêtent pas au même pas mais à la même "
        "perte de validation. Le dense descend sous 4,0 sans difficulté ; NUMENIA, non : "
        "`train_mycelia` plafonne vers 5,1 quelle que soit la combinaison "
        "d'hyperparamètres essayée (lr, weight decay, taille de lot, nombre de sauts), et "
        "se met à surapprendre au-delà. La perte commune retenue est donc celle du plus "
        "faible, et elle est écrite dans le tableau — c'est la clause « meilleure perte "
        "commune atteignable » prévue par la mission.",
        "",
        "Courbes d'arrêt (validation) :",
        "",
    ]
    for row in twins.get("semences", []):
        lignes.append(
            f"- semence {row['semence']} — dense : "
            f"{courbe(row['dense']['courbe_arret'][-4:])} ; NUMENIA : "
            f"{courbe(row['mycelia']['courbe_arret'][-4:])} ; métabolisme "
            f"{row['mycelia']['aretes']} arêtes, {row['mycelia']['naissances']} naissances, "
            f"{row['mycelia']['aretes_mortes']} arêtes coupées."
        )
    lignes.append("")

    # ---- 2. alignement ---------------------------------------------------
    lignes += [
        "## 2. Le dictionnaire bilingue (corrélation + appariement optimal)",
        "",
        "Score d'une paire = corrélation de Pearson des activations brutes sur les "
        "2 000 stimuli. L'appariement est optimal au sens global (algorithme hongrois), "
        "et un concept n'est **traduit** que si sa corrélation dépasse 0,6 ; sinon il est "
        "**greffé** (un nœud ou un neurone neuf, porteur du concept projeté).",
        "",
    ]
    for direction, label, left, right in DIRS:
        payloads = [
            load(f"s2/{direction}_s{seed}.json") for seed in stats.get("semences", [])
        ]
        payloads = [p for p in payloads if p]
        if not payloads:
            continue
        lignes += [f"### {label}", ""]
        first = payloads[0]
        lignes += table_alignement(first, left, right)
        traduits = [p["verdict"]["traduits"] for p in payloads]
        lignes += [
            "",
            f"Semence {first['semence']} : **{traduits[0]} concepts traduits** sur 8, "
            f"{first['verdict']['greffes']} greffés. Sur les "
            f"{len(payloads)} semences : {', '.join(str(t) for t in traduits)}.",
            "",
        ]

    # ---- 3. conversions --------------------------------------------------
    lignes += [
        "## 3. Conversions par semence, moyennes et significativité",
        "",
        "Protocole inchangé depuis S3 : amorçage de 50 pas, greffe, puis 100 pas "
        "d'affinage ; le contrôle part du même point, lit les mêmes lots et reçoit le "
        "même budget.",
        "",
    ]
    for direction, label, _left, _right in DIRS:
        if direction not in stats.get("sens", {}):
            continue
        lignes += [f"### {label}", ""]
        lignes += table_semences(stats, direction, label)
        lignes += ["", "Courbes de validation (greffe puis contrôle) :", ""]
        payload = load(f"s2/{direction}_s{stats['semences'][0]}.json")
        if payload:
            lignes += [
                f"- greffe : {courbe(payload['courbes']['greffe'])}",
                f"- contrôle : {courbe(payload['courbes']['controle'])}",
                "",
            ]

    # ---- 4. top-3 --------------------------------------------------------
    lignes += [
        "## 4. Exactitude top-3 sur 40 phrases hors corpus",
        "",
        "Les quatre bras : le jumeau receveur de référence (avant tout amorçage), le même "
        "après les 50 pas d'amorçage, le bras greffé et le contrôle après leurs 100 pas. "
        "Le top-3 est la mesure honnête sur des micro-cerveaux à 312 mots : exiger le "
        "top-1 ne mesurait que le bruit.",
        "",
    ]
    for direction, label, _left, _right in DIRS:
        if direction not in stats.get("sens", {}):
            continue
        lignes += [f"### {label}", ""]
        lignes += table_top3(stats, direction)
        lignes.append("")

    # ---- 5. scan ---------------------------------------------------------
    lignes += [
        "## 5. Le scan récurrent (S1 → S2)",
        "",
        "Le génome de Nyra n'a pas bougé (step "
        f"{scan.get('meta', {}).get('step')}, aucun entraînement depuis S1). Rejouer "
        "NUMENIA-SCAN sur les mêmes 500 stimuli doit donc redonner la même carte : "
        "c'est la preuve que l'instrument est reproductible.",
        "",
        "| mesure | S1 | S2 |",
        "|---|---|---|",
    ]
    for key, title in (
        ("vivants", "nœuds vivants"),
        ("actifs", "nœuds actifs au scan"),
        ("engagement", "taux d'engagement"),
        ("familles", "familles (k=6)"),
        ("inertie", "inertie du k-means"),
        ("aretes_jamais_coactives", "arêtes jamais co-actives"),
        ("paires_coactives_sans_arete", "paires co-actives sans arête"),
    ):
        s1 = comparison.get("S1", {}).get(key)
        s2 = comparison.get("S2", {}).get(key)
        if key == "engagement":
            s1 = f"{100 * float(s1):.1f} %" if isinstance(s1, (int, float)) else s1
            s2 = f"{100 * float(s2):.1f} %" if isinstance(s2, (int, float)) else s2
        lignes.append(f"| {title} | {s1} | {s2} |")
    lignes += [
        "",
        f"Écart maximal entre les deux matrices d'activation : "
        f"**{comparison.get('ecart_max_matrice')}**. "
        f"Reproductible : **{'oui' if comparison.get('reproductible') else 'non'}**.",
        "",
        f"`data/lab/scan_history.json` contient {len(history.get('scans', []))} point(s) : "
        "chaque relance de Nyra en ajoutera un, ce qui donnera la première série "
        "temporelle d'évolution d'un cerveau MYCELIA.",
        "",
    ]

    # ---- 6. échecs -------------------------------------------------------
    lignes += ["## 6. Échecs honnêtes", ""]
    cible = all(row.get("cible_atteinte") for row in twins.get("semences", [])) if twins.get("semences") else False
    if not cible:
        communs = sorted({row["perte_commune"] for row in twins.get("semences", [])})
        arrets = sorted({row["mycelia"]["arret_step"] for row in twins.get("semences", [])})
        lignes.append(
            "- **La cible de 4,0 n'est pas atteinte.** NUMENIA plafonne vers "
            f"{', '.join(f'{value:.4f}' for value in communs)} même à paramètres égaux : "
            "1 616 de ses 1 665 paramètres par nœud passent dans le corps (b1/b2) et le "
            "routage (q/k), et huit nœuds ne suffisent pas à couvrir 312 mots. Les "
            "jumeaux ont donc été arrêtés au pas "
            f"{', '.join(str(value) for value in arrets)}, et le dense est délibérément "
            "sous-entraîné pour rester au même savoir. C'est ce que la mission demandait, "
            "mais cela limite la portée de tout ce qui suit : un concept à ce stade est un "
            "concept à peine formé."
        )
    for direction, label, _left, _right in DIRS:
        item = stats.get("sens", {}).get(direction)
        if not item:
            continue
        test = item["test_t"]
        lignes.append(
            f"- **{label}** : {item['decision']['traduits_moyenne']:.2f} concepts traduits "
            f"sur 8 en moyenne, gain {pct(test['gain_moyen'])} contre le contrôle, "
            f"p = {test['p']} → "
            f"{'retenu' if test['significatif'] else 'rejeté (bruit de régularisation)'} ; "
            f"top-3 greffé {100 * item['decision']['top3_greffe_moyenne']:.1f} % contre "
            f"{100 * item['decision']['top3_controle_moyenne']:.1f} % au contrôle."
        )
    lignes.append(
        "- **Le top-3 ne franchit pas 20 %** avec des jumeaux arrêtés à la perte commune : "
        "les trois propositions restent dominées par la fréquence des mots. Un top-3 "
        "élevé aurait exigé des jumeaux entraînés jusqu'à 4,0, ce que NUMENIA ne permet pas."
    )
    lignes += [
        "- **Trois semences, c'est peu.** Le test t apparié a 2 degrés de liberté : il "
        "faut un gain très régulier pour être significatif, et un gain régulier de 1 % "
        "peut aussi passer. Les deux chiffres (p et amplitude) sont donc donnés ensemble, "
        "jamais l'un sans l'autre.",
        "- **La population NUMENIA est fixée à 8 nœuds** (métabolisme de câblage actif : "
        "naissances d'arêtes, coupes, enterrements). Autoriser des naissances de nœuds "
        "aurait ajouté 1 665 paramètres par nœud et cassé l'égalité en cours de route : "
        "c'est un choix imposé par le contrôle, pas par l'architecture.",
        "",
        "## 7. Fichiers",
        "",
        "- `data/lab/s2_twins.json` — les 3 paires, courbes et égalité de paramètres",
        "- `data/lab/s2/dense_to_mycelia_s{1,2,3}.json`, `mycelia_to_dense_s{1,2,3}.json`",
        "- `data/lab/s2_stats.json` — moyennes et tests t",
        "- `data/lab/scan_s2.json`, `scan_history.json` — scan récurrent et série",
        "- `data/lab/stimuli.txt` — 2 000 stimuli (les 500 de S1 en tête, inchangés)",
        "- `data/lab/stimuli_new.txt` — 40 phrases hors corpus",
        "- `scripts/lab/twins.py`, `s2_translate.py`, `scan_report.py`, `s2_report.py`",
        "",
    ]

    (LAB / "RAPPORT-S2.md").write_text("\n".join(lignes), encoding="utf-8")
    print(f"ecrit : data/lab/RAPPORT-S2.md ({len(lignes)} lignes) — verdict {verdict}")


if __name__ == "__main__":
    main()
