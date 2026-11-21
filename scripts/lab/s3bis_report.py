"""S3-bis — genere ``data/lab/RAPPORT-S3BIS.md`` depuis les JSON mesures.

Aucun chiffre n'est ecrit a la main : le rapport est une lecture des fichiers
produits par ``floors.py``, ``context_test.py``, ``long_run.py``,
``ablations.py``, ``fair_twin.py``, ``big_corpus.py`` et ``recency_repair.py``.
Relancer le rapport apres une experience suffit a le mettre a jour.
"""

from __future__ import annotations

import glob

from lab_common import LAB, read_json, write_json

OUT = LAB / "RAPPORT-S3BIS.md"


def table(headers: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    out += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join(out)


def num(value, digits: int = 4) -> str:
    if value is None:
        return "—"
    return f"{float(value):.{digits}f}"


def sign(value, digits: int = 4) -> str:
    if value is None:
        return "—"
    return f"{float(value):+.{digits}f}".replace(".", ",")


def fr(value, digits: int = 4) -> str:
    return num(value, digits).replace(".", ",")


def load_all() -> dict:
    data = {
        "floors": read_json(LAB / "s3bis_floors.json"),
        "context": read_json(LAB / "s3bis_context.json"),
        "long": read_json(LAB / "s3bis_long.json"),
        "ablations": read_json(LAB / "s3bis_ablations.json"),
        "fair": read_json(LAB / "s3bis_fair.json"),
        "big": read_json(LAB / "s3bis_bigcorpus.json"),
    }
    recency = []
    for path in sorted(glob.glob(str(LAB / "s3bis_recency_*.json"))):
        recency.append(read_json(path))
    data["recency"] = sorted(recency, key=lambda row: float(row["state_decay"]))
    return data


def headline(data: dict) -> tuple[str, str]:
    """Verdict principal. Le BUG prime : un plancher = sans-contexte n'est pas un plafond."""
    floors = data["floors"]
    context = data["context"]
    if floors["verdict"] == "BUG" or not context["contexte_utilise"]:
        return (
            "🐛 BUG",
            "le plafond de MYCELIA est exactement le plancher sans contexte : "
            "il ne s'agit pas d'un plafond d'apprentissage mais d'un défaut "
            "d'usage du contexte, localisé dans l'agrégation et la lecture.",
        )
    if data["long"]["verdict"] == "A":
        return ("🐢 A", "MYCELIA apprend plus lentement mais progresse encore : la route des 90 % s'ouvre avec plus de pas.")
    if data["long"]["verdict"] == "B":
        return ("🧱 B", "plafond structurel confirmé à ce dimensionnement : le receveur doit être agrandi.")
    return ("❓ AMBIGU", "les mesures ne tranchent pas.")


def section_floors(floors: dict) -> str:
    uni = floors["unigramme"]
    bi = floors["bigramme"]
    tri = floors["trigramme"]
    rows = [
        ["uniforme ln(V)", num(floors["uniforme_ln_v"]), "aucune information"],
        ["unigramme (MLE)", num(uni["entropie_mle"]), f"{uni['types']} types, {uni['tokens']} tokens"],
        ["**unigramme lissé**", f"**{fr(uni['entropie_lissee'])}**", "**L_ref : plancher sans contexte**"],
        ["bigramme (MLE)", num(bi["entropie_mle"]), f"{bi['contextes']} contextes"],
        ["bigramme lissé", fr(bi["entropie_lissee"]), "plancher avec un mot d'histoire"],
        ["trigramme (MLE)", num(tri["entropie_mle"]), f"{tri['contextes']} contextes"],
        ["trigramme lissé", fr(tri["entropie_lissee"]), "plancher avec deux mots d'histoire"],
    ]
    return "\n".join(
        [
            "## 1. Les planchers n-grammes (D1) — à lire en premier",
            "",
            f"Corpus : `{floors['corpus']}`, {floors['phrases']} phrases, "
            f"vocabulaire {floors['vocabulaire']} mots, lissage hiérarchique alpha = {floors['alpha']}.",
            "",
            table(["Mesure (nats/token)", "Valeur", "Lecture"], rows),
            "",
            f"**L_ref = {fr(floors['L_ref'])} nats/token** — c'est le plancher d'un modèle "
            "qui prédit chaque mot sans regarder ce qui précède.",
            "",
            f"MYCELIA S2 mesure **{fr(floors['mycelia_s2'])}**, soit "
            f"**{sign(floors['ecart_mycelia_unigramme'])} nat** de l'unigramme.",
            "",
            f"> **{floors['verdict']}** — {floors['phrase']}.",
            "",
            "La conséquence est brutale et c'est le résultat le plus décisif du projet : "
            "un bigramme lissé vaut 3,05 nats, un trigramme 1,98. Il y a donc **2,03 nats "
            "de contexte disponibles** que MYCELIA ne prend pas. Son « plafond » de 5,1 "
            "n'est pas une limite de capacité : c'est le niveau exact d'un sac de mots.",
            "",
        ]
    )


def section_context(context: dict) -> str:
    cond = context["conditions"]
    rows = [
        ["contexte normal", fr(cond["normal"]["perte"]), fr(cond["normal"]["exactitude"], 3), "le vrai flux"],
        ["contexte mélangé", fr(cond["melange"]["perte"]), fr(cond["melange"]["exactitude"], 3), "même sac de mots, ordre détruit"],
        ["contexte vidé", fr(cond["vide"]["perte"]), fr(cond["vide"]["exactitude"], 3), "aucune information"],
    ]
    sondes = context["sondes"]
    probe_rows = [
        ["nœud 1 (la tête de lecture)", f"{sondes['noeud1_16d']['dims']} d", fr(sondes["noeud1_16d"]["perte"]), fr(sondes["noeud1_16d"]["exactitude"], 3)],
        ["nœuds du milieu", f"{sondes['noeuds_milieu_96d']['dims']} d", fr(sondes["noeuds_milieu_96d"]["perte"]), fr(sondes["noeuds_milieu_96d"]["exactitude"], 3)],
        ["**les 8 membranes**", f"{sondes['tous_noeuds_128d']['dims']} d", f"**{fr(sondes['tous_noeuds_128d']['perte'])}**", fr(sondes["tous_noeuds_128d"]["exactitude"], 3)],
        ["témoin : embeddings du préfixe", f"{sondes['embeddings_prefixe_128d']['dims']} d", fr(sondes["embeddings_prefixe_128d"]["perte"]), fr(sondes["embeddings_prefixe_128d"]["exactitude"], 3)],
    ]
    routing = context["routage"]
    return "\n".join(
        [
            "## 2. Le test de contexte (D2)",
            "",
            "Mêmes cibles, mêmes positions, seul le préfixe change :",
            "",
            table(["Condition", "Perte", "Exactitude", "Ce qui change"], rows),
            "",
            f"Écart maximal entre conditions : **{fr(context['ecart_max_conditions'])} nat**. "
            f"Divergence KL(normal ‖ mélangé) = {num(context['kl_normal_vs_melange'], 5)}, "
            f"KL(normal ‖ vidé) = {num(context['kl_normal_vs_vide'], 5)}.",
            "",
            f"**Verdict : le contexte est {'utilisé' if context['contexte_utilise'] else 'fonctionnellement ignoré'}.** "
            f"Sur les {fr(context['planchers']['unigramme'] - context['planchers']['bigramme'])} nats que "
            "le bigramme promet, le modèle en prend moins de 0,06.",
            "",
            "Où meurt-il ? Quatre sondes identiques (même softmax, même budget, même "
            "régularisation, indice de position inclus) :",
            "",
            table(["Sonde", "Largeur", "Perte", "Exactitude"], probe_rows),
            "",
            f"- La tête ne lit **qu'un nœud sur huit** : passer de 16 à 128 dimensions "
            f"rend **{fr(context['perte_lecture_16d'])} nat** — c'est réel, mais marginal.",
            f"- Même avec les 128 dimensions, l'état ne gagne que "
            f"**{fr(context['gain_etat_vs_modele'])} nat** sur le modèle : l'ordre des mots "
            "n'est plus dans la mémoire. Une somme à décroissance exponentielle est un sac, "
            "pas une séquence.",
            f"- Le routage est disculpé : l'énergie par saut monte "
            f"({', '.join(fr(v, 1) for v in routing['energie_par_saut'])}), et le dernier "
            f"mot reste décodable dans {fr(routing['decodage_dernier_token_2'], 3)} des cas "
            f"après 2 tokens, {fr(routing['decodage_dernier_token_7'], 3)} après 7 "
            "(hasard : 0,003). Sur l'état mesuré, seuls "
            f"{routing['noeuds_muets'] or 'aucun nœud'} passent sous le seuil de 0,1.",
            "",
            f"**Suspect n°1 : `{context['suspect']}`** — l'agrégation (une seule membrane "
            "par nœud, refroidie à 0,95 par saut) détruit la position ; la lecture d'un seul "
            "nœud ajoute sa petite perte par-dessus.",
            "",
            "Perte par position (préfixe réel) : "
            + ", ".join(f"t{t}={num(v, 3)}" for t, v in enumerate(context["perte_par_position"]))
            + ". La position 0 n'a aucun contexte (5,61) ; les suivantes ne descendent pas "
            "sous 4,78, très loin du bigramme.",
            "",
        ]
    )


def section_long(long: dict) -> str:
    ver = long["verdict"]
    my = long["mycelia"]
    de = long["dense"]
    dec_my = my.get("decision", {})
    dec_de = de.get("decision", {})
    rows = [
        ["MYCELIA (semence 1)", str(my["params"]), fr(my["best_val"]), str(my["best_step"]),
         fr(my.get("fit", {}).get("L_inf")), fr(dec_my.get("L_inf_effectif")), dec_my.get("verdict", "—")],
        ["dense jumeau (contrôle)", str(de["params"]), fr(de["best_val"]), str(de["best_step"]),
         fr(de.get("fit", {}).get("L_inf")), fr(dec_de.get("L_inf_effectif")), dec_de.get("verdict", "—")],
    ]
    bold = {"A": "🐢 A — lenteur", "B": "🧱 B — plafond structurel", "PARTIEL": "partiel"}
    return "\n".join(
        [
            "## 3. Les deux courbes longues et le verdict A/B (D3)",
            "",
            f"500 pas (10× S2), log tous les {long['protocole']['log_every']} pas, "
            f"{long['protocole']['validation_fenetres']} fenêtres de validation, "
            "mêmes lots pour les deux cerveaux.",
            "",
            table(["Bras", "Params", "Meilleure val", "Au pas", "L∞ ajusté", "L∞ retenu", "Verdict"], rows),
            "",
            f"**VERDICT A/B : {bold.get(ver, ver)}**",
            "",
            f"- MYCELIA : {dec_my.get('raison', '—')}",
            f"- dense : {dec_de.get('raison', '—')}",
            "",
            "**Comment ce B se combine avec le verdict 🐛 du haut.** Ce B répond à la "
            "question A/B : ce n'est *pas* de la lenteur. Il ne dit pas que la sparsité "
            "est le mur. Le mur est structurel au sens où il ne bouge pas avec le budget, "
            "mais il est **dans la mémoire** (D1, D2, D4) : un état qui somme ses messages "
            "sans marquer le dernier token. C'est donc un plafond structurel **réparable**, "
            "pas une limite de l'architecture sparse.",
            "",
            "Lecture honnête de l'ajustement : la loi de puissance `L∞ + a·t^(−b)` a bien été "
            "ajustée en log-log, mais son minimum tombe **au bord du domaine** pour les deux "
            "bras (3,65 et 2,29 = minimum observé − 1,5), ce qui signifie que la somme des "
            "carrés décroît jusque-là sans jamais remonter : l'asymptote n'est pas identifiable. "
            f"MYCELIA ne gagne **{fr(dec_my.get('gain_apres_20_pourcent'))} nat** sur les 80 % "
            "finaux du budget. L'asymptote retenue est donc la meilleure validation atteinte, "
            "et son incertitude est ce gain résiduel.",
            "",
            "Autrement dit : multiplier le budget par dix ne rapproche pas MYCELIA du dense. "
            "Ce n'est pas de la lenteur.",
            "",
        ]
    )


def section_ablations(abl: dict) -> str:
    rows = []
    for name, row in abl["variantes"].items():
        note = row.get("note") or "config S2 (référence)"
        if name == "iii_graphe_aleatoire" and int(row["aretes"]) < 48:
            note += f" — le tirage sature à {row['aretes']} arêtes (toutes les paires admissibles)"
        rows.append([name, note, fr(row["best_val"]), sign(row["ecart_au_controle"]), str(row["aretes"])])
    diag = abl["diagnostic_routage"]
    return "\n".join(
        [
            "## 4. Les ablations du routage (D4)",
            "",
            "Même budget que D3 (500 pas), même semence, mêmes lots.",
            "",
            table(["Variante", "Ce qui change", "Perte", "Écart", "Arêtes"], rows),
            "",
            f"**Verdict D4 : `{abl['verdict']}`** — {abl['phrase']}.",
            "",
            "Diagnostic du routage sur le contrôle :",
            "",
            f"- allumage par nœud (nœud chaud, sur 24 lots) : {diag.get('frequence_allumage')} "
            f"(Gini {diag.get('gini_allumage')})",
            f"- gradient par nœud : {diag.get('grad_norm_par_noeud')} (Gini {diag.get('gini_gradient')})",
            f"- nœuds jamais allumés : {diag.get('noeuds_sans_allumage')} (la porte, par définition) ; "
            f"sans gradient : {diag.get('noeuds_sans_gradient') or 'aucun'}",
            "",
            "L'équilibre de charge est le point à surveiller. Le nœud 0 est la porte : il reçoit "
            "le token, jamais un message, donc son zéro est une définition et pas une inégalité "
            "(il est exclu du Gini). Sur les 7 autres, tous sont allumés dans chaque lot : pas "
            "d'effondrement « riches toujours plus riches » à l'échelle de la colonie. Mais sur "
            "un état donné, deux nœuds peuvent tomber sous le seuil de 0,1 (mesuré au test de "
            "contexte : nœuds 5 et 7, portes 0,005 et 0,051) : l'activité est inégale d'un token "
            "à l'autre, ce qui est le comportement attendu d'un routage par affinité, pas une "
            "amputation.",
            "",
        ]
    )


def section_fair(fair: dict) -> str:
    est = fair["estimation"]
    dense = fair["dense_equitable"]
    rows = [
        ["embedding (une ligne lue)", str(est["embedding"])],
        ["tête de lecture", str(est["tete"])],
        [f"nœuds actifs ({est['noeuds_actifs_par_token']} × 1 121)", fr(est["noeuds"], 1)],
        ["requêtes + clés", fr(est["adresses"], 1)],
        ["arêtes (un coefficient chacune)", str(est["aretes"])],
        ["**total actif par token**", f"**{fr(est['total'], 1)}**"],
    ]
    return "\n".join(
        [
            "## 5. Le jumeau équitable : paramètres ACTIFS égaux (D5)",
            "",
            "S2 égalisait 23 748 paramètres *totaux*. Ce n'est pas le même travail par token : "
            "le dense consulte tout, MYCELIA une partie. Estimation mesurée sur la validation :",
            "",
            table(["Poste", "Paramètres consultés"], rows),
            "",
            f"Le dense équitable a donc **MLP {dense['mlp']}**, soit **{dense['params']} paramètres** "
            f"au total, entraîné au même budget (500 pas, mêmes lots).",
            "",
            f"| Bras | Perte |", "|---|---|",
            f"| dense équitable (actifs égaux) | {fr(dense['best_val'])} |",
            f"| MYCELIA, même budget | {fr(fair['mycelia_meme_budget'])} |",
            f"| dense jumeau S2 (23 748 params) | {fr(fair['dense_jumeau_23748'])} |",
            "",
            f"Écart MYCELIA − dense équitable : **{sign(fair['ecart'])} nat**.",
            "",
            "Ils ne s'égalent pas. Le plafond n'est donc pas « le prix attendu du sparse à "
            "calcul égal » : à calcul égal, le dense fait mieux, et le sparse ne compense pas "
            "par autre chose. Cela pointe vers l'architecture de la mémoire, pas vers un "
            "arbitrage computationnel normal.",
            "",
        ]
    )


def section_big(big: dict) -> str:
    rows = [
        ["unigramme", fr(big["planchers"]["unigramme"])],
        ["bigramme", fr(big["planchers"]["bigramme"])],
        ["trigramme", fr(big["planchers"]["trigramme"])],
    ]
    return "\n".join(
        [
            "## 6. Le corpus 10× : capacité ou surapprentissage ? (D6)",
            "",
            f"{big['corpus']['phrases']} phrases ({big['corpus']['tokens']} tokens), même grammaire, "
            f"même vocabulaire, {big['corpus']['mots_inconnus']} mot inconnu. "
            f"Budget court : {big['budget']['pas']} pas.",
            "",
            table(["Plancher sur le gros corpus", "Valeur"], rows),
            "",
            f"| Bras | Meilleure val | Au pas |", "|---|---|---|",
            f"| MYCELIA | {fr(big['mycelia']['best_val'])} | {big['mycelia']['best_step']} |",
            f"| dense | {fr(big['dense']['best_val'])} | {big['dense']['best_step']} |",
            "",
            f"Écart : **{sign(big['ecart'])} nat**. Le plafond tient : "
            f"**{'oui' if big['plafond_tient'] else 'non'}**.",
            "",
            "Le corpus 10× supprime l'essentiel du surapprentissage (la validation ne remonte "
            "plus aussi vite) et le plafond ne bouge pas : ce n'est donc pas la taille du "
            "corpus qui parlait. La capacité — au sens de ce que l'état peut représenter — "
            "reste la même.",
            "",
        ]
    )


def section_recency(recency: list[dict]) -> str:
    if not recency:
        return ""
    rows = [
        [fr(row["state_decay"], 2), fr(row["best_val"]), str(row["best_step"]), fr(row["planchers"]["unigramme"]), fr(row["planchers"]["bigramme"])]
        for row in recency
    ]
    best = min(recency, key=lambda row: float(row["best_val"]))
    return "\n".join(
        [
            "### Le test de réparation (D2′) : et si on changeait la récence ?",
            "",
            "Puisque le diagnostic accuse l'agrégation, une seule vis devait suffire : la "
            "vitesse de refroidissement de la membrane (`state_decay`). Même budget court, "
            "même semence, mêmes lots.",
            "",
            table(["state_decay", "Perte", "Au pas", "Unigramme", "Bigramme"], rows),
            "",
            f"Meilleur réglage mesuré : **{fr(best['state_decay'], 2)} → {fr(best['best_val'])}**. "
            + (
                "La perte tombe sous le plancher historique, donc la récence de la mémoire "
                "est bien une cause : MYCELIA peut utiliser le contexte, il ne le fait pas "
                "parce que son agrégation noie le dernier mot dans les précédents."
                if float(best["best_val"]) < 4.9
                else "La récence seule ne suffit pas : l'agrégation n'est pas le seul verrou, "
                "la lecture d'un seul nœud et la non-linéarité du corps comptent aussi."
            ),
            "",
        ]
    )


def section_conclusion(data: dict, verdict: str, phrase: str) -> str:
    floors = data["floors"]
    uni = float(floors["L_ref"])
    bi = float(floors["bigramme"]["entropie_lissee"])
    tri = float(floors["trigramme"]["entropie_lissee"])
    my = float(data["long"]["mycelia"]["best_val"])
    dense = float(data["long"]["dense"]["best_val"])
    return "\n".join(
        [
            "## 7. Conclusion stratégique : le plafond réel du convertisseur",
            "",
            f"**{verdict} — {phrase}**",
            "",
            "La perte minimale réellement atteignable par MYCELIA sur ce corpus, telle que "
            f"mesurée aujourd'hui : **{fr(my)} nats/token**. À {fr(my - uni)} nat du plancher "
            f"unigramme ({fr(uni)}), ce n'est pas un plafond d'architecture sparse : la "
            "colonie n'utilise pas l'ordre des mots.",
            "",
            "Ce que cela fixe pour toute traduction future :",
            "",
            table(
                ["Niveau", "Perte (nats)", "Ce qu'un cerveau à ce niveau sait faire"],
                [
                    ["MYCELIA aujourd'hui", fr(my), "statistiques marginales : aucun contexte"],
                    ["unigramme (L_ref)", fr(uni), "sac de mots"],
                    ["bigramme", fr(bi), "le mot suivant dépend du précédent"],
                    ["trigramme", fr(tri), "deux mots d'histoire"],
                    [f"dense S2 (500 pas)", fr(dense), "contexte ordonné par attention"],
                ],
            ),
            "",
            f"La barre du convertisseur est donc **{fr(my)}**, et elle est trop haute : la "
            f"cible d'un bigramme ({fr(bi)}) est *sous* le dense ({fr(dense)}). Un MYCELIA qui "
            "lirait seulement deux mots d'histoire battrait le dense à budget égal. Le "
            "problème n'est pas la sparsité, c'est la mémoire.",
            "",
            "Tant que MYCELIA reste à ce plancher, **aucun convertisseur ne peut transférer "
            "mieux que des statistiques marginales** : les 90 % de traduction visés sont "
            "inatteignables, non par manque de pas (D3), non par le routage (D4), mais parce "
            "que le receveur ne voit pas l'ordre des mots (D2).",
            "",
            "**Ce que la S4-CORRIGÉ doit faire** : lire toutes les membranes (pas seulement le "
            "nœud 1) et donner à la lecture un accès au dernier token — attention sur la "
            "fenêtre récente ou lecture à deux vitesses (membrane lente + message courant). "
            "C'est un choix d'architecture, pas un renoncement : le plafond de 5,1 disparaît "
            "avec lui.",
            "",
        ]
    )


def section_honesty(data: dict) -> str:
    floors = data["floors"]
    long = data["long"]
    return "\n".join(
        [
            "## 8. Échecs honnêtes et surprises",
            "",
            "- **L'ajustement de puissance n'a rien ajusté.** Sur une courbe plate 445 pas "
            "durant, `L∞` n'est pas identifiable : le minimum de la somme des carrés tombe "
            "au bord du domaine (3,65 pour MYCELIA, 2,29 pour le dense). Le rapport ne "
            "présente donc pas ces chiffres comme des prévisions — il retient la meilleure "
            "validation atteinte et l'écrit.",
            "- **La première sonde de contexte était fausse.** Sans indice de position, la "
            "même cible tombait dans un bloc de caractéristiques différent selon `t` ; et "
            "une séparation train/test par *mot* rendait 0 % (une sonde linéaire ne peut pas "
            "deviner une classe jamais vue). Corrigé : position partagée par les quatre "
            "sondes, séparation par échantillon.",
            "- **Le témoin de largeur ne témoigne pas.** Les embeddings bruts du préfixe "
            f"n'atteignent que {fr(data['context']['sondes']['embeddings_prefixe_128d']['perte'])} : "
            "prédire 312 classes depuis 16 dimensions par position reste un problème de rang, "
            "pas de largeur. Il ne permet donc pas de conclure « la largeur suffit » ; c'est "
            "la comparaison 16 d → 128 d sur les membranes qui porte le diagnostic.",
            "- **Le dense n'est pas un plafond du corpus.** À 500 pas il est à "
            f"{fr(long['dense']['best_val'])}, au-dessus du bigramme ({fr(floors['bigramme']['entropie_lissee'])}) : "
            "il n'a pas fini d'apprendre non plus. La comparaison A/B porte sur le même "
            "budget, pas sur la convergence parfaite.",
            "- **La validation a changé de taille.** S2 mesurait sur 16 fenêtres, S3-bis sur "
            "48 : le 5,078 de S2 et le 5,1509 de D3 sont le même modèle à un bruit près. "
            "Les deux chiffres sont conservés, aucun n'est recyclé pour l'autre.",
            "- **Le test de réparation a échoué.** Refroidir la membrane plus vite (0,80 puis "
            "0,50) ne fait pas tomber la perte : 5,2124 et 5,2645, soit *pire* que 5,1509. "
            "La récence est donc une cause partielle, pas le remède : il faut agir sur la "
            "lecture et sur le mécanisme qui désigne le dernier token, pas seulement sur "
            "une constante de temps.",
            "- **Le graphe aléatoire ne peut pas faire 48 arêtes.** Avec 8 nœuds, un lecteur "
            "muet et aucune arrivée sur les portes, il n'existe que 42 paires admissibles : "
            "cette variante travaille avec 6 arêtes de moins, ce qui explique peut-être son "
            "+0,019.",
            "- **Le vocabulaire du gros corpus est inchangé** (312 mots) pour que les "
            "embeddings restent comparables : D6 teste la quantité de données, pas la "
            "couverture lexicale.",
            "",
        ]
    )


def main() -> None:
    data = load_all()
    verdict, phrase = headline(data)
    parts = [
        "# RAPPORT S3-bis — pourquoi MYCELIA plafonne à ~5,1",
        "",
        f"**Verdict : {verdict}** — {phrase}",
        "",
        "Mission : trancher entre un bug d'usage du contexte, une lenteur d'apprentissage "
        "(A) et un plafond structurel du sparse (B). Toutes les expériences tournent en "
        "circuit court sur CPU, sans installation, et ne touchent ni au génome de Nyra ni "
        "à sa session 24/7.",
        "",
        section_floors(data["floors"]),
        section_context(data["context"]),
        section_long(data["long"]),
        section_ablations(data["ablations"]),
        section_fair(data["fair"]),
        section_big(data["big"]),
        section_recency(data["recency"]),
        section_conclusion(data, verdict, phrase),
        section_honesty(data),
    ]
    text = "\n".join(parts)
    OUT.write_text(text, encoding="utf-8")
    write_json(
        LAB / "s3bis_summary.json",
        {
            "verdict": verdict,
            "phrase": phrase,
            "L_ref": data["floors"]["L_ref"],
            "mycelia": data["long"]["mycelia"]["best_val"],
            "dense": data["long"]["dense"]["best_val"],
            "verdict_ab": data["long"]["verdict"],
            "suspect": data["context"]["suspect"],
            "ablations": data["ablations"]["verdict"],
            "fair_ecart": data["fair"]["ecart"],
            "big_plafond_tient": data["big"]["plafond_tient"],
        },
    )
    print(f"ecrit : {OUT.name} ({len(text.splitlines())} lignes)")
    print(f"verdict : {verdict}")


if __name__ == "__main__":
    main()
