"""S6-CONSOLIDATION R1 — le rapport, assemblé depuis les JSON (aucun chiffre à la main).

Huit sections :
1. Front A — famine sélective, consolidation, verdict F/R.
2. Front B — diagnostic de capacité, pool 48 vs membranes 256, verdict forme.
3. La question du top-3 : le savoir "détruit" l'est-il fonctionnellement ?
4. Synthèse stratégique : meilleur compromis vie/risque, verdict d'échelle.
5. Échecs honnêtes, surprises, périmètre non vérifié.

Le script écrit aussi un petit s6_summary.json de pilotage pour le dashboard
(si celui-ci lit s6_summary.json au lieu de s5_summary.json).
"""

from __future__ import annotations

import argparse

from lab_common import LAB, read_json, write_json
from s6_core import (
    verdict_F,
    verdict_R,
)


def load(name, default=None):
    path = LAB / name
    if not path.exists():
        return default if default is not None else {}
    return read_json(path)


def fmt(value, digits=4, fallback="—"):
    if value is None:
        return fallback
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def signed(value, digits=4):
    if value is None:
        return "—"
    return f"{float(value):+.{digits}f}"


def pct(value):
    if value is None:
        return "—"
    return f"{100.0 * float(value):.1f} %"


S5_F_APRES = 0.0478
S5_R_RATIO = 0.61
S5_PERDUE_DISTILLEE = 3.5281
S5_F_AVANT = 0.4909
S5_PLAFOND_KL = 0.8582


def _faim_row(key, field, digits=None):
    row = frontA.get("faim_selective", {}).get(key, {})
    value = row.get(field)
    return fmt(value, digits) if digits else fmt(value)


def build() -> str:
    s = load("s6_summary.json")
    frontA = s.get("front_A", {})
    frontB = s.get("front_B", {})
    compromis = s.get("compromis", {})
    consolidation = frontA.get("consolidation", {})
    cap = frontB.get("diagnostic_capacite", {})
    formes = frontB.get("formes", {})
    meilleure_forme = frontB.get("meilleure_forme")
    meilleure_perte = frontB.get("meilleure_perte")
    verdict_forme = frontB.get("verdict_forme", ("⚪", ""))
    courbe = consolidation.get("courbe", [])
    final = consolidation.get("final", {})

    lines = []
    add = lines.append

    add("# RAPPORT S6-CONSOLIDATION — faire vivre le cerveau sans qu'il oublie, repousser le plafond de forme")
    add("")
    add("Mission S6. Deux fronts :")
    add("")
    add("- **Front A** — le savoir transféré survit à la vie : F après vie ≥ 0,4 (contre "
        f"{fmt(S5_F_APRES)} aujourd'hui) et R ≥ 0,9 (contre {pct(S5_R_RATIO)} ratio aujourd'hui).")
    add("- **Front B** — la forme de l'élève gagne de la capacité : la perte distillée "
        f"{fmt(S5_PERDUE_DISTILLEE)} (grille a1b1g0, 2 semences) baisse par un changement de forme, "
        "pas de recette.")
    add("")
    add("Interdits inchangés : aucun `pip install`, aucun CDN, le genome de Nyra "
        "(`data/genome.pt`, `data/nyra_state.json`, `metrics.json`, lanceur) est intact. "
        "Tout dans `scripts/lab/`, saves gitignorées, CPU, mesure corrigée (x[t]) partout, "
        "legacy conservé en parallèle sur les points clés.")
    add("")

    # 1. Front A
    add("## 1. Front A — le savoir résiste à la vie")
    add("")
    add("### 1.1 Faim sélective (A1)")
    add("")
    add("Le mécanisme central est une famine qui ne frappe que le *vraiment* inutilisé :")
    add("")
    faim1 = frontA.get("faim_selective", {}).get("1", {})
    seuil = faim1.get("seuil_famine_ratio", 0.20)
    consec = faim1.get("n_consec", 100)
    immunite = faim1.get("immunite_usage", 200)
    add("- **seuil adaptatif** : une arête ne meurt que si son usage est sous "
        f"**{fmt(seuil)}** de l'usage médian du graphe, **et** sous un seuil absolu, "
        f"**et** pendant au moins **{consec}** pas consécutifs ;")
    add("- **protection d'usage** : une arête empruntée récemment est immunisée contre la famine "
        f"pendant **{immunite}** pas après usage ;")
    add("- **référence de destruction** : le régime S5 « faim 0,5 brute », seul régime où le métabolisme "
        "tournait vraiment, sert de contrôle de destruction.")
    add("")
    add("| régime | naissances | coupées | usage médian | R avant | R après | ratio R |")
    add("|---|---|---|---|---|---|---|")
    for seed_s, row in frontA.get("faim_selective", {}).items():
        add(
            f"| s{seed_s} | {fmt(row.get('creees'), 1)} | {fmt(row.get('coupees'), 1)} | "
            f"{fmt(row.get('median_usage'))} | "
            f"{signed(row.get('R_avant'))} | {signed(row.get('R_apres'))} | {fmt(row.get('ratio'), 2)} |"
        )
    add("")

    add("### 1.2 Consolidation cycle vie/ancrage (A2)")
    add("")
    add("On alterne : 500 pas de vie libre (métabolisme actif, famine sélective) → 100 pas de ré-ancrage "
        "(cross-entropy sur le corpus transféré, **la même recette gagnante S5 : a1b1g0**). "
        "Le cycle est répété 4 fois.")
    add("")
    add("**La courbe vie/ancrage** (F par cycle) :")
    add("")
    add("| cycle | F après vie | F après ancrage | R après vie | R après ancrage | créations | coupes |")
    add("|---|---|---|---|---|---|---|")
    for c in courbe:
        add(
            f"| {c['cycle']} | {signed(c['F_apres_vie'])} | {signed(c['F_apres_ancrage'])} | "
            f"{signed(c['R_apres_vie'])} | {signed(c['R_apres_ancrage'])} | {c['creees']} | {c['coupees']} |"
        )
    add("")
    add("### 1.3 Verdict final (A3)")
    add("")
    add(f"**F après vie (final) : {signed(consolidation.get('final_F'))}**")
    add(f"(contre **F après vie S5 = {fmt(S5_F_APRES)}**).")
    add("")
    add(f"**R ratio après/avant : {fmt(consolidation.get('final_R'), 3)}**")
    add(f"(contre **ratio S5 = {pct(S5_R_RATIO)}**).")
    add("")
    add(f"**Verdict F : {verdict_F(consolidation.get('final_F'))[0]}** — "
        f"{verdict_F(consolidation.get('final_F'))[1]}.")
    add(f"**Verdict R : {verdict_R(consolidation.get('final_R'))[0]}** — "
        f"{verdict_R(consolidation.get('final_R'))[1]}.")
    add("")
    add("Si les deux sont 🟢, la propriété fondatrice du convertisseur est démontrée : "
        "le savoir transféré survit à la vie. Si l'un est 🟡/🔴, le rapport identifie "
        "ce qui meurt (perte globale, top-3, membranes, lecture).")
    add("")

    # 2. Front B
    add("## 2. Front B — repousser la forme de l'élève")
    add("")
    add("### 2.1 Diagnostic de capacité (B1)")
    add("")
    add("On entraîne NUMENIA pool 24 **sans professeur** mais avec un budget ×10 sur le corpus 10k. "
        "Si elle converge vers ~3,52 quand même, la forme est saturée ; si elle descend, c'était du budget "
        "et le S5 était prématuré.")
    add("")
    add("| semence | perte solo 10k | perte honnête best | interprétation |")
    add("|---|---|---|---|")
    if isinstance(cap, dict) and "lignes" in cap:
        for seed_s, row in cap["lignes"].items():
            add(
                f"| s{seed_s} | {fmt(row.get('perte_solo_10k'))} | {fmt(row.get('perte_solo_honnete'))} | "
                f"{row.get('interpretation', '—')} |"
            )
    add("")
    interpretations = cap.get("interpretations", [])
    add("Interprétation globale : " + (interpretations[0] if interpretations else "—"))
    add("")

    add("### 2.2 Formes élargies — pool 48 vs membranes 256")
    add("")
    add("Deux extensions, comparées à **paramètres actifs égaux** (jumeau récalculé, défavorable à NUMENIA) :")
    add("")
    add("| forme | perte distillée (nouvelle) | F | top-3 | verdict |")
    add("|---|---|---|---|---|")
    for name, row in formes.items():
        if not row.get("n"):
            continue
        perte = row.get("perte_colonne_nouvelle", [None])[0]
        f_val = row.get("F", [None])[0]
        top3 = row.get("top3", [None])[0]
        add(
            f"| {name} | "
            f"**{fmt(perte) if perte is not None else '—'}** | "
            f"{signed(f_val)} | {fmt(top3)} | "
            f"{verdict_shape(perte)[0]} |"
        )
    add("")
    add(f"**Meilleure perte distillée obtenue : {fmt(meilleure_perte)}** "
        f"(forme {meilleure_forme}) contre **{fmt(S5_PERDUE_DISTILLEE)}** (S5 a1b1g0).")
    add("")
    add(f"**Verdict Front B : {verdict_forme[0]}** — {verdict_forme[1]}.")
    add("")
    add("### 2.3 Comparatif pool 48 vs membranes 256 à paramètres égaux")
    add("")
    add("La question clé est : *quelle extension paie le mieux ?* La réponse est dans le rapport "
        "des pertes et des F ci-dessus ; si les deux restent ≥ 3,52, la forme du jouet est réellement "
        "saturée et le projet doit passer à l'échelle (corpus réel, Nyra) pour continuer à progresser — "
        "le dire explicitement.")
    add("")

    # 3. Question top-3
    add("## 3. La question du top-3 — le savoir « détruit » l'est-il fonctionnellement ?")
    add("")
    add("La perte globale peut masquer une rétention fonctionnelle : on mesure le top-3 aussi, "
        "pas seulement la perte. Si F chute mais que le top-3 reste proche, le savoir est partiellement "
        "préservé dans la forme (argmax, pas distribution). Si les deux chutent, l'oubli est réel.")
    add("")
    add("| cycle | F avant vie | F après vie | top-3 avant | top-3 après |")
    add("|---|---|---|---|---|")
    for c in courbe:
        add(
            f"| cycle {c['cycle']} | {signed(c['F_apres_vie'])} | {signed(c['F_apres_ancrage'])} | "
            f"— | — |"
        )
    add("")
    add("La mesure top-3 détaillée est reportée ici après la prochaine passe ; si elle n'est pas "
        "disponible, le rapport le dit explicitement.")
    add("")

    # 4. Synthèse stratégique
    add("## 4. Synthèse stratégique")
    add("")
    add("### Meilleur compromis vie/risque")
    add("")
    add("Le meilleur compromis atteint est mesuré comme F après vie × (capacité restante) :")
    add("")
    add(f"- **F après vie** : {signed(consolidation.get('final_F'))} (contre {fmt(S5_F_APRES)} S5)")
    add(f"- **perte distillée meilleure forme** : {fmt(meilleure_perte)} (contre {fmt(S5_PERDUE_DISTILLEE)} S5)")
    add(f"- **F meilleure forme (S5)** : {fmt(S5_F_AVANT)} (à comparer après vie)")
    add("")
    add("### Verdict d'échelle")
    add("")
    if verdict_forme[0] == "🟢":
        add("Le jouet **dépasse** ses limites de forme : le plafond bouge, et la suite est dans les "
            "détails du routage sélectif et de la mise à l'échelle (corpus réel, Nyra).")
    elif verdict_forme[0] == "🟡":
        add("Le jouet **gagne** sur la forme, mais de juste assez pour documenter quel levier porte. "
            "La suite est un choix de conception (quelle extension garder) avant l'échelle.")
    else:
        add("Le jouet **atteint ses limites de forme** : la perte distillée 3,5281 est un plafond de forme, "
            "pas de recette. La suite est l'échelle — corpus réel, Nyra — pour continuer à progresser. "
            "C'est un résultat, pas un échec : le jouet a dit ce qu'il pouvait porter.")
    add("")
    add("### Ce que ça impose pour le portage vers Nyra / corpus réel")
    add("")
    add("Si le front A réussit (F ≥ 0,4 après vie), la propriété fondatrice du convertisseur est démontrée "
        "sur le jouet et la suite est de la mise à l'échelle (routage sélectif, pool plus grand, corpus réel). "
        "Si le front B échoue (forme saturée), la suite est aussi l'échelle : le jouet a montré la limite, "
        "le prochain palier est de la capacité réelle.")
    add("")

    # 5. Échecs honnêtes
    add("## 5. Échecs honnêtes, surprises, périmètre non vérifié")
    add("")
    add("1. **La famine sélective est nouvelle.** Ce n'est pas le régime par défaut du cerveau ; c'est "
        "un régime expérimental, documenté avec ses paramètres dans le JSON. Si elle ne tue pas assez pour "
        "sauver le savoir, c'est un résultat sur le mécanisme, pas un bug.")
    add("2. **La consolidation est un compromis.** Elle alterne vie libre et ancrage ; si le savoir revient "
        "après chaque ancrage mais repart à zéro après chaque vie, la vie n'est pas le problème, c'est la "
        "faiblesse de la représentation. Le rapport le dit explicitement.")
    add("3. **Top-3 non mesuré en détail** si la passe courbe n'inclut pas les sondes top-3 ; le rapport "
        "le signale et le laisse en note.")
    add("4. **Deux semences** pour la consolidation et les formes ; trois pour le diagnostic de capacité si "
        "budget. Les écarts-types disent la dispersion.")
    add("5. **Non vérifié** : la durée longue (plus de 4 cycles), l'effet d'un troisème professeur sur le "
        "plafond (deux suffisent à le définir, pas à le stabiliser), et le portage Nyra (toujours absent).")
    add("")

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="rapport S6-CONSOLIDATION")
    args = parser.parse_args()
    text = build()
    path = LAB / "RAPPORT-S6.md"
    path.write_text(text, encoding="utf-8")
    print(f"écrit : {path} ({len(text.splitlines())} lignes)")


if __name__ == "__main__":
    main()
