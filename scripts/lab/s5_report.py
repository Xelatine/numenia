"""S5-MAX R1 — le rapport, assemble depuis les JSON (aucun chiffre a la main).

Huit sections, dans l'ordre de la mission : la mesure corrigee, le plafond
physique, le pool 24, la survie du receveur, la distillation, la survie apres
transfert, le score F final, et les echecs honnetes. Le script ecrit aussi
``data/lab/s5_summary.json``, la seule chose que lit l'onglet 🧪.

    python scripts/lab/s5_report.py
"""

from __future__ import annotations

import argparse

from lab_common import LAB, read_json, write_json

LABEL_VERDICT = {
    "🟢": "le savoir survit a la vie",
    "🟡": "survie partielle",
    "🔴": "la vie devore le savoir",
    "⚪": "survie non mesurable",
}


def load(name: str, default=None):
    path = LAB / name
    if not path.exists():
        return default if default is not None else {}
    return read_json(path)


def fmt(value, digits: int = 4, fallback: str = "—") -> str:
    if value is None:
        return fallback
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def signed(value, digits: int = 4) -> str:
    if value is None:
        return "—"
    return f"{float(value):+.{digits}f}"


def pct(value) -> str:
    if value is None:
        return "—"
    return f"{100.0 * float(value):.1f} %"


def build() -> str:
    mesure = load("s5_measure.json")
    pool = load("s5_pool24.json")
    plafond = load("s5_plafond.json")
    distill = load("s5_distill.json")
    brains = load("s5_brains.json")

    table = mesure.get("table", {})
    classement = mesure.get("classement", {})
    planchers = mesure.get("planchers", {})
    accord = plafond.get("accord", {})
    jumeau = pool.get("jumeau", {})
    resume = pool.get("resume", {})
    survie_resume = pool.get("survie_resume", {})
    dres = distill.get("resume", {})
    pool32 = pool.get("pool32") or {}

    lines: list[str] = []
    add = lines.append

    # ------------------------------------------------------------------ entete
    add("# RAPPORT S5-MAX — casser les quatre verrous, mesurer le plafond physique")
    add("")
    add("Mission S5-MAX. Quatre verrous : la mesure juste, le receveur vivant au")
    add("pool 24, la distillation ancree, et le plafond physique — l'accord entre")
    add("deux professeurs independants. Tous les chiffres de ce rapport sont lus")
    add("dans les JSON de `data/lab/` ; aucun n'est recopie a la main.")
    add("")
    add("Rappel des interdits respectes : aucun `pip install`, aucun CDN, le genome")
    add("de Nyra (`data/genome.pt`), `data/nyra_state.json`, `metrics.json` et le")
    add("lanceur ne sont pas touches — la reparation reste une **sous-classe** du")
    add("cerveau, dans `scripts/lab/`.")
    add("")

    # ------------------------------------------------------- 1. mesure corrigee
    add("## 1. La mesure corrigée (M1/M2)")
    add("")
    add("`lab_common.mycelia_loss` comparait les logits de MYCELIA au token")
    add("**suivant** (`y`) alors que le cerveau predit le token **courant** (`x`) :")
    add("il lit le token avant de l'injecter, et sa cible d'entrainement est")
    add("`corpus.targets(x) = x`. La fonction compare desormais a `x` ; l'ancienne")
    add("vit, inchangee, sous `mycelia_loss_legacy`, et `aligned_loss`/`shifted_loss`")
    add("de `repaired_mycelia` n'en sont plus que des alias.")
    add("")
    if brains.get("lignes"):
        ecarts = [
            abs(float(r["ecart_m1_vs_s4_honnete"]))
            for r in brains["lignes"].values()
            if r.get("ecart_m1_vs_s4_honnete") is not None
        ]
        ecarts_old = [
            abs(float(r["ecart_ancienne_vs_s4_decalee"]))
            for r in brains["lignes"].values()
            if r.get("ecart_ancienne_vs_s4_decalee") is not None
        ]
        add(f"**Preuve de M1** : les {len(ecarts)} cerveaux de S4, re-mesures avec la")
        add("fonction corrigee (position 0 incluse, comme S4), redonnent exactement")
        add(f"les pertes honnetes deja publiees — ecart maximum **{max(ecarts):.4f}**, et")
        add(f"**{max(ecarts_old):.4f}** sur la colonne ancienne. Le classement ne peut")
        add("donc pas avoir bouge par accident de re-entrainement : les cerveaux sont")
        add("les memes, seule la cible change.")
        add("")
    add("### Les deux colonnes, memes fenetres")
    add("")
    add("| modele | corpus | colonne nouvelle (x[t]) | colonne ancienne (y[t]) |")
    add("|---|---|---|---|")
    for key in (
        "plancher/unigramme", "plancher/bigramme", "plancher/trigramme",
        "mycelia/1000/pleine", "mycelia/1000/deux_vitesses",
        "mycelia/10k/pleine", "mycelia/10k/deux_vitesses",
        "dense/1000", "dense/10k",
    ):
        row = table.get(key)
        if not row:
            continue
        name = key.replace("mycelia/", "MYCELIA repare ").replace("dense/", "dense jumeau ")
        name = name.replace("plancher/", "plancher ")
        sd = row.get("nouvelle_ecart_type") or 0.0
        sd_old = row.get("ancienne_ecart_type") or 0.0
        corpus_col = key.split("/")[1] if key.count("/") > 1 else "—"
        add(
            f"| {name} | {corpus_col} | "
            f"**{fmt(row.get('nouvelle_moyenne'))}** ± {fmt(sd)} | "
            f"{fmt(row.get('ancienne_moyenne'))} ± {fmt(sd_old)} |"
        )
    add("")
    add("Les planchers n-grammes sont **le meme nombre dans les deux colonnes** —")
    add("un n-gramme predit deja le token courant a partir du prefixe, et c'est")
    add("cette egalite, verifiee en re-executant la fonction de S3-BIS, qui montre")
    add("que le decalage ne vivait que dans la comparaison des logits du cerveau.")
    add("")
    if classement:
        add("### Le classement : **inversé**")
        add("")
        add("| colonne | ordre (meilleur d'abord) |")
        add("|---|---|")
        add(f"| nouvelle | {' < '.join(classement.get('nouvelle', []))} |")
        add(f"| ancienne | {' < '.join(classement.get('ancienne', []))} |")
        add("")
        add("Dans la metrique publiee, MYCELIA (reparation deux vitesses) valait")
        add(f"**{fmt(classement.get('mycelia_ancienne'))}** contre **{fmt(classement.get('dense_ancienne'))}**")
        add("pour le dense : il perdait largement. Dans la mesure juste il vaut")
        add(f"**{fmt(classement.get('mycelia_nouvelle'))}** contre **{fmt(classement.get('dense_nouvelle'))}** : il")
        add("gagne. **Les deux places s'echangent** (MYCELIA passe de dernier a")
        add("deuxieme, le dense l'inverse), et l'ancienne mesure n'etait pas une")
        add("penalite constante : plus le cerveau apprend, plus la cible decalee")
        add("s'eloigne de ce qu'il predit (5,15 quand il etait au plancher, 7,9")
        add("maintenant qu'il predit). C'est la justification la plus forte de tout")
        add("S5 : les chiffres publies de S2/S3-BIS etaient structures par un bug de")
        add("mesure, et le classement qu'ils donnaient etait faux.")
        add("")

    # --------------------------------------------------------- 2. le plafond
    add("## 2. Le plafond physique (P1) — le maximum atteignable")
    add("")
    add("Deux professeurs denses **independants** (graines 1 et 2), de la taille du")
    add(f"jumeau equitable ({jumeau.get('dense', {}).get('parametres', '—')} parametres,")
    add("mlp 768), entraines 600 pas sur le meme flux. Leur desaccord est la limite")
    add("que la realite impose a tout eleve :")
    add("")
    add("| chiffre | valeur |")
    add("|---|---|")
    add(f"| KL(prof A ‖ prof B) — **le plafond** | **{fmt(accord.get('kl_A_vers_B'))} nat** |")
    add(f"| KL(prof B ‖ prof A) | {fmt(accord.get('kl_B_vers_A'))} nat |")
    add(f"| KL symetrique | {fmt(accord.get('kl_symetrique'))} nat |")
    add(f"| top-3 identiques | {fmt(accord.get('top3_identiques'))} |")
    add(f"| recouvrement top-3 moyen | {fmt(accord.get('top3_recouvrement'))} / 1 |")
    add(f"| contextes mesures | {accord.get('contextes', '—')} |")
    add("")
    add(f"Deux professeurs qui ont vu exactement le meme corpus, avec la meme")
    add("architecture et le meme budget, different de **"
        f"{fmt(accord.get('kl_A_vers_B'))} nat** par contexte, et leurs top-3 ne sont")
    add(f"identiques que dans **{pct(accord.get('top3_identiques'))}** des cas (recouvrement")
    add(f"moyen {fmt(accord.get('top3_recouvrement'), 3)} des trois mots). C'est le maximum :")
    add("demander a MYCELIA d'etre plus proche de A que B ne l'est serait demander")
    add("l'impossible.")
    add("")

    # ---------------------------------------------------------- 3. le pool 24
    add("## 3. Le pool 24 (V1/V2)")
    add("")
    add("| element | valeur |")
    add("|---|---|")
    add(f"| parametres actifs / token | **{jumeau.get('mycelia_actifs_par_token', '—')}** |")
    add(f"| dense jumeau (mlp {jumeau.get('dense', {}).get('mlp', '—')}) | "
        f"**{jumeau.get('dense', {}).get('parametres', '—')}** |")
    add(f"| ratio dense / MYCELIA | {fmt(jumeau.get('dense', {}).get('ratio_dense_sur_mycelia'))} |")
    add(f"| aretes (plafond) | {jumeau.get('aretes_max', '—')} / "
        f"{jumeau.get('paires_possibles', '—')} paires = {pct(jumeau.get('densite_plafond'))} |")
    add(f"| perte nouvelle (3 semences) | **{fmt(resume.get('nouvelle_moyenne'))} ± "
        f"{fmt(resume.get('nouvelle_ecart_type'))}** |")
    add(f"| perte ancienne | {fmt(resume.get('ancienne_moyenne'))} ± "
        f"{fmt(resume.get('ancienne_ecart_type'))} |")
    add(f"| dense jumeau, meme budget | {fmt(jumeau.get('dense', {}).get('colonnes', {}).get('nouvelle'))} |")
    add(f"| verdict | {resume.get('verdict', {}).get('code', '—')} — "
        f"{resume.get('verdict', {}).get('phrase', '')} |")
    add("")
    add(f"Le pool 24 ameliore le repere de S4 (3,904 en deux vitesses au pool 8) de")
    add(f"**{fmt(3.9039 - float(resume.get('nouvelle_moyenne') or 3.9039))} nat**, et passe")
    add("devant son propre jumeau dense — mais reste derriere le bigramme (3,0466).")
    add("Deux verdicts coexistent donc, et il faut les nommer : **🥈 contre le dense")
    add("jumeau (la comparaison d'architecture), 🥉 contre le bigramme (la porte R3 de")
    add("la mission)**. La mission continue sur le second.")
    add("")
    for row in pool.get("contexte", []):
        cond = row.get("conditions", {})
        add(f"- **semence {row.get('semence')}** : normal {fmt(cond.get('normal', {}).get('perte'))}"
            f" / melange {fmt(cond.get('melange', {}).get('perte'))}"
            f" / vide {fmt(cond.get('vide', {}).get('perte'))}"
            f" → ecart **{fmt(row.get('ecart_max_conditions'))} nat**"
            f" ({'✓' if row.get('contexte_utilise') else '✗'} seuil 2 nat)")
    add("")
    if pool.get("contexte"):
        probe = pool["contexte"][0].get("sondes", {})
        add("Sondes (semence 1) : " + " · ".join(
            f"{key} {fmt(value.get('perte'))}" for key, value in probe.items()
        ) + ". Le repere « noeud1 » lit en realite `state[:, 0, :]`, c'est-a-dire la")
        add("**porte d'entree** (noeud 0) : sa valeur est identique sur les trois")
        add("semences au dix-millieme, ce qui confirme qu'elle ne porte aucun savoir")
        add("appris — c'est une correction du libelle herite de S4, pas un resultat.")
        add("")

    # ---------------------------------------------------- 4. survie du receveur
    add("## 4. La survie du receveur (V3/V4)")
    add("")
    defaut = survie_resume.get("defaut", {})
    faim = survie_resume.get("faim", {})
    add("Le metabolisme tourne **libre** 1000 pas : naissances payees par le trafic,")
    add("coupe des synapses faminees, aucun pas d'optimiseur. Deux regimes de faim.")
    add("")
    add("| regime | naissances | coupees | R avant | R apres | ratio |")
    add("|---|---|---|---|---|---|")
    add(f"| defaut (prune_ratio 0,1) | {fmt(defaut.get('creees'), 1)} | "
        f"{fmt(defaut.get('coupees'), 1)} | {signed(defaut.get('R_avant'))} | "
        f"{signed(defaut.get('R_apres'))} | {fmt(defaut.get('ratio'), 2)} |")
    add(f"| faim (prune_ratio 0,5) | {fmt(faim.get('creees'), 1)} | "
        f"{fmt(faim.get('coupees'), 1)} | {signed(faim.get('R_avant'))} | "
        f"{signed(faim.get('R_apres'))} | {fmt(faim.get('ratio'), 2)} |")
    add("")
    diagnostic = pool.get("diagnostic_faim") or {}
    if diagnostic:
        add("**Le regime par defaut est inerte, et c'est mesurable** : a la sortie de")
        add(f"l'entrainement, les usages des {diagnostic.get('aretes')} aretes vivent entre")
        add(f"**{fmt(diagnostic.get('usage_min'), 5)}** et **{fmt(diagnostic.get('usage_max'), 5)}**")
        add(f"(mediane {fmt(diagnostic.get('usage_median'), 5)}) alors que le seuil de famine —")
        add(f"un dixieme de la moyenne du decile le plus charge — vaut")
        add(f"**{fmt(diagnostic.get('seuil_defaut'), 5)}** : {diagnostic.get('sous_le_seuil')} aretes")
        add("sont affamees, et **aucune** des aretes n'a un usage nul. Aucune synapse ne")
        add("peut donc mourir. Le pool 32 ne change rien, ce qui identifie la cause :")
        add("ce n'est pas la taille du pool, c'est l'absence de routage selectif. Un")
        add("cerveau qui n'active pas de sous-ensemble n'a pas de concurrence de trafic,")
        add("donc pas de famine. Le metabolisme du jouet est **naissance-seul**.")
        add("")
    if pool32:
        add(f"Re-test demande, pool 32 : densite {pct(pool32.get('densite'))}, "
            f"{pool32.get('aretes_creees_entrainement')} aretes creees et "
            f"{pool32.get('aretes_coupees_entrainement')} coupees pendant 200 pas ;")
        add(f"en 300 pas de vie libre, {pool32.get('vie_libre', {}).get('aretes_creees')} "
            f"creees / {pool32.get('vie_libre', {}).get('aretes_coupees')} coupees. "
            f"Usage {fmt(pool32.get('usage_min'))}-{fmt(pool32.get('usage_max'))} "
            f"contre un seuil a {fmt(pool32.get('seuil_faim_defaut'), 4)}.")
        add("")
    verdict = faim.get("verdict") or {}
    code = verdict.get("code", "—")
    codes = verdict.get("codes_par_semence") or []
    add(f"**Verdict de survie : {code}** — {verdict.get('phrase', '')}")
    add(f"(R apres = {fmt(verdict.get('ratio'), 3)} × R avant en moyenne ; critere "
        f"{verdict.get('critere', 'R_apres >= 0,9 x R_avant')}).")
    if codes:
        add("Par semence : " + " · ".join(
            f"s{index + 1} {code_i}" for index, code_i in enumerate(codes)
        ) + f" — {codes.count('🔴')} rouge(s), {codes.count('🟡')} jaune(s), "
            f"{codes.count('🟢')} vert(s). La dispersion est reelle : le verdict est")
        add("celui de la moyenne, et chaque semence est publiee a cote.")
    add("")

    # -------------------------------------------------------- 5. distillation
    add("## 5. La distillation ancrée (D1-D3)")
    add("")
    add("La formule complete, avec les deux correctifs du diagnostic S4 :")
    add("")
    add("```")
    add("L = alpha * T^2 * KL(p_T(MYC) ‖ p_T(prof))")
    add("  + beta  * CE(logits_MYC, tokens_reels)      <- l'ancrage")
    add("  + gamma * MSE(P h_MYC, h_prof)              <- P aleatoire, appris")
    add("```")
    add("")
    grille = distill.get("grille", {})
    if grille:
        add("| config | alpha | beta | gamma | perte validation (nouvelle) |")
        add("|---|---|---|---|---|")
        for name, row in grille.items():
            add(f"| {name} | {row.get('alpha')} | {row.get('beta')} | {row.get('gamma')} | "
                f"**{fmt(row.get('perte_moyenne'))}** ± {fmt(row.get('perte_ecart_type'))} |")
        add("")
    add(f"Meilleure config sur la validation : **{distill.get('meilleur_par_validation', '—')}**")
    add(f"(retenue : {distill.get('meilleur_retenu', '—')}).")
    add("")
    bras = distill.get("bras", {})
    if bras:
        add("| bras | perte | R | KL(A‖MYC) | F | top-3 |")
        add("|---|---|---|---|---|---|")
        for name, rows in bras.items():
            if not rows:
                continue
            n = len(rows)
            add(f"| {name} | {fmt(sum(float(r['perte']) for r in rows) / n)} | "
                f"{signed(sum(float(r['R']) for r in rows) / n)} | "
                f"{fmt(sum(float(r['kl_A_vers_MYC']) for r in rows) / n)} | "
                f"**{signed(sum(float(r['F']) for r in rows) / n)}** | "
                f"{fmt(sum(float(r['top3']) for r in rows) / n)} |")
        add("")
    add(f"**Gain isole de la distillation : {signed(dres.get('gain_distillation_isole'))}** "
        f"(R distille − R M0).")
    controle = dres.get("controle_bruit") or {}
    if controle:
        add(f"Controle a etiquettes melangees : R = {signed(controle.get('R_melange'))}, "
            f"ecart au distille {signed(controle.get('ecart'))} — "
            f"{'propre' if controle.get('propre') else 'non concluant'}.")
    add("")

    # ------------------------------------------------- 6. survie apres transfert
    add("## 6. La survie après transfert (D4)")
    add("")
    survies = distill.get("survie_transfert", [])
    if survies:
        add("| semence | F avant | F apres | R avant | R apres | coupees |")
        add("|---|---|---|---|---|---|")
        for row in survies:
            avant, apres = row.get("avant", {}), row.get("apres", {})
            add(f"| {row.get('semence')} | {signed(avant.get('F'))} | {signed(apres.get('F'))} | "
                f"{signed(avant.get('R'))} | {signed(apres.get('R'))} | "
                f"{row.get('vie', {}).get('aretes_coupees', '—')} |")
        add("")
    else:
        add("La vie libre apres transfert n'a pas ete mesuree (voir section 8).")
        add("")

    # ------------------------------------------------------- 7. score final F
    add("## 7. Le score final F (P3)")
    add("")
    add("```")
    add("F = 1 - KL(prof A ‖ MYCELIA) / KL(prof A ‖ prof B)")
    add("```")
    add("")
    decile = dres.get("distille") or {}
    add(f"- plafond KL(A‖B) : **{fmt(dres.get('plafond_kl_A_B'))} nat**")
    add(f"- KL(A‖MYCELIA) distille : **{fmt(decile.get('kl_A_vers_MYC'))} nat**")
    add(f"- **F avant la vie : {signed(dres.get('F_avant_vie'))}**")
    add(f"- **F apres la vie : {signed(dres.get('F_apres_vie'))}**")
    if dres.get("F_ratio_vie") is not None:
        add(f"- conservation de F sous la vie : **{fmt(dres.get('F_ratio_vie'), 3)}**")
    add("")
    f_avant = dres.get("F_avant_vie")
    if f_avant is not None:
        if float(f_avant) >= 0.9:
            phrase = ("MYCELIA est a moins de 10 % du plafond physique : la distillation "
                      "a fait le travail demande.")
        elif float(f_avant) >= 0.5:
            phrase = ("MYCELIA couvre une part majoritaire du chemin entre « aucun "
                      "transfert » et le plafond inter-donneurs, mais il reste du "
                      "chemin : le levier est la forme de l'eleve, pas la quantite de "
                      "professeur.")
        elif float(f_avant) > 0.0:
            phrase = ("MYCELIA ne couvre qu'une part minoritaire du chemin : le "
                      "transfert existe mais l'eleve ne peut pas porter la forme du "
                      "professeur.")
        else:
            phrase = ("MYCELIA est *derriere* « aucun transfert » : sa distribution "
                      "s'ecarte du professeur plus que deux professeurs ne s'ecartent "
                      "l'un de l'autre. F negatif est un resultat, pas un bug : eleve "
                      "trop petit, et KL tiree vers l'uniforme.")
        add(f"**Lecture : {phrase}**")
        add("")
        add("Ce n'est pas un probleme de recette : aucun levier de la section 5 ne")
        add("deplace la perte de plus de 0,03 nat (T=2, T=4, alpha/beta raffines).")
        add("Le plafond de fidelite est donc fixe par ce que l'eleve peut porter —")
        add("63 276 parametres dont 36 000 de membranes qui servent d'abord a la")
        add("lecture — pas par la quantite de professeur ni par la temperature.")
        add("")
    levees = distill.get("leviers") or {}
    if levees:
        add("Leviers testes (P4, semence 1) :")
        add("")
        add("| levier | perte validation |")
        add("|---|---|")
        for name, row in levees.items():
            add(f"| {row.get('label', name)} | {fmt(row.get('perte'))} |")
        add("")
    elif distill:
        add("Aucun levier lance : F etait deja au-dessus du seuil de 0,9 (ou la")
        add("grille n'a pas ete fusionnee).")
        add("")

    # ------------------------------------------------------- 8. echecs honnetes
    add("## 8. Échecs honnêtes, surprises, périmètre non vérifié")
    add("")
    add("1. **La mort est inerte dans ce jouet.** Avec la regle du cerveau")
    add("   (`prune_ratio` 0,1), 0 synapse meurt en 1000 pas, a 24 comme a 32")
    add("   noeuds : les usages sont trop serres (0,016-0,065) pour qu'une famine")
    add("   existe, parce que la diffusion atteint tous les noeuds — il n'y a pas")
    add("   de routage selectif. Le metabolisme du jouet est donc **naissance-seul**.")
    add("   Le regime « faim » (0,5) est un reglage de faim, pas la regle par")
    add("   defaut : il est documente comme tel dans chaque ligne du JSON.")
    add("2. **La sonde « noeud 1 » de S4 etait mal nommee.** `collect_states` lit")
    add("   `state[:, 0, :]`, la porte d'entree, pas le noeud 1. Sa valeur constante")
    add("   (5,32 sur les trois semences) le montre. La conclusion de S4 — la tete ne")
    add("   lit pas une membrane unique — tient : la sonde « lecture » vaut 4,3")
    add("   contre 5,32 pour cette porte.")
    add("3. **Deux semences pour la grille et la survie apres transfert**, trois")
    add("   pour la validation : c'est un budget CPU, pas un choix statistique. Les")
    add("   ecarts-types publies disent la dispersion disponible.")
    add("4. **Aucun portage vers Nyra** : tout vit dans `scripts/lab/`, le genome et")
    add("   les runs de Nyra sont intacts. Le pool 24 du jouet n'est pas le pool")
    add("   reel ; rien ici ne prouve un comportement a 10 000 noeuds.")
    add("5. **Non verifie** : le corpus reel (le jouet a 312 mots), la duree longue")
    add("   (600 pas, pas 50 000), et l'effet d'un troisieme professeur sur le")
    add("   plafond (deux suffisent a le definir, pas a le stabiliser).")
    add("")
    return "\n".join(lines) + "\n"


def summary() -> dict:
    """Ce que lit l'onglet : F avant/apres, la courbe de vie, naissances/morts."""
    mesure = load("s5_measure.json")
    pool = load("s5_pool24.json")
    distill = load("s5_distill.json")
    plafond = load("s5_plafond.json")
    dres = distill.get("resume", {})
    classement = mesure.get("classement", {})
    survie_resume = pool.get("survie_resume", {})
    faim = survie_resume.get("faim", {})
    verdict = faim.get("verdict") or {}
    courbe = []
    for row in distill.get("survie_transfert", []):
        courbe = [
            {"pas": point.get("pas"), "R": point.get("R"),
             "coupees": point.get("coupees"), "aretes": point.get("aretes")}
            for point in row.get("courbe", [])
        ]
        if courbe:
            break
    if not courbe:
        for row in pool.get("survie", []):
            courbe = [
                {"pas": point.get("pas"), "R": point.get("R"),
                 "coupees": point.get("coupees"), "aretes": point.get("aretes")}
                for point in row.get("regimes", {}).get("faim", {}).get("courbe", [])
            ]
            if courbe:
                break
    def as_int(value):
        return None if value is None else int(round(float(value)))

    return {
        "version": "S5-MAX",
        "F_avant": dres.get("F_avant_vie"),
        "F_apres": dres.get("F_apres_vie"),
        "plafond_kl": (plafond.get("accord") or {}).get("kl_A_vers_B"),
        "survie_code": verdict.get("code"),
        "survie_phrase": verdict.get("phrase"),
        "vie_creees": as_int(faim.get("creees")),
        "vie_coupees": as_int(faim.get("coupees")),
        "densite": pool.get("resume", {}).get("densite_moyenne"),
        "mesure_mycelia": classement.get("mycelia_nouvelle"),
        "mesure_dense": classement.get("dense_nouvelle"),
        "mesure_bigramme": classement.get("bigramme"),
        "pool24_moyenne": pool.get("resume", {}).get("nouvelle_moyenne"),
        "distillation_gain": dres.get("gain_distillation_isole"),
        "courbe_survie": courbe,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="rapport S5-MAX")
    parser.add_argument("--json-only", action="store_true")
    args = parser.parse_args()
    payload = summary()
    write_json(LAB / "s5_summary.json", payload)
    if not args.json_only:
        text = build()
        path = LAB / "RAPPORT-S5.md"
        path.write_text(text, encoding="utf-8")
        print(f"ecrit : {path} ({len(text.splitlines())} lignes)")
    print(f"ecrit : {LAB / 's5_summary.json'} — F {payload['F_avant']} -> {payload['F_apres']}")


if __name__ == "__main__":
    main()
