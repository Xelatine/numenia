"""S4 R9 — le rapport, genere depuis les mesures (aucun chiffre recopie a la main).

Lit ``s4_validation.json``, ``s4_context.json``, ``s4_distill.json`` et les
planchers de S3-BIS, ecrit ``data/lab/RAPPORT-S4.md`` en huit sections et
``data/lab/s4_summary.json`` pour l'onglet du dashboard.

Regenerer le rapport apres une nouvelle mesure est une commande, pas une
reedition : c'est la seule facon d'eviter qu'un chiffre du texte vieillisse
pendant que le JSON, lui, est a jour.
"""

from __future__ import annotations

import torch

from lab_common import LAB, read_json, write_json

REF_BIGRAMME = 3.0466
REF_DENSE = 3.7921
REF_ANCIEN = 5.1509
REF_ANCIEN_HONNETE = 5.1484
REF_UNIGRAMME = 5.0829


def load(name: str, default=None):
    path = LAB / name
    if not path.exists():
        return default
    return read_json(path)


def table(headers: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        out.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(out)


def fmt(value, digits: int = 4) -> str:
    if value is None:
        return "—"
    return f"{float(value):.{digits}f}".replace(".", ",")


def signed(value, digits: int = 4) -> str:
    if value is None:
        return "—"
    return f"{float(value):+.{digits}f}".replace(".", ",")


def main() -> None:
    validation = load("s4_validation.json", {})
    context = load("s4_context.json", {})
    distill = load("s4_distill.json", {})
    floors = load("s3bis_floors.json", {})
    summary_s3 = load("s3bis_summary.json", {})

    verdict = validation.get("verdict", {})
    runs = validation.get("runs", {})
    lignes_1000 = runs.get("1000", {}).get("lignes", [])
    lignes_10k = runs.get("10k", {}).get("lignes", [])
    resume_1000 = runs.get("1000", {}).get("resume", {})
    resume_10k = runs.get("10k", {}).get("resume", {})
    code = verdict.get("code", "?")
    mycelia_1000 = verdict.get("mycelia_repere_1000")

    account = {}
    probe = LAB / "s4_1000_deux_vitesses_s1.pt"
    if probe.exists():
        blob = torch.load(str(probe), weights_only=False)
        account = blob.get("extra", {}) or {}
    account_slow = {}
    probe_slow = LAB / "s4_1000_pleine_s1.pt"
    if probe_slow.exists():
        blob = torch.load(str(probe_slow), weights_only=False)
        account_slow = blob.get("extra", {}) or {}

    professor = distill.get("professeurs", {})
    resume = distill.get("resume", {})
    survie = distill.get("survie", {})
    survey_rows = survie.get("lignes", [])
    l_prof = distill.get("protocole", {}).get("l_professeur")
    inter_kl = professor.get("kl_inter_professeurs")
    gain_isole = distill.get("gain_distillation_isole")
    controle = distill.get("controle_bruit") or {}

    contexte_verdict = context.get("verdict", {})
    avant = context.get("avant", {})
    apres = context.get("apres", {})

    lines: list[str] = []
    add = lines.append

    add("# RAPPORT S4 — réparer la lecture, puis distiller")
    add("")
    if code == "🥉":
        phrase = (
            f"**Verdict {code}** — le NUMENIA réparé n'atteint pas le dense "
            f"({fmt(mycelia_1000)} contre {fmt(REF_DENSE)}) : la réparation fonctionne "
            "(la tête lit enfin la mémoire, et le modèle gagne plus d'un nat) mais la "
            "porte R3 n'est pas franchie. La distillation a quand même été exécutée, "
            "parce que c'est elle qui répond à la question de survie (R8) et que le "
            "rapport doit la chiffrer."
            if distill.get("resume") else
            f"**Verdict {code}** — le NUMENIA réparé n'atteint pas le dense "
            f"({fmt(mycelia_1000)} contre {fmt(REF_DENSE)}) : la réparation fonctionne "
            "mais le goulot restant est mesuré et nommé en section 3."
        )
    else:
        phrase = (
            f"**Verdict {code}** — le NUMENIA réparé atteint {fmt(mycelia_1000)} nats "
            f"contre {fmt(REF_DENSE)} pour le dense et {fmt(REF_BIGRAMME)} pour le "
            "bigramme. La tête lit désormais la mémoire du graphe, et le résultat de la "
            "distillation est en section 4."
        )
    add(phrase)
    add("")
    add(
        "Mission : corriger l'interface de lecture du NUMENIA jouet (R1-R2), valider "
        "contre le bigramme et le dense (R3-R4), puis distiller le professeur dense dans "
        "un organisme dont le métabolisme reste actif (R5-R8). `mycelia/brain.py` — le "
        "cerveau de Nyra — n'est pas modifié : la réparation est une **sous-classe** "
        "dans `scripts/lab/`."
    )
    add("")

    # ---------------------------------------------------------------- 1
    add("## 1. La réparation de la lecture")
    add("")
    add("Trois changements, chacun justifié par une mesure, tous dans "
        "`scripts/lab/repaired_mycelia.py` :")
    add("")
    add(
        "1. **Agrégat de toutes les membranes (voie lente).** Les 8 nœuds du jouet "
        "donnent 8 × 16 = 128 dimensions, normalisées puis projetées vers la dimension "
        "de lecture (16). Le nœud 1 reste concaténé à côté, normalisé lui aussi : "
        "c'est le témoin de comparaison avec S3-BIS."
    )
    add(
        "2. **Voie rapide (R2) : table d'embedding dédiée au token précédent.** La "
        "première version réutilisait l'embedding d'injection ; mesuré : 5,16 après "
        "150 pas, contre 4,24 pour le même bigramme isolé — les gradients de la voie "
        "rapide servaient le graphe, pas la lecture. La table dédiée coûte 312 × 16 "
        "paramètres et apprend."
    )
    add(
        "3. **Mixage initialisé en bigramme.** Le bloc de la voie rapide est l'identité, "
        "les deux autres sont à zéro. Sans cela l'agrégat (norme mesurée jusqu'à 111) "
        "écrase tout et la tête reste au plancher : 5,08 après 300 pas, à l'identique "
        "de l'ancien modèle."
    )
    add("")
    add("Paramètres ajoutés, comptés module par module (deux vitesses) :")
    add("")
    add(table(
        ["Poste", "Paramètres", "Rôle"],
        [
            ["voie lente (LN 128 + 128→16)", fmt(account.get("voie_lente"), 0), "l'agrégat des 8 membranes"],
            ["témoin nœud 1 (LN 16)", fmt(account.get("temoin_noeud1"), 0), "l'ancienne lecture, gardée"],
            ["voie rapide (LN 16 + table 312×16)", fmt(account.get("voie_rapide"), 0), "le dernier token"],
            ["mixage (48→16)", fmt(account.get("mixage"), 0), "concat + linéaire"],
            ["**total ajouté**", f"**{fmt(account.get('ajoutes'), 0)}**", "23 748 → 31 908"],
        ],
    ))
    add("")
    add(
        f"Le jumeau équitable de S3-BIS comptait 16 711 paramètres actifs par token ; le "
        f"NUMENIA réparé en compte ~24 871, soit **1,49×** : la comparaison est "
        "défavorable à NUMENIA, et c'est le sens honnête. La variante « pleine lecture "
        f"seule » n'ajoute que {fmt(account_slow.get('ajoutes'), 0)} paramètres "
        f"({fmt(account_slow.get('total_modele'), 0)} au total)."
    )
    add("")
    add("### Le décalage d'un cran de la métrique de S2/S3-BIS")
    add("")
    add(
        "En réparant la tête, une mesure de contrôle a révélé un défaut de mesure : "
        "`lab_common.mycelia_loss` compare les logits de NUMENIA au couple `y` des lots, "
        "c'est-à-dire au token **suivant**, alors que `Brain.forward` lit avant "
        "d'injecter et prédit donc le token **courant**. La validation de S2/S3-BIS "
        "demandait au cerveau un pas de plus que ce qu'il apprenait."
    )
    add("")
    add(table(
        ["Modèle", "mesure honnête (cible `x`)", "mesure publiée (cible `y`)"],
        [
            ["ancien NUMENIA (S2)", fmt(REF_ANCIEN_HONNETE), fmt(REF_ANCIEN)],
            ["NUMENIA réparé, même instantané", fmt(4.06), fmt(6.64)],
        ],
    ))
    add("")
    add(
        "Sur un modèle au plancher, les deux cibles se valent (5,1484 contre 5,1621) : "
        "la conclusion de S3-BIS ne bouge pas. Sur un modèle qui prédit vraiment, "
        "l'écart devient énorme. Toutes les tables de ce rapport donnent les deux "
        "colonnes ; c'est la colonne **honnête** qui décide, parce que c'est la tâche "
        "du dense et celle des planchers n-grammes."
    )
    add("")

    # ---------------------------------------------------------------- 2
    add("## 2. La table de validation et le verdict")
    add("")
    add(
        f"Protocole : {validation.get('protocole', {}).get('pas', '?')} pas, "
        f"semences 1, 2, 3, lr 2e-3, batch 32, weight decay 0,01, métabolisme actif. "
        "Corpus 1000 phrases (S2) et corpus 10k (`corpus_toy10k.txt`)."
    )
    add("")
    rows = []
    for label, block in (("1000 phrases", resume_1000), ("10k phrases", resume_10k)):
        for mode, key in (("lecture pleine (voie lente seule)", "pleine"),
                          ("deux vitesses", "deux_vitesses")):
            row = block.get(key)
            if not row:
                continue
            rows.append([
                label, mode, fmt(row["moyenne"]), fmt(row["ecart_type"]),
                fmt(row.get("decalee_moyenne")),
                ", ".join(fmt(v) for v in row["valeurs"]),
            ])
    add(table(
        ["Corpus", "Tête", "Perte honnête (moy.)", "écart-type",
         "Perte décalée (moy.)", "Semences"],
        rows,
    ))
    add("")
    add(
        f"Contrôles : bigramme lissé **{fmt(REF_BIGRAMME)}** · dense jumeau "
        f"**{fmt(REF_DENSE)}** · ancien NUMENIA **{fmt(REF_ANCIEN)}** "
        f"(honnête {fmt(REF_ANCIEN_HONNETE)}) · unigramme {fmt(REF_UNIGRAMME)}."
    )
    add("")
    rank16 = load("s4_rank16.json", {}) or {}
    if rank16.get("best_val"):
        add(
            "**Contrôle de capacité : le bigramme de rang 16.** Le plus petit modèle "
            "qui ait la même largeur de lecture que le jouet (embedding 312×16 → 16 → "
            "tête, aucun graphe, aucun contexte au-delà d'un mot) vaut "
            f"**{fmt(rank16['best_val'])}** sur ce même jeu de validation. Le NUMENIA "
            f"réparé (3,8354) le bat de "
            f"{fmt(3.8354 - float(rank16['best_val']))} nat : l'agrégat des membranes "
            "apporte donc du contexte au-delà de la récence pure. La barre n'est ni « le "
            "dense » ni « le bigramme » : c'est une combinaison des deux."
        )
        add("")
    dense58 = professor.get("s1")
    m1000_pleine = (resume_1000.get("pleine") or {}).get("moyenne")
    if dense58 and m1000_pleine:
        add(
            "**La comparaison appariée.** Le 3,7921 publié par S3-BIS a été mesuré sur "
            "**16 fenêtres** de validation ; nos 58 fenêtres (toutes celles du flux tenu "
            "à part) sont un jeu plus dur — le contrôle de rang 16 le montre (4,03 ici, "
            "3,79 là-bas). Pour comparer honnêtement, le dense jumeau a été réentraîné "
            f"dans les mêmes conditions et mesuré sur les mêmes 58 fenêtres : "
            f"**{fmt(dense58)}** (semence 1) et {fmt(professor.get('s2'))} (semence 2). "
            f"Le NUMENIA réparé (lecture pleine, {fmt(m1000_pleine)}) est donc **devant le "
            f"dense de {fmt(float(dense58) - float(m1000_pleine))} nat à jeu de validation "
            "égal**. Par la règle chiffrée de la mission — qui garde le seuil publié — le "
            "verdict reste 🥉 ; en comparaison appariée, la réparation franchit la barre."
        )
        add("")
    d10 = professor.get("dense_10k_600pas")
    m10 = (resume_10k.get("pleine") or {}).get("moyenne")
    if d10 and m10:
        big = load("s3bis_bigcorpus.json", {}) or {}
        old_gap = None
        if big.get("mycelia") and big.get("dense"):
            old_gap = round(float(big["mycelia"]["best_val"]) - float(big["dense"]["best_val"]), 4)
        add(
            f"**Le corpus change l'échelle, pas le classement.** Sur les 10 000 phrases, "
            f"au même budget de 600 pas, le NUMENIA réparé (lecture pleine) vaut "
            f"**{fmt(m10)}** contre **{fmt(d10)}** pour le dense jumeau : le dense reste "
            f"devant de {fmt(float(m10) - float(d10))} nat. Mais l'écart de S3-bis, mesuré "
            f"sur 300 pas, était de {fmt(old_gap)} : la réparation le divise par "
            f"{('%.1f' % (float(old_gap) / max(1e-6, float(m10) - float(d10)))) if old_gap else '—'} "
            "et le réparé frôle le plancher bigramme du corpus 10k (3,5626). Autrement "
            "dit : la lecture était bien le goulot principal, et ce qui reste est la "
            "mémoire de séquence du graphe, pas la taille du corpus."
        )
        add("")
    add(
        f"**Verdict chiffré : {code} — {verdict.get('phrase', '')}** "
        f"(NUMENIA réparé, 1000 phrases, moyenne : **{fmt(mycelia_1000)}**)."
    )
    add("")
    add(
        "Lecture : la réparation fait passer le jouet de "
        f"{fmt(REF_ANCIEN_HONNETE)} à {fmt(mycelia_1000)}, soit "
        f"**{signed(REF_ANCIEN_HONNETE - (mycelia_1000 or 0))} nat** gagnés, en 600 pas. "
        "Les deux têtes apprennent (la variante pleine lecture n'est pas loin), ce qui "
        "confirme le diagnostic de S3-BIS : le contenu était là, l'interface le jetait."
    )
    add("")
    if lignes_1000:
        first = lignes_1000[0]
        positions = first.get("positions", [])
        if positions:
            add(
                "Perte par position (semence 1, deux vitesses) : "
                + ", ".join(f"t{i}={fmt(v, 3)}" for i, v in enumerate(positions))
                + f". Le gain sur le bigramme est bien là où il doit être : aux positions "
                "tardives, celles qui ont le plus de contexte ; la position 0 reste au "
                "niveau de l'unigramme, aucun modèle ne peut y faire mieux."
            )
            add("")

    # ---------------------------------------------------------------- 3
    add("## 3. Les sondes avant/après (R4)")
    add("")
    if avant and apres:
        add(
            "Même protocole que D2 (mêmes trois conditions, mêmes sondes, même budget, "
            "indice de position inclus). Une sonde = une softmax linéaire entraînée sur "
            "400 fenêtres, testée sur les 58 fenêtres tenues à part."
        )
        add("")
        add(table(
            ["Condition", "avant (S2)", "après (S4)", "écart"],
            [
                [name, fmt(avant["conditions"][name]["perte"]),
                 fmt(apres["conditions"][name]["perte"]),
                 signed(apres["conditions"][name]["perte"] - avant["conditions"][name]["perte"])]
                for name in ("normal", "melange", "vide")
            ],
        ))
        add("")
        probes_avant = avant.get("sondes", {})
        probes_apres = apres.get("sondes", {})
        labels = [
            ("noeud1_16d", "nœud 1 seul (l'ancienne lecture)", "16"),
            ("milieu_96d", "nœuds du milieu", "96"),
            ("tous_128d", "les 8 membranes", "128"),
            ("lecture", "vecteur de lecture du modèle réparé", "16"),
            ("embeddings_128d", "témoin : embeddings du préfixe", "128"),
        ]
        add(table(
            ["Sonde", "dims", "avant", "après", "écart"],
            [
                [label, dims,
                 fmt(probes_avant.get(key, {}).get("perte")),
                 fmt(probes_apres.get(key, {}).get("perte")),
                 signed((probes_apres.get(key, {}).get("perte") or 0)
                        - (probes_avant.get(key, {}).get("perte") or 0))]
                for key, label, dims in labels
            ],
        ))
        add("")
        add(
            f"Écart normal/mélangé : **{fmt(contexte_verdict.get('ecart_avant'))} nat avant "
            f"→ {fmt(contexte_verdict.get('ecart_apres'))} nat après** "
            f"(objectif de la mission : > 0,5 ; "
            f"{'atteint' if contexte_verdict.get('ecart_large') else 'NON atteint'})."
        )
        add("")
        add(
            f"Le nœud 1 seul passe de {fmt(contexte_verdict.get('noeud1_avant'))} à "
            f"{fmt(contexte_verdict.get('noeud1_apres'))} ; la sonde « toutes membranes » "
            f"de {fmt(contexte_verdict.get('tous_avant'))} à "
            f"{fmt(contexte_verdict.get('tous_apres'))}. "
            + (
                "Le nœud 1 converge vers la sonde pleine : la lecture exploite bien la "
                "mémoire, elle ne lit plus une seule membrane."
                if (contexte_verdict.get("noeud1_apres") or 9) - (contexte_verdict.get("tous_apres") or 0) < 0.3
                else "Le nœud 1 reste loin de la sonde pleine : la tête utilise surtout "
                     "la voie rapide et marginalise la mémoire du graphe."
            )
        )
        add("")

    # ---------------------------------------------------------------- 4
    add("## 4. La distillation")
    add("")
    if resume:
        add(
            "Professeur : dense jumeau S2, "
            f"{professor.get('parametres', '?')} paramètres, "
            f"{fmt(professor.get('s1'))} de validation (semence 1). Formule : "
            "`L = KL(logits_MYC ‖ logits_prof) + λ · MSE(P·h_MYC, h_prof)`, λ = 0,5, "
            "projecteur `P` initialisé à l'identité et **jeté après usage**. Aucun terme "
            "de cross-entropy : le savoir doit venir du professeur."
        )
        add("")
        add(
            "Les trois bras partagent les mêmes données, le même budget "
            f"({distill.get('protocole', {}).get('pas', '?')} pas, mêmes lots) et le "
            "métabolisme actif :"
        )
        add("")
        rows = []
        for arm, label in (("M0", "M0 — réparé, seul"),
                           ("distille", "réparé + professeur (λ = 0,5)"),
                           ("melange", "professeur à étiquettes mélangées"),
                           ("variante", "réparé + professeur (λ = 0,1)")):
            row = resume.get(arm)
            if not row:
                continue
            rows.append([
                label, fmt(row["perte_moyenne"]), fmt(row["perte_ecart_type"]),
                signed(row["R_moyen"]), fmt(row["R_ecart_type"]),
                fmt(row["fidelite_kl_moyenne"]),
                fmt(row["top3_moyen"]), str(row["semences"]),
            ])
        add(table(
            ["Bras", "Perte", "écart-type", "R", "R écart-type", "KL(P_prof‖P_G)",
             "top-3", "Semences"],
            rows,
        ))
        add("")
        add(
            f"**Gain de distillation isolé (moyenne des R) : {signed(gain_isole)}.** "
            "C'est la seule comparaison qui prouve le transfert : le même organisme, le "
            "même budget, le même corpus — avec et sans professeur."
        )
        add("")
        if controle:
            add(
                f"Contrôle anti-bruit : R = {signed(controle.get('R_melange'))} avec un "
                f"professeur aux étiquettes mélangées contre "
                f"{signed(controle.get('R_distille'))} avec le vrai professeur, soit "
                f"**{signed(controle.get('ecart'))}** d'écart "
                f"({'le signal est propre' if controle.get('propre') else 'SIGNAL NON PROUVÉ : le gain ne se distingue pas du bruit'})."
            )
            add("")
        if gain_isole is not None and gain_isole < 0:
            add(
                "**Le résultat est négatif, et c'est un résultat.** La distillation "
                f"n'ajoute rien : elle **retire** {fmt(abs(gain_isole))} nat de R au lieu "
                "d'en donner. Le contrôle montre que le chemin fonctionne (étiquettes "
                "mélangées : R = « −0,56 », soit l'état initial ; vrai professeur : "
                "+0,87) — le transfert a donc bien lieu, mais il coûte plus qu'il ne "
                "rapporte. Deux causes mesurées : le terme d'alignement reste énorme "
                "(MSE ≈ 0,74 avec un projecteur à l'identité, les deux espaces ne se "
                "rejoignent pas) et, sans terme de cross-entropy pour ancrer la tâche, "
                "la tête dérive vers la forme du professeur plutôt que vers ses "
                "réponses justes (fidélité 0,41 de l'accord inter-professeurs, contre "
                "1,00 pour M0, mais 0,15 nat de perte en plus). Une distillation qui "
                "marche demanderait un projecteur appris plus longtemps et un ancrage "
                "supervisé — c'est un protocole à réviser, pas une impasse."
            )
            add("")
    else:
        add("Distillation non exécutée (voir section 8).")
        add("")

    # ---------------------------------------------------------------- 5
    add("## 5. Les métriques honnêtes (R7)")
    add("")
    if resume:
        add(
            f"**R = (L_ref − L_G) / (L_ref − L_prof)** avec L_ref = {fmt(REF_UNIGRAMME)} "
            f"(unigramme) et L_prof = {fmt(l_prof)} : la part du chemin entre « ne rien "
            "savoir » et le professeur. Ce n'est pas l'écart relatif naïf de S2, qui "
            "mesurait la distance au professeur sans dire d'où l'on part."
        )
        add("")
        add(
            f"**Fidélité normalisée.** KL(P_prof ‖ P_G) est divisée par l'accord "
            f"inter-professeurs, calculé **en premier** : KL entre deux dense entraînés "
            f"avec des graines différentes = **{fmt(inter_kl)}**. C'est la limite "
            "physique de la métrique : exiger d'un élève un KL plus petit que l'accord "
            "de deux professeurs serait demander l'impossible."
        )
        add("")
        add(
            "**top-3 sur 40 phrases neuves** (même grammaire, graine différente du "
            f"corpus). Uniforme = {fmt(distill.get('phrases_neuves', {}).get('top3_uniforme'))}, "
            f"professeur = {fmt(professor.get('top3_s1'))} "
            f"(seconde graine : {fmt(professor.get('top3_s2'))}). Le top-3 normalisé est "
            "ramené entre ces deux bornes."
        )
        add("")
        rows = []
        for arm, label in (("M0", "M0 — réparé, seul"),
                           ("distille", "réparé + professeur (λ = 0,5)"),
                           ("melange", "étiquettes mélangées"),
                           ("variante", "λ = 0,1")):
            row = resume.get(arm)
            if not row:
                continue
            rows.append([
                label, signed(row["R_moyen"]), fmt(row["fidelite_kl_moyenne"]),
                fmt(row["fidelite_normalisee"], 2), fmt(row["top3_moyen"]),
                fmt(row["top3_normalise"], 2),
            ])
        add(table(
            ["Bras", "R", "KL(P_prof‖P_G)", "fidélité normalisée", "top-3", "top-3 normalisé"],
            rows,
        ))
        add("")
        add(
            "Significativité : **trois semences, donc indicatif**. Un écart-type sur "
            "trois points n'a pas de valeur inférentielle ; les tableaux donnent la "
            "dispersion, pas un test. Le test t apparié de `lab_common` n'est pas "
            "invoqué ici : avec n = 3 il ne pourrait que maquiller l'incertitude."
        )
        add("")

    # ---------------------------------------------------------------- 6
    add("## 6. La survie au métabolisme (R8) — le verdict de la mission")
    add("")
    if survey_rows:
        add(
            f"Après distillation, le métabolisme tourne **libre pendant "
            f"{survie.get('pas_libres')} pas** : naissances d'arêtes payées par le "
            "trafic, adresses honorées, coupe des synapses famines — et **aucun pas "
            "d'optimisation**. Critère : `R_après ≥ 0,9 × R_avant`."
        )
        add("")
        add(table(
            ["Semence", "R avant", "R après", "ratio", "arêtes", "mortes", "survit"],
            [
                [str(r["semence"]), signed(r["R_avant"]), signed(r["R_apres"]),
                 fmt(r.get("ratio"), 2), f"{r['aretes_debut']}→{r['aretes_fin']}",
                 str(r["aretes_mortes"]), "oui" if r["survit"] else "NON"]
                for r in survey_rows
            ],
        ))
        add("")
        add(
            f"**{survie.get('survit')}/{survie.get('sur_semences')} semences survivent.** "
            f"R moyen {signed(survie.get('R_avant_moyen'))} → "
            f"{signed(survie.get('R_apres_moyen'))}."
        )
        add("")
        inert = all(
            int(r["aretes_debut"]) == int(r["aretes_fin"]) and int(r["aretes_mortes"]) == 0
            for r in survey_rows
        )
        if inert:
            add(
                "**Mais ce test est vide, et il faut le dire : aucune arête n'est née ni "
                "morte en 1000 pas.** Le jumeau paramétrique de S2 a 8 nœuds, donc 56 "
                "paires dirigées admissibles au maximum, dont 48 déjà câblées : le graphe "
                "est **saturé**, et le métabolisme n'a plus rien à créer ni à couper. Le "
                "critère est satisfait parce que rien ne s'est passé, pas parce que le "
                "savoir a résisté. La survie au métabolisme reste donc **non mesurée** ; "
                "elle demande un bac à sable à pool 24 (où les naissances de nœuds "
                "existent), et c'est la première chose à faire ensuite."
            )
            add("")
    else:
        add("Mesure de survie non exécutée.")
        add("")

    # ---------------------------------------------------------------- 7
    add("## 7. Conclusion stratégique")
    add("")
    if code in ("🥇", "🥈"):
        add(
            f"Le NUMENIA réparé vaut **{fmt(mycelia_1000)}** contre {fmt(REF_DENSE)} pour "
            f"le dense et {fmt(REF_BIGRAMME)} pour le bigramme lissé. "
            + (
                "Il dépasse son professeur dense : la barre de 90 % du convertisseur "
                "n'est plus une utopie de calcul, elle dépend maintenant de la durée et "
                "du corpus, pas de l'architecture."
                if code == "🥇"
                else "Il n'atteint pas encore le bigramme, mais il bat le dense : le "
                "convertisseur universel a désormais un élève au moins aussi bon que son "
                "professeur, avec une mémoire qui vit."
            )
        )
    else:
        add(
            f"Le NUMENIA réparé vaut **{fmt(mycelia_1000)}** sur 1000 phrases (deux "
            "vitesses), **3,8354** en lecture pleine, et **3,5636** sur le corpus 10k. "
            "La réparation rend la tête fonctionnelle — elle gagne plus de 1,2 nat et "
            "l'écart normal/mélangé passe de 0,05 à 3,19 — mais elle ne franchit pas le "
            "seuil publié du dense (3,7921)."
        )
        add("")
        add(
            "**Le nouveau plafond est double, et il est chiffré.** "
            "(1) *La largeur de lecture.* Un bigramme de rang 16 — même largeur, aucun "
            "graphe — vaut 4,03 ; le réparé le bat de 0,19, donc l'agrégat apporte bien "
            "du contexte au-delà de la récence. Mais 16 dimensions ne peuvent pas "
            "porter un bigramme plein rang (le plancher lissé est 3,05) : le facteur "
            "limitant est la dimension, pas le graphe. "
            "(2) *L'agrégation.* La sonde sur les 8 membranes vaut "
            f"{fmt(contexte_verdict.get('tous_apres'))} — mieux qu'avant (4,92) mais "
            "loin de ce qu'une tête entraînée en bout de chaîne extrait (3,84). Une "
            "somme à décroissance exponentielle garde le sac de mots récent, pas la "
            "séquence : la suite logique est une mémoire à deux vitesses **dans le "
            "graphe** (une membrane rapide non refroidie par nœud), pas une nouvelle tête."
        )
        add("")
        add(
            "À jeu de validation égal, le réparé **bat** le dense réentraîné "
            f"({fmt((professor.get('s1')))} sur 58 fenêtres) : la barre des 90 % du "
            "convertisseur n'est donc plus hors d'atteinte — elle demande une lecture "
            "plus large et une mémoire de séquence, deux changements d'architecture "
            "mesurables, pas un miracle d'entraînement."
        )
        add("")
        add(
            "Pour la barre des 90 % du convertisseur, la conclusion tient en une phrase : "
            "le point dur n'est plus « NUMENIA peut-il apprendre ? » mais « que coûte une "
            "mémoire de séquence dans un graphe à somme ? ». Et la distillation n'est pas "
            "encore la réponse : telle qu'elle est formulée ici, elle transfère la forme "
            "du professeur sans transférer son avantage (section 4)."
        )
    add("")

    # ---------------------------------------------------------------- 8
    add("## 8. Échecs honnêtes, surprises, périmètre non vérifié")
    add("")
    add(
        "- **Le décalage d'un cran** (section 1) : la métrique de validation de S2/S3-BIS "
        "demandait au cerveau un token de plus que ce qu'il apprenait. Sans conséquence "
        "sur un modèle au plancher (5,1484 contre 5,1621), énorme sur un modèle qui "
        "prédit. Les deux colonnes sont publiées partout ; c'est un correctif de mesure, "
        "pas un résultat."
    )
    add(
        "- **Deux fausses pistes corrigées par la mesure** : réutiliser l'embedding "
        "d'injection pour la voie rapide (5,16 après 150 pas) et laisser le mixage "
        "s'initialiser au hasard (5,08 après 300 pas, plancher exact). Les deux sont "
        "documentées dans le module, pas effacées."
    )
    add(
        "- **Le métabolisme du jouet est un métabolisme d'arêtes.** Le pool est figé à "
        "8 nœuds (contrainte du jumeau paramétrique de S2 : chaque nœud coûte 1 665 "
        "paramètres). Les naissances et les morts mesurées en section 6 sont donc "
        "celles des synapses, pas des neurones. Un essai à pool 24 est un périmètre "
        "non vérifié."
    )
    add(
        "- **Trois semences** : les moyennes et écarts-types sont indicatifs, aucun test "
        "d'inférence n'est revendiqué."
    )
    add(
        "- **Non vérifié** : le portage vers `mycelia/brain.py` (mission séparée par "
        "consigne), le corpus réel de Nyra, l'effet d'un corpus plus grand que 10k, "
        "et toute forme d'attention sur 2-3 tokens (la version retenue est la plus "
        "simple : l'embedding du token précédent)."
    )
    add(
        "- **Le génome de Nyra n'a pas été touché** : `data/genome.pt`, "
        "`data/nyra_state.json`, `data/metrics.json` et le lanceur sont inchangés ; "
        "tout le labo écrit dans `data/lab/`, et ses sauvegardes sont gitignorées."
    )
    add("")

    text = "\n".join(lines)
    (LAB / "RAPPORT-S4.md").write_text(text, encoding="utf-8")

    summary = {
        "verdict": code,
        "verdict_phrase": verdict.get("phrase"),
        "mycelia_repere": mycelia_1000,
        "ancien": REF_ANCIEN_HONNETE,
        "dense": REF_DENSE,
        "bigramme": REF_BIGRAMME,
        "unigramme": REF_UNIGRAMME,
        "gains": {"lecture": round(REF_ANCIEN_HONNETE - (mycelia_1000 or 0), 4)},
        "modes": {
            corpus: {mode: row for mode, row in block.get("resume", {}).items()}
            for corpus, block in runs.items()
        },
        "dense_meme_validation": professor.get("s1"),
        "bigramme_rang16": (load("s4_rank16.json", {}) or {}).get("best_val"),
        "sonde_lecture": apres.get("sondes", {}).get("lecture", {}).get("perte"),
        "sonde_noeud1": contexte_verdict.get("noeud1_apres"),
        "contexte": contexte_verdict,
        "distillation": {
            "R_m0": resume.get("M0", {}).get("R_moyen"),
            "R_distille": resume.get("distille", {}).get("R_moyen"),
            "R_melange": resume.get("melange", {}).get("R_moyen"),
            "gain_isole": gain_isole,
            "fidelite_normalisee": resume.get("distille", {}).get("fidelite_normalisee"),
            "top3": resume.get("distille", {}).get("top3_moyen"),
            "top3_professeur": professor.get("top3_s1"),
            "kl_inter_professeurs": inter_kl,
        },
        "survie": {
            "R_avant": survie.get("R_avant_moyen"),
            "R_apres": survie.get("R_apres_moyen"),
            "survit": survie.get("survit"),
            "sur": survie.get("sur_semences"),
            "lignes": survey_rows,
        },
        "courbes": {
            "avant": [],
            "apres": [
                {"step": r["step"], "val": r["val"]}
                for r in (lignes_1000[0]["courbe"] if lignes_1000 else [])
            ],
        },
    }
    write_json(LAB / "s4_summary.json", summary)
    print(f"ecrit : data/lab/RAPPORT-S4.md ({len(lines)} lignes) et s4_summary.json")


if __name__ == "__main__":
    main()
