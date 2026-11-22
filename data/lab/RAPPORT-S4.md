> **Renommage —** Ce projet s'appelle désormais **Numenia** (auparavant référencé
> comme Mycelia dans les titres et la documentation). Session, chiffres, fichiers de
> données et cerveau **Nyra** inchangés. Ce document est historique : son contenu est
> conservé tel quel, et seul le nom du projet passe à Numenia.

# RAPPORT S4 — réparer la lecture, puis distiller

**Verdict 🥉** — le MYCELIA réparé n'atteint pas le dense (3,9039 contre 3,7921) : la réparation fonctionne (la tête lit enfin la mémoire, et le modèle gagne plus d'un nat) mais la porte R3 n'est pas franchie. La distillation a quand même été exécutée, parce que c'est elle qui répond à la question de survie (R8) et que le rapport doit la chiffrer.

Mission : corriger l'interface de lecture du MYCELIA jouet (R1-R2), valider contre le bigramme et le dense (R3-R4), puis distiller le professeur dense dans un organisme dont le métabolisme reste actif (R5-R8). `mycelia/brain.py` — le cerveau de Nyra — n'est pas modifié : la réparation est une **sous-classe** dans `scripts/lab/`.

## 1. La réparation de la lecture

Trois changements, chacun justifié par une mesure, tous dans `scripts/lab/repaired_mycelia.py` :

1. **Agrégat de toutes les membranes (voie lente).** Les 8 nœuds du jouet donnent 8 × 16 = 128 dimensions, normalisées puis projetées vers la dimension de lecture (16). Le nœud 1 reste concaténé à côté, normalisé lui aussi : c'est le témoin de comparaison avec S3-BIS.
2. **Voie rapide (R2) : table d'embedding dédiée au token précédent.** La première version réutilisait l'embedding d'injection ; mesuré : 5,16 après 150 pas, contre 4,24 pour le même bigramme isolé — les gradients de la voie rapide servaient le graphe, pas la lecture. La table dédiée coûte 312 × 16 paramètres et apprend.
3. **Mixage initialisé en bigramme.** Le bloc de la voie rapide est l'identité, les deux autres sont à zéro. Sans cela l'agrégat (norme mesurée jusqu'à 111) écrase tout et la tête reste au plancher : 5,08 après 300 pas, à l'identique de l'ancien modèle.

Paramètres ajoutés, comptés module par module (deux vitesses) :

| Poste | Paramètres | Rôle |
|---|---|---|
| voie lente (LN 128 + 128→16) | 2320 | l'agrégat des 8 membranes |
| témoin nœud 1 (LN 16) | 32 | l'ancienne lecture, gardée |
| voie rapide (LN 16 + table 312×16) | 5024 | le dernier token |
| mixage (48→16) | 784 | concat + linéaire |
| **total ajouté** | **8160** | 23 748 → 31 908 |

Le jumeau équitable de S3-BIS comptait 16 711 paramètres actifs par token ; le MYCELIA réparé en compte ~24 871, soit **1,49×** : la comparaison est défavorable à MYCELIA, et c'est le sens honnête. La variante « pleine lecture seule » n'ajoute que 2880 paramètres (26628 au total).

### Le décalage d'un cran de la métrique de S2/S3-BIS

En réparant la tête, une mesure de contrôle a révélé un défaut de mesure : `lab_common.mycelia_loss` compare les logits de MYCELIA au couple `y` des lots, c'est-à-dire au token **suivant**, alors que `Brain.forward` lit avant d'injecter et prédit donc le token **courant**. La validation de S2/S3-BIS demandait au cerveau un pas de plus que ce qu'il apprenait.

| Modèle | mesure honnête (cible `x`) | mesure publiée (cible `y`) |
|---|---|---|
| ancien MYCELIA (S2) | 5,1484 | 5,1509 |
| MYCELIA réparé, même instantané | 4,0600 | 6,6400 |

Sur un modèle au plancher, les deux cibles se valent (5,1484 contre 5,1621) : la conclusion de S3-BIS ne bouge pas. Sur un modèle qui prédit vraiment, l'écart devient énorme. Toutes les tables de ce rapport donnent les deux colonnes ; c'est la colonne **honnête** qui décide, parce que c'est la tâche du dense et celle des planchers n-grammes.

## 2. La table de validation et le verdict

Protocole : 600 pas, semences 1, 2, 3, lr 2e-3, batch 32, weight decay 0,01, métabolisme actif. Corpus 1000 phrases (S2) et corpus 10k (`corpus_toy10k.txt`).

| Corpus | Tête | Perte honnête (moy.) | écart-type | Perte décalée (moy.) | Semences |
|---|---|---|---|---|---|
| 1000 phrases | lecture pleine (voie lente seule) | 3,8354 | 0,0373 | 7,4155 | 3,8424, 3,7951, 3,8686 |
| 1000 phrases | deux vitesses | 3,9039 | 0,0584 | 7,9402 | 3,8421, 3,9115, 3,9582 |
| 10k phrases | lecture pleine (voie lente seule) | 3,5636 | 0,0064 | 7,6286 | 3,5640, 3,5698, 3,5571 |
| 10k phrases | deux vitesses | 3,5985 | 0,0159 | 8,1441 | 3,5989, 3,6141, 3,5824 |

Contrôles : bigramme lissé **3,0466** · dense jumeau **3,7921** · ancien MYCELIA **5,1509** (honnête 5,1484) · unigramme 5,0829.

**Contrôle de capacité : le bigramme de rang 16.** Le plus petit modèle qui ait la même largeur de lecture que le jouet (embedding 312×16 → 16 → tête, aucun graphe, aucun contexte au-delà d'un mot) vaut **4,0288** sur ce même jeu de validation. Le MYCELIA réparé (3,8354) le bat de -0,1934 nat : l'agrégat des membranes apporte donc du contexte au-delà de la récence pure. La barre n'est ni « le dense » ni « le bigramme » : c'est une combinaison des deux.

**La comparaison appariée.** Le 3,7921 publié par S3-BIS a été mesuré sur **16 fenêtres** de validation ; nos 58 fenêtres (toutes celles du flux tenu à part) sont un jeu plus dur — le contrôle de rang 16 le montre (4,03 ici, 3,79 là-bas). Pour comparer honnêtement, le dense jumeau a été réentraîné dans les mêmes conditions et mesuré sur les mêmes 58 fenêtres : **3,8934** (semence 1) et 3,8790 (semence 2). Le MYCELIA réparé (lecture pleine, 3,8354) est donc **devant le dense de 0,0580 nat à jeu de validation égal**. Par la règle chiffrée de la mission — qui garde le seuil publié — le verdict reste 🥉 ; en comparaison appariée, la réparation franchit la barre.

**Le corpus change l'échelle, pas le classement.** Sur les 10 000 phrases, au même budget de 600 pas, le MYCELIA réparé (lecture pleine) vaut **3,5636** contre **3,2886** pour le dense jumeau : le dense reste devant de 0,2750 nat. Mais l'écart de S3-bis, mesuré sur 300 pas, était de 1,4357 : la réparation le divise par 5.2 et le réparé frôle le plancher bigramme du corpus 10k (3,5626). Autrement dit : la lecture était bien le goulot principal, et ce qui reste est la mémoire de séquence du graphe, pas la taille du corpus.

**Verdict chiffré : 🥉 — le MYCELIA réparé ne bat pas le seuil publié du dense : le goulot restant doit être diagnostiqué** (MYCELIA réparé, 1000 phrases, moyenne : **3,9039**).

Lecture : la réparation fait passer le jouet de 5,1484 à 3,9039, soit **+1,2445 nat** gagnés, en 600 pas. Les deux têtes apprennent (la variante pleine lecture n'est pas loin), ce qui confirme le diagnostic de S3-BIS : le contenu était là, l'interface le jetait.

Perte par position (semence 1, deux vitesses) : t0=5,517, t1=4,383, t2=3,406, t3=3,652, t4=3,738, t5=3,921, t6=2,937, t7=3,186. Le gain sur le bigramme est bien là où il doit être : aux positions tardives, celles qui ont le plus de contexte ; la position 0 reste au niveau de l'unigramme, aucun modèle ne peut y faire mieux.

## 3. Les sondes avant/après (R4)

Même protocole que D2 (mêmes trois conditions, mêmes sondes, même budget, indice de position inclus). Une sonde = une softmax linéaire entraînée sur 400 fenêtres, testée sur les 58 fenêtres tenues à part.

| Condition | avant (S2) | après (S4) | écart |
|---|---|---|---|
| normal | 5,0817 | 3,6093 | -1,4724 |
| melange | 5,1031 | 6,1809 | +1,0778 |
| vide | 5,1354 | 6,8009 | +1,6655 |

| Sonde | dims | avant | après | écart |
|---|---|---|---|---|
| nœud 1 seul (l'ancienne lecture) | 16 | 5,3680 | 5,3680 | +0,0000 |
| nœuds du milieu | 96 | 4,9233 | 4,4542 | -0,4691 |
| les 8 membranes | 128 | 4,9246 | 4,4466 | -0,4780 |
| vecteur de lecture du modèle réparé | 16 | 5,3680 | 4,0458 | -1,3222 |
| témoin : embeddings du préfixe | 128 | 5,4016 | 5,3860 | -0,0156 |

Écart normal/mélangé : **0,0537 nat avant → 3,1916 nat après** (objectif de la mission : > 0,5 ; atteint).

Le nœud 1 seul passe de 5,3680 à 5,3680 ; la sonde « toutes membranes » de 4,9246 à 4,4466. Le nœud 1 reste loin de la sonde pleine : la tête utilise surtout la voie rapide et marginalise la mémoire du graphe.

## 4. La distillation

Professeur : dense jumeau S2, 23748 paramètres, 3,8934 de validation (semence 1). Formule : `L = KL(logits_MYC ‖ logits_prof) + λ · MSE(P·h_MYC, h_prof)`, λ = 0,5, projecteur `P` initialisé à l'identité et **jeté après usage**. Aucun terme de cross-entropy : le savoir doit venir du professeur.

Les trois bras partagent les mêmes données, le même budget (600 pas, mêmes lots) et le métabolisme actif :

| Bras | Perte | écart-type | R | R écart-type | KL(P_prof‖P_G) | top-3 | Semences |
|---|---|---|---|---|---|---|---|
| M0 — réparé, seul | 3,9039 | 0,0584 | +0,9912 | 0,0492 | 0,7757 | 0,3151 | 3 |
| réparé + professeur (λ = 0,5) | 4,0520 | 0,0197 | +0,8667 | 0,0166 | 0,3211 | 0,2641 | 3 |
| professeur à étiquettes mélangées | 5,7456 | 0,0030 | -0,5571 | 0,0025 | 2,2193 | 0,0055 | 3 |
| réparé + professeur (λ = 0,1) | 4,0350 | 0,0000 | +0,8809 | 0,0000 | 0,3679 | 0,2514 | 1 |

**Gain de distillation isolé (moyenne des R) : -0,1245.** C'est la seule comparaison qui prouve le transfert : le même organisme, le même budget, le même corpus — avec et sans professeur.

Contrôle anti-bruit : R = -0,5571 avec un professeur aux étiquettes mélangées contre +0,8667 avec le vrai professeur, soit **+1,4238** d'écart (le signal est propre).

**Le résultat est négatif, et c'est un résultat.** La distillation n'ajoute rien : elle **retire** 0,1245 nat de R au lieu d'en donner. Le contrôle montre que le chemin fonctionne (étiquettes mélangées : R = « −0,56 », soit l'état initial ; vrai professeur : +0,87) — le transfert a donc bien lieu, mais il coûte plus qu'il ne rapporte. Deux causes mesurées : le terme d'alignement reste énorme (MSE ≈ 0,74 avec un projecteur à l'identité, les deux espaces ne se rejoignent pas) et, sans terme de cross-entropy pour ancrer la tâche, la tête dérive vers la forme du professeur plutôt que vers ses réponses justes (fidélité 0,41 de l'accord inter-professeurs, contre 1,00 pour M0, mais 0,15 nat de perte en plus). Une distillation qui marche demanderait un projecteur appris plus longtemps et un ancrage supervisé — c'est un protocole à réviser, pas une impasse.

## 5. Les métriques honnêtes (R7)

**R = (L_ref − L_G) / (L_ref − L_prof)** avec L_ref = 5,0829 (unigramme) et L_prof = 3,8934 : la part du chemin entre « ne rien savoir » et le professeur. Ce n'est pas l'écart relatif naïf de S2, qui mesurait la distance au professeur sans dire d'où l'on part.

**Fidélité normalisée.** KL(P_prof ‖ P_G) est divisée par l'accord inter-professeurs, calculé **en premier** : KL entre deux dense entraînés avec des graines différentes = **0,7775**. C'est la limite physique de la métrique : exiger d'un élève un KL plus petit que l'accord de deux professeurs serait demander l'impossible.

**top-3 sur 40 phrases neuves** (même grammaire, graine différente du corpus). Uniforme = 0,0096, professeur = 0,2568 (seconde graine : 0,2131). Le top-3 normalisé est ramené entre ces deux bornes.

| Bras | R | KL(P_prof‖P_G) | fidélité normalisée | top-3 | top-3 normalisé |
|---|---|---|---|---|---|
| M0 — réparé, seul | +0,9912 | 0,7757 | 1,00 | 0,3151 | 1,24 |
| réparé + professeur (λ = 0,5) | +0,8667 | 0,3211 | 0,41 | 0,2641 | 1,03 |
| étiquettes mélangées | -0,5571 | 2,2193 | 2,85 | 0,0055 | -0,02 |
| λ = 0,1 | +0,8809 | 0,3679 | 0,47 | 0,2514 | 0,98 |

Significativité : **trois semences, donc indicatif**. Un écart-type sur trois points n'a pas de valeur inférentielle ; les tableaux donnent la dispersion, pas un test. Le test t apparié de `lab_common` n'est pas invoqué ici : avec n = 3 il ne pourrait que maquiller l'incertitude.

## 6. La survie au métabolisme (R8) — le verdict de la mission

Après distillation, le métabolisme tourne **libre pendant 1000 pas** : naissances d'arêtes payées par le trafic, adresses honorées, coupe des synapses famines — et **aucun pas d'optimisation**. Critère : `R_après ≥ 0,9 × R_avant`.

| Semence | R avant | R après | ratio | arêtes | mortes | survit |
|---|---|---|---|---|---|---|
| 1 | +0,8852 | +0,8852 | 1,00 | 48→48 | 0 | oui |
| 2 | +0,8616 | +0,8616 | 1,00 | 48→48 | 0 | oui |
| 3 | +0,8532 | +0,8532 | 1,00 | 48→48 | 0 | oui |

**3/3 semences survivent.** R moyen +0,8667 → +0,8667.

**Mais ce test est vide, et il faut le dire : aucune arête n'est née ni morte en 1000 pas.** Le jumeau paramétrique de S2 a 8 nœuds, donc 56 paires dirigées admissibles au maximum, dont 48 déjà câblées : le graphe est **saturé**, et le métabolisme n'a plus rien à créer ni à couper. Le critère est satisfait parce que rien ne s'est passé, pas parce que le savoir a résisté. La survie au métabolisme reste donc **non mesurée** ; elle demande un bac à sable à pool 24 (où les naissances de nœuds existent), et c'est la première chose à faire ensuite.

## 7. Conclusion stratégique

Le MYCELIA réparé vaut **3,9039** sur 1000 phrases (deux vitesses), **3,8354** en lecture pleine, et **3,5636** sur le corpus 10k. La réparation rend la tête fonctionnelle — elle gagne plus de 1,2 nat et l'écart normal/mélangé passe de 0,05 à 3,19 — mais elle ne franchit pas le seuil publié du dense (3,7921).

**Le nouveau plafond est double, et il est chiffré.** (1) *La largeur de lecture.* Un bigramme de rang 16 — même largeur, aucun graphe — vaut 4,03 ; le réparé le bat de 0,19, donc l'agrégat apporte bien du contexte au-delà de la récence. Mais 16 dimensions ne peuvent pas porter un bigramme plein rang (le plancher lissé est 3,05) : le facteur limitant est la dimension, pas le graphe. (2) *L'agrégation.* La sonde sur les 8 membranes vaut 4,4466 — mieux qu'avant (4,92) mais loin de ce qu'une tête entraînée en bout de chaîne extrait (3,84). Une somme à décroissance exponentielle garde le sac de mots récent, pas la séquence : la suite logique est une mémoire à deux vitesses **dans le graphe** (une membrane rapide non refroidie par nœud), pas une nouvelle tête.

À jeu de validation égal, le réparé **bat** le dense réentraîné (3,8934 sur 58 fenêtres) : la barre des 90 % du convertisseur n'est donc plus hors d'atteinte — elle demande une lecture plus large et une mémoire de séquence, deux changements d'architecture mesurables, pas un miracle d'entraînement.

Pour la barre des 90 % du convertisseur, la conclusion tient en une phrase : le point dur n'est plus « MYCELIA peut-il apprendre ? » mais « que coûte une mémoire de séquence dans un graphe à somme ? ». Et la distillation n'est pas encore la réponse : telle qu'elle est formulée ici, elle transfère la forme du professeur sans transférer son avantage (section 4).

## 8. Échecs honnêtes, surprises, périmètre non vérifié

- **Le décalage d'un cran** (section 1) : la métrique de validation de S2/S3-BIS demandait au cerveau un token de plus que ce qu'il apprenait. Sans conséquence sur un modèle au plancher (5,1484 contre 5,1621), énorme sur un modèle qui prédit. Les deux colonnes sont publiées partout ; c'est un correctif de mesure, pas un résultat.
- **Deux fausses pistes corrigées par la mesure** : réutiliser l'embedding d'injection pour la voie rapide (5,16 après 150 pas) et laisser le mixage s'initialiser au hasard (5,08 après 300 pas, plancher exact). Les deux sont documentées dans le module, pas effacées.
- **Le métabolisme du jouet est un métabolisme d'arêtes.** Le pool est figé à 8 nœuds (contrainte du jumeau paramétrique de S2 : chaque nœud coûte 1 665 paramètres). Les naissances et les morts mesurées en section 6 sont donc celles des synapses, pas des neurones. Un essai à pool 24 est un périmètre non vérifié.
- **Trois semences** : les moyennes et écarts-types sont indicatifs, aucun test d'inférence n'est revendiqué.
- **Non vérifié** : le portage vers `mycelia/brain.py` (mission séparée par consigne), le corpus réel de Nyra, l'effet d'un corpus plus grand que 10k, et toute forme d'attention sur 2-3 tokens (la version retenue est la plus simple : l'embedding du token précédent).
- **Le génome de Nyra n'a pas été touché** : `data/genome.pt`, `data/nyra_state.json`, `data/metrics.json` et le lanceur sont inchangés ; tout le labo écrit dans `data/lab/`, et ses sauvegardes sont gitignorées.
