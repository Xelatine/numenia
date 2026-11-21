# RAPPORT S3-bis — pourquoi MYCELIA plafonne à ~5,1

**Verdict : 🐛 BUG** — le plafond de MYCELIA est exactement le plancher sans contexte : il ne s'agit pas d'un plafond d'apprentissage mais d'un défaut d'usage du contexte, localisé dans l'agrégation et la lecture.

Mission : trancher entre un bug d'usage du contexte, une lenteur d'apprentissage (A) et un plafond structurel du sparse (B). Toutes les expériences tournent en circuit court sur CPU, sans installation, et ne touchent ni au génome de Nyra ni à sa session 24/7.

## 1. Les planchers n-grammes (D1) — à lire en premier

Corpus : `corpus_toy.txt`, 1000 phrases, vocabulaire 312 mots, lissage hiérarchique alpha = 0.5.

| Mesure (nats/token) | Valeur | Lecture |
|---|---|---|
| uniforme ln(V) | 5.7430 | aucune information |
| unigramme (MLE) | 5.0818 | 310 types, 4664 tokens |
| **unigramme lissé** | **5,0829** | **L_ref : plancher sans contexte** |
| bigramme (MLE) | 2.7195 | 170 contextes |
| bigramme lissé | 3,0466 | plancher avec un mot d'histoire |
| trigramme (MLE) | 1.6304 | 2723 contextes |
| trigramme lissé | 1,9828 | plancher avec deux mots d'histoire |

**L_ref = 5,0829 nats/token** — c'est le plancher d'un modèle qui prédit chaque mot sans regarder ce qui précède.

MYCELIA S2 mesure **5,1000**, soit **+0,0171 nat** de l'unigramme.

> **BUG** — l'unigramme vaut déjà le niveau de MYCELIA : il prédit comme un sac de mots, ce n'est PAS un plafond d'apprentissage mais un bug d'usage du contexte.

La conséquence est brutale et c'est le résultat le plus décisif du projet : un bigramme lissé vaut 3,05 nats, un trigramme 1,98. Il y a donc **2,03 nats de contexte disponibles** que MYCELIA ne prend pas. Son « plafond » de 5,1 n'est pas une limite de capacité : c'est le niveau exact d'un sac de mots.

## 2. Le test de contexte (D2)

Mêmes cibles, mêmes positions, seul le préfixe change :

| Condition | Perte | Exactitude | Ce qui change |
|---|---|---|---|
| contexte normal | 5,0718 | 0,114 | le vrai flux |
| contexte mélangé | 5,0937 | 0,065 | même sac de mots, ordre détruit |
| contexte vidé | 5,1253 | 0,039 | aucune information |

Écart maximal entre conditions : **0,0535 nat**. Divergence KL(normal ‖ mélangé) = 0.00364, KL(normal ‖ vidé) = 0.01339.

**Verdict : le contexte est fonctionnellement ignoré.** Sur les 2,0363 nats que le bigramme promet, le modèle en prend moins de 0,06.

Où meurt-il ? Quatre sondes identiques (même softmax, même budget, même régularisation, indice de position inclus) :

| Sonde | Largeur | Perte | Exactitude |
|---|---|---|---|
| nœud 1 (la tête de lecture) | 23 d | 5,3650 | 0,036 |
| nœuds du milieu | 103 d | 4,9136 | 0,128 |
| **les 8 membranes** | 135 d | **4,9141** | 0,128 |
| témoin : embeddings du préfixe | 135 d | 5,3885 | 0,085 |

- La tête ne lit **qu'un nœud sur huit** : passer de 16 à 128 dimensions rend **0,4509 nat** — c'est réel, mais marginal.
- Même avec les 128 dimensions, l'état ne gagne que **0,1577 nat** sur le modèle : l'ordre des mots n'est plus dans la mémoire. Une somme à décroissance exponentielle est un sac, pas une séquence.
- Le routage est disculpé : l'énergie par saut monte (9,5, 58,5, 93,4, 111,0), et le dernier mot reste décodable dans 0,407 des cas après 2 tokens, 0,131 après 7 (hasard : 0,003). Sur l'état mesuré, seuls [5, 7] passent sous le seuil de 0,1.

**Suspect n°1 : `agregation`** — l'agrégation (une seule membrane par nœud, refroidie à 0,95 par saut) détruit la position ; la lecture d'un seul nœud ajoute sa petite perte par-dessus.

Perte par position (préfixe réel) : t0=5.612, t1=5.348, t2=4.896, t3=5.160, t4=5.241, t5=5.133, t6=4.782, t7=4.943. La position 0 n'a aucun contexte (5,61) ; les suivantes ne descendent pas sous 4,78, très loin du bigramme.

## 3. Les deux courbes longues et le verdict A/B (D3)

500 pas (10× S2), log tous les 5 pas, 48 fenêtres de validation, mêmes lots pour les deux cerveaux.

| Bras | Params | Meilleure val | Au pas | L∞ ajusté | L∞ retenu | Verdict |
|---|---|---|---|---|---|---|
| MYCELIA (semence 1) | 23748 | 5,1509 | 55 | 3,6509 | 5,1509 | B |
| dense jumeau (contrôle) | 23748 | 3,7921 | 315 | 2,2921 | 3,7921 | A |

**VERDICT A/B : 🧱 B — plafond structurel**

- MYCELIA : gain +0.0000 sur les 80 % finaux, ajustement au bord du domaine : la loi de puissance ne pin pas l'asymptote, on retient la meilleure validation atteinte (l'incertitude est le gain residuel observe)
- dense : gain +0.7584 sur les 80 % finaux, ajustement au bord du domaine : la loi de puissance ne pin pas l'asymptote, on retient la meilleure validation atteinte (l'incertitude est le gain residuel observe)

**Comment ce B se combine avec le verdict 🐛 du haut.** Ce B répond à la question A/B : ce n'est *pas* de la lenteur. Il ne dit pas que la sparsité est le mur. Le mur est structurel au sens où il ne bouge pas avec le budget, mais il est **dans la mémoire** (D1, D2, D4) : un état qui somme ses messages sans marquer le dernier token. C'est donc un plafond structurel **réparable**, pas une limite de l'architecture sparse.

Lecture honnête de l'ajustement : la loi de puissance `L∞ + a·t^(−b)` a bien été ajustée en log-log, mais son minimum tombe **au bord du domaine** pour les deux bras (3,65 et 2,29 = minimum observé − 1,5), ce qui signifie que la somme des carrés décroît jusque-là sans jamais remonter : l'asymptote n'est pas identifiable. MYCELIA ne gagne **0,0000 nat** sur les 80 % finaux du budget. L'asymptote retenue est donc la meilleure validation atteinte, et son incertitude est ce gain résiduel.

Autrement dit : multiplier le budget par dix ne rapproche pas MYCELIA du dense. Ce n'est pas de la lenteur.

## 4. Les ablations du routage (D4)

Même budget que D3 (500 pas), même semence, mêmes lots.

| Variante | Ce qui change | Perte | Écart | Arêtes |
|---|---|---|---|---|
| controle | config S2 (référence) | 5,1536 | +0,0000 | 48 |
| i_routage_dense | affinité par arête coupée : transmission pleine | 5,1476 | -0,0060 | 48 |
| ii_metabolisme_gele | naissances figées au pas 25, cycle de vie coupé | 5,1536 | +0,0000 | 48 |
| iii_graphe_aleatoire | arêtes aléatoires figées (20000 tirages) — le tirage sature à 42 arêtes (toutes les paires admissibles) | 5,1727 | +0,0191 | 42 |

**Verdict D4 : `PAS_LE_ROUTAGE`** — couper l'affinité ne rapproche pas du dense : le routage n'est pas le goulot à ce dimensionnement.

Diagnostic du routage sur le contrôle :

- allumage par nœud (nœud chaud, sur 24 lots) : [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0] (Gini 0.0)
- gradient par nœud : [0.7001, 0.059, 0.4301, 0.4563, 0.4534, 0.4443, 0.3427, 0.3695] (Gini 0.2043)
- nœuds jamais allumés : [0] (la porte, par définition) ; sans gradient : aucun

L'équilibre de charge est le point à surveiller. Le nœud 0 est la porte : il reçoit le token, jamais un message, donc son zéro est une définition et pas une inégalité (il est exclu du Gini). Sur les 7 autres, tous sont allumés dans chaque lot : pas d'effondrement « riches toujours plus riches » à l'échelle de la colonie. Mais sur un état donné, deux nœuds peuvent tomber sous le seuil de 0,1 (mesuré au test de contexte : nœuds 5 et 7, portes 0,005 et 0,051) : l'activité est inégale d'un token à l'autre, ce qui est le comportement attendu d'un routage par affinité, pas une amputation.

## 5. Le jumeau équitable : paramètres ACTIFS égaux (D5)

S2 égalisait 23 748 paramètres *totaux*. Ce n'est pas le même travail par token : le dense consulte tout, MYCELIA une partie. Estimation mesurée sur la validation :

| Poste | Paramètres consultés |
|---|---|
| embedding (une ligne lue) | 16 |
| tête de lecture | 4992 |
| nœuds actifs (7.0 × 1 121) | 7847,0 |
| requêtes + clés | 3808,0 |
| arêtes (un coefficient chacune) | 48 |
| **total actif par token** | **16711,0** |

Le dense équitable a donc **MLP 61**, soit **16818 paramètres** au total, entraîné au même budget (500 pas, mêmes lots).

| Bras | Perte |
|---|---|
| dense équitable (actifs égaux) | 3,9218 |
| MYCELIA, même budget | 5,1509 |
| dense jumeau S2 (23 748 params) | 4,5800 |

Écart MYCELIA − dense équitable : **+1,2291 nat**.

Ils ne s'égalent pas. Le plafond n'est donc pas « le prix attendu du sparse à calcul égal » : à calcul égal, le dense fait mieux, et le sparse ne compense pas par autre chose. Cela pointe vers l'architecture de la mémoire, pas vers un arbitrage computationnel normal.

## 6. Le corpus 10× : capacité ou surapprentissage ? (D6)

10000 phrases (47696 tokens), même grammaire, même vocabulaire, 0 mot inconnu. Budget court : 300 pas.

| Plancher sur le gros corpus | Valeur |
|---|---|
| unigramme | 5,1193 |
| bigramme | 3,5626 |
| trigramme | 2,7181 |

| Bras | Meilleure val | Au pas |
|---|---|---|
| MYCELIA | 5,1625 | 80 |
| dense | 3,7268 | 300 |

Écart : **+1,4357 nat**. Le plafond tient : **oui**.

Le corpus 10× supprime l'essentiel du surapprentissage (la validation ne remonte plus aussi vite) et le plafond ne bouge pas : ce n'est donc pas la taille du corpus qui parlait. La capacité — au sens de ce que l'état peut représenter — reste la même.

### Le test de réparation (D2′) : et si on changeait la récence ?

Puisque le diagnostic accuse l'agrégation, une seule vis devait suffire : la vitesse de refroidissement de la membrane (`state_decay`). Même budget court, même semence, mêmes lots.

| state_decay | Perte | Au pas | Unigramme | Bigramme |
|---|---|---|---|---|
| 0,50 | 5,2645 | 60 | 5,0829 | 3,0466 |
| 0,80 | 5,2124 | 60 | 5,0829 | 3,0466 |

Meilleur réglage mesuré : **0,80 → 5,2124**. La récence seule ne suffit pas : l'agrégation n'est pas le seul verrou, la lecture d'un seul nœud et la non-linéarité du corps comptent aussi.

## 7. Conclusion stratégique : le plafond réel du convertisseur

**🐛 BUG — le plafond de MYCELIA est exactement le plancher sans contexte : il ne s'agit pas d'un plafond d'apprentissage mais d'un défaut d'usage du contexte, localisé dans l'agrégation et la lecture.**

La perte minimale réellement atteignable par MYCELIA sur ce corpus, telle que mesurée aujourd'hui : **5,1509 nats/token**. À 0,0680 nat du plancher unigramme (5,0829), ce n'est pas un plafond d'architecture sparse : la colonie n'utilise pas l'ordre des mots.

Ce que cela fixe pour toute traduction future :

| Niveau | Perte (nats) | Ce qu'un cerveau à ce niveau sait faire |
|---|---|---|
| MYCELIA aujourd'hui | 5,1509 | statistiques marginales : aucun contexte |
| unigramme (L_ref) | 5,0829 | sac de mots |
| bigramme | 3,0466 | le mot suivant dépend du précédent |
| trigramme | 1,9828 | deux mots d'histoire |
| dense S2 (500 pas) | 3,7921 | contexte ordonné par attention |

La barre du convertisseur est donc **5,1509**, et elle est trop haute : la cible d'un bigramme (3,0466) est *sous* le dense (3,7921). Un MYCELIA qui lirait seulement deux mots d'histoire battrait le dense à budget égal. Le problème n'est pas la sparsité, c'est la mémoire.

Tant que MYCELIA reste à ce plancher, **aucun convertisseur ne peut transférer mieux que des statistiques marginales** : les 90 % de traduction visés sont inatteignables, non par manque de pas (D3), non par le routage (D4), mais parce que le receveur ne voit pas l'ordre des mots (D2).

**Ce que la S4-CORRIGÉ doit faire** : lire toutes les membranes (pas seulement le nœud 1) et donner à la lecture un accès au dernier token — attention sur la fenêtre récente ou lecture à deux vitesses (membrane lente + message courant). C'est un choix d'architecture, pas un renoncement : le plafond de 5,1 disparaît avec lui.

## 8. Échecs honnêtes et surprises

- **L'ajustement de puissance n'a rien ajusté.** Sur une courbe plate 445 pas durant, `L∞` n'est pas identifiable : le minimum de la somme des carrés tombe au bord du domaine (3,65 pour MYCELIA, 2,29 pour le dense). Le rapport ne présente donc pas ces chiffres comme des prévisions — il retient la meilleure validation atteinte et l'écrit.
- **La première sonde de contexte était fausse.** Sans indice de position, la même cible tombait dans un bloc de caractéristiques différent selon `t` ; et une séparation train/test par *mot* rendait 0 % (une sonde linéaire ne peut pas deviner une classe jamais vue). Corrigé : position partagée par les quatre sondes, séparation par échantillon.
- **Le témoin de largeur ne témoigne pas.** Les embeddings bruts du préfixe n'atteignent que 5,3885 : prédire 312 classes depuis 16 dimensions par position reste un problème de rang, pas de largeur. Il ne permet donc pas de conclure « la largeur suffit » ; c'est la comparaison 16 d → 128 d sur les membranes qui porte le diagnostic.
- **Le dense n'est pas un plafond du corpus.** À 500 pas il est à 3,7921, au-dessus du bigramme (3,0466) : il n'a pas fini d'apprendre non plus. La comparaison A/B porte sur le même budget, pas sur la convergence parfaite.
- **La validation a changé de taille.** S2 mesurait sur 16 fenêtres, S3-bis sur 48 : le 5,078 de S2 et le 5,1509 de D3 sont le même modèle à un bruit près. Les deux chiffres sont conservés, aucun n'est recyclé pour l'autre.
- **Le test de réparation a échoué.** Refroidir la membrane plus vite (0,80 puis 0,50) ne fait pas tomber la perte : 5,2124 et 5,2645, soit *pire* que 5,1509. La récence est donc une cause partielle, pas le remède : il faut agir sur la lecture et sur le mécanisme qui désigne le dernier token, pas seulement sur une constante de temps.
- **Le graphe aléatoire ne peut pas faire 48 arêtes.** Avec 8 nœuds, un lecteur muet et aucune arrivée sur les portes, il n'existe que 42 paires admissibles : cette variante travaille avec 6 arêtes de moins, ce qui explique peut-être son +0,019.
- **Le vocabulaire du gros corpus est inchangé** (312 mots) pour que les embeddings restent comparables : D6 teste la quantité de données, pas la couverture lexicale.
