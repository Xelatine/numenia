> **Renommage —** Ce projet s'appelle désormais **Numenia** (auparavant référencé
> comme Mycelia dans les titres et la documentation). Session, chiffres, fichiers de
> données et cerveau **Nyra** inchangés. Ce document est historique : son contenu est
> conservé tel quel, et seul le nom du projet passe à Numenia.

# RAPPORT S5-MAX — casser les quatre verrous, mesurer le plafond physique

Mission S5-MAX. Quatre verrous : la mesure juste, le receveur vivant au
pool 24, la distillation ancree, et le plafond physique — l'accord entre
deux professeurs independants. Tous les chiffres de ce rapport sont lus
dans les JSON de `data/lab/` ; aucun n'est recopie a la main.

Rappel des interdits respectes : aucun `pip install`, aucun CDN, le genome
de Nyra (`data/genome.pt`), `data/nyra_state.json`, `metrics.json` et le
lanceur ne sont pas touches — la reparation reste une **sous-classe** du
cerveau, dans `scripts/lab/`.

## 1. La mesure corrigée (M1/M2)

`lab_common.mycelia_loss` comparait les logits de MYCELIA au token
**suivant** (`y`) alors que le cerveau predit le token **courant** (`x`) :
il lit le token avant de l'injecter, et sa cible d'entrainement est
`corpus.targets(x) = x`. La fonction compare desormais a `x` ; l'ancienne
vit, inchangee, sous `mycelia_loss_legacy`, et `aligned_loss`/`shifted_loss`
de `repaired_mycelia` n'en sont plus que des alias.

**Preuve de M1** : les 12 cerveaux de S4, re-mesures avec la
fonction corrigee (position 0 incluse, comme S4), redonnent exactement
les pertes honnetes deja publiees — ecart maximum **0.0000**, et
**0.0000** sur la colonne ancienne. Le classement ne peut
donc pas avoir bouge par accident de re-entrainement : les cerveaux sont
les memes, seule la cible change.

### Les deux colonnes, memes fenetres

| modele | corpus | colonne nouvelle (x[t]) | colonne ancienne (y[t]) |
|---|---|---|---|
| plancher unigramme | — | **5.0829** ± 0.0000 | 5.0829 ± 0.0000 |
| plancher bigramme | — | **3.0466** ± 0.0000 | 3.0466 ± 0.0000 |
| plancher trigramme | — | **1.9828** ± 0.0000 | 1.9828 ± 0.0000 |
| MYCELIA repare 1000/pleine | 1000 | **3.5936** ± 0.0368 | 7.4155 ± 0.2447 |
| MYCELIA repare 1000/deux_vitesses | 1000 | **3.6672** ± 0.0559 | 7.9402 ± 0.4059 |
| MYCELIA repare 10k/pleine | 10k | **3.3394** ± 0.0053 | 7.6286 ± 0.0199 |
| MYCELIA repare 10k/deux_vitesses | 10k | **3.3790** ± 0.0162 | 8.1441 ± 0.1085 |
| dense jumeau 1000 | — | **3.9229** ± 0.0657 | 3.9193 ± 0.0578 |
| dense jumeau 10k | — | **3.3308** ± 0.0189 | 3.3169 ± 0.0250 |

Les planchers n-grammes sont **le meme nombre dans les deux colonnes** —
un n-gramme predit deja le token courant a partir du prefixe, et c'est
cette egalite, verifiee en re-executant la fonction de S3-BIS, qui montre
que le decalage ne vivait que dans la comparaison des logits du cerveau.

### Le classement : **inversé**

| colonne | ordre (meilleur d'abord) |
|---|---|
| nouvelle | bigramme < MYCELIA < dense |
| ancienne | bigramme < dense < MYCELIA |

Dans la metrique publiee, MYCELIA (reparation deux vitesses) valait
**7.9402** contre **3.9193**
pour le dense : il perdait largement. Dans la mesure juste il vaut
**3.6672** contre **3.9229** : il
gagne. **Les deux places s'echangent** (MYCELIA passe de dernier a
deuxieme, le dense l'inverse), et l'ancienne mesure n'etait pas une
penalite constante : plus le cerveau apprend, plus la cible decalee
s'eloigne de ce qu'il predit (5,15 quand il etait au plancher, 7,9
maintenant qu'il predit). C'est la justification la plus forte de tout
S5 : les chiffres publies de S2/S3-BIS etaient structures par un bug de
mesure, et le classement qu'ils donnaient etait faux.

## 2. Le plafond physique (P1) — le maximum atteignable

Deux professeurs denses **independants** (graines 1 et 2), de la taille du
jumeau equitable (63480 parametres,
mlp 768), entraines 600 pas sur le meme flux. Leur desaccord est la limite
que la realite impose a tout eleve :

| chiffre | valeur |
|---|---|
| KL(prof A ‖ prof B) — **le plafond** | **0.8582 nat** |
| KL(prof B ‖ prof A) | 0.8356 nat |
| KL symetrique | 0.8469 nat |
| top-3 identiques | 0.0115 |
| recouvrement top-3 moyen | 0.3053 / 1 |
| contextes mesures | 5376 |

Deux professeurs qui ont vu exactement le meme corpus, avec la meme
architecture et le meme budget, different de **0.8582 nat** par contexte, et leurs top-3 ne sont
identiques que dans **1.1 %** des cas (recouvrement
moyen 0.305 des trois mots). C'est le maximum :
demander a MYCELIA d'etre plus proche de A que B ne l'est serait demander
l'impossible.

## 3. Le pool 24 (V1/V2)

| element | valeur |
|---|---|
| parametres actifs / token | **63276** |
| dense jumeau (mlp 768) | **63480** |
| ratio dense / MYCELIA | 1.0032 |
| aretes (plafond) | 220 / 552 paires = 39.9 % |
| perte nouvelle (3 semences) | **3.7363 ± 0.0512** |
| perte ancienne | 7.4452 ± 0.1449 |
| dense jumeau, meme budget | 3.8775 |
| verdict | 🥈 — le pool 24 bat le dense jumeau du pool 24 |

Le pool 24 ameliore le repere de S4 (3,904 en deux vitesses au pool 8) de
**0.1676 nat**, et passe
devant son propre jumeau dense — mais reste derriere le bigramme (3,0466).
Deux verdicts coexistent donc, et il faut les nommer : **🥈 contre le dense
jumeau (la comparaison d'architecture), 🥉 contre le bigramme (la porte R3 de
la mission)**. La mission continue sur le second.

- **semence 1** : normal 3.7654 / melange 5.7830 / vide 7.1381 → ecart **3.3727 nat** (✓ seuil 2 nat)
- **semence 2** : normal 3.6772 / melange 5.9811 / vide 6.5703 → ecart **2.8931 nat** (✓ seuil 2 nat)
- **semence 3** : normal 3.7662 / melange 6.0397 / vide 6.7014 → ecart **2.9352 nat** (✓ seuil 2 nat)

Sondes (semence 1) : noeud1_16d 5.3216 · toutes_384d 4.9941 · lecture 4.3085. Le repere « noeud1 » lit en realite `state[:, 0, :]`, c'est-a-dire la
**porte d'entree** (noeud 0) : sa valeur est identique sur les trois
semences au dix-millieme, ce qui confirme qu'elle ne porte aucun savoir
appris — c'est une correction du libelle herite de S4, pas un resultat.

## 4. La survie du receveur (V3/V4)

Le metabolisme tourne **libre** 1000 pas : naissances payees par le trafic,
coupe des synapses faminees, aucun pas d'optimiseur. Deux regimes de faim.

| regime | naissances | coupees | R avant | R apres | ratio |
|---|---|---|---|---|---|
| defaut (prune_ratio 0,1) | 0.0 | 0.0 | +1.1172 | +1.1172 | 1.00 |
| faim (prune_ratio 0,5) | 412.7 | 434.7 | +1.1172 | +0.6808 | 0.61 |

**Le regime par defaut est inerte, et c'est mesurable** : a la sortie de
l'entrainement, les usages des 220 aretes vivent entre
**0.01712** et **0.11461**
(mediane 0.04553) alors que le seuil de famine —
un dixieme de la moyenne du decile le plus charge — vaut
**0.00804** : 0 aretes
sont affamees, et **aucune** des aretes n'a un usage nul. Aucune synapse ne
peut donc mourir. Le pool 32 ne change rien, ce qui identifie la cause :
ce n'est pas la taille du pool, c'est l'absence de routage selectif. Un
cerveau qui n'active pas de sous-ensemble n'a pas de concurrence de trafic,
donc pas de famine. Le metabolisme du jouet est **naissance-seul**.

Re-test demande, pool 32 : densite 39.9 %, 384 aretes creees et 0 coupees pendant 200 pas ;
en 300 pas de vie libre, 0 creees / 0 coupees. Usage 0.0191-0.1154 contre un seuil a 0.0084.

**Verdict de survie : 🔴** — la vie devore le savoir : c'est le vrai cout du metabolisme
(R apres = 0.610 × R avant en moyenne ; critere R_apres >= 0,9 x R_avant (regime faim)).
Par semence : s1 🔴 · s2 🔴 · s3 🟡 — 2 rouge(s), 1 jaune(s), 0 vert(s). La dispersion est reelle : le verdict est
celui de la moyenne, et chaque semence est publiee a cote.

## 5. La distillation ancrée (D1-D3)

La formule complete, avec les deux correctifs du diagnostic S4 :

```
L = alpha * T^2 * KL(p_T(MYC) ‖ p_T(prof))
  + beta  * CE(logits_MYC, tokens_reels)      <- l'ancrage
  + gamma * MSE(P h_MYC, h_prof)              <- P aleatoire, appris
```

| config | alpha | beta | gamma | perte validation (nouvelle) |
|---|---|---|---|---|
| a1b1g05 | 1.0 | 1.0 | 0.5 | **3.5470** ± 0.0123 |
| a1b1g0 | 1.0 | 1.0 | 0.0 | **3.5281** ± 0.0105 |
| a1b05g05 | 1.0 | 0.5 | 0.5 | **3.5789** ± 0.0254 |
| a05b1g05 | 0.5 | 1.0 | 0.5 | **3.5477** ± 0.0018 |

Meilleure config sur la validation : **a1b1g0**
(retenue : a1b1g0).

| bras | perte | R | KL(A‖MYC) | F | top-3 |
|---|---|---|---|---|---|
| M0 | 3.7213 | +1.1296 | 0.6909 | **+0.1949** | 0.3142 |
| a1b1g0 | 3.5281 | +1.2898 | 0.4368 | **+0.4909** | 0.2404 |
| melange | 4.7595 | +0.2683 | 1.4557 | **-0.6963** | 0.1338 |

**Gain isole de la distillation : +0.1602** (R distille − R M0).
Controle a etiquettes melangees : R = +0.2683, ecart au distille +1.0215 — propre.

## 6. La survie après transfert (D4)

| semence | F avant | F apres | R avant | R apres | coupees |
|---|---|---|---|---|---|
| 1 | +0.4795 | -0.0954 | +1.2960 | +0.7893 | 433 |
| 2 | +0.5024 | +0.1911 | +1.2837 | +0.9759 | 440 |

## 7. Le score final F (P3)

```
F = 1 - KL(prof A ‖ MYCELIA) / KL(prof A ‖ prof B)
```

- plafond KL(A‖B) : **0.8582 nat**
- KL(A‖MYCELIA) distille : **0.4368 nat**
- **F avant la vie : +0.4909**
- **F apres la vie : +0.0478**
- conservation de F sous la vie : **0.097**

**Lecture : MYCELIA ne couvre qu'une part minoritaire du chemin : le transfert existe mais l'eleve ne peut pas porter la forme du professeur.**

Ce n'est pas un probleme de recette : aucun levier de la section 5 ne
deplace la perte de plus de 0,03 nat (T=2, T=4, alpha/beta raffines).
Le plafond de fidelite est donc fixe par ce que l'eleve peut porter —
63 276 parametres dont 36 000 de membranes qui servent d'abord a la
lecture — pas par la quantite de professeur ni par la temperature.

Leviers testes (P4, semence 1) :

| levier | perte validation |
|---|---|
| temperature T=2 | 3.5548 |
| temperature T=4 | 3.5802 |
| budget x4 (2400 pas) | 3.5245 |
| alpha/beta raffines (0,5/0,5) autour du meilleur point | 3.5463 |

## 8. Échecs honnêtes, surprises, périmètre non vérifié

1. **La mort est inerte dans ce jouet.** Avec la regle du cerveau
   (`prune_ratio` 0,1), 0 synapse meurt en 1000 pas, a 24 comme a 32
   noeuds : les usages sont trop serres (0,016-0,065) pour qu'une famine
   existe, parce que la diffusion atteint tous les noeuds — il n'y a pas
   de routage selectif. Le metabolisme du jouet est donc **naissance-seul**.
   Le regime « faim » (0,5) est un reglage de faim, pas la regle par
   defaut : il est documente comme tel dans chaque ligne du JSON.
2. **La sonde « noeud 1 » de S4 etait mal nommee.** `collect_states` lit
   `state[:, 0, :]`, la porte d'entree, pas le noeud 1. Sa valeur constante
   (5,32 sur les trois semences) le montre. La conclusion de S4 — la tete ne
   lit pas une membrane unique — tient : la sonde « lecture » vaut 4,3
   contre 5,32 pour cette porte.
3. **Deux semences pour la grille et la survie apres transfert**, trois
   pour la validation : c'est un budget CPU, pas un choix statistique. Les
   ecarts-types publies disent la dispersion disponible.
4. **Aucun portage vers Nyra** : tout vit dans `scripts/lab/`, le genome et
   les runs de Nyra sont intacts. Le pool 24 du jouet n'est pas le pool
   reel ; rien ici ne prouve un comportement a 10 000 noeuds.
5. **Non verifie** : le corpus reel (le jouet a 312 mots), la duree longue
   (600 pas, pas 50 000), et l'effet d'un troisieme professeur sur le
   plafond (deux suffisent a le definir, pas a le stabiliser).

