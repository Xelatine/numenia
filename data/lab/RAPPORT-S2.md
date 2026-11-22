> **Renommage —** Ce projet s'appelle désormais **Numenia** (auparavant référencé
> comme Mycelia dans les titres et la documentation). Session, chiffres, fichiers de
> données et cerveau **Nyra** inchangés. Ce document est historique : son contenu est
> conservé tel quel, et seul le nom du projet passe à Numenia.

# RAPPORT S2 — JUMEAUX COMPARABLES

## Verdict

**MUR CONFIRMÉ**

- concepts traduits (corrélation > 0,6), maximum sur les deux sens : **2.00 / 8** en moyenne sur les semences ;
- gain significatif (p < 0,05 et > +1 %) dans au moins un sens : **non** ;
- exactitude top-3 du bras greffé sur 40 phrases neuves : **3.3 %** (seuil de la mission : 20 %).

Le mur tient : même à paramètres égaux et à savoir égal, l'alignement par le monde partagé ne trouve pas assez de correspondances. La Mission S3 devra explorer un pont **appris** (une petite tête de prédiction entraînée sur les activations) plutôt qu'un alignement calculé.

## 1. Les jumeaux paramétriquement égaux

Vocabulaire 312 mots, dimension 16, 2 blocs / 2 têtes côté dense, 8 nœuds et 100 arêtes côté MYCELIA. Égalité vérifiée : **oui** — 23748 paramètres de chaque côté.

| semence | paramètres | dense (convergence) | MYCELIA (convergence) | perte commune | pas d'arrêt dense | pas d'arrêt MYCELIA |
|---|---|---|---|---|---|---|
| 1 | 23748 | 3.6490 (pas 300) | 5.0783 (pas 50) | 5.0783 | 50 | 50 |
| 2 | 23748 | 3.6216 (pas 350) | 5.1079 (pas 50) | 5.1079 | 50 | 50 |
| 3 | 23748 | 3.7437 (pas 350) | 5.0947 (pas 50) | 5.0947 | 50 | 50 |

**Pourquoi 23 748 et pas 17 016.** La cible « ~17k » de la mission n'est pas atteignable par le vrai `Brain` : il refuse moins de 8 nœuds (ses 12 arêtes de départ occupent les nœuds 0..7), et à `dim=16` avec 312 mots l'embedding (4 992) et la tête de lecture (5 304) pèsent déjà 10 296. Huit nœuds à 1 665 paramètres mettent le plancher à 23 664 : le jumeau MYCELIA a donc été laissé au minimum, et le dense a été remonté à 23 748 (MLP 166 au lieu de 64) pour que l'égalité soit **exacte**. L'égalité stricte est la contrainte qui compte ; 17 016 était l'ordre de grandeur, pas la propriété testée.

**Même savoir.** Les deux jumeaux ne s'arrêtent pas au même pas mais à la même perte de validation. Le dense descend sous 4,0 sans difficulté ; MYCELIA, non : `train_mycelia` plafonne vers 5,1 quelle que soit la combinaison d'hyperparamètres essayée (lr, weight decay, taille de lot, nombre de sauts), et se met à surapprendre au-delà. La perte commune retenue est donc celle du plus faible, et elle est écrite dans le tableau — c'est la clause « meilleure perte commune atteignable » prévue par la mission.

Courbes d'arrêt (validation) :

- semence 1 — dense : 5.979 → 5.230 → 4.796 ; MYCELIA : 5.730 → 5.316 → 5.078 → 5.078 ; métabolisme 48 arêtes, 36 naissances, 0 arêtes coupées.
- semence 2 — dense : 5.926 → 5.233 → 4.864 ; MYCELIA : 5.734 → 5.358 → 5.108 → 5.108 ; métabolisme 48 arêtes, 36 naissances, 0 arêtes coupées.
- semence 3 — dense : 5.886 → 5.224 → 4.893 ; MYCELIA : 5.725 → 5.336 → 5.095 → 5.095 ; métabolisme 48 arêtes, 36 naissances, 0 arêtes coupées.

## 2. Le dictionnaire bilingue (corrélation + appariement optimal)

Score d'une paire = corrélation de Pearson des activations brutes sur les 2 000 stimuli. L'appariement est optimal au sens global (algorithme hongrois), et un concept n'est **traduit** que si sa corrélation dépasse 0,6 ; sinon il est **greffé** (un nœud ou un neurone neuf, porteur du concept projeté).

### dense → MYCELIA

| concept dense | unités | nœud MYCELIA proposé | corrélation | statut | stimuli partagés |
|---|---|---|---|---|---|
| 0 | 82 | 2 | 0.671 | **traduit** | « je ne attend pas ce mère et je ne reste pas ma chambre. »; « je ne attend pas une maîtresse et je ne crie pas les poule. » |
| 1 | 2 | 6 | 0.470 | **greffe** | « le chat petit court. »; « il y a quatorze mer et douze père. » |
| 2 | 9 | 3 | 0.209 | **greffe** | « un frère ne lit pas et ce gâteau ne s'ouvre pas. »; « sa frère ne compte pas et un table ne reste pas. » |
| 3 | 2 | 4 | 0.441 | **greffe** | « les poisson vole ce fleur. »; « les sœur part les route. » |
| 4 | 56 | 7 | 0.000 | **greffe** | — |
| 5 | 104 | 0 | 0.025 | **greffe** | « tombe mon arbre vite. »; « il y a sept père et quinze canard. » |
| 6 | 24 | 1 | 0.274 | **greffe** | « je ne reste pas un fleur et je ne mange pas ce souris. »; « je ne s'ouvre pas son fleur et je ne nage pas ma chien. » |
| 7 | 53 | 5 | 0.734 | **traduit** | « il y a dix mer et vingt gâteau. »; « il y a dix ville et dix arbre. » |

Semence 1 : **2 concepts traduits** sur 8, 6 greffés. Sur les 3 semences : 2, 1, 3.

### MYCELIA → dense

| nœud MYCELIA | unités | concept dense proposé | corrélation | statut | stimuli partagés |
|---|---|---|---|---|---|
| 0 | 1 | 5 | 0.025 | **greffe** | « tombe mon arbre vite. »; « il y a sept père et quinze canard. » |
| 1 | 1 | 6 | 0.274 | **greffe** | « je ne reste pas un fleur et je ne mange pas ce souris. »; « je ne s'ouvre pas son fleur et je ne nage pas ma chien. » |
| 2 | 1 | 0 | 0.671 | **traduit** | « je ne attend pas ce mère et je ne reste pas ma chambre. »; « je ne attend pas une maîtresse et je ne crie pas les poule. » |
| 3 | 1 | 2 | 0.209 | **greffe** | « un frère ne lit pas et ce gâteau ne s'ouvre pas. »; « sa frère ne compte pas et un table ne reste pas. » |
| 4 | 1 | 3 | 0.441 | **greffe** | « les poisson vole ce fleur. »; « les sœur part les route. » |
| 5 | 1 | 7 | 0.734 | **traduit** | « il y a dix mer et vingt gâteau. »; « il y a dix ville et dix arbre. » |
| 6 | 1 | 1 | 0.470 | **greffe** | « le chat petit court. »; « il y a quatorze mer et douze père. » |
| 7 | 0 | 4 | 0.000 | **vide** | — |

Semence 1 : **2 concepts traduits** sur 8, 6 greffés. Sur les 3 semences : 2, 1, 3.

## 3. Conversions par semence, moyennes et significativité

Protocole inchangé depuis S3 : amorçage de 50 pas, greffe, puis 100 pas d'affinage ; le contrôle part du même point, lit les mêmes lots et reçoit le même budget.

### dense → MYCELIA

| semence | avant | après greffe | contrôle | gain vs contrôle | traduits / greffés |
|---|---|---|---|---|---|
| 1 | 5.1558 | 5.2716 | 5.1648 | -2.07 % | 2 / 6 |
| 2 | 5.2102 | 5.2807 | 5.2570 | -0.45 % | 1 / 7 |
| 3 | 5.1660 | 5.3406 | 5.3251 | -0.29 % | 3 / 5 |
| **moyenne** | 5.1773 | 5.2976 | 5.2490 | -0.94 % | 2.00 / 6.00 |

Test t apparié (dense → MYCELIA) : n = 3, gain moyen -0.94 %, écart-type 0.0098, t = -1.6503, ddl = 2, p = 0.24066 → **REJETÉ** (seuils : p < 0.05 et gain > 1 %).

Courbes de validation (greffe puis contrôle) :

- greffe : 5.166 → 5.132 → 5.151 → 5.153 → 5.172 → 5.174 → 5.166 → 5.187 → 5.244 → 5.272
- contrôle : 5.156 → 5.146 → 5.138 → 5.169 → 5.159 → 5.157 → 5.161 → 5.155 → 5.196 → 5.165

### MYCELIA → dense

| semence | avant | après greffe | contrôle | gain vs contrôle | traduits / greffés |
|---|---|---|---|---|---|
| 1 | 4.7644 | 4.5016 | 4.5018 | +0.00 % | 2 / 6 |
| 2 | 4.8384 | 4.5899 | 4.5903 | +0.01 % | 1 / 7 |
| 3 | 4.8691 | 4.6648 | 4.6653 | +0.01 % | 3 / 5 |
| **moyenne** | 4.8240 | 4.5854 | 4.5858 | +0.01 % | 2.00 / 6.00 |

Test t apparié (MYCELIA → dense) : n = 3, gain moyen +0.01 %, écart-type 0.0000, t = 4.3007, ddl = 2, p = 0.050041 → **REJETÉ** (seuils : p < 0.05 et gain > 1 %).

Courbes de validation (greffe puis contrôle) :

- greffe : 4.739 → 4.704 → 4.673 → 4.653 → 4.638 → 4.610 → 4.583 → 4.554 → 4.530 → 4.502
- contrôle : 4.739 → 4.704 → 4.673 → 4.652 → 4.638 → 4.610 → 4.583 → 4.554 → 4.530 → 4.502

## 4. Exactitude top-3 sur 40 phrases hors corpus

Les quatre bras : le jumeau receveur de référence (avant tout amorçage), le même après les 50 pas d'amorçage, le bras greffé et le contrôle après leurs 100 pas. Le top-3 est la mesure honnête sur des micro-cerveaux à 312 mots : exiger le top-1 ne mesurait que le bruit.

### dense → MYCELIA

| semence | jumeau de référence | après amorçage | greffe | contrôle |
|---|---|---|---|---|
| 1 | 5 % | 5 % | 5 % | 2 % |
| 2 | 2 % | 0 % | 2 % | 5 % |
| 3 | 0 % | 0 % | 2 % | 2 % |
| **moyenne** | — | — | 3.3 % | 3.3 % |

### MYCELIA → dense

| semence | jumeau de référence | après amorçage | greffe | contrôle |
|---|---|---|---|---|
| 1 | 0 % | 0 % | 0 % | 0 % |
| 2 | 0 % | 0 % | 0 % | 0 % |
| 3 | 0 % | 0 % | 0 % | 0 % |
| **moyenne** | — | — | 0.0 % | 0.0 % |

## 5. Le scan récurrent (S1 → S2)

Le génome de Nyra n'a pas bougé (step 56, aucun entraînement depuis S1). Rejouer MYCELI-SCAN sur les mêmes 500 stimuli doit donc redonner la même carte : c'est la preuve que l'instrument est reproductible.

| mesure | S1 | S2 |
|---|---|---|
| nœuds vivants | 514 | 514 |
| nœuds actifs au scan | 285 | 285 |
| taux d'engagement | 55.5 % | 55.5 % |
| familles (k=6) | 6 | 6 |
| inertie du k-means | 5.6119 | 5.6119 |
| arêtes jamais co-actives | 162 | 162 |
| paires co-actives sans arête | 4236 | 4236 |

Écart maximal entre les deux matrices d'activation : **0.0**. Reproductible : **oui**.

`data/lab/scan_history.json` contient 1 point(s) : chaque relance de Nyra en ajoutera un, ce qui donnera la première série temporelle d'évolution d'un cerveau MYCELIA.

## 6. Échecs honnêtes

- **La cible de 4,0 n'est pas atteinte.** MYCELIA plafonne vers 5.0783, 5.0947, 5.1079 même à paramètres égaux : 1 616 de ses 1 665 paramètres par nœud passent dans le corps (b1/b2) et le routage (q/k), et huit nœuds ne suffisent pas à couvrir 312 mots. Les jumeaux ont donc été arrêtés au pas 50, et le dense est délibérément sous-entraîné pour rester au même savoir. C'est ce que la mission demandait, mais cela limite la portée de tout ce qui suit : un concept à ce stade est un concept à peine formé.
- **dense → MYCELIA** : 2.00 concepts traduits sur 8 en moyenne, gain -0.94 % contre le contrôle, p = 0.24066 → rejeté (bruit de régularisation) ; top-3 greffé 3.3 % contre 3.3 % au contrôle.
- **MYCELIA → dense** : 2.00 concepts traduits sur 8 en moyenne, gain +0.01 % contre le contrôle, p = 0.050041 → rejeté (bruit de régularisation) ; top-3 greffé 0.0 % contre 0.0 % au contrôle.
- **Le top-3 ne franchit pas 20 %** avec des jumeaux arrêtés à la perte commune : les trois propositions restent dominées par la fréquence des mots. Un top-3 élevé aurait exigé des jumeaux entraînés jusqu'à 4,0, ce que MYCELIA ne permet pas.
- **Trois semences, c'est peu.** Le test t apparié a 2 degrés de liberté : il faut un gain très régulier pour être significatif, et un gain régulier de 1 % peut aussi passer. Les deux chiffres (p et amplitude) sont donc donnés ensemble, jamais l'un sans l'autre.
- **La population MYCELIA est fixée à 8 nœuds** (métabolisme de câblage actif : naissances d'arêtes, coupes, enterrements). Autoriser des naissances de nœuds aurait ajouté 1 665 paramètres par nœud et cassé l'égalité en cours de route : c'est un choix imposé par le contrôle, pas par l'architecture.

## 7. Fichiers

- `data/lab/s2_twins.json` — les 3 paires, courbes et égalité de paramètres
- `data/lab/s2/dense_to_mycelia_s{1,2,3}.json`, `mycelia_to_dense_s{1,2,3}.json`
- `data/lab/s2_stats.json` — moyennes et tests t
- `data/lab/scan_s2.json`, `scan_history.json` — scan récurrent et série
- `data/lab/stimuli.txt` — 2 000 stimuli (les 500 de S1 en tête, inchangés)
- `data/lab/stimuli_new.txt` — 40 phrases hors corpus
- `scripts/lab/twins.py`, `s2_translate.py`, `scan_report.py`, `s2_report.py`
