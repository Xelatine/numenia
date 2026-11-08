# L'ablation centrale — la croissance sert-elle ? Protocole, et où elle en est

**8 novembre 2026.** Question posée : une colonie qui aurait démarré avec le graphe final
(3 997 arêtes, ~514 nœuds), câblé **au hasard** et **figé**, aurait-elle fait aussi bien ?
Si oui, les naissances ne sont qu'une décoration coûteuse, et c'est le résultat le plus
important du projet — dans ce sens-là. Ce document fixe le protocole et l'état exact de
l'exécution.

## 1. Ce qui est codé, et comment

- `mycelia/ablation.py::freeze_random_graph` : remplace le câblage par un graphe aléatoire
  sparse de `edges` synapses, conserve le chemin de départ `0 -> 2..7 -> 1` (sinon la
  porte et le lecteur sont déconnectés et l'arm mesure un cerveau muet), garantit ≥ 1
  arête entrante par nœud, puis coupe `max_births`, `max_node_births` et
  `allow_node_growth`.
- Drapeau de configuration : `growth.mode: grown | frozen_random` et
  `growth.frozen_edges` (`configs/config.yaml`, défauts dans `mycelia/config.py`). Le
  gel n'est appliqué qu'à un cerveau **neuf** : un run repris garde son propre câblage.
- `scripts/ablate_growth.py --arm grown|frozen` : deux arms, même graine, même corpus,
  même `seq 32`, même `batch 8`, même budget de pas, même générateur de données. Chaque
  arm écrit sa courbe (`logs_ablation_*.txt`), son JSON et sa perplexité de validation.

## 2. Validation du code (faite)

Un cerveau jouet de 128 nœuds avec `growth.frozen_edges = 300` :

```
arêtes 300 | max_births 0 | allow_node_growth False | births 0 | enable_growth False
3 pas ok | perte 4.6054 | arêtes finales 300 | nœuds 128
```

Le graphe ne bouge pas d'une arête en trois pas, et la croissance est bien désactivée :
le drapeau fait ce qu'il annonce.

## 3. Ce qui n'est pas encore mesuré, et pourquoi

La règle de carte a été **révisée le 8 novembre** : un run est autorisé si (a) aucun
processus étranger ne calcule à plus de 5 % et (b) la VRAM libre couvre le besoin du run
plus 1,5 Go de marge. Besoin de l'arm grandi : **4 908 Mo réservés mesurés** au pas 500 du
run comparatif (`GPU 85/4908 Mo`), arrondi à 5 000 Mo ; marge 1 500 Mo ; seuil **6 500 Mo
libres**. Relevés de la session :

```
1 306 MiB utilisés, 18 % d'utilisation   (début de session, ancienne règle)
1 297 MiB utilisés, 13 % d'utilisation   (après les bancs CPU)
1 804 MiB utilisés, 6 224 MiB libres, 25 % d'utilisation, 29 processus étrangers  (lancement tenté)
```

Verdict mesuré du lancement : **(a) faux**, 25 % d'utilisation avec 29 processus étrangers
(navigateurs, Discord, Steam, overlays) ; **(b) faux**, 6 224 Mo libres pour 6 500 requis,
soit **276 Mo manquants**. Aucun processus n'a été tué. Les deux arms restent à lancer :

```bash
python scripts/ablate_growth.py --arm grown  --steps 500
python scripts/ablate_growth.py --arm frozen --steps 500
```

Durée attendue, **estimée** d'après les 1 464 ms/pas du banc du 6 novembre à 2 200 arêtes
(carte libre) : l'arm figé a 3 997 arêtes, donc ~2 fois plus de travail par pas, soit
~3 s/pas → **25 min par arm**. Cette estimation n'est pas une mesure et ne doit pas être
présentée comme telle.

Ce qu'il faut pour débloquer, chiffré : libérer **≥ 300 Mo de VRAM** et faire tomber
l'utilisation sous 5 % (fermer un navigateur à accélération matérielle suffit en général),
puis relancer la mesure de condition. **Si l'écart entre les deux arms est ≤ 2 %**, la
règle fixée d'avance impose de rejouer les deux à 2 000 pas avant de conclure — soit
~2 h par arm **[estimé]**, et une seconde session de carte libre.

## 4. La règle de verdict, fixée d'avance

- si l'arm **figé** atteint une perplexité de validation ≤ celle de l'arm grandi (± 2 %),
  alors **la croissance n'a pas encore payé à ce budget** : c'est un résultat, et il sera
  écrit tel quel ;
- si l'arm **grandi** est meilleur de plus de 2 %, la croissance a trouvé une topologie
  utile, et l'écart est la mesure de son apport ;
- dans les deux cas, le budget de 500 pas est **court** : la croissance a besoin de la
  boucle naissance/élagage pour explorer, et l'arm figé n'a, par construction, aucun
  mécanisme d'exploration. Un verdict négatif à 500 pas ne condamnerait donc pas la
  croissance — il dirait qu'elle ne s'est pas encore distinguée.

Les deux courbes seront publiées côte à côte, sans lissage et sans moyenne mobile.

## 5. Règle de carte v2 (décision humaine du 8 novembre, soir)

La règle v1 (< 1 Go utilisé, < 10 % d'utilisation) confondait deux besoins différents et
bloquait la mesure principale pour 276 Mo. Elle est remplacée par deux classes :

**Classe « résultat »** — ablation, entraînement, courbes de perte, perplexités. Exige
seulement **VRAM libre ≥ besoin du run + 1 000 Mo**. L'utilisation en % est **sans objet** :
un navigateur qui dessine à 20 % ne change pas une perte. Les deux arms d'une ablation
tournent dans la **même session, même carte, même état**, sans autre charge.

**Classe « chronométrage »** — ms/pas, ETA, bancs. Garde la règle stricte (< 1 Go utilisé,
< 10 % d'utilisation) : c'est la leçon du 7 novembre, un débit mesuré sur une carte
partagée n'est pas un débit.

Justification : le 7 novembre, la contamination avait faussé des **temps** (3 832 → 10 755
ms/pas pour la même colonie) et provoqué des OOM ; elle n'a jamais changé une **perte**, qui
ne dépend que du graphe, des données et de la graine. Bloquer une courbe de perte au nom
d'un chronométrage sale, c'est payer deux fois.

Mesure de condition pour cette session (classe résultat, seuil 6 000 Mo) :

```
1 434 MiB utilisés, 6 585 MiB libres, 16 %  -> libre 6 587 Mo >= 6 000 : AUTORISÉ
```
## 6. Résultat mesuré : la croissance a payé à 500 pas (cas A)

**Mesuré le 8 novembre 2026.** Deux arms dans la **même session**, même carte, même graine
(42), `seq 32`, `batch 8`, 500 pas chacun, condition prise avant lancement sous la règle v2
(1 434 MiB utilisés, 6 585 MiB libres → autorisé). Relevés de carte consignés dans
`ablation_conditions.txt` : 1 450/6 569 au départ, 7 913/106 à la bascule entre les arms,
1 661/6 358 à la fin. Journaux : `data/per_step_grown.csv`, `data/per_step_frozen.csv`,
500 lignes chacun.

| arm | perte au pas 1 | perte au pas 500 | val ppl | arêtes | nœuds | durée |
|---|---:|---:|---:|---:|---:|---:|
| grandi | 9,1964 | 7,6172 | **1 943,301** | 3 983 | 514 | 3 051 s |
| figé aléatoire | 9,1955 | 7,6736 | **2 071,823** | 3 997 | 532 | 3 712 s |

Écart de perplexité de validation : **+6,61 % en faveur du grandi**, soit **0,0564 nat** de
perte par pas. La règle du §4 (fixée avant les runs) donne donc le **cas A** : nous sommes
au-dessus des 2 %. Le rejeu à 2 000 pas du cas B **n'est pas déclenché**.

### 6.1 Les deux courbes, sans lissage

| pas | grandi perte | grandi arêtes | figé perte | figé arêtes | écart |
|---:|---:|---:|---:|---:|---:|
| 50 | 9,1856 | 413 | 9,1482 | 3 997 | +0,0374 (figé devant) |
| 100 | 9,0859 | 812 | 9,0689 | 3 997 | +0,0170 (figé devant) |
| 150 | 8,9880 | 1 212 | 8,9849 | 3 997 | +0,0031 (figé devant) |
| 200 | 8,8547 | 1 612 | 8,8587 | 3 997 | −0,0040 |
| 250 | 8,6679 | 2 011 | 8,6772 | 3 997 | −0,0093 |
| 300 | 8,5017 | 2 411 | 8,5303 | 3 997 | −0,0286 |
| 350 | 8,3188 | 2 810 | 8,3545 | 3 997 | −0,0357 |
| 400 | 8,0656 | 3 210 | 8,1144 | 3 997 | −0,0488 |
| 450 | 7,7753 | 3 593 | 7,8320 | 3 997 | −0,0567 |
| 500 | 7,6172 | 3 993 | 7,6736 | 3 997 | −0,0564 |

Lecture : le figé mène jusqu'au pas 150 — normal, il démarre avec **3 997 arêtes quand le
grandi en a 20**. Le croisement tombe entre 150 et 200 et l'écart se creuse ensuite de
façon presque monotone, jusqu'à se stabiliser à −0,056 nat sur la dernière fenêtre.
L'avantage arrive avec la densité : ce sont les quelque 1 400 dernières arêtes nées qui le
portent, pas les premières.

### 6.2 Ce qui explique l'écart, et ce qui ne l'explique pas

**Ce n'est pas la capacité.** Le figé a plus d'arêtes (3 997 contre 20 → 3 993) et plus de
nœuds vivants (532 contre 514) **à tous les pas**. Tout avantage de taille joue contre le
grandi : l'écart est obtenu malgré ce handicap, ce qui le renforce.

**Ce n'est pas un hub.** Degré entrant moyen 7,73 contre 7,51 ; Gini 0,2173 contre 0,1946 ;
maximum 16 contre 19. La distribution du grandi est un peu plus inégale, mais le figé n'est
pas assez concentré pour expliquer 6,6 % en sa défaveur. (Le génome du pas 500 vient du run
jumeau : l'arm n'a pas sauvegardé le sien, voir §6.3.)

Deux candidats restent, que cette expérience **ne sépare pas** :

- **(a) le routage** — une arête naît là où un message est passé et meurt quand elle ne
  sert plus (17 arêtes coupées, 31 nœuds enterrés, 31 ressuscités sur 500 pas). Le graphe
  aléatoire tire ses arêtes uniformément : une partie d'entre elles route du bruit.
- **(b) le programme de capacité** — 20 arêtes au pas 1 → 3 993 au pas 500 est une montée
  en complexité progressive, qui peut aider l'optimisation au début (moins d'interférence
  entre paramètres).

Pour les séparer, il faut un troisième arm : **même calendrier d'arêtes, identités tirées
au hasard et re-tirées** à chaque pas — croissance sans routage. [estimé] ~1 h de calcul,
même règle de carte. C'est la prochaine ablation proposée, pas une conclusion.

### 6.3 Ce que ce résultat ne dit pas

- **500 pas seulement.** Les deux courbes sont encore en pente raide (les fenêtres 401–450
  et 451–500 baissent encore de 0,29 et 0,16 nat) : aucun des deux arms n'a convergé, un
  croisement ultérieur n'est pas exclu.
- **Variance run-to-run, première mesure du projet** : deux runs *grandi* indépendants
  (comparaison du 8 novembre et cet arm) donnent 1 977,843 et 1 943,301 → **1,75 %**
  d'écart de perplexité, soit 0,018 nat. L'écart mesuré (6,61 %) vaut ~3,7× cette
  dispersion — mais elle n'est estimée que sur **deux** runs, et l'arm figé n'a **aucune**
  réplique. Signal, pas loi. Une part de cet écart vient d'ailleurs d'un détail de code :
  le run de comparaison appelait `speak()` à chaque rapport, qui tirait dans le RNG global
  et décalait les naissances ; l'arm d'ablation ne génère aucun échantillon.
- **Le génome du grandi n'a pas été sauvegardé** par `ablate_growth.py` : la distribution
  des degrés vient du run jumeau (même protocole). À corriger avant la prochaine ablation.
- Le graphe figé a été reconstruit pour l'analyse avec `freeze_random_graph(edges=3997,
  seed=42)` : même fonction, même graine que l'arm, donc même graphe — vérifié, 3 997
  arêtes, et le journal montre 3 997 arêtes à chaque pas avec **0 naissance**.

### 6.4 Coût

Grandi **3 051 s** (6,1 s/pas en moyenne), figé **3 712 s** (7,4 s/pas), pour le même
budget de pas : la croissance coûte **18 % de temps en moins**, parce qu'elle n'entretient
pas 3 997 arêtes dès le premier pas. Le coût par pas en fin de course est le même (~9 s à
~4 000 arêtes).

### 6.5 Ce que ça change à la feuille de route

Le run texte **3 000 pas** devient la priorité suivante. ETA [estimé] : 6,1 s/pas → **5,1 h** ;
si le graphe continue de grossir (plafond 20 000 arêtes), 9 s/pas → **7,5 h**. À convertir
en mesure par un banc de 50 pas **en classe chronométrage** (carte quasi vide) avant
lancement — sinon l'ETA reste une estimation.
