# Distribution des degrés — le routage ne s'effondre pas

**8 novembre 2026.** Question : la règle « un message choisit sa destination » (F2)
produit-elle un graphe où les nœuds riches s'enrichissent (effondrement), ou un milieu
qui s'équilibre ? Réponse mesurée sur trois checkpoints réels : **équilibre proche d'un
Poisson, avec un très léger excès de variance**. Aucun effondrement.

## 1. Mesures

`python scripts/degree_stats.py data/genome_comparaison.pt data/genome_mort-pas1502.pt data/genome.pt`

| checkpoint | pas | arêtes | nœuds | degré entrant moyen | max | orphelins | Gini | part du 1 % le plus connecté | CV |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `genome_comparaison.pt` (texte) | 500 | 3 971 | 514 | 7,73 | 16 | 1 | **0,2173** | 2,0 % | **0,386** |
| `genome_mort-pas1502.pt` (toy) | 1 502 | 5 265 | 518 | 10,16 | 24 | 1 | **0,1953** | 2,1 % | **0,354** |
| `genome.pt` (toy v5) | 2 000 | 4 890 | 519 | 9,41 | 24 | 1 | **0,2236** | 2,3 % | **0,405** |

Histogramme du degré entrant (nombre de nœuds par tranche), texte au pas 500 :
`[0-5) 141`, `[5-10) 250`, `[10-15) 106`, `[15-20) 12`, `[20-30) 4`, `[30-50) 0`.

## 2. Ce que ces chiffres disent

Un graphe aléatoire de même taille a un degré entrant de Poisson, donc un coefficient de
variation `1/√λ` :

| checkpoint | λ mesuré | CV d'un Poisson | CV mesuré | écart |
|---|---:|---:|---:|---:|
| texte, pas 500 | 7,73 | 0,360 | 0,386 | +7 % |
| toy, pas 1 502 | 10,16 | 0,314 | 0,354 | +13 % |
| toy v5, pas 2 000 | 9,41 | 0,326 | 0,405 | +24 % |

- **Pas d'effondrement** : le nœud le plus connecté reçoit 16 à 24 arêtes, soit 2 à 3 fois
  la moyenne, jamais 100 fois. Le 1 % le plus connecté concentre 2,0 à 2,3 % des arêtes
  là où une distribution uniforme en donnerait 1 %.
- **Un léger excès de variance** apparaît avec l'âge du run (CV +7 % à 500 pas, +24 % à
  2 000) : le routage préfère un peu les adresses déjà fréquentées, sans jamais basculer
  en loi de puissance.
- **Un orphelin entrant** sur ~515 nœuds dans les trois cas : un nœud vivant que personne
  n'adresse. L'isolement est traité par le cycle de vie (grâce de 300 pas), pas par le
  routage.

## 3. Conséquence pour l'ablation

L'ablation de l'étape B compare la colonie grandie à un graphe aléatoire **de même
taille** (3 997 arêtes, 532 slots). Ce tableau dit que la colonie grandie n'est pas, à
500 pas, un graphe aléatoire « meilleur » : elle en est très proche en degrés. Si l'arm
figé fait aussi bien, ce ne sera donc pas parce que la topologie grandi est
pathologique — ce sera parce que la croissance n'a pas encore eu le temps de la
distinguer. C'est une information dont le verdict de l'ablation a besoin.

## 4. Limite de la mesure

Trois checkpoints, deux régimes (texte et toy), un seul run par régime : ce n'est pas une
loi, c'est un relevé. La question « la distribution se concentre-t-elle avec le temps ? »
demanderait des checkpoints intermédiaires du **même** run, qui n'ont pas été conservés
(seul le dernier et un état de mort accidentelle l'ont été).
