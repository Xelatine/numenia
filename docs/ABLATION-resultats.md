# Ablation croissance contre graphe figé — résultats bruts

**8 novembre 2026.** Deux arms, même corpus, même graine (42), `seq 32`, `batch 8`,
500 pas chacun, même carte, même session. Journal par pas :
`data/per_step_grown.csv` et `data/per_step_frozen.csv` (500 lignes chacun).

## 1. Verdict (règle fixée d'avance, `docs/ABLATION-croissance.md` §5)

**LA CROISSANCE A PAYÉ (écart > 2 % en faveur du grandi)** — cas **A**.

| arm | perte initiale | perte finale | validation ppl | arêtes | nœuds | durée |
|---|---:|---:|---:|---:|---:|---:|
| grandi | 9.1964 | 7.6172 | **1943.301** | 3983 | 514 | 3 051 s |
| figé aléatoire | 9.1955 | 7.6736 | **2071.823** | 3997 | 532 | 3 712 s |

Écart de perplexité de validation : **+6.61%**. En pertes (nats) : +0.0564 nat par pas de la part du figé, soit la même différence vue en unités lisibles.

## 2. Courbes, pas par pas

| pas | grandi perte | grandi arêtes | figé perte | figé arêtes |
|---:|---:|---:|---:|---:|
| 50 | 9.1856 | 413 | 9.1482 | 3997 |
| 100 | 9.0859 | 812 | 9.0689 | 3997 |
| 150 | 8.9880 | 1212 | 8.9849 | 3997 |
| 200 | 8.8547 | 1612 | 8.8587 | 3997 |
| 250 | 8.6679 | 2011 | 8.6772 | 3997 |
| 300 | 8.5017 | 2411 | 8.5303 | 3997 |
| 350 | 8.3188 | 2810 | 8.3545 | 3997 |
| 400 | 8.0656 | 3210 | 8.1144 | 3997 |
| 450 | 7.7753 | 3593 | 7.8320 | 3997 |
| 500 | 7.6172 | 3993 | 7.6736 | 3997 |

## 3. Distribution des degrés entrants

| arm | degré entrant moyen | max | Gini | orphelins |
|---|---:|---:|---:|---:|
| grandi (`data/genome_comparaison.pt`, pas 500) | 7.73 | 16 | 0.2173 | 1 |
| figé (reconstruit, graine 42, 3997 arêtes) | 7.51 | 19 | 0.1946 | 0 |

Le génome du pas 500 vient du run jumeau (`COMPARAISON-500`) : l'arm d'ablation
n'a pas sauvegardé le sien. Même protocole, écart de reproductibilité 1,75 %.

## 4. Vie du graphe lue dans `per_step.csv`

- **figé** : arêtes constantes à [3997] sur les 500 pas, 0 naissance cumulée → le drapeau tient, et l'arm est bien exemptée de la propriété E\* ≈ naissances/pas × 1 000.
- **grandi** (compteurs cumulés au pas 500) : 3998 naissances d'arêtes, 17 morts d'arêtes, 31 morts de nœuds, 31 résurrections, cimetière 0, 3993 arêtes et 514 nœuds vivants.
- naissances par pas : **8.0 à 8.0** sur les dix fenêtres — la colonie ne naît pas *parce que* la perte stagne, elle naît huit fois par pas par construction (`max_births 8`). Ce n'est pas un mécanisme adaptatif, c'est une montée en capacité régulière.

### Pente de perte et taille du graphe, par fenêtre de 50 pas (grandi)

| fenêtre | Δ perte | naissances/pas | arêtes moyennes |
|---|---:|---:|---:|
| 1–50 | -0.0108 | 8.0 | 221 |
| 51–100 | -0.0996 | 8.0 | 617 |
| 101–150 | -0.0979 | 8.0 | 1016 |
| 151–200 | -0.1333 | 8.0 | 1416 |
| 201–250 | -0.1867 | 8.0 | 1815 |
| 251–300 | -0.1662 | 8.0 | 2215 |
| 301–350 | -0.1829 | 8.0 | 2614 |
| 351–400 | -0.2532 | 8.0 | 3014 |
| 401–450 | -0.2903 | 8.0 | 3398 |
| 451–500 | -0.1581 | 8.0 | 3797 |

Corrélation Δ perte / arêtes moyennes entre fenêtres : **-0.82** (négative = les fenêtres où le graphe est le plus gros sont celles où la perte baisse le plus ; n = 10, descriptif, pas une preuve).
## 5. Coût

Grandi 3 051 s, figé 3 712 s pour le même budget de pas : la croissance n'a pas coûté de temps, elle en a économisé (18 %), parce qu'elle n'entretient pas 3 997 arêtes dès le premier pas. Le coût par pas en fin de course est le même (~9 s à ~4 000 arêtes).

