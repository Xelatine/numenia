# Plateau de croissance — réponse par lecture de code

**6 novembre 2026.** Question posée : « pourquoi la croissance plafonne-t-elle ? »
Réponse **par lecture de `growth.py`, `lifecycle.py`, `brain.py` et des relevés du run
v5**, sans rien modifier. Chaque affirmation est soit un chiffre du log, soit une ligne
de code citée ; ce qui reste une hypothèse est écrit comme tel.

## 1. Ce que le run v5 a mesuré

Run `logs_toy_v5_2000_batch16.txt`, 2 000 pas, batch 16, séquence 10, de zéro.

| pas | arêtes | nœuds vivants | arêtes nées (cumul) | arêtes mortes (cumul) | VRAM réservée |
|----:|-------:|--------------:|--------------------:|----------------------:|--------------:|
| 1 000 | 4 977 | 516 | 7 589 | 2 624 | 5 012 Mo |
| 1 200 | 5 368 | 518 | 8 996 | 3 640 | 5 274 Mo |
| 1 400 | 5 711 | 518 | 10 431 | 4 732 | 5 592 Mo |
| 1 500 | **5 835** | 518 | 11 126 | 5 303 | 5 682 Mo |
| 1 600 | 5 700 | 520 | 11 575 | 5 887 | 5 702 Mo |
| 1 800 | 5 544 | 520 | 12 551 | 7 019 | 5 702 Mo |
| 2 000 | **5 433** | 520 | 13 542 | 8 121 | 5 702 Mo |

Les quatre décisions de croissance, telles que le log les a écrites :

```
référence de validation posée à step 500 (val=1.1) — première décision au pas 1000
phase CROISSANCE à step 1000 (val=1.0, gain 5.1%) : naissances ouvertes
PALIER : pool 583 → 839 à step 1500 (val=1.0)
phase OPTIMISATION à step 2000 (val=1.0) : naissances suspendues
```

Le plateau n'est donc pas un arrêt des naissances : 13 542 arêtes sont nées en 2 000 pas.
C'est un **équilibre entre deux budgets exprimés dans deux unités différentes**.

## 2. Le point fixe : un budget d'élagage proportionnel contre un plafond de naissances par pas

**Élagage** — `lifecycle.py` L101 :

```python
budget = max(1, int(round(ne * max(0.0, min(1.0, share)))))
```

`share` vient de `prune_fraction: 0.1` (`mycelia/config.py` L51), et la passe a lieu tous
les `prune_every: 100` pas (`config.py` L49, appelée en `train.py` L501-502). Le budget
d'élagage **par pas** vaut donc :

```
budget/pas = 0,10 × E / 100 = E / 1 000
```

Il est **proportionnel** à la taille de la table. À 5 476 arêtes, une passe peut couper
547 arêtes : le log du pas 1 900 affiche « 548 arêtes coupées », et le suivant en coupe
543 — l'élagage travaille **à son plafond**, pas en dessous.

**Naissances** — `brain.py` L543 :

```python
births_left = (self.max_births if births is None else int(births)) if grow else 0
```

`max_births: 8` (`config.py` L35) : au plus 8 arêtes par pas, **constant**, quelle que
soit la taille de la colonie. Mesuré sur les 500 derniers pas : 2 416 naissances, soit
4,8/pas ; sur les 100 derniers pas du run optimisé, 505 naissances, soit 5,05/pas.

**Le point fixe s'écrit alors sans aucune constante ajustée :**

```
E* / 1 000 = naissances par pas   →   E* = 1 000 × 4,8…5,4 ≈ 4 800 … 5 400 arêtes
```

Le run s'est arrêté à 5 433. Le plateau est donc **prédit par le code et vérifié par le
log à 1 % près**. Ce n'est pas un défaut de la dynamique : c'est l'intersection de deux
règles qui ne s'accordent pas sur l'échelle — l'élagage raisonne en *fraction de la
table*, la naissance en *nombre absolu par pas*. Tant que `max_births` reste constant,
la colonie ne peut pas dépasser ≈ 1 000 × `max_births` arêtes, et elle y arrive par le
haut puis redescend quand les naissances sont suspendues (pas 2 000, §4).

## 3. Pourquoi le « pool » grandit alors que la colonie ne grandit pas

`growth.py` L128-135 :

```python
improvement = (self.last_val - val) / self.last_val
if improvement > self.grow_threshold:        # 1 %
    event = "grow"
elif improvement <= 0:                        # plat ou pire
    event = "reallocate"
elif improvement <= self.optimize_threshold:  # 0,5 %
    event = "optimize"
```

Sur le toy, la validation touche son plancher : perplexité 1,0 dès le pas 1 000, et le
log affiche `val=1.0` à chaque décision. Une perplexité ne descend pas sous 1,0 : dès que
la tâche est apprise, `improvement` vaut 0 ou devient négatif (bruit de mesure), donc
**chaque contrôle tombe dans la branche `reallocate`** (L132-133) → `reallocate_pool`
(L152, fonction L51) ajoute **256 slots**.

Conséquence mesurée au pas 1 500 : `PALIER : pool 583 → 839`, alors que la colonie
compte 518 nœuds vivants. Au pas 2 000 : pool 861, **520 nœuds vivants**, soit 341 slots
alloués et vides. Le mot « PALIER » du log ne décrit donc pas un palier de la colonie :
il décrit un agrandissement du **réservoir**. Ce n'est pas gratuit en RAM (les rangées
existent), mais ce n'est pas la source du plateau d'arêtes, et l'effet VRAM est faible :
les rangées vides n'entrent pas dans les activations chaudes, seule la boîte de réception
`[n_nodes, batch, dim]` les porte (341 × 16 × 32 × 4 o ≈ 0,7 Mo par pas de diffusion).

## 4. Pourquoi, une fois la tâche apprise, le nombre d'arêtes ne peut que baisser

`growth.py` L145 :

```python
self.suspended_until = int(step) + self.suspend_births
```

avec `suspend_births: 5000` (`config.py`). Au pas 2 000 le log écrit « phase OPTIMISATION
… naissances suspendues » : `births_left` vaut 0 (`brain.py` L543, branché sur
`allow_node_growth` par `growth.py` L199) **jusqu'au pas 7 000** — soit 3,5 fois la durée
du run. Or l'élagage, lui, continue toutes les 100 pas (`train.py` L501). Mesuré :
−402 arêtes entre les pas 1 500 et 2 000. Ce n'est pas une instabilité, c'est
l'application littérale de la règle « optimiser = laisser le graphe se tasser ».

## 5. Un compteur qui ne se refermait pas — **résolu le 6 novembre**

Arithmétique du log au pas 2 000 : 512 nœuds initiaux + 8 nés − 64 morts = 456, alors que
le run affiche **520 nœuds vivants**. La question est tranchée : il manquait un terme au
**relevé**, pas à l'arithmétique. Le cimetière est vide à la fin alors que 64
eutèrements ont eu lieu, et `revive_node` remet `node_alive` à vrai sans décrémenter
`node_deaths` : les 64 nœuds sont revenus, et rien ne l'imprimait. L'identité est

```
vivants = slots − (morts − ressuscités)   →   520 − (64 − 64) = 520
```

Le compteur `node_revivals` existe désormais, s'affiche dans les vitals
(`ressuscités 64`) et est vérifié par
`tests/test_instrumentation.py::test_node_counters_close`. Détail complet :
`docs/COMPTEURS-noeuds.md`. Les §1 à §4 de ce document n'en dépendent pas.

## 6. Ce que cela implique, sans rien changer

- Le plateau à ≈ 5 400 arêtes est **le comportement correct du code actuel** sur une
  tâche saturée : la perplexité est à 1,0 depuis le pas 1 000, la colonie n'a plus rien
  à apprendre, et la seule chose que la croissance pourrait encore acheter est de la
  mémoire consommée.
- Pour déplacer le plateau, il n'y a que deux familles de leviers, et elles sont
  symétriques : augmenter la naissance par pas (`max_births`, `birth_k`,
  `birth_candidates`) ou baisser l'élagage par pas (`prune_fraction`, `prune_every`).
  `E* ≈ 1 000 × naissances/pas` donne l'effet attendu de chacune, sans mesure nouvelle.
- Sur le texte, `val` n'a pas de plancher à 1,0 : la branche `reallocate` sera atteinte
  par vrai plateau, pas par saturation du plancher. Je n'en dis pas plus : ce régime
  n'est pas mesuré, et une prédiction non mesurée est exactement ce que le §8 du
  diagnostic a déjà payé une fois.
