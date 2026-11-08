# Session du 8 novembre (c) — l'ablation, mesurée

**Verdict en une phrase** : à 500 pas, `seq 32`, `batch 8`, graine 42, même carte, la colonie
qui **grandit** atteint **1 943,301** de perplexité de validation là où le **graphe aléatoire
figé de même taille** atteint **2 071,823** — la croissance a payé, **+6,61 %**, soit le
**cas A** de la règle fixée d'avance ; le rejeu à 2 000 pas du cas B n'est donc pas déclenché,
et la réserve reste entière : 500 pas est un **signal**, pas une loi.

Ce rapport suit l'ordre demandé : verdict, étapes avec chiffres, les deux courbes, ce qui
n'a pas été fait et pourquoi, prochaine action chiffrée. Les mots **mesuré**, **estimé** et
**supposé** sont posés là où ils s'appliquent.

## 1. Conditions de la mesure

Classe **« résultat »** de la règle v2 : seul compte l'espace libre, pas le pourcentage
d'utilisation. Mesures prises avant chaque arm, consignées dans `ablation_conditions.txt` :

| moment | VRAM utilisée | VRAM libre | utilisation |
|---|---:|---:|---:|
| avant les arms | 1 450 MiB | 6 569 MiB | 18 % |
| entre les deux arms (fin grandi) | 7 913 MiB | 106 MiB | — |
| après l'arm figé | 1 661 MiB | 6 358 MiB | — |

Le seuil de la classe résultat était de 6 000 Mo libres : **autorisé** [mesuré]. Les deux
arms ont tourné **dans la même session, sur la même carte, dans le même état**, sans aucune
autre charge (ni pytest, ni génération, ni banc) — c'est ce qui rend la comparaison loyale.

## 2. Les deux arms

| | grandi | figé aléatoire |
|---|---:|---:|
| arêtes au pas 1 | 20 | 3 997 |
| arêtes au pas 500 | 3 993 (3 983 en fin de session) | 3 997 |
| nœuds vivants | 514 (pool 532) | 532 (pool 532) |
| perte au pas 1 | 9,1964 | 9,1955 |
| perte au pas 500 | **7,6172** | 7,6736 |
| **perplexité de validation** | **1 943,301** | **2 071,823** |
| durée | **3 051 s** | 3 712 s |
| journal | `data/per_step_grown.csv` (500 lignes) | `data/per_step_frozen.csv` (500 lignes) |

Écart : **+6,61 % de perplexité** (0,0564 nat de perte par pas) en faveur du grandi.
Règle du protocole : ≤ 2 % → rejeu à 2 000 pas avant tout verdict ; > 2 % en faveur du
grandi → résultat fondateur. **Cas A** [mesuré].

Contrôle du drapeau : le journal de l'arm figé montre **3 997 arêtes à chacun des 500 pas**
et **0 naissance cumulée**. Le gel est réel, et cet arm est bien exempté de la propriété
E\* ≈ naissances/pas × 1 000 [mesuré].

## 3. Les deux courbes, pas par pas

| pas | grandi perte | grandi arêtes | figé perte | figé arêtes |
|---:|---:|---:|---:|---:|
| 50 | 9,1856 | 413 | 9,1482 | 3 997 |
| 100 | 9,0859 | 812 | 9,0689 | 3 997 |
| 150 | 8,9880 | 1 212 | 8,9849 | 3 997 |
| 200 | 8,8547 | 1 612 | 8,8587 | 3 997 |
| 250 | 8,6679 | 2 011 | 8,6772 | 3 997 |
| 300 | 8,5017 | 2 411 | 8,5303 | 3 997 |
| 350 | 8,3188 | 2 810 | 8,3545 | 3 997 |
| 400 | 8,0656 | 3 210 | 8,1144 | 3 997 |
| 450 | 7,7753 | 3 593 | 7,8320 | 3 997 |
| 500 | 7,6172 | 3 993 | 7,6736 | 3 997 |

Le figé mène jusqu'au pas 150 : il démarre avec **3 997 arêtes quand le grandi en a 20**.
Le croisement tombe entre 150 et 200, puis l'écart se creuse presque monotone jusqu'à
−0,056 nat. Tracé complet et lecture par fenêtre : `docs/ABLATION-resultats.md`,
résultat fondateur détaillé : `docs/ABLATION-croissance.md` §6.

## 4. Topologie à la fin

| arm | degré entrant moyen | max | Gini | orphelins |
|---|---:|---:|---:|---:|
| grandi (génome du run jumeau, pas 500) | 7,73 | 16 | 0,2173 | 1 |
| figé (reconstruit, graine 42, 3 997 arêtes) | 7,51 | 19 | 0,1946 | 0 |

La distribution du grandi est un peu plus inégale, mais **le figé n'est pas moins concentré
au point d'expliquer 6,6 %** : l'écart ne vient donc pas d'un hub, ni de la capacité (le
figé a plus d'arêtes **et** plus de nœuds à tous les pas). Deux explications restent, que
cette expérience ne sépare pas : **le routage** (une arête naît là où un message est passé)
et **le programme de capacité** (20 → 3 993 arêtes en 500 pas). Les séparer demande un
troisième arm « croissance sans routage » — voir §8.

## 5. Vie du graphe, lue dans le journal par pas

- **figé** : 0 naissance, 0 mort, 0 résurrection — rien ne bouge, comme voulu.
- **grandi**, cumulés au pas 500 : 3 998 naissances d'arêtes, 17 arêtes coupées, 31 nœuds
  enterrés, 31 résurrections, cimetière vide à la fin, 514 nœuds vivants.
- les naissances valent **8,0 par pas sur chacune des dix fenêtres** : la colonie ne naît
  pas *parce que* la perte stagne (`max_births 8` par pas, par construction). Ce n'est pas
  un mécanisme adaptatif, c'est une montée en capacité régulière — et cela seul interdit de
  conclure que « la croissance s'adapte ».
- corrélation entre Δ perte par fenêtre et taille moyenne du graphe : **−0,82** (n = 10,
  descriptif) : les fenêtres où le graphe est le plus gros sont celles où la perte baisse
  le plus. Autrement dit, l'avantage du grandi arrive avec la densité, pas au début.

## 6. Coût

Grandi **3 051 s**, figé **3 712 s** pour le même budget de pas : la croissance coûte
**18 % de temps en moins** [mesuré], parce qu'elle n'entretient pas 3 997 arêtes dès le
premier pas. Le coût par pas en fin de course est identique (~9 s à ~4 000 arêtes).

## 7. Ce qui n'a pas été fait, et pourquoi

- **Rejeu des deux arms à 2 000 pas** : non déclenché. La règle ne l'impose que si l'écart
  est ≤ 2 % ; il vaut 6,61 % [mesuré].
- **Observation 300 pas `seq 128` / `batch 16` / `tbptt 4`** : **refusée par la règle**, et
  c'est la règle qui a raison. Elle appartient à la classe **« chronométrage »**, qui exige
  une carte quasi vide (< 1 Go utilisé, < 10 %). Mesure du 8 novembre à 18 h 07 :
  **1 686 MiB utilisés, 6 333 MiB libres, 18 % d'utilisation**, avec une trentaine de
  processus GPU étrangers (navigateurs, Discord, Steam, overlays, Epic). Condition non
  satisfaite → **aucun ms/pas, aucune ETA mesurée**, et l'on s'arrête là comme demandé.
  Conséquence directe : l'ETA du run 3 000 pas reste **estimée**, pas mesurée. Aucun
  processus étranger n'a été tué.
- **Génome du grandi non sauvegardé** par `scripts/ablate_growth.py` : la distribution des
  degrés du §4 vient du run jumeau (même protocole, écart de reproductibilité 1,75 %).
  À corriger avant la prochaine ablation.
- **Aucun échantillon de texte** : aucune génération n'a tourné. Pendant les arms c'était
  interdit (charge concurrente) ; après, la carte était encore à 1,7 Go utilisés et le
  protocole demandait de ne rien lancer d'autre.
- **Première mesure de variance run-to-run du projet** : deux runs *grandi* indépendants
  donnent 1 977,843 et 1 943,301 → **1,75 %**, soit 0,018 nat. L'écart mesuré de 6,61 %
  vaut ~3,7× cette dispersion, mais elle n'est estimée que sur deux runs et l'arm figé n'a
  aucune réplique. Une part de l'écart vient d'un détail : le run de comparaison appelait
  `speak()` à chaque rapport, qui tirait dans le RNG global et décalait les naissances.

## 8. Prochaine action, chiffrée

1. **Banc de 50 pas, classe chronométrage**, carte quasi vide (exiger < 1 Go utilisé et
   < 10 %), pour convertir l'ETA en mesure : coût/pas à ~4 000 arêtes et coût `tbptt 4`
   contre `tbptt 16`. [estimé] 10–15 min sur une carte libre.
2. **Run texte 3 000 pas** — la priorité suivante, puisque la croissance a une valeur
   mesurée. ETA [estimé] : 6,1 s/pas → **5,1 h** ; si le graphe continue de grossir jusqu'au
   plafond, 9 s/pas → **7,5 h**. Échantillons aux pas 200 / 500 / 1 000 / 2 000 / 3 000,
   rapport intermédiaire au pas 1 000.
3. **Troisième arm de l'ablation** : croissance **sans routage** (même calendrier d'arêtes,
   identités tirées au hasard) — c'est lui qui dira si l'écart de 6,61 % vient du routage ou
   de la seule montée en capacité. [estimé] ~1 h, même règle de carte.
4. **Sauvegarder le génome** en fin d'arm dans `ablate_growth.py`, pour que la prochaine
   ablation mesure ses propres degrés.

## 9. Livraison

- code : `scripts/ablate_growth.py` (journal par arm, UTF-8 forcé, JSON + texte de fin),
  `scripts/ablation_report.py` (verdict, courbes, degrés, vie du graphe), `mycelia/train.py`
  (journal par pas) ;
- mesure : `data/per_step_grown.csv`, `data/per_step_frozen.csv`, `ablation_grown.json`,
  `ablation_frozen.json`, `ablation_conditions.txt` ;
- documentation : `docs/ABLATION-croissance.md` §5–§6, `docs/ABLATION-resultats.md`,
  ce rapport ;
- `pytest -q` complet, arbre propre, ZIP horodaté `mycelia-v5-2026-11-08-c.zip`.
