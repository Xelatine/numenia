# Session du 9 novembre (e) — diagnostic mémoire et capacité

**Verdict en une phrase** : le poste dominant est le **graphe autograd du pas** —
**3 924 Mo, soit 93 % du pic** de 4 205 Mo à 3 000 arêtes en `seq 32`/`tbptt 16` [mesuré] —,
le mur du run texte (`seq 128`/`batch 16`) est à **4 425 arêtes** (8 707 Mo mesurés à 4 000,
OOM à 4 425) et l'écart à combler pour tenir ce run à **4 000 arêtes** est de **1 257 Mo**
dans les 7 450 Mo réellement disponibles ; pour atteindre **E\* ≈ 8 000** arêtes à
`seq 32`/`batch 8` il manque **≈ 4 853 Mo** [estimé par régression sur cinq points mesurés].

Tout chiffre de ce rapport porte son étiquette : **[mesuré]**, **[estimé]** (calcul sur des
mesures) ou **[supposé]** (mécanisme non prouvé ici). Aucune ligne de `mycelia/` n'a été
modifiée : le harnais accroche ses mesures de l'extérieur (hooks, méthodes patchées à
l'instance), et l'unique fichier de code ajouté est `scripts/mem_profile.py`.

---

## 1. La carte, et les conditions de chaque campagne

| | valeur | étiquette |
|---|---|---|
| carte | **NVIDIA GeForce RTX 3070 Laptop GPU** | [mesuré] |
| VRAM totale | **8 192 MiB** | [mesuré] |
| occupé au repos, session entière | 700–741 MiB (**8,6–9,0 %**) | [mesuré] |
| libre au repos | 7 278–7 307 MiB | [mesuré] |

**Classe « chronométrage » : AUTORISÉE pour la première fois.** La règle v2 exige < 1 Go
utilisé et < 10 % : 741 MiB et 9 % la satisfont. Le banc de chronométrage refusé le 8 novembre
(2 025 MiB, 18 %) a donc pu être remplacé par des mesures directes. Chaque campagne a écrit son
bloc `nvidia-smi` **avant** de tourner dans `ablation_conditions.txt` (blocs « 9 novembre 2026
(e) », tous à 8,7–9,0 %). Aucun processus étranger n'a été tué.

Les mesures de **mémoire** (`memory_allocated` / `memory_reserved`) sont par processus et
restent exactes sur une carte occupée ; les mesures de **temps** n'ont été prises qu'à 9 %
d'occupation. Une réserve : Windows affiche le bureau sur cette carte, donc un « 0 MiB » est
inatteignable — 700 MiB est le plancher de cette machine [mesuré].

---

## 2. L'inventaire statique — le cerveau est petit

532 nœuds, 4 000 arêtes, vocabulaire 10 000, `dim 32`, après **un pas** (les moments AdamW
n'existent qu'après le premier pas). Liste complète : `data/mem_report/static_inventory.json`
(84 lignes), table : `static_inventory.md`. Top 20 par octets :

| # | tenseur | shape | Mo | grad | persistant | bloc du checkpoint |
|---:|---|---|---:|:--:|:--:|---|
| 1 | `bank.b1_w` | 558×64×32 | 4,359 | oui | oui | `state_dict.node_bank` |
| 2 | `bank.b2_w` | 558×32×64 | 4,359 | oui | oui | `state_dict.node_bank` |
| 3 | `optim.?.exp_avg` | 532×64×32 | 4,156 | non | oui | `optimizer` |
| 4 | `optim.?.exp_avg_sq` | 532×64×32 | 4,156 | non | oui | `optimizer` |
| 5 | `optim.?.exp_avg` | 532×32×64 | 4,156 | non | oui | `optimizer` |
| 6 | `optim.?.exp_avg_sq` | 532×32×64 | 4,156 | non | oui | `optimizer` |
| 7 | `bank.q_w` | 558×32×32 | 2,180 | oui | oui | `state_dict.node_bank` |
| 8 | `bank.k_w` | 558×32×32 | 2,180 | oui | oui | `state_dict.node_bank` |
| 9 | `optim.?.exp_avg` (q) | 532×32×32 | 2,078 | non | oui | `optimizer` |
| 10 | `optim.?.exp_avg_sq` (q) | 532×32×32 | 2,078 | non | oui | `optimizer` |
| 11 | `optim.?.exp_avg` (k) | 532×32×32 | 2,078 | non | oui | `optimizer` |
| 12 | `optim.?.exp_avg_sq` (k) | 532×32×32 | 2,078 | non | oui | `optimizer` |
| 13 | `embedding.weight` | 10 000×32 | 1,221 | oui | oui | `state_dict.vocab` |
| 14 | `output_head.weight` | 10 000×32 | 1,221 | oui | oui | `state_dict.vocab` |
| 15–18 | `optim.{embedding,output_head}.*` | 10 000×32 | 4×1,221 | non | oui | `optimizer` |
| 19 | `attr._last_out` | 530×8×32 | **0,518** | **oui** | **non** | *non sauvegardé* |
| 20 | `edge_index` | 2×20 000 | 0,305 | non | oui | `state_dict.wiring` |

### Les trois sommes, et l'écart expliqué

| mesure | Mo | étiquette |
|---|---:|---|
| somme des tenseurs inventoriés | **48,44** | [mesuré] |
| dont persistants (checkpoint) | 47,73 | [mesuré] |
| `torch.cuda.memory_allocated()` au repos après le pas | **391,75** | [mesuré] |
| checkpoint classe L écrit par `save_genome` | 21,78 | [mesuré] |
| écart 391,75 − 48,44 | **343,31** | [mesuré] |

L'écart **n'est pas** de la fragmentation : c'est le **graphe autograd retenu par la
télémétrie**. Mesuré directement, dans le même processus, sur 532 nœuds / 4 000 arêtes :

```
après .to(cuda)      :  16,01 Mo
après Trainer        :  16,01 Mo
après 1 pas          : 359,67 Mo
après gc.collect()   : 359,67 Mo      <- ce n'est pas un cycle Python
après _last_out=None :  45,06 Mo      <- 314,61 Mo rendus
```

`brain._last_out` (dernière sortie vivante d'un nœud, au sens autograd) et
`brain.last_node_io` sont réassignés **pendant** le forward et jamais lâchés après le pas :
ils épinglent le graphe entier du dernier bloc. À `seq 32` cela coûte 315 Mo ; à `seq 128`,
≈ 1,3 Go [estimé par proportionnalité, mesuré à 32 seulement]. Le checkpoint est plus petit que
l'état (21,78 Mo) uniquement parce que `torch.save` compresse.

**Conséquence pour la spec `.myc`** : la liste des tenseurs d'état dynamique est close
(84 entrées), et le format n'a **pas** de sections `.myc` — le génome est un dict `torch.save`
(`FORMAT_VERSION 3`). La colonne « bloc » ci-dessus nomme donc le bloc de ce dict, c'est-à-dire
ce qu'un écrivain `.myc` devra reprendre. Deux attributs seulement échappent au `state_dict`
(`_last_out`, `last_node_io`) et ce sont précisément ceux qui coûtent.

---

## 3. Où va la mémoire d'un pas — la ventilation

Harnais : `Trainer.step_once` réel, mesuré par brackets (`forward` par hook de module,
`backward` par `Tensor.backward` patché, `optimizer.step`, `loss` par `cross_entropy`,
`births` par `_birth_edges`, `lifecycle` par `apply_lifecycle`). Graphes synthétiques de
taille imposée (graine 7, degré moyen 7,5, pas de hub). `data/mem_report/pass_breakdown.csv`.

### 3.1 Le tableau (seq 32, batch 8, 2 pas mesurés)

| arêtes | pic du pas (Mo) | forward | backward | loss | births | penalties | optimizer | lifecycle | ms/pas |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 500 | **125,5** | 1 290 | 1 368 | 8 | 6 | 7 | 3 | — | 2 694 |
| 1 000 | **252,7** | 1 310 | 1 377 | 7 | 8 | 6 | 4 | — | 2 721 |
| 2 000 | **948,1** | 1 717 | 2 025 | 8 | 5 | 6 | 4 | — | 3 782 |
| 3 000 | **1 691,0** | 2 210 | 2 758 | 7 | 5 | 6 | 3 | — | 5 010 |
| 4 000 | **2 079,5** | 2 556 | 3 374 | 9 | 4 | 7 | 4 | — | 5 980 |
| 4 000 (élagage chaque pas) | 2 416,2 | 1 752 | 2 443 | 2 | 5 | 6 | 3 | **3,5** | 4 225 |

Temps en **ms par pas** (la colonne brute du CSV couvre les deux pas ; divisée par deux ici).
Le pic est lu sur `max_memory_allocated` du pas entier, les brackets remettant le compteur à
zéro : le maximum courant est récolté **avant** chaque remise à zéro, sans quoi le pic du pas
valait celui du dernier poste (200 Mo au lieu de 2 365 — bug corrigé, et les quatre points
« nœuds » qu'il avait faussés ont été re-mesurés le 9 novembre).

**Forward + backward = 5 930 ms sur 5 980, soit 99,2 % du temps** [mesuré]. La perte, l'Adam,
les naissances, l'élagage, le gel des nouveau-nés et le clip : moins de 1 % à eux tous.
L'élagage coûte 3,5 ms par pas, `optimizer.step` 3,7 ms, les naissances 4,3 ms. Le calcul est
la diffusion, rien d'autre.

### 3.2 La part du graphe dans le pic

Point de référence (`one`, 532 nœuds, 3 000 arêtes, `seq 32`, `batch 8`, `tbptt 16`) :

| grandeur | Mo | étiquette |
|---|---:|---|
| allocation au repos avant le pas | 280,5 | [mesuré] |
| pic du pas | **4 204,9** | [mesuré] |
| **graphe autograd** | **3 924,4** | [mesuré] |
| part du graphe dans le pic | **93,3 %** | [mesuré] |
| réservé après le pas | 4 224,0 | [mesuré] |

Le graphe pèse **≈ 1,8 Go par bloc de 16 tokens** à 3 000 arêtes, et **2,79 Go** à seulement
2 000 arêtes en `tbptt 16` [mesuré]. C'est lui, et lui seul, qui décide du mur.

### 3.3 TBPTT : le coefficient, enfin chiffré (2 000 arêtes, seq 32, batch 8)

| bloc (tokens) | pic (Mo) | ms/pas |
|---:|---:|---:|
| 1 | **359,4** | 1 983 |
| 2 | **690,4** | 3 202 |
| 4 | **948,1** | 3 782 |
| 16 | **2 785,7** | 4 204 |

De 1 à 16 tokens de bloc, le pic est multiplié par **7,75** et le temps par **2,1** : la
mémoire suit la longueur du bloc (moins un plancher fixe de ~250 Mo d'état + graphe retenu),
le temps suit le nombre total de tokens. `tbptt 16` coûte **2,9× la mémoire de `tbptt 4`** et
**1,1× son temps** — le diagnostic chiffre, il ne tranche pas.

### 3.4 Sparse ou dense ? — tranché par la mesure

À **arêtes constantes** (2 000) et en ne faisant varier que la population :

| nœuds | pic (Mo) | ms/pas | degré entrant moyen |
|---:|---:|---:|---:|
| 200 | **3 229,6** | 6 175 | 10,0 |
| 532 | **2 418,6** | 5 162 | 3,8 |
| 1 000 | **391,9** | 3 694 | 2,0 |
| 2 000 | **410,7** | 3 617 | 1,0 |

**Le coût BAISSE quand la population monte.** Une implémentation dense (matrice N×N, ou
score sur tous les slots) aurait un coût en N² : à 2 000 nœuds elle demanderait 2 000² = 4 M
scores par pas et par tour, soit **×2 000** le coût à 1 000 nœuds ; mesuré, le temps est
**×0,98**. Le message passing est donc **sparse, par liste d'arêtes** [mesuré], et l'hypothèse
dense est réfutée d'un facteur ≈ 2 000. Le surcoût d'une petite colonie est un effet réel et
contre-intuitif : à arêtes fixes, moins de nœuds ⇒ plus d'arêtes chaudes par tour ⇒ un front
d'onde plus épais (200 nœuds coûtent 1,34× le pic de 532, mesuré).

---

## 4. Les courbes de mise à l'échelle

Chaque point est un processus neuf, mesuré sur 2 pas après 1 pas de chauffe.
`data/mem_report/scaling.json`. `tbptt 16` partout (la valeur du fichier de config).

### 4.1 Arêtes (532 nœuds, seq 32, batch 8) — sweep `edges`

| arêtes | pic (Mo) | ms/pas | note |
|---:|---:|---:|---|
| 1 000 | 402,2 | 2 954 | |
| 2 000 | 2 418,6 | 4 140 | |
| 3 000 | 4 266,5 | 5 744 | |
| 4 000 | 5 725,1 | 8 040 | |
| 6 000 | **8 807,5** | 14 624 | **au-delà des 8 192 Mo de la carte** |
| 8 000 | — | — | **OOM** |

Le saut 1 000 → 2 000 (×6 en mémoire) n'est pas une erreur : sous 1 000 arêtes le front
d'onde meurt en deux tours et le graphe n'a presque rien à retenir.

### 4.2 Nœuds (ratio arêtes/nœuds 7,5 — seq 32, batch 8) — sweep `nodes`

| nœuds | arêtes | pic (Mo) | ms/pas |
|---:|---:|---:|---:|
| 200 | 1 500 | 2 482,1 | 5 422 |
| 300 | 2 250 | 3 398,1 | 6 614 |
| 400 | 3 000 | 4 441,5 | 7 570 |
| 532 | 3 990 | 5 717,0 | 8 646 |
| 800 | 6 000 | **8 052,8** | 11 877 |
| 1 000 | 7 500 | **9 860,5** | 20 534 |

À 800 nœuds la mesure dépasse déjà la VRAM totale : **le pilote Windows déborde en mémoire
système** au lieu d'échouer, et le pas ralentit de 40 % [mesuré pour les Mo ; mécanisme WDDM
[supposé], corroboré par 9 860 Mo alloués sur une carte de 8 192].

### 4.3 Séquence (2 000 arêtes, batch 8) — sweep `seq`

| seq | pic (Mo) | ms/pas |
|---:|---:|---:|
| 32 | 2 418,6 | 4 183 |
| 64 | 2 526,0 | 8 234 |
| 128 | 2 526,0 | 16 154 |
| 256 | 2 526,0 | 31 859 |

**La mémoire est plate, le temps double à chaque doublement.** Avec `tbptt` fixe, la longueur
de séquence n'est pas un axe de mémoire du tout : 128 tokens font 8 blocs de 16, pas un bloc
de 128. Le run texte n'était donc pas bloqué par `seq 128` — c'est un fait mesuré.

### 4.4 Batch (seq 128, 2 000 arêtes) — sweep `batch`

| batch | pic (Mo) | ms/pas |
|---:|---:|---:|
| 4 | 1 982,5 | 17 069 |
| 8 | 2 526,0 | 20 789 |
| 16 | **3 620,0** | 20 934 |
| 32 | **5 924,5** | 20 547 |

Le temps sature (le GPU est plein), la mémoire croît de **+1 094 Mo pour 8 → 16** puis
**+2 304 Mo pour 16 → 32** [mesuré]. `batch 16` en `seq 128` **tient** à 2 000 arêtes dans
7 450 Mo — c'est la croissance du graphe qui le tuera, pas le batch seul.

### 4.5 Le croisement décisif (seq 128 / batch 16) — sweep `cross`

| arêtes | pic (Mo) | ms/pas | état |
|---:|---:|---:|---|
| 1 000 | 562,6 | 15 033 | tient |
| 2 000 | 3 620,0 | 20 137 | tient |
| 3 000 | 6 472,8 | 27 452 | tient, à ~1 000 Mo du plafond |
| 4 000 | 8 707,1 | 38 509 | tient **en débordant** hors VRAM |

### 4.6 Le mur, par dichotomie — `bisect.json`

| arêtes | pic (Mo) | ms/pas | verdict |
|---:|---:|---:|---|
| 400 | 157,6 | 13 707 | tient |
| 3 200 | 6 840,1 | 29 287 | tient (dans la VRAM) |
| 3 900 | 8 373,9 | 39 600 | tient, **déborde** (> 8 192) |
| 4 250 | 9 173,3 | 49 624 | tient, déborde de 981 Mo |
| **4 425** | — | — | **OOM** (commit hôte épuisé) |

**Le mur mesuré du run texte est 4 425 arêtes** ; la plus grande taille qui tienne *dans la
VRAM* est **≈ 3 200** (6 840 Mo), celle qui tienne *du tout* est **4 250**.

### 4.7 Extrapolation contre mesure — les courbes sont-elles une loi ?

Régression linéaire sur les quatre points mesurés du croisement :
`pic ≈ 2,73 × arêtes − 1 981` (Mo).

| arêtes | pic mesuré | pic prédit | écart |
|---:|---:|---:|---:|
| 1 000 | 562,6 | 749,5 | +33 % |
| 2 000 | 3 620,0 | 3 479,5 | −3,9 % |
| 3 000 | 6 472,8 | 6 209,5 | −4,1 % |
| 4 000 | 8 707,1 | 8 939,5 | +2,7 % |

La loi prédit le pic de 9 173 Mo (mesuré à 4 250) à **4 086 arêtes**, soit **−4 %** de l'OOM
réel (4 425) [estimé vs mesuré]. Entre 2 000 et 4 000 arêtes la droite est utilisable à ~4 % —
pas mieux, et le point 1 000 en est exclu (le front d'onde meurt).

---

## 5. Réductible / verrouillé — le tableau de livraison

| poste | Mo [mesuré] | % du pic | réductible ? | levier | prix | verrou sémantique |
|---|---:|---:|---|---|---|---|
| **Graphe autograd du bloc** | **3 924** (3 000 arêtes, `tbptt 16`) | **93,3 %** | **oui** | checkpointing par tour ; `tbptt` plus court ; blocs plus courts | re-forward (+30–40 % de temps [estimé]) | *activations* seulement : l'état stocké reste float32 exact (continuité 1e-6) |
| Graphe retenu par la télémétrie (`_last_out`, `last_node_io`) | **315** (532/4 000, `tbptt 4`) ; ≈1,3 Go à `seq 128` [estimé] | 15 % / ~45 % | **oui** | lâcher les deux références après le pas | **rien** (télémétrie) | le dashboard lit `last_node_io` entre deux pas — à décider, pas un verrou |
| État statique (params + buffers + Adam) | **48,4** | 2,3 % | non (verrouillé) | — | — | continuité 1e-6 : l'état doit rester **float32 exact** |
| dont moments AdamW | **34,0** | 1,6 % | oui | offload CPU des moments | trafic hôte↔GPU à chaque pas | l'état de l'optimiseur doit revenir identique (reprise) |
| dont vocabulaire (embedding + tête + Adam) | 7,3 | 0,35 % | oui | bf16/fp16 du vocabulaire | perte de précision des logits | — (négligeable, la piste est close) |
| Cimetière | **0** sur ce graphe ; **28** nœuds en cimetière sur le run `schedule`, 0 sur `grandi` (32 nœuds enterrés, tous rappelés par un message) | < 0,1 % | **non** | paging des nœuds vivants vers le disque | — | **résurrection par le trafic** : un nœud « inactif » peut recevoir un message ; la seule frontière VRAM/disque est le statut cimetière |
| Capacité `edge_index` (20 000 slots) | 0,31 | < 0,1 % | non | — | — | — (piste close) |
| Colonie de 200 nœuds (au lieu de 532) | **2 482** vs 5 717 à ratio 7,5 (**−57 %**) | 44 % | oui | réduire la population | perte de capacité ; à arêtes fixes une petite colonie coûte **plus cher** (3 230 vs 2 419, mesuré) | aucun verrou, mais **qualité non mesurée** (cf. §7) |
| `tbptt 16 → 1` (diagnostic seulement) | **359** vs 2 786 à 2 000 arêtes (**−87 %**) | — | oui | — | gradient tronqué : autre apprentissage | **choix humain du 7 novembre** : le diagnostic chiffre, il ne tranche pas |
| `batch 16` en `seq 128` | 3 620 (2 000 arêtes) | — | non | — | — | **choix humain du 7 novembre** |
| bf16 sur les activations | 1 777 vs 2 703 (**−34 %**) | — | oui | `autocast` | **+17 % de temps** et perte quantifiée à 1/16 de nat : 17 pas sur 20 rendent **la même valeur** (9,18750) | acceptable en entraînement, interdit en reprise |

### Le test bf16, en détail (2 000 arêtes, seq 32, batch 8, 20 pas, même graine)

| | pic (Mo) | ms/pas | pertes (1er → 20e) |
|---|---:|---:|---|
| f32 | 2 702,9 | 5 410 | 9,20848 → 9,15119 (monotone, −0,057 nat) |
| bf16 | **1 777,5** | 6 344 | 9,18750 → 9,12726, dont **17 pas consécutifs à 9,18750** |

Le gain mémoire est réel (−34 %) ; le prix est double : **17 % plus lent** sur cette carte et
une **perte quantifiée** qui rend la courbe par pas inutilisable pour juger un progrès
inférieur à 0,06 nat.

---

## 6. La cible : ce qu'il manque, et dans quel ordre

| objectif | besoin | disponible | manque | étiquette |
|---|---:|---:|---:|---|
| `seq 128`/`batch 16` à **3 000** arêtes | **6 473** | 7 450 | **0** — tient avec 977 Mo de marge | [mesuré] |
| `seq 128`/`batch 16` à **4 000** arêtes | **8 707** | 7 450 | **1 257** | [mesuré] |
| `seq 128`/`batch 16` au-delà de **4 425** arêtes | > 9 173 | 7 450 | OOM | [mesuré] |
| `seq 32`/`batch 8` à **E\* ≈ 8 000** arêtes | **12 303** | 7 450 | **4 853** | [estimé, régression sur 5 points] |

Les leviers, classés par gain **mesuré** quand il existe, **estimé** sinon :

1. **Lâcher la télémétrie après le pas** — gain **315 Mo** à `seq 32`, ≈ 1,3 Go à `seq 128`
   [estimé], **coût nul**, aucun verrou sémantique (ce sont des vues de graphe, pas de l'état).
   Mesuré aussi : −20 % de temps par pas sur 4 000 arêtes (8 610 → 6 856 ms), sans changer le
   pic — le graphe retenu n'est pas dans le pic, il est dans le *prochain* pas.
   **Candidat n°1, quasi gratuit.**
2. **Checkpointing par tour** (recalculer au backward) — cible le poste de 3 924 Mo : à −50 %
   il rend 1 960 Mo à 3 000 arêtes et fait passer le pic de 4 000 arêtes de 8 707 à ≈ 6 750 Mo,
   c'est-à-dire **sous les 7 450 disponibles**. Prix : un re-forward, +30–40 % de temps
   [estimé depuis la part du backward, 56 %]. Aucun verrou : seules les activations bougent,
   l'état reste exact.
3. **Colonie réduite** — gain **−57 %** (2 482 vs 5 717 Mo à ratio 7,5) mais **la qualité n'est
   pas mesurée** : sur 30 pas, la perte du 200 nœuds est identique à celle du 532 **à la
   cinquième décimale** (9,20848 → 9,09665 dans les deux cas) — le lecteur n'a pas encore senti
   le graphe, cet essai ne discrimine rien.
4. **bf16** — gain **−34 %** mesuré, prix +17 % de temps et une perte quantifiée.
5. **`tbptt 1`** — gain **−87 %** mesuré à 2 000 arêtes, mais c'est un autre apprentissage :
   chiffré pour information, interdit de décision unilatérale.

---

## 7. Ce qui n'a pas été mesuré, et pourquoi

- **Le mur « 5,2 Go libres » du 8 novembre n'a pas été reproduit** : la config exacte
  (`seq 32`, `batch 8`, 3 000 arêtes, `tbptt 16`) culmine à **4 205 Mo** [mesuré] et tourne.
  Avec 7,3 Go libres elle tient ; avec les 6,1 Go du 8 novembre elle aurait dû tenir aussi.
  L'OOM de ce jour-là vient donc de l'état de la carte à cet instant (1 894 MiB pris par
  Windows + segments réservés + commit) et **non** du besoin intrinsèque du système
  [supposé — non reproductible aujourd'hui sans saturer volontairement la mémoire].
- **Pas de mesure sur une carte totalement vide** : impossible, le bureau Windows y réside
  (700 MiB de plancher). Toutes les campagnes ont été prises à 8,7–9,0 % [mesuré].
- **Le checkpointing par tour n'a pas été implémenté** : ce serait modifier le forward, ce que
  la consigne interdit. Son gain est **estimé** depuis la part mesurée du backward.
- **L'offload CPU des moments Adam n'a pas été mesuré** (changement de code) : le gain plafonne
  à 34 Mo, soit 1,6 % du pic — la piste est **chiffrée et close**.
- **bf16 n'a été mesuré qu'en `autocast`**, 20 pas, sans poids maîtres ni optimiseur bf16 :
  c'est un prix, pas un run.
- **Rien au-delà de 2 000 nœuds ni de 8 000 arêtes** : 8 000 arêtes en `seq 32`/`batch 8` OOM,
  et 1 000 nœuds dépassent déjà la VRAM (9 860 Mo alloués). La cible des 140 000 nœuds reste
  hors de portée de cette carte, sans extrapolation hasardeuse.
- **La qualité de la colonie de 200 nœuds** n'est pas établie (30 pas identiques au 532).
- **Aucune mesure de débit disque / mmap** : hors sujet ici, `mmap` n'est actif qu'à l'échelle 3.

---

## 8. Prochaine action chiffrée

1. **Appliquer le levier 1** (lâcher `_last_out`/`last_node_io` en fin de pas) : gain
   **315 Mo à `seq 32`, ≈ 1,3 Go à `seq 128`** [estimation par proportionnalité], coût nul,
   et −20 % de temps par pas déjà mesurés.
2. **Mesurer, pas supposer, le checkpointing par tour** : instrumenter un forward qui ne garde
   que l'entrée de chaque tour et rejoue — cible **−1 960 Mo à 3 000 arêtes** (pic 4 205 →
   2 245) pour un re-forward borné par les 56 % de backward déjà mesurés. Si le gain dépasse
   **1 257 Mo**, le run texte à 4 000 arêtes en `seq 128`/`batch 16` devient possible dans les
   7 450 Mo actuels.
3. **Ne pas relancer le run texte à 3 000 pas avant (1) et (2)** : à 3 000 arêtes il tient
   aujourd'hui (6 473 Mo, 27,5 s/pas → 22,9 h pour 3 000 pas [estimé]), mais la croissance le
   pousse vers 4 425 arêtes où il meurt.

---

*Mesures : `scripts/mem_profile.py`, 9 novembre 2026. Données :
`data/mem_report/{static_inventory.json,static_inventory.md,pass_breakdown.csv,pass_breakdown.json,scaling.json,bisect.json,amp.json,reduced.json}`.
Conditions de carte : `ablation_conditions.txt`, blocs « 9 novembre 2026 (e) ».*
