# Session du 8 novembre 2026 — ablations, comparaison terminée, baseline démasqué deux fois

**8 novembre 2026.** Consigne : condition bloquante sur l'état de la carte, cinq ablations
quasi gratuites, l'ablation centrale croissance contre graphe figé, la fin de la
comparaison, l'observation puis le run texte, verdict et livraison. Étiquette de chaque
chiffre : **[mesuré]**, **[estimé]**, **[supposé]**.

## 1. Verdict en une phrase

La session a **établi cinq faits mesurés** (pas de fuite de membrane, déterminisme GPU à
4,77e-07, pics de perte sans corrélation avec la vie du graphe, routage non concentré,
comparaison terminée), **trouvé deux bugs de plus dans le baseline** (identité, puis
échelle d'initialisation : ppl de validation 6 060 → 159,97), et **n'a pas pu mesurer
l'ablation centrale ni l'observation** parce que la carte n'est jamais descendue sous
1 297 Mo et 13 % — condition posée en tête de session.

## 2. Étape par étape

### Étape A — les cinq ablations quasi gratuites

**A.1 Fuite de membrane aux frontières de documents : NON [mesuré].** Détail dans
`docs/MEMBRANE-frontieres.md`.

| Mesure | Valeur |
|---|---|
| lignes du corpus / distinctes | 29 402 / **29 402** (0 doublon) |
| tokens train / validation | 4 955 606 / 260 950 |
| fenêtres train / validation | 38 715 / 2 038 |
| fenêtres **identiques** | **0** |
| `val[0:128] == train[-128:]` | vrai → 1 fenêtre sur 2 038 = **0,049 %** |

Preuves de ligne : `brain.py` L548 (`_fresh_state` si `state is None`), L421-429 (zéros) ;
`train.py` `step_once` (`state = None` à chaque pas), `validate` (un appel **sans état** par
fenêtre). Il n'y a **aucune frontière de document** dans le code (`data.py`, docstring de
`TextDataset` : flux continu). Verdict : rien à corriger avant un run texte.

**A.2 Déterminisme de reprise sur CPU et GPU : OK 1e-6 sur les deux [mesuré].**
`python scripts/check_determinism.py --steps 50` (16 nœuds, 208 arêtes, protocole de
`test_resume_continuity`) :

```
cpu : écart max 4.768e-07 entre run continu et reprise -> OK 1e-6
cuda: écart max 4.768e-07 entre run continu et reprise -> OK 1e-6
```

La crainte d'`index_add_` non déterministe ne se matérialise pas à cette taille. Le 1e-6
reste la référence ; il n'y a pas de tolérance GPU à élargir.

**A.3 Pics de perte contre événements : hypothèse Adam ÉCARTÉE, cause non concluante
[mesuré + non concluant].** Détail dans `docs/ADAM-evenements.md`.

| pas | perte | nœud né | nœud mort | résurrection | pool |
|---:|---:|---|---|---|---|
| 1 300 | **0,0365** | non | non (64 depuis 1 100) | non | inchangé |
| 1 800 | **0,0332** | non | non | non | inchangé (839 depuis 1 500) |

Le mécanisme Adam existe bien (`checkpoint.py` L109-133 : moments ré-attachés seulement si
la forme est identique, donc lignes nouvelles à moments nuls), mais **aucune forme de
paramètre ne change à ces deux pas** : l'hypothèse ne peut pas expliquer ces pics. Les pics
suivent en revanche des rafales d'élagage (555 et 554 arêtes coupées), mais l'élagage a lieu
tous les 100 pas sans provoquer de pic : **je ne conclus pas**. Il manque deux journaux :
perte par pas, et compteurs par pas.

**A.4 Distribution des degrés : pas d'effondrement [mesuré].** Détail dans
`docs/ROUTAGE-degres.md`.

| checkpoint | pas | arêtes | degré entrant moyen | max | Gini | CV mesuré | CV Poisson | 1 % le plus connecté |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| texte | 500 | 3 971 | 7,73 | 16 | 0,217 | 0,386 | 0,360 | 2,0 % |
| toy | 1 502 | 5 265 | 10,16 | 24 | 0,195 | 0,354 | 0,314 | 2,1 % |
| toy v5 | 2 000 | 4 890 | 9,41 | 24 | 0,224 | 0,405 | 0,326 | 2,3 % |

Un léger excès de variance apparaît avec l'âge (+7 % à 500 pas, +24 % à 2 000), jamais une
loi de puissance : le nœud le plus connecté reçoit 2 à 3 fois la moyenne, pas 100 fois.
Un orphelin entrant par checkpoint, traité par le cycle de vie, pas par le routage.

**A.5 Constante de mémoire : protocole invalide, arrêté [mesuré, puis STOP].**
`scripts/memory_delay.py`, régime réduit (32 nœuds, dim 16, 6 sauts, 200 arêtes, seq 110,
batch 4, 80 pas) :

```
délai   5 | perte 4.606 -> 4.587 | exactitude 0.0086 | 251 s
hasard = 0.0102
```

À 80 pas, le modèle **n'apprend rien** (exactitude au niveau du hasard) : la courbe ne
mesurerait pas une mémoire, elle mesurerait l'absence d'apprentissage. Le banc est arrêté
plutôt que rempli de points non informatifs. Un protocole valide demande le vrai régime
(512 nœuds, ~2 200 arêtes, ~2 000 pas comme le run v5) : **1 à 1,7 h par délai [estimé]**,
soit 5 à 8 h pour cinq délais, carte libre. Une première tentative en cinq processus
parallèles a échoué proprement (`DefaultCPUAllocator: not enough memory` pour 2 Mo) : la
machine ne supporte pas cinq PyTorch à la fois.

### Étape B — l'ablation centrale : **CODÉE, VALIDÉE, NON MESURÉE**

`mycelia/ablation.py` + `growth.mode: frozen_random` + `scripts/ablate_growth.py` +
`tests/test_ablation.py` (2 tests verts). Validation du drapeau [mesuré] :

```
arêtes 300 | max_births 0 | allow_node_growth False | births 0 | enable_growth False
3 pas ok | perte 4.6054 | arêtes finales 300 | nœuds 128
```

Le graphe ne bouge pas d'une arête. Les deux arms n'ont pas tourné : la carte n'a jamais
rempli la condition de la consigne.

| relevé | mémoire | utilisation |
|---|---:|---:|
| début de session | 1 306 Mo / 8 192 | 18 % |
| après les bancs CPU | 1 297 Mo / 8 192 | 13 % |
| fin de session | 1 668 Mo / 8 192 | 16 % |

Aucun processus de calcul étranger : c'est la bureautique et les navigateurs. La règle
posée était < 1 Go et < 10 % : elle n'est pas remplie, donc **rien n'a été lancé**. Les deux
commandes prêtes, dans `docs/ABLATION-croissance.md` :

```bash
python scripts/ablate_growth.py --arm grown  --steps 500
python scripts/ablate_growth.py --arm frozen --steps 500
```

Durée **estimée** : ~3 s/pas à 3 997 arêtes → 25 min par arm ; la règle de verdict
(≤ 2 % d'écart = la croissance n'a pas payé à ce budget) est fixée d'avance dans le
document.

### Étape C — la comparaison, terminée, et un baseline démasqué deux fois

**C.1 Le run de 500 pas est terminé [mesuré] — mon rapport d'hier était faux.**
`MYCELIA_EXIT=0`, checkpoint de fin au pas 500, 3 764 s de run, `history.json` complet. Le
7 novembre j'avais écrit « arrêté au pas 250 » : l'arrêt avait été demandé, le processus
avait déjà franchi le pas 500 et écrit son checkpoint. Corrigé dans les trois documents
concernés.

**C.2 L'init du Transformer était cassée [mesuré].** Perte initiale **21,90** pour un
uniforme à `ln 10 000 = 9,21` : embeddings `N(0,1)` et tête liée aux embeddings donnent des
logits d'écart-type ~11. Init GPT-2 (`reset_parameters`) : **9,2349 au pas 0**. Après
500 pas, seq 32, batch 8, CPU 30 ms/pas : perte **4,8725**, **ppl de validation 159,974**
(contre 6 060,503 avant la correction).

**C.3 Comparaison à tokens égaux (128 000 tokens chacun) [mesuré].**

| pas | colonie perte | colonie ppl | arêtes | Transformer perte | Transformer ppl |
|---:|---:|---:|---:|---:|---:|
| 100 | 9,0859 | 8 830 | 812 | 6,1718 | 479 |
| 200 | 8,8547 | 7 007 | 1 612 | 5,6345 | 280 |
| 300 | 8,4842 | 4 838 | 2 419 | 5,1833 | 178 |
| 400 | 8,0484 | 3 129 | 3 217 | 4,9621 | 143 |
| 500 | **7,4379** | **1 699** | **3 997** | **4,8725** | **131** |

```
colonie      ppl val  1 977,8
Transformer  ppl val    159,974     -> x12,4 en faveur du Transformer
```

**Inférence de 40 tokens, CPU, machine au repos, un forward complet par token [mesuré]** :

```
colonie      : 3,390 s ->  11,80 tok/s (3 971 arêtes vivantes)
Transformer  : 0,124 s -> 321,90 tok/s (1,68 M paramètres)
rapport      : x27,3 en faveur du Transformer
```

L'espoir « seuls les nœuds traversés calculent » ne se matérialise pas à cette échelle :
la colonie est 27 fois plus lente **en inférence** aussi, et douze fois moins bonne après
128 000 tokens. Ce n'est pas une condamnation de l'architecture (500 pas n'est pas un
régime de croissance), c'est le point de départ chiffré qui manquait au projet.

**C.4 Décodeur propre [fait].** `clean_text` est appliqué à l'affichage dans la boucle
d'entraînement (`train.py`), les dix échantillons (pas 50 à 500) sont régénérés sans
artefact, et le pas 500 a été reproduit depuis le checkpoint avec la même graine.

### Étape D — observation puis run texte : **BLOQUÉE**

L'observation 300 pas (seq 128, batch 16, tbptt 4, `configs/config_observation.yaml`) n'a
pas été lancée : c'est un **chronométrage**, et la carte n'était pas libre. Les seuils ne
sont donc pas appliqués. ETA toujours **estimée** : 3 000 pas ≈ **3,9–6,1 h**, d'après
`logs_text.txt` du 3 novembre (4 633 → 7 326 ms/pas, carte libre, tbptt 16) ; 10 000 pas
≈ 13–20 h ; le coût de `tbptt 4` n'est pas mesuré **[supposé : il réduit la mémoire du pas
sans réduire son temps]**. Aucun run de 3 000 pas lancé, aucune extension à 10 000.

### Étape E — verdict : **FAITE**

- **Palier 1 : non**, sans ambiguïté. L'exemple le plus net est le pas 500 : « Il était une
  fois rivières voitures merveilleuses colis doué coucha perdre démêler seule autres rempli
  poires enneigée droites châteaux fer jambon Éric d trompeusesaves une Lulu deux Dans
  oisillon Dodo Ilseau pourquoi Après les ta se fonds vivre quatres aller montra gant
  équilibré ». Des mots français, aucune syntaxe, des morphèmes collés
  (`trompeusesaves`, `quatres`).
- État du cerveau au pas 500 [mesuré] : 3 997 arêtes (4 006 naissances, 21 morts), 514
  nœuds vivants sur 532 slots, 2 nœuds nés d'un message, 30 morts, 30 résurrections,
  cimetière vide. E* prédit ≈ 1 000 × naissances/pas ≈ 4 000 : **atteint**.

## 3. Échantillons, tous, en clair

42 tokens chacun, amorce `Il était une fois`, graine = numéro du pas ; artefacts de
décodage retirés à l'affichage.

**Pas 50** — `Il était une foisadmorage bul vivaient attiraitait ferméaboyerève illuminantâce sembler bi fauteuil odeurastiques profond exauça rougit donnantblè hors fruitier tourbillonner carr apparu étroitchèrent dedans tourbillon allu suffprise Framboiseteaux aperçut brusquement toupindre offrirent bruy`

**Pas 100** — `Il était une fois enveloppa quant comp cabossé réconforter remerciéuoué Jojo planterlat levaitenroulaoudouN Un Croa ro échap ponts plantèrent Basilearmon voilàUtilise BongoumesTiti fois Martine voiturette]iendrait peign Quelques permettrait pes amitié apais courageuse séparerinalement`

**Pas 150** — `Il était une fois luge crinière repassé faisantouver fermier sourire Al Zizi fâcheux igloo demandeBien souhaitait quartier Cher pensent vaches Gast Dupont ryth rais glaces amer empl fissurangeantassiette lettresersonne rencontreettait mélangées poussi Fl œil création réchauffa Dansjet impressionnés`

**Pas 200** — `Il était une fois passerait Fin regardermienvelop ballon réun rouges réisous pleuv meilleursaraignéeb tandis chanta éclat| Pao leçons coeur ramasser rassuré exam peint déc ard Coin bouillir avançaitCoin baignamusa traceD géantesoignoncheurherbesesses objet`

**Pas 250** — `Il était une fois rencontraraient sœursifférent détestait brisé propreté brique perles attendait lèvres cerisesotes franavocatitait emport soudainè insista réfrigérateur cur réunissaient Pierre lionceauordball termina aimons Griquer déb adorée soucier pédaleronge glissa Peux avalé ouverte essayer`

**Pas 300** — `Il était une foiszar planter chéri illuminerce oreiller racontait jupe rivièresby foul calme gard grognpérée tomba pensant ab déchets impatétais saine pique ordinaire restent ravi debout sû miaulementenvoler jouait ramenéûte penser commençait fripé compta nidanes voudrais chapeau pirate`

**Pas 350** — `Il était une fois trouvaitarrêtèrent testelop ilsorm tuyauinquiétraper ramenapê l Toute cubesécureuil renardérateurest solution chanter tristement rennes calée nez lo éch, tandis roller caronnantes murmurer;ais vil raton incic épin mangeaenaient Lily`

**Pas 400** — `Il était une foisorigine désag ballonouta il clou toujours étroit betterave faite aideotement assister crièrent heureuse rêve qu essayait brillants coupant Peuxonses estaide Grincheux protégeait souvent enseigna jet Leurs momentères Kiki glacée asse accueillant calmement rentrer libre lourd referma policier`

**Pas 450** — `Il était une fois réfléchit fourm Bubbleeux passe déchets rond Piaf noir était quelque apprisinquiète brise musique lumièreeureusementaime tiroir Bleuette Écureuileuv approchait débr décourager préférées sansonnais Toby< exaucer clign reç contrari grognant visagevé toutes rendant besoin conn créèrent`

**Pas 500** — `Il était une fois rivières voitures merveilleuses colis doué coucha perdre démêler seule autres rempli poires enneigée droites châteaux fer jambon Éric d trompeusesaves une Lulu deux Dans oisillon Dodo Ilseau pourquoi Après les ta se fonds vivre quatres aller montra gant équilibré`

## 4. Ce qui n'a PAS été fait, et pourquoi (le STOP est un résultat)

| Non fait | Raison mesurée |
|---|---|
| ablation croissance vs graphe figé | carte à 1 297–1 668 Mo et 13–16 %, condition < 1 Go / < 10 % non remplie |
| observation 300 pas seq 128 | même raison ; c'est un chronométrage, interdit sur carte occupée |
| seuils automatiques | aucune ETA mesurée : les appliquer serait décider sur une estimation |
| run 3 000 pas, baseline 10 000 pas | dépendent des seuils |
| constante de mémoire (A.5) | protocole réduit invalide (exactitude au hasard), vrai régime ≈ 5–8 h carte libre |
| cause des pics de perte (A.3) | journaux par pas absents ; conclusion impossible sans eux |
| `max_births` 8 → 4 | dépend des seuils ; déplacer E* est un choix humain |

## 5. Livraison

- **Tests** : `pytest -q` nu → **45 passed, 1 warning en 86,68 s** [mesuré] (43 + les 2
  tests de l'ablation).
- **nouveaux fichiers de code** : `mycelia/ablation.py`, `scripts/ablate_growth.py`,
  `scripts/check_determinism.py`, `scripts/degree_stats.py`, `scripts/memory_delay.py`,
  `tests/test_ablation.py`.
- **modifiés** : `mycelia/train.py` (décodeur propre, drapeau d'ablation),
  `mycelia/config.py` et `configs/config.yaml` (`growth.mode`, `growth.frozen_edges`),
  `scripts/train_baseline.py` (init GPT-2, `--device`, `--save`).
- **documents** : `docs/MEMBRANE-frontieres.md`, `docs/ADAM-evenements.md`,
  `docs/ROUTAGE-degres.md`, `docs/ABLATION-croissance.md`,
  `docs/COMPARAISON-500.md` (réécrit), `docs/BILAN-run-texte.md` (réécrit),
  `docs/echantillons-texte.md` (régénéré), ce rapport, et la correction datée du rapport
  du 7 novembre.

## 6. Prochaine action recommandée, chiffrée, et ce qu'il faut trancher

1. **Libérer la carte** (ou déplacer le travail) : seul blocage de l'ablation et de
   l'observation. Sans cela, rien de neuf n'est mesurable.
2. **Ablation** : deux runs de 500 pas, ~25 min par arm [estimé], commandes prêtes.
3. **Observation** : 300 pas en seq 128 / batch 16 / tbptt 4, puis ETA mesurée → si
   < 48 h, lancer 3 000 pas (3,9–6,1 h [estimé]).
4. **Décision humaine** : accepter ou non que le projet continue sur une architecture qui,
   à 128 000 tokens, est 12 fois moins bonne et 27 fois plus lente en inférence que le
   Transformer de référence. C'est la question que la comparaison pose maintenant, et
   aucune ablation ne peut y répondre à la place d'un humain.
