# Session du 8 novembre 2026 (suite) — l'ablation bloquée par la carte, et ce qui a été livré

**8 novembre 2026, 20 h.** Consigne : l'ablation centrale en priorité absolue, puis
l'observation, puis le run texte, avec une règle de carte révisée (aucun processus étranger
à plus de 5 % de calcul, VRAM libre ≥ besoin + 1,5 Go). Étiquette de chaque chiffre :
**[mesuré]**, **[estimé]**, **[supposé]**.

## 1. Verdict en une phrase

**L'ablation n'a pas pu être mesurée** : la carte a échoué la règle révisée à 276 Mo près
et avec 25 % d'utilisation due à 29 processus étrangers, donc aucun des deux arms n'a été
lancé — mais le code de l'ablation est validé, la règle de verdict est fixée d'avance, et
le journal par pas que l'analyse des pics réclamait est désormais livré et vérifié.

## 2. La condition de carte, mesurée et consignée

| Grandeur | Valeur mesurée |
|---|---|
| VRAM utilisée | **1 804 Mo / 8 192** |
| VRAM libre | **6 224 Mo** |
| Utilisation GPU | **25 %** |
| Processus de calcul listés | 29, tous étrangers (Firefox, Tor Browser, Discord, Edge, Steam, overlays) |
| Besoin de l'arm grandi | **4 908 Mo** réservés, mesurés au pas 500 du run comparatif (`GPU 85/4908 Mo`) → arrondi 5 000 |
| Marge exigée | 1 500 Mo |
| Seuil | **6 500 Mo libres** |

- **(a) aucun processus étranger > 5 %** : **faux** (25 % d'utilisation, 29 processus).
- **(b) libre ≥ besoin + marge** : **faux** — 6 224 < 6 500, il manque **276 Mo**.

Aucun processus n'a été tué. Conformément à la consigne, le lancement s'arrête là. Les
relevés sont écrits dans `docs/ABLATION-croissance.md` §3, avec ce qu'il faut pour
débloquer (libérer ≥ 300 Mo et faire tomber l'utilisation sous 5 %).

## 3. Étape par étape

### Étape 1 — ablation centrale : **CODE VALIDÉ, MESURE BLOQUÉE**

- `mycelia/ablation.py` + `growth.mode: frozen_random` + `scripts/ablate_growth.py`.
- Deux tests dédiés (`tests/test_ablation.py`) : le graphe est figé, sans self-loop,
  chaque nœud a une arête entrante, le chemin de départ survit ; un cerveau figé ne gagne
  ni arête ni nœud en cinq forwards.
- Validation du drapeau sur un cerveau de 128 nœuds [mesuré] :

```
arêtes 300 | max_births 0 | allow_node_growth False | births 0 | enable_growth False
3 pas ok | perte 4.6054 | arêtes finales 300 | nœuds 128
```

- Les deux arms n'ont pas tourné. Durée attendue : ~25 min par arm **[estimé]** à
  ~3 s/pas pour 3 997 arêtes.
- La règle de verdict reste celle fixée d'avance : écart ≤ 2 % → la croissance n'a pas
  payé à ce budget, et il faut alors rejouer les deux arms à 2 000 pas (~2 h par arm
  **[estimé]**) avant de conclure.

### Étape 2 — observation : **NON LANCÉE**

C'est un chronométrage : interdit sur une carte qui échoue la condition. Donc ni la courbe
ms/pas vs arêtes, ni l'ETA bornée, ni le coût de `tbptt 4` contre `tbptt 16` ne sont
mesurés. Le `[supposé]` du 8 novembre (« tbptt 4 réduit la mémoire sans réduire le temps »)
**reste étiqueté supposé** : il n'a pas été transformé en mesure, et il ne doit pas être lu
comme tel. ETA toujours **estimée** : 3 000 pas ≈ 3,9–6,1 h ; 10 000 pas ≈ 13–20 h.

### Étape 3 — run texte 3 000 pas : **NON LANCÉ**, mais son instrument est livré

Le journal réclamé par l'analyse A.3 existe désormais : `mycelia/train.py` écrit
**`data/per_step.csv`**, une ligne par pas, onze colonnes, en append sur reprise et
tronqué sur run neuf. Vérifié sur un run jouet de 6 pas, chemins redirigés pour ne pas
toucher aux artefacts réels [mesuré] :

```
step,loss,edges,nodes,edge_births,node_births,edge_deaths,node_deaths,node_revivals,cemetery,lr
1,4.6156816482543945,18,512,6,0,0,0,0,0,0.001
2,4.592458724975586,28,514,16,2,0,0,0,0,0.001
...
6,4.5641655921936035,60,514,48,2,0,0,0,0,0.001
```

Ce que ce journal permet, et ce qu'il ne permet pas : il tranchera la cause des pics sur
les runs **à venir**. Il ne rétro-instrumente pas le run v5, qui n'a écrit que des relevés
au pas 100 : la cause des pics 1 300 et 1 800 reste **non concluante**, et l'hypothèse
« élagage » reste une hypothèse.

### Étape 4 — verdict : **RIEN DE NOUVEAU, ET C'EST DIT**

- Pas de verdict d'ablation : la mesure n'existe pas.
- Palier 1 inchangé : **non**, sur les échantillons du pas 500 (aucune nouvelle génération
  n'a été produite cette session).
- Le critère « une séquence de 5 mots grammaticalement enchaînés = presque » n'a pas eu
  l'occasion de s'appliquer.

## 4. Ce qui n'a PAS été fait, et pourquoi

| Non fait | Raison mesurée |
|---|---|
| arm `grown` 500 pas | condition de carte (a) et (b) fausses, 276 Mo manquants |
| arm `frozen` 500 pas | même raison |
| arms à 2 000 pas | dépend du verdict à 500 pas, qui n'existe pas |
| variante « grandi puis figé puis réentraîné » | dépend de la carte et du temps ; non entamée |
| observation 300 pas, coût `tbptt 4` vs 16 | chronométrage interdit sur carte en condition échouée |
| run 3 000 pas, baseline 10 000 pas | dépendent des seuils, qui dépendent de l'observation |

## 5. Livraison

- **Tests** : `pytest -q` nu → **45 passed, 1 warning en 89,97 s** [mesuré], après l'ajout
  du journal par pas.
- **Code de cette session** : `mycelia/train.py` (journal `data/per_step.csv`,
  `per_step_line`, ouverture/fermeture propre), `docs/ABLATION-croissance.md` (règle de
  carte révisée et refus mesuré), `docs/ADAM-evenements.md` (mise à jour datée : le journal
  existe, les pics anciens restent inexpliqués), ce rapport.
- **Déjà livré le même jour, avant le blocage** : `mycelia/ablation.py`,
  `scripts/ablate_growth.py`, `scripts/check_determinism.py`, `scripts/degree_stats.py`,
  `scripts/memory_delay.py`, `tests/test_ablation.py`, init GPT-2 du baseline,
  `docs/MEMBRANE-frontieres.md`, `docs/ROUTAGE-degres.md`, `docs/COMPARAISON-500.md`.

## 6. Prochaine action, chiffrée

1. **Libérer la carte** : ≥ 300 Mo de VRAM et utilisation < 5 % (fermer un ou deux
   navigateurs à accélération matérielle, Discord, les overlays Steam/Epic). Re-mesurer
   avec la même commande, consigner les deux valeurs dans `docs/ABLATION-croissance.md`.
2. **Lancer les deux arms** : `--steps 500` chacun, ~25 min par arm **[estimé]**.
3. **Appliquer la règle** : écart ≤ 2 % → rejouer à 2 000 pas (~2 h par arm **[estimé]**)
   avant tout verdict ; écart > 2 % en faveur du grandi → documenter comme résultat
   fondateur, courbes et distribution des degrés à l'appui.
4. **Ensuite seulement** : observation 300 pas (seq 128 / tbptt 4) et son ETA mesurée.

À trancher humainement : **faut-il attendre une carte libre, ou déplacer le travail sur une
machine dédiée ?** C'est la troisième session de suite où le même blocage coûte la mesure
principale, et aucun réglage logiciel ne le contourne.
