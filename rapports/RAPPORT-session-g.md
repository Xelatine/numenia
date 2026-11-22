> **Renommage —** Ce projet s'appelle désormais **Numenia** (auparavant référencé
> comme Mycelia dans les titres et la documentation). Session, chiffres, fichiers de
> données et cerveau **Nyra** inchangés. Ce document est historique : son contenu est
> conservé tel quel, et seul le nom du projet passe à Numenia.

# Rapport — session G « le temps »

**Date** : 10 novembre 2026. **Machine** : portable, RTX 3070 Laptop (8 192 Mio),
pilote 610.88, torch 2.11.0+cu128, Python 3.12.10. **Commit de départ** : `5081127`.

Chaque chiffre porte son étiquette : **[mesuré]**, **[estimé]**, **[supposé]**.

---

## 0. Avertissement de mesure — à lire avant tout chiffre de temps

La règle 7 exige une occupation étrangère **< 1 Go** pour une campagne de
chronométrage. Elle n'a **pas** été respectée, et ce n'est pas contournable :
pendant toute la session la carte a porté entre **1 854 et 7 728 Mio** occupés par
des applications utilisateur (navigateurs, Steam, Discord, Epic, overlay NVIDIA),
avec un GPU à **48-78 %** d'utilisation. Conditions complètes :
`data/mem_report/session-g_conditions.txt`.

Conséquences, à ne jamais oublier en lisant ce qui suit :

1. Les **mémoires** (pic alloué) restent fiables : ce sont nos allocations, pas
   celles des autres processus. Elles reproduisent d'ailleurs exactement la
   session F : 352,45 Mo à 1 000 arêtes, 311,63 Mo à 3 000 avec le drapeau ON.
2. Les **temps absolus sont gonflés et bruités**. Preuve interne : à 1 000 arêtes
   le mode off donne 7 963 ms/pas, et à 2 000 arêtes 7 602 ms/pas — un pas devient
   *plus rapide* avec deux fois plus d'arêtes. Le bruit atteint donc ±20 % et rend
   toute comparaison à la session F illégitime.
3. En revanche les **rapports mesurés dos à dos** (même séance, mêmes conditions,
   quelques minutes d'écart) gardent leur sens : ils sont l'objet de ce rapport,
   et ils sont reproduits deux fois pour le levier (a).

---

## 1. Tableau OFF / ON / ON+sélectif

Config : 532 nœuds, croissance coupée (`--no-growth`), 2 pas mesurés par point,
un processus par point, graine 7 pour le graphe synthétique. **[mesuré]**

| arêtes | seq/b | OFF | ON (complet) | ON+sélectif | ×ON | ×sélectif | pic OFF | pic ON | pic sélectif |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 000 | 32/b8 | 7 963 ms | 8 967 ms | **8 331 ms** | ×1,13 | **×1,05** | 352,45 Mo | 187,02 Mo | 292,17 Mo |
| 2 000 | 32/b8 | 7 602 ms | 11 293 ms | **11 049 ms** | ×1,49 | **×1,45** | 2 376,13 Mo | 264,47 Mo | 919,84 Mo |
| 3 000 | 32/b8 | 10 409 ms | 16 155 ms | **14 590 ms** | ×1,55 | **×1,40** | 4 212,70 Mo | 311,63 Mo | 1 183,84 Mo |
| 3 000 | **128/b16** | non mesuré | 64 542 ms | 67 209 ms | — | — | — | 483,73 Mo | 1 695,71 Mo |

Lecture honnête :

- Le sélectif **gagne peu en temps** : 7,1 % plus rapide que le complet à
  1 000 arêtes, 2,2 % à 2 000, **9,7 %** à 3 000. L'estimation de la consultation
  (+10-25 % de surcoût au lieu de +91 %) n'est donc **pas** tenue : le surcoût
  réel du sélectif est de +40 à +45 % à 2 000-3 000 arêtes, pas +10-25 %.
- Le sélectif **coûte cher en mémoire** : son pic est 2,6× (1 000), 3,5× (2 000)
  et 3,8× (3 000) celui du checkpoint complet. C'est le prix attendu — les
  activations de nœud restent dans le graphe — mais il est plus élevé que prévu.
- **À seq 128/b16, le sélectif perd sur les deux axes** : +4,1 % de temps *et*
  3,5× le pic (1 695,71 vs 483,73 Mo). Le mécanisme est clair : par pas, les
  activations de nœud conservées valent `positions × hops × nœuds × batch × dim`,
  et à 128 positions ce terme dépasse tout ce que le replay économise.

## 2. Verdict du critère de succès (déclaré d'avance)

Critère : à 3 000 arêtes, surcoût temps du sélectif **≤ ×1,4** *et* pic **< 1 500 Mo**.

| critère | seuil | mesuré | verdict |
|---|---|---|---|
| surcoût temps | ≤ ×1,4 | **×1,4016** | **RATÉ de 0,1 %** |
| pic | < 1 500 Mo | **1 183,84 Mo** | tenu |

Je ne l'arrondis pas en sa faveur : ×1,4016 est au-dessus de ×1,4. C'est, dans
l'incertitude de cette carte (±20 % de bruit sur le temps), indistinguable du
seuil — ce qui veut dire que **le sélectif n'apporte pas la marge attendue**, et
qu'il ne la justifie pas à seq 128 où il est plus lent et 3,5× plus gourmand.

**Recommandation** : pour le run texte (seq 128/b16), garder le **checkpoint
complet** (`--checkpoint-turns` seul). Le sélectif reste utile si la contrainte
est la mémoire *et* que la séquence est courte (≤ 32), cas où il rend le pas 7-10 %
plus rapide que le complet pour un pic divisé par 3,6.

## 3. Les autres leviers temps

### a. Taille de chunk (vectorisation du scatter) — **gain réel, le seul de la session**

Le message passing découpe les arêtes en chunks (`edge_chunk_size`, défaut 512) ;
chaque chunk lance ses kernels. Mesuré à 3 000 arêtes, seq 32/b8, mode off, deux
séries dos à dos **[mesuré]** :

| chunk | ms/pas | écart vs 512 | pic |
|---:|---:|---:|---:|
| 512 (défaut) | 12 778,5 | — | 4 212,70 Mo |
| 1 024 | 12 343,9 | −3,4 % | 4 209,97 Mo |
| **2 048** | **6 955,5** | **−45,6 %** | 4 261,32 Mo |
| 8 192 | 9 574,1 | −25,1 % | 4 277,25 Mo |

Le gain est **conforme au diagnostic** (petits kernels dominés par les
lancements), il est **gratuit en mémoire** (+1,2 % de pic) et il a un optimum :
au-delà de 2 048 le pic stagne mais le temps remonte (moins de parallélisme,
kernels trop gros). **Défaut inchangé** (512) : c'est un levier opt-in
(`--chunk 2048`), à valider par une mesure à seq 128 avant tout run — je ne l'ai
mesuré qu'à seq 32.

### b. Dataloader préchargé / pinned memory — **non pertinent, mesuré**

Un lot coûte **0,0742 ms** (`TextDataset.sample_batch`, seq 128/b16, 2 000
prélèvements) soit **0,0007 %** d'un pas de 10 409 ms. Il n'y a d'ailleurs aucun
`DataLoader` dans le projet : le corpus est un tenseur d'entiers en RAM, prélevé
par `randint` + indexation. Le GPU n'attend jamais le CPU. **Rien implémenté**,
et c'est la conclusion correcte.

### c. `torch.compile` — **échec, abandonné proprement**

Tenté sur `_route` (le message passing), config courte (1 000 arêtes, seq 32/b8,
1 pas), mode par défaut, borne de 120 s **[mesuré]** : erreur Dynamo au premier
appel, aucun résultat produit (27,5 s puis sortie vide). La cause est structurelle
et était prévue : la topologie change à chaque pas, donc les formes des tenseurs
d'arêtes changent, donc il faut recompiler — exactement le piège identifié. Le
message d'erreur brut est conservé dans `session-g.json` (`leviers_temps`).

### d. CUDA graphs — **analyse seule, aucune implémentation**

Inapplicable en l'état, et je documente pourquoi plutôt que de l'essayer :
un CUDA graph capture une suite d'appels **et d'adresses** fixes. Ici (i) le
nombre d'arêtes actives `eids` change à chaque tour, (ii) les naissances
réaffectent `edge_index` en place, (iii) le checkpointing rejoue un forward
supplémentaire au backward. Il faudrait une capture par forme, donc un cache de
graphes indexé par (nombre d'arêtes actives, forme de la frontière) — c'est-à-dire
reconstruire `torch.compile` à la main, avec le même échec attendu. Non tenté.

## 4. Durée du run texte 3 000 pas, recalculée

Base mesurée aujourd'hui, seq 128/b16, 3 000 arêtes, checkpoint complet :
**64 542 ms/pas** **[mesuré, sous contention]**.

- 3 000 pas × 64,542 s = **193 626 s ≈ 53,8 h** **[estimé depuis le mesuré]**.
- Session F, carte plus propre : 11,2 s/pas → **≈ 9,3 h** **[mesuré en F, estimé
  pour 3 000 pas]**. L'écart (×5,8) est l'occupation étrangère de la carte, pas
  un changement du code : la même config donne le même pic mémoire à 0,01 Mo près.

**Conséquence directe, et c'est la phrase la plus importante de la session : le
temps mesuré aujourd'hui ne justifie pas de lancer le run.** Lancer 54 h sur une
carte à moitié occupée par des navigateurs, c'est payer six fois le prix d'un run
qui en vaut neuf heures. Ce n'est pas un « non » définitif : c'est un « pas dans
ces conditions ».

## 5. Préparation du run — commande exacte (non lancée)

```bash
python -m mycelia.train --data text --scale 1 --steps 3000 \
  --seq-len 128 --batch-size 16 --checkpoint-turns \
  --save-every 250 --seed 42
```

- `--checkpoint-turns` **seul** : le sélectif est écarté à seq 128 (§1, §2).
- `--chunk 2048` **non inclus** : le gain est mesuré à seq 32 uniquement ; l'ajouter
  sans mesure à seq 128 serait appliquer un chiffre hors de son domaine.
- `--save-every 250` : la sauvegarde réelle suit l'horloge (`save_seconds`,
  600 s par défaut) ; ce drapeau n'est qu'un déclencheur supplémentaire.
- Graine fixée (42) : la reprise est prouvée à 1e-6, y compris avec les drapeaux
  de checkpoint (t6 de la session F, s4 de cette session).

**Plan de surveillance** (journal horaire, une ligne par heure) :
`date, step, loss, edges, nodes, gpu_used_mib, gpu_temp_c, commit_mb`.
La dérive à surveiller est le **COMMIT** (mémoire du processus), pas le working
set alloué : c'est la leçon WDDM des sessions précédentes — un working set plat
avec un commit qui monte annonce l'OOM.

**Critère de verdict Palier 1** (rappel, inchangé) : cinq mots grammaticalement
enchaînés en complétion, ou l'honnêteté du « presque ».

**État** : *prêt, en attente de décision humaine.* Le run n'est pas lancé et ne le
sera pas sans un « lance » explicite.

## 6. Trous nommés

1. **Chronométrage contaminé** : aucun chiffre de temps de cette session n'est
   comparable aux sessions précédentes ; seuls les rapports internes le sont.
2. **seq 128 sous-mesuré** : un seul point (3 000 arêtes), pas de mode off
   (impossible : le pic off à 3 000 arêtes est 4 212,70 Mo, la carte n'avait plus
   la place à ce moment), et le levier chunk n'y a pas été testé.
3. **8 000 arêtes à seq 128 non refait** : la session F l'avait mesuré à
   101,8 s/pas ; refaire ce point aurait coûté plus de 200 s par pas sur une carte
   à 48-78 % d'occupation étrangère.
4. **L0 (télémétrie conservée) aux grandes tailles** : le groupe `l0` du pilote
   existe mais n'a pas été exécuté, faute de temps de carte propre.
5. **Levier (a) à seq 128** : non mesuré, donc non recommandé pour le run.

## 7. Ce que la session a produit

- `--checkpoint-selective` (défaut OFF, subordonné à `--checkpoint-turns`),
  `_hop` scindé en `forward_rows` + `_route`, cinq tests s1-s5 verts
  (dont l'encadrement strict du pic et la continuité de reprise à 1e-6).
- 12 tests CPU verts, 2 tests GPU verts, aucune régression sur t1-t8.
- `scripts/sessiong.py` (pilote : un processus par point, écriture après chaque
  point) et `data/mem_report/session-g.json` (11 points, 4 leviers).
- Un levier temps réellement gagnant : `--chunk 2048`, −45,6 % à 3 000 arêtes
  seq 32, à mémoire constante.
- Un verdict qui contredit l'hypothèse de départ : le checkpointing sélectif
  n'atteint pas son critère (×1,4016 pour ×1,4 exigé) et perd à seq 128.
