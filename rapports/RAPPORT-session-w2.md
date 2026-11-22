> **Renommage —** Ce projet s'appelle désormais **Numenia** (auparavant référencé
> comme Mycelia dans les titres et la documentation). Session, chiffres, fichiers de
> données et cerveau **Nyra** inchangés. Ce document est historique : son contenu est
> conservé tel quel, et seul le nom du projet passe à Numenia.

# Session W clarifiée — 10 020 nœuds, 10 % éveillés

**Le jumeau B2 tourne. Le smoke est vert. Le chat parle.**

Le point dur annoncé — « si le run charge les 10 020 nœuds, le mmap est mal
utilisé, c'est un bug » — a été tranché par la mesure, pas par une intuition :
les 10 020 nœuds **tiennent** en 246 Mo de poids. Ce qui coûte, ce n'est pas la
capacité, c'est l'optimiseur Adam. Chiffres ci-dessous.

## 1. Architecture

| Élément | Valeur |
|---|---|
| Nœuds | **10 020** (10 × la référence 1 002) |
| Routeur | projection apprise du token courant, 32 × 10 020 |
| Éveillés par token | **1 007–1 008** (10,05 %) |
| Dormants | ni forward, ni message : arêtes filtrées aux deux bouts |
| Exploration | 4 nœuds rotatifs par forward, curseur global |
| Batch / seq | 1 / 32 (même budget de tokens que le dense) |

Le routage est **appris** : les nœuds éveillés reçoivent un poids sigmoïde qui
laisse un gradient au routeur ; la sélection dure, elle, est une politique.

## 2. Smoke 100 pas : VERT

Fichier `data/mem_report/smoke_twin10020.json`.

| Mesure | Valeur |
|---|---|
| Couverture | **10 020 / 10 020** |
| Jamais éveillés | **0** |
| Éveillés au dernier pas | 1 007 (10,05 %) |
| Concentration top-200 | **2,0 %** des éveils (uniforme : pas de collapse MoE) |
| Cimetière | 0 |
| Paramètres fp32 | **246,0 Mo** |
| RSS en régime | **1 331 Mo** (plat de 20 à 100 pas) |
| HWM pendant l'entraînement | **1 731–1 756 Mo** |
| HWM au rapport (sauvegarde comprise) | **2 070 Mo** |
| Rythme au pas 100 | 8,8 s/pas |

### Le mmap, la RAM et le vrai piège

Le protocole attendait ~1 Go. Mesuré : 2,07 Go au pic. Décomposition :

- poids des 10 020 nœuds : **246 Mo** ;
- gradients : **246 Mo** ;
- états Adam (2 ×) : **492 Mo** ;
- activations, cache, sauvegarde : le reste.

Les nœuds dormants ne sont **jamais** calculés : seul ~10 % du bank passe dans
`forward_rows`, et le RSS est **plat** (1 324 → 1 331 Mo sur 80 pas). Le coût
mémoire vient des **poids et de l'optimiseur**, pas de la capacité activée. À
cette échelle, un mmap des poids ne ferait gagner que 246 Mo et ajouterait de la
latence disque ; c'est l'optimiseur qu'il faudrait sortir de la RAM pour viser
1 Go. Je ne présente donc pas le mmap comme « bien utilisé » : il n'est **pas
utilisé**, et c'est un constat mesuré, pas un contournement.

Honnêteté : le HWM dépasse l'attendu de ~1 Go. Le run tient (7,3 Go de RAM,
4,3 Go libres) ; si le verdict exige 1 Go, la prochaine étape est un optimiseur
8 bits ou hors-mémoire, pas un réglage du seuil.

### Deux bugs trouvés par le smoke, pas par le verdict

1. **Désynchronisation `wave`/`hot`** : quand aucune arête n'est traversable,
   `_route` renvoyait un nœud chaud avec un signal vide ; le premier pas du
   run 10 020 mourait sur `baddbmm` (batch 0). Corrigé : zéro arête → zéro nœud
   chaud.
2. **Six nœuds morts-nés** avec le premier compteur d'exploration : `tbptt`
   répète les positions, donc `clock × 32 + position` ne balayait pas tout.
   Remplacé par un **curseur global** incrémenté à chaque forward : couverture
   complète au pas 80.

## 3. Run 3 000 pas : LANCÉ

```
setsid .venv/bin/python -u scripts/run_twin_10020.py --mode run
PID 245331 | log data/mem_report/twin10020.log
checkpoint data/genome_twin10020_run.pt tous les 500 pas
```

Seuil consigné avant : **perte finale ≤ 0,002342** (référence dense 1 002).
Étage 2 : inférence à **10 / 5 / 2 %** éveillé, dégradation constatée. Aucun
anneau 30 %→10 % : la couverture est déjà complète sans lui, et un anneau
changerait le coût moyen — donc le protocole.

Le run 2 004 / 50 % a été **arrêté** : la vision clarifiée le remplace. Son
smoke reste dans `data/mem_report/smoke_twin2004.json`.

Le verdict n'est pas écrit. Le rythme croît avec les arêtes (6,0 s/pas à 20,
8,8 à 100) : annoncer une heure de fin serait une invention. Le JSON
`data/mem_report/twin10020.json` apparaîtra à la fin, avec
`stage1_admissible` et, si vert, `stage2`.

## 4. Premier chat : inchangé, revalidé

`scripts/chat.py`, même hiérarchie, essayé sur les vrais modèles :

```
toi> qui es-tu ?
mycelia> [identite] Je suis sans-nom (à nommer au premier Sommeil). Né le
2026-11-13, âge mesuré 0.6 jours humains (0.00 jour CPU)...
toi> quel est mon poison préféré ?
mycelia> [qwen] Je ne peux pas m'exprimer sur des activités personnelles...
toi> quel est mon poison préféré ?
mycelia> [memoire] Je ne peux pas m'exprimer sur des activités personnelles...
```

Le contenu est faible (Qwen 0,5B) ; le **routage par étage** et la mémoire sont,
eux, vérifiés : second passage identique → `[memoire]`, sans réseau.

## 5. Tests

| Suite | Résultat |
|---|---|
| `test_chat.py` + `test_brain.py` | **13 passed, 1 skipped** |
| Smoke 10 020 | couverture **10 020/10 020**, routage uniforme |
| Test de reprise routée (session précédente) | OK |

## État consolidé

| Élément | État |
|---|---|
| Jumeau B2 10 020 / 10 % | **run en cours**, PID 245331 |
| Smoke B2 | **VERT**, HWM 2 070 Mo, couverture totale |
| Chat | **VERT**, 13 tests |
| Nom | toujours **sans-nom** |
| Ingestion / corpus | non lancées, scripts toujours absents |
| Sens | non intégrés |

## Prochaine action

1. lire `data/mem_report/twin10020.json` à la fin du run ;
2. si `stage1_admissible` : lire l'étage 2 (10/5/2 %) ;
3. si échec : consigner B2, puis routage par similarité de question
   (EmbeddingGemma est vert), puis nested dropout ;
4. ne pas toucher au seuil 0,002342 après coup.
