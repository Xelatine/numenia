> **Renommage —** Ce projet s'appelle désormais **Numenia** (auparavant référencé
> comme Mycelia dans les titres et la documentation). Session, chiffres, fichiers de
> données et cerveau **Nyra** inchangés. Ce document est historique : son contenu est
> conservé tel quel, et seul le nom du projet passe à Numenia.

# Session W — 2 004 nœuds, la moitié éveillée, et le premier chat

**Deux livraisons, un run en cours :**

- le jumeau B1 est **construit et lancé** (smoke vert : 2 004/2 004 nœuds
  couverts, 1 003 éveillés par token, HWM **703 Mo** mesuré) ;
- le premier chat **fonctionne** : `[identite]` → `[qwen]` → `[memoire]`,
  hors réseau, avec écriture de chaque échange ;
- le verdict d'admissibilité du run 3 000 pas **n'est pas encore écrit** :
  il tourne, PID 243610.

## 1. Architecture : routage conditionnel, pas amputation

Le gate existant (sigmoïde, seuil 0,1 jamais atteint) ne pouvait pas servir tel
quel : il décide *par nœud*, pas *par entrée*. Un routeur a été ajouté.

| Élément | Implémentation |
|---|---|
| Capacité | **2 004 nœuds** fixes (`max_node_births=0`) |
| Routeur | `router_w` (32 × 2 004) + `router_b`, projection du **token courant** |
| Sélection | top-k dur des nœuds vivants, `k = round(0,50 × vivants)` |
| Toujours éveillés | nœud d'entrée et nœud de sortie |
| Dormants | ni forward, ni message : **arêtes filtrées aux deux bouts** |
| Gradient du routeur | poids sigmoïde appliqué à la sortie des nœuds éveillés |

Deux choix à consigner, pas à cacher :

1. **batch 1, seq 32** (au lieu de batch 2 × seq 16 du dense). Avec un masque
   partagé sur un batch, deux questions différentes ne peuvent pas avoir deux
   moitiés différentes : le « 50 % par question » serait faux. Même budget de
   tokens par pas (32), routage réellement conditionnel.
2. **le routage lit le token courant**, jamais la séquence entière : pas de fuite
   du futur dans le choix des nœuds.

Limite assumée : `route_fraction` exige `batch_size=1` et le dit par une erreur
explicite, plutôt que de simuler un routage par lot.

## 2. Smoke 100 pas : VERT

Fichier `data/mem_report/smoke_twin2004.json`.

| Mesure | Valeur |
|---|---|
| Nœuds couverts | **2 004 / 2 004** |
| Jamais éveillés | **0** |
| Éveillés au dernier pas | **1 003** (50,05 %) |
| Nœuds vivants | 2 004, cimetière 0 |
| Arêtes | 731 au pas 100 (812 après la passe de croissance) |
| HWM mesuré | **703 Mo** |
| Rythme au pas 100 | 7,12 s/pas |
| Perte au pas 100 | 4,3808 (début d'apprentissage, sans valeur de verdict) |

Le HWM attendu (~1,5 Go, 2× le dense) n'est **pas** observé : 703 Mo. Explication
mesurée, pas supposée : au pas 100 le graphe ne porte que 731 arêtes et la moitié
des nœuds est dormante ; le pic suivra la croissance des arêtes. Le chiffre
consigné est celui du smoke, pas une projection.

## 3. Run 3 000 pas : LANCÉ, verdict en attente

```
setsid .venv/bin/python -u scripts/run_twin_2004.py --mode run
PID 243610 | log data/mem_report/twin2004.log
checkpoint data/genome_twin2004_run.pt tous les 500 pas
```

Seuil consigné **avant** : perte finale ≤ **0,002342**, la référence dense
1 002 — pas 2× le seuil, pas un réglage après coup. Étage 2 (fractions
100/50/25/10 %) ne s'exécutera que si l'étage 1 est vert.

Le rythme croît avec les arêtes (3,97 s/pas à 20, 7,12 à 100) : annoncer une
heure de fin serait une invention. Le JSON `data/mem_report/twin2004.json`
s'écrira à la fin, avec `stage1_admissible` et, si vert, `stage2`.

## 4. Premier chat : fonctionnel, hors réseau

`scripts/chat.py`, REPL texte, testé sur les vrais modèles :

```
toi> qui es-tu ?
mycelia> [identite] Je suis sans-nom (à nommer au premier Sommeil). Né le
2026-11-13, âge mesuré 0.6 jours humains (0.00 jour CPU). Sexe déclaré :
indéterminé. Génome myc-1:34caba7813561650030199010018e7ae.
toi> quelle est ma couleur préférée ?
mycelia> [qwen] Je suis un chat et je n'ai pas de couleur préférée...
toi> quelle est ma couleur préférée ?
mycelia> [memoire] Je suis un chat et je n'ai pas de couleur préférée...
```

- **identité** : `self.json`, aucun modèle, aucun réseau ;
- **niveau 1** : retrieval Gemma, seuil **0,30** calibré la session précédente ;
- **niveau 2** : Qwen 0,5B GGUF local ;
- **niveaux 3-4** : refus explicite, aucun client instancié ;
- **mémoire** : chaque échange est encodé (la question porte l'index, le texte
  porte `Q:`/`R:`) et sauvé dans `data/memory_chat.npz` — le second passage
  identique ressort en `[memoire]`.

La réponse de Qwen est faible (0,5B) ; c'est un fait du modèle, pas du routage.

## 5. Tests

| Suite | Résultat |
|---|---|
| `tests/test_chat.py` | **6/6** |
| `tests/test_brain.py` + `test_external.py` | **11 passed, 1 skipped** |
| Golden Gemma (session U) | 5/5, inchangé |
| Smoke routage | **couverture 2 004/2 004** |

Le test « zéro réseau » intercepte `socket.socket` et vérifie que l'escalade
3-4 refuse sans jamais appeler : c'est le comportement testé, pas une intention.

## État consolidé

| Élément | État |
|---|---|
| Jumeau B1 2 004 / 50 % | **run en cours** (PID 243610), verdict non écrit |
| Smoke routage | **VERT**, HWM 703 Mo |
| Chat | **VERT**, 6/6, essayé sur modèles réels |
| Identité | **hors ligne**, âge mesuré |
| Nom | toujours **sans-nom** (décision utilisateur) |
| Ingestion / corpus | toujours non lancées (verdict B1 en attente) |
| Sens | non intégrés |

## Prochaine action

1. lire `data/mem_report/twin2004.json` quand le run finit ;
2. si `stage1_admissible` : lire l'étage 2, puis seulement envisager l'ingestion
   (dont les scripts `run_ingestion.py` / `run_corpus.py` restent **absents**) ;
3. si échec : consigner B1 comme échec, puis nested dropout, puis Top-KAST —
   sans toucher au seuil.
