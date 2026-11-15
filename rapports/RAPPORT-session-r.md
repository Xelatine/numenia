# Session R — architecture tiercée out-of-core : ce qui tient, et ce qui casse

**Honnêteté en tête** : MYCELIA n'est pas Claude ni GPT-4. Ces modèles doivent
leur intelligence à des centaines de milliards de paramètres ; BIO-1 ne peut pas
les imiter en taille. La cible est le compromis évolutif : découpler la mémoire
(vécue, persistante, sur disque) du calcul (borné, sélectif). Ce rapport mesure
l'infrastructure de ce compromis — et deux de ses leviers échouent.

## 1. Débit disque (fondation des projections)

- SSD SATA `Verbatim Vi560`, `ROTA 0` (pas de rotation), 802 Go libres.
- Écriture directe 500 Mo : 0,165 s ≈ **3,2 Go/s** (rafale, cache du disque).
- `hdparm -t` : **Permission denied** (root indisponible).
- `dd if=… iflag=direct` : **« Invalid input »** — le système de fichiers refuse
  l'I/O direct sur ce fichier. La lecture disque froide n'a donc **pas** été
  mesurée ; le 7,7 Go/s observé est du page cache, pas du disque.
- **Conséquence** : toutes les projections de latence disque restent non
  calibrées. Le seul chiffre solide est l'écriture en rafale.

## 2a. Activité par hop et par position (le chiffre qui manquait)

Sur le génome réel (1 129 nœuds, 6 246 arêtes), forward de 16 positions :

| Hop | Fraction moyenne | Max |
|---|---|---|
| 0 | 0,025 | 0,025 |
| 1 | 0,167 | 0,167 |
| 2 | 0,591 | 0,591 |
| 3 | 0,875 | 0,875 |
| 4-7 | 0,889 | 0,889 |

La fraction cumulée de la session Q (0,999) cachait cette montée : **dès le hop
3, 87,5 % du graphe propage**. Un top-k à 5 % couperait donc l'essentiel de la
propagation, pas une frange inactive. Le levier ×20 annoncé n'existe pas dans
ce régime.

## 2b-2c. Top-k branché : REJETÉ par la mesure

Implémenté dans le vrai forward (`Brain.topk_fraction`, sélection détachée sur
l'énergie de membrane, routeur calculé sur tous les nœuds). Un bug réel a été
trouvé par le test : `hot` était réduit sans réduire `wave` (667 lignes contre
564 adresses) → corrigé.

Perte sur 4 batchs fixes du graphe entraîné (run 3 000, perte de référence
0,002342) :

| Fraction | Perte | Écart relatif |
|---|---|---|
| aucune | 0,002342 | — |
| 50 % | 1,5943 | **×679** |
| 25 % | 3,0917 | ×1 319 |
| 10 % | 4,3253 | ×1 846 |
| **5 %** | **4,2084** | **×1 796** |
| 1 % | 6,1622 | ×2 630 |

Même à 50 %, la fonction apprise est détruite. **Le top-k n'est pas un défaut de
config** : il reste désactivé par défaut. Réserve honnête : la sélection utilise
l'énergie de membrane comme proxy ; une sélection sur les magnitudes du routeur
n'a pas été testée. Mais l'ordre de grandeur (×1 800 à 5 %) interdit de la
présenter comme « un peu de perte ».

## 3. Format `genome.myc/` — PROUVÉ en fp32, rejeté en fp16

`mycelia/genome_io.py`, format `myc-1`, converti depuis `genome_run3000.pt` :

```
meta.json          config, offsets, md5 par fichier, hash global
topology.csr       indptr int32 [N+1], indices int32 [E], poids fp32 [E]
embeddings.bin     organes des nœuds [N, 6401] fp32, aligné 4 Ko
token_embedding.bin, output_head.bin, misc.bin
energy.bin / salience.bin (fp16), cemetery.bin
```

| Conversion | max_abs logits | Verdict |
|---|---|---|
| poids **fp16** | 5,905e-03 | **REJETÉ** (> atol 1e-3) |
| poids **fp32** | **4,768e-06** | **ACCEPTÉ** |

Taille mesurée à 1 129 nœuds : **29 501 537 o**, hash global
`34caba7813561650030199010018e7ae`. Le format ne change pas le programme ; la
version fp16, elle, le changeait — c'est la mesure qui l'a dit, pas la lecture.

## 3b. int8 — accepté pour le disque

Quantification par ligne de nœud (échelle symétrique par nœud) :

- forward d'équité : écart relatif moyen **0,546 %** (< 5 %) → **accepté** ;
- golden retrieval : **2/5 avant, 2/5 après** → inchangé ;
- taille divisée par 4 (organes), contre un facteur 2 pour la topologie.

## 4. mmap — seuil de bascule CONSTATÉ à 200 k nœuds

Graphe **SIMULÉ** à 200 000 nœuds, organes int8 (1,28 Go) :

| | Valeur |
|---|---|
| Écriture | 2,8 s |
| Chargement complet | 0,59 s, **RSS +1 220,9 Mo** |
| mmap froid | **2,24 µs/ligne** (10 000 lignes) |
| Seuil projeté fp32 | ~207 000 nœuds (5,3 Go dispo) |
| Seuil projeté int8 | ~828 000 nœuds |

Le seuil est donc **constaté par la RSS mesurée**, pas décrété : à 200 k nœuds
int8, le cerveau tient en 1,22 Go ; à 1 M, il faut le mmap. Réserve : les
lectures « froides » restent aidées par le page cache (`fadvise` ne garantit pas
l'éviction) ; le débit SATA réel n'est pas mesuré (voir §1).

## 5. Sommeil / réorganisation — gain non prouvé, non adopté

Simulation à communautés plantées, 50 000 nœuds, 2 500 lignes chaudes :

- ordre aléatoire : 5,8-7,7 ms ; ordre par communauté : 5,0-5,1 ms ;
- **gain 12 % à 35 % selon la passe** — instable, dominé par le cache, sous le
  seuil de 20 % exigé pour adopter.

Décision : la réorganisation physique reste **optionnelle**, pas un défaut du
Sommeil. Le compactage et la consolidation épisodique→synapses ne sont **pas**
implémentés dans cette session.

## 6. Dimensionnement (calcul exact, pas une phrase)

Organes = 6 401 valeurs/nœud (dim 32) ; topologie = 20 arêtes × 12 o.

| Nœuds | Organes fp32 | Organes int8 | Topologie | Fenêtre 5 % int8 |
|---|---|---|---|---|
| 10 000 | 0,26 Go | 0,06 Go | 2,4 Mo | 3,2 Mo |
| 100 000 | 2,56 Go | 0,64 Go | 24 Mo | 32 Mo |
| 1 000 000 | 25,6 Go | 6,4 Go | 240 Mo | 320 Mo |

## Le chiffre final — et sa limite

**Le « quelques centaines de Mo de RAM pour un cerveau arbitraire » n'est pas
établi.** Il dépendait du top-k, qui est rejeté : sans sélecteur fonctionnel,
la fenêtre active n'est pas 5 % du graphe, elle est le graphe. Ce qui est
acquis, mesuré :

- un cerveau de 200 000 nœuds tient en **1,22 Go** d'organes int8, chargé et
  vérifié ;
- à 1 000 000 de nœuds, le disque est de **6,4 Go** (organes) + 240 Mo
  (topologie), et le mmap devient nécessaire ;
- le format `.myc` préserve le forward à 4,8e-6 près, et l'int8 à 0,55 %.

La capacité devient une variable de disque ; **la sélectivité, elle, reste un
problème ouvert** — c'est le résultat principal de la session R.

## Ce qui n'est pas fait (et ne doit pas être maquillé)

- Encodeur toujours ROUGE : ingestion toujours fermée.
- Mesures d'échelle sur graphe **simulé** ; le graphe réel a 1 129 nœuds.
- Débit lecture disque non mesuré (O_DIRECT refusé, root absent).
- Compactage du Sommeil non implémenté ; réorganisation non adoptée.
- Top-k mesuré sur la tâche jouet (perte basse, signal net), pas sur le texte :
  le rejet à ×1 800 rend la campagne texte inutile.
- Aucune promesse « type Claude » : le créneau reste mémoire vécue par watt,
  pas intelligence brute.
