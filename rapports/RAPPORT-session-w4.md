# Session W4 — A et C mesurés, B2 corrigé après effondrement

**Trois résultats, aucun déplacé :**

- pilote A (encodeur distillé) : **0,667** pour un seuil de 0,70 → ROUGE ;
- piste C (routage sémantique) : **0,4805** pour un seuil de 0,50 → ROUGE de
  0,019, mais l'écart aux questions sans rapport (0,371) et la couverture
  (49,4 %) passent ;
- run B2 : **effondré au pas 300** (9 558 nœuds enterrés, 462 vivants), cause
  trouvée, architecture corrigée, smoke vert, run relancé.

## 1. Pilote A — le traducteur appris ne franchit pas le mur

MLP 896→256→32, distillation de cosinus sur 1 200 phrases du corpus, 500 pas,
évaluation sur les 20 phrases tenues à l'écart.

| | Valeur | Seuil consigné |
|---|---|---|
| Spearman | **0,667** | ≥ 0,70 |
| Erreur cosinus | **0,167** | ≤ 0,15 |
| Retrieval hit@1 | 2/5 | ≥ 4/5 |
| Qwen seul | 2/5 | — |

C'est mieux que la projection linéaire (0,597) et la PCA (0,60), mais toujours
sous le seuil. **Le mur est la dimension 32**, pas la forme de la traduction :
un MLP ne crée pas l'information que 32 nombres ne peuvent pas porter. Aucun
poids de MYCELIA n'est modifié ; la greffe 1 reste fermée.

## 2. Piste C — le routage sémantique rate son propre seuil de 0,019

`mycelia/semantic_router.py` : clés aléatoires gelées 10 020×768, sélection des
1 002 nœuds les plus proches de l'embedding de la question (Gemma, seuil 5/5).

| | Valeur | Seuil |
|---|---|---|
| Jaccard moyen paraphrases | **0,4805** | ≥ 0,50 |
| Jaccard questions sans rapport | 0,109 | — |
| Écart | **0,371** | ≥ 0,30 |
| Couverture des 10 questions | **49,4 %** | ≥ 30 % |

Quatre paires sur cinq sont entre 0,50 et 0,59 ; « qu'est-ce qui m'énerve ? »
tombe à **0,207** et fait échouer la moyenne. Le seuil n'est pas bougé. C'est
une piste **presque passante**, pas une piste verte : la prochaine itération
devra être jugée sur des paraphrases inédites, avec un nouveau seuil écrit
avant.

## 3. B2 — l'effondrement, la cause, la correction

Au pas 300, le log montrait `éveillés 62` et `nœuds 462`, cimetière 9 558.
Deux causes distinctes :

1. **mort par isolement** : `kill_isolated_nodes` enterre tout nœud sans arête
   entrante ou sortante après 300 pas. À 10 020 nœuds et 2 000 arêtes, la
   majorité n'a jamais été câblée — 90 % de la capacité a été enterrée vivante.
   Un nœud dormant n'est pas un nœud mort.
2. **injection au seul nœud 0** : même sans enterrement, un nœud éveillé sans
   arête ne recevait jamais le token. Le routage était donc largement décoratif.

### Correction (mesurée)

- **injection broadcast** : le token est envoyé à **tous** les nœuds éveillés,
  pondéré par leur gate ; les arêtes ne servent plus qu'au mélange ;
- **readout moyenné** : la prédiction lit la moyenne des sorties des experts
  éveillés, pas la membrane d'un seul nœud ;
- **fin de la mort par isolement** pour un cerveau routé : la capacité reste
  fixe par conception.

### Smoke après correction — VERT

| | Avant | Après |
|---|---|---|
| Couverture | 10 020/10 020 | **10 020/10 020** |
| Enterrés au pas 100 | 0 (mais 9 558 au pas 300) | **0** |
| Perte au pas 100 | 4,4597 | **2,7801** |
| Rythme | 8,8 s/pas | **5,9 s/pas** |
| HWM | 2 070 Mo | **2 108 Mo** |
| Éveillés | 1 007 | 1 008 |

La perte chute deux fois plus vite : la capacité est réellement utilisée.
Run relancé : **PID 247374**, seuil **0,002342** inchangé, checkpoint/500.

## 4. Décision sur le run

Le run précédent a été arrêté (invalidé par l'effondrement). Le nouveau run
part de zéro et laissera `data/mem_report/twin10020.json` à la fin. Je n'ai pas
introduit de palier : ce serait un changement de protocole, et il t'appartient.
À 5-6 s/pas au début, la croissance des arêtes ralentira encore.

## Ce qui n'est toujours pas fait

- greffe 2 (D) non codée : le mur de dimension n'est pas contourné par A ;
- `[greffon]` / `[mycelia]` inactifs ; `[voix]` reste Qwen ;
- ingestion/corpus toujours non lancés (scripts absents) ;
- nom toujours sans-nom.
