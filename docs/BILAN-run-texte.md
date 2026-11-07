# Bilan du run texte — 8 novembre 2026

**Aucun run texte long (3 000 pas) n'a été lancé.** Ce document dit pourquoi, avec les
mesures, et ce qui a été mesuré à la place.

## 1. Où en est la décision des seuils

Le plan prévoyait : observation de 300 pas en régime texte réel (seq 128, batch 16,
tbptt 4) → ETA bornée → seuils automatiques. **L'observation n'est toujours pas lancée**,
et la raison est mesurée : la carte n'est pas libre.

| relevé | mémoire utilisée | utilisation |
|---|---:|---:|
| 7 novembre, pendant le run | 7 929 Mo / 8 192 | 90 % (jeu, PID 32564) |
| 8 novembre, début de session | 1 306 Mo / 8 192 | 18 % |
| 8 novembre, après les bancs CPU | 1 297 Mo / 8 192 | 13 % |

La consigne de session exige **< 1 Go et < 10 %** avant toute mesure : elle n'est pas
remplie, donc aucun chronométrage n'a été pris et le run d'observation n'a pas été lancé.
Un run seq 128 / batch 16 tiendrait probablement dans les 6,9 Go libres, mais « probablement
tenir » n'est pas « mesurer sur une carte libre ».

Conséquence assumée : les seuils **ne sont pas appliqués**, faute de la mesure qui les
alimente. L'ETA reste **estimée** : 3 000 pas ≈ 3,9–6,1 h, d'après `logs_text.txt`
(3 novembre, carte libre, seq 128 / tbptt 16, 4 633 → 7 326 ms/pas). Le coût de `tbptt 4`
n'est pas mesuré.

## 2. Ce qui a été mesuré à la place

### 2.1 La comparaison de 500 pas est terminée

Contrairement à ce que disait le rapport du 7 novembre, le run **n'a pas été arrêté au pas
250** : `MYCELIA_EXIT=0`, checkpoint de fin au pas 500, `history.json` complet. Courbe et
lecture dans `docs/COMPARAISON-500.md`.

| pas | colonie perte | arêtes | Transformer perte |
|---:|---:|---:|---:|
| 100 | 9,0859 | 812 | 6,1718 |
| 250 | 8,6679 | 2 011 | 5,5172 (interpolé) |
| 500 | **7,4379** | **3 997** | **4,8725** |

Perplexités de validation : **colonie 1 977,8** contre **Transformer 159,97**, à 128 000
tokens chacun.

### 2.2 Le baseline était faux deux fois

1. **Identité** : cibles `(x, x)` contre un Transformer causal → perte 0,0000 et ppl 1,000
   au pas 50. Corrigé par le décalage `logits[:, :-1]` / `y[:, 1:]`.
2. **Échelle** : init par défaut (embeddings `N(0,1)`, tête liée) → perte initiale 21,90
   pour un uniforme à 9,21. Corrigé par une init GPT-2 → **9,2349 au pas 0**.

Les deux corrections vont dans le même sens : sans elles, la comparaison était soit
truquée en faveur de la colonie, soit flattée dans les deux sens.

### 2.3 Coût par pas de la colonie (contaminé)

3 832 ms/pas à 413 arêtes, 10 755 ms/pas à 3 997 : ces chiffres mesurent *colonie + jeu*,
pas la colonie. Ils ne sont pas comparables au banc du 6 novembre (1 464 ms/pas à
2 200 arêtes, carte libre).

## 3. État final du cerveau (pas 500)

| Grandeur | Mesuré |
|---|---|
| arêtes | 3 997 (4 006 naissances, 21 morts d'arêtes) |
| nœuds vivants | 514 sur un pool de 532 |
| nœuds nés d'un message | 2 |
| morts-nœuds / résurrections | 30 / 30 (cimetière vide) |
| E* prédit | ≈ 1 000 × naissances/pas ≈ 4 000 — **atteint à 3 997** |

Le plateau documenté est donc cohérent avec la prédiction à ce budget : la table a crû
linéairement (8 naissances/pas) jusqu'à ~4 000 arêtes, sans effondrement.

## 4. Verdict Palier 1 (échantillons)

**Non.** Les 42 tokens du pas 500 (`docs/echantillons-texte.md`) restent des mots français
enchaînés sans syntaxe : « Il était une fois rivières voitures merveilleuses colis doué
coucha perdre démêler seule autres rempli poires enneigée... ». Aucune phrase grammaticale.
C'est le résultat attendu à 500 pas sur 24 M de caractères, et ce n'est pas un résultat sur
l'architecture.

## 5. Recommandation

1. **Libérer la carte** : c'est le seul blocage, aucun hyperparamètre ne le contourne.
2. Lancer les deux arms de l'ablation (`docs/ABLATION-croissance.md`) : ils ne dépendent
   plus que de la carte, le code est écrit et validé.
3. Puis l'observation 300 pas en seq 128 / tbptt 4, et appliquer les seuils sur l'ETA
   **mesurée**.
4. Enfin, si l'ETA passe le seuil : 3 000 pas, baseline 10 000 pas en parallèle.
