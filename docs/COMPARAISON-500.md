# Première preuve comparative — colonie contre Transformer, 500 pas

**8 novembre 2026 (révision).** Même corpus (`data/corpus_fr.txt`, LeCarnet), même
tokenizer BPE de 10 000 tokens, même fenêtre (seq 32), même batch (8), même graine (42).
Deux modèles de taille voisine : la colonie à 512 nœuds (512 × 6 401 paramètres de nœud)
et le Transformer de `scripts/train_baseline.py` (1 680 896 paramètres, 2 couches,
dim 128, 4 têtes). Budget identique : 500 pas × 32 tokens × 8 = **128 000 tokens** chacun.

## 1. Deux corrections avant de lire une courbe

**1.1 Le baseline apprenait l'identité.** Premier lancement : perte 0,0000 et perplexité
de validation 1,000 dès le pas 50. Cause : `TextDataset.sample_batch` renvoie `(x, x)`.
C'est juste pour la colonie, qui lit sa prédiction *avant* d'injecter (prouvé par
`test_causality`) et faux pour un Transformer causal, dont le masque `triu(diagonal=1)`
laisse `t` s'attentionner elle-même. Corrigé par le décalage `logits[:, :-1]` contre
`y[:, 1:]`.

**1.2 Le baseline démarrait à une échelle absurde.** Même corrigé, il partait à **perte
21,90** là où l'uniforme vaut `ln(10 000) = 9,21` : embeddings `N(0,1)` et tête liée aux
embeddings donnent des logits d'écart-type ~11. La perte initiale n'était pas un mauvais
départ, c'était une échelle cassée, et les cent premiers pas ne servaient qu'à revenir
vers l'uniforme. `TinyTransformer.reset_parameters` (init GPT-2, std 0,02, projections
résiduelles divisées par `√(2·layers)`) donne **9,2349 au pas 0**, à 0,025 de l'uniforme.

## 2. Les deux courbes, à tokens égaux

| pas | colonie perte | colonie ppl | arêtes | Transformer perte | Transformer ppl |
|---:|---:|---:|---:|---:|---:|
| 0 (init) | 9,1856* | 9 755 | 413 | **9,2349** | 10 231 |
| 100 | 9,0859 | 8 830 | 812 | 6,1718 | 479 |
| 200 | 8,8547 | 7 007 | 1 612 | 5,6345 | 280 |
| 300 | 8,4842 | 4 838 | 2 419 | 5,1833 | 178 |
| 400 | 8,0484 | 3 129 | 3 217 | 4,9621 | 143 |
| 500 | **7,4379** | **1 699** | **3 997** | **4,8725** | **131** |

\* premier relevé de la colonie, au pas 50 (son init n'est pas journalisée séparément).

**Perplexités de validation** (les seules comparables, même fenêtres tenues à l'écart) :

```
colonie       ppl val  1 977,8    [mesuré, pas 500]
Transformer   ppl val    159,974  [mesuré, pas 500, init corrigée]
hasard uniforme       10 000
```

Rapport : **×12,4 en faveur du Transformer**, à budget de tokens strictement égal.

## 3. Lecture honnête

- Le Transformer corrigé apprend beaucoup plus vite : il divise sa perte par 1,9 entre
  les pas 100 et 500, la colonie par 1,22.
- La colonie n'est pas immobile : sa perte passe de 9,19 à 7,44 et sa table croît de 413
  à 3 997 arêtes. Elle paie chaque pas dix fois plus cher (7 à 11 s/pas sur une carte
  partagée, contre 30 ms/pas CPU pour le Transformer), pour un résultat douze fois moins
  bon à ce budget.
- **Aucun avantage de la colonie ne peut apparaître ici** : 500 pas sur 128 000 tokens,
  c'est quelques minutes de calcul dense, et la colonie consacre une partie de ce budget
  à chercher sa topologie. Ce n'est pas une excuse, c'est la raison pour laquelle ce
  chiffre ne tranche pas l'architecture — il fixe le point de départ.
- Ce que la comparaison a réellement produit de solide : **deux bugs du baseline**, dont
  l'un (l'identité) aurait « prouvé » que la colonie gagne contre un tricheur, et l'autre
  (l'échelle) aurait fait paraître la colonie compétitive face à un modèle bridé.

## 4. Débit — mesuré, mais pas utilisable comme référence

| pas | colonie ms/pas (moyenne 20) |
|---:|---:|
| 50 | 3 832 |
| 100 | 4 308 |
| 150 | 5 062 |
| 200 | 5 389 |
| 250 | 8 242 |
| 300 | 11 266 |
| 350 | 8 669 |
| 400 | 7 044 |
| 450 | 7 426 |
| 500 | 10 755 |

Pendant tout le run, `SkyrimSE.exe` a occupé la carte (7 929 Mo sur 8 192, 90 %). Ces
chiffres décrivent *colonie + jeu*, jamais la colonie seule, et ne doivent pas être
comparés aux 1 464 ms/pas du banc du 6 novembre (carte libre, 2 200 arêtes). Le
Transformer, lui, a tourné sur CPU à 30 ms/pas : la comparaison de temps entre les deux
systèmes n'a donc aucun sens dans cet état, et n'est pas présentée comme un résultat.

## 5. État du run

Run de 500 pas **terminé** (`MYCELIA_EXIT=0`, checkpoint de fin au pas 500, 3 764 s de
run). `history.json` porte les dix relevés ; `data/genome_comparaison.pt` est le génome
final (3 997 arêtes, 514 nœuds vivants, 30 morts, 30 résurrections).

Note de méthode : le rapport du 7 novembre disait ce run « arrêté au pas 250 ». C'était
faux — l'arrêt avait été demandé, le processus avait déjà franchi le pas 500 et écrit son
checkpoint final. La correction est consignée ici et dans le rapport de session.
