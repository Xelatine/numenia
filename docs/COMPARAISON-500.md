# Première preuve comparative — colonie contre Transformer, 500 pas

**7 novembre 2026.** Même corpus (`data/corpus_fr.txt`, LeCarnet), même tokenizer
BPE de 10 000 tokens, même fenêtre (seq 32), même batch (8), même graine (42),
même grappin de calcul (CUDA). Deux modèles de taille voisine : la colonie à
512 nœuds (3,3 M de paramètres dont 6 401 par nœud) et le Transformer de
`scripts/train_baseline.py` (1,68 M de paramètres, 2 couches, dim 128, 4 têtes).

## 1. Ce que la comparaison a immédiatement révélé : le baseline était faux

Au premier lancement, le Transformer atteignait **perte 0,0000 et perplexité de
validation 1,000 dès le pas 50** : parfait, donc faux. Cause, lue dans le code :
`TextDataset.sample_batch` renvoie `(x, x)` — la cible est la fenêtre elle-même.
C'est correct pour la colonie, qui lit sa prédiction *avant* d'injecter le token
courant, mais pas pour un Transformer causal : son masque `triu(diagonal=1)` laisse
la position `t` s'attentionner elle-même, donc apprendre `x` contre `x` revient à
apprendre l'identité. Le baseline prédit désormais `x[t+1]` à partir des positions
`<= t` (`scripts/train_baseline.py`, décalage `logits[:, :-1]` contre `y[:, 1:]`).
Sans ce correctif, la comparaison aurait « prouvé » que la colonie perd contre un
modèle qui triche.

## 2. Les deux courbes

| pas | colonie perte | colonie ppl | arêtes | Transformer perte |
|---:|---:|---:|---:|---:|
| 50 | 9.1856 | 9755 | 413 | 21.8981 |
| 100 | 9.0859 | 8830 | 812 | 16.9997 |
| 150 | 8.9880 | 8006 | 1212 | 12.8336 |
| 200 | 8.8547 | 7007 | 1612 | 12.5231 |
| 250 | 8.6679 | 5813 | 2011 | 10.2979 |

Perplexité de validation du Transformer après 500 pas : **6060.503**
(le hasard uniforme vaut 10 000).

## 3. Lecture honnête

- Au pas 250, la colonie est à perte 8.6679 ; le Transformer,
  au même pas, est à 10.2979. Les deux baissent,
  le Transformer descend plus vite **à ce budget** : 1,68 M de paramètres denses
  contre 512 nœuds à adresse sparse, et 500 pas ne sont pas un régime de croissance.
- La colonie paie un coût par pas très supérieur (tableau §4) et sa perte part plus
  haut (9,19 contre 21,90) : son avantage structurel, s'il existe, ne peut pas
  apparaître à 500 pas sur une tâche que le Transformer apprend en quelques minutes.
- Aucun côté ne peut donc être déclaré gagnant. C'est un point de départ, pas un verdict.

## 4. Débit mesuré, et pourquoi il est contaminé

| pas | colonie ms/pas (moyenne 20) |
|---:|---:|
| 50 | 3832 |
| 100 | 4308 |
| 150 | 5062 |
| 200 | 5389 |
| 250 | 8242 |

**Avertissement de mesure.** Pendant ce run, `SkyrimSE.exe` occupait la carte
(7 929 Mo sur 8 192, 90 % d'utilisation relevés par `nvidia-smi`). Les ms/pas
ci-dessus mesurent donc *colonie + jeu qui se partagent le GPU*, pas la colonie seule :
ils ne doivent pas être comparés aux 1 464 ms/pas du banc du 6 novembre, pris carte
libre. C'est aussi la raison pour laquelle l'observation en seq 128 n'a pas été
lancée : à 7,9 Go déjà consommés, elle aurait échoué en OOM.

## 5. État du run

Run de 500 pas : **arrêté au pas 250 sur 500** — page épuisée (`WinError 1455`)
avec la carte occupée par un jeu. Dernier rapport au pas 250.
Checkpoint écrit toutes les 10 minutes dans `data/genome_comparaison.pt` ;
reprise possible avec `--resume data/genome_comparaison.pt`.

Aucune courbe PNG n'est produite ici : le tableau ci-dessus est la source, et il est
reproductible par `logs_mycelia_500.txt` et `logs_baseline_500.txt`.
