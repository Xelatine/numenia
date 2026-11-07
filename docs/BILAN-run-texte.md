# Bilan du run texte — 7 novembre 2026

**Aucun run texte long n'a été lancé.** Ce document dit pourquoi, avec les mesures,
et ce qui a été mesuré à la place.

## 1. Où en est la décision des seuils

Le plan prévoyait : observation de 300 pas en régime texte réel (seq 128, batch 16,
tbptt 4) → ETA bornée → seuils automatiques. **L'observation n'a pas été lancée**, pour
une raison mesurée et non négociable : pendant la session, `nvidia-smi` relève
**7 929 Mo occupés sur 8 192 et 90 % d'utilisation**, par `SkyrimSE.exe` (PID 32564),
un processus qui n'appartient pas à ce travail. À ce niveau d'occupation, un run
seq 128 / batch 16 échoue en OOM dès les premières activations ; lancer quand même
aurait produit un crash, pas une mesure.

Conséquence assumée : les seuils **ne sont pas appliqués**, faute de la mesure qui les
alimente. Décider sur une ETA non mesurée est exactement l'erreur que le §8 du
diagnostic du 5 novembre a déjà coûtée une fois.

## 2. Ce qui a été mesuré à la place

### 2.1 Preuve comparative 500 pas (seq 32, batch 8)

Voir `docs/COMPARAISON-500.md`. Résumé : le Transformer descend plus vite à ce budget
(perte 7,94 contre 8,67 au pas 250 pour la colonie), et la comparaison a surtout servi
à **trouver un bug du baseline** : il apprenait l'identité (perte 0,0000, ppl 1,000)
parce que `TextDataset` renvoie `(x, x)`, ce qui est juste pour la colonie et faux pour
un Transformer causal. Corrigé par un décalage `logits[:, :-1]` / `y[:, 1:]`.

### 2.2 Coût par pas de la colonie en régime texte, mesuré ce jour

| pas | arêtes | ms/pas (moyenne 20) |
|---:|---:|---:|
| 50 | 413 | 3 832 |
| 100 | 812 | 4 308 |
| 150 | 1 212 | 5 062 |
| 200 | 1 612 | 5 389 |
| 250 | 2 011 | 8 242 |

Ces chiffres sont **contaminés** par le jeu qui partage la carte : ils ne décrivent pas
la colonie seule. Ils montrent en revanche la forme de la courbe — le coût suit le
nombre d'arêtes, pas le compteur de pas, ce que le profil du README annonçait déjà.

### 2.3 Le seul point d'ancrage non contaminé pour seq 128

`logs_text.txt` (run réel du 3 novembre, carte libre, seq 128, batch 16, tbptt 16) :
**4 633 → 7 326 ms/pas** entre les pas 25 et 150. Extrapolation, **étiquetée estimée** :
3 000 pas à ce débit ≈ **3,9 à 6,1 h** ; 10 000 pas ≈ 13 à 20 h. Le coût de `tbptt 4`
n'a **pas** été mesuré : il réduit la mémoire du pas, pas nécessairement son temps, et
je ne prétends pas savoir lequel des deux effets domine.

## 3. Ce que le run 500 pas a produit et laisse en l'état

- `logs_mycelia_500.txt` : courbe de perte, arêtes, nœuds, échantillons aux pas 50 à 250.
- `data/genome_comparaison.pt` : checkpoint toutes les 10 minutes, reprise par
  `python mycelia/train.py --data text --resume data/genome_comparaison.pt`.
- Le run de 500 pas **a été arrêté au pas 250 sur 500** au moment d'écrire ces lignes (dernier rapport
  au pas 250, sur 500). La cause est consignée : en plus du jeu qui occupait la carte, la machine a épuisé
  son fichier de pagination (`OSError WinError 1455` au chargement de
  `cublas64_12.dll`), ce qui a empêché jusqu'à la relance des tests. Il est reprenable
  tel quel (`data/genome_comparaison.pt`, écrit au pas 250) : ce n'est ni un crash du
  modèle ni une décision d'architecture.

## 4. Verdict Palier 1 (échantillons)

**Non**, sans ambiguïté. Les 42 tokens produits (`docs/echantillons-texte.md`) sont des
mots français enchaînés sans syntaxe : « Il était une fois rencontraraient sœursifférent
détestait brisé propreté brique… ». Aucune phrase grammaticale et compréhensible. C'est
le résultat attendu à 250 pas sur un corpus de 24 M de caractères, et ce n'est pas un
résultat sur l'architecture.

## 5. État final du cerveau

| Grandeur | Valeur mesurée au pas 250 |
|---|---|
| arêtes | 2 011 (naissances 1 999, aucune mort) |
| nœuds vivants | 514 (512 initiaux, 2 nés d'un message) |
| morts-arêtes / morts-nœuds | 0 / 0 (l'élagage commence au pas 100, la mortalité n'a pas encore eu lieu) |
| E* prédit | ≈ 1 000 × naissances/pas ; à 5 naissances/pas ≈ 5 000 arêtes — non atteint ici |

Le plateau documenté (`docs/PLATEAU-croissance.md`) n'est pas encore observable à
250 pas : la table n'a pas fini de croître.

## 6. Recommandation

1. **Ne rien lancer sur cette carte tant qu'elle est occupée.** Le premier chantier est
   une machine libre, pas un hyperparamètre.
2. Quand le GPU est libre : lancer l'observation telle que spécifiée (300 pas, seq 128,
   batch 16, tbptt 4, `configs/config_observation.yaml`), puis appliquer les seuils sur
   l'ETA mesurée, pas sur celle du §2.3.
3. Terminer d'abord le run comparatif de 500 pas (`--resume`) pour clore la première
   preuve comparative sur une courbe complète.
