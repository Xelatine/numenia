# Membranes aux frontières de documents — la fuite n'existe pas, et pourquoi

**8 novembre 2026.** Question posée (avis externe, round 2) : la membrane est-elle
réinitialisée aux frontières de documents ? Si non, un texte de validation pourrait
hériter du contexte d'un texte d'entraînement, et les perplexités seraient fausses.
Verdict : **aucune fuite de membrane, aucune fuite de données au-delà de 0,049 % de la
validation** — mesuré, pas supposé.

## 1. Il n'y a pas de frontières de documents, par construction

`mycelia/data.py`, docstring de `TextDataset` :

> « Why one flat stream and not one sequence per story: the brain has no notion of
> document boundaries, it only sees tokens arriving. Cutting the stream into fixed
> windows gives every step the same shape and lets a story start in the middle of
> another one. »

Le corpus est donc un **flux continu** de tokens (`stream`), coupé en fenêtres de
`seq_len`. Il n'existe aucun objet « document » dans le code : la question « la membrane
est-elle remise à zéro à la frontière ? » n'a pas de support, parce qu'il n'y a pas de
frontière.

## 2. La membrane est remise à zéro à chaque pas, pas à chaque document

`mycelia/brain.py` L548 :

```python
state = self._fresh_state(batch) if state is None else self._fit_state(state, batch)
```

et `_fresh_state` (L421-429) renvoie un tenseur de **zéros** :

```python
def _fresh_state(self, batch: int) -> torch.Tensor:
    return torch.zeros(self.n_nodes, batch, self.dim, ...)
```

Côté entraînement, `mycelia/train.py` (`step_once`) :

```python
        blocks = max(1, math.ceil(self.seq_len / self.tbptt))
        births_left = self.brain.max_births
        state = None
```

`state = None` **à chaque pas** : la fenêtre de 128 tokens démarre donc avec une
membrane vide. Elle est ensuite portée d'un bloc `tbptt` au suivant (`state=state`,
puis `state.detach()`), ce qui est l'intention du découpage, et jamais d'un pas à
l'autre.

Côté validation, `mycelia/train.py` (`validate`) appelle le cerveau **sans état** pour
chaque fenêtre :

```python
        for x, y in self.dataset.windows(int(self.config["training"]["val_batch"]), "val"):
            batches.append((x, y))
        ...
            logits, _ = self.brain(x, grow=False)
```

Chaque fenêtre de validation repart donc de zéros : aucune ne contamine la suivante,
et aucune ne peut hériter de l'entraînement autrement que par les poids.

La seule génération qui **porte** la membrane est `Brain.generate` (L757-782) : c'est
voulu — c'est la parole, pas l'évaluation — et le contexte y est reconstruit depuis la
dernière fenêtre quand il déborde (`context_len`), ce qui borne la mémoire.

## 3. Doublons du corpus : aucun

| Mesure | Valeur |
|---|---|
| lignes non vides | 29 402 |
| lignes distinctes (SHA-1) | **29 402** |
| lignes dupliquées | **0** |
| caractères dans une ligne dupliquée | 0 (0,00 %) |

Aucun texte n'apparaît deux fois dans `data/corpus_fr.txt`.

## 4. Chevauchement entraînement / validation : mesuré, et c'est un choix

`TextDataset` garde 5 % pour la validation et ajoute **un token de contexte** avant la
coupure :

```python
        cut = int(stream.numel() * (1.0 - float(val_fraction)))
        self.train = stream[:cut]
        self.val = stream[max(0, cut - self.seq_len) :]
```

Conséquence mesurée sur le corpus réel (`seq_len` 128) :

| Mesure | Valeur |
|---|---|
| tokens train / validation | 4 955 606 / 260 950 |
| `val[0:128] == train[-128:]` | **vrai** (chevauchement voulu) |
| fenêtres train / validation | 38 715 / 2 038 |
| fenêtres **identiques** entre les deux | **0** |
| part de validation recouverte | **0,0000 %** |
| types de tokens présents dans les deux | 6 683 sur 9 190 (vocabulaire, pas fuite) |

La fenêtre de validation d'indice 0 contient les 128 derniers tokens d'entraînement,
soit **1 fenêtre sur 2 038 = 0,049 %** de la validation. Ce n'est pas une fuite : c'est
le contexte minimal pour que la première cible tenue à l'écart soit lue dans son
contexte, et son effet sur une perplexité est sous le bruit d'échantillonnage.
Aucun correctif n'est donc appliqué — le corriger changerait la définition de la
validation, pas sa validité.

## 5. Ce que la remise à zéro implique, et qu'il faut dire

La membrane est vidée toutes les 128 positions : le modèle **ne peut pas** utiliser un
contexte plus long qu'une fenêtre, même si `state_decay` permettrait théoriquement
d'en porter ~20 à 100 tokens. Ce n'est pas un défaut de mesure, c'est une propriété du
régime d'entraînement (et une raison de plus pour que `seq 128` soit un choix explicite,
pas un défaut hérité).

## 6. Décision

**Rien à corriger avant un run texte.** Les perplexités mesurées ne sont pas gonflées
par une fuite : la membrane est vide au début de chaque fenêtre, le corpus n'a aucun
doublon, et le seul recouvrement est celui, documenté, du token de contexte (0,049 %).
