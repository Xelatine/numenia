# Parallélisation du front d'onde — réponse par lecture de code

**6 novembre 2026.** Question posée : les 128 positions de séquence sont-elles
indépendantes pendant un tour, au point de traiter `batch × seq` en parallèle sur
les 8 sauts au lieu de 1 024 itérations séquentielles ?

**Verdict : NON.** Des trois prémisses de l'avis externe, la première est vraie,
la deuxième est **fausse**, et la troisième est vraie mais insuffisante : la
membrane n'est pas seulement une mémoire passive, elle est la **clé de routage du
destinataire**. Paralléliser les positions demanderait de changer la sémantique,
ce que la consigne interdit. Aucun code n'a été touché.

## 1. Réponses aux trois questions

### (a) L'émission lit-elle la membrane, ou seulement ce qui vient d'arriver ?

**Seulement ce qui vient d'arriver. C'est vrai, et c'est une propriété voulue.**

`mycelia/brain.py` L591-592 :

```python
                x = wave
                out, _gate = self.bank.forward_rows(x, hot)
```

`wave` ne contient jamais la membrane : il vaut l'injection à l'entrée
(L583 `wave = injection.unsqueeze(0)`) puis, à chaque tour, exactement ce qui a
été délivré au tour précédent (L663 `wave = inbox[alive_inbox]`). Et
`NodeBank.forward_rows` (`mycelia/node.py`) ne lit que son argument `x` :

```python
gate = torch.sigmoid(_blinear(x, self.gate_w[ids], self.gate_b[ids]))
hidden = F.gelu(_blinear(x, self.b1_w[ids], self.b1_b[ids]))
transformed = _blinear(hidden, self.b2_w[ids], self.b2_b[ids])
out = (x + _layernorm(transformed, self.ln_w[ids], self.ln_b[ids])) * gate
```

C'est la correction historique du README (« Early attempts made nodes emit their
whole membrane, which turned any cycle into an amplifier ») : un nœud émet son
entrée, donc une boucle perd 10 % par tour au lieu de s'amplifier.

### (b) La porte / le routage dépend-il de la membrane ?

**Oui — de la membrane du destinataire. C'est le point qui bloque tout.**

`mycelia/brain.py` L639-641, dans la boucle de scatter :

```python
                    key_c = self.bank.keys(state[recv_c], recv_c)
                    query_c = self.bank.queries(out[rows_c], src_e[start:stop])
                    score_c = torch.sigmoid(
                        (query_c * key_c).sum(dim=-1, keepdim=True) * self.inv_sqrt_dim
                    )
```

La requête est bien tirée de l'émission (`out`, donc de ce qui vient d'arriver),
mais **la clé est tirée de `state[recv_c]`**, c'est-à-dire de la membrane du nœud
qui reçoit. Le score multiplie ensuite le message (L647-649) :

```python
                    message_c = (
                        out[rows_c] * score_c * mask_c.view(-1, 1, 1) * self.dissipation
                    )
```

Donc « un nœud ne dépend que de ce qu'il vient de recevoir » est faux au niveau
du graphe : *où* un message atterrit dépend de l'état accumulé de la cible. La
membrane n'est pas un lecteur passif en bout de chaîne, elle pèse sur le routage à
chaque tour. C'est F2 lu au sens fort : l'adresse d'un nœud est sa clé, et sa clé
est calculée sur sa membrane.

### (c) Y a-t-il un couplage entre positions autrement que par la boucle ?

Pas de masque causal, pas d'attention croisée : la causalité est **structurelle**,
pas tensorielle. Mais les positions ne sont pas indépendantes, parce que la
membrane est portée de l'une à l'autre :

`mycelia/brain.py` L573-575 :

```python
        for position in range(seq_len):
            # Read before injecting: the prediction of token t must not see it.
            predictions.append(self.predict(state))
```

et L660-663, à la fin de chaque tour :

```python
                state = state * self.state_decay
                if hot.numel():
                    state = state.index_add(0, hot, inbox[alive_inbox])
                wave = inbox[alive_inbox]
```

`state` entre dans la position `t + 1` tel qu'il est sorti de la position `t`, et
la porte de la position `t + 1` lit ce `state` (L639). La position `t + 1` ne peut
donc pas démarrer avant que la position `t` ait fini ses 8 tours.

## 2. Schéma des dépendances

```
position t ─────────────────────────────────────────────────────┐
                 tour 0        tour 1              tour 7        │
  state_{t-1} ──► key(639) ──► key(639) ──► ... ──► key(639) ────┤ état porté
       ▲             │            │                   │           │ vers t+1
       │             ▼            ▼                   ▼           │
       │          inbox_0 ──► inbox_1 ──► ... ──► inbox_7         │
       │             │            │                   │           │
       └─────────────┴────────────┴───────────────────┘           │
              state ← state·0.995 + inbox (660-662)               │
                                                                 ▼
position t+1 ── tour 0 : key(639) lit le state laissé par t ──────┘
```

Deux récurrences imbriquées, et **aucune des deux n'est parallélisable sur
l'axe des positions** :

1. verticale (tours) : `state_h` alimente la porte du tour `h + 1` ;
2. horizontale (positions) : `state_t` alimente la porte du tour 0 de `t + 1`.

Le batch, lui, est déjà vectorisé partout (`[n_nodes, batch, dim]`), et les nœuds
chauds aussi (F10) : il ne reste séquentiel que les 1 024 itérations
`128 positions × 8 tours`, exactement le chiffre que le profil du README
désignait déjà comme le coût dominant.

## 3. Pourquoi le « scan » proposé ne sauve pas la situation

L'avis externe propose de reconstruire la membrane par un scan (somme
exponentielle décroissante). Ce serait valable si `state` n'était qu'un
accumulateur : la somme décroissante est associative et se calcule en log n.

Or la ligne linéaire (L660-662) n'est qu'**une** des deux utilisations de `state` :
l'autre (L639) le fait passer dans une sigmoïde qui multiplie le message avant
l'accumulation. La récurrence réelle est

```
state_h = state_{h-1} · 0,995 + Σ_{bords vers n} out_e · σ(q_e · K(state_{h-1}[n])) · m_e · 0,9
```

`K` et `σ` sont non linéaires et le produit couple membrane et message : le scan
n'est pas applicable. Il faudrait soit geler la clé sur `out` (mais `state_h`
entre alors encore dans le tour suivant), soit geler la membrane pendant tout un
pas — ce qui supprime le routage dépendant de l'état, donc une partie de F2, et
change la sémantique. Refusé.

## 4. Ce que la parallélisation aurait coûté, et ce qui reste possible

- Le gain annoncé (~×100) supposait l'indépendance des positions. Elle n'existe
  pas : le plafond réel d'un tel chantier serait nul sans changement de
  sémantique.
- **Ce qui reste vrai** : la parallélisation *inter-batch* est déjà en place, et
  l'unique axe encore séquentiel est `positions × tours`. Le levier mesuré pour
  ce coût reste celui du 5 novembre : `tbptt 4` à 2 200 arêtes (2 186 ms/pas,
  1 240 Mo), et non une réécriture du forward.

## 5. Décision

Étape 1 close : **non parallélisable sans changement de sémantique**, donc
conformément à la consigne, pas d'implémentation, pas de drapeau `wave_parallel`,
pas de `test_wave_parallel_equivalence`. Le reste de la session traite l'étape 2
(forme) puis les étapes 3 et 4 (preuve comparative), sans toucher au forward.
