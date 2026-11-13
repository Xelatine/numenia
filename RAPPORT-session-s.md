# Session S — sparsité : le diagnostic est fait, la preuve par contraste ne l'est pas

**Honnêteté en tête** : l'expérience centrale de la session — le jumeau sparse
de 3 000 pas (~11 h CPU) — **n'a pas été exécutée**. Ce rapport contient le
diagnostic (décisif), l'implémentation de l'activation sparse, et un pilote de
50 pas qui ne permet **pas** de trancher la doctrine. Aucun chiffre ne prétend
le contraire.

## Action 1 — diagnostic de l'activation : CONSTATÉ

**Fonctions réellement appliquées** :

- gate : `sigmoid(gate_w·x + b)` puis seuil dur `GATE_THRESHOLD = 0.1` ;
- body : **GELU** (lisse, jamais exactement zéro) ;
- routage : `sigmoid(q·k/√dim)` ;
- lecture : LayerNorm + Linear.

**Zéros exacts par hop sur le génome du run 3 000** : `0,0` à tous les hops, et
**gates fermés : 0,0** — le seuil dur n'a jamais tiré. Le cerveau est dense non
parce qu'il a « appris dense », mais parce que **rien dans son activation ne
peut produire un zéro**. La chaîne causale de la session R est confirmée : pas
de zéros natifs → propagation cascade → top-k destructeur.

## Action 1b — le forward est déjà sparse en arêtes (pas de gain CSR à récolter)

Temps d'un forward de 8 tokens, N = 1 129 constant :

| Arêtes | Temps |
|---|---|
| 3 123 | 0,186 s |
| 6 246 | 0,975 s |
| 12 492 | 2,120 s |
| 24 984 | 4,083 s |

Le coût suit les **arêtes** (×2 d'arêtes → ×2 de temps à partir de 6 246), pas
N². Le forward est déjà un calcul de liste d'arêtes, pas une matrice dense : la
conversion CSR n'apporterait rien. Le 3 123 → 6 246 (×5,2) est un artefact de
démarrage/chauffe, pas une loi.

## Action 2 — sparsité temporelle : ABSENTE

`delta[n] = |état(t) − état(t−1)| / (|état(t−1)| + 1e-8)` sur 16 positions :

| ε | Fraction changeante | Delta médian |
|---|---|---|
| 1e-3 | **88,9 %** | 0,098 |
| 1e-2 | **88,9 %** | 0,098 |

**Verdict : régime dense total.** Une couche Delta ne skipperait que ~11 % des
nœuds ; ce n'est pas le levier annoncé. La piste est écartée **par la mesure**,
pas par intuition — et Action 5 n'est donc pas implémentée.

## Action 3 — l'activation sparse : implémentée, pilote non concluant

`NodeBank(activation="relu"|"gelu", output_sparse=bool)`, exposé dans `Brain` et
donc dans le génome. `output_sparse` applique un ReLU sur la sortie du nœud —
sans lui, le résidu `x + norm(...)` interdit tout zéro exact, quel que soit
l'activation interne.

Pilote **50 pas**, même tâche jouet, même seed, deux cerveaux neufs :

| Bras | Zéros exacts | Perte (campagne) | Top-k 50 % → 5 % |
|---|---|---|---|
| dense (GELU) | **0,0 %** | 4,550 | ×1,000 → ×0,996 |
| sparse (ReLU + sortie ReLU) | **48,0 %** | 4,593 | ×1,000 → ×0,996 |

Deux lectures :

- **la sparsité d'activation est réelle** : 48 % de zéros exacts dès la
  naissance, contre 0 % pour le jumeau dense ;
- **le contraste top-k est nul** — mais parce qu'après 50 pas les deux bras sont
  à la perte du hasard (ln 100 = 4,605 ; mesuré 4,550 et 4,593). Un modèle qui
  ne sait rien ne perd rien quand on le coupe. **Ce pilote ne prouve ni ne
  réfute la doctrine sparse.**

Le jumeau de 3 000 pas (condition A ReLU, condition B k-WTA) n'a pas été lancé :
~11 h CPU, hors de cette session. La question « un cerveau né sparse résiste-t-il
au top-k là où le dense explose ? » reste **ouverte**.

## Actions 4 à 6 — non faites, avec leur raison

- **Action 4 (gate entraîné)** : requiert un cerveau sparse entraîné pour
  comparer ; sans le jumeau, ce serait un second inconnu empilé sur le premier.
- **Action 5 (couche Delta)** : écartée par Action 2 (88,9 % de changement).
- **Action 6 (memory layer)** : non prototypée ; le golden reste à 2/5 lexical
  et 3/5 graphe, et une structure de recherche ne corrige pas un encodeur
  absent.

## État

| Piste | Verdict |
|---|---|
| Diagnostic activation | **zéro exact = 0 %, gate jamais fermé** |
| Forward dense/sparse | **déjà sparse en arêtes** (pas de N²) |
| Sparsité temporelle | **88,9 % changeant** → Delta écartée |
| Activation sparse | **implémentée**, 48 % de zéros au pilote |
| Jumeau 3 000 pas | **non exécuté** → doctrine non prouvée |
| Encodeur | **toujours rouge**, ingestion fermée |

**Ce que la session S établit** : la cause racine est l'absence de zéros
natifs, mesurée ; le levier temporel n'existe pas ; l'activation sparse est
disponible et produit des zéros. **Ce qu'elle n'établit pas** : que cela suffise
à rendre le top-k inoffensif. Il faudra les 11 h du jumeau — et le dire avant de
l'avoir fait est exactement ce que ce rapport refuse de faire.
