# Session Q — l'encodeur Qwen mesuré : 3/5, et la porte reste fermée

**Hypothèse testée** : « encodées par Qwen, HIT@1 attendu 5/5 ».
**Résultat CONSTATÉ** : **3/5**. L'hypothèse est réfutée par la mesure, pas par
un doute. Conformément à la règle de la session P (« si 5/5 »), l'étape 2 —
brancher la mémoire sur l'encodeur — **n'a pas été exécutée**.

## 1️⃣ Golden test Qwen (dim 896, GGUF déjà sur BIO-1, md5 vérifié)

| Pooling | HIT@1 | Cosinus bon souvenir | Cosinus meilleur faux |
|---|---|---|---|
| mean (1) | **3/5** | 0,8644 – 0,9293 | jusqu'à 0,9228 |
| CLS (2) | **2/5** | 0,6546 – 0,7423 | jusqu'à 0,7808 |
| **last token (3)** | **3/5** | 0,8506 – 0,9293 | jusqu'à 0,9051 |

Deux lectures, toutes deux défavorables :

- **Le classement est faux, pas le seuil.** HIT@1 ne dépend d'aucun seuil : ce
  n'est donc ni le 0,55 de `grounding.py` ni le découpage en claims qui sont en
  cause. Les phrases correctes et incorrectes se disputent le top-1.
- **L'espace est tassé.** Les cosinus vont de 0,85 à 0,93 pour des textes sans
  rapport ; aucune valeur ne sépare les bons des mauvais (le max des faux,
  0,9051, dépasse le min des bons, 0,8506). C'est l'anisotropie classique des
  états cachés d'un petit LM causal, pas un encodeur de retrieval.

**Conclusion** : Qwen2.5-0.5B-Instruct n'est pas un encodeur sémantique. Le
« 20 tok/s » mesuré en session O dit sa vitesse de génération, pas sa qualité
d'embedding. Utiliser ses états cachés comme organe des sens du retrieval
serait un sens emprunté qui ne distingue rien.

## 2️⃣ Mémoire : inchangée, et c'est volontaire

La mémoire reste mono-vecteur (32-d). Le schéma `emb_index` (896 fp16) +
`emb_graph` (32) n'est **pas** branché : il attend un encodeur qui passe le
golden. Les 7/7 tests de la session P restent valides ; aucun test n'a été
affaibli pour faire entrer Qwen.

## 3️⃣ Fraction active mesurée — le top-k n'est pas décoratif

Sur le génome du run 3 000 (1 129 nœuds, 6 246 arêtes), un forward de 64 tokens :

```
nœuds vivants 1 005 | ayant émis 1 004 | fraction 0,999
top-k 5 % : 56 nœuds sur 1 129
```

**99,9 % du graphe émet** (cumulé sur les 64 positions). Le top-k à 5 % n'est
donc pas un garde-fou sans effet : il y a un facteur ~20 disponible. Réserve
honnête : c'est une fraction **cumulée sur la séquence**, pas par position ; la
mesure par hop doit être faite avant de promettre le facteur 20 sur la RAM
active.

## 4️⃣ Re-mesure à 2 000 arêtes — non lancée, et pourquoi

Elle coûte ~250 pas × 30,6 s ≈ **2 h 10** et sert à autoriser l'ingestion. Or
l'ingestion est bloquée en amont par l'encodeur : verdir la config d'un run qui
ne peut pas démarrer serait une mesure sans emploi. Elle sera relancée dès
qu'un encodeur passe le golden — pas avant. La mesure de session P (1 270 Mo à
813 arêtes) reste la référence, avec sa réserve n°2 intacte.

## Ce qui débloque l'encodeur, chiffré

| Piste | Coût | Qualité attendue |
|---|---|---|
| GGUF d'embedding dédié (bge-m3, e5 multilingue) via llama.cpp | 0,5-2,3 Go à télécharger, md5 à consigner ; même runtime déjà compilé | élevée : ces modèles sont entraînés pour la paraphrase |
| Fine-tuner Qwen 0.5B pour le retrieval | des paires, du temps, un GPU absent | inconnue, disproportionné |
| Entraîner les embeddings du graphe pendant l'ingestion | des semaines, et l'ingestion est bloquée par le retrieval | circulaire aujourd'hui |

L'option C (hybride) reste la bonne architecture ; c'est son segment B qui
change de modèle : un encodeur d'embedding dédié, pas le chat 0,5B.

## État

| Étape | État |
|---|---|
| 1. Golden Qwen | **ROUGE — 3/5**, cosinus non séparables |
| 2. Mémoire + encodeur | **non exécutée** (condition 5/5 non remplie) |
| 3. Fraction active | **MESURÉE — 0,999**, levier top-k réel |
| 4. Config à 2 000 arêtes | **différée**, ingestion bloquée en amont |

**Prochaine action** : choisir et mesurer un encodeur d'embedding dédié sur le
même golden. La porte de l'ingestion n'a pas bougé ; elle a maintenant un
diagnostic précis : ce n'est pas la mémoire, ce n'est pas le seuil, c'est le
modèle qui encode.
