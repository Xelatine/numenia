> **Renommage —** Ce projet s'appelle désormais **Numenia** (auparavant référencé
> comme Mycelia dans les titres et la documentation). Session, chiffres, fichiers de
> données et cerveau **Nyra** inchangés. Ce document est historique : son contenu est
> conservé tel quel, et seul le nom du projet passe à Numenia.

# Session P — mesure d'abord : verdict config texte, retrieval rouge, mémoire verte

**Ordre respecté** : la mesure avant toute ingestion, le test golden avant de
construire dessus, l'intégration ensuite. Aucun chiffre de ce rapport n'est
hérité d'un brief : tout est relevé sur BIO-1.

## 0️⃣ Mesure de la config texte (batch 16, seq 128) — VERDICT : AUTORISÉ

100 pas, `Trainer(data="text")` réel (pas un forward simulé), `checkpoint_turns=True`,
TBPTT 16 (8 blocs), corpus Gutenberg (20 257 629 tokens train), CPU.

| | Valeur |
|---|---|
| **VmHWM** | **1 270 Mo** (seuil 2 500) |
| Marge | **1 230 Mo** |
| RSS de fin de pas | 686 Mo, plat |
| Arêtes | 10 → 813 (croissance ~8/pas) |
| Perte | 9,2098 → 9,1004 (cerveau neuf) |
| Rythme | **30,6 s/pas** (100 pas = 3 061 s) |
| Nœuds / vocab | 1 031 / 10 000 |

Verdict consigné : **ingestion autorisée en batch 16**, `text_config.json` +
`text_config.log` dans `data/mem_report/`.

### Trois réserves, écrites avant qu'on les oublie

1. **HWM inclut le chargement du corpus** (~600 Mo d'encodage) ; il est plat dès
   le pas 1 et ne bouge plus sur 100 pas. Le pic d'entraînement seul est ≤ 1 270.
2. **813 arêtes mesurées, 20 000 possibles.** En session O, `checkpoint_turns` a
   maintenu le pic plat de 3 100 à 6 246 arêtes ; la même invariance est attendue
   ici, mais elle n'est pas encore mesurée à 2 000 arêtes. Re-mesure à 2 000
   arêtes avant le run long, pas avant.
3. **Le cerveau mesuré est neuf** (vocab 10 000). Le génome du run 3 000 a un
   vocabulaire de 100 (tâche jouet) : il ne peut pas ingérer le tokenizer réel
   sans transplantation de vocabulaire, non mesurée. Le point de départ de la
   phase 2 reste donc à trancher : cerveau texte neuf (mesuré ici) ou greffe
   (à mesurer).

### Ce que le prototype fourni n'aurait pas mesuré

Le `checkpoint_sequential(model, segments=4)` du script initial est inopérant :
la protection mémoire réelle vit dans `Brain.checkpoint_turns`, et la perte
`out.pow(2).mean()` n'est pas la perte du modèle. Le script livré mesure le vrai
`step_once()` (entropie croisée, 8 blocs, `trim_heap`), sinon le chiffre serait
celui d'un autre programme.

## 2️⃣ Test de retrieval — ROUGE, et c'est le résultat

| Encodeur | HIT@1 | Ce que ça prouve |
|---|---|---|
| `hash` | hasard | rien (mécanisme seul) |
| `lexical` (sac-de-mots) | **2/5** | le pipeline tourne, le lexique ne suffit pas |
| `graph` (moyenne des embeddings du cerveau, non entraîné) | **3/5** | base de la phase 2, pas un encodeur sémantique |

**Conclusion CONSTATÉE : aucun encodeur sémantique n'existe aujourd'hui.** Le
mécanisme (cosinus, top-k, consolidation à l'usage) fonctionne — les tests
mémoire le prouvent — mais la qualité de recherche par paraphrase n'est pas
acquise. Conformément à la règle du script, **on ne construit rien dessus** :
l'ingestion devra soit entraîner les embeddings pendant la phase 2 (puis
relancer ce test), soit recevoir un encodeur sémantique externe (les embeddings
du GGUF Qwen sont disponibles et mesurables) — décision à prendre, pas à
supposer.

## 1️⃣ Mémoire à deux poids — VERT (7/7)

`mycelia/memory.py` + `tests/test_memory.py` sur le serveur :

```
....... [100%]  7 passed in 0.06s
```

Dont le test négatif exigé : **salience 0,95 / énergie 0, survit à 1 000 pas de
decay** (`prune()` renvoie 0, cimetière 0) — l'anti-piège LRU est testé, pas
seulement écrit. Le retrieval ne filtre jamais par salience ; seule l'énergie
plancher le fait.

Deux défauts trouvés par les tests, pas par la lecture :

- `np.append(…, 1.0)` promouvait les poids en **float64** : le round-trip NPZ
  perdait la valeur exacte. Corrigé en float32 explicite.
- Mon assertion de décroissance comparait les valeurs restantes au lieu des
  **taux** (deux exponentielles) : corrigée sur `log(salience)/log(énergie)`.

## 3️⃣ Protection SI, top-k, grounding — écrits, pas encore éprouvés

- `mycelia/protection.py` : importance = Σ|grad|², pénalité quadratique autour
  de l'ancre, seuil des synapses protégées. **Non branché** : il s'accumule
  pendant les derniers pas du graphe, puis se pénalise avant le LoRA.
- `mycelia/route_topk.py` : top-k strict + `active_fraction()`. **Mesure
  manquante** : le % de nœuds activés par pas. Si ≤ 5 %, le gain est nul et le
  module reste un garde-fou — c'est écrit dans le fichier.
- `mycelia/grounding.py` : sans `encode`, renvoie `ratio=None` et
  `grounded=None`. Il ne fabrique jamais un vert ; le seuil 0,55 reste à
  calibrer sur les golden tests.

## État

| Étape | État |
|---|---|
| 0. Mesure config texte | **VERT** — HWM 1 270 ≤ 2 500, batch 16 autorisé |
| 2. Retrieval golden | **ROUGE** — 2/5 lexical, 3/5 graphe non entraîné |
| 1. Mémoire deux poids | **VERT** — 7/7, anti-LRU inclus |
| 3. Protection / top-k / grounding | écrits, non éprouvés |

**Prochaine action, dans l'ordre** : décider de l'encodeur (entraîner ou
externe), relancer le golden jusqu'à 5/5, puis seulement ingérer. La mesure de
config, elle, ne bloque plus rien.
