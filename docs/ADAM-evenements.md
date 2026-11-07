# Pics de perte et événements de vie — hypothèse Adam testée, et écartée

**8 novembre 2026.** Le run v5 (toy, 2 000 pas) montre deux pics isolés de perte, au pas
1 300 et au pas 1 800. L'avis externe proposait une cause : les moments d'Adam créés à
ces événements. Vérification par lecture de code et par croisement avec le journal :
**l'hypothèse ne tient pas pour ces deux pics.** Les pics coïncident en revanche avec
des rafales d'élagage, ce qui n'est pas une preuve — l'élagage a lieu tous les 100 pas.

## 1. Les pics, mesurés

`logs_toy_v5_2000_batch16.txt` (relevés tous les 100 pas, perte d'entraînement) :

| pas | perte | arêtes | nœuds | morts-nœuds | cimetière |
|---:|---:|---:|---:|---:|---:|
| 1 200 | 0,0011 | 5 368 | 518 | 63 | 0 |
| **1 300** | **0,0365** | 5 553 | 517 | 64 | 1 |
| 1 400 | 0,0077 | 5 711 | 518 | 64 | 0 |
| 1 700 | 0,0004 | 5 617 | 520 | 64 | 0 |
| **1 800** | **0,0332** | 5 544 | 520 | 64 | 0 |
| 1 900 | 0,0015 | 5 476 | 520 | 64 | 0 |

Soit +0,035 et +0,033 sur une perte de l'ordre de 10⁻³ : deux sauts d'un facteur ~30,
puis un retour au niveau antérieur.

## 2. L'hypothèse Adam, et ce que le code en dit

Les moments sont conservés par **nom**, avec vérification de forme
(`mycelia/checkpoint.py` L109-133) :

```python
        compatible = all(
            not torch.is_tensor(value)
            or value.dim() == 0  # AdamW's step counter is a scalar, not a moment
            or tuple(value.shape) == tuple(param.shape)
            for value in saved.values()
        )
        if not compatible:
            continue
```

Donc, quand la population grandit ou qu'un nœud ressuscite, le tenseur de paramètres
change de forme et **les moments ne sont pas re-attachés** : les lignes nouvelles
démarrent avec des moments nuls. C'est un vrai mécanisme de pic — un paramètre dont le
second moment est nul fait un pas de la taille du taux d'apprentissage, sans amortissement.

Mais il faut que l'événement ait lieu. Or :

| pas du pic | naissance de nœud | mort de nœud | résurrection | changement de pool |
|---:|---|---|---|---|
| 1 300 | non (`nœuds_nés` 6 depuis 1 100) | non (64 depuis 1 100) | `cimetière` 1→0 au pas 1 400 | non |
| 1 800 | non (8 depuis 1 500) | non (64 depuis 1 100) | non | non (pool 839 depuis 1 500) |

Le seul changement de pool du run a lieu au pas **1 500** (`PALIER : pool 583 → 839`), et
il ne produit aucun pic (perte 0,0008). Les rares changements de forme sont donc
**décorrélés** des pics : l'hypothèse Adam est écartée pour 1 300 et 1 800.

## 3. Ce avec quoi les pics coïncident — et pourquoi ce n'est pas encore une preuve

Les deux pics suivent une rafale d'élagage :

```
  555 arêtes coupées, 0 nœuds enterrés      (avant le relevé 1 300)
  554 arêtes coupées, 0 nœuds enterrés      (avant le relevé 1 800)
```

Couper 555 synapses sur ~5 500 change la fonction de ~10 % : un pic de perte est
plausible. Mais l'élagage tourne **tous les 100 pas** et les relevés 1 000, 1 100, 1 200,
1 400, 1 500, 1 600, 1 700, 1 900 ne montrent aucun pic. La co-occurrence n'est donc pas
spécifique, et **je ne conclus pas**.

## 4. Verdict et ce qu'il faudrait pour trancher

**Mesuré** : les pics existent ; ils ne coïncident avec aucun événement de vie ; les
moments Adam ne peuvent pas être la cause de ces deux-là, puisqu'aucune forme de
paramètre ne change à ces pas.

**Non concluant** : la cause réelle. Le journal ne descend pas sous le pas 100, et
l'élagage est lui aussi quantifié au pas. Deux instruments manquent, et ils sont
identifiés :

1. écrire la perte **à chaque pas** (un fichier `losses.csv`), pas tous les 100 ;
2. écrire `edge_deaths`, `node_deaths`, `node_revivals` **à chaque pas**, pour que la
   corrélation soit testable et non contemplée.

Tant que ces deux journaux n'existent pas, toute explication des pics serait une
hypothèse déguisée en résultat. C'est écrit ici pour que personne ne la présente comme
mesurée.

## 5. Mise à jour du 8 novembre (soir) — le journal existe

L'instrument manquant est livré : `mycelia/train.py` écrit `data/per_step.csv`, une
ligne par pas, onze colonnes (`step, loss, edges, nodes, edge_births, node_births,
edge_deaths, node_deaths, node_revivals, cemetery, lr`), en append sur reprise et
tronqué sur run neuf. Vérifié sur un run jouet de 6 pas [mesuré] :

```
step,loss,edges,nodes,edge_births,node_births,edge_deaths,node_deaths,node_revivals,cemetery,lr
1,4.6156816482543945,18,512,6,0,0,0,0,0,0.001
2,4.592458724975586,28,514,16,2,0,0,0,0,0.001
```

Ce journal tranchera la question sur les runs **à venir**. Il ne rétro-instrumente pas
le run v5, qui n'a écrit que des relevés au pas 100 : la cause des pics 1 300 et 1 800
reste **non concluante**, et l'hypothèse « rafale d'élagage » reste une hypothèse.
