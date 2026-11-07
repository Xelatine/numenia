# L'ablation centrale — la croissance sert-elle ? Protocole, et où elle en est

**8 novembre 2026.** Question posée : une colonie qui aurait démarré avec le graphe final
(3 997 arêtes, ~514 nœuds), câblé **au hasard** et **figé**, aurait-elle fait aussi bien ?
Si oui, les naissances ne sont qu'une décoration coûteuse, et c'est le résultat le plus
important du projet — dans ce sens-là. Ce document fixe le protocole et l'état exact de
l'exécution.

## 1. Ce qui est codé, et comment

- `mycelia/ablation.py::freeze_random_graph` : remplace le câblage par un graphe aléatoire
  sparse de `edges` synapses, conserve le chemin de départ `0 -> 2..7 -> 1` (sinon la
  porte et le lecteur sont déconnectés et l'arm mesure un cerveau muet), garantit ≥ 1
  arête entrante par nœud, puis coupe `max_births`, `max_node_births` et
  `allow_node_growth`.
- Drapeau de configuration : `growth.mode: grown | frozen_random` et
  `growth.frozen_edges` (`configs/config.yaml`, défauts dans `mycelia/config.py`). Le
  gel n'est appliqué qu'à un cerveau **neuf** : un run repris garde son propre câblage.
- `scripts/ablate_growth.py --arm grown|frozen` : deux arms, même graine, même corpus,
  même `seq 32`, même `batch 8`, même budget de pas, même générateur de données. Chaque
  arm écrit sa courbe (`logs_ablation_*.txt`), son JSON et sa perplexité de validation.

## 2. Validation du code (faite)

Un cerveau jouet de 128 nœuds avec `growth.frozen_edges = 300` :

```
arêtes 300 | max_births 0 | allow_node_growth False | births 0 | enable_growth False
3 pas ok | perte 4.6054 | arêtes finales 300 | nœuds 128
```

Le graphe ne bouge pas d'une arête en trois pas, et la croissance est bien désactivée :
le drapeau fait ce qu'il annonce.

## 3. Ce qui n'est pas encore mesuré, et pourquoi

La condition posée en tête de session est explicite : `nvidia-smi` doit montrer
**< 1 Go utilisé et < 10 % d'utilisation**, et aucune mesure ne doit être prise sur une
carte contaminée. Relevés pendant la session :

```
1306 MiB, 18 %   (début de session)
1297 MiB, 13 %   (après les bancs CPU)
```

La carte n'est donc **pas libre** au sens de la consigne (bureautique et navigateurs,
aucun processus de calcul étranger). Les deux arms restent à lancer :

```bash
python scripts/ablate_growth.py --arm grown  --steps 500
python scripts/ablate_growth.py --arm frozen --steps 500
```

Durée attendue, **estimée** d'après les 1 464 ms/pas du banc du 6 novembre à 2 200 arêtes
(carte libre) : l'arm figé a 3 997 arêtes, donc ~2 fois plus de travail par pas, soit
~3 s/pas → **25 min par arm**. Cette estimation n'est pas une mesure et ne doit pas être
présentée comme telle.

## 4. La règle de verdict, fixée d'avance

- si l'arm **figé** atteint une perplexité de validation ≤ celle de l'arm grandi (± 2 %),
  alors **la croissance n'a pas encore payé à ce budget** : c'est un résultat, et il sera
  écrit tel quel ;
- si l'arm **grandi** est meilleur de plus de 2 %, la croissance a trouvé une topologie
  utile, et l'écart est la mesure de son apport ;
- dans les deux cas, le budget de 500 pas est **court** : la croissance a besoin de la
  boucle naissance/élagage pour explorer, et l'arm figé n'a, par construction, aucun
  mécanisme d'exploration. Un verdict négatif à 500 pas ne condamnerait donc pas la
  croissance — il dirait qu'elle ne s'est pas encore distinguée.

Les deux courbes seront publiées côte à côte, sans lissage et sans moyenne mobile.
