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

La règle de carte a été **révisée le 8 novembre** : un run est autorisé si (a) aucun
processus étranger ne calcule à plus de 5 % et (b) la VRAM libre couvre le besoin du run
plus 1,5 Go de marge. Besoin de l'arm grandi : **4 908 Mo réservés mesurés** au pas 500 du
run comparatif (`GPU 85/4908 Mo`), arrondi à 5 000 Mo ; marge 1 500 Mo ; seuil **6 500 Mo
libres**. Relevés de la session :

```
1 306 MiB utilisés, 18 % d'utilisation   (début de session, ancienne règle)
1 297 MiB utilisés, 13 % d'utilisation   (après les bancs CPU)
1 804 MiB utilisés, 6 224 MiB libres, 25 % d'utilisation, 29 processus étrangers  (lancement tenté)
```

Verdict mesuré du lancement : **(a) faux**, 25 % d'utilisation avec 29 processus étrangers
(navigateurs, Discord, Steam, overlays) ; **(b) faux**, 6 224 Mo libres pour 6 500 requis,
soit **276 Mo manquants**. Aucun processus n'a été tué. Les deux arms restent à lancer :

```bash
python scripts/ablate_growth.py --arm grown  --steps 500
python scripts/ablate_growth.py --arm frozen --steps 500
```

Durée attendue, **estimée** d'après les 1 464 ms/pas du banc du 6 novembre à 2 200 arêtes
(carte libre) : l'arm figé a 3 997 arêtes, donc ~2 fois plus de travail par pas, soit
~3 s/pas → **25 min par arm**. Cette estimation n'est pas une mesure et ne doit pas être
présentée comme telle.

Ce qu'il faut pour débloquer, chiffré : libérer **≥ 300 Mo de VRAM** et faire tomber
l'utilisation sous 5 % (fermer un navigateur à accélération matérielle suffit en général),
puis relancer la mesure de condition. **Si l'écart entre les deux arms est ≤ 2 %**, la
règle fixée d'avance impose de rejouer les deux à 2 000 pas avant de conclure — soit
~2 h par arm **[estimé]**, et une seconde session de carte libre.

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
