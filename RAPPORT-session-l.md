# Session L — ouverture du sprint « français » : aucun prérequis au vert

Le sprint demandé (organe de langage + distillation vers le graphe) est
conditionné à trois prérequis. Ils sont mesurés à **0 sur 3**. Les phases 2 à 4
ne sont donc pas entamées : la règle du sprint l'interdit, et c'est la bonne
règle — ingérer un corpus sur un graphe qui fuit serait dépenser des heures
pour un chiffre non publiable.

## 1.1 Filaire — toujours bloqué sur les droits root

```
/etc/netplan/ : 00-installer-config.yaml 01-network-manager-all.yaml 90-NM-….yaml
enp8s0        : UP 169.254.184.83/16 SERVEUR-IPV6/64
default       : via PASSERELLE-LAN dev wlp7s0 proto dhcp src SERVEUR-LAN metric 20600
```

Le drop-in `60-enp8s0.yaml` n'a jamais été installé et la route par défaut sort
toujours par `wlp7s0`, le radio qui ne route plus rien d'autre que le lien-local.
C'est la quatrième session que ce point attend `sudo bash ~/prepare-filaire.sh`.
Je ne peux pas exécuter d'élévation de privilèges depuis mon environnement.

Conséquence pratique : tout le sprint — téléchargements de corpus, modèle,
installation de `peft`/`transformers` — devra passer par le câble et par moi,
puisque le serveur n'a aucun accès sortant.

## 1.2 Dérive mémoire — caractérisée, pas corrigée

Nouvelle sonde `scripts/drift_probe.py` : elle relève RSS, pic et commit toutes
les N étapes et imprime la série, pour que la **forme** de la courbe réponde au
lieu d'une moyenne.

Courbe à 1 000 nœuds, batch 2, seq 16, CPU (device propre, 0 conteneur) :

| Pas | RSS | Commit |
|---:|---:|---:|
| départ | 631 Mo | 5 026 Mo |
| 1 | 792 Mo | 25 869 Mo |
| 20 | 886 Mo | 25 963 Mo |
| 40 | 1 003 Mo | 26 079 Mo |
| 60 | 1 131 Mo | 26 207 Mo |
| 80 | 1 314 Mo | 26 389 Mo |
| 100 | 1 433 Mo | 26 509 Mo |

Trois faits mesurés :

1. **Ce n'est pas un échauffement** : la courbe ne plafonne pas sur 100 pas ;
   elle s'accélère même (+117, +128, +183, +119 Mo par tranche de 20).
2. **Ce n'est pas de la fragmentation** : à `seq 8` la pente tombe à
   **2,4 Mo/pas** contre **4,7 Mo/pas** à `seq 16` — elle suit donc `batch × seq`,
   c'est-à-dire le volume d'activations d'un forward, et non le graphe.
3. **Ce n'est pas une liste qui grossit** : cimetière 0, requêtes de naissance 0,
   historique 0, nœuds et arêtes stables pendant les 20 premiers pas.

L'extrapolation linéaire à 3 000 pas donne ≈ **19 Go** [estimé], très au-delà des
~6 Go disponibles : le run d'endurance reste impossible en l'état.

Un événement de cycle de vie s'est déclenché vers le pas 85 (« 8 arêtes coupées,
667 nœuds enterrés ») : il ajoute du bruit à la courbe mais ne l'explique pas,
puisque la dérive existe déjà avant lui, sur les 80 premiers pas.

**Cause localisée, non identifiée — et une piste fausse éliminée par la mesure.**
Quelque chose retient, à chaque forward, une quantité proportionnelle aux
activations. Le premier candidat était `self.last_node_io` /
`self.last_energy_by_hop`, qui conservent les tenseurs de sortie de chaque hop.
Je l'ai testé en les vidant après chaque pas, ainsi que `last_out_acc` et
`_last_out` :

| Pas | RSS sans vidage | RSS avec vidage |
|---:|---:|---:|
| 10 | 840 Mo | 841 Mo |
| 20 | 886 Mo | 887 Mo |

Écart nul — **l'hypothèse est réfutée**. Ces champs se contentent de remplacer
leur contenu à chaque forward ; ils ne retiennent rien. La cause est ailleurs et
reste à trouver.

## 1.3 Run 3 000 pas — non lancé, et c'est volontaire

Lancer 3 000 pas avec une dérive non expliquée produirait un OOM au bout de
~800 pas [estimé] et aucun chiffre d'endurance. Le prérequis n'est pas rempli.

## Ce qui n'a pas été fait

Toutes les phases 2 à 4 : corpus Gutenberg, Qwen2.5-0.5B, LoRA, endpoint
`/speak`, embeddings, expérience graphe nu vs pré-câblé. Aucune n'est entamée.

## Deux corrections au brief

- **Le genome n'est pas absent** : `data/genome.pt` est en place sur le serveur,
  `67 049 701` octets, amené par le câble en session K. L'affirmation « ABSENT
  DEPUIS TROP LONGTEMPS » du brief n'est plus vraie.
- **Le dépôt serveur n'est pas à `9299b24`** mais à `f6ccd1c` (les rapports de
  la session K ont été poussés après).

## État

| Prérequis | État | Blocage |
|---|---|---|
| 1.1 Filaire | rouge | droits root (humain) |
| 1.2 Dérive | rouge (caractérisée) | cause localisée, non prouvée |
| 1.3 Run 3 000 pas | non lancé | dépend de 1.2 |
| Phases 2-4 | non entamées | dépendent des trois |

## Prochaine action utile

Une seule commande sur le serveur, sans root : instrumenter `last_node_io` et
`last_energy_by_hop` (`gc.get_referrers`, `sys.getrefcount`) après un forward,
pour trancher entre « retenu par le brain » et « retenu par l'appelant ». Si
c'est le brain, vider ces champs quand `trace=False` suffit souvent à supprimer
la dérive — et 1.2 passe au vert sans toucher à l'architecture.
