> **Renommage —** Ce projet s'appelle désormais **Numenia** (auparavant référencé
> comme Mycelia dans les titres et la documentation). Session, chiffres, fichiers de
> données et cerveau **Nyra** inchangés. Ce document est historique : son contenu est
> conservé tel quel, et seul le nom du projet passe à Numenia.

# Session K — le chiffre de référence, à 1 000 nœuds

Objectif de la session : un chiffre de benchmark en conditions propres, et un
graphe qui tient en mémoire. Les deux sont atteints, et le diagnostic mémoire de
la session précédente était faux — c'est le point le plus important de ce
rapport.

## 1. L'échelle 1 000 existe maintenant

Décision prise avec l'humain : ajouter le nominal plutôt que renommer
l'existant. `SCALE_NODES` passe de `{0: 512, 1: 302, 2: 3000, 3: 140 000}` à
`{0: 512, 1: 302, 2: 3000, 3: 140 000, 4: 1_000}`, avec le nom
« nominal (1 000 nodes) ». Renuméroter 0..3 aurait invalidé tous les chiffres
déjà publiés contre ces échelles.

Les trois points d'entrée qui figeaient la liste (`mycelia/train.py`,
`scripts/bench_step.py`, `scripts/dashboard.py`) acceptent désormais 4. Commit
`9299b24`, poussé.

## 2. Pourquoi le benchmark mourait : ce n'était pas le graphe

La session J concluait à une empreinte du graphe lui-même (6,45 Go, OOM noyau à
512 nœuds). **C'était faux.** L'empreinte vient du produit `batch × seq` :

| Config | Nœuds | RAM mesurée | Résultat |
|---|---:|---:|---|
| batch 16, seq 128 | 512 | **6 450 Mo** | tué par l'OOM killer |
| batch 2, seq 16 | 302 | 2 385 Mo | OK |
| batch 2, seq 16 | 512 | 2 034 Mo | OK |
| batch 2, seq 16 | **1 000** | **1 045 Mo** | OK |

Autrement dit : ajouter des nœuds n'ajoute pas de mémoire ; c'est la séquence
de 128 positions traitée dans un seul graphe qui coûtait 6 Go. Le docstring de
`bench_step.py` le disait déjà — « a text sequence of 128 positions in ONE graph
is exactly what the run never does ». La session J a mesuré une configuration
que le projet ne fait jamais tourner, puis en a conclu un problème de graphe.

Aucune optimisation de code n'a donc été nécessaire : ni dtype, ni mmap, ni
détachement d'états.

## 3. Mesures de référence

Conditions attestées, relevées juste avant et juste après chaque point :

- 0 conteneur actif (23 stoppés, 24 en `restart=no`) ;
- charge avant : 0,06 ; 6 142 Mo disponibles ; swap 723 Mo ;
- GPU : 3 MiB utilisés sur 2 048, 0 % ;
- aucun processus `bench_step.py` résiduel ;
- accès par le câble (IPv6 lien-local), 0 ms, 0 % de perte.

| Nœuds | Arêtes | batch | seq | itérations | Médiane | Amplitude | RAM | Dérive commit |
|---:|---:|---:|---:|---:|---:|---|---:|---:|
| 302 | 1500 | 2 | 16 | 20 | **2 932 ms/pas** | 2 810–3 074 | 2 385 Mo | +10 Mo/pas |
| 512 | 1500 | 2 | 16 | 10 | **2 826 ms/pas** | 2 751–3 026 | 2 034 Mo | +18 Mo/pas |
| **1 000** | 1500 | 2 | 16 | 10 | **2 099 ms/pas** | 2 010–2 238 | 1 045 Mo | +13 Mo/pas |

Toutes en CPU (`--only step`, `CUDA_VISIBLE_DEVICES=-1`), dimension 32, 8 hops,
tbptt 16, données `toy`. Étiquette : [mesuré], conditions propres.

**Anomalie consignée, pas expliquée** : 1 000 nœuds est *plus rapide* (2 099 ms)
que 302 nœuds (2 932 ms) à nombre d'arêtes égal. À 1 500 arêtes fixes, le graphe
de 1 000 nœuds est bien plus dilué : moins de routes actives par hop. C'est
cohérent, mais ce n'est pas démontré — il faudrait balayer les arêtes à nœuds
fixes pour l'établir.

## 4. Ce qui reste ouvert

**Dérive mémoire** : +13 Mo/pas à 1 000 nœuds. Sur 10 pas c'est négligeable ;
sur 3 000 pas, cela ferait ≈ 38 Go [estimé par extrapolation linéaire], donc le
run long reste hors de portée tant que la dérive n'est pas expliquée. C'est
maintenant le vrai chantier mémoire, et il est mesurable en une commande.

**Durée du run 3 000 pas** : 2 099 ms × 3 000 ≈ **1 h 45** [estimé], sans
compter les checkpoints. Jamais lancé.

**Filaire** : toujours pas de bail DHCP. `/etc/netplan/` ne contient que ses
trois fichiers d'origine ; `enp8s0` reste en `169.254.184.83/16`. Le script
`~/prepare-filaire.sh` attend toujours une exécution avec les droits root. Le
WiFi est associé mais ne route rien : tout passe par le lien-local IPv6.

**Genome `.myc`** : absent du dépôt, local comme serveur.

## État

| Élément | État |
|---|---|
| Échelle 1 000 | implémentée, commit `9299b24` poussé |
| Benchmark 302 / 512 / 1000 | mesuré, conditions propres |
| OOM | expliqué (batch × seq), non reproductible à b2/s16 |
| Dérive +13 Mo/pas | mesurée, non expliquée |
| Netplan | en attente des droits root |
| Run 3 000 pas | jamais lancé, ≈ 1 h 45 estimées |
| Genome `.myc` | absent |
