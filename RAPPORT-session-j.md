# Session J — le GPU est constaté, la mesure ne tient pas en mémoire

Suite de la phase 3. Trois choses ont changé : le filaire, les conteneurs, et
surtout le GPU. Une chose n'a pas abouti : le chiffre du benchmark dans des
conditions propres.

## 1. Salle blanche : le vrai coupable était docker

Les services systemd (`clickhouse-server`, `litellm`, `open-webui`, `speaches`,
`hermes`) étaient déjà `inactive` : ils ne tenaient rien. C'était **23
conteneurs `ods-*`** (Langfuse, qdrant, n8n, whisper, tts, minio, perplexica…)
qui occupaient la mémoire, avec le swap plein à 94 %.

| Mesure | Avant | Après | Étiquette |
|---|---|---|---|
| Conteneurs actifs | 23 | **0** (24 en `restart=no`) | [mesuré] |
| Mémoire disponible | 1 366 Mo | **5 743–6 216 Mo** | [mesuré] |
| Swap utilisé | 3 797 Mo | **305 Mo** | [mesuré] |
| Charge moyenne | 56 à 102 | **1,08 → 2,66** | [mesuré] |
| GPU | 3 MiB / 0 % | 3 MiB / 0 % | [mesuré] |

`docker stop` et `docker update --restart=no` ne demandent pas les droits root
(`utilisateur` est dans le groupe docker) : c'est réversible par `--restart=always`
puis `start`.

## 2. GPU : `(5, 0)` constaté, plus jamais supposé

Le diagnostic de la session précédente était incomplet, et c'est important de
le dire : `nccl 2.29.3` et `cudnn 9.10.2.21` **existent bien**, mais sous des
étiquettes de plateforme que mes filtres ne fournissaient pas
(`manylinux_2_18_x86_64` et `manylinux_2_27_x86_64`). Pip les déclarait donc
inexistantes. Seul `triton 3.8` est réellement absent en cp314 — il s'arrête à
cp313. La conclusion opérationnelle reste la même : il faut Python 3.12.

Réalisé par le câble, le serveur n'ayant plus aucun accès sortant :

- CPython 3.12.16 portable (`python-build-standalone`, 68 163 679 o, md5
  `fe3c763ce578c2f25f52e0bf1e686a16`) ;
- lot complet de 27 roues cu126 en cp312, **3 636 963 137 o** ;
- venv `.venv-cu126`, torch `2.14.0+cu126`.

Résultat, mesuré sur la machine :

```
version     2.14.0+cu126
arch_list   ['sm_50', 'sm_60', 'sm_70', 'sm_75', 'sm_80', 'sm_86', 'sm_90']
disponible  True
cartes      1
nom         NVIDIA GeForce GTX 860M
capacite    (5, 0)
```

`sm_50` est dans la liste des architectures compilées et `(5, 0)` est **constaté
par `get_device_capability`**, plus supposé [mesuré]. C'est la première fois que
la 860M est réellement vue par torch dans ce projet.

## 3. Benchmark : une mesure, mais pas dans les conditions requises

Trois obstacles, tous mesurés, et aucun n'est un caprice de méthode :

**a. L'échelle « 1 000 nœuds » n'existe pas dans le code.**
`SCALE_NODES = {0: 512, 1: 302, 2: 3000, 3: 140 000}`. Il n'y a aucune échelle à
1 000 nœuds : le brief et le code ne parlent pas de la même taille. Les tailles
mesurables sont 302 (C. elegans) et 512 (prototype).

**b. La machine ne peut pas porter 512 nœuds / 4 000 arêtes.**
Le processus a été tué par le noyau :

```
Out of memory: Killed process 146380 (python)
  total-vm:14846704kB, anon-rss:6451000kB, oom_score_adj:0
```

6,45 Go de mémoire anonyme pour un total de 7 350 Mo sur la machine. Ce n'est
pas la charge résiduelle : c'est le graphe lui-même. Deux autres tentatives ont
fini de la même façon (processus disparu, journal arrêté après l'en-tête).

**c. La seule mesure complète a été obtenue avec des processus résiduels.**
Un relevé de 20 itérations a produit :

```
device cpu | batch 2 | seq 16 | arêtes 1500 | nœuds 302 | dim 32 | hops 8 | tbptt 16 | data toy
step_once : médiane 3214 ms/pas [2985–5433] | RAM 2382 Mo (pic 2382, commit 27461, +10/pas)
```

Cette valeur est **publiée avec sa réserve** : au lancement, quatre processus
`bench_step.py` résiduels étaient encore actifs (« actifs avant : 4 »), donc la
médiane mesure aussi leur contention [mesuré, conditions non propres]. La
reprise propre lancée ensuite n'a produit aucune ligne — le processus est mort
avec la session SSH.

**Aucun chiffre du benchmark n'est donc publié comme mesure de référence.**

## 4. Filaire : toujours pas de bail, et le WiFi ne route plus

L'étape 1 reste **non faite** : le script `~/prepare-filaire.sh` n'a pas tourné
avec les droits root, et `/etc/netplan/` ne contient toujours que ses trois
fichiers d'origine. `enp8s0` reste en `169.254.184.83/16`.

Aggravation mesurée pendant la session : le WiFi est associé au point d'accès
(SSID `wifi`, 5 GHz) mais **ne route plus rien** — passerelle à 100 % de perte,
PyPI et PyTorch en `000`. Tout l'accès passe donc par le lien-local IPv6 du
câble, `SERVEUR-IPV6%15`, à 0 ms et 0 % de perte.

## 5. État et livrables

| Élément | État |
|---|---|
| Conteneurs | 0 actif, 24 en `restart=no` |
| Swap | 305 Mo utilisés sur 4 095 |
| torch GPU | `2.14.0+cu126`, `(5, 0)` constaté |
| torch CPU | `2.14.0+cpu` dans `.venv` (Python 3.14) |
| Tests | 58 passed, 4 skipped (mesurés en session I) |
| Benchmark | mesuré une fois, conditions non propres |
| Run 3 000 pas | jamais lancé |
| Netplan / swapoff | en attente des droits root |
| genome `.myc` | absent du dépôt serveur |

## Ce qu'il faut faire ensuite

1. Lancer `sudo bash ~/prepare-filaire.sh` (étape 1 toujours ouverte).
2. Mesurer le benchmark sur 302 nœuds **sans aucun processus résiduel**, et
   consigner la charge et la mémoire juste avant et juste après.
3. Décider ce que « 1 000 nœuds » veut dire, puisque aucune échelle ne le
   fournit : soit on mesure 512, soit on ajoute une échelle au code.
