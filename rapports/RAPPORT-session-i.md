# Session I — le câble était déjà branché

Reprise après la session H, qui s'était arrêtée sur un transfert impossible
(19 Ko/s au mieux, 25 % puis 60 % de perte radio, puis plus aucune connexion).

## Ce qui a débloqué la situation

Le câble Ethernet **était branché depuis le début**. La carte filaire du
serveur, `enp8s0`, était `UP, LOWER_UP`, carrier à 1, vitesse négociée
1 000 Mb/s — mais un profil netplan nommé `netplan-enp8s0` lui donne une
adresse lien-local `169.254.184.83/16` et **aucun bail DHCP**. Tout le trafic
continuait donc de passer par `wlp7s0`, le radio qui perdait les paquets.

Le chemin filaire est pourtant utilisable sans aucun droit root, en IPv6
lien-local : `fe80::226a:8aff:fea3:5e67%15` (index 15 = carte Ethernet du
poste). C'est ce chemin qui a servi à tout le reste de la session.

| Mesure | Valeur | Étiquette |
|---|---|---|
| Ping filaire vers le serveur | 4/4, 0 %, 0 ms | [mesuré] |
| Wheel torch 869 180 118 o par `scp` IPv6 | **10 s**, soit ≈ 82,9 MiB/s (663 Mbit/s) | [mesuré] |
| Intégrité de la wheel reçue | md5 `b7da24fc1e10e598215e358666d31174` conforme | [mesuré] |
| Wheelhouse complète 879 677 434 o | 10 s | [mesuré] |
| 44 roues de dépendances, 101 410 221 o | 3 s | [mesuré] |
| Fermeture CUDA 12.6, 2 483 565 691 o | 24 s | [mesuré] |

Comparaison directe avec la session H : 19 Ko/s au mieux cette nuit, 86,9 Mo/s
maintenant — facteur ≈ 4 500. Le goulot n'a jamais été la méthode de transfert,
c'était l'adresse.

## Ce qui a marché

- `scp` IPv6 filaire : fonctionne, y compris pour 869 Mo d'un coup.
- Dépôt serveur amené à `e998821`, arbre propre, par bundle de 3 958 o.
- 10 modules du projet installés (numpy 2.5.4, pytest, fastapi, pydantic,
  httpx, uvicorn, datasets, tokenizers, pyarrow, PyYAML).
- **Suite de tests : 58 passed, 4 skipped en 102,67 s**, sur le serveur, en CPU.

## Ce qui n'a pas marché : le GPU

La wheel `torch-2.14.0+cu126` ne peut pas être liée sur cette machine :

1. `import torch` réclame successivement `libcublasLt`, `libcudnn.so.9`,
   `libcufile.so.0` — puis `ldd libtorch_cuda.so` donne la liste complète :
   `libcufft.so.11`, `libcufile.so.0`, `libcupti.so.12`, `libcurand.so.10`,
   `libcusparseLt.so.0`, `libcusparse.so.12`, `libnccl.so.2`,
   `libnvshmem_host.so.3`.
2. Ces huit bibliothèques ont été installées (16 roues, 2,48 Go).
3. `ldd` ne signale alors plus aucun manquant, mais l'édition de liens casse :
   `undefined symbol: ncclCommGrow`.

Cause : la wheel épingle `nccl 2.29.3`, `cudnn 9.10.2.21` et `triton ~3.8.0`,
et **ces versions ne sont publiées pour aucune plateforme cp314**, ni sur PyPI
ni sur `download.pytorch.org/whl/cu126`. Le meilleur nccl disponible est
2.27.7, qui ne connaît pas le symbole attendu. Le build est donc incohérent
avec lui-même tel que publié — ce n'est pas une erreur de manipulation.

Repli assumé : `torch-2.14.0+cpu` (196 Mo) installé à la place. L'import passe,
mais `torch.cuda.is_available()` est **False** par construction. La capacité
`(5, 0)` de la 860M reste donc **[supposé]**, jamais constatée.

## Ce qui n'a pas été mesuré : le benchmark

**Aucun chiffre de temps par pas n'est publié dans cette session.** Deux
lancements, aucun résultat :

- lancement 1 (`--iters 100`) : processus mort sans écrire une ligne ;
- lancement 2 (`--iters 8`, sortie non tamponnée) : processus lancé (pid 66475),
  puis `sshd` est devenu injoignable — 6 tentatives, aucune bannière.

La raison est nommable et grave : **la machine n'est pas une salle blanche**.
Au moment des mesures elle faisait tourner ClickHouse, litellm, open-webui,
speaches, hermes, docker et une douzaine de services, avec une charge moyenne
entre 56 et 102 et 1 575 Mo de RAM disponibles sur 7 350. La règle du projet
— « pas d'entraînement sur la machine de mesure » — est violée par l'état de la
machine elle-même, pas par la procédure.

La carte graphique, elle, était propre : 3 MiB utilisés sur 2 048, 0 %
d'utilisation. C'est la seule condition du protocole qui soit remplie.

## Erreurs de conduite de cette session

1. `scp -r .server/cuda-min host:~/cuda-min` a créé `~/cuda-min/cuda-min` :
   seuls 5 paquets ont été installés, d'où deux cycles perdus à chercher des
   bibliothèques qui étaient pourtant arrivées. Corrigé en copiant les fichiers.
2. Un glob `~/cpu/*.whl` non expansé a été passé tel quel à `uv` alors que le
   répertoire distant n'existait pas encore : une installation perdue.
3. Deux benchmarks lancés sans `python -u` : la sortie est restée dans le
   tampon, ce qui a fait croire à un blocage.
4. Trois itérations à installer les bibliothèques CUDA une par une avant de
   penser à `ldd`, qui donne la liste complète en une commande.

## État à la fin de la session

| Élément | État |
|---|---|
| Transfert de fichiers | **résolu** (câble, IPv6 lien-local, 82,9 MiB/s) |
| Dépôt serveur | `e998821`, propre |
| torch | `2.14.0+cpu` installé, import OK |
| Tests CPU | 58 passed, 4 skipped |
| GPU sm_50 | bloqué par les pins de la wheel, non constaté |
| Benchmark 100 pas | non mesuré, machine saturée |
| Run 3 000 pas | jamais lancé |

## Ce qu'il faut faire ensuite

1. Rendre la machine à l'état de salle blanche : arrêter ClickHouse, litellm,
   open-webui, speaches, hermes et les conteneurs avant toute mesure. Sans
   cela, tout chiffre de temps mesurera la contention, pas le graphe.
2. Rendre le lien filaire permanent : `dhcp4: true` pour `enp8s0` dans le
   profil netplan (une commande avec droits root), pour ne plus dépendre d'une
   adresse lien-local et d'un index d'interface.
3. Pour le GPU, la seule voie propre est un build `cu126` dont les dépendances
   existent pour cp314 — c'est-à-dire attendre une publication corrigée, ou
   construire depuis les sources.
