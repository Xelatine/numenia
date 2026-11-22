> **Renommage —** Ce projet s'appelle désormais **Numenia** (auparavant référencé
> comme Mycelia dans les titres et la documentation). Session, chiffres, fichiers de
> données et cerveau **Nyra** inchangés. Ce document est historique : son contenu est
> conservé tel quel, et seul le nom du projet passe à Numenia.

# Session H — « la plomberie »

10 novembre 2026. Objectif : faire passer torch sur la 860M, puis produire la
première mesure de temps propre du projet. **Résultat : la plomberie s'arrête au
transfert.** Aucune mesure de temps n'a été prise, aucun pas n'a tourné sur la
carte, et c'est ce que ce rapport documente.

## Avertissement de mesure, avant tout chiffre

La règle 7 est **satisfaite** : la carte est quasi vide (3 MiB utilisés, 0 %
d'occupation, relevé deux fois à quatre heures d'intervalle). La salle blanche
existe et elle est constatée — [mesuré]. Elle n'a servi à rien, parce que rien ne
pouvait y être exécuté : torch n'est pas installé.

Aucun temps de cette session n'est un temps d'entraînement. Les seuls chiffres de
temps ci-dessous mesurent des **transferts de fichiers**, pas des pas de calcul.

## Ce qui est fait, et vérifié

| Élément | État | Étiquette |
|---|---|---|
| Dépôt `~/mycelia` sur la 860M à `a516706`, arbre propre | fait, `git bundle verify` OK | [mesuré] |
| `uv 0.13.7` dans `~/.local/bin`, sans privilèges | installé | [mesuré] |
| venv `~/mycelia/.venv`, Python 3.14.4 | créé par uv | [mesuré] |
| Wheel `torch-2.14.0+cu126-cp314` + 9 dépendances sur le poste | 869 180 118 octets, md5 `b7da24fc1e10e598215e358666d31174` | [mesuré] |
| Wheel sur le serveur | **non** | — |
| torch importable sur le serveur | **non** | — |
| Carte propre | GTX 860M, 2 048 MiB, 3 MiB utilisés, 0 % | [mesuré] |

Deux obstacles système ont été levés sans root : `python3.14 -m venv` échoue
(`ensurepip` absent, paquet `python3.14-venv` non installé), et le Python système
est en PEP 668 ; `uv` résout les deux puisqu'il installe les paquets sans passer
par pip.

## Le vrai adversaire : le lien radio du serveur

Diagnostic mesuré, et il est net :

| Cible | Paquets | Perte | Latence |
|---|---:|---:|---|
| Passerelle PASSERELLE-LAN | 20 | **0 %** | 1 ms |
| Serveur SERVEUR-LAN | 30 | **16,7 %** | 3 à 130 ms, 66 ms de moyenne |

La passerelle est parfaite. C'est donc le lien radio du serveur qui perd des
paquets, pas le LAN. Conséquence : toute pile TCP s'effondre dès qu'on lui demande
quelques centaines de mégaoctets.

## Les sept méthodes essayées, et ce qu'elles ont donné

| Méthode | Débit | Atteint | Verdict |
|---|---:|---:|---|
| `uv pip install` cu126 (serveur → Internet) | 1,07 Mo/s en rafale puis 0 | ~14 Mo | échec : `connection reset`, uv **repart de zéro** à chaque essai |
| `curl -C -` wheel CPU 169 Mo (serveur → Internet) | ~600 Ko/s puis 0 | 45,5 Mo / 169,1 Mo | échec : resets en boucle |
| `scp` LAN, 1 Mo puis 50 Mo | 0 | 0 | échec : reset à 24 s et 25 s |
| `scp` 4 Mo × 6 flux | 186 Ko/s utiles | 12 / 208 tranches | fonctionne, lent (39 envois, 12 succès) |
| `scp` 4 Mo × 8 flux | ~75 Ko/s utiles | **104 071 168 octets** | meilleur résultat de la session |
| `scp` 4 Mo × 24 flux | 0 | 0 tranche | échec : `sshd` étranglé, 1040 envois en 113 s tous refusés |
| flux `tar` unique de 40 Mo | 0 | 0 | échec : coupure à 24 s |
| serveur → Internet, 8 × Range 2 Mo | 92 Ko/s | 14 292 585 octets | fonctionne, plus lent que le push |
| serveur → Internet, 1 × Range 8 Mo | 48 Ko/s | 4 321 174 octets | `206` confirmé, le débit ne suit pas |
| reprise par offset `tail -c +N \| ssh 'cat >>'` | 10 Ko/s | 5 235 035 octets | le plus économe en connexions, lien au plus bas |
| HTTP, le serveur tire depuis le poste | 13 Ko/s | 0,7 Mo / 55 s | pire que le push, abandonnée |

Le meilleur transfert atteint **12 %** de la wheel. Le fichier complet fait
869 180 118 octets ; il en manque 863 944 783.

## Retest de la même nuit

Second essai, deux heures après le premier, pour savoir si l'échec était
conjoncturel ou structurel.

| Mesure | Valeur | Étiquette |
|---|---|---|
| Perte ping vers le serveur | **25 %** (9 reçus sur 12), 3 à 991 ms, 158 ms de moyenne | [mesuré] |
| Connexions `ssh` acceptées | **1 sur 6** | [mesuré] |
| `scp` de 8 Ko | refusé (« Connection closed », 19 s) | [mesuré] |
| `scp` d'une tranche de 4 Mo | refusé | [mesuré] |
| Flux `ssh 'cat >>'` par offset | 5 647 992 octets en 290 s = **19 Ko/s** | [mesuré] |

Le diagnostic se précise : `scp` est désormais refusé, mais le flux `ssh` à
sortie redirigée passe encore. Le fichier est à **22 245 730 octets** sur
869 180 118 (2,6 %). Au débit du retest il reste 846 934 388 octets, soit
**≈ 12 h** [estimé] ; au meilleur débit jamais observé (38 Ko/s), ≈ 6 h.

## Troisième relevé — le lien s'effondre

Troisième essai, vingt minutes après le second.

| Mesure | Valeur | Étiquette |
|---|---|---|
| Perte ping vers le serveur | **60 %** (4 reçus sur 10), 45 à 112 ms | [mesuré] |
| Connexions `ssh` acceptées | **0 sur 6** | [mesuré] |
| `ssh 'cat >>'` (1 896 octets, le bundle) | échec aux 5 essais, fichier distant vide | [mesuré] |

Trajectoire de la nuit, sur la même liaison : 16,7 % de perte, puis 25 %, puis
60 %. Aucun transfert n'aboutit plus, pas même 1 896 octets. Le fichier est resté
à **22 245 730 octets** sur 869 180 118 (2,6 %).

```
## Deux erreurs de conduite, consignées

1. J'ai supprimé les 104 071 168 octets déjà reçus (`rm -f ~/<wheel>`) en
   repassant à la méthode par offset, perdant le meilleur acquis de la session.
2. J'ai lancé 24 flux parallèles alors que 6 flux montraient déjà des échecs :
   résultat zéro tranche et un `sshd` saturé. Le parallélisme n'a jamais battu
   les 8 flux.

## Les trois missions, sans enjolivure

**Mission 1 — faire entrer torch.** Partielle. L'outillage est en place, la wheel
est prête et vérifiée sur le poste, le transfert est bloqué par le lien. La
compatibilité sm_50 de `torch 2.14.0+cu126` vient du commentaire de
`requirements.txt` : elle n'a **pas** été constatée sur la machine — [supposé].

**Mission 2 — le benchmark de 100 pas.** Non fait. Il exige torch, torch exige
869 Mo, et 869 Mo ne passent pas. Le critère avait été déclaré d'avance (100 pas
sans OOM, pic cohérent avec 483,73 Mo, temps par pas sur carte à 0 %) : il reste
entier, non mesuré.

**Mission 3 — la campagne G-bis.** Non faite, même raison. `--chunk 2048` à
seq 128/b16 et le point OFF à 1 000 arêtes attendent toujours. Le OFF à 3 000
reste un trou structurel de cette machine : 4 212,70 Mo [mesuré en session G] ne
tiennent pas dans 2 048 MiB.

**Le run 3 000 pas.** Non lancé — aucun « lance » humain, et la machine n'est pas
capable de l'exécuter aujourd'hui.

## Ce qu'il faut pour finir

Le fichier à faire arriver : `torch-2.14.0+cu126-cp314-cp314-manylinux_2_28_x86_64.whl`,
869 180 118 octets, md5 `b7da24fc1e10e598215e358666d31174`. Il est déjà sur le
poste, dans `.server/wheels/`.

```bash
# reprise de l'envoi (script du dépôt, reprend par offset, budget illimité)
bash .server/upstream.sh ".server/wheels/torch-2.14.0+cu126-cp314-cp314-manylinux_2_28_x86_64.whl" \
     "~/torch-2.14.0+cu126-cp314-cp314-manylinux_2_28_x86_64.whl" 36000

# état vu de l'extérieur, en une commande
ssh -i .server/cle-ssh utilisateur@SERVEUR-LAN 'cat ~/dl-status.txt'

# une fois le md5 concordant
ssh ... 'cd ~/mycelia && ~/.local/bin/uv pip install --python .venv/bin/python \
         ~/torch-2.14.0+cu126-cp314-cp314-manylinux_2_28_x86_64.whl'
ssh ... 'cd ~/mycelia && CUDA_VISIBLE_DEVICES=-1 .venv/bin/python -m pytest -q'
ssh ... 'cd ~/mycelia && .venv/bin/python -c "import torch; print(torch.__version__, \
         torch.cuda.is_available(), torch.cuda.get_device_capability(0))"'
```

Durée estimée du transfert restant : **3 à 24 heures** selon la qualité du lien
[estimé depuis 10-75 Ko/s mesurés]. Réserves : ces débits varient d'un facteur 10
dans la même heure et ne sont pas extrapolables.

**Le remède structurel n'est pas logiciel** : la passerelle répond à 1 ms sans
perte, seul le lien radio du serveur perd des paquets. Un câble Ethernet sur la
860M supprimerait le problème à la racine, et c'est la seule action qui rende
cette machine utilisable comme machine de mesure du projet.

## État final

**Bloqué sur le transfert de torch — rapport d'échec documenté, run non lancé.**
La voie la plus avancée est l'envoi par tranches vérifiées (104 Mo atteints), la
méthode restante est la reprise par offset, et le fichier exact à obtenir est
identifié par son md5.
