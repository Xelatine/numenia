# Session U — verdict en attente, porte dorée bloquée par le réseau

**Trois faits, dans l'ordre où ils ont été constatés :**

1. le jumeau sparse **tourne encore** (2 640/3 000 au dernier relevé) : son JSON
   n'existe pas, donc **aucun verdict n'est lisible** ;
2. le réseau de BIO-1 est **tombé** (DNS et route absents) : l'encodeur
   EmbeddingGemma et les organes sensoriels ne peuvent pas être téléchargés ;
3. le mécanisme de nom est **implémenté et testé** (4/4), sans nommer
   l'organisme à sa place.

Aucune de ces trois lignes n'est un contournement : ce sont les mesures du jour.

## 1. Jumeau sparse : EN COURS, verdict non rendu

| | |
|---|---|
| État | PID 238894 vivant, 10 h 08 d'exécution |
| Dernier pas | **2 640 / 3 000** |
| Rythme | ~17 s/pas → reste ~1 h 40 |
| HWM | **750 Mo** (plafond 800) |
| Perte au pas 2 600 | **0,1824** (référence dense finale : 0,002342) |
| Arêtes / nœuds | 6 929 / 1 008, cimetière 4 |
| Seuil étage 1 | perte finale ≤ 0,004684 |

La perte courante (0,18) est encore **77×** au-dessus du seuil d'admissibilité,
mais elle baisse (4,64 au pas 100 → 0,18 au pas 2 600) et le dense avait la même
trajectoire avant de chuter. **Prédire le verdict serait exactement ce que le
protocole interdit.** Il sera dans `data/mem_report/sparse_3000.json` :

```
ssh -i .server/id_ed25519_mycelia -6 dieux@fe80::226a:8aff:fea3:5e67%15 \
  'cat ~/mycelia/data/mem_report/sparse_3000.json'
```

Si `stage1_admissible` est vrai, la campagne top-k s'exécute automatiquement ;
si faux, le Plan B (Top-KAST / gate Switch) s'applique — et sera consigné comme
échec de la piste A.

## 2. Golden 5/5 : BLOQUÉ PAR L'INFRASTRUCTURE, pas par la mesure

`enp8s0` ne porte qu'une adresse lien-local `169.254.184.83/16`, aucune route
par défaut, Wi-Fi `DOWN`. `nmcli connection up` répond **« Not authorized to
control networking »** (root/polkit indisponible). `huggingface.co` et `pypi.org`
ne résolvent plus.

Conséquence : **EmbeddingGemma-300M et bge-m3 ne peuvent pas être obtenus**, donc
le golden reste rouge et l'ingestion reste fermée. Ce n'est pas un échec de la
piste encodeur — c'est une panne réseau constatée.

Dernière tentative **offline** sur Qwen (déjà présent) : les préfixes
d'instruction n'aident pas, ils détruisent :

| Variante | HIT@1 |
|---|---|
| brut | 3/5 |
| « Instruct: Retrieve the passage… » | **1/5** |
| « task: search result \| query: » | **1/5** |

Aucun encodeur local ne passe la porte. Rien n'est maquillé en « presque 5/5 ».

## 3. Le nom : mécanisme B défini, testé, non exécuté sur l'organisme

`mycelia/naming.py` — l'organisme se nomme **au premier Sommeil**, à partir de sa
structure, jamais au hasard :

- signature = (nœuds vivants, arêtes, suite des degrés **triée**) ;
- projection SHA-256 sur une table de syllabes fixe (24 × 10 × 12) ;
- **indépendant de l'ordre des arêtes** (testé par permutation) ;
- **figé dès l'écriture** : l'ingestion peut changer le graphe, le nom ne bouge
  plus (testé : second graphe → même nom) ;
- survit à la copie/compaction (testé).

4/4 tests. `self.json` conserve pour l'instant
« sans-nom (à nommer au premier Sommeil) » : **la machine n'a pas nommé
l'organisme à la place de l'utilisateur**.

## 4. Ce qui n'a pas été fait, et pourquoi

| Action | État | Raison |
|---|---|---|
| Verdict jumeau | **en attente** | run à 88 %, non interrompu |
| Golden EmbeddingGemma | **bloqué** | réseau/DNS morts, root absent |
| Ingestion 21,3 M tokens | **non lancée** | golden rouge + verdict absent |
| Encodage corpus | **non lancé** | même condition |
| Whisper / SmolVLM | **non intégrés** | téléchargements impossibles |
| Niveaux 3-4 | **stubs** | pas de détecteur d'ignorance |
| Test final | **impossible** | dépend de tout le reste |

## État consolidé

| Élément | État |
|---|---|
| Jumeau sparse | **en cours**, HWM 750 Mo, perte 0,18 au pas 2 600 |
| Golden encodeur | **rouge** — Qwen 3/5, variantes 1/5, aucun encodeur dédié téléchargeable |
| Nom | **mécanisme prêt**, non exécuté (choix utilisateur ou premier Sommeil) |
| Identité offline | **verte** (8 tests + 4 tests de nom) |
| Organes externes | **câblés, stubs** |
| Réseau BIO-1 | **HS** (lien-local seul, pas de route, nmcli non autorisé) |

La prochaine session n'a qu'un seul préalable non négociable : **rétablir le
réseau** (connexion filaire ou Wi-Fi avec privilèges) pour obtenir un encodeur.
Sans lui, ni le golden, ni l'ingestion, ni les sens ne peuvent avancer — et le
dire est plus utile que de lancer une tâche dense « en attendant ».
