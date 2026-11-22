> **Renommage —** Ce projet s'appelle désormais **Numenia** (auparavant référencé
> comme Mycelia dans les titres et la documentation). Session, chiffres, fichiers de
> données et cerveau **Nyra** inchangés. Ce document est historique : son contenu est
> conservé tel quel, et seul le nom du projet passe à Numenia.

# Session N — Dérive mémoire : fragmentation glibc, pas fuite Python

**Date** : 12 novembre 2026 — **Machine** : BIO-1 — **Code** : `mycelia/train.py` patché (md5 `05c67c0f…`, serveur = poste).

## Résumé exécutif (CONSTATÉ)

- **Les deux lignes désignées par tracemalloc sont innocentes.** `brain.py:345` (`_edge_keys.add`) : **+0,01 Mo / +120 objets** sur 15 pas (set borné à `max_edges=20000`, il en contenait 172). `brain.py:956` (`last_firing_nodes = sorted(...)`) : **+0,01 Mo / +129 objets** (liste bornée par `n_nodes`). La mémoire Python tracée est **plate** : 72,44 → 72,48 Mo.
- **La croissance est native et c'est de la fragmentation du tas glibc.** Sur 20 pas : `uordblks` (octets en usage) **plat** 195,2 → 195,3 Mo ; `fordblks` (libre non rendu) 39,7 → 103,7 Mo ; `arena` 234,9 → 299,0 Mo — soit exactement la pente RSS. Tenseurs vivants : 111 → 117 objets, **98,9 Mo constants**.
- **Mono-thread identique** (+4,39 vs +4,38 Mo/pas) → ce ne sont pas les arènes de threads.
- **`malloc_trim(0)` après chaque pas : +0,07 Mo/pas** (RSS 435→436 Mo sur 15 pas). `MALLOC_MMAP_THRESHOLD_=131072` : +4,21 Mo/pas (sans effet).
- **Patch** : `mycelia/train.py` appelle `trim_heap()` en fin de `step_once()` (glibc uniquement, no-op ailleurs).
- **Vérification 100 pas (patch actif)** : pente **+0,19 Mo/pas** tout compris, dont le saut du prune au pas 10100 (8 arêtes coupées, 667 nœuds enterrés) ; **hors prune, +0,02 Mo/pas** (RSS 444 → 445 Mo du pas 40 au pas 90), `uordblks` plat.

## Ce qui a été fait, dans l'ordre

### Voie 0 — lecture des lignes (le déblocage)
`sed`/`nl` sur le serveur (`mycelia/brain.py`, 1123 lignes) :
- **345** : `self._edge_keys.add(source * self.max_population + target)` — set d'ints, membres bornés par `max_edges` (20 000).
- **956** : `self.last_firing_nodes = sorted(int(i) for i in fired_nodes.nonzero()...)` — liste d'ints ≤ `n_nodes`, réaffectée à chaque forward.

Aucune structure non auditée qui s'accumule : le diagnostic écrit dans le code était « deux allocations bornées ».

### H3 — sonde par stdin SSH, sans transfert de fichier
Script injecté par `base64 -d` sur stdin SSH (aucun scp, aucun crochet IPv6 dans un chemin), exécuté par le venv du projet (`.venv/bin/python`, torch 2.14.0+cpu), 20 pas, tracemalloc à 25 trames, snapshots aux pas 5 et 20.

```
pas  1 | RSS 599 Mo | py_cur 79,13 Mo | edge_keys  20 | firing   7
pas  5 | RSS 859 Mo | py_cur 72,44 Mo | edge_keys  52 | firing  37
pas 20 | RSS 923 Mo | py_cur 72,48 Mo | edge_keys 172 | firing 140
DIFF pas5→pas20 : brain.py:345 +0,01 Mo (+120 objets) ; brain.py:956 +0,01 Mo (+129 objets)
```

Verdict : **tracemalloc voyait des sites d'allocation bornés, pas une rétention** — la croissance RSS (+2,7 Mo/pas dans ce run, snapshot tracemalloc compris) est hors Python.

### H4 — census des tenseurs vivants + mallinfo2
20 pas, census aux pas 1/5/10/15/20 :

```
pas  1 | RSS 448 | uordblks 188,6 | fordblks  22,9 | arena 211,5 | tenseurs 100 / 98,8 Mo
pas 20 | RSS 529 | uordblks 188,8 | fordblks 103,6 | arena 292,5 | tenseurs 117 / 98,9 Mo
pente RSS : 4,38 Mo/pas (4 threads) — 4,39 Mo/pas (OMP_NUM_THREADS=1)
```

Aucune rétention : les octets en usage et les tenseurs vivants sont constants ; **seul le libre dans l'arène grandit**.

### H4b — les deux leviers anti-fragmentation, mesurés
| Variante (20 pas) | Pente RSS | `uordblks` | `fordblks` |
|---|---|---|---|
| Baseline | **+4,30 Mo/pas** | 195,2 → 195,3 | 39,7 → 103,7 |
| `malloc_trim(0)` chaque pas | **+0,07 Mo/pas** | 195,2 → 195,3 | 31,7 → 95,8 |
| `MALLOC_MMAP_THRESHOLD_=128K` | +4,21 Mo/pas | 117,8 → 117,9 | 26,9 → 89,7 |

`MALLOC_MMAP_THRESHOLD_` déplace les allocations mais ne rend pas les pages : sans effet sur la pente. `malloc_trim` rend les pages libres au noyau : effet immédiat.

### Patch
```python
def trim_heap() -> None:
    """Give freed heap pages back to the OS (glibc only, no-op elsewhere)."""
    global _MALLOC_TRIM
    if _MALLOC_TRIM is None:
        try:
            _MALLOC_TRIM = ctypes.CDLL("libc.so.6").malloc_trim
        except (OSError, AttributeError):
            _MALLOC_TRIM = False
    if _MALLOC_TRIM:
        _MALLOC_TRIM(0)
```
Appelé dans `Trainer.step_once()`, juste après `_release_telemetry()` et **avant** la lecture de `process_memory()`, pour que la RAM rapportée soit celle que le pas suivant réutilise. Coût mesuré : ~+40 ms/pas à l'échelle 4.

### H5 — vérification 100 pas (critère du protocole : dérive ~0 ou plafonnée)
Patch transféré par stdin SSH, **md5 vérifié identique** poste/serveur (`05c67c0f59140013ebc2c26fbfe7f916`) ; sauvegarde serveur dans `/tmp/train.py.pre-h4n`.

```
pas  10 | RSS 445 | uordblks 195,2 | ne  92
pas  40 | RSS 444 | uordblks 188,9 | ne 332
pas  90 | RSS 445 | uordblks 189,4 | ne 732
pas 100 | RSS 462 | uordblks 207,5 | ne 804   ← prune : 8 arêtes coupées, 667 nœuds enterrés
pente RSS pas10→100 : +0,19 Mo/pas   (hors prune : +0,02 Mo/pas)
```

## Interprétation

Ce n'était pas une fuite : rien ne s'accumule (uordblks plat, tenseurs vivants constants, Python plat). C'était **l'allocateur glibc qui étendait son arène** à chaque pas et ne rendait jamais les pages libérées au noyau. Pourquoi H1 (session L) échouait : `MALLOC_TRIM_THRESHOLD_` ne déclenche qu'au sommet du tas, et les trous libres sont fragmentés au milieu ; `MALLOC_ARENA_MAX` ne change rien à une arène unique. Il fallait un **appel explicite** `malloc_trim(0)`.

## Limites (à ne pas confondre avec des acquis)

- **CONSTATÉ** sur 100 pas, échelle 4 (1000 nœuds), toy, batch 2, seq 16 — pas sur la config texte (batch 16, seq 128) ni sur 3 000 pas.
- Le prune du pas 10100 a produit un saut ponctuel de RSS (+18 Mo, cimetière de 667 nœuds) : c'est un événement, pas une pente.
- Le dashboard n'était pas actif pendant les mesures.
- GPU non vérifié cette session : `nvidia-smi` échoue (`NVML: driver/library version mismatch`).
- Swap non relevé (commande `awk` échouée dans le relevé de salle).
- Salle au moment du run : charge 1,71 ; RAM totale 7 350 Mo, disponible 5 875 Mo ; **0 conteneur actif**. **Identité réconciliée le 12 novembre** : `hostnamectl` donne l'hôte `IA` (Acer Aspire VN7-591G, i7-4710HQ, machine-id `76af723a…`), et **toutes les archives H→L mesurent 7 350 Mo au total**. Le « 16 Go » venait du brief initial, corrigé dès la session J (« 7 350 Mo au total et non 16 Go comme annoncé dans le brief »). Aucun changement de machine ; la mention de mon premier jet était l'erreur.

## État des trois prérequis

| Prérequis | État | Note |
|---|---|---|
| **1.1 Filaire** | **rouge** | `sudo bash ~/prepare-filaire.sh` toujours en attente (4e session) |
| **1.2 Dérive** | **vert (corrigé et mesuré)** | cause = fragmentation glibc ; patch `malloc_trim` ; 100 pas à +0,19 Mo/pas, hors prune +0,02 |
| **1.3 Run 3 000 pas** | **débloqué, non lancé** | dépendait de 1.2 ; reste la confirmation finale à l'échelle nominale |

Phases 2-4 du sprint français : toujours verrouillées par 1.1 (et par la confirmation 3 000 pas).
