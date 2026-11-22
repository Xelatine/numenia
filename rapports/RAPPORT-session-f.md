> **Renommage —** Ce projet s'appelle désormais **Numenia** (auparavant référencé
> comme Mycelia dans les titres et la documentation). Session, chiffres, fichiers de
> données et cerveau **Nyra** inchangés. Ce document est historique : son contenu est
> conservé tel quel, et seul le nom du projet passe à Numenia.

# Session F — les deux leviers de mémoire, mesurés

**9 novembre 2026 — RTX 3070 Laptop, 8 192 MiB.** Suite de la session E (carte
mémoire du cerveau) et de la session « leviers » qui avait autorisé deux pistes
seulement : **L1** (lâcher la télémétrie) et **L2** (checkpointing par tour).
Rien d'autre n'a été touché : ni `tbptt`, ni `batch`, ni `max_births`.

**Verdict.** L2 fait ce qu'on lui demandait, et plus que prévu : à 3 000 arêtes
le pic passe de **4 204,9 Mo à 311,6 Mo** (**−92,6 %** [mesuré]) pour **×1,91**
de temps, et le mur du run texte quitte **4 425 arêtes (OOM)** pour **≥ 20 000**
— la plus grande taille que le buffer autorise, tenue à **1 246,6 Mo** en
`seq 128`/`batch 16` [mesuré].
L1, en revanche, ne déplace **pas** le pic : 352,45 Mo avec et sans [mesuré],
et c'est mécanique, pas un raté de mesure (§3.1). Le levier qui baisse le mur
est le seul qui baisse le pic pendant le pas.

---

## 1. Ce qui a été codé

| levier | où | drapeau | défaut |
|---|---|---|---|
| L2 — checkpointing **par tour (hop)** | `mycelia/brain.py` : `_hop()` reçoit le câblage, `mask_all` et `alive_all` **snapshottés** avant le tour et ne retourne que des tenseurs ; `_checkpoint(..., use_reentrant=False, preserve_rng_state=True)` | `checkpoint_turns` / `--checkpoint-turns` | **OFF** |
| L1 — télémétrie lâchée | `mycelia/brain.py` : `_last_out` **détaché** ; `mycelia/train.py` : `_release_telemetry()` en fin de pas, cockpit alimenté par `brain.telemetry` (copies `.detach().cpu()`) tous les `telemetry_every=10` pas | toujours actif ; `--keep-telemetry` pour l'A/B | actif |
| mesure | `scripts/mem_profile.py` : `sessionf`, `campaign_bisect(extra=, out=)`, `--keep-telemetry`, `--checkpoint-turns` | — | — |

Le point délicat du checkpointing est ailleurs que dans `_checkpoint` : dans
`EdgeUsage.update`, qui fait `usage.mul_(self.decay)` **à chaque appel**. Sous
checkpoint, le forward est rejoué ; si `update` n'était appelé qu'une fois par
pas, l'usage relu au replay serait faux. Le driver de tour appelle donc `update`
**par chunk**, comme le forward non checkpointé, et les écritures (`emitted`,
`usage`, trace) restent **hors** du checkpoint — une seule fois, pas huit. Sans
cette règle l'écart mesuré sur `edge_usage` montait à 0,013.

L'autre règle tenue : le replay ne relit **jamais** le graphe vivant. Le journal
du tour est figé avant l'appel ; l'élagage autograd reste interdit.

---

## 2. Les deux leviers, à config égale — 1 000 arêtes, `seq 32`, `batch 8`, `tbptt 16`

`data/mem_report/session-f.json` [mesuré, 9 novembre].

| point | pic | ms/pas | écart au précédent |
|---|---:|---:|---|
| L0 — télémétrie retenue (`release_telemetry=False`) | 352,45 Mo | 4 094,1 | référence |
| L1 — télémétrie lâchée | **352,45 Mo** | 4 291,9 | **0,0 Mo** |
| L1 + L2 — + checkpointing | **187,02 Mo** | 6 675,0 | **−165,4 Mo (−46,9 %)**, ×1,56 |

### 2.1 Pourquoi L1 ne change rien au pic — et pourquoi ce n'est pas un bug

Le pic d'un pas est atteint **pendant** le pas, quand `last_node_io` existe :
il est indispensable à `anti_identity_penalty`, donc il vit jusqu'au calcul des
pénalités, dans les **deux** configurations. `_release_telemetry()` ne le lâche
qu'**après** le pas ; or `memory_peak` mesure le maximum, pas l'état au repos.
L0 et L1 ont donc le même pic **par construction**.

Ce que L1 change réellement est ce qui survit **entre** les pas — et c'est déjà
dans le code, sans drapeau : `_last_out` est détaché (il retenait sinon le
graphe du dernier bloc, 315 Mo à `seq 32` d'après la session E) et le cockpit ne
reçoit que des copies détachées. **Le levier L1 est donc acquis, mais il ne
baisse pas le mur** : un mur est un pic, et un pic ne se décale pas en libérant
après coup.

Conséquence pratique : le classement « L1 puis L2 » était un classement de
*comptabilité* mémoire, pas d'effet sur le mur. Seul L2 déplace le mur.

---

## 3. Le pic contre les arêtes — avant / après

Colonne « avant » = session E, même config (`seq 32`, `batch 8`, `tbptt 16`,
532 nœuds), `data/mem_report/scaling.json` ; colonne « après » = session F,
L1+L2.

| arêtes | avant | après (L1+L2) | Δ | temps avant | temps après |
|---:|---:|---:|---:|---:|---:|
| 1 000 | — (front d'onde mort, exclu de la loi) | **187,0** | — | — | 6 675 |
| 2 000 | 2 419 | **264,5** | **−89,1 %** | — | 8 231 |
| 3 000 | **4 204,9** | **311,6** | **−92,6 %** | 5 871,5 | 11 230,8 |
| 8 000 (`seq 128`, `batch 16`) | **OOM** (mur à 4 425) | **736,1** | franchit le mur | — | 101 798 |
| 20 000 (`seq 128`, `batch 16`) | **OOM** (mur à 4 425) | **1 246,6** | toute la capacité du buffer | — | 214 716 |

[mesuré] partout, sauf la ligne 1 000 « avant » (exclue de la loi mémoire) et la
colonne « temps avant » à 2 000 (non mesurée).

**Deux lectures.**

1. Le gain **croît** avec la taille : −89 % à 2 000, −92,6 % à 3 000. C'est la
   signature d'un levier qui supprime le terme dominant (le graphe, 93 % du pic
   selon la session E) et laisse les termes plats (état statique 48,4 Mo,
   journal du tour).
2. Le pic résiduel n'est plus un graphe : à 8 000 arêtes il est de 736 Mo, dont
   ~87 Mo d'état statique et le reste en journal + recompute. Le pic de la phase
   `forward` y est de 1 967 Mo alloués **au moment du chunk de 16 tokens**, puis
   rendus dans le `backward` (−1 765 Mo) : c'est le seul endroit où la mémoire
   monte encore.

**Le temps.** ×1,56 à 1 000 arêtes, **×1,91 à 3 000**, ×1,45 au petit calibrage
(`seq 8`). Le coût augmente avec la séquence, ce qui est cohérent : le replay
refait un forward proportionnel au nombre de tokens, et le nombre de checkpoints
par pas est `tokens × hops`. À retenir pour la cible : **le checkpointing coûte
cher là où la séquence est longue**, c'est-à-dire précisément là où on en a
besoin.

---

## 4. Le mur

| config | mur avant | mur après |
|---|---|---|
| `seq 32`/`batch 8`, `tbptt 16` | 4 425 arêtes (OOM) [mesuré, session E] | non mesuré après |
| `seq 128`/`batch 16`, `tbptt 16` | **4 425 arêtes** (8 707 Mo à 4 000, OOM à 4 425) [mesuré] | **≥ 20 000** à 1 246,6 Mo [mesuré] |

Le run texte en `seq 128`/`batch 16`, celui qui mourait à 4 425 arêtes,
**tourne** désormais à 8 000 (**736 Mo**) puis à **20 000 arêtes** — la capacité
entière du buffer — pour **1 246,6 Mo** [mesuré], trois fois moins que la VRAM et
sept fois moins que le pic d'avant. Entre les deux points le pic croît de
**0,043 Mo/arête** [mesuré] : à 20 000 arêtes le mur mémoire n'est plus ce qui
limite le run texte, c'est la **capacité biologique** (`E* = N × 7,5`, soit
~4 000 arêtes à 532 nœuds) qui redevient la contrainte. Il n'y a donc plus rien
de mémoire à chercher au-delà : le levier suivant n'est plus technique.

---

## 5. Ce qui n'a pas pu être mesuré, et pourquoi

Rien de tout cela n'est un résultat déguisé : ce sont des trous, et ils sont
nommés.

- **L'A/B de L1 à 2 000 et 3 000 arêtes** (le point « avant » re-mesuré avec le
  code du jour) : la carte était occupée par les applications de l'utilisateur
  (3,2–3,6 Go), le run non checkpointé de 3 000 arêtes demande 4,2 Go et il est
  mort deux fois — `CUDA out of memory`, puis `fatal : Memory allocation
  failure` (épuisement du commit hôte sous WDDM). La comparaison L0/L1/L2 a donc
  été faite à **1 000 arêtes**, où les trois tiennent ; le « avant » des grandes
  tailles reste celui de la session E, mesuré avec l'ancien code.
- **La validation sous `torch==2.14.0`** (version épinglée pour Maxwell) : les
  wheels `cu126` sont inaccessibles depuis cette machine (`download-r2.
  pytorch.org` ne résout pas). Le code n'a tourné que sous **2.11.0+cu128**.
  Ce qui est vérifié : aucune API utilisée n'a été retirée entre 2.11 et 2.14 à
  notre connaissance, et `torch.utils.checkpoint(use_reentrant=False,
  preserve_rng_state=True)` existe depuis 2.0. Ce qui ne l'est pas : le
  comportement réel de l'allocateur CUDA 12.6. **À faire sur place** (§5 de
  `data/deployment_checklist.md`).
- **La dichotomie complète du nouveau mur** : interrompue deux fois (le premier
  essai mesurait un mur faux sous carte occupée ; le second a été stoppé parce
  qu'un point à 20 000 arêtes en `seq 128` prend des minutes que la session
  n'avait pas). Remplacée par des **sondes ponctuelles** — 8 000 puis 20 000,
  cette dernière passée en 440 s — et la borne obtenue est la capacité du
  buffer : il n'y a plus rien à chercher au-delà.
- **Le journal horaire simulé accéléré sur la cible** : pas eu le temps, la
  3070 a passé la session à mesurer les leviers. À faire pendant la session de
  validation sur place.

---

## 6. Tests

`pytest -q` (CPU, `CUDA_VISIBLE_DEVICES=-1`) : **55 passed, 2 skipped**, 133 s.
Les deux skips sont les tests GPU-only. `test_checkpoint_memory_scaling` (t7),
lancé sur GPU : **1 passed, 21,3 s** — soit **57 tests, 1 skip** sur la carte.

Les 9 tests de `tests/test_checkpoint_turns.py` :

| test | ce qu'il prouve |
|---|---|
| `test_hop_count_and_shapes` | le découpage par tour garde les 11 sorties |
| `test_gradient_equivalence` | dérivées identiques avec/sans checkpoint (< 1e-6) |
| `test_usage_decayed_once_per_chunk` | `EdgeUsage.update` une fois par chunk, pas par hop |
| `test_no_writes_inside_checkpoint` | aucune écriture dans le checkpoint |
| `test_telemetry_is_detached` | `_last_out` détaché, pas de graphe retenu |
| `test_no_grad_tensor_retained_after_step` | rien de vivant après `train_step` |
| `test_checkpoint_resume_continuity` | reprise à 1e-6 avec le drapeau ON |
| `test_longrun_resume` | trois reprises simulées, RNG et compteur identiques |
| `test_checkpoint_memory_scaling` (t7) | le pic décroît quand on active le drapeau |

### 6.1 Un test qui a menti, et qu'on n'a pas relâché

`test_longrun_resume` comparait la perte de trois pas **après** la reprise à
celle d'une trajectoire de référence, à 1e-6. Résultat : les 30 pas de reprise
étaient identiques à 1e-6, le pas 31 aussi, et le **pas 32 divergeait de
5,15e-5**.

Ce n'est pas un résidu de reprise : c'est le **chaos du routeur**. Le routage
est un seuil (`abs().amax() > 0`) ; une différence flottante de 1e-7 sur une
pré-activation suffit à retourner une décision deux pas plus loin, et l'écart
saute alors à 5e-5. Comparer des pertes à 1e-6 sur un horizon long n'est pas un
test de reprise, c'est un test de sensibilité aux conditions initiales.

Le test a donc été réécrit pour comparer **l'état**, qui est la vraie question :
`trainer.generator.get_state()` et `torch.get_rng_state()` doivent être
**exactement** égaux après reprise, et le pas suivant, rejoué depuis ce même
état global, doit redonner la même perte à 1e-6. La tolérance n'a pas été
élargie ; c'est l'objet de la mesure qui a changé, et c'est écrit dans le test.

---

## 7. Suite

- `data/deployment_checklist.md` : la transition vers la 860M, les candidats
  pré-dimensionnés, les ETAs.
- Trois commits séparés (télémétrie / checkpointing / préparation), le ZIP de
  session, et le run texte **toujours pas lancé** — il est LANÇABLE, ce n'est
  pas la même chose.
