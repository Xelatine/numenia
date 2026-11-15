# Session L — ouverture du sprint « français » : état des trois prérequis, caractérisation de la dérive mémoire, protocole H1/H2

## 1.1 Filaire — toujours bloqué sur les droits root

Même état que précédemment : `enp8s0` en `169.254.184.83/16`, netplan jamais appliqué, route sort par `wlp7s0`. Toujours besoin de `sudo bash ~/prepare-filaire.sh`.

## 1.2 Dérive mémoire — caractérisée, H1 et H2 toutes deux réfutées par la mesure

### Mesures de base (session L, suite)
Courbe RSS à 1 000 nœuds, batch 2, seq 16, CPU :
- Pas 1 → 100 : **631 → 1 433 Mo** (+8,0 Mo/pas)
- Séquence 8 → pente ~2,4 Mo/pas (suivant `batch × seq`)
- Aucun plateau sur 100 pas
- Structures visibles : cimetière 0, requêtes 0, historique 0

### Protocole H1 — allocateur GLIBC
`MALLOC_ARENA_MAX=2 MALLOC_TRIM_THRESHOLD_=16M` sur 10 pas :
- RSS **identique** sans les variables d'environnement (841 Mo pas 10) et avec
- L'hypothèse que la dérive vient du glibc malloc arena est **réfutée** — on ne peut
  pas la "trimmer" away avec des variables d'environnement.

### Protocole H2 — graphe autograd retient activations
`torch.no_grad()` wrapper autour de `step_once()` sur 10 pas :
- RSS **identique** sans et avec le wrapper (841 Mo pas 10)
- L'hypothèse que la dérive vient de la rétention des graphes computationnels
  autograd est **réfutée** — désactiver les gradients n'a aucun effet sur la pente.

### Ce qui a été éliminé (par la mesure, pas par l'intuition)
1. **Listes Python visibles qui grossissent** : `cimetière` 0, `pending_node_requests` 0,
   `history` 0, nœuds et arêtes stables. Tester en vidant `last_node_io`,
   `last_energy_by_hop`, `last_out_acc`, `_last_out` après chaque pas → écart nul
   (840→886 sans vidage vs 841→887 avec, soit ~1 Mo sur 20 pas).
2. **Allocateur GLIBC** : variables d'environnement `MALLOC_ARENA_MAX` / `MALLOC_TRIM_THRESHOLD_`
   n'affectent pas la pente RSS.
3. **Graphe autograd / suivi des gradients** : `torch.no_grad()` autour du forward
   n'affecte pas la pente RSS.

### Ce qui reste à investiguer (H3 — "rétention invisible")
Le protocole M prévoit H3 (`gc.get_referrers` / `objgraph` sur les activations
capturées pendant un forward), mais les trois premières pistes naturelles ont
été écartées par mesure. La cause reste localisée mais non identifiée ; elle
pourrait être :
- Une subtilité d'interaction entre l'allocation CPU/CUDA non visible en RSS
- Un motif de rétention dans les hooks `register_forward_post_hook` ou
  `saved_tensors` non détecté par nos comptages
- Quelque chose dans le cycle de vie du Trainer que nous n'avons pas encore
  inspecté avec `gc.get_referrers`

**Critère de sortie du sprint phase 1** : dérive ~0 Mo/pas OU plafonnée.
Pour l'instant la dérive est ~8 Mo/pas et ne plafonne pas → les trois
prérequis restent rouges, phases 2-4 verrouillées.

## 1.3 Run 3 000 pas — non lancé, dépendant de la correction de 1.2

Extrapolation linéaire à 3 000 pas ≈ 19 Go [estimé] → impossible avec ~6 Go dispo.

## État mis à jour

| Prérequis | État | Note |
|---|---|---|
| 1.1 Filaire | rouge | `sudo bash ~/prepare-filaire.sh` attendu |
| 1.2 Dérive | rouge (caractérisée, H1/H2 réfutées) | Cause toujours localisée mais non identifiée |
| 1.3 Run 3 000 pas | non lancé | Dépend de 1.2 |
| Phases 2-4 | non entamées | Verrouillées par la règle du sprint |

## Livrables
`RAPPORT-session-l.md` (mis à jour), `data/mem_report/session-l.json` (mis à jour),
`scripts/drift_probe.py` (rétabli à son état d'origine), commit **4336a10** poussé
sur `origin/main`.

## Prochaine action du protocole
Si l'utilisateur souhaite continuer l'investigation : lancer H3 (`gc.get_referrers`
sur des tenseurs d'activation capturés pendant `step_once()`, suivi de la chaîne
de références pour nommer le coupable racine). C'est le test "chirurgical" qui
nommerait littéralement l'objet qui retient. Sans objgraph installé, on peut
tenter une approche manuelle : capturer un tenseur juste après `step_once()`, puis
utiliser `gc.get_referrers(t)` pour voir qui il référence, itérer jusqu'à atteindre
une racine connue (module, classe Trainer, etc.).