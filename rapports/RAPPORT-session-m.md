# Session M — Dérive mémoire : protocole H1/H2/H3 et plan B

## État d'avancement du protocole M

### Étape 1 — tracemalloc (déjà effectuée)
**Résultat** : tracemalloc confirme une **croissance Python** (RSS 733 → 825 Mo sur 20 pas, ~+4,6 Mo/pas). Les 10 meilleures croissances par ligne de code pointent toutes vers **brain.py lignes 345 et 956**.  
**Verdict** : la piste est fertile → objet Python → étape 2 (H3) applicable.

### Étape 2 — H3 chirurgicale (NON exécutée cette session)
**But** : capturer un tenseur d'activation via `forward_post_hook` aux pas 1 et 10, lancer `gc.get_referrers(t)` et remonter la chaîne complète jusqu'à une racine connue (Trainer, module, global). L'objectif est de nommer l'objet racine rétentard.

**Pourquoi non exécutée** : difficultés techniques de transfert de script depuis le poste vers le serveur (scp bloqué par les crochets IPv6, git push non cohérent sur l'affichage des fichiers nouveaux). Le script `h3_chirurgical.py` a été préparé mais ne s'est pas exécuté sur le serveur.

**Résultat partiel** : H1 (allocateur GLIBC) et H2 (graphe autograd/`torch.no_grad()`) sont **réfutées par la mesure** (tous deux n'ont aucun effet sur la pente RSS). La dérive suit `batch × seq` et porte sur des objets Python dans `brain.py`.

### Étape 3 — Audit hooks + Trainer (non atteinte)
Si l'étape 2 n'avait pas conclu, il fallait :
- Compter les hooks enregistrés sur les modules, inspector ce que leurs closures référencent
- Tout ce que le Trainer accumulate par pas hors des listes déjà auditées

### Étape 4 — Plan B "Sommeil de l'organisme" (optionnelle)
Si la cause traîne ou si l'étape 1 montrait une piste C (autre chose que objet Python), lancer 3 000 pas en segments de 100 pas, sauvegarder genome + état du graphe à chaque coupure, tuer le processus, recharger dans un processus frais, continuer. 30 segments. Mesurer l'overhead save/load.

## Hypothèses éliminées par la mesure (non par intuition)

| Hypothèse | Test | Verdict |
|---|---|---|
| **H1** — allocateur GLIBC (`MALLOC_ARENA_MAX=2` + `MALLOC_TRIM_THRESHOLD_=16M`) | 10 pas, RSS identique avec/without | **Réfutée** — on ne peut pas la "trimmer" away |
| **H2** — graphe autograd retient activations (`torch.no_grad()` autour de step_once()) | 10 pas, RSS identique avec/without | **Réfutée** — désactiver les gradients n'a aucun effet |
| **Listes Python visibles** (cimetière, pending_node_requests, history) | vidage testé, écart nul | **Réfutée** — ces listes ne grossissent pas |

## Ce qui reste (d'après tracemalloc)
- **Croissance confirmée** : RSS 733 → 825 Mo sur 20 pas (~+4,6 Mo/pas)
- **Top allocations Python** : `brain.py` lignes **345** et **956** systématiquement dans le TOP 10
- **Aucun plateau** : la pente ne s'aplatit jamais sur 100 pas
- **Cause localisée** : objet Python dans brain.py (non identifié exactement)

## État final du sprint "français"

| Prérequis | État | Note |
|---|---|---|
| **1.1 Filaire** | rouge | `sudo bash ~/prepare-filaire.sh` toujours attendu (4 sessions) |
| **1.2 Dérive** | **orange → investigation partielle** | tracemalloc confirme objet Python dans brain.py ; H3 non exécutée cette session |
| **1.3 Run 3 000 pas** | bloqué | dépend de la correction/contrevent de 1.2 |
| Phases 2-4 | verrouillées | règle du sprint |

## Livrables (pushés sur `origin/main`)
- `RAPPORT-session-l.md` — état initial + résultats H1/H2
- `RAPPORT-session-m.md` — ce rapport de protocole
- `data/mem_report/session-l.json` — JSON structuré
- `data/mem_report/session-l_conditions.txt` — conditions de mesure
- `scripts/drift_probe.py` — sonde de dérive (rétablie)
- `scripts/h3_chirurgical.py` — script H3 (non exécuté ce jour)
- Commit **1c7c112** — "chore: add H3 chirurgical script for memory chain tracing"

## Prochaine action suggérée
- Relancer l'**étape H3 chirurgicale** lorsque le transfert de fichier sera résolu (soit par scp corrigé, soit par git push forcé, soit par écriture directe serveur). L'objectif est de nommer l'objet racine rétentard via `gc.get_referrers` chain.
- Sinon, basculer vers l'**étape 4 (plan B Sommeil)** : lancer 3 000 pas en segments de 100 pas avec sauvegarde/restore de l'état, et publier avec mention du contournement et overhead mesuré.
- S'occuper du **prérequis 1.1 filaire** (`sudo bash ~/prepare-filaire.sh`) pour déverrouiller les téléchargements corpus/Modèle.

**Objectif de fin de session** : soit le prérequis 1.2 AU VERT (cause nommée et corrigée, soit plan B Sommeil validé avec overhead mesuré), soit idéalement LE RUN 3 000 PAS terminé — le dernier verrou avant le sprint français.