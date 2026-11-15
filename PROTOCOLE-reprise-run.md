# PROTOCOLE — reprise du run B2 (10 020 nœuds)

Écrit le 15 novembre 2026, **avant** de savoir lequel des trois cas s'appliquera.
Seuil de référence : **0,002342**. Objectif : **3 000 pas**. Checkpoint : tous les **500 pas**.
Jamais de seuil déplacé après coup, jamais de reprise maquillée.

## Machine de relance

Le lanceur `scripts/run_twin_10020.py` lit `/proc/self/status` (`hwm_mb()`) : il est
**Linux uniquement**. Les checkpoints 10 020 nœuds (775 Mo) n'existent que sur le serveur
`192.168.50.53`. Le PC hôte Windows n'a aucun checkpoint twin/10020 et son disque C: est
plein (493 Mo libres). **Conclusion : B2 vit et revit sur le serveur, nulle part ailleurs.**

Interpréteur : prendre le venv qui contient réellement torch (`.venv-cu126`, Python 3.12),
pas `.venv` (Python 3.14, sans torch).

## Les trois cas

### Cas 1 — le run est vivant

```bash
ps aux | grep run_twin_10020
```

→ **Ne rien relancer.** Jamais deux runs sur le même génome (écrasement mutuel des
checkpoints). Consigner `{etat: vivant, pid, step_dernier_log}` et surveiller le log.
Le nouveau PID devient le **sacré** de la règle « ne jamais tuer le PID ».

### Cas 2 — mort + checkpoint de run présent

```bash
ls -la data/genome_twin10020_run.pt        # écrit tous les 500 pas
```

→ Reprise, **jamais de zéro** :

```bash
setsid .venv-cu126/bin/python -u scripts/run_twin_10020.py --mode run \
  --resume data/genome_twin10020_run.pt > data/mem_report/twin10020.log 2>&1 &
```

Vérifier dans les premières lignes : `REPRISE | ... | step X -> 3000` puis
`reprise à step X` (log de `Trainer.resume`). Si le log dit « initialisation » ou
`step 0`, **arrêter immédiatement** : la reprise a échoué.
Consigner : `{cause, checkpoint_reprise, pas_perdus, nouveau_PID}`.

### Cas 3 — mort + aucun checkpoint de run

Redémarrage à zéro **autorisé si et seulement si** :

1. le run mort n'a jamais atteint son premier checkpoint/500 (donc < 500 pas perdus) ;
2. le génome de base est vérifié intact — **hash de `genome_latest.pt` consigné
   avant** relance (775 851 218 octets, 14 nov 15:48 au dernier relevé) ;
3. l'incident est consigné : `{cause, pas_perdus: <500, redemarrage_zero_autorise_par: ce protocole}`.

Seuil 0,002342, 3 000 pas, checkpoint/500 : **inchangés**.

```bash
sha256sum data/genome_latest.pt            # consigner AVANT
setsid .venv-cu126/bin/python -u scripts/run_twin_10020.py --mode run \
  > data/mem_report/twin10020.log 2>&1 &
echo "PID: $!"                             # consigner — il devient le sacré
```

Le zéro n'est pas un maquillage quand il n'y a rien à reprendre : c'est un redémarrage
honnête, documenté. Le maquillage serait de prétendre reprendre alors qu'on a tout
relancé — ou l'inverse.

## État du mécanisme de reprise

- `--resume` câblé dans `scripts/run_twin_10020.py` (appel `Trainer.resume`, pas de zéro).
- Testé le 15 nov 2026 sur un cerveau de 64 nœuds : save step 7 → resume step 7,
  log `reprise à step 7 | pool 64 | arêtes 68 | cimetière 0`. **VERT.**
- Test pleine échelle (smoke 10 020 avec `--resume`) **non exécuté** : exige le
  checkpoint de 775 Mo, présent uniquement sur le serveur, et le disque local est plein.

---

# INCIDENT — nuit du 14 au 15 novembre 2026

## Cause

`taskkill /F /IM python.exe` (commande globale, sans cible) lancée sur le PC hôte
Windows pour arrêter le chat local. Elle a tué **tous** les Python de la machine.

## Ce qui a réellement été tué

Sur le PC hôte : les instances Python locales, dont deux serveurs Flask périmés qui
occupaient encore `127.0.0.1:8080` (PID 23368 et 7904, tués ensuite proprement par PID).
**B2 n'était pas sur cette machine** : le lanceur est Linux-only et les checkpoints
10 020 nœuds n'existent que sur le serveur. Le run n'a donc **pas** été tué par ce
taskkill ; son état réel reste à vérifier dès que SSH répond.

## Enquête — preuves

1. `scripts/run_twin_10020.py` ouvre `/proc/self/status` → Linux uniquement.
2. Checkpoints 10 020 nœuds sur le serveur : `genome_twin10020_smoke.pt` (775 852 137 o,
   14 nov 15:47) et `genome_latest.pt` (775 851 218 o, 14 nov 15:48). En local : aucun
   fichier twin/10020 ; le plus gros `genome_latest.pt` fait 49 426 941 o (7 nov).
3. `RAPPORT-session-w4.md` : lancement consigné à 15:50.
4. `data/genome_twin10020_run.pt` **absent** au dernier listing réussi → le run n'a jamais
   atteint 500 pas (ou n'a jamais démarré) ; `data/mem_report/twin10020.json` absent aussi.
5. Deux venvs sur le serveur : `.venv` (Python 3.14, **sans** torch) et `.venv-cu126`
   (Python 3.12, **avec** torch). Le `ModuleNotFoundError: No module named 'torch'` venait
   du mauvais interpréteur — pas d'une installation manquante.

## Effets de bord

- Un `pip install torch --index-url .../nightly/cpu` a été lancé sur le `.venv` du serveur
  avant l'audit des venvs. **Résultat non confirmé** (SSH fermé avant vérification).
  À contrôler et, si présent, désinstaller : un torch inadapté dans le mauvais venv est
  une bombe à retardement (un script qui prendrait `.venv` par réflexe importerait un
  faux torch au lieu d'échouer bruyamment).
- SSH est devenu injoignable en pré-authentification (`Connection closed` / `Connection
  reset`) après des tentatives répétées : signature fail2ban / MaxStartups. Martèlement
  arrêté. Le bannissement se lève seul (10 min à 1 h) ou via console.
- Clé SSH **non installée** (`ssh-copy-id` a échoué, puis le serveur a coupé).
  Le mot de passe a transité en clair deux fois → **à changer**, et la clé à installer
  en une seule session manuelle.

## Interdictions ajoutées (permanentes)

- **`taskkill python.exe` / tout kill global : INTERDIT.** Toujours `taskkill /PID <pid>`
  après avoir vérifié que ce n'est pas le run.
- **Mot de passe en clair dans un fichier, script, log ou consignation : INTERDIT.**
  Clé SSH obligatoire, secrets hors du projet.
- **Toute commande globale sans cible vérifiée : INTERDITE.** Chaque opération porte son
  PID, son chemin, sa cible.

## Coût réel de l'incident

Au pire : moins de 500 pas d'entraînement (le run venait de démarrer, perte 2,78 au pas
100). Le trésor — le génome corrigé 10 020/10 020 de 775 Mo — est intact. Le coût d'un
redémarrage à zéro est de quelques minutes de calcul. **Presque rien n'est perdu.**

---

# DÉCISION — le seuil 0,002342 (écrite le 15 novembre 2026, avant tout run)

Constat d'audit, vérifiable dans les artefacts déjà présents :

- `data/mem_report/session-r.json` → `topk.mesures` donne, pour `fraction: null`
  (aucun routage, tous les nœuds éveillés), `perte: 0.002342`. **C'est de là que
  vient le seuil.** Ce n'est pas la perte d'un jumeau routé : c'est la perte
  **dense** d'un génome de **1 129 nœuds** (même fichier, `activite_par_hop`).
- Le même fichier mesure le coût du routage sur ce génome : `0.5 → 1.594`
  (×680), `0.25 → 3.091` (×1 320), `0.10 → 4.325` (**×1 847**), `0.01 → 6.162`.
- Le smoke B2 corrigé (10 % éveillé, 10 020 nœuds, 100 pas) : `final_loss: 2.7801`,
  contre `4.4597` avant correction (`data/mem_report/session-w4.json`,
  `b2_correction`). Le routage coûte, mais la correction a divisé la perte du
  pas 100 par 1,6.

## Ce que cela veut dire

L'écart « 2,7801 contre 0,002342 » **n'est pas une preuve de panne** : le seuil
compare une perte sans routage à une perte avec 10 % de nœuds éveillés. Sur un
génome comparable, le routage à 10 % coûtait à lui seul un facteur ~1 850. Un
tel écart est donc **attendu par construction** : le seuil est **mal apparié**.

## La règle, dans cet ordre — aucune étape sautée

1. **Le seuil de l'étage 1 reste 0,002342.** Il a été consigné avant le run ;
   le déplacer maintenant serait exactement le maquillage interdit.
2. **Le run 3 000 pas se lance tel quel** (protocole ci-dessus, `--resume`
   compris) et **écrit son verdict**, quel qu'il soit.
3. **Si la perte finale est > 0,002342** — c'est le cas attendu — alors l'étage 1
   est **échoué tel que consigné**, et cela ne dit rien de la piste B2. On ouvre
   alors un **étage 1bis apparié**, dont le critère est écrit et daté **avant**
   son propre run :
   - soit comparaison **à fraction égale** (10 % routé contre 10 % routé) ;
   - soit comparaison **à pas égal** avec la référence dense (perte aux pas
     100 / 500 / 1 000 / 3 000), et non une perte finale contre une finale.
4. **Si la perte ne descend pas** (stagnation sur ≥ 500 pas) ou si le cimetière
   se remplit encore au pas 300, c'est l'hypothèse **structurelle** : on
   l'audite avant toute relance, et on ne relance pas 3 000 pas « pour voir ».

## Ce qui manque pour trancher proprement, et où le prendre

La référence dense n'existe dans le dépôt que par sa **valeur finale**
(0,002342) : sa courbe pas-à-pas n'y est pas. La comparaison à pas égal ne peut
donc pas être faite depuis le PC. Elle doit être produite **sur le serveur, dans
la même session** que le run B2 (même génome, mêmes 3 000 pas), sinon l'étage
1bis serait jugé sur deux courbes mesurées à des moments différents.

**Interdit maintenu** : relancer en espérant. Un run sans critère apparié est un
run qui ne peut pas conclure.
