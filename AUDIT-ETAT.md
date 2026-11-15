# AUDIT-ETAT — MYCELIA / Nyra

Audit réalisé le 15 novembre 2026, **en lecture seule**, depuis le PC principal Windows
(`G:\mycelia`, branche `main`, HEAD `e029d4c`).
Aucun fichier de code n'a été modifié, déplacé ou supprimé. Aucune connexion au serveur.
Aucun entraînement lancé. Ce fichier est le seul ajout au dépôt.

**Méthode** : lecture du code, `git log` / `git diff`, grep ciblés, et deux essais
d'exécution autorisés — `python scripts/run_twin_10020.py --help` et un smoke du chat
(démarrage, 1 GET, 1 POST, arrêt par PID ciblé `taskkill //PID`). Suite pytest
**collectée** (`--collect-only`), pas exécutée.

---

## 1. SCORE GLOBAL

| Verdict | Nombre |
|---|---:|
| ✅ VÉRIFIÉ | **12** |
| ⚠️ PARTIEL | **5** |
| ❌ FAUX / ABSENT | **11** |
| ❓ INVÉRIFIABLE depuis ce PC | **0** dans A→H — mais **7 vérifications serveur** listées au §6 |
| **Total affirmations auditées (A→H)** | **28** |

Trois affirmations sont vraies dans les fichiers de travail **mais pas commitées** :
`git diff --stat` montre `mycelia/brain.py` (9 l.), `mycelia/lifecycle.py` (6 l.),
`scripts/run_twin_10020.py` (32 l.) modifiés. **Le `--resume` de B2 n'existe que dans
l'arbre de travail.**

> **MISE À JOUR — même journée, après correctifs.** Cet audit reste l'instantané du
> 15/11 *avant* travaux. Ce qui a été corrigé depuis, et vérifié : **C5** (le
> `POST /chat` répond **200** et écrit son journal), **D4** (`awake.flag` expire à
> 300 s et disparaît à l'arrêt propre), `sshpass` retiré de `.server/server.bat`
> (relais vers `server-key.bat`, connexion par clé), README régénéré (90 tests
> collectés), seuil 0,002342 documenté dans `PROTOCOLE-reprise-run.md`, et tout
> l'arbre commité (`1d72cb4`). Le reste des ❌ est **inchangé** : `self.json`,
> Métabolisme, DPO, Garde-manger et prompt maître n'ont toujours pas une ligne de
> code.

---

## 2. TABLEAU DES AFFIRMATIONS

### A. LE CERVEAU

| # | Affirmation | Verdict | Preuve |
|---|---|---|---|
| A1 | Architecture « cerveau routé » définie | ✅ | `mycelia/brain.py:71` `class Brain(nn.Module)` ; `n_nodes` param. `:76` ; `route_fraction` `:112` et validé `:219-222` ; routeur (q/k + biais) `:261-266` ; `awake_seen` `:230` ; `last_awake_nodes` `:224`. Cible fixée dans le lanceur : `scripts/run_twin_10020.py:44` `NODES = 10_020`, `FRACTION = 0.10` `:45` |
| A2 | Cible « 10 020 / 10 020 nœuds éveillés » | ⚠️ | La **couverture** est mesurée et vraie : `data/mem_report/smoke_twin10020.json` → `nodes_ever_awake: 10020`, `never_awake: 0`, `awake_last: 1008`, `cemetery: 0`, `routing_ok: true` (smoke 100 pas). **Mais** : la perte finale du smoke est **2,7801** contre un seuil de **0,002342**, et `data/mem_report/twin10020.json` (résultat du run 3 000 pas) **n'existe pas**. La cible de nœuds est atteinte ; la cible de perte ne l'est pas, et le run qui devait la trancher n'a produit aucun artefact |
| A3 | Corrections post-effondrement B2 dans le code | ✅ (avec réserve) | Commit `e029d4c` : « le run B2 s'effondrait au pas 300 : 9 558 nœuds enterrés… Le token est désormais broadcasté aux éveillés, le readout moyenné, et les nœuds routés ne meurent plus par isolement ». Code : broadcast `mycelia/brain.py:934-943` ; readout = moyenne des experts `:225` et `:530` ; garde d'isolation `mycelia/lifecycle.py:147` (`if getattr(brain, "route_fraction", None) is not None`). **Réserve** : 3 fichiers restent modifiés non commités (dont le `--resume`) |
| A4 | `Trainer.resume()` existe + log « reprise à step X » | ✅ | `mycelia/train.py:787` `def resume(...)` (classmethod) ; restaure cerveau, optimiseur par nom, croissance, historique, RNG `:807-818` ; log exact `mycelia/train.py:818-820` : `f"reprise à step {trainer.step} | pool … | arêtes … | cimetière …"` |
| A5 | `--resume` câblé dans `run_twin_10020.py` | ✅ | `scripts/run_twin_10020.py:78` `def resume_trainer(...)` → `Trainer.resume(...)` `:85` ; argument `:204` ; branchement `:207` `resume_trainer(args.resume) if args.resume else build_trainer(...)` ; log `REPRISE \| … \| step X -> N` `:209`. **Test réel** : `python scripts/run_twin_10020.py --help` → `--resume RESUME  checkpoint de run à reprendre (jamais de zéro si présent)` ✅ |

### B. LE NOM ET L'IDENTITÉ

| # | Affirmation | Verdict | Preuve |
|---|---|---|---|
| B1 | `self.json` existe et contient « Nyra » | ❌ | Aucun fichier `self.json` sur le PC (`find . -name "self*.json"` → vide) ; `data/genome_myc/` **n'existe pas**. Le code sait l'écrire (`mycelia/identity.py:94` `identity.save(Path(genome_dir) / "self.json")`) mais rien ne l'a jamais appelé sur cette machine. Le nom « Nyra » n'apparaît dans **aucun** `self.json` — il n'y en a pas |
| B2 | `self.json` est lu à chaque éveil | ❌ | Un seul lecteur dans tout le dépôt : `scripts/chat.py:123` `Identity.load(path)` (chat **terminal**, non lancé et sans rapport avec `webchat.py`). `scripts/webchat.py` ne contient **aucune** référence à `self.json` / `Identity`. Le nom affiché est une constante : `scripts/webchat.py:2`, `:153`, `:203`, `:220`, `:222` |
| B3 | Étiquette affichée `[nyra]` dans `webchat.py` (attendu ≥ 2) | ✅ | `grep -c -i "nyra" scripts/webchat.py` → **11**. Étiquette explicite : `scripts/webchat.py:142-148` `label = "[nyra]"` (deux branches) |
| B4 | Restes de `[mycelia]` dans les affichages | ✅ (aucun reste) | `grep -rn "\[mycelia\]"` → **0** occurrence dans tout le dépôt. `mycelia` n'apparaît que comme **nom de paquet Python** (`from mycelia.brain import Brain`, etc.) — usage légitime, pas une étiquette visible |

### C. LE CHAT

| # | Affirmation | Verdict | Preuve |
|---|---|---|---|
| C1 | `webchat.py` : lignes + endpoints | ✅ | **284 lignes**. 5 routes : `/chat` (`:105`), `/feedback` (`:165`), `/history` (`:176`), `/debug` (`:183`), `/awake` (`:189`) — confirmé à l'exécution : `GET /debug` → `{"routes":["/static/<path:filename>","/chat","/feedback","/history","/debug","/awake"]}` |
| C2 | `render_template_string` importé (bug récent corrigé) | ✅ | `scripts/webchat.py:19` `from flask import Flask, render_template, render_template_string, request, jsonify` ; utilisé `:153`. (`render_template` reste importé mais inutilisé) |
| C3 | Bind `0.0.0.0:8080` | ✅ | `scripts/webchat.py:29` `FLASK_HOST = "0.0.0.0"`, `:30` `FLASK_PORT = 8080`, `:196` `app.run(host=FLASK_HOST, port=FLASK_PORT, threaded=True)`. À l'exécution le serveur annonce `http://192.168.50.222:8080` → **joignable depuis le LAN**. Aucune authentification : `:102` `REQUIRE_TOKEN = False`, jamais utilisé |
| C4 | Le chat démarre localement | ✅ | Smoke : serveur UP en **4 s**, `GET /awake` → `{"actif":true}`, `GET /debug` → 200. Arrêt par `taskkill //PID 37040 //F` (PID vérifié `python.exe`), port 8080 libéré derrière |
| C5 | Le chat peut **répondre** à un message | ❌ | `POST /chat` (message=bonjour) → **HTTP 500**. Traceback réel : `scripts/webchat.py:122 in chat` → `scripts/webchat.py:81 in _append_access` → `ACCESS_LOG.append_text(line)` → `AttributeError: 'WindowsPath' object has no attribute 'append_text'`. **Le chat démarre mais aucun message ne peut aboutir** |

### D. LE MÉTABOLISME

| # | Affirmation | Verdict | Preuve |
|---|---|---|---|
| D1 | Code du Métabolisme, 4 nourritures | ❌ | `grep -rni "métabolisme\|metabolism\|nourriture\|garde-manger"` sur tout le dépôt (`.py`, `.md`, `.sh`) : **aucun module, aucune fonction, aucune donnée**. Seules occurrences : trois chaînes d'affichage du badge dans `scripts/webchat.py:150,239,277`. Les 4 nourritures ne sont **ni codées ni même écrites** dans un `.md` du dépôt |
| D2 | Gardes-fous dans le code | ❌ | Aucun garde-fou de Métabolisme dans le code. Le seul document normatif est `PROTOCOLE-reprise-run.md:120-126` (interdiction du kill global, interdiction du mot de passe en clair) — c'est un `.md`, pas une garde exécutable |
| D3 | Journal (`metabolism.log` ou équivalent) écrit par le code | ❌ | `data/metabolism.log` : **inexistant**. Aucune ligne de code n'écrit un journal métabolique. Le seul journal du chat est `ACCESS_LOG` (`data/webchat/access.log`), et son écriture **plante** (cf. C5) |
| D4 | `awake.flag` : créé / lu / supprimé | ⚠️ | Créé : `scripts/webchat.py:88-93` `_set_awake(True)` → `AWAKE_FLAG.write_text("")`. Lu : `:95-96` `_is_awake()` ; route `/awake` `:189-191`. **Supprimé : jamais** — `_set_awake(False)` n'est appelé nulle part, et le commentaire `:112-113` renvoie à un `setTimeout`/`beforeUnload` **absent du JS** (`:237-245`). Conséquence observable : `data/awake.flag` existe depuis le 14/11 (0 octet) et le badge affiche « actif » en permanence. Lien Métabolisme : **nominal seulement** |

### E. LE DPO À ANNOTATEUR UNIQUE

| # | Affirmation | Verdict | Preuve |
|---|---|---|---|
| E1 | Code de collecte des paires | ❌ | `scripts/webchat.py:165-174` : la route `/feedback` lit `choix`, appelle `_append_access` (qui plante, cf. C5) et **retourne `ok=True` sans rien stocker**. Commentaire explicite `:170-172` : « Écriture dans preferences.jsonl seulement si IP consignée du user… Pour l'instant on journalise simplement ». Aucune paire (chosen/rejected) n'est constituée nulle part |
| E2 | Format `preferences.jsonl` défini dans le code | ❌ | `preferences.jsonl` **n'existe pas** sur le disque et n'est **jamais ouvert** en lecture ou écriture. Deux mentions, en commentaire seulement : `scripts/webchat.py:170`. Aucun schéma, aucun sérialiseur, aucun test |
| E3 | Bouton 👍 câblé | ❌ | Les boutons existent : `scripts/webchat.py:234-235`. Mais ils sont dans `<div class="goodbad">` (`:233`), **hors du `<form>`**, sans `type="submit"`, sans écouteur JS (le seul `addEventListener` est sur le formulaire de chat, `:242`), donc sans action. De plus le formulaire de chat utilise `type="envoyer"` (`:231`), invalide |

### F. LE GARDE-MANGER

| # | Affirmation | Verdict | Preuve |
|---|---|---|---|
| F1 | Code de chargement Claire / GSM8K / ARC-AGI-1 | ❌ | `grep -rn "GSM8K\|gsm8k"` → **0**. `grep -rn "ARC-AGI\|arc_agi"` → **0**. `grep -rn "\bClaire\b\|CLAIRE"` (hors corpus) → **0**. Ces trois jeux de données ne sont **mentionnés nulle part**, ni en code ni en doc. `mycelia/data.py` ne connaît que : le corpus texte local (`:73` `load_tokenizer`, `:98` `TextDataset`, `:177` `get_text_batch`) et la tâche jouet de copie différée |
| F2 | Corpus présents localement | ✅ | `data/corpus_fr.txt` **24,7 Mo** (3/11), `data/corpus_gutenberg.txt` **61,4 Mo** (13/11), `data/tokenizer.json` **709 Ko**, `data/train.parquet` **17,8 Mo**, `data/models/qwen2.5-0.5b-instruct-q8_0.gguf` **676 Mo** (13/11) |

### G. LE PROMPT MAÎTRE

| # | Affirmation | Verdict | Preuve |
|---|---|---|---|
| G1 | Prompt maître localisé, contenant les Phases 0-6 | ❌ | `grep -rni "prompt maître\|prompt maitre"` → **0** occurrence. `grep -rn "Phase 0\|Phases 0\|Phase 6"` sur tous les `.md` → **0**. Aucun fichier de prompt, aucune constante de prompt dans le code. Le prompt maître n'est pas dans le dépôt |
| G2 | Interdictions + « rapport exigé » présents | ⚠️ | Les **interdictions** existent, mais dans un seul document et sous forme locale : `PROTOCOLE-reprise-run.md:120-126` (« `taskkill python.exe` / tout kill global : INTERDIT », « mot de passe en clair… INTERDIT », « toute commande globale sans cible vérifiée : INTERDITE »). Aucune trace d'une clause « rapport exigé » nulle part |

### H. LE LANCEUR SERVEUR

| # | Affirmation | Verdict | Preuve |
|---|---|---|---|
| H1 | `--mode run` et `--mode smoke` | ✅ | `scripts/run_twin_10020.py:199-200` `parser.add_argument("--mode", choices=("smoke","run"), default="run")` ; 100 pas en smoke, 3 000 en run `:212` ; rapports distincts `smoke_twin10020.json` / `twin10020.json` `:213`. `--help` le confirme à l'exécution |
| H2 | `hwm_mb()` via `/proc/self/status` (Linux-only) | ⚠️ | Présent : `scripts/run_twin_10020.py:50-51` (`open("/proc/self/status")`, `VmHWM`). **Conséquence** : le script **ne peut pas tourner sur Windows** — il lève `FileNotFoundError` au premier callback ; il est donc **serveur-only**. C'est dit dans `PROTOCOLE-reprise-run.md:9-13`. Le smoke B2 existant (`hwm_mb: 2107.5`) a donc bien été produit sur Linux |
| H3 | Venv cible documenté | ✅ | `PROTOCOLE-reprise-run.md:14-16` : prendre `.venv-cu126` (Python 3.12, avec torch), **pas** `.venv` (Python 3.14, sans torch). Cohérent avec le PC, où `python` système = 3.12.10 + torch 2.11.0+cu128, et `.venv` = 3.14 |

---

## 3. LES MENTIONS À EFFACER

Affirmations que le code ne soutient pas encore. Constat, sans jugement :

| Mention | Ce que le code montre |
|---|---|
| « le chat de Nyra » / interface fonctionnelle | `POST /chat` → HTTP 500 (`webchat.py:81`). L'interface démarre et s'affiche ; elle ne traite aucun message |
| « Nyra » comme identité | « Nyra » est une **chaîne littérale** dans `webchat.py` (11 occurrences). Aucun `self.json`, aucune lecture d'identité par le chat |
| « self.json : l'identité du corps » | Le module existe (`identity.py`), l'intégration n'existe pas : rien n'appelle `Identity.birth`/`load` hors du chat terminal et des tests |
| « le Métabolisme » (4 nourritures, gardes-fous, journal) | **Zéro ligne de code**, zéro ligne de doc. Seul un badge d'affichage |
| « DPO à annotateur unique » | Boutons 👍/👎 inertes ; `/feedback` répond `ok` sans écrire ; `preferences.jsonl` n'existe pas et n'a aucun format défini |
| « le Garde-manger » (Claire, GSM8K, ARC-AGI-1) | Aucun des trois noms n'apparaît dans le dépôt. Seuls corpus présents : LeCarnet FR + Gutenberg FR + GGUF Qwen |
| « le Prompt maître » (Phases 0-6) | Introuvable : 0 occurrence des mots, 0 occurrence des phases |
| « B2 : 10 020 nœuds, seuil 0,002342 » | Couverture 10 020/10 020 **mesurée au smoke** ; perte **2,7801** ≫ 0,002342 ; run 3 000 pas **sans résultat** (`twin10020.json` absent) |
| « reprise testée, VERT » (`PROTOCOLE:72-73`) | Le test 64 nœuds invoqué **n'a laissé aucune trace** dans le dépôt (aucun log, aucun JSON). Le mécanisme est codé et son `--help` passe ; l'essai lui-même est **non reproductible depuis ce dépôt** |

---

## 4. LES TROUS DE SÉCURITÉ

| Point | Constat | Fichier |
|---|---|---|
| `xela` | **Aucune** occurrence dans les sources. Les seuls fichiers qui contiennent cette chaîne sont des wheels CUDA binaires (`.server/cuda-min/*.whl`) — faux positif, pas un secret | — |
| `sshpass` | **1 occurrence** : une commande SSH qui pointe vers un fichier de mot de passe **situé hors du projet**. Le secret lui-même n'est pas dans le dépôt, mais une commande non interactive l'expose en clair dans l'historique de commandes | `.server/server.bat:1` |
| `password` | Aucun dans le code. Occurrences uniquement dans le vocabulaire d'un tokenizer Hugging Face (`.cache/…/tokenizer.json`) — pas des secrets | — |
| Clé privée | **Clé privée OpenSSH en clair** dans le dossier projet | `.server/id_ed25519_mycelia` (couverte par `.gitignore:33` → jamais commitée) |
| Réseau sortant | **1 seul programme sortant** : collecte Gutendex (`urllib.request`, `https://gutendex.com/books/?languages=fr`) — assumé, documenté, script de corpus | `scripts/fetch_gutenberg_fr.py:20,25,41-42` |
| Réseau sortant (trompeur) | Le chat sonde `8.8.8.8:80` pour trouver l'IP locale, alors que son docstring et 3 lignes d'affichage promettent « aucune sortie internet ». UDP : aucun paquet émis, mais l'intention contredit la promesse | `scripts/webchat.py:258-262` |
| Kills globaux | **Aucun `taskkill`/`pkill`/`killall` dans le code exécutable.** `start.py:101` `process.terminate()` cible son propre enfant ; `mycelia/train.py:154` `os.kill(pid, 0)` est une **sonde** (documentée `CONTRIBUTING.md:23`), pas un meurtre. L'interdiction du kill global est écrite `PROTOCOLE-reprise-run.md:123` | — |
| PID codés en dur | **Aucun** | — |
| Exposition réseau | Bind `0.0.0.0:8080`, sans auth, avec un `/debug` qui énumère les routes, et un `/chat` qui plante. Accessible depuis le LAN (annoncé `192.168.50.222:8080`) | `scripts/webchat.py:29,102,183-187` |

---

## 5. LES CONTRADICTIONS CODE / DOCS

| Doc | Ce qu'elle affirme | Réalité vérifiée |
|---|---|---|
| `README.md:108` | `tests/  7 files, 43 tests` | **17 fichiers**, **90 tests collectés** (`pytest --collect-only -q` → `90 tests collected in 1.17s`) |
| `README.md:188-189` | `pytest -q` → `43 passed in ~70 s` | 90 tests collectés ; le chiffre est périmé d'environ deux fois |
| `README.md` (tout) | Décrit le projet v5 du 7/11 : dashboard, `--scale`, corpus LeCarnet | Le code a depuis produit Nyra, `webchat.py`, B2 10 020, `run_twin_10020.py`, la greffe Qwen, le routage sémantique. **Le README ignore ces cinq chantiers** |
| `data/deployment_checklist.md:21` | « 55 passed (2 skippés) » | 90 tests collectés aujourd'hui |
| `rapports/INDEX.md:6` | « ce dossier est désormais le **seul endroit** où les rapports vivent » | `RAPPORT-session-w4.md`, `RAPPORT-session-f…w2.md` (20 fichiers) sont à la **racine** ; seul W4 est dans les deux. `docs/RAPPORT-session-08-11-c.md` existe aussi, **vide (0 octet)** alors que `rapports/RAPPORT-session-08-11-c.md` fait 8,7 Ko |
| `rapports/INDEX.md:8` | « 21 documents » | Vue obsolète (« jusqu'au 8 novembre ») : depuis, 20 `RAPPORT-session-*` ont été produits à la racine, jusqu'à W4 (14/11) |
| `PROTOCOLE-reprise-run.md:72-73` | « Testé le 15 nov sur un cerveau de 64 nœuds : save step 7 → resume step 7… **VERT** » | Aucun artefact de ce test dans le dépôt (pas de log, pas de JSON, pas de test automatisé le couvrant côté `scripts/`) |
| `PROTOCOLE-reprise-run.md:71` | « `--resume` câblé » | Vrai **dans l'arbre de travail**, **faux dans `HEAD`** : `git diff scripts/run_twin_10020.py` montre l'ajout entier de `resume_trainer` et de l'argument |
| `README.md` (Run, §6) | Le cockpit se lance sur `127.0.0.1:8080` via `scripts/dashboard.py` | Le port 8080 est désormais occupé par `webchat.py` (`0.0.0.0:8080`). Deux applications, un seul port, doc qui n'en mentionne qu'une |
| `docs/README.{ar,de,es,ja,pt,ru,zh}.md` | Traductions du README | Le fichier dit lui-même être un **gel du 5/11** ; il décrit une architecture antérieure au routage |

---

## 6. CE QUI EST INVÉRIFIABLE DEPUIS LE PC

Ces 7 points exigent **une seule session SSH** sur `192.168.50.53`. Aucun n'a été testé ici.

1. **Le run B2 est-il vivant ?** `ps aux | grep run_twin_10020` → PID + dernier step du log. (Le protocole interdit de relancer si un run vit.)
2. **Checkpoint de reprise présent ?** `ls -la data/genome_twin10020_run.pt` (écrit tous les 500 pas). Son absence ⇒ le run n'a jamais passé 500 pas.
3. **Hash du génome de base avant toute relance** : `sha256sum data/genome_latest.pt` → attendu **775 851 218 octets** au dernier relevé.
4. **Le génome 10 020 est-il intact ?** `genome_twin10020_smoke.pt` (775 852 137 o attendus) et `genome_latest.pt` (775 851 218 o).
5. **`.venv-cu126` contient-il réellement torch ?** `.venv-cu126/bin/python -c "import torch;print(torch.__version__)"` — et **le `.venv` (3.14) est-il pollué** par le `pip install torch --index-url …/nightly/cpu` lancé avant l'audit (effet de bord non confirmé du protocole) ?
6. **Le résultat du run 3 000 pas existe-t-il côté serveur ?** `ls -la data/mem_report/twin10020.json` — absent en local, jamais produit.
7. **État SSH** : bannissement fail2ban/MaxStartups levé ? clé `id_ed25519_mycelia` installée côté serveur ? (`.ssh/` local est **vide**.)

---

## 7. LA CARTE DU CHEMIN RESTANT

De l'état vérifié aujourd'hui à la « première parole [nyra] ». Dans cet ordre — chaque
case dépend de la précédente.

| # | Case | État | Pourquoi c'est bloquant |
|---|---|---|---|
| 1 | **Réparer `webchat.py`** : `Path.append_text` → écriture réelle (`webchat.py:81`), suppression de `awake.flag`, JS qui n'injecte plus la page entière (`:242`), boutons dans un formulaire (`:233-235`), auth ou bind `127.0.0.1` | ❌ 0 % | Aujourd'hui **aucun message n'aboutit** (HTTP 500). Tout le reste est invisible sans ça |
| 2 | **Brancher l'identité** : créer `self.json` via `Identity.birth`, le lire au démarrage du chat, remplacer la constante « Nyra » | ❌ 0 % | Sans ça, « Nyra » est un titre de page, pas une identité. Le test « débrancher le câble et demander qui es-tu » n'a pas de support dans le chat |
| 3 | **Décoder du texte, pas des identifiants** : `webchat.py:130` fait `chr(max(32,min(126,pred)))` sur un vocabulaire de 100 jetons → la « réponse » ne peut pas être une phrase. Il faut `load_tokenizer` (`mycelia/data.py:73`) + décodage | ❌ 0 % | C'est la différence entre un badge et une parole. Le tokenizer 709 Ko est présent, il n'est pas branché |
| 4 | **Obtenir un cerveau qui parle** : le run 3 000 pas n'a **aucun résultat** ; le smoke est à 2,7801 contre un seuil de 0,002342 (facteur ~1 190) | ❌ | Tant que l'étage 1 n'est pas tranché sur le serveur, il n'y a rien à décoder. C'est **la** question scientifique ouverte du projet |
| 5 | **Métabolisme** : 4 nourritures, gardes-fous exécutables, journal écrit par le code, `awake.flag` avec cycle de vie complet | ❌ 0 % | Aucune ligne n'existe. Le badge affiche « actif » en continu |
| 6 | **DPO** : format `preferences.jsonl`, écriture des paires, boutons câblés | ❌ 0 % | Dépend de 1 (le chat doit répondre pour qu'on juge une réponse) |
| 7 | **Garde-manger** : chargeurs Claire / GSM8K / ARC-AGI-1 | ❌ 0 % | Aucune de ces trois sources n'est nommée dans le dépôt |
| 8 | **Prompt maître** : écrire le document (Phases 0-6, interdictions, rapport exigé) | ❌ 0 % | Les interdictions existent en morceau dans le protocole ; le reste est absent |
| 9 | **Committer l'état réel** : `brain.py`, `lifecycle.py`, `run_twin_10020.py` sont modifiés non commités | ⚠️ | Le `--resume` que le protocole déclare « câblé » n'est pas dans `HEAD` |

**Ce qui est solide et n'a pas besoin d'être refait** : le cerveau routé (structure, routeur,
couverture 10 020/10 020), `Trainer.resume` avec son log, le harnais B2 (modes, HWM, seuils
consignés avant), le corpus FR + Gutenberg, les 90 tests collectés, le protocole de reprise
en 3 cas avec hash obligatoire.

---

## 8. EN UNE PHRASE

Le cerveau est réel et sa couverture 10 020/10 020 est mesurée, mais **Nyra n'existe pas
encore comme être** : le nom est une constante, l'identité n'est jamais lue, le chat plante
sur chaque message, le Métabolisme, le DPO, le Garde-manger et le Prompt maître sont à zéro
ligne de code, et la seule question qui compte — le run B2 tient-il son seuil — n'a toujours
pas de résultat.
