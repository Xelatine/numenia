# NUMENIA

Knowledge distillation from dense AI models into a living, sparse neural network that
survives its own life — an AI-to-AI translator.

NUMENIA is not a stack of layers. It is a sparse colony of small nodes that route
messages, grow and cut synapses, and read out what they have understood — and the
project's central question is whether a *dense* model's knowledge can be distilled
into such a living network well enough that it survives its own metabolism.

The brain that speaks is called **Nyra** (its state lives in `data/nyra_state.json`,
its voice in `scripts/webchat.py`). The project that builds it is **Numenia**.

---

## Sessions S1 → S8

| Session | Titre | Verdict | Chiffre clé |
|---|---|---|---|
| S1 | Scan récurrent NUMENIA-SCAN — reproductibilité de l'instrument | 🟢 | Taux d'engagement 55,5 % ; écart maximal entre deux scans : 0,0 (reproductible) |
| S2 | Jumeaux comparables (dense ↔ Numenia, 23 748 paramètres égaux) | 🔴 Mur confirmé | Numenia plafond ~5,1 (5,078–5,108) ; 2,00 concepts traduits / 8 ; top-3 3,3 % |
| S3-bis | Pourquoi Numenia plafonne à ~5,1 | 🐛 BUG (plafond réparable) | Plancher unigramme 5,0829 ; Numenia 5,1509 = sac de mots ; contexte ignoré (écart normal/mélangé 0,0535 nat) |
| S4 | Réparer la lecture, puis distiller | 🥉 | Numenia réparé 3,9039 vs dense 3,7921 ; gain +1,2445 nat ; distillation : gain isolé −0,1245 |
| S5 | Plafond physique, pool 24, score F | 🥈 / 🔴 | Plafond KL(A‖B) = 0,8582 ; F 0,49 → 0,048 (avant → après vie) ; perte distillée a1b1g0 = 3,5281 ; pool 24 : 3,7363 vs dense jumeau 3,8775 |
| S6 | Consolidation — faire vivre sans oublier | ⚪ Non mesuré (jouet) | Front A (F ≥ 0,4 après vie) et Front B (perte < 3,5281) non livrés sur le jouet ; plafond de forme 3,5281 |
| S7 | Premier traducteur IA→IA | 🟢 (pipeline) | Audit Phase 0 terminé ; 10/10 contrats de couche ; pipeline prêt (professeur dense, receveur Numenia pool 24, distillation a1b1g0, test aveugle, vie 2000 pas) |
| S8 | Banc multi-donateurs (4 IA divergentes × 3 semences) | 🟡 En cours | 12 traductions prévues (distillation a1b1g0 + 600 pas, F avant vie, top-3 sur 5000 contextes) ; premiers bancs A enregistrés, reste en cours |

**Chiffres phares du projet :** plafond KL inter-professeurs = **0,8582 nat** · score F **0,49 → 0,048** (avant → après vie) · perte distillée étendue **3,5281** · **10/10** contrats de couche S7.

---

## Le cerveau : Nyra

Le runtime s'appelle **Nyra**. Ce qui existe, tel que vérifié par l'audit
(`AUDIT-ETAT.md`) :

| Pièce | État |
|---|---|
| Cerveau routé, 10 020 nœuds, 10 % éveillés par token | ✅ `mycelia/brain.py` (`n_nodes`, `route_fraction`) — couverture mesurée 10 020/10 020 au smoke |
| Interface web locale (chat, dashboard, panel) | ✅ `scripts/webchat.py` + `scripts/nyra_panel.py` + `dashboard/` — serveur localhost, réponses et logs |
| Métabolisme, DPO, Garde-manger, Prompt maître | ❌ aucune ligne de code (ticketés dans l'audit) |

Le cockpit est `scripts/dashboard.py` ; le chat est `scripts/webchat.py`. Tous deux
utilisent le port 8080, donc ils ne peuvent pas tourner en même temps.

---

## Représentation

| Niveau | Ce que c'est | Poids |
|---|---|---|
| **Nœud** | L'équivalent d'un neurone biologique : il reçoit, décide, transmet, naît, meurt. Largeur 32. | un nœud |
| **Organes internes** | `gate`, `body`, `router` — paramètres mathématiques, *pas* des neurones. | ~6 400 paramètres par nœud |

---

## Le signal

Il n'y a **aucune matrice dense**. Le câblage est un tampon d'arêtes préalloué
`edge_index` de forme [2, 20 000] plus un compteur `ne`, et une force entraînable
par slot. La matrice 140 000² = 19,6 milliards de cellules n'est **jamais**
allouée.

Le passage avant est une **frontale d'onde**, token par token :

1. Le nœud 0 injecte l'embedding du token dans la frontale (`E = 1`).
2. À chaque saut, chaque nœud chaud transforme *ce qu'il vient de recevoir* par
   sa gate et son body, score ses cibles avec sa propre query contre leurs keys, et
   envoie `out × score × strength × 0,9` au saut suivant.
3. Seuls les nœuds ayant reçu quelque chose calculent. Un signal qui ne reçoit
   rien meurt ; il n'y a pas de règle anti-boucle.
4. Le nœud 1 accumule chaque message dans sa **membrane**, qui refroidit par
   0,995 par saut (~0,96 par token, une vingtaine de tokens de mémoire).
5. À la position `t`, la prédiction se lit sur le nœud 1 *avant* d'injecter le
   token `t`. C'est toute la causalité : prédiction du token courant depuis le
   passé seulement, donc aucun décalage n'est nécessaire dans le jeu de données.
6. La trace `(saut, source, cible)` est enregistrée et streamée vers le cockpit.

---

## Structure du repo

```
mycelia/                 # paquet Python du cerveau (NE PAS renommer — imports)
  brain.py               # graphe, frontale, routeur, naissances, lecture, parole
  node.py                # un neurone (référence) + NodeBank (empilé, croissant, batché)
  lifecycle.py           # usage EMA, pruning, isolation, cimetière, revival, taxes
  growth.py              # réallocation, naissance d'un nœud par un message, gestionnaire par paliers
  checkpoint.py          # génome versionné, optimiseur par nom, état RNG complet
  data.py                # tâche jouet delayed-copy + flux texte français (5 % validation)
  train.py               # Trainer (CLI, dashboard et tests le partagent), OOM, SIGINT
  ...
scripts/                 # scripts — entraînement, génération, dashboard, webchat, labo
dashboard/               # cockpit : runner.py (threadé), server.py (FastAPI + WS), static/index.html
tests/                   # 17 fichiers, 90 tests collectés
configs/                 # config.yaml — tous les seuils, modifiable sans toucher le code
docs/                    # README.fr.md (autoritaire) + traductions gelées (ES, DE, ZH, JA, PT, RU, AR)
rapports/                # rapports de session, verdicts, notes d'ablation
data/                    # checkpoints, corpus, runs/ — reconstruits localement, jamais versionnés
data/lab/                # artefacts du labo (sessions, conversions, S1→S8) — gitignorés
```

---

## Lancement

```bash
pip install -r requirements.txt

# 1. Corpus français (LeCarnet) + BPE 10 000 jetons — téléchargé une fois
python scripts/train_tokenizer.py

# 2. Régression architecture : copie différée, secondes par pas
python mycelia/train.py                       # 2 000 pas, scale 0
python mycelia/train.py --scale 1             # 302 nœuds (C. elegans)
python mycelia/train.py --scale 2             # 3 000 nœuds (larve de mouche)

# 3. Français : 10 000 pas, batch 16, seq 128, backprop tronqué 16
python mycelia/train.py --data text --scale 0

# 4. Parole
python scripts/generate.py --prompt "Il était une fois" --tokens 100

# 5. Le nombre sans lequel rien n'est interprétable
python scripts/train_baseline.py --steps 10000

# 6. Le cockpit
python scripts/dashboard.py --train --data text --scale 0
#    → http://127.0.0.1:8080

# 7. Tout doit rester vert
pytest -q
```

---

## Status honnête

Le cerveau est réel et sa couverture 10 020/10 020 est mesurée. La traduction
dense → réseau vivant est **en cours** : S5 a mesuré le plafond physique (F 0,49 →
0,048 après vie) et S7 a validé le pipeline (10/10 contrats), mais la première
traduction complète bout-en-bout (phase 1 S7) et le banc multi-donateurs S8 ne sont
pas encore terminés. Les chiffres publiés sont ceux qui ont été mesurés ; le reste
est marqué comme en attente, jamais inventé.

---

## Licence

MIT. Voir [LICENSE](LICENSE).

Des conditions additionnelles s'appliquent et prévalent sur le texte MIT : l'auteur
se réserve un **droit de veto** sur toute utilisation, et toute vente (revente,
licence commerciale, produit payant construit sur ce projet) doit être
**notifiée à l'avance, avec une copie adressée à l'auteur**. Voir
[CONDITIONS-ADDITIONNELLES.md](CONDITIONS-ADDITIONNELLES.md).

---

> **Renommage —** Ce projet a été renommé **Numenia** (auparavant référencé comme
> Numenia dans les titres et la documentation). La session, les chiffres, les fichiers
> de données et le cerveau **Nyra** sont inchangés. Les rapports historiques
> (dont `AUDIT-ETAT.md`) conservent leur contenu d'origine ; un bandeau en tête les
> signale. Seul le nom du projet/ dépôt passe à Numenia.
