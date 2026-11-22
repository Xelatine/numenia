# RAPPORT S8-REPAIR-AD — réparer A, améliorer D, rouvrir le chat

*2026-11-22. Contraintes respectées : aucun `pip install`, aucun CDN, génome de Nyra
intact, clefs JSON et identifiants `mycelia_*` inchangés, aucun
`load_state_dict(strict=False)`.*

## Résumé en trois lignes

- **A / seed1-2** : c'était un **fossile**, pas un résultat. Re-traduits :
  4,6669 → **0,4481** et 4,7532 → **0,4638**.
- **D / seed1-3** : c'était un **bug vivant**, pas un plafond physique.
  `teacher_parts` décalait le professeur cerveau d'un token. Corrigé :
  0,8542 / 0,9455 / 0,8990 → **0,2600 / 0,2752 / 0,2572**.
- **Chat S8** : le 404 n'était pas la mémoire mais un `NameError` (`torch` importé
  dans une seule fonction). Les tags `trad_X` chargent maintenant un seul fichier
  à la demande. Testé, réponses cohérentes, serveur laissé up.

Table finale : **12 / 12 en 🟢**.

## Phase 1 — A : diagnostic « fossile ou bug vivant ? »

Le fichier remplacé portait trois indices internes qui se contredisaient :

| indice | valeur | ce qu'elle dit |
|---|---|---|
| perte d'entraînement | 3,5241 / 3,5392 | identique à A/seed3 (3,5582) → l'élève **avait appris** |
| top-3 neuf | 0,2416 / 0,2809 | normal (A/seed3 : 0,2697) |
| KL | 4,6669 / 4,7532 | or `ln(312) = 5,74` : deux distributions quasi disjointes |

Une perte de 3,52 avec une KL de 4,7 est incompatible avec le même couple de
modèles : la **mesure** était fausse, pas l'apprentissage. La cause est le
décalage d'un cran déjà corrigé dans `s8_run_bench._teacher_logits` (le dense
prédit `x[t+1]` au pas `t`, le cerveau prédit `x[t]`) ; seule la re-traduction
manquait. Les valeurs fossiles sont conservées dans
`data/lab/s8/historique_fossiles_A.json`.

**Résultat** : 0,4481 et 0,4638, au niveau de A/seed3 (0,4635) qui avait servi de
référence. Aucun bug vivant propre à A.

## Phase 2 — D : diagnostic « protocole ou plafond ? »

### Ce qui plaidait pour le plafond (hypothèse a)

La KL de D (0,8542 / 0,8990 / 0,9455) **encadre exactement** le plafond pool-24
mesuré en S5 : `0,8582` (contrat C5 de `test_chain.py`, ré-exécuté aujourd'hui :
`KL(A||MYC)=0,8318, plafond=0,8582`). Un donneur « au plafond » est l'explication
la plus naturelle — et c'est celle qui avait été retenue.

### Ce qui l'a écartée

D et son receveur ont **la même architecture** : 24 nœuds, 220 arêtes,
`two_speeds=True`. Un élève de même taille peut copier son professeur
exactement ; si le plafond était une limite de capacité, il serait ~0, pas 0,85.
Le plafond de 0,8582 vaut pour un professeur **dense** (16 dimensions de lecture
contre 24 membranes), pas pour un professeur de même forme.

### La cause réelle (hypothèse b, confirmée)

`teacher_parts` rendait `logits[:, :-1]` pour un cerveau — le cran du **dense**.
Or `ToyCorpus.targets(x) = x` : un cerveau prédit le token `t` au pas `t`
(il lit avant d'injecter). Donc :

- dense : le pas `t-1` prédit le token `t` → `logits[:, :-1]` ✔
- cerveau : le pas `t` prédit le token `t` → `logits[:, 1:]` ✘ (le code prenait `[:-1]`)

Le professeur enseignait donc le token **précédent**, et la ligne 0 (aucun
contexte) servait de cible au pas 1 : deux défauts en un.

**Correctif** (une seule variable) : `repaired_mycelia.teacher_parts`, branche
cerveau → `logits[:, 1:], hidden[:, 1:]`, avec le commentaire qui explique
pourquoi les deux familles n'ont pas le même cran à couper.

**Contrôles d'identité exécutés avant de relancer** (pas après) :

```
brain : max|teacher_parts(D) - D(x)[0][:, 1:]| = 0.0        OK
dense : max|teacher_parts(B) - parts(B)[1][:, :-1]| = 0.0   OK (branche dense intacte)
```

**Résultat** : 0,2600 / 0,2752 / 0,2572 ; top-3 identiques 1,8 % → 11,1 % (×6) ;
recouvrement top-3 24,7 % → 52,8 %. D devient le **meilleur** donneur du banc et
passe **sous** le plafond de 0,8582, ce qui confirme que ce plafond ne
s'appliquait pas à ce cas. Aucun contournement : ni température, ni epochs, ni
exemples ajoutés — la ligne fausse, corrigée.

### Témoin (règle appliquée)

Toute modification du protocole est re-testée sur un donneur sain, avant d'être
crue. Ici le témoin est **B/seed1** (dense, chemin de code non touché) :

| | avant | après | écart |
|---|---|---|---|
| KL | 0,4450 | 0,4434 | 0,4 % (bruit numérique CPU) |
| top-3 identiques | 0,0318 | 0,0383 | même ordre |
| perte | 3,4901 | 3,4816 | 0,2 % |

Aucune dérive du protocole : le correctif ne touche que la branche cerveau.

### Bonus de cohérence : B/seed2

`bench_B_seed2.json` était le seul fichier du banc sans la clef `contextes` et
avec `top3_identiques = top3_recouvrement = 0.0` alors que sa KL (0,4458) est
celle de B/seed1 (0,43 de recouvrement) — improbable sur 22 400 contextes.
Re-mesuré, protocole inchangé : **0,4453**, top-3 0,0290 / 0,4523, clef
rétablie. Anciennes valeurs dans `historique_B_seed2_anomalie.json`.

## Phase 3 — chat S8 : le 404 n'était pas la mémoire

Trois défauts se cumulaient, tous dans `scripts/webchat.py` :

1. **`torch` et `F` n'étaient importés que dans une seule fonction** → tout le
   bloc S8 levait `NameError: name 'torch' is not defined`, avalé par les
   `except Exception` et rendu en 404. C'était la vraie cause depuis le début ;
   la piste mémoire était une fausse piste.
2. **`_predict_brain` lisait `out.dim()`** alors que `Brain.forward` rend
   **toujours** un tuple → `AttributeError` sur le premier token, donc 404 même
   quand le fichier existait.
3. **Aucun tag `trad_X` ne correspondait à un fichier** : les traduits vivent
   dans `recipient_X_seedN.pt`, pas dans `donors/trad_X.pt`.

Correctifs :

- imports `torch` / `torch.nn.functional as F` au niveau module (commentés) ;
- `_predict_brain` réécrit : état porté le long du prompt, préfixe **complet**
  (token 0 inclus, comme `teacher_parts`), lecture de la dernière position ;
- `_s8_predict_tag` : routage `trad_X` → `s8/recipient_X_seed{seed}.pt`,
  **un seul** fichier lu par requête puis relâché (272 Ko), aucune ouverture de
  donneur ; message d'erreur exact (`fichier absent : …`, `lecture impossible
  (X)`, `chargement impossible (X)`) au lieu d'un 404 muet.

### Tests live (curl, serveur 8080)

| requête | réponse |
|---|---|
| `GET /api/status` | `voice=numenia, nodes=532, step=56, awake=true` |
| `POST trad_D seed1 ["le chat dort"]` | `?, est, ferme` — entropie 3,77 |
| `POST D ["le chat dort"]` | `vole., ferme, ferme.` — entropie 4,3918 |
| `POST trad_A seed1 ["pourquoi le"]` | `et, ?, se` — entropie 3,1187 |
| `POST trad_B seed9` (absent) | **404** `fichier absent : recipient_B_seed9.pt` |

Les deux valeurs de `le chat dort` (donneur 4,3918 / traduit 3,77) sont
**identiques** à celles de `DEMO-NUMENIA.md` : le chat et la démo répondent à la
même question sur le même préfixe. Le serveur reste up à la fin de la session.

## Table finale — 12 / 12 en 🟢

| Donneur | Sem. | KL avant (valeur remplacée) | KL après | Top-3 id. | Recouv. | Verdict |
|---|---|---|---|---|---|---|
| A | 1 | 4,6669 | **0,4481** | 0,0255 | 0,4488 | 🟢 |
| A | 2 | 4,7532 | **0,4638** | 0,0255 | 0,4182 | 🟢 |
| A | 3 | — | **0,4635** | 0,0237 | 0,4358 | 🟢 |
| B | 1 | — | **0,4434** | 0,0383 | 0,4348 | 🟢 |
| B | 2 | 0,4458 | **0,4453** | 0,0290 | 0,4523 | 🟢 |
| B | 3 | — | **0,4237** | 0,0305 | 0,4387 | 🟢 |
| C | 1 | — | **0,5718** | 0,0007 | 0,2433 | 🟢 |
| C | 2 | — | **0,5719** | 0,0005 | 0,1950 | 🟢 |
| C | 3 | — | **0,5999** | 0,0002 | 0,2050 | 🟢 |
| D | 1 | 0,8542 | **0,2600** | 0,1111 | 0,5282 | 🟢 |
| D | 2 | 0,9455 | **0,2752** | 0,1174 | 0,5261 | 🟢 |
| D | 3 | 0,8990 | **0,2572** | 0,0939 | 0,5407 | 🟢 |

## Vérifications exécutées

- `py_compile` : `s8_run_bench.py`, `repaired_mycelia.py`, `showcase.py`, `webchat.py` — OK
- `test_chain.py` : **10/10 contrats** (28,4 s)
- génome : `data/genome.pt`, 49 425 917 octets, md5 `ffa53e100c1d4f642a0290517d99f856` — **inchangé**
- `grep -r "repaired_numenia|columns_numenia"` : **0** occurrence
- clefs JSON du banc : identiques aux fichiers de référence (B/seed2 re-mesuré avec sa clef)

## Ce qui reste ouvert (dit franchement)

- **Qwen** : toujours impossible ici (`transformers` absent, vocabulaire 151 840
  contre 312). Rien ne l'a débloqué.
- **Reproductibilité** : CPU non bit-à-bit (B/seed1 varie de 0,4 %). Le protocole
  n'a pas changé, seule la branche cerveau de `teacher_parts`.
- **Sans vie** : les 12 traductions restent mesurées avant vie (600 pas), comme
  le protocole S8 simplifié.
- Le `_predict_brain` **mort** (première copie du bloc S8, masquée par la
  seconde) garde encore son ancien `out.dim()` : sans effet aujourd'hui, mais à
  supprimer lors d'un prochain nettoyage du fichier.
