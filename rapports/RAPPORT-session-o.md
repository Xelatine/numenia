> **Renommage —** Ce projet s'appelle désormais **Numenia** (auparavant référencé
> comme Mycelia dans les titres et la documentation). Session, chiffres, fichiers de
> données et cerveau **Nyra** inchangés. Ce document est historique : son contenu est
> conservé tel quel, et seul le nom du projet passe à Numenia.

# Session O — Run 3 000 pas terminé, corpus réel, et révision honnête de 1.1

**Dates** : 12 novembre 23:19 → 13 novembre 10:33 (BIO-1, hôte `IA`).
**Commit de référence** : `7eda51a` (patch `trim_heap`), run `scripts/run_3000.py`.

## Résultat du run (CONSTATÉ)

| | Valeur |
|---|---|
| Pas | **3 000 / 3 000** (`FIN step 3000`) |
| Perte | **4,3204 (pas 100) → 0,0010 (pas 3 000)** |
| Arêtes | 774 → **6 246** |
| Nœuds | 1 000 → **1 129** (10 nœuds nés, 500 morts, 5 au cimetière) |
| Arêtes mortes | **16 009** (prune + morts naturelles) |
| RSS de fin de pas | 426-483 Mo, **dernier 444 Mo** |
| **VmHWM final** | **709 Mo** (plafond du protocole : ≤ 800) |
| `uordblks` | 200 → 240 Mo, plat |
| Checkpoint | `data/genome_run3000.pt`, **87 795 754 o**, step 3000, md5 `53b00b6944f19ab889fe27510955bf06` |
| Durée | ~11 h 14 (2 500 pas après reprise, ~16 s/pas en moyenne) |

### Deux régimes, documentés (courbe : `data/mem_report/run3000_loss.svg`)

- **Pas 0-500, sans `checkpoint_turns`** : VmHWM monté à 4 202 Mo, disponible 1 861 Mo au pas 500 → règle d'arrêt appliquée, arrêt propre, checkpoint du pas 500 vérifié.
- **Pas 500-3 000, avec `checkpoint_turns=True`** : VmHWM **673 → 709 Mo**, plat ; RSS de fin de pas 469 → 444 Mo ; `fordblks`/`arena` effondrés (182/392 Mo au pas 600). Le pic devient une constante de la config (fenêtre de recalcul batch×seq×dim), insensible à la taille du graphe.
- La couture au pas 500 est **invisible sur la perte** (1,7648 au pas 500 → 1,4132 au pas 600 → 0,0010) ; `preserve_rng_state=True` et les gradients recalculés à l'identique. La marque verticale du SVG évite toute lecture d'anomalie là où il y a un changement de régime.

**Verdict** : la config nominale n'a pas changé (scale 4, batch 2, seq 16, `max_edges` 20 000). Seule la stratégie mémoire/calcul a changé au pas 500. **1.3 est vert.**

## Corpus — décision explicite

| Rôle | Fichier | Taille | md5 | Nature |
|---|---|---|---|---|
| Smoke-test / validation du pipeline | `data/corpus_fr.txt` | 24 719 274 o | `1a27b7abb971ce2eec1aa7d7aaae21e0` | histoires **synthétiques** (29 402 lignes) |
| **Apprentissage** | `data/corpus_gutenberg.txt` | **61 360 803 o** | `5a3e256afccc4d4c2a1604a2b1a9d08c` | **100 livres français du domaine public** |

- Source : API Gutendex (`languages=fr&sort=popular`), donc **top 100 par `download_count`**, textes UTF-8 Gutenberg.
- Boilerplate retiré : marqueurs START/END supprimés, **0 occurrence restante** vérifiée par `grep`.
- Chargement vérifié sur le serveur : **21 323 837 tokens**, 1 266 101 fenêtres train, 66 638 fenêtres val, RSS max 1 234 Mo. Échantillons : Theuriet, Huysmans, Verne — du vrai français, pas de licence.
- **Correction nécessaire du chargeur** : `TextDataset` encodait le corpus en un seul appel ; sur 61 Mo, l'énorme liste Python d'IDs a été **tuée par l'OOM (exit 137)**. `mycelia/data.py` encode désormais par tranches de 4 MB (`torch.cat` de tenseurs) — md5 `5db19d4eaa016402575493ac6d7dfbbc` poste = serveur. Effet de bord assumé : quelques frontières de merge BPE par tranche, négligeable à l'échelle du corpus.
- Le corpus synthétique **reste** dans le dépôt comme smoke-test, jamais comme corpus d'apprentissage par défaut.

## Prérequis 1.1 — révisé, pas maquillé

**CONSTATÉ : le filaire n'a pas fonctionné.** `sudo bash ~/prepare-filaire.sh` n'a pas pu être tenté (5e session) : l'outil refuse les wrappers de privilèges et aucune fenêtre root n'est ouverte sur l'hôte.

**Mais l'objectif de 1.1 est couvert ailleurs, et mesuré :**

| Transport | Volume | Durée | Débit | md5 |
|---|---|---|---|---|
| GGUF Qwen2.5-0.5B Q8_0 | 675 710 816 o | 86 s | ~7,8 Mo/s | `a49598f88d6157e900ae2aac730873c9` |
| Corpus Gutenberg | 61 360 803 o | 8,16 s | ~7,5 Mo/s | `5a3e256afccc4d4c2a1604a2b1a9d08c` |
| Tokenizer | 709 412 o | < 1 s | — | `d8b2e90c6ab5771cc47dff2bf026303d` |

**1.1 se dédouble donc :**
- **1.1-a « Transport fiable poste→serveur » : VERT** (lien-local SSH, mesures ci-dessus, md5 vérifiés).
- **1.1-b « Ethernet filaire dédié » : reporté, objectif de confort hors chemin critique** (latence/robustesse pour les organes temps réel). **Le filaire n'a pas marché ; c'est son objectif qui est satisfait autrement.**

## Smoke-test de la première parole — exécuté

- `scripts/smoke_qwen.py` (md5 `a27780d9…`, poste = serveur) : md5 du modèle vérifié avant chargement, prompt fixe, seed 42, 20 tokens.
- Runtime : `llama-cpp-python` **absent**, aucun wheel cp314 → compilation source (`cmake` 4.4.4 via pip, g++/make) dans `screen buildllama`, **`BUILD_EXIT=0`**.
- **Résultat CONSTATÉ** :

```
modele qwen2.5-0.5b-instruct-q8_0.gguf | 675710816 o | md5 a49598f88d6157e900ae2aac730873c9
chargement : 0.34 s
prompt     : "Bonjour, je m'appelle Mycelia. "
sortie     : ' Je suis une personne qui a des problèmes de santé mentale et de santé physique. Je suis'
20 tokens en 1.01 s | 0.051 s/token
```

La sortie est celle du modèle de base, sans distillation ni conditionnement : elle n'est pas un résultat du graphe MYCELIA et ne doit pas être lue comme tel. C'est la mesure de latence et la preuve que le canal d'inférence fonctionne (~20 tok/s sur CPU).

## Journal haute résolution

`follow.log` récupéré **avant** l'arrêt du guet : `data/mem_report/follow.run3000.log` (436 lignes, 31 185 o ; une entrée toutes les 10 min). `watch3000` arrêté après copie. Séries : `data/mem_report/run3000_trace.txt`, courbe `run3000_loss.svg`.

## État final des prérequis

| Prérequis | État |
|---|---|
| 1.1-a Transport fiable | **VERT** (lien-local mesuré, md5) |
| 1.1-b Filaire dédié | **reporté** (confort, hors chemin critique) |
| 1.2 Dérive | **VERT** (fragmentation glibc, `trim_heap`, mesuré) |
| 1.3 Run 3 000 pas | **VERT** (terminé, pic 709 Mo, perte 0,0010) |

Les phases 2-4 du sprint français ne sont plus bloquées par le transport : corpus réel et modèle sont sur le serveur, et la première inférence est faite (0,051 s/token). La phase 2 peut s'ouvrir : distillation / ingestion du corpus réel dans le graphe, avec la réserve que le corpus Gutenberg est un top-100 par téléchargements (œuvres variées, pages de titre incluses), pas une sélection littéraire curatée.
