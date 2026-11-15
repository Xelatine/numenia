# Session U — verdict rouge, encodeur vert, nom non imposé

**Trois chiffres suffisent :**

- jumeau sparse : **0,0747** de perte finale pour un plafond de **0,004684** →
  `stage1_admissible: false`, top-k non exécuté ;
- golden : **5/5** avec EmbeddingGemma-300M (md5 `29ebd01c…`, 333 590 944 o),
  **4/5** avec les préfixes officiels — la configuration mesurée est consignée ;
- nom : le mécanisme produirait **« Bou »** → je ne l'impose pas, l'organisme
  reste sans-nom jusqu'à ta décision.

## 1. Verdict du jumeau : ÉCHEC DE LA PISTE A

`data/mem_report/sparse_3000.json`, lu tel quel :

| | |
|---|---|
| Pas | 3 000 |
| Perte finale | **0,0747** |
| Référence dense | 0,002342 |
| Plafond étage 1 | 0,004684 |
| Verdict | **`stage1_admissible: false`** |
| Top-k 50/25/10/5 % | **non exécuté** (« étage 1 échoué (plan B) ») |
| HWM | 749,5 Mo |
| Durée | 42 518,6 s (11 h 48) |

L'écart au plafond est de **×15,9**. Le sparse n'est pas « presque » admissible :
il est seize fois au-dessus. Conséquence doctrinale : **pas d'ingestion dense
« en attendant »**, pas de top-k inventé, **Plan B** (Top-KAST / gate Switch)
comme prochaine piste. Le contraste avec le dense (×679 à 50 %) ne peut pas être
lu puisque la campagne top-k n'a pas eu lieu — et c'est un résultat en soi.

## 2. Encodeur : EmbeddingGemma-300M, golden 5/5

`google/embeddinggemma-300m` est **gated** (accord manuel). Le GGUF Q8_0 de
`ggml-org/embeddinggemma-300M-GGUF` est libre :

| | |
|---|---|
| Taille | 333 590 944 o |
| md5 (calculé, identique Windows/BIO-1) | `29ebd01c9362065f36405ef185c7755d` |
| dim | 768 |
| HWM mesuré | **664,4 Mo** |
| Golden | **5/5** (hit@1) |

Le détail qui compte : les préfixes officiels **dégradent** le golden.

| Configuration | Golden | Held-out (5 paires inédites) |
|---|---|---|
| Qwen 0,5B brut (baseline) | 3/5 | — |
| Gemma, préfixes query + document | 4/5 | 5/5 |
| Gemma, préfixe document seul | **5/5** | **5/5** |
| Gemma, aucun préfixe | 4/5 | — |

La paire difficile est « Je déteste qu'on me coupe la parole » / « qu'est-ce qui
l'énerve ? » : cos 0,322 contre 0,279 pour le meilleur faux. Le 5/5 tient à
**0,043 de marge** sur cette paire. Le held-out 5/5 dans les deux configs dit que
le choix n'est pas un sur-ajustement grotesque, mais la marge reste mince et je
ne la présente pas comme confortable.

### Détecteur d'ignorance : seuil 0,55 → 0,30

Le seuil hérité 0,55 aurait déclaré **inconnus deux souvenirs corrects** (0,521
et 0,322). Mesure sur 5 connus + 5 inconnus :

- connus : 0,741 / 0,563 / 0,521 / 0,616 / 0,322 → **min 0,322** ;
- inconnus : 0,132 / 0,260 / 0,187 / 0,199 / 0,159 → **max 0,260** ;
- séparation **0,062**, seuil **0,30** : 5/5 connus acceptés, 5/5 inconnus rejetés.

`ExternalOrgans.confidence_threshold` passe donc à **0,30**. Les niveaux 3-4 sont
**armés** (condition golden remplie), mais aucun client Gemini/web n'est appelé
sans clé : le code refuse toujours explicitement.

## 3. Nom : « Bou » — refusé, décision laissée à l'utilisateur

Le mécanisme (signature structurelle, 4/4 tests) appliqué au génome dense vivant
(`genome_run3000.pt`, 3 000 pas, 1 005 nœuds vivants, 6 246 arêtes) donne
**« Bou »**. Ce n'est pas un nom, c'est un artefact de table de syllabes.

Je ne l'écris pas. `self.json` garde « sans-nom (à nommer au premier Sommeil) ».
Aucun Sommeil n'est implémenté à ce jour : le déclencheur n'existe pas encore.
**Le nom t'appartient** — dis-le et je l'écris ; sinon le mécanisme devra être
corrigé (longueur minimale, table revue) avant de servir.

## 4. Les deux longues tâches : NON LANCÉES, et pas seulement à cause du verdict

| Script attendu | Réalité |
|---|---|
| `scripts/run_ingestion.py` | **n'existe pas** |
| `scripts/run_corpus.py` | **n'existe pas** |

Même avec un verdict vert, les commandes de la séquence auraient échoué. Il
faudra les écrire (reprise, checkpoints/500, suivi RSS 24 h) avant de parler de
7-8 jours. Rien n'a été lancé : ni ingestion, ni encodage, ni dense « en
attendant ». La doctrine est respectée.

## 5. Sens : non intégrés, et c'était le bon ordre

Whisper et SmolVLM n'ont de sens que **pendant** un long run. Il n'y a pas de
long run. Les télécharger maintenant aurait ajouté deux modèles non mesurés à un
organisme qui n'ingère pas. Rien n'a été maquillé en « prêt ».

## 6. Réseau : réparé par la mesure, pas par le câble

`enp8s0` est toujours en lien-local (DHCP non abouti) ; le trafic passe par
`wlp7s0`. Débit serveur : 27 Ko/s, effondré à 8 Ko/s en parallèle. Le lien local
Windows → BIO-1 mesure **4,5 Mo/s** et le téléchargement Windows **20 Mo/s**.
Le GGUF a donc été téléchargé sur Windows (6 s) puis poussé par SSH (38 s), md5
vérifié des deux côtés.

## État consolidé

| Élément | État |
|---|---|
| Jumeau sparse | **ROUGE** — 0,0747 vs 0,004684, top-k non exécuté |
| Encodeur | **VERT** — Gemma 5/5, held-out 5/5, HWM 664,4 Mo |
| Détecteur d'ignorance | **calibré 0,30**, marge 0,062 |
| Niveaux 3-4 | **armés**, clients absents (aucun appel) |
| Nom | **non imposé** — « Bou » rejeté, décision utilisateur |
| Ingestion / corpus | **scripts absents**, non lancés |
| Sens | **non intégrés** (pas de run à couvrir) |
| Plan B | **prochaine étape** — non commencé |

## Prochaine action, dans l'ordre

1. écrire le **Plan B** sparse (Top-KAST ou gate Switch) et le mesurer comme le
   jumeau : perte ≤ 0,004684 à 50 % de zéros, HWM, équité ;
2. écrire `run_ingestion.py` et `run_corpus.py` avec reprise et checkpoints ;
3. alors seulement : ingestion sparse, encodage, puis whisper/SmolVLM pendant
   le run ;
4. le nom : ta décision, ou un générateur corrigé au premier Sommeil.
