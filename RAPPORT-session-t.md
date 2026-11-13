# Session T — jumeau sparse lancé, identité et organes câblés

**État du jumeau : EN COURS, verdict non rendu.** Le run de 3 000 pas tourne
détaché (PID 238894) ; il écrira `data/mem_report/sparse_3000.json` à la fin.
Ce rapport ne prétend pas connaître un résultat qui n'existe pas encore.

## 1. Smoke 100 pas : COMPATIBLE (mesuré avant le run long)

Cerveau neuf, `activation="relu"`, `output_sparse=True`, pruning + croissance +
mémoire actifs ensemble :

| Pas | Perte | Arêtes | Vivants | Cimetière | Zéros | HWM |
|---|---|---|---|---|---|---|
| 20 | 4,642 | 169 | 1 002 | 0 | 48,6 % | 447 Mo |
| 60 | 4,555 | 479 | 1 002 | 0 | 48,9 % | 470 Mo |
| **100** | **4,429** | **790** | **1 002** | **0** | **50,0 %** | **541 Mo** |

25 arêtes coupées, **0 nœud enterré** : les nœuds dormants (sortie nulle) ne
sont pas pris pour des morts. Verdict : **compatible**, le run long n'est pas
différé. Artefact : `data/mem_report/smoke_sparse_100.json`.

## 2. Le jumeau 3 000 pas : LANCÉ, pas terminé

- Commande : `setsid .venv/bin/python -u scripts/run_sparse_3000.py`
- Journal : `data/mem_report/sparse_3000.log` (PACE tous les 20 pas, TRACE tous
  les 100, checkpoint tous les 500 → `data/genome_sparse_3000.pt`)
- Étage 1 : perte finale ≤ **0,004684** (2 × 0,002342). Le verdict est écrit
  dans le JSON, pas décidé ici.
- Étage 2 : si l'étage 1 est vert, la campagne top-k {50, 25, 10, 5} % est
  rejouée automatiquement sur ce cerveau, mêmes batchs fixes que la session R.

À la vitesse du smoke (~3,6 s/pas au début, croissante avec les arêtes), le run
prend plusieurs heures. **Aucune notification automatique** : pour suivre,

```
ssh -i .server/id_ed25519_mycelia -6 dieux@fe80::226a:8aff:fea3:5e67%15 \
  'tail -f ~/mycelia/data/mem_report/sparse_3000.log'
```

Le résultat sera dans `data/mem_report/sparse_3000.json` (`stage1_admissible`,
puis `topk`). Tant qu'il n'existe pas, la doctrine sparse reste **non prouvée**.

## 3. self.json : l'identité est offline et mesurée

`mycelia/identity.py` — créé dans `data/genome_myc/self.json` :

```json
{
  "name": "sans-nom (à nommer au premier Sommeil)",
  "birth_ts": "2026-11-13T23:15:27+00:00",
  "declared_sex": "indéterminé",
  "declared_age_fn": "now() - birth_ts",
  "version_genome": "myc-1:34caba7813561650030199010018e7ae",
  "stable": true
}
```

Réponse hors ligne : « Je suis sans-nom (à nommer au premier Sommeil). Né le
2026-11-13, âge mesuré 0.0 jours humains (0.00 jour CPU). Sexe déclaré :
indéterminé. Génome myc-1:34caba78… »

- l'âge est `now() − birth_ts`, jamais un chiffre décrété ;
- le sexe est une déclaration libre, jamais déduite ;
- le nom n'est **jamais** aléatoire : utilisateur ou premier Sommeil ;
- test de copie/compactage : le `self.json` survit à l'identique.

## 4. Organes externes : hiérarchie câblée, niveaux 3-4 STUBBÉS

`mycelia/organs/external.py` :

1. mémoire locale (toujours d'abord, seuil de confiance) ;
2. modèle local (Qwen, sans réseau) ;
3. Gemini — **refus explicite** tant que le détecteur d'ignorance n'existe pas :
   `« escalade bloquée : pas de détecteur d'ignorance (golden 5/5 non atteint) »` ;
4. recherche web — même condition, et le résultat est un **input à mémoriser**,
   jamais une réponse affichée brute.

Vie privée : `sanitize()` ne laisse passer qu'une question nue, bornée ; le
contexte local n'est jamais transmis. Aucun appel réel n'est effectué : le test
mocke Gemini et vérifie qu'il **n'est pas appelé** sans détecteur.

## Tests exécutés

```
tests/test_identity.py + tests/test_external.py : 8 passed
smoke 100 pas : compatible (0 nœud enterré, 50 % de zéros)
```

## Ce qui n'est pas fait

- **Le verdict du jumeau** : le run tourne, il n'a pas fini. La doctrine n'est
  ni prouvée ni réfutée à l'heure de ce rapport.
- **Plan B** (Top-KAST / gate Switch) : seulement si l'étage 1 échoue.
- **Encodeur toujours rouge** : ingestion fermée, niveaux 3-4 inertes.
- Aucun appel réseau réel (Gemini, web) n'a été effectué ; c'est volontaire.
