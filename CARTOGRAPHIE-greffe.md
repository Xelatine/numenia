# Greffe paramétrique Qwen → MYCELIA — extraction, cartographie, verdict

**Résumé : l'étape 1 est faite, l'étape 2 est conçue, l'étape 3 est ROUGE.**
La projection linéaire 896 → 32 détruit l'ordre des similarités de Qwen
(Spearman 0,38 pour un seuil consigné à 0,70). Aucun poids de MYCELIA n'a été
modifié. La greffe 1 n'est pas rejetée par principe : elle est rejetée par la
mesure, et une cartographie révisée est proposée plus bas.

## Étape 1 — extraction (faite)

`mycelia/gguf_reader.py` lit le GGUF sans dépendance (PyPI ne résout pas depuis
BIO-1) ; `scripts/graft_extract.py` écrit `data/mem_report/qwen_tensors.json`.

| | |
|---|---|
| Fichier | `qwen2.5-0.5b-instruct-q8_0.gguf`, 675 710 816 o |
| Architecture | qwen2, **24 couches**, dim **896**, 14 têtes (2 KV), FFN 4864, vocab 151 936 |
| Tenseurs | **291** — 170 Q8_0, 121 F32 |
| Embedding | `token_embd.weight` **(896, 151 936)** Q8_0, 137,9 Mo |
| Couche type | `attn_q` (896,896), `attn_k/v` (896,128), `attn_output` (896,896), `ffn_gate/up` (896,4864), `ffn_down` (4864,896), norms F32 (896,) |

Le JSON contient les 291 tenseurs avec nom, forme, type, offset et octets : la
matière première de la greffe est inventoriée, pas supposée.

## Étape 2 — cartographie (conçue, pas codée d'un coup)

### GREFFE 1 — l'embedding comme poids d'entrée

Pour un token MYCELIA `t` de chaîne `s_t` :

1. tokeniser `s_t` avec le tokenizer Qwen → ids `q_1..q_k` ;
2. `e_t = moyenne(E[q_i])`, `E` = table `token_embd` déquantifiée (896) ;
3. `P ∈ R^{32×896}` (PCA ajustée sur des phrases, ou projection aléatoire) ;
4. `w_t = normalise(P (e_t − μ)) × σ_mycelia` ;
5. écrire `w_t` dans `embedding.weight[t]` — les nœuds reçoivent le langage
   déjà projeté.

Le nœud ne change pas pour la greffe 1 : c'est son **entrée** qui change.

### GREFFE 2 — les têtes d'attention comme clusters

Une tête `h` de la couche `ℓ` a `W_q,W_k,W_v ∈ R^{64×896}`, `W_o ∈ R^{896×64}`.
Avec `R_in ∈ R^{896×32}` (PCA) et `R_out ∈ R^{64×32}` :

```
A_q = R_out^T W_q_h R_in      [32×32]
A_k = R_out^T W_k_h R_in      [32×32]
A_v = R_out^T W_v_h R_in      [32×32]
A_o = R_in^T  W_o_h R_out     [32×32]
```

Chaque tête devient **un nœud** portant quatre matrices 32×32 : 14 nœuds par
couche, **336 nœuds greffés** sur 24 couches. Évolution du nœud : un champ
`mix_w` [32×32] appliqué en résidu (`out += mix_w @ normalise(x)`) ; les arêtes
du cluster portent les scores d'attention. Coût : 4×1024 = 4 096 paramètres par
nœud greffé, soit ~1,4 M — négligeable.

### Routeur et étiquettes

Le routeur apprend à envoyer les tokens de langue vers le cluster greffé ; on
mesure la part des éveils qui y vont. Étiquettes : `[voix]` = Qwen appelé,
`[greffon]` = région greffée utilisée seule, `[mycelia]` = routeur maître.

## Étape 3 — validation : seuils consignés AVANT, verdict ROUGE

Seuils écrits dans `scripts/graft_pilot.py` avant le calcul :

- **S1** Spearman des cosinus par paires (20 phrases, 190 paires) ≥ **0,70** ;
- **S2** erreur cosinus moyenne ≤ **0,15** ;
- **S3** retrieval hit@1 ≥ **4/5**, et pas plus d'un point sous Qwen.

| Cartographie | Spearman | Err. cos | S1 | S2 | hit@1 |
|---|---|---|---|---|---|
| Projection aléatoire 32d | **0,348** | **0,114** | non | **oui** | 2/5 (Qwen 2/5) |
| PCA tokens 32d | 0,376 | 0,671 | non | non | 2/5 |
| PCA phrases 32d | 0,597 | 0,157 | non | non | 1/5 |

Courbe de dimension (mêmes seuils, même jeu) :

| Dim | Aléatoire Spearman / err | PCA-phrases Spearman / err |
|---|---|---|
| 32 | 0,348 / 0,114 | 0,597 / 0,157 |
| 64 | 0,456 / 0,088 (hit@1 **4/5**) | 0,638 / 0,134 |
| 128 | 0,663 / 0,058 | 0,677 / 0,132 |

**Aucune dimension n'atteint 0,70**, et 128d est infaisable : ~99 000 paramètres
par nœud × 10 020 ≈ **4 Go de poids** avant Adam, sur une machine de 7,3 Go.
64d coûterait ~251 M paramètres (~1 Go) avant optimiseur, déjà tendu.

### Ce que le chiffre dit vraiment

La moyenne des embeddings d'entrée n'est **pas** la représentation sémantique de
Qwen : le retrieval Qwen lui-même ne fait que **2/5** sur ces paires. Le test
mesure donc la projection, pas le comportement du modèle. Deux conclusions :

1. la greffe d'entrée à 32d **ne peut pas** « voir avec les yeux de Qwen » de
   façon fidèle : l'ordre des similarités est perdu ;
2. le critère S1 était peut-être trop dur pour une moyenne d'embeddings, mais il
   a été consigné avant — on ne le déplace pas après.

## Cartographie révisée (proposée, non implémentée)

| Piste | Principe | Coût / risque |
|---|---|---|
| **A. Encodeur partagé distillé** | MLP 896→32 appris pour reproduire les similarités de Qwen, au lieu d'une projection linéaire | petit (~30 k paramètres), mais ce n'est plus « transplanter les poids » |
| **B. Dim 64, nœuds réduits** | 64d atteint hit@1 4/5 mais Spearman 0,46 ; il faudrait ~2 500 nœuds pour tenir la RAM | capacité ×2,5 au lieu de ×10 |
| **C. Qwen comme routeur, pas comme perception** | l'embedding de question sert au routage sémantique (EmbeddingGemma est vert) ; les nœuds gardent leur propre entrée | respecte la matière, ne greffe rien |
| **D. GREFFE 2 seule** | têtes d'attention → clusters, sans greffer l'embedding | même mur de dimension, à mesurer |

**Recommandation mesurée** : ne pas écrire la greffe 1 dans le cerveau. Faire
un pilote A (encodeur distillé) avec le même protocole S1/S2/S3, et en parallèle
C (routage par similarité de question), qui est la seule piste déjà verte
(EmbeddingGemma 5/5). Si A échoue aussi, la greffe paramétrique est consignée
comme non viable à cette dimension — pas contournée.

## Ce qui n'a pas été fait

- aucun poids de MYCELIA modifié ;
- GREFFE 2 non codée (elle attend une dimension qui tienne) ;
- étiquettes `[greffon]` / `[mycelia]` non activées : `[voix]` reste l'appel Qwen ;
- aucun test de comportement (top-1) : impossible sans GREFFE 2.
