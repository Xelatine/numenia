# Levier 2 — checkpointing par tour : mesuré, refusé

**6 novembre 2026.** Consigne : implémenter le recalcul du tour de diffusion au backward,
mesurer au banc à 2 200 arêtes, **s'arrêter et rapporter si le gain est inférieur à ÷2**.
Gain mesuré : **÷1,32**. Le levier est refusé, le code est retiré du dépôt, et le run
texte n'est **pas** lancé.

## 1. Verdict en une phrase

Le checkpointing par tour est **exact** (mêmes logits, mêmes gradients à moins de 1e-6)
mais **pas rentable** : il divise le pic de VRAM par 1,32 pour un coût de +22 % en temps,
très loin du ÷2 qui était la condition d'adoption.

## 2. Ce qui a été implémenté, puis retiré

Implémentation testée : `torch.utils.checkpoint.checkpoint(..., use_reentrant=False)`
autour de `bank.forward_rows(x, hot)` dans la boucle des tours de `Brain.forward`, armée
par un paramètre `grad_checkpoint` (défaut `False`), exposée dans la config (`brain`),
dans `train.py` et par `--hop-checkpoint / --no-hop-checkpoint` dans
`scripts/bench_step.py`. Le garde exigeait `self.training`, le grad activé et
`x.requires_grad`, et le checkpoint est posé **sans** traverser la décision de naissance
(qui reste hors graphe, D1.1). `forward_rows` ne contient ni dropout ni bruit : le
recalcul est la même arithmétique, donc `test_resume_continuity` n'était pas en danger.

Tout cela a été **retiré** après la mesure (§5) : aucun chemin mort, aucune option
fantôme, `git status` propre.

## 3. Mesures au banc — processus neufs, 2 200 arêtes

`python scripts/bench_step.py --batch 16 --seq 10 --edges 2200 --only step [--hop-checkpoint]`
(GPU libre, toy v5 terminé.)

| régime | ms/pas (médiane, 5 mesures) | VRAM réservée | commit | RAM hôte |
|---|---:|---:|---:|---:|
| sans checkpointing | **1 464** [1 462–1 486] | **2 930 Mo** | 5 894 Mo | 1 670 Mo |
| avec checkpointing | **1 783** [1 744–1 829] | **2 280 Mo** | 5 200 Mo | 1 661 Mo |
| rapport | ×1,22 (plus lent) | **÷1,29** | −694 Mo | −9 Mo |

Mesure indépendante, dans le test (pic *alloué*, `empty_cache()` et
`reset_peak_memory_stats()` entre les deux passes, graine identique, graphe identique) :

```
sans 2 782 Mo → avec 2 107 Mo   (÷1,32, attendu ≥ ÷2)
```

<details>
<summary>Sorties brutes du banc</summary>

```
device cuda | batch 16 | seq 10 | arêtes 2200 | nœuds 512 | dim 32 | hops 8 | tbptt 16 | ckpt False | data toy
step_once : médiane    1464 ms/pas [1462–1486] | RAM 1670 Mo (pic 1670, commit 5894, +16/pas) | GPU 70/2930 Mo

device cuda | batch 16 | seq 10 | arêtes 2200 | nœuds 512 | dim 32 | hops 8 | tbptt 16 | ckpt True | data toy
step_once : médiane    1783 ms/pas [1744–1829] | RAM 1661 Mo (pic 1661, commit 5200, +20/pas) | GPU 70/2280 Mo
```

</details>

## 4. Équivalence : le levier est bon, il n'est pas rentable

Test sur 400 arêtes, même graphe, seule la bascule change : logits identiques à moins de
1e-6 et **tous** les gradients identiques à moins de 1e-6 (test vert avant retrait). Le
levier ne dégrade donc rien ; il ne rapporte simplement pas ce qu'il promettait.

## 5. Pourquoi le gain est faible — ce qui est mesuré, ce qui est une hypothèse

**Mesuré** : avec 8 tours, si les intérieurs de tour étaient le poste dominant, le gain
serait proche de ÷8. Il est de ÷1,3, donc au plus ~15 % du pic viennent des intérieurs de
tour recalculés ; ~85 % du pic restent ailleurs.

**Hypothèse, non instrumentée** : le reste est le graphe du pas complet — logits de toute
la séquence, pénalités (identité, redondance) et état du dernier tour. Je ne l'affirme
pas : le mesurer demanderait une sonde par poste, que la consigne ne demandait pas et que
je n'ai pas posée.

## 6. Étape 0 — le run v5 a fini proprement, et le critère (g) est tenu sans levier

Run `logs_toy_v5_2000_batch16.txt`, 2 000 pas de zéro, batch 16, 5 273 s :

```
step 2000 | perte 0.0004 | arêtes 5433 | nœuds 520 | RAM 1783 Mo (commit 8928)
          | GPU 106/5702 Mo | 2934 ms/step
chemin : 0 -> 2 -> 1
```

- **critère (g) : VRAM 5 702 Mo < 6 Go — tenu**, sans aucun levier de mémoire ;
- fin propre (checkpoint final écrit, verrou relâché, pas d'OOM, pas de mort WDDM) ;
- la prédiction du §8 du diagnostic — « mur vers 8 000 arêtes », « mort entre les pas
  1 400 et 1 700 » — est **fausse** : l'élagage a stabilisé la table vers 5 400 arêtes et
  le run est allé au bout. C'est corrigé au §9 du diagnostic.

## 7. Étape 4 — run texte : NON lancé

La condition était « gain ≥ ÷2 **et** toy fini proprement ». Le toy est fini proprement,
le gain est de ÷1,32 : la condition n'est pas remplie, donc
`python mycelia/train.py --data text --scale 0` n'a pas été lancé. Les deux options
restantes sont à discuter avec l'humain, aucune n'est engagée ici :

1. **tbptt en repli** (16 → 8 ou 4) : déjà mesuré au banc le 5 novembre — `tbptt 4` à
   2 200 arêtes donne 2 186 ms/pas pour **1 240 Mo** de VRAM, soit un ordre de grandeur
   au-dessus de ce que le checkpointing apporte, au prix de la mémoire longue ;
2. **accepter le régime v5** tel quel : il tient 6 Go sur le toy, mais le texte est une
   autre pente (séquences de 128, 1 200 arêtes).

## 8. État du dépôt après ce rapport

`git status` propre, `pytest tests` → **41/41** en 67 s. Aucun fichier de code modifié par
rapport au commit `4a6f673` ; ce rapport, `docs/PLATEAU-croissance.md` et le §9 du
diagnostic sont les seuls ajouts.
