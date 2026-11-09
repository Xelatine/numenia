# Transition vers la machine cible (GTX 860M, ~2 Go) — checklist

**Rien ici n'a été mesuré sur la 860M** : elle n'est pas disponible. Tout ce qui
pouvait être établi sur la 3070 l'a été le 9 novembre (session F) ; le reste est
étiqueté [estimé] ou [supposé] et **doit être vérifié sur place** avant tout long
run. La transition se fait en deux temps : cette préparation, puis une **courte
session de validation sur place** (checklist ci-dessous + 20 pas avec le candidat
retenu). Découvrir une incompatibilité PyTorch/Maxwell après trois jours serait
la pire façon de l'apprendre.

---

## 1. À vérifier sur place, dans cet ordre (bloquant)

| # | point | commande / action | attendu | statut |
|---:|---|---|---|---|
| 1 | la carte est bien en **compute capability 5.0** | `nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version --format=csv` | `5.0`, ~2 048 MiB | à faire |
| 2 | **pilote** : dernière branche supportant Maxwell | `nvidia-smi` ; la branche **R580** est la dernière à couvrir Maxwell/Pascal (NVIDIA, 2025) [supposé — à confirmer], et c'est elle qui permet un runtime CUDA 12.x | `≥ 580.xx` | à faire |
| 3 | **PyTorch** : `2.14.0` + **build CUDA 12.6** | `pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cu126` | `torch.version.cuda == '12.6'`, `torch.cuda.is_available()` vrai | à faire |
| 4 | le noyau sm_50 est réellement dans le wheel | `python -c "import torch;print(torch.cuda.get_arch_list())"` | `sm_50` présent dans la liste | à faire |
| 5 | **suite de tests complète** sur cette version | `python -m pytest -q` | 55 passed (2 skippés : les tests GPU-only seuls s'exécutent) | à faire |
| 6 | RAM système libre (offload éventuel) | `wmic OS get TotalVisibleMemorySize` ou le Gestionnaire des tâches | ≥ 8 Go | à faire |
| 7 | **PCIe** : micro-bench de copie pinned | voir §4 | noter Go/s | à faire |
| 8 | température de repos et sous charge | `nvidia-smi --query-gpu=temperature.gpu,power.draw --format=csv -l 1` | < 90 °C | à faire |

### Pourquoi 2.14.0 exactement

`torch==2.14.0` est **la dernière release dont un wheel publié compile encore
pour Maxwell (sm_50)**. Trois dates le fixent :

- les wheels **cu128** ont supprimé Maxwell/Pascal dès **2.8** [doc PyTorch] ;
- **CUDA 13.x** ne descend plus sous Turing (sm_75) ; le build PyPI **par défaut**
  de 2.14 est en CUDA 13.0 — il **ne voit pas** la 860M ;
- le wheel **cu126** de 2.14 contient `5.0;6.0;6.1;7.0;7.5;8.0;8.6;9.0`, et
  c'est le dernier : **2.15 supprime le build CUDA 12.6** (annonce dev-discuss
  PyTorch, septembre 2026).

D'où l'épinglage dans `requirements.txt`, avec l'index à utiliser. **Cette
machine-ci (RTX 3070) tourne toujours sur 2.11.0+cu128** : ne pas réinstaller ici.

> **Non vérifié dans cette session** : le téléchargement du wheel cu126 a échoué
> (`download-r2.pytorch.org` ne résout pas depuis cette machine). Le code n'a donc
> **pas** été exécuté sous 2.14.0 ; la vérification API reste à faire sur place
> (point 5). Ce qui *a* été vérifié ici : le code ne dépend d'aucune API retirée
> entre 2.11 et 2.14 connue de nous, et `torch.utils.checkpoint(use_reentrant=
> False, preserve_rng_state=True)` existe depuis 2.0.

---

## 2. Candidats pré-dimensionnés pour ~1,8 Go utiles

Pic et débit **estimés** à partir des lois mesurées le 9 novembre, jamais d'une
mesure sur la cible. Le pic avec checkpointing vient de la mesure de session F
(532 nœuds, `seq 32`, `batch 8`, 3 000 arêtes, `tbptt 16` : **312 Mo** [mesuré]),
ramené au nombre d'arêtes par proportionnalité [estimé].

| candidat | nœuds | arêtes | tbptt | seq × batch | pic estimé | s/pas estimé (3070) | s/pas estimé (860M) | ETA 3 000 pas |
|---|---:|---:|---:|---|---:|---:|---:|---:|
| **A** — checkpointing ON | 532 | 1 000 | 16 | 32 × 8 | **~230 Mo** | ~6,0 | ~50–90 | **1,7–3,1 jours** |
| **B** — sans checkpointing | 532 | 1 000 | 4 | 32 × 8 | ~140 Mo | ~3,0 | ~25–45 | 0,9–1,6 jour |
| **C** — petite colonie | 250 | 750 | 4 | 32 × 8 | ~50 Mo | ~1,5 | ~12–22 | 0,4–0,8 jour |
| **D** — maximal sûr | 532 | 2 000 | 8 | 32 × 8 | ~380 Mo | ~7,0 | ~60–105 | 2,1–3,6 jours |

Ratios utilisés : mémoire ∝ arêtes à `tbptt` fixe [mesuré] ; `tbptt 16 → 4` divise
le pic par ~2,9 [mesuré] ; le checkpointing coûte **×1,45 à ×2,6** en temps selon
la longueur de séquence [mesuré] ; la 860M est prise à **÷8 à ÷15** du débit de
la 3070 pour ce type de calcul (petits `baddbmm`, donc dominé par les lancements
et les 640 cœurs contre 5 120) [supposé — **à mesurer sur place avec 20 pas**, et
c'est le point 5 de la session de validation].

**Capacité** : E\* ≈ N × 7,5 arêtes, donc 532 nœuds ≈ 4 000 arêtes de capacité
biologique et 250 nœuds ≈ 1 875. Le candidat **C** ne « fait » donc pas la même
colonie que les ablations à 532 : c'est un compromis assumé, pas une équivalence.

**Le levier à préférer** : à arêtes fixes, une **petite colonie coûte plus cher**
(3 230 Mo contre 2 419 à 2 000 arêtes et 532 nœuds, mesuré le 9 novembre : moins
de nœuds ⇒ plus d'arêtes chaudes par tour ⇒ un front d'onde plus épais). Pour
tenir dans 2 Go, réduire **d'abord `tbptt`, `seq` et le batch, et activer le
checkpointing** ; réduire la population en dernier, et en connaissant le coût.

---

## 3. Le jour J — ordre d'exécution

```bash
# 1. environnement
python -m venv .venv && .venv/Scripts/activate
pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt          # le reste : numpy, tokenizers, fastapi…
python -c "import torch;print(torch.__version__, torch.version.cuda, torch.cuda.get_arch_list())"

# 2. la suite, sans GPU d'abord (le CPU suffit à prouver l'API)
set CUDA_VISIBLE_DEVICES=-1 && python -m pytest -q

# 3. 20 pas avec le candidat retenu, en regardant le pic mémoire
python mycelia/train.py --data text --steps 20 --checkpoint-turns \
    --batch-size 8 --seq-len 32 --batch-size 8 --report-every 5

# 4. si les 20 pas tiennent : le run long, avec reprise toutes les 250 pas
python mycelia/train.py --data text --steps 3000 --checkpoint-turns --save-every 250
```

Le drapeau `--checkpoint-turns` est **OFF par défaut** : il échange du temps
contre de la mémoire, et la session F a mesuré ce prix. `--save-every 250` passe
par le `--resume` existant, dont la continuité à 1e-6 est prouvée (et re-prouvée
avec le drapeau ON par `test_checkpoint_resume_continuity`).

---

## 4. Micro-bench PCIe (à lancer sur place, 30 s)

```python
# python pcie.py  — copie pinned hôte -> GPU et retour, 512 Mo
import torch, time
n = 512 * 1024 * 1024 // 4
host = torch.empty(n, dtype=torch.float32, pin_memory=True).uniform_()
device = torch.empty(n, dtype=torch.float32, device="cuda")
torch.cuda.synchronize()
t = time.perf_counter()
for _ in range(5):
    device.copy_(host)
torch.cuda.synchronize()
up = 5 * 512 / (time.perf_counter() - t)
t = time.perf_counter()
for _ in range(5):
    host.copy_(device)
torch.cuda.synchronize()
down = 5 * 512 / (time.perf_counter() - t)
print(f"{up:.2f} Go/s hôte→GPU, {down:.2f} Go/s GPU→hôte")
```

Un laptop de cette génération est en **PCIe 3.0 x16 au mieux, souvent x8 ou
Gen 2** [supposé] : c'est ce chiffre, et lui seul, qui dira si un offload
`save_on_cpu` (levier c du menu) est viable. Tant qu'il n'est pas mesuré, aucun
chiffrage d'offload n'est publié.

---

## 5. Longue durée (le vrai sujet)

- **Reprise** : `--save-every 250` (4 sauvegardes par 1 000 pas) ; le genome
  contient poids, cimetière, moments AdamW, compteurs et **état RNG complet**.
  Testé cette session : `test_longrun_resume` (trois crashes simulés, 30 pas
  identiques à 1e-6, compteur et RNG repris à l'identique).
- **Journal d'exécution** : sur la cible, une ligne par heure dans
  `logs_target.csv` —
  `date,step,loss,edges,nodes,gpu_used_mib,gpu_free_mib,gpu_temp_c,ram_used_mb,commit_mb`.
  La dérive à surveiller est celle du **commit**, pas du working set : sur WDDM
  c'est lui qui tue un run sans traceback (le 8 novembre, une config à 4,2 Go de
  pic est morte sur une carte où il restait 6,1 Go).
  *Non simulé accéléré cette session* : la 3070 a passé son temps à mesurer les
  deux leviers. À faire pendant la session de validation sur place, où 20 pas
  suffisent à voir si la mémoire est plate.
- **ETA** : voir le tableau §2. Le candidat **A** est le seul à dépasser deux
  jours ; **si l'ETA dépasse 4 jours, c'est une décision humaine, pas un
  paramètre** — le noter et choisir explicitement entre durée, taille de colonie
  et `tbptt`.
- **Ce qu'il ne faut pas faire** : lancer 3 000 pas sans avoir mesuré 20 pas sur
  place. La loi mémoire est fiable à ±4 % *sur la 3070* ; rien ne garantit que
  l'allocateur CUDA 12.6 sur Maxwell se comporte à l'identique (pas de
  `expandable_segments` utile en 2 Go, fragmentation plus sévère) [supposé].
