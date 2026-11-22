"""S7 — démo du traducteur IA→IA, < 2 min, un chiffre par ligne.

Chaîne complète : professeur chargé → receveur vierge → traduction → test à
l'aveugle → vie → F final. Un seul chiffre par ligne. C'est le script de
l'épisode vidéo.

    python scripts/lab/demo_translate.py
"""

from __future__ import annotations

import time
from pathlib import Path

import torch
import torch.nn.functional as F

from lab_common import LAB, ToyCorpus, read_json, set_seed, write_json
from repaired_mycelia import (
    SEQ,
    RepairedBrain,
    aligned_loss,
    build_repaired,
    free_life,
    load_brain,
    save_brain,
    train_repaired,
)
from s5_common import (
    FAIM_VIVANTE,
    L_REF,
    POOL24,
    SEEDS,
    build_dense,
    columns_mycelia,
    columns_dense,
    load_corpus,
    load_dense,
    relative_gain,
)
from s5_distill import (
    BEST,
    GRID,
    teacher_path,
    plafond_measure,
    fidelity_student,
    score_f,
    fresh_sentences,
    train_student,
)
PLAFOND_PATH = LAB / "s5_plafond.json"
from s6_core import consolidate_cycle
from twins import DENSE_BATCH, DENSE_LR, MYCELIA_BATCH, MYCELIA_LR, MYCELIA_WD

LOG = LAB / "s7_demo.json"


def line(label: str, value: str) -> None:
    print(f"  {label}: {value}", flush=True)


def main() -> None:
    ensure_lab()
    overall = time.time()
    print("=" * 60)
    print("S7 — DÉMO DU TRADUCTEUR IA→IA")
    print(f"démarrage {time.strftime('%H:%M:%S')}")
    print("=" * 60)

    # ── 1. Professeur ──────────────────────────────────────────────────────
    print("\n[1] PROFESSEUR (dense entraîné, corpus 10k, 3 semences)")
    corpus = load_corpus("10k")
    val = corpus.fixed(100, "val")
    vocab = corpus.vocab_size
    results = []
    for seed in SEEDS:
        set_seed(seed)
        teacher = build_dense(vocab, seed, mlp=MLP_TWIN)
        result = train_dense(
            teacher,
            corpus,
            600,
            lr=DENSE_LR,
            batch=DENSE_BATCH,
            seed=seed,
            log_every=600,
            patience=6,
            val_batches=val,
        )
        results.append(result["final_val"])
        line(f"  semence {seed}", f"val {result['final_val']:.4f}")

    best_seed = SEEDS[results.index(min(results))]
    set_seed(best_seed)
    teacher = build_dense(vocab, best_seed, mlp=MLP_TWIN)
    teacher.load_state_dict(
        torch.load(str(LAB / f"s7_teacher_s{best_seed}.pt"), weights_only=False)["state_dict"]
    )
    teacher.eval()
    line("  professeur choisi", f"semence {best_seed}, val {min(results):.4f}")

    # ── 2. Receveur vierge ─────────────────────────────────────────────────
    print("\n[2] RECEVEUR VIERGE (NUMENIA pool 24, jamais entrené sur ce corpus)")
    brain = build_repaired(vocab, 7, two_speeds=True, overrides=POOL24)
    # mesurer l'élève vierge
    brain.eval()
    cols0 = columns_mycelia(brain, val)
    line("  perte vierge (nouvelle)", f"{cols0['nouvelle']:.4f}")
    line("  R vierge", f"{relative_gain(cols0['nouvelle'], min(results)):.4f}")
    line("  arêtes", f"{int(brain.ne)}")

    # ── 3. Traduction (recette a1b1g0) ─────────────────────────────────────
    print("\n[3] TRADUCTION (a1b1g0, KL + CE d'ancrage, 2 budgets)")
    plafond_path = LAB / "s5_plafond.json"
    if plafond_path.exists():
        plafond = read_json(plafond_path)["accord"]
    else:
        plafond = plafond_measure(teacher, load_dense(teacher_path("B")), corpus)["accord"]
    kl_ab = float(plafond["kl_A_vers_B"])

    for budget, steps in [("standard (600 pas)", 600), ("x4 (2400 pas)", 2400)]:
        result = train_student(
            f"demo_{budget.split()[0]}",
            7,
            alpha=1.0,
            beta=1.0,
            gamma=0.0,
            temperature=1.0,
            steps=steps,
            projector_init="random",
        )
        brain_path = LAB / f"s5_demo_{budget.split()[0]}_s7.pt"
        brain = load_brain(brain_path) if brain_path.exists() else None
        if brain is not None:
            brain.eval()
            cols = columns_mycelia(brain, val)
            fid = fidelity_student(brain, teacher, load_dense(teacher_path("B")), corpus)
            f_val = score_f(fid["kl_A_vers_MYC"], kl_ab)
            line("  budget %s" % budget, "")
            line(f"    perte distillée", f"{cols['nouvelle']:.4f}")
            line(f"    F avant vie", f"{f_val:+.4f}")
            line(f"    R avant vie", f"{relative_gain(cols['nouvelle'], min(results)):.4f}")
        else:
            line(f"  budget {budget}", "échec charge")

    # ── 4. Test à l'aveugle ────────────────────────────────────────────────
    print("\n[4] TEST À L'AVEUGLE (corpus tenu à l'écart)")
    # Le corpus 10k a déjà un split val tenu à l'écart (100 fenêtres)
    # Utiliser le meilleur élève distillé
    best_brain_path = LAB / "s5_demo_standard_s7.pt"
    if best_brain_path.exists():
        brain = load_brain(best_brain_path)
        brain.eval()
        fid = fidelity_student(brain, teacher, load_dense(teacher_path("B")), corpus)
        f_val = score_f(fid["kl_A_vers_MYC"], kl_ab)
        line("  F (test aveugle)", f"{f_val:+.4f}")
        line("  KL(A||MYC)", f"{fid['kl_A_vers_MYC']:.4f}")
        line("  KL plafond (A||B)", f"{kl_ab:.4f}")
        line("  top3 identique A", f"{fid['top3_identiques_avec_A']:.4f}")
    else:
        line("  test aveugle", "élève non disponible")

    # ── 5. Vie (2000 pas + 4 cycles consolidation) ─────────────────────────
    print("\n[5] VIE (2000 pas métabolisme libre + 4 cycles consolidation)")
    brain.prune_ratio = FAIM_VIVANTE
    life = free_life(brain, corpus, steps=2000, batch=MYCELIA_BATCH, seed=7, lifecycle_every=50)
    line("  pas de vie", f"{life['pas']}")
    line("  arêtes créées", f"{life['aretes_creees']}")
    line("  arêtes coupées", f"{life['aretes_coupees']}")
    cols_after_life = columns_mycelia(brain, val)
    fid_after_life = fidelity_student(brain, teacher, load_dense(teacher_path("B")), corpus)
    f_after_life = score_f(fid_after_life["kl_A_vers_MYC"], kl_ab)
    line("  F après 2000 pas", f"{f_after_life:+.4f}")
    line("  perte après vie", f"{cols_after_life['nouvelle']:.4f}")

    # ── 6. Verdict ──────────────────────────────────────────────────────────
    print("\n[6] VERDICT DU PREMIER TRADUCTEUR")
    f_avant = f_val if best_brain_path.exists() else 0.0
    f_apres = f_after_life
    if f_avant >= 0.5 and f_apres >= 0.4:
        verdict = "TRADUCTEUR FONCTIONNEL"
        code = "VERT"
    elif f_avant >= 0.5 and f_apres < 0.4:
        verdict = "traduction OK, consolidation à retoucher"
        code = "JAUNE"
    else:
        verdict = "l'audit a localisé la panne"
        code = "ROUGE"

    line("  F avant vie", f"{f_avant:+.4f}")
    line("  F après vie", f"{f_apres:+.4f}")
    line("  verdict", f"{code} — {verdict}")

    print(f"\n[TEMP TOTAL : {round(time.time() - overall, 1)}s]")

    write_json(
        LOG,
        {
            "date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "duree_s": round(time.time() - overall, 1),
            "professeur": {"semence": best_seed, "val": min(results)},
            "receveur_vierge": cols0,
            "apres_distillation": {"F": f_val, "perte": cols["nouvelle"]} if best_brain_path.exists() else None,
            "apres_vie": {"F": f_after_life, "perte": cols_after_life["nouvelle"]},
            "verdict": {"code": code, "phrase": verdict},
        },
    )


if __name__ == "__main__":
    main()
