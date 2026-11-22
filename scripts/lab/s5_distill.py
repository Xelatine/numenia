"""S5-MAX Verrous 3 et 4 — la distillation ancree et le plafond physique.

D1. Deux professeurs denses **independants** (graines 1 et 2), a la taille du
    jumeau equitable du pool 24 (mlp 768, 63 480 parametres). C'est le fondement
    du plafond : deux professeurs ne predisent pas la meme chose, et un eleve ne
    peut pas etre plus fidele que la realite ne le permet.

D2. La formule complete, les deux correctifs du diagnostic S4 :

        L = alpha * T^2 * KL(p_T(MYC) || p_T(prof))
          + beta  * CE(logits_MYC, tokens_reels)   <- l'ancrage
          + gamma * MSE(P h_MYC, h_prof)           <- P aleatoire, appris

    Grille courte (4 configs x 2 semences), choix sur la perte de VALIDATION
    (colonne nouvelle).

D3. Controles : M0 seul (meme budget, memes lots) et professeur a etiquettes
    melangees. Gain isole = distille - M0.

D4. Survie apres transfert : 1000 pas de metabolisme libre (regime de faim
    vivante du verrou 2), R et F avant / apres.

P1-P3. Le plafond : accord inter-donneurs KL(A||B) et top-3 croise sur ~5 400
    contextes = le maximum atteignable ; fidelite de NUMENIA mesuree pareil ;
    score **F = 1 - KL(A||MYC) / KL(A||B)**, rapporte avant et apres la vie.

P4. Si F < 0,9 : budget x4, temperatures T=2 et T=4, alpha/beta raffines autour
    du meilleur point. Chaque levier est une bras separe.

    python scripts/lab/s5_distill.py --arm teacherB
    python scripts/lab/s5_distill.py --arm grid:a1b1g05:1
    python scripts/lab/s5_distill.py --arm melange:2
    python scripts/lab/s5_distill.py --arm metriques
    python scripts/lab/s5_distill.py --arm levier:T2
    python scripts/lab/s5_distill.py --arm survie:1
    python scripts/lab/s5_distill.py --arm merge
"""

from __future__ import annotations

import argparse
import time

import torch
import torch.nn.functional as F

from lab_common import LAB, ensure_lab, read_json, write_json
from micro_dense import train_dense
from repaired_mycelia import (
    SEQ,
    build_repaired,
    free_life,
    load_brain,
    save_brain,
    train_repaired,
)
from s4_distill import fresh_sentences, top3_mycelia
from s5_common import (
    FAIM_VIVANTE,
    L_REF,
    MLP_TWIN,
    POOL24,
    SEEDS,
    build_dense,
    columns_mycelia,
    kl_direction,
    load_corpus,
    load_dense,
    mean_sd,
    relative_gain,
    save_dense,
    topk_overlap,
)
from twins import DENSE_BATCH, DENSE_LR, MYCELIA_BATCH, MYCELIA_LR, MYCELIA_WD

VAL_WINDOWS = 58
FREE_STEPS = 1000
FID_BATCHES = 12
FID_WINDOWS = 64
GRID = {
    "a1b1g05": (1.0, 1.0, 0.5),
    "a1b1g0": (1.0, 1.0, 0.0),
    "a1b05g05": (1.0, 0.5, 0.5),
    "a05b1g05": (0.5, 1.0, 0.5),
}
GRID_SEEDS = (1, 2)
# Choix protocolaire : la meilleure perte de VALIDATION de la grille (3,528 pour
# a1b1g0 contre 3,547 pour a1b1g05, soit 0,019 nat — au bord de la dispersion des
# semences, mais c'est la regle annoncee). Le terme d'alignement (gamma) gagne
# donc zero : une fois le corpus ancre, le projecteur appris n'aide plus.
BEST = "a1b1g0"


def fragment(name: str):
    return LAB / f"s5_{name}.json"


def teacher_path(name: str):
    return LAB / f"s5_teacher{name}.pt"


# ---------------------------------------------------------------------------
# D1 + P1 : les deux professeurs et l'accord inter-donneurs
# ---------------------------------------------------------------------------
def arm_teacherB() -> None:
    corpus = load_corpus("1000")
    val = corpus.fixed(VAL_WINDOWS, "val")
    model = build_dense(corpus.vocab_size, 2, mlp=MLP_TWIN)
    result = train_dense(
        model, corpus, 600, lr=DENSE_LR, batch=DENSE_BATCH, seed=2,
        log_every=100, patience=0, val_batches=val,
    )
    save_dense(model, teacher_path("B"), {"val": result["final_val"], "seed": 2})
    print(f"professeur B : val {result['final_val']:.4f} — {result['seconds']}s", flush=True)


@torch.no_grad()
def plafond_measure(a, b, corpus) -> dict:
    """Accord inter-donneurs sur ~5 400 contextes (positions 1..7)."""
    kl_ab = 0.0
    kl_ba = 0.0
    kl_sym = 0.0
    identical = 0.0
    cover = 0.0
    rows = 0
    generator = torch.Generator().manual_seed(4242)
    for _ in range(FID_BATCHES):
        x, _y = corpus.batch(FID_WINDOWS, generator=generator)
        _h1, l1 = a.parts(x)
        _h2, l2 = b.parts(x)
        l1, l2 = l1[:, :-1], l2[:, :-1]
        kl_ab += kl_direction(l1, l2)
        kl_ba += kl_direction(l2, l1)
        kl_sym += (kl_direction(l1, l2) + kl_direction(l2, l1)) / 2.0
        flat1 = l1.reshape(-1, l1.shape[-1])
        flat2 = l2.reshape(-1, l2.shape[-1])
        overlap = topk_overlap(flat1, flat2)
        identical += overlap["identiques"]
        cover += overlap["recouvrement"]
        rows += 1
    return {
        "contextes": int(rows * FID_WINDOWS * 7),
        "kl_A_vers_B": round(kl_ab / max(1, rows), 4),
        "kl_B_vers_A": round(kl_ba / max(1, rows), 4),
        "kl_symetrique": round(kl_sym / max(1, rows), 4),
        "top3_identiques": round(identical / max(1, rows), 4),
        "top3_recouvrement": round(cover / max(1, rows), 4),
        "top3_uniforme": round(3.0 / corpus.vocab_size, 4),
    }


def arm_plafond() -> None:
    corpus = load_corpus("1000")
    a = load_dense(teacher_path("A"))
    b = load_dense(teacher_path("B"))
    block = plafond_measure(a, b, corpus)
    twin = read_json(fragment("twin")) if fragment("twin").exists() else {}
    payload = {
        "professeurs": {
            "A": {"graine": 1, "mlp": MLP_TWIN, "parametres": 12_792 + 66 * MLP_TWIN},
            "B": {"graine": 2, "mlp": MLP_TWIN, "parametres": 12_792 + 66 * MLP_TWIN},
        },
        "accord": block,
        "definition": "KL comportementale entre deux professeurs independants : "
                      "c'est le plafond theorique de la fidelite d'un eleve",
        "jumeau": twin.get("dense"),
    }
    write_json(fragment("plafond"), payload)
    print(f"accord inter-donneurs : KL(A||B) {block['kl_A_vers_B']:.4f}, "
          f"KL(B||A) {block['kl_B_vers_A']:.4f}, symetrique "
          f"{block['kl_symetrique']:.4f} ; top-3 identiques "
          f"{block['top3_identiques']:.4f}, recouvrement "
          f"{block['top3_recouvrement']:.4f} sur {block['contextes']} contextes", flush=True)


# ---------------------------------------------------------------------------
# D2 : la grille
# ---------------------------------------------------------------------------
def train_student(name: str, seed: int, alpha: float, beta: float, gamma: float,
                  temperature: float = 1.0, steps: int = 600, permuted: bool = False,
                  projector_init: str = "random") -> dict:
    corpus = load_corpus("1000")
    val = corpus.fixed(VAL_WINDOWS, "val")
    teacher = load_dense(teacher_path("A"))
    perm = None
    if permuted:
        perm = torch.randperm(
            corpus.vocab_size, generator=torch.Generator().manual_seed(77)
        )
    brain = build_repaired(
        corpus.vocab_size, int(seed), two_speeds=True, overrides=POOL24
    )
    started = time.time()
    result = train_repaired(
        brain, corpus, steps, lr=MYCELIA_LR, batch=MYCELIA_BATCH, seed=int(seed),
        log_every=100, weight_decay=MYCELIA_WD, val_batches=val, lifecycle_every=100,
        teacher=teacher, teacher_perm=perm, alpha=alpha, beta=beta, gamma=gamma,
        temperature=temperature, projector_init=projector_init,
    )
    brain = result["brain"]
    save_brain(
        LAB / f"s5_{name}_s{seed}.pt", brain, step=result["best_step"],
        extra={"alpha": alpha, "beta": beta, "gamma": gamma, "T": temperature,
               "steps": steps, "permute": permuted},
    )
    cols = columns_mycelia(brain, val)
    row = {
        "bras": name,
        "semence": int(seed),
        "alpha": float(alpha),
        "beta": float(beta),
        "gamma": float(gamma),
        "temperature": float(temperature),
        "pas": int(steps),
        "projecteur": projector_init,
        "etiquettes_melangees": bool(permuted),
        **cols,
        "honnete_0_7": result["best_val"],
        "pas_meilleur": int(result["best_step"]),
        "positions": result["per_position"],
        "aretes": int(result["edges"]),
        "aretes_creees": int(result["births"]),
        "aretes_coupees": int(result["edge_deaths"]),
        "secondes": round(time.time() - started, 1),
        "courbe": [
            {"step": r["step"], "val": r["val"], "kl": r["kl"], "ce": r["ce"],
             "align": r["align"], "aretes": r["edges"]}
            for r in result["curve"]
        ],
    }
    write_json(fragment(f"{name}_s{seed}"), row)
    print(f"{name} s{seed} : nouvelle {cols['nouvelle']:.4f} / ancienne "
          f"{cols['ancienne']:.4f} au pas {row['pas_meilleur']} ; dernier KL/CE/align "
          f"{row['courbe'][-1]['kl']}/{row['courbe'][-1]['ce']}/"
          f"{row['courbe'][-1]['align']} — {row['secondes']}s", flush=True)
    return row


# ---------------------------------------------------------------------------
# P2/P3 : fidelite et score F
# ---------------------------------------------------------------------------
@torch.no_grad()
def fidelity_student(brain, teacher_a, teacher_b, corpus) -> dict:
    """KL(A||MYC), KL(B||MYC), top-3 et la perte, sur les memes contextes."""
    tot_a = 0.0
    tot_b = 0.0
    top_a = 0.0
    cover = 0.0
    rows = 0
    generator = torch.Generator().manual_seed(4242)
    for _ in range(FID_BATCHES):
        x, _y = corpus.batch(FID_WINDOWS, generator=generator)
        logits = brain(x, grow=False, trace=False)[0][:, 1:]
        _ha, la = teacher_a.parts(x)
        _hb, lb = teacher_b.parts(x)
        la, lb = la[:, :-1], lb[:, :-1]
        tot_a += kl_direction(la, logits)
        tot_b += kl_direction(lb, logits)
        flat = logits.reshape(-1, logits.shape[-1])
        flata = la.reshape(-1, la.shape[-1])
        overlap = topk_overlap(flata, flat)
        top_a += overlap["identiques"]
        cover += overlap["recouvrement"]
        rows += 1
    return {
        "kl_A_vers_MYC": round(tot_a / max(1, rows), 4),
        "kl_B_vers_MYC": round(tot_b / max(1, rows), 4),
        "top3_identiques_avec_A": round(top_a / max(1, rows), 4),
        "top3_recouvrement_avec_A": round(cover / max(1, rows), 4),
        "contextes": int(rows * FID_WINDOWS * 7),
    }


def score_f(kl_student: float, kl_plafond: float) -> float:
    """F = 1 - KL(A||MYC) / KL(A||B) : 1 = le plafond physique, 0 = aucun transfert."""
    if abs(float(kl_plafond)) < 1e-9:
        return 0.0
    return round(1.0 - float(kl_student) / float(kl_plafond), 4)


def arm_metriques() -> None:
    corpus = load_corpus("1000")
    val = corpus.fixed(VAL_WINDOWS, "val")
    plafond = read_json(fragment("plafond"))["accord"]
    kl_ab = float(plafond["kl_A_vers_B"])
    a = load_dense(teacher_path("A"))
    b = load_dense(teacher_path("B"))
    sentences = fresh_sentences(corpus.vocab)
    twin = read_json(fragment("twin"))
    l_prof = float(twin["dense"]["colonnes"]["nouvelle"])

    arms: dict = {}
    candidates = [("M0", LAB / f"s5_pool24_s{seed}") for seed in GRID_SEEDS]
    candidates += [(BEST, LAB / f"s5_grid_{BEST}_s{seed}") for seed in GRID_SEEDS]
    candidates += [("melange", LAB / f"s5_melange_s{seed}") for seed in GRID_SEEDS]
    for name, path in candidates:
        source = path if str(path).endswith(".pt") else path.with_suffix(".pt")
        if not source.exists():
            continue
        brain = load_brain(source)
        cols = columns_mycelia(brain, val)
        fid = fidelity_student(brain, a, b, corpus)
        top = top3_mycelia(brain, sentences)
        arms.setdefault(name, []).append({
            "modele": source.name,
            "perte": cols["nouvelle"],
            "perte_ancienne": cols["ancienne"],
            "R": relative_gain(cols["nouvelle"], l_prof),
            **fid,
            "F": score_f(fid["kl_A_vers_MYC"], kl_ab),
            "top3": top["top3"],
        })
        print(f"  {name:10s} {source.name:28s} perte {cols['nouvelle']:.4f} "
              f"R {arms[name][-1]['R']:+.4f} KL(A||MYC) "
              f"{fid['kl_A_vers_MYC']:.4f} F {arms[name][-1]['F']:+.4f} "
              f"top3 {top['top3']:.4f}", flush=True)
    payload = {
        "plafond": plafond,
        "l_professeur": l_prof,
        "l_ref": L_REF,
        "phrases_neuves": {
            "n": len(sentences),
            "top3_uniforme": round(3.0 / corpus.vocab_size, 4),
        },
        "bras": arms,
    }
    write_json(fragment("metriques"), payload)
    print(f"ecrit : {fragment('metriques')}", flush=True)


# ---------------------------------------------------------------------------
# D4 / P3 : la survie apres transfert
# ---------------------------------------------------------------------------
def arm_survie(seed: int) -> None:
    corpus = load_corpus("1000")
    val = corpus.fixed(VAL_WINDOWS, "val")
    plafond = read_json(fragment("plafond"))["accord"]
    kl_ab = float(plafond["kl_A_vers_B"])
    a = load_dense(teacher_path("A"))
    b = load_dense(teacher_path("B"))
    twin = read_json(fragment("twin"))
    l_prof = float(twin["dense"]["colonnes"]["nouvelle"])
    brain = load_brain(LAB / f"s5_grid_{BEST}_s{seed}.pt")
    brain.prune_ratio = FAIM_VIVANTE

    def snap(model, step) -> dict:
        cols = columns_mycelia(model, val)
        return {
            "pas": step,
            "perte": cols["nouvelle"],
            "R": relative_gain(cols["nouvelle"], l_prof),
            "aretes": int(model.ne),
            "coupees": int(model.edge_deaths),
            "creees": int(model.births),
        }

    before_cols = columns_mycelia(brain, val)
    before_fid = fidelity_student(brain, a, b, corpus)
    before = {
        "perte": before_cols["nouvelle"],
        "R": relative_gain(before_cols["nouvelle"], l_prof),
        "kl_A_vers_MYC": before_fid["kl_A_vers_MYC"],
        "F": score_f(before_fid["kl_A_vers_MYC"], kl_ab),
        "aretes": int(brain.ne),
        "coupees": int(brain.edge_deaths),
        "creees": int(brain.births),
    }
    started = time.time()
    life = free_life(
        brain, corpus, steps=FREE_STEPS, batch=MYCELIA_BATCH, seed=int(seed),
        lifecycle_every=50, probe=snap, probe_every=100,
    )
    after_cols = columns_mycelia(brain, val)
    after_fid = fidelity_student(brain, a, b, corpus)
    after = {
        "perte": after_cols["nouvelle"],
        "R": relative_gain(after_cols["nouvelle"], l_prof),
        "kl_A_vers_MYC": after_fid["kl_A_vers_MYC"],
        "F": score_f(after_fid["kl_A_vers_MYC"], kl_ab),
        "top3_identiques_avec_A": after_fid["top3_identiques_avec_A"],
        "aretes": int(brain.ne),
        "coupees": int(brain.edge_deaths),
        "creees": int(brain.births),
    }
    payload = {
        "semence": int(seed),
        "modele": f"s5_grid_{BEST}_s{seed}.pt",
        "kl_plafond": kl_ab,
        "regime": {"prune_ratio": FAIM_VIVANTE, "pas_libres": FREE_STEPS},
        "avant": before,
        "apres": after,
        "R_ratio": round(after["R"] / before["R"], 4) if abs(before["R"]) > 1e-9 else None,
        "F_ratio": round(after["F"] / before["F"], 4) if abs(before["F"]) > 1e-9 else None,
        "vie": {k: v for k, v in life.items() if k != "courbe"},
        "courbe": life["courbe"],
        "secondes": round(time.time() - started, 1),
    }
    write_json(fragment(f"survie_transfert_s{seed}"), payload)
    print(f"survie apres transfert s{seed} : F {before['F']:+.4f} -> {after['F']:+.4f} "
          f"(R {before['R']:+.4f} -> {after['R']:+.4f}) ; creees "
          f"{life['aretes_creees']}, coupees {life['aretes_coupees']}", flush=True)


# ---------------------------------------------------------------------------
# P4 : les leviers
# ---------------------------------------------------------------------------
LEVIERS = {
    "T2": {"temperature": 2.0, "label": "temperature T=2"},
    "T4": {"temperature": 4.0, "label": "temperature T=4"},
    "budget": {"steps": 2400, "label": "budget x4 (2400 pas)"},
    "alpha": {"alpha": 0.5, "beta": 0.5, "gamma": 0.0,
              "label": "alpha/beta raffines (0,5/0,5) autour du meilleur point"},
}


def arm_levier(name: str) -> None:
    spec = LEVIERS[name]
    alpha, beta, gamma = GRID[BEST]
    alpha = float(spec.get("alpha", alpha))
    beta = float(spec.get("beta", beta))
    gamma = float(spec.get("gamma", gamma))
    row = train_student(
        f"levier_{name}", 1, alpha, beta, gamma,
        temperature=float(spec.get("temperature", 1.0)),
        steps=int(spec.get("steps", 600)),
    )
    row["levier"] = spec["label"]
    write_json(fragment(f"levier_{name}_s1"), row)
    print(f"levier {spec['label']} : nouvelle {row['nouvelle']:.4f}", flush=True)


# ---------------------------------------------------------------------------
# Fusion
# ---------------------------------------------------------------------------
def arm_merge() -> None:
    plafond = read_json(fragment("plafond"))
    metriques = read_json(fragment("metriques"))
    grid_rows = {}
    for config in GRID:
        for seed in GRID_SEEDS:
            path = fragment(f"grid_{config}_s{seed}")
            if path.exists():
                grid_rows[f"{config}_s{seed}"] = read_json(path)
    melange_rows = {}
    for seed in GRID_SEEDS:
        path = fragment(f"melange_s{seed}")
        if path.exists():
            melange_rows[f"melange_s{seed}"] = read_json(path)
    survies = []
    for seed in GRID_SEEDS:
        path = fragment(f"survie_transfert_s{seed}")
        if path.exists():
            survies.append(read_json(path))
    leviers = {}
    for name in LEVIERS:
        path = fragment(f"levier_{name}_s1")
        if path.exists():
            leviers[name] = read_json(path)

    def arm_stats(name: str) -> dict:
        rows = metriques["bras"].get(name, [])
        if not rows:
            return {}
        return {
            "n": len(rows),
            "perte_moyenne": mean_sd([r["perte"] for r in rows])[0],
            "perte_ecart_type": mean_sd([r["perte"] for r in rows])[1],
            "R_moyen": mean_sd([r["R"] for r in rows])[0],
            "kl_A_vers_MYC": mean_sd([r["kl_A_vers_MYC"] for r in rows])[0],
            "F_moyen": mean_sd([r["F"] for r in rows])[0],
            "F_valeurs": [round(float(r["F"]), 4) for r in rows],
            "top3_moyen": mean_sd([r["top3"] for r in rows])[0],
        }

    m0 = arm_stats("M0")
    distille = arm_stats(BEST)
    melange = arm_stats("melange")
    gain_isole = None
    if m0 and distille:
        gain_isole = round(distille["R_moyen"] - m0["R_moyen"], 4)
    controle = None
    if distille and melange:
        ecart = round(distille["R_moyen"] - melange["R_moyen"], 4)
        controle = {
            "R_melange": melange["R_moyen"],
            "R_distille": distille["R_moyen"],
            "ecart": ecart,
            "propre": bool(ecart > 0.05),
        }

    grid_summary = {}
    for config, (alpha, beta, gamma) in GRID.items():
        rows = [grid_rows[key] for key in grid_rows if key.startswith(f"{config}_")]
        if rows:
            grid_summary[config] = {
                "alpha": alpha, "beta": beta, "gamma": gamma,
                "n": len(rows),
                "perte_moyenne": mean_sd([r["nouvelle"] for r in rows])[0],
                "perte_ecart_type": mean_sd([r["nouvelle"] for r in rows])[1],
                "valeurs": [round(float(r["nouvelle"]), 4) for r in rows],
            }
    best_by_val = min(grid_summary.items(), key=lambda kv: kv[1]["perte_moyenne"])[0] \
        if grid_summary else BEST

    f_avant = distille.get("F_moyen")
    f_apres = mean_sd([r["apres"]["F"] for r in survies])[0] if survies else None
    levees = {}
    for name, row in leviers.items():
        levees[name] = {
            "label": row.get("levier", name),
            "perte": row["nouvelle"],
            "temperature": row.get("temperature"),
            "pas": row.get("pas"),
        }
    payload = {
        "protocole": {
            "professeurs": "deux denses independants (graines 1 et 2), mlp 768",
            "formule": "alpha*T^2*KL + beta*CE + gamma*MSE(P h, h_prof), P aleatoire appris",
            "grille": {k: {"alpha": v[0], "beta": v[1], "gamma": v[2]} for k, v in GRID.items()},
            "semences": list(GRID_SEEDS),
            "pas": 600,
            "fidelite": f"{FID_BATCHES} lots x {FID_WINDOWS} fenetres x 7 positions",
            "score": "F = 1 - KL(A||MYC) / KL(A||B)",
            "vie": "1000 pas libres, prune_ratio 0,5",
        },
        "plafond": plafond,
        "grille": grid_summary,
        "meilleur_par_validation": best_by_val,
        "meilleur_retenu": BEST,
        "bras": metriques["bras"],
        "resume": {
            "M0": m0,
            "distille": distille,
            "melange": melange,
            "gain_distillation_isole": gain_isole,
            "controle_bruit": controle,
            "plafond_kl_A_B": plafond["accord"]["kl_A_vers_B"],
            "F_avant_vie": f_avant,
            "F_apres_vie": f_apres,
            "F_ratio_vie": (round(f_apres / f_avant, 4)
                            if f_avant and f_apres and abs(f_avant) > 1e-9 else None),
        },
        "leviers": levees,
        "survie_transfert": survies,
        "optimisation_restante": bool(f_avant is not None and f_avant < 0.9),
    }
    write_json(LAB / "s5_distill.json", payload)
    print("— distillation ancree")
    for name in ("M0", BEST, "melange"):
        block = arm_stats(name)
        if block:
            print(f"  {name:10s} perte {block['perte_moyenne']:.4f} R "
                  f"{block['R_moyen']:+.4f} KL(A||MYC) {block['kl_A_vers_MYC']:.4f} "
                  f"F {block['F_moyen']:+.4f} top3 {block['top3_moyen']:.4f}", flush=True)
    print(f"  gain isole {gain_isole} ; plafond KL(A||B) "
          f"{plafond['accord']['kl_A_vers_B']}", flush=True)
    print(f"  **F = {f_avant} avant la vie / {f_apres} apres**", flush=True)
    print(f"  ecrit : {LAB / 's5_distill.json'}", flush=True)


def main() -> None:
    ensure_lab()
    parser = argparse.ArgumentParser(description="S5 verrous 3-4 : distillation et plafond")
    parser.add_argument("--arm", action="append", default=[])
    args = parser.parse_args()
    for arm in args.arm:
        parts = arm.split(":")
        kind = parts[0]
        if kind == "teacherB":
            arm_teacherB()
        elif kind == "plafond":
            arm_plafond()
        elif kind == "grid":
            config = parts[1]
            seed = int(parts[2]) if len(parts) > 2 else 1
            alpha, beta, gamma = GRID[config]
            train_student(f"grid_{config}", seed, alpha, beta, gamma)
        elif kind == "melange":
            seed = int(parts[1]) if len(parts) > 1 else 1
            alpha, beta, gamma = GRID[BEST]
            train_student(f"melange", seed, alpha, beta, gamma, permuted=True)
        elif kind == "metriques":
            arm_metriques()
        elif kind == "levier":
            arm_levier(parts[1])
        elif kind == "survie":
            arm_survie(int(parts[1]) if len(parts) > 1 else 1)
        elif kind == "merge":
            arm_merge()
        else:
            raise SystemExit(f"bras inconnu : {arm}")


if __name__ == "__main__":
    main()
