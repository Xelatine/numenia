"""S5-MAX Verrou 2 — le receveur qui vit : NUMENIA au pool 24.

V1. ``RepairedBrain`` porte au **pool 24** (meme sous-classe, ``brain.py`` de
    Nyra intact). La lecture lit les 24 membranes (384 dims) plus le noeud 1 et
    la voie rapide. Le jumeau equitable est **recalcule** : on compte les
    parametres actifs par token, puis on choisit la taille MLP du dense pour
    qu'il soit au moins aussi gros (comparaison defavorable a NUMENIA).

V2. Meme budget que S4 (600 pas, 32 par lot, memes reglages), mesure corrigee.
    Le test de contexte est refait sur le cerveau du pool 24.

V3. La survie reelle : 1000 pas de metabolisme LIBRE, naissances et morts
    comptees. Deux regimes, et c'est la decouverte du verrou :

    * ``defaut`` (``prune_ratio`` 0,1, le reglage du cerveau) : aucune synapse
      n'est jamais affamee -- les usages mesures vivent entre 0,016 et 0,065 et
      le seuil vaut 0,0056, sous le plancher. La mort est **inerte** ;
    * ``faim`` (``prune_ratio`` 0,5) : le seuil passe dans la distribution, des
      synapses meurent a chaque battement et les naissances re-remplissent les
      slots. C'est le seul regime ou le metabolisme tourne vraiment, et c'est
      lui qui porte le verdict V4.

    Le pool 32 est re-teste comme la mission le demande : il ne change rien a la
    faim (meme diffusion exhaustive), ce qui identifie la cause -- l'absence de
    routage selectif, pas la taille du pool.

V4. R (gain relatif a l'unigramme, mesure corrigee) avant -> apres :
    🟢 >= 0,9 x R avant · 🟡 0,7-0,9 · 🔴 < 0,7.

    python scripts/lab/s5_pool24.py --arm twin
    python scripts/lab/s5_pool24.py --arm train:1
    python scripts/lab/s5_pool24.py --arm context:1
    python scripts/lab/s5_pool24.py --arm survie:1
    python scripts/lab/s5_pool24.py --arm pool32
    python scripts/lab/s5_pool24.py --arm merge
"""

from __future__ import annotations

import argparse
import random
import time

import numpy as np
import torch

from lab_common import LAB, SEED, ensure_lab, read_json, write_json
from micro_dense import n_parameters, train_dense
from repaired_mycelia import (
    SEQ,
    aligned_loss,
    build_repaired,
    free_life,
    load_brain,
    per_position_loss,
    readout_accounting,
    save_brain,
    train_repaired,
)
from s4_context import CONDITIONS, collect_readings, collect_states, condition_loss, softmax_probe
from s5_common import (
    FAIM_DEFAUT,
    FAIM_VIVANTE,
    MLP_TWIN,
    POOL24,
    POOL32,
    SEEDS,
    STEPS,
    build_dense,
    columns_mycelia,
    load_corpus,
    mean_sd,
    relative_gain,
    save_dense,
)
from twins import DENSE_BATCH, DENSE_LR, MYCELIA_BATCH, MYCELIA_LR, MYCELIA_WD

VAL_WINDOWS = 58
PROBE_TRAIN_WINDOWS = 150
FREE_STEPS = 1000


def brain_path(seed: int):
    return LAB / f"s5_pool24_s{seed}.pt"


def fragment(name: str):
    return LAB / f"s5_{name}.json"


# ---------------------------------------------------------------------------
# V1 : le jumeau equitable, recalcule
# ---------------------------------------------------------------------------
def arm_twin() -> None:
    corpus = load_corpus("1000")
    val = corpus.fixed(VAL_WINDOWS, "val")
    brain = build_repaired(corpus.vocab_size, 1, two_speeds=True, overrides=POOL24)
    account = readout_accounting(brain)
    active = int(sum(p.numel() for p in brain.parameters()))

    mlp = MLP_TWIN
    dense = build_dense(corpus.vocab_size, 1, mlp=mlp)
    dparams = int(n_parameters(dense))
    result = train_dense(
        dense, corpus, 600, lr=DENSE_LR, batch=DENSE_BATCH, seed=1,
        log_every=100, patience=0, val_batches=val,
    )
    save_dense(dense, LAB / "s5_teacherA.pt", {"val": result["final_val"], "seed": 1})
    from s5_common import columns_dense

    cols = columns_dense(dense, val)
    payload = {
        "pool": int(brain.n_nodes),
        "aretes_max": int(brain.max_edges),
        "paires_possibles": int(brain.n_nodes * (brain.n_nodes - 1)),
        "densite_plafond": round(brain.max_edges / max(1, brain.n_nodes * (brain.n_nodes - 1)), 4),
        "mycelia": account,
        "mycelia_actifs_par_token": active,
        "dense": {
            "mlp": int(mlp),
            "parametres": dparams,
            "parametres_formule": 12_792 + 66 * int(mlp),
            "ratio_dense_sur_mycelia": round(dparams / max(1, active), 4),
            "val_600pas": result["final_val"],
            "colonnes": cols,
            "secondes": result["seconds"],
        },
        "egalite": bool(dparams >= active),
        "regle": "le dense est au moins aussi gros que NUMENIA : la comparaison "
                 "reste defavorable a NUMENIA",
    }
    write_json(fragment("twin"), payload)
    print(f"NUMENIA pool 24 : {active} parametres actifs ({account['ajoutes']} ajoutes "
          f"par la lecture) ; dense mlp {mlp} : {dparams} "
          f"({payload['dense']['ratio_dense_sur_mycelia']}x)", flush=True)
    print(f"professeur A (dense jumeau) : val 600 pas {result['final_val']:.4f} ; "
          f"colonnes {cols}", flush=True)


# ---------------------------------------------------------------------------
# V2 : l'entrainement
# ---------------------------------------------------------------------------
def arm_train(seed: int) -> None:
    corpus = load_corpus("1000")
    val = corpus.fixed(VAL_WINDOWS, "val")
    brain = build_repaired(corpus.vocab_size, int(seed), two_speeds=True, overrides=POOL24)
    account = readout_accounting(brain)
    started = time.time()
    result = train_repaired(
        brain, corpus, STEPS, lr=MYCELIA_LR, batch=MYCELIA_BATCH, seed=int(seed),
        log_every=50, weight_decay=MYCELIA_WD, val_batches=val, lifecycle_every=100,
    )
    brain = result["brain"]
    save_brain(brain_path(int(seed)), brain, step=result["best_step"], extra=account)
    cols = columns_mycelia(brain, val)
    row = {
        "semence": int(seed),
        **cols,
        "honnete_0_7": result["best_val"],
        "decalee_0_7": result["best_val_s3bis"],
        "pas": int(result["best_step"]),
        "positions": result["per_position"],
        "parametres_ajoutes": account["ajoutes"],
        "parametres_total": account["total_modele"],
        "pool": int(result["pool"]),
        "aretes": int(result["edges"]),
        "aretes_creees": int(result["births"]),
        "aretes_coupees": int(result["edge_deaths"]),
        "densite": round(result["edges"] / max(1, POOL24["n_nodes"] * (POOL24["n_nodes"] - 1)), 4),
        "secondes": round(time.time() - started, 1),
        "courbe": [{"step": r["step"], "val": r["val"], "aretes": r["edges"]}
                   for r in result["curve"]],
    }
    write_json(fragment(f"pool24_s{seed}"), row)
    print(f"pool 24 s{seed} : nouvelle {cols['nouvelle']:.4f} / ancienne {cols['ancienne']:.4f} "
          f"au pas {row['pas']} ; aretes {row['aretes']} ({row['densite']:.1%}), "
          f"creees {row['aretes_creees']}, coupees {row['aretes_coupees']} — "
          f"{row['secondes']}s", flush=True)


# ---------------------------------------------------------------------------
# V2 bis : le test de contexte au pool 24
# ---------------------------------------------------------------------------
def arm_context(seed: int) -> None:
    corpus = load_corpus("1000")
    brain = load_brain(brain_path(int(seed)))
    val = corpus.fixed(VAL_WINDOWS, "val")
    vocab = corpus.vocab_size
    results = {}
    for index, condition in enumerate(CONDITIONS):
        rng = random.Random(SEED + index)
        results[condition] = condition_loss(brain, val, condition, rng)
    normal, shuffle, empty = (results[c] for c in CONDITIONS)

    train_fixed = corpus.fixed(PROBE_TRAIN_WINDOWS, "train")
    Xtr, ntr = collect_states(brain, train_fixed)
    Xva, nva = collect_states(brain, val)
    Rtr, rtr = collect_readings(brain, train_fixed)
    Rva, rva = collect_readings(brain, val)
    width = int(brain.nodes_read) * int(brain.dim)

    def wide(X):
        return np.hstack([X[:, :width], X[:, -7:]])

    probes = {
        "noeud1_16d": softmax_probe(Xtr[:, 0:16], ntr, Xva[:, 0:16], nva, vocab),
        f"toutes_{width}d": softmax_probe(wide(Xtr), ntr, wide(Xva), nva, vocab),
        "lecture": softmax_probe(Rtr, rtr, Rva, rva, vocab),
    }
    spread = max(
        abs(normal["perte"] - shuffle["perte"]),
        abs(normal["perte"] - empty["perte"]),
    )
    payload = {
        "semence": int(seed),
        "pool": int(brain.n_nodes),
        "aretes": int(brain.ne),
        "fenetres_validation": len(val),
        "fenetres_sonde": len(train_fixed),
        "conditions": {
            name: {
                "perte": results[name]["perte"],
                "exactitude": results[name]["exactitude"],
                "entropie_predite": results[name]["entropie_predite"],
            }
            for name in CONDITIONS
        },
        "ecart_max_conditions": round(spread, 4),
        "ecart_normal_melange": round(abs(normal["perte"] - shuffle["perte"]), 4),
        "perte_par_position": per_position_loss(brain, val),
        "sondes": probes,
        "seuil_ecart": 2.0,
        "contexte_utilise": bool(spread >= 2.0),
        "colonnes": columns_mycelia(brain, val),
        "honnete_0_7": round(aligned_loss(brain, val), 4),
    }
    write_json(fragment(f"context24_s{seed}"), payload)
    print(f"contexte s{seed} : normal {normal['perte']:.4f} / melange "
          f"{shuffle['perte']:.4f} / vide {empty['perte']:.4f} — ecart {spread:.4f} "
          f"({'OK' if spread >= 2.0 else 'FAIBLE'})", flush=True)
    print(f"  sondes : { {k: v['perte'] for k, v in probes.items()} }", flush=True)


# ---------------------------------------------------------------------------
# V3/V4 : la vie libre, deux regimes de faim
# ---------------------------------------------------------------------------
def arm_survie(seed: int) -> None:
    corpus = load_corpus("1000")
    val = corpus.fixed(VAL_WINDOWS, "val")
    twin = read_json(fragment("twin"))
    l_prof = float(twin["dense"]["colonnes"]["nouvelle"])
    l_ref = 5.082888

    def loss_of(brain) -> float:
        return columns_mycelia(brain, val)["nouvelle"]

    def probe(brain, step):
        loss = loss_of(brain)
        return {
            "perte": loss,
            "R": relative_gain(loss, l_prof, l_ref),
            "aretes": int(brain.ne),
            "coupees": int(brain.edge_deaths),
            "creees": int(brain.births),
        }

    rows = {"semence": int(seed), "l_professeur": l_prof, "l_ref": l_ref, "regimes": {}}
    for name, ratio in (("defaut", FAIM_DEFAUT), ("faim", FAIM_VIVANTE)):
        brain = load_brain(brain_path(int(seed)))
        brain.prune_ratio = float(ratio)
        before_loss = loss_of(brain)
        before_R = relative_gain(before_loss, l_prof, l_ref)
        edges_before = int(brain.ne)
        degrees_before = torch.bincount(
            brain.edge_index[0, : brain.ne], minlength=brain.n_nodes
        ).tolist()
        started = time.time()
        life = free_life(
            brain, corpus, steps=FREE_STEPS, batch=MYCELIA_BATCH, seed=int(seed),
            lifecycle_every=50, probe=probe, probe_every=100,
        )
        after_loss = loss_of(brain)
        after_R = relative_gain(after_loss, l_prof, l_ref)
        degrees_after = torch.bincount(
            brain.edge_index[0, : brain.ne], minlength=brain.n_nodes
        ).tolist()
        ratio_R = round(after_R / before_R, 4) if abs(before_R) > 1e-9 else None
        rows["regimes"][name] = {
            "prune_ratio": float(ratio),
            "perte_avant": before_loss,
            "perte_apres": after_loss,
            "R_avant": before_R,
            "R_apres": after_R,
            "ratio": ratio_R,
            **{k: v for k, v in life.items() if k != "courbe"},
            "courbe": life["courbe"],
            "degres_avant": degrees_before,
            "degres_apres": degrees_after,
            "noeuds_amputes": [
                int(i) for i, (a, b) in enumerate(zip(degrees_before, degrees_after))
                if b == 0 and a > 0
            ],
            "secondes": round(time.time() - started, 1),
        }
        print(f"vie s{seed} [{name}] : R {before_R:+.4f} -> {after_R:+.4f} "
              f"(ratio {ratio_R}) ; aretes {edges_before}->{life['aretes_fin']}, "
              f"creees {life['aretes_creees']}, coupees {life['aretes_coupees']}, "
              f"densite {life['densite_fin']:.1%}", flush=True)

    verdict_regime = rows["regimes"]["faim"]
    ratio = verdict_regime["ratio"]
    if ratio is None:
        code, phrase = "⚪", "R avant nul : la survie n'est pas mesurable"
    elif ratio >= 0.9:
        code, phrase = "🟢", "le savoir survit a la vie : le metabolisme ne coute pas le savoir"
    elif ratio >= 0.7:
        code, phrase = "🟡", "survie partielle : la faim coupe des porteurs du savoir"
    else:
        code, phrase = "🔴", "la vie devore le savoir : c'est le vrai cout du metabolisme"
    rows["verdict"] = {
        "code": code,
        "phrase": phrase,
        "ratio": ratio,
        **{"critere": "R_apres >= 0,9 x R_avant (regime faim)"},
    }
    write_json(fragment(f"survie_s{seed}"), rows)
    print(f"verdict s{seed} : {code} {phrase} (ratio {ratio})", flush=True)


# ---------------------------------------------------------------------------
# V3 bis : le pool 32, comme la mission le demande si le graphe sature
# ---------------------------------------------------------------------------
def arm_pool32() -> None:
    corpus = load_corpus("1000")
    val = corpus.fixed(VAL_WINDOWS, "val")
    brain = build_repaired(corpus.vocab_size, 1, two_speeds=True, overrides=POOL32)
    started = time.time()
    result = train_repaired(
        brain, corpus, 200, lr=MYCELIA_LR, batch=MYCELIA_BATCH, seed=1,
        log_every=100, weight_decay=MYCELIA_WD, val_batches=val, lifecycle_every=50,
        restore_best=False,
    )
    brain = result["brain"]
    life = free_life(
        brain, corpus, steps=300, batch=MYCELIA_BATCH, seed=1, lifecycle_every=50
    )
    payload = {
        "pool": int(brain.n_nodes),
        "aretes_max": int(brain.max_edges),
        "paires_possibles": int(brain.n_nodes * (brain.n_nodes - 1)),
        "densite": round(int(brain.ne) / max(1, brain.n_nodes * (brain.n_nodes - 1)), 4),
        "pas_entrainement": 200,
        "aretes_creees_entrainement": int(result["births"]),
        "aretes_coupees_entrainement": int(result["edge_deaths"]),
        "vie_libre": life,
        "usage_min": float(brain.edge_usage[: brain.ne].min()),
        "usage_max": float(brain.edge_usage[: brain.ne].max()),
        "seuil_faim_defaut": round(float(brain.edge_usage[: brain.ne].topk(
            max(1, brain.ne // 10)).values.mean()) * FAIM_DEFAUT, 6),
        "secondes": round(time.time() - started, 1),
    }
    write_json(fragment("pool32"), payload)
    print(f"pool 32 : densite {payload['densite']:.1%}, coupees entrainement "
          f"{payload['aretes_coupees_entrainement']}, vie libre creees "
          f"{life['aretes_creees']} coupees {life['aretes_coupees']} — "
          f"usage {payload['usage_min']:.4f}-{payload['usage_max']:.4f} vs seuil "
          f"{payload['seuil_faim_defaut']:.4f}", flush=True)


# ---------------------------------------------------------------------------
# Fusion
# ---------------------------------------------------------------------------
def arm_merge() -> None:
    twin = read_json(fragment("twin"))
    # Diagnostic de la faim : pourquoi la mort est inerte au reglage du cerveau.
    diagnostic = {}
    if brain_path(1).exists():
        probe_brain = load_brain(brain_path(1))
        usage = probe_brain.edge_usage[: probe_brain.ne]
        busiest = max(1, int(probe_brain.ne) // 10)
        cut = float(usage.topk(busiest).values.mean()) * FAIM_DEFAUT
        diagnostic = {
            "aretes": int(probe_brain.ne),
            "usage_min": round(float(usage.min()), 6),
            "usage_median": round(float(usage.median()), 6),
            "usage_max": round(float(usage.max()), 6),
            "seuil_defaut": round(cut, 6),
            "sous_le_seuil": int((usage < cut).sum()),
            "aretes_a_usage_nul": int((usage == 0).sum()),
        }
        print(f"diagnostic faim : usage {diagnostic['usage_min']:.5f}-"
              f"{diagnostic['usage_max']:.5f} contre un seuil a "
              f"{diagnostic['seuil_defaut']:.5f} -> "
              f"{diagnostic['sous_le_seuil']} aretes affamees", flush=True)
    trains = [read_json(fragment(f"pool24_s{s}")) for s in SEEDS
              if fragment(f"pool24_s{s}").exists()]
    contexts = [read_json(fragment(f"context24_s{s}")) for s in SEEDS
                if fragment(f"context24_s{s}").exists()]
    survies = [read_json(fragment(f"survie_s{s}")) for s in SEEDS
               if fragment(f"survie_s{s}").exists()]
    pool32 = read_json(fragment("pool32")) if fragment("pool32").exists() else None

    legacy = read_json(LAB / "s4_validation.json")
    s4_mean = legacy["runs"]["1000"]["resume"]["deux_vitesses"]["moyenne"]
    s4_full = legacy["runs"]["1000"]["resume"]["pleine"]["moyenne"]
    bigram = 3.046586

    def mean(rows, key):
        return mean_sd([float(r[key]) for r in rows])

    new_mean, new_sd = mean(trains, "nouvelle")
    old_mean, old_sd = mean(trains, "ancienne")
    if new_mean < bigram:
        verdict, phrase = "🥇", "le pool 24 bat le bigramme"
    elif new_mean < float(twin["dense"]["colonnes"]["nouvelle"]):
        verdict, phrase = "🥈", "le pool 24 bat le dense jumeau du pool 24"
    else:
        verdict, phrase = "🥉", "le pool 24 reste derriere son jumeau dense"

    def regime_mean(name, key):
        rows = [r["regimes"][name] for r in survies if name in r.get("regimes", {})]
        return mean_sd([float(r[key]) for r in rows]) if rows else (0.0, 0.0)

    codes = [str(r.get("verdict", {}).get("code")) for r in survies]
    ratio_moyen = regime_mean("faim", "ratio")[0]
    if ratio_moyen >= 0.9:
        code_moyen, phrase_moyen = "🟢", "le savoir survit a la vie"
    elif ratio_moyen >= 0.7:
        code_moyen, phrase_moyen = "🟡", "survie partielle : la faim coupe des porteurs du savoir"
    else:
        code_moyen, phrase_moyen = "🔴", "la vie devore le savoir : c'est le vrai cout du metabolisme"
    payload = {
        "protocole": {
            "pool": POOL24["n_nodes"],
            "aretes_max": POOL24["max_edges"],
            "pas": STEPS,
            "semences": list(SEEDS),
            "lr": MYCELIA_LR,
            "batch": MYCELIA_BATCH,
            "weight_decay": MYCELIA_WD,
            "mesure": "colonne nouvelle (token courant)",
            "s4_reference_deux_vitesses": s4_mean,
            "s4_reference_pleine": s4_full,
            "bigramme": bigram,
        },
        "jumeau": twin,
        "entrainements": trains,
        "contexte": contexts,
        "survie": survies,
        "pool32": pool32,
        "diagnostic_faim": diagnostic,
        "resume": {
            "nouvelle_moyenne": new_mean,
            "nouvelle_ecart_type": new_sd,
            "ancienne_moyenne": old_mean,
            "ancienne_ecart_type": old_sd,
            "parametres_actifs": twin["mycelia_actifs_par_token"],
            "dense_parametres": twin["dense"]["parametres"],
            "aretes_moyennes": mean(trains, "aretes")[0],
            "densite_moyenne": mean(trains, "densite")[0],
            "verdict": {"code": verdict, "phrase": phrase},
        },
        "survie_resume": {
            "defaut": {
                "R_avant": regime_mean("defaut", "R_avant")[0],
                "R_apres": regime_mean("defaut", "R_apres")[0],
                "ratio": regime_mean("defaut", "ratio")[0],
                "coupees": regime_mean("defaut", "aretes_coupees")[0],
                "creees": regime_mean("defaut", "aretes_creees")[0],
            },
            "faim": {
                "R_avant": regime_mean("faim", "R_avant")[0],
                "R_apres": regime_mean("faim", "R_apres")[0],
                "ratio": regime_mean("faim", "ratio")[0],
                "coupees": regime_mean("faim", "aretes_coupees")[0],
                "creees": regime_mean("faim", "aretes_creees")[0],
                "verdict": {
                    "code": code_moyen,
                    "phrase": phrase_moyen,
                    "ratio": ratio_moyen,
                    "codes_par_semence": codes,
                    "critere": "R_apres >= 0,9 x R_avant (regime faim)",
                },
            },
        },
    }
    write_json(LAB / "s5_pool24.json", payload)
    print(f"— pool 24 : nouvelle {new_mean:.4f} ± {new_sd:.4f} / ancienne "
          f"{old_mean:.4f} ± {old_sd:.4f} — {verdict} {phrase}", flush=True)
    for name in ("defaut", "faim"):
        block = payload["survie_resume"][name]
        print(f"  vie [{name}] : R {block['R_avant']:+.4f} -> {block['R_apres']:+.4f} "
              f"(ratio {block['ratio']}) ; creees {block['creees']}, coupees {block['coupees']}",
              flush=True)
    print(f"  ecrit : {LAB / 's5_pool24.json'}", flush=True)


def main() -> None:
    ensure_lab()
    parser = argparse.ArgumentParser(description="S5 verrou 2 : le pool 24 vivant")
    parser.add_argument("--arm", action="append", default=[])
    args = parser.parse_args()
    for arm in args.arm:
        parts = arm.split(":")
        kind = parts[0]
        seed = int(parts[1]) if len(parts) > 1 else 1
        if kind == "twin":
            arm_twin()
        elif kind == "train":
            arm_train(seed)
        elif kind == "context":
            arm_context(seed)
        elif kind == "survie":
            arm_survie(seed)
        elif kind == "pool32":
            arm_pool32()
        elif kind == "merge":
            arm_merge()
        else:
            raise SystemExit(f"bras inconnu : {arm}")


if __name__ == "__main__":
    main()
