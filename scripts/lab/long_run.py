"""S3-bis D3 — la courbe longue, et le verdict A/B.

S2 arretait MYCELIA apres 50 pas : on ne savait pas si 5,078 etait un plancher
ou un point d'arret premature. Ici le meme MYCELIA (semence 1, config S2 exacte)
court **500 pas**, dix fois plus, et on enregistre toute la courbe. Le dense jumeau
court le meme budget, avec les memes lots de validation, comme controle.

Le modele est ajuste a la main :

    L(t) = L_inf + a * t^(-b)

Pourquoi ajuster l'**enveloppe** (meilleure validation atteinte jusqu'au pas t)
et pas la courbe brute : la validation est bruitee et, apres le minimum, elle
remonte (surapprentissage). Une loi de puissance decrit une progression, pas une
degringolade ; l'enveloppe est monotone par construction et c'est elle que la
loi modelise. La pente ``b`` et ``a`` sortent d'une regression lineaire en
log-log a ``L_inf`` fixe ; ``L_inf`` est choisi par balayage (celui qui minimise
la somme des carres des residus en log), puis son incertitude vient d'un
bootstrap sur les points (200 reechantillonnages, intervalle a 90 %).

Verdict ecrit dans le JSON :

* ``L_inf >= 4,95`` -> **B**, plafond structurel a ce dimensionnement ;
* ``L_inf < 4,80``  -> **A**, lenteur : MYCELIA y arrive, il lui faut plus de pas ;
* entre les deux    -> **partiel**, zone d'ambiguite assumee.

Ecrit ``data/lab/s3bis_long.json``.
"""

from __future__ import annotations

import argparse
import math
import time

import numpy as np
import torch

from lab_common import LAB, ToyCorpus, ensure_lab, read_json, set_seed, write_json
from micro_dense import n_parameters as dense_parameters, train_dense
from micro_mycelia import n_parameters as mycelia_parameters, train_mycelia
from twins import (
    DENSE_BATCH, DENSE_LR, MYCELIA_BATCH, MYCELIA_LR, MYCELIA_WD, SEQ,
    dense_twin, mycelia_twin,
)

STEPS = 500
LOG_EVERY = 5
VAL_WINDOWS = 48
SEED = 1


def envelope(curve: list[dict]) -> list[dict]:
    """Meilleure validation atteinte jusqu'au pas t (monotone, sans bruit)."""
    out: list[dict] = []
    best = math.inf
    for row in curve:
        value = float(row["val"])
        if value < best:
            best = value
        out.append({"step": int(row["step"]), "val": round(best, 6)})
    return out


def _loglog_fit(steps, values, l_inf: float):
    """Regression log-log a ``L_inf`` fixe : rend (a, b, sse)."""
    residual = values - l_inf
    if np.any(residual <= 1e-9):
        return None
    x = np.log(steps)
    y = np.log(residual)
    slope, intercept = np.polyfit(x, y, 1)
    predicted = slope * x + intercept
    sse = float(((y - predicted) ** 2).sum())
    return math.exp(float(intercept)), -float(slope), sse


def fit_power_law(curve: list[dict], start: int = 10, resamples: int = 200) -> dict:
    """Ajuste ``L_inf + a t^-b`` sur l'enveloppe, avec intervalle de confiance.

    Le balayage de ``L_inf`` va de ``min(val) - 1,5`` a ``min(val) - 0,001`` :
    une asymptote au-dessus du minimum observe n'a aucun sens (le modele l'a
    deja depassee).
    """
    points = [row for row in curve if int(row["step"]) >= int(start)]
    if len(points) < 5:
        return {"ok": False, "raison": "moins de 5 points exploitables"}
    steps = np.asarray([row["step"] for row in points], dtype=np.float64)
    values = np.asarray([row["val"] for row in points], dtype=np.float64)
    floor = float(values.min())
    grid = np.linspace(floor - 1.5, floor - 0.001, 1500)
    best = None
    for l_inf in grid:
        fitted = _loglog_fit(steps, values, float(l_inf))
        if fitted is None:
            continue
        a, b, sse = fitted
        if best is None or sse < best[3]:
            best = (float(l_inf), a, b, sse)
    if best is None:
        return {"ok": False, "raison": "aucune asymptote ne laisse des residus positifs"}
    l_inf, a, b, sse = best

    rng = np.random.default_rng(20261120)
    draws: list[float] = []
    for _ in range(int(resamples)):
        pick = rng.integers(0, len(steps), len(steps))
        sub_steps, sub_values = steps[pick], values[pick]
        sub_floor = float(sub_values.min())
        local = None
        for candidate in np.linspace(sub_floor - 1.5, sub_floor - 0.001, 300):
            fitted = _loglog_fit(sub_steps, sub_values, float(candidate))
            if fitted is None:
                continue
            if local is None or fitted[2] < local[1]:
                local = (float(candidate), fitted[2])
        if local is not None:
            draws.append(local[0])
    if draws:
        low, high = np.percentile(draws, [5, 95])
    else:
        low = high = l_inf
    # Un minimum pose sur le bord du domaine n'est pas un ajustement : c'est le
    # signe que la somme des carres decroit jusque-la sans jamais remonter. Sur
    # une courbe plate, c'est le cas (mesure : 3,6509 = minimum observe - 1,5
    # pour MYCELIA, 2,2921 pour le dense). Le drapeau le dit au rapport.
    at_bound = abs(float(l_inf) - float(floor - 1.5)) < 1e-6

    if l_inf >= 4.95:
        verdict = "B"
    elif l_inf < 4.80:
        verdict = "A"
    else:
        verdict = "PARTIEL"
    return {
        "ok": True,
        "L_inf": round(l_inf, 4),
        "a": round(a, 4),
        "b": round(b, 4),
        "sse_log": round(sse, 6),
        "points": int(len(points)),
        "minimum_observe": round(floor, 4),
        "L_inf_ic90": None if float(high) - float(low) < 1e-6 else [round(float(low), 4), round(float(high), 4)],
        "au_bord": bool(at_bound),
        "L_inf_draws": len(draws),
        "verdict_ajuste": verdict,
    }


def decide(fit: dict, curve: list[dict]) -> dict:
    """Verdict honnete : la loi de puissance n'est retenue que si elle est identifiable.

    Mesure : une courbe qui ne bouge plus sur les 80 % finaux du budget ne dit
    rien de son asymptote. Exemple mesure : 5,1509 au pas 55, puis 445 pas sans
    le moindre gain ; la loi de puissance peut alors poser L_inf a 3,65 sans
    contredire les points (la decroissance aurait eu lieu avant le pas 55). Ce
    n'est pas une prevision, c'est un parametre libre. Dans ce cas l'asymptote
    retenue est la meilleure validation atteinte, et le rapport le dit.
    """
    if not fit.get("ok"):
        return {"L_inf_effectif": None, "identifiable": False, "raison": fit.get("raison")}
    end_step = int(curve[-1]["step"])
    end_val = float(curve[-1]["val"])
    start = max(10, int(0.2 * end_step))
    base = next(float(row["val"]) for row in curve if int(row["step"]) >= start)
    gain = round(base - end_val, 4)
    identifiable = bool(gain >= 0.02 and not fit.get("au_bord"))
    if identifiable:
        l_inf = float(fit["L_inf"])
        reason = "la courbe decroit encore sur la fin du budget : l'asymptote ajustee est retenue"
    else:
        l_inf = end_val
        reason = (
            f"gain {gain:+.4f} sur les 80 % finaux, ajustement "
            f"{'au bord du domaine' if fit.get('au_bord') else 'non identifiable'} : "
            "la loi de puissance ne pin pas l'asymptote, on retient la meilleure "
            "validation atteinte (l'incertitude est le gain residuel observe)"
        )
    if l_inf >= 4.95:
        verdict = "B"
    elif l_inf < 4.80:
        verdict = "A"
    else:
        verdict = "PARTIEL"
    return {
        "L_inf_effectif": round(l_inf, 4),
        "identifiable": bool(identifiable),
        "incertitude_empirique": round(float(gain), 4),
        "gain_apres_20_pourcent": gain,
        "verdict": verdict,
        "raison": reason,
    }


def run_mycelia(corpus: ToyCorpus, val) -> dict:
    set_seed(SEED)
    brain = mycelia_twin(corpus.vocab_size, SEED)
    started = time.time()
    result = train_mycelia(
        brain, corpus, STEPS, lr=MYCELIA_LR, batch=MYCELIA_BATCH, seed=SEED,
        log_every=LOG_EVERY, val_batches=val, weight_decay=MYCELIA_WD,
        patience=0, restore_best=True,
    )
    curve = [{"step": r["step"], "val": r["val"], "loss": r["loss"]} for r in result["curve"]]
    return {
        "params": int(mycelia_parameters(result["brain"])),
        "best_val": result["best_val"],
        "best_step": result["best_step"],
        "secondes": round(time.time() - started, 1),
        "aretes": int(result["edges"]),
        "pool": int(result["pool"]),
        "curve": curve,
        "envelope": envelope(curve),
    }


def run_dense(corpus: ToyCorpus, val) -> dict:
    set_seed(SEED)
    model = dense_twin(corpus.vocab_size, SEED)
    started = time.time()
    result = train_dense(
        model, corpus, STEPS, lr=DENSE_LR, batch=DENSE_BATCH, seed=SEED,
        log_every=LOG_EVERY, val_batches=val, patience=0,
    )
    curve = [{"step": r["step"], "val": r["val"], "loss": r["loss"]} for r in result["curve"]]
    return {
        "params": int(dense_parameters(model)),
        "best_val": result["final_val"],
        "best_step": result["best_step"],
        "secondes": round(time.time() - started, 1),
        "curve": curve,
        "envelope": envelope(curve),
    }


def refit_existing() -> None:
    """Recalcule les ajustements depuis les courbes deja enregistrees.

    Utile quand seule l'interpretation change (ici : la degenerescence de la loi
    de puissance sur une courbe plate). Aucun entrainement n'est rejoue.
    """
    blob = read_json(LAB / "s3bis_long.json")
    for key in ("mycelia", "dense"):
        blob[key]["fit"] = fit_power_law(blob[key]["envelope"])
        blob[key]["decision"] = decide(blob[key]["fit"], blob[key]["envelope"])
    blob["verdict"] = blob["mycelia"]["decision"]["verdict"]
    write_json(LAB / "s3bis_long.json", blob)
    for key in ("mycelia", "dense"):
        fit = blob[key]["fit"]
        decision = blob[key]["decision"]
        print(f"{key:8s} : meilleure {blob[key]['best_val']:.4f} — L_inf ajuste "
              f"{fit.get('L_inf')} (IC90 {fit.get('L_inf_ic90')}), effectif "
              f"{decision['L_inf_effectif']} — verdict {decision['verdict']}")
        print(f"           {decision['raison']}")
    print(f"VERDICT A/B : {blob['verdict']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="courbe longue et verdict A/B (S3-bis D3)")
    parser.add_argument("--refit", action="store_true")
    args = parser.parse_args()
    if args.refit:
        refit_existing()
        return
    ensure_lab()
    corpus = ToyCorpus(seq_len=SEQ)
    val = corpus.fixed(VAL_WINDOWS, "val")
    print(f"courbe longue : {STEPS} pas, log tous les {LOG_EVERY}, "
          f"{VAL_WINDOWS} fenetres de validation")

    print("MYCELIA (semence 1) ...")
    mycelia = run_mycelia(corpus, val)
    mycelia["fit"] = fit_power_law(mycelia["envelope"])
    print(f"  meilleure validation {mycelia['best_val']:.4f} au pas {mycelia['best_step']} "
          f"en {mycelia['secondes']} s — L_inf {mycelia['fit'].get('L_inf')} "
          f"({mycelia['fit'].get('verdict')})")

    print("dense jumeau (controle) ...")
    dense = run_dense(corpus, val)
    dense["fit"] = fit_power_law(dense["envelope"])
    print(f"  meilleure validation {dense['best_val']:.4f} au pas {dense['best_step']} "
          f"en {dense['secondes']} s — L_inf {dense['fit'].get('L_inf')} "
          f"({dense['fit'].get('verdict')})")

    mycelia["decision"] = decide(mycelia["fit"], mycelia["envelope"])
    dense["decision"] = decide(dense["fit"], dense["envelope"])
    verdict = mycelia["decision"]["verdict"]
    payload = {
        "protocole": {
            "pas": STEPS,
            "log_every": LOG_EVERY,
            "validation_fenetres": VAL_WINDOWS,
            "semence": SEED,
            "mycelia_lr": MYCELIA_LR,
            "mycelia_batch": MYCELIA_BATCH,
            "mycelia_weight_decay": MYCELIA_WD,
            "dense_lr": DENSE_LR,
            "dense_batch": DENSE_BATCH,
            "forme": "L(t) = L_inf + a * t^-b sur l'enveloppe (meilleure validation)",
        },
        "mycelia": mycelia,
        "dense": dense,
        "verdict": verdict,
        "decision": mycelia["decision"],
        "gain_dense_final": round(float(dense["best_val"]) - float(mycelia["best_val"]), 4),
    }
    write_json(LAB / "s3bis_long.json", payload)
    print(f"VERDICT A/B : {verdict} ({mycelia['decision']['raison']})")
    print("ecrit : data/lab/s3bis_long.json")


if __name__ == "__main__":
    main()
