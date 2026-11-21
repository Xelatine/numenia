"""S4 R5-R8 — la distillation, ses controles, ses metriques, sa survie.

Le professeur est le dense jouet bien entraine (jumeau S2, 23 748 parametres,
memes lots, meme tache). La formule est exactement celle de la mission :

    L = KL(logits_MYC || logits_prof) + lambda * d(P h_MYC, h_prof)

ou ``P`` est un projecteur lineaire initialise a l'identite et **jete apres
usage** (il ne fait pas partie du cerveau). Pas de terme de cross-entropy : le
savoir doit venir du professeur, sinon le gain ne serait pas distinguable d'un
entrainement supervise ordinaire.

Trois bras, memes donnees, meme budget, metabolisme actif dans les trois :

* ``M0``       — le MYCELIA repare entraine SEUL : c'est le bras
  ``deux_vitesses`` de la validation (memes pas, memes lots, memes semences).
  Le controle cle : sans lui, rien ne prouve le transfert.
* ``distille`` — MYCELIA repare + professeur, lambda = 0,5.
* ``melange``  — le meme professeur, **etiquettes permutees** : forme et entropie
  identiques, lien mot -> distribution detruit. Le gain doit tomber a ~zero.

Les trois metriques de R7 :

* ``R = (L_ref - L_G) / (L_ref - L_prof)``, ``L_ref`` = unigramme 5,0829 ;
* fidelite ``KL(P_prof || P_G)`` sur ~5 000 contextes, **normalisee** par le KL
  entre deux professeurs de semences differentes (calcule en premier) ;
* top-3 sur 40 phrases **neuves** (grammaire du corpus, graine differente),
  normalise entre l'uniforme (3/312) et le professeur.

R8 : apres distillation, le metabolisme tourne LIBRE 1000 pas (naissances et
morts actives, aucun entrainement) ; R avant / R apres. Critere :
``R_apres >= 0,9 * R_avant``.

Le script tourne **par bras** (comme la validation : chaque appel reste court) :

    --arm teachers | distille:1 | melange:2 | variante | metriques | survie:1 | merge
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from lab_common import LAB, ToyCorpus, ensure_lab, read_json, set_seed, write_json
from make_lab_data import build_sentences
from micro_dense import DenseTransformer, n_parameters as dense_parameters, train_dense
from repaired_mycelia import (
    SEQ,
    aligned_loss,
    build_repaired,
    free_life,
    load_brain,
    save_brain,
    train_repaired,
)
from twins import (
    DENSE_BATCH,
    DENSE_LR,
    MYCELIA_BATCH,
    MYCELIA_LR,
    MYCELIA_OVERRIDES,
    MYCELIA_WD,
    dense_twin,
)

STEPS = 600
SEEDS = (1, 2, 3)
LAMBDA = 0.5
LAMBDA_ALT = 0.1
L_REF = 5.0829
FIDELITY_WINDOWS = 64
FIDELITY_BATCHES = 10
TOP3_SENTENCES = 40
TOP3_SEED = 20261123
FREE_STEPS = 1000


# ---------------------------------------------------------------------------
# Professeurs
# ---------------------------------------------------------------------------
def teacher_path(seed: int) -> Path:
    return LAB / f"s4_teacher_s{seed}.pt"


def train_teacher(corpus: ToyCorpus, seed: int, val) -> tuple:
    """Le dense jouet, memes reglages que le jumeau S2 (600 pas, patience 6)."""
    set_seed(seed)
    model = dense_twin(corpus.vocab_size, seed)
    result = train_dense(
        model, corpus, 600, lr=DENSE_LR, batch=DENSE_BATCH, seed=seed,
        log_every=100, patience=6, val_batches=val,
    )
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": {"dim": 16, "layers": 2, "heads": 2, "mlp": 166, "seq": SEQ},
            "vocab": list(corpus.vocab),
            "seed": int(seed),
            "best_val": result["final_val"],
            "best_step": result["best_step"],
            "curve": result["curve"],
        },
        str(teacher_path(seed)),
    )
    model.eval()
    return model, result


def load_teacher(seed: int):
    blob = torch.load(str(teacher_path(seed)), weights_only=False)
    model = DenseTransformer(
        len(blob["vocab"]), dim=blob["config"]["dim"], layers=blob["config"]["layers"],
        heads=blob["config"]["heads"], mlp=blob["config"]["mlp"], seq=blob["config"]["seq"],
    )
    model.load_state_dict(blob["state_dict"])
    model.eval()
    return model, blob


# ---------------------------------------------------------------------------
# Metriques
# ---------------------------------------------------------------------------
def relative_gain(loss_g: float, loss_prof: float, l_ref: float = L_REF) -> float:
    """R = (L_ref - L_G) / (L_ref - L_prof) : la part du chemin parcourue."""
    denom = float(l_ref) - float(loss_prof)
    if abs(denom) < 1e-9:
        return 0.0
    return round((float(l_ref) - float(loss_g)) / denom, 4)


@torch.no_grad()
def fidelity(student, teacher, corpus: ToyCorpus) -> dict:
    """KL(P_prof || P_G) sur les positions 1..7 de ~5 000 contextes."""
    total = 0.0
    count = 0
    generator = torch.Generator().manual_seed(4242)
    for _ in range(FIDELITY_BATCHES):
        x, _y = corpus.batch(FIDELITY_WINDOWS, generator=generator)
        s_logits = student(x, grow=False, trace=False)[0][:, 1:]
        _hidden, t_logits = teacher.parts(x)
        t_logits = t_logits[:, :-1]
        p = F.softmax(t_logits, dim=-1)
        logp = F.log_softmax(t_logits, dim=-1)
        logq = F.log_softmax(s_logits, dim=-1)
        total += float((p * (logp - logq)).sum(-1).sum())
        count += int(p.shape[0] * p.shape[1])
    return {"kl": round(total / max(1, count), 4), "contextes": int(count)}


@torch.no_grad()
def teacher_cross_kl(first, second, corpus: ToyCorpus) -> float:
    """L'accord inter-donneurs : KL symetrique entre deux professeurs.

    C'est la limite physique de la fidelite : deux dense entraines avec des
    graines differentes ne predisent pas la meme chose, donc exiger d'un eleve
    un KL plus petit que cet accord serait demander l'impossible.
    """
    total = 0.0
    count = 0
    generator = torch.Generator().manual_seed(4242)
    for _ in range(FIDELITY_BATCHES):
        x, _y = corpus.batch(FIDELITY_WINDOWS, generator=generator)
        _h1, l1 = first.parts(x)
        _h2, l2 = second.parts(x)
        l1, l2 = l1[:, :-1], l2[:, :-1]
        p1, p2 = F.softmax(l1, -1), F.softmax(l2, -1)
        kl12 = (p1 * (F.log_softmax(l1, -1) - F.log_softmax(l2, -1))).sum(-1).sum()
        kl21 = (p2 * (F.log_softmax(l2, -1) - F.log_softmax(l1, -1))).sum(-1).sum()
        total += float(kl12 + kl21) / 2.0
        count += int(p1.shape[0] * p1.shape[1])
    return round(total / max(1, count), 4)


def fresh_sentences(vocab: list[str], count: int = TOP3_SENTENCES) -> list[list[int]]:
    """40 phrases NEUVES de la meme grammaire (graine differente du corpus)."""
    stoi = {word: i for i, word in enumerate(vocab)}
    out: list[list[int]] = []
    for line in build_sentences(count * 4, seed=TOP3_SEED):
        words = line.split()
        if all(word in stoi for word in words) and len(words) >= 3:
            out.append([stoi[word] for word in words])
        if len(out) >= count:
            break
    if len(out) < count:
        raise SystemExit(f"pas assez de phrases neuves : {len(out)}/{count}")
    return out


@torch.no_grad()
def top3_mycelia(brain, sentences: list[list[int]]) -> dict:
    """Top-3 en lecture libre du vrai prefixe, token par token."""
    hits = 0
    total = 0
    for tokens in sentences:
        # Etat neuf : la premiere prediction n'a aucun contexte (comme dans
        # l'entrainement, ou la position 0 voit une membrane vierge).
        state = brain._fresh_state(1)
        for token in tokens:
            logits = brain.predict(state)[0]
            hits += int(token in torch.topk(logits, 3).indices.tolist())
            total += 1
            _, state, _ = brain(
                torch.tensor([[int(token)]]), state=state, grow=False, trace=False,
                return_state=True,
            )
    return {"top3": round(hits / max(1, total), 4), "tokens": int(total)}


@torch.no_grad()
def top3_dense(model, sentences: list[list[int]]) -> dict:
    """Le dense lit le prefixe vrai ; son pas ``t-1`` predit le token ``t``."""
    hits = 0
    total = 0
    for tokens in sentences:
        for index, token in enumerate(tokens):
            total += 1
            if index == 0:
                continue  # sans contexte : aucun modele n'a d'information
            prefix = torch.tensor([tokens[:index]], dtype=torch.long)
            logits = model(prefix)[0, -1]
            hits += int(token in torch.topk(logits, 3).indices.tolist())
    return {"top3": round(hits / max(1, total), 4), "tokens": int(total)}


def normalised(score: float, floor: float, ceiling: float) -> float:
    span = float(ceiling) - float(floor)
    if abs(span) < 1e-9:
        return 0.0
    return round((float(score) - float(floor)) / span, 4)


def measure(student, teacher, corpus: ToyCorpus, sentences, l_prof: float, inter_kl: float) -> dict:
    """Les trois metriques de R7 pour un eleve donne."""
    loss = aligned_loss(student, corpus.fixed(58, "val"))
    fid = fidelity(student, teacher, corpus)
    top = top3_mycelia(student, sentences)
    return {
        "perte": round(float(loss), 4),
        "R": relative_gain(loss, l_prof),
        "fidelite_kl": fid["kl"],
        "fidelite_contextes": fid["contextes"],
        "fidelite_normalisee": round(fid["kl"] / max(inter_kl, 1e-9), 4),
        "top3": top["top3"],
    }


# ---------------------------------------------------------------------------
# Bras
# ---------------------------------------------------------------------------
def arm_teachers(corpus: ToyCorpus) -> None:
    val = corpus.fixed(58, "val")
    teacher1, blob1 = train_teacher(corpus, 1, val)
    teacher2, blob2 = train_teacher(corpus, 2, val)
    # Controle de la section 2 : le dense jumeau sur le corpus 10k, **au meme
    # budget de 600 pas** que le MYCELIA repare (S3-bis ne l'avait entraine que
    # 300 pas : 3,7268). Sans ce chiffre, comparer 3,57 a 3,73 serait comparer
    # deux budgets.
    corpus10k = ToyCorpus(seq_len=SEQ, path=LAB / "corpus_toy10k.txt")
    val10k = corpus10k.fixed(100, "val")
    set_seed(1)
    dense10k = dense_twin(corpus10k.vocab_size, 1)
    dense10k_result = train_dense(
        dense10k, corpus10k, 600, lr=DENSE_LR, batch=DENSE_BATCH, seed=1,
        log_every=100, patience=0, val_batches=val10k,
    )
    print(f"dense 10k (600 pas) : {dense10k_result['final_val']:.4f}", flush=True)
    inter_kl = teacher_cross_kl(teacher1, teacher2, corpus)
    sentences = fresh_sentences(corpus.vocab)
    payload = {
        "s1": float(blob1["final_val"]),
        "s1_pas": int(blob1["best_step"]),
        "s2": float(blob2["final_val"]),
        "s2_pas": int(blob2["best_step"]),
        "kl_inter_professeurs": inter_kl,
        "top3_s1": top3_dense(teacher1, sentences)["top3"],
        "top3_s2": top3_dense(teacher2, sentences)["top3"],
        "parametres": int(dense_parameters(teacher1)),
        "phrases_neuves": len(sentences),
        "graine_phrases": TOP3_SEED,
        "dense_10k_600pas": round(float(dense10k_result["final_val"]), 4),
        "dense_10k_pas": int(dense10k_result["best_step"]),
    }
    write_json(LAB / "s4_teachers.json", payload)
    print(f"professeurs : s1 {payload['s1']:.4f} / s2 {payload['s2']:.4f} — "
          f"KL inter {inter_kl:.4f}, top3 {payload['top3_s1']} / {payload['top3_s2']}", flush=True)


def arm_train(corpus: ToyCorpus, kind: str, seed: int) -> None:
    teacher, _blob = load_teacher(1)
    val = corpus.fixed(58, "val")
    lam = LAMBDA_ALT if kind == "variante" else LAMBDA
    perm = None
    if kind == "melange":
        perm = torch.randperm(corpus.vocab_size, generator=torch.Generator().manual_seed(77))
    brain = build_repaired(corpus.vocab_size, seed, two_speeds=True, overrides=MYCELIA_OVERRIDES)
    started = time.time()
    result = train_repaired(
        brain, corpus, STEPS, lr=MYCELIA_LR, batch=MYCELIA_BATCH, seed=seed,
        log_every=50, weight_decay=MYCELIA_WD, val_batches=val,
        teacher=teacher, lam=lam, teacher_perm=perm,
    )
    name = "distille_lam0.1" if kind == "variante" else kind
    path = LAB / f"s4_{name}_s{seed}.pt"
    save_brain(path, result["brain"], step=result["best_step"])
    fragment = {
        "bras": kind, "semence": seed, "lambda": lam,
        "pas": result["best_step"], "secondes": round(time.time() - started, 1),
        "val_honnete": result["best_val"],
        "courbe": [
            {"step": r["step"], "val": r["val"], "kl": r["kl"], "align": r["align"]}
            for r in result["curve"]
        ],
    }
    write_json(LAB / f"s4_train_{name}_s{seed}.json", fragment)
    print(f"{kind} s{seed} : val honnete {result['best_val']:.4f} au pas {result['best_step']} "
          f"({fragment['secondes']}s) — dernier KL/align "
          f"{result['curve'][-1]['kl']}/{result['curve'][-1]['align']}", flush=True)


def arm_metrics(corpus: ToyCorpus) -> None:
    teachers = read_json(LAB / "s4_teachers.json")
    teacher, _blob = load_teacher(1)
    sentences = fresh_sentences(corpus.vocab)
    inter_kl = float(teachers["kl_inter_professeurs"])
    l_prof = float(teachers["s1"])
    rows: dict = {}
    for arm in ("M0", "distille", "melange", "variante"):
        rows[arm] = []
        # La variante lambda n'a qu'une semence : la boucler sur trois
        # afficherait trois fois le meme modele.
        seeds = (1,) if arm == "variante" else SEEDS
        for seed in seeds:
            if arm == "M0":
                path = LAB / f"s4_1000_deux_vitesses_s{seed}.pt"
            elif arm == "variante":
                path = LAB / "s4_distille_lam0.1_s1.pt"
            else:
                path = LAB / f"s4_{arm}_s{seed}.pt"
            if not path.exists():
                continue
            brain = load_brain(path)
            row = measure(brain, teacher, corpus, sentences, l_prof, inter_kl)
            row["nom"] = f"{arm} s{seed if arm != 'variante' else 1}"
            row["semence"] = seed if arm != "variante" else 1
            rows[arm].append(row)
            print(f"  {row['nom']:12s} perte {row['perte']:.4f} R {row['R']:+.4f} "
                  f"KL {row['fidelite_kl']:.3f} top3 {row['top3']}", flush=True)
    payload = {
        "professeurs": teachers,
        "bras": rows,
        "phrases_neuves": {
            "n": len(sentences), "graine": TOP3_SEED,
            "top3_professeur": teachers["top3_s1"],
            "top3_professeur_2": teachers["top3_s2"],
            "top3_uniforme": round(3.0 / corpus.vocab_size, 4),
        },
    }
    write_json(LAB / "s4_metriques.json", payload)
    print("ecrit : data/lab/s4_metriques.json", flush=True)


def arm_survival(corpus: ToyCorpus, seed: int) -> None:
    teachers = read_json(LAB / "s4_teachers.json")
    l_prof = float(teachers["s1"])
    val = corpus.fixed(58, "val")
    brain = load_brain(LAB / f"s4_distille_s{seed}.pt")
    before = relative_gain(aligned_loss(brain, val), l_prof)
    ne_before = int(brain.ne)
    life = free_life(brain, corpus, steps=FREE_STEPS, batch=MYCELIA_BATCH, seed=seed)
    after = relative_gain(aligned_loss(brain, val), l_prof)
    row = {
        "semence": seed,
        "R_avant": before,
        "R_apres": after,
        "ratio": round(after / before, 4) if abs(before) > 1e-9 else None,
        "survit": bool(abs(before) > 1e-9 and after >= 0.9 * before),
        "aretes_debut": ne_before,
        **life,
    }
    write_json(LAB / f"s4_survie_s{seed}.json", row)
    print(f"survie s{seed} : R {before:+.4f} -> {after:+.4f} (ratio {row['ratio']}), "
          f"aretes {ne_before}->{life['aretes_fin']}, morts {life['aretes_mortes']}, "
          f"survit {row['survit']}", flush=True)


def merge(corpus: ToyCorpus) -> None:
    metriques = read_json(LAB / "s4_metriques.json")
    rows = metriques["bras"]

    def mean_of(values):
        values = [float(v) for v in values]
        mean = sum(values) / len(values)
        sd = 0.0 if len(values) < 2 else (
            sum((v - mean) ** 2 for v in values) / (len(values) - 1)
        ) ** 0.5
        return round(mean, 4), round(sd, 4)

    summary = {}
    for arm, arm_rows in rows.items():
        if not arm_rows:
            continue
        perte, perte_sd = mean_of([r["perte"] for r in arm_rows])
        gain, gain_sd = mean_of([r["R"] for r in arm_rows])
        top3, top3_sd = mean_of([r["top3"] for r in arm_rows])
        fid, fid_sd = mean_of([r["fidelite_kl"] for r in arm_rows])
        summary[arm] = {
            "perte_moyenne": perte, "perte_ecart_type": perte_sd,
            "R_moyen": gain, "R_ecart_type": gain_sd,
            "top3_moyen": top3, "top3_ecart_type": top3_sd,
            "top3_normalise": normalised(
                top3,
                metriques["phrases_neuves"]["top3_uniforme"],
                metriques["phrases_neuves"]["top3_professeur"],
            ),
            "fidelite_kl_moyenne": fid, "fidelite_ecart_type": fid_sd,
            "fidelite_normalisee": round(
                fid / max(float(metriques["professeurs"]["kl_inter_professeurs"]), 1e-9), 4
            ),
            "semences": len(arm_rows),
        }

    survival_rows = []
    for seed in SEEDS:
        path = LAB / f"s4_survie_s{seed}.json"
        if path.exists():
            survival_rows.append(read_json(path))
    survie = {
        "pas_libres": FREE_STEPS,
        "lignes": survival_rows,
        "R_avant_moyen": mean_of([r["R_avant"] for r in survival_rows])[0] if survival_rows else None,
        "R_apres_moyen": mean_of([r["R_apres"] for r in survival_rows])[0] if survival_rows else None,
        "survit": sum(1 for r in survival_rows if r["survit"]),
        "sur_semences": len(survival_rows),
        "critere": "R_apres >= 0.9 * R_avant",
    }

    trains = {}
    for name in ("distille", "melange"):
        for seed in SEEDS:
            path = LAB / f"s4_train_{name}_s{seed}.json"
            if path.exists():
                trains[f"{name}_s{seed}"] = read_json(path)
    gain_isole = None
    if "distille" in summary and "M0" in summary:
        gain_isole = round(summary["distille"]["R_moyen"] - summary["M0"]["R_moyen"], 4)
    controle = None
    if "distille" in summary and "melange" in summary:
        ecart = round(summary["distille"]["R_moyen"] - summary["melange"]["R_moyen"], 4)
        controle = {
            "R_melange": summary["melange"]["R_moyen"],
            "R_distille": summary["distille"]["R_moyen"],
            "ecart": ecart,
            "propre": bool(ecart > 0.05),
        }
    payload = {
        "protocole": {
            "pas": STEPS, "semences": list(SEEDS), "lambda": LAMBDA,
            "lambda_alterne": LAMBDA_ALT, "l_ref": L_REF,
            "l_professeur": float(metriques["professeurs"]["s1"]),
            "formule": "KL(logits_MYC || logits_prof) + lambda * MSE(P h_MYC, h_prof)",
            "projecteur": "initialise a l'identite, jete apres usage",
            "donnees": "corpus 1000 phrases, logits du professeur calcules a la volee",
            "M0": "meme budget et memes lots que la distillation, sans professeur "
                  "(bras deux_vitesses de la validation)",
        },
        "professeurs": metriques["professeurs"],
        "bras": metriques["bras"],
        "entrainements": trains,
        "resume": summary,
        "gain_distillation_isole": gain_isole,
        "controle_bruit": controle,
        "survie": survie,
        "phrases_neuves": metriques["phrases_neuves"],
    }
    write_json(LAB / "s4_distill.json", payload)
    print("— resume")
    for arm, row in summary.items():
        print(f"  {arm:8s} perte {row['perte_moyenne']:.4f} ± {row['perte_ecart_type']:.4f} "
              f"R {row['R_moyen']:+.4f} fid {row['fidelite_normalisee']:.2f} "
              f"top3 {row['top3_moyen']:.4f} (norm {row['top3_normalise']:.2f})", flush=True)
    print(f"gain de distillation isole : {gain_isole}", flush=True)
    print(f"survie : {survie['R_avant_moyen']} -> {survie['R_apres_moyen']} "
          f"({survie['survit']}/{survie['sur_semences']})", flush=True)
    print("ecrit : data/lab/s4_distill.json", flush=True)


def main() -> None:
    ensure_lab()
    parser = argparse.ArgumentParser(description="distillation S4")
    parser.add_argument("--arm", action="append", default=[],
                        help="teachers | distille:N | melange:N | variante | metriques | survie:N | merge")
    args = parser.parse_args()
    corpus = ToyCorpus(seq_len=SEQ)
    for arm in args.arm:
        parts = arm.split(":")
        kind = parts[0]
        seed = int(parts[1]) if len(parts) > 1 else 1
        if kind == "teachers":
            arm_teachers(corpus)
        elif kind in ("distille", "melange", "variante"):
            arm_train(corpus, kind, seed)
        elif kind == "metriques":
            arm_metrics(corpus)
        elif kind == "survie":
            arm_survival(corpus, seed)
        elif kind == "merge":
            merge(corpus)
        else:
            raise SystemExit(f"bras inconnu : {arm}")


if __name__ == "__main__":
    main()
