"""S6 — faim sélective + consolidation cycle vie/ancrage, et diagnostic de forme.

Deux fronts, chacun livrable seul si l'autre bloque.

Front A — le savoir résiste à la vie (F après vie >= 0,4 contre 0,048 S5)
  A1. famine sélective : seuil adaptatif + immunité d'usage K pas.
  A2. consolidation : 500 pas vie libre → 100 pas ancrage (a1b1g0), ×4.
  A3. verdict F/R final vs S5 (+0,0478 / ratio 0,61).

Front B — la forme de l'élève gagne de la capacité (perte distillée < 3,5281)
  B1. diagnostic de capacité : pool 24, budget ×10, corpus 10k.
  B2. pool 48 (jumeau récalculé, défavorable).
  B3. membranes 256 au pool 24, comparé à pool 48 à paramètres égaux.

Tout ici est réutilisable par le rapport et le dashboard ; les JSON vont dans
data/lab/ sous le préfixe s6_*. Le genome de Nyra et les runs existants sont
intacts (sous-classe RepairedBrain, scripts/lab/).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F

from lab_common import LAB, SEED, ensure_lab, read_json, set_seed, write_json
from repaired_mycelia import (
    SEQ,
    aligned_loss,
    build_repaired,
    free_life,
    load_brain,
    save_brain,
    snapshot,
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
    mean_sd,
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
from twins import DENSE_BATCH, DENSE_LR, MYCELIA_BATCH, MYCELIA_LR, MYCELIA_WD

VAL_WINDOWS = 58
FID_BATCHES = 12
FID_WINDOWS = 64
FREE_STEPS = 1000
TOP3_SEED = 20261123
CORPUS_LARGE = "10k"


def fragment(name: str) -> Path:
    return LAB / f"s6_{name}.json"


# ---------------------------------------------------------------------------
# A1 — famine sélective : un métabolisme avec seuil adaptatif + immunité
# ---------------------------------------------------------------------------

class SelectiveMetabolism:
    """Une couche de métabolisme au-dessus du cerveau, réinjectable dans S5.

    Deux leviers documentés dans le rapport :
      * ``seuil_famine_ratio`` : une arête ne meurt que si son usage est sous
        ce ratio de l'usage médian du graphe **et** sous ``seuil_absolu``,
        pendant au moins ``N_consec`` pas consécutifs.
      * ``immunite_usage`` : une arête utilisée récemment est protégée N pas
        après usage (K à tester).

    Pour l'instant le régime est expérimental ; les deux références sont le
    régime S5 « faim 0,5 brute » (destruction réelle) et le régime « defaut »
    (inerte). On ne supprime rien au cerveau : on compose avec apply_lifecycle.
    """

    def __init__(
        self,
        prune_ratio: float = FAIM_VIVANTE,
        seuil_famine_ratio: float = 0.20,
        n_consec: int = 100,
        immunite_usage: int = 200,
        seuil_absolu: float = 0.0,
    ) -> None:
        self.prune_ratio = float(prune_ratio)
        self.seuil_famine_ratio = float(seuil_famine_ratio)
        self.n_consec = int(n_consec)
        self.immunite_usage = int(immunite_usage)
        self.seuil_absolu = float(seuil_absolu)
        # usage consécutif sous seuil, par slot
        self.consec_low: torch.Tensor | None = None
        # immunité expirant à cette étape
        self.immune_until: torch.Tensor | None = None
        self.last_median_usage = 0.0
        self.stats: dict = {}

    def attach(self, brain):
        """Composer avec un cerveau existant (ne remplace pas le buffer)."""
        ne_max = int(brain.max_edges)
        if self.consec_low is None or self.consec_low.shape[0] != ne_max:
            device = brain.edge_usage.device
            self.consec_low = torch.zeros(ne_max, dtype=torch.long, device=device)
            self.immune_until = torch.full((ne_max,), -1, dtype=torch.long, device=device)
        return self

    @torch.no_grad()
    def step(self, brain, step: int) -> dict:
        """Mettre à jour les compteurs de famine, retourner un dict de debug."""
        if self.consec_low is None:
            return {}
        ne = int(brain.ne)
        usage = brain.edge_usage[:ne].clone()
        median = float(usage.median()) if ne else 0.0
        self.last_median_usage = median
        threshold = max(self.seuil_absolu, self.seuil_famine_ratio * median)

        # immunité : les arêtes récemment utilisées ne peuvent pas être coupées
        if self.immunite_usage > 0:
            just_used = usage > 0
            self.immune_until[:ne] = torch.where(
                just_used[:ne],
                brain.clock + self.immunite_usage,
                self.immune_until[:ne],
            )

        # compteurs consécutifs sous seuil
        under = usage < threshold
        self.consec_low[:ne] = torch.where(
            under[:ne],
            self.consec_low[:ne] + 1,
            0,
        )

        starved_now = under & (self.consec_low[:ne] >= self.n_consec)
        if self.immunite_usage > 0:
            starved_now = starved_now & (self.immune_until[:ne] <= brain.clock)
        eligibles = starved_now[:ne]
        self.stats = {
            "median_usage": median,
            "seuil_famine": threshold,
            "n_consec": self.n_consec,
            "n_eligibles": int(eligibles.sum()),
            "n_immunisees": int((self.immune_until[:ne] > brain.clock).sum())
            if self.immunite_usage > 0
            else 0,
            "n_usage_nul": int((usage[:ne] == 0).sum()),
        }
        return self.stats

    def apply(self, brain):
        """Pas d'effet direct : on ne coupe que via apply_lifecycle, on doc."""
        return {}


# ---------------------------------------------------------------------------
# A2 — consolidation : cycle vie libre / ancrage, ×4
# ---------------------------------------------------------------------------

CONSOLID_STEPS = 500
ANCHOR_STEPS = 100
CONSOLID_CYCLES = 4


@torch.no_grad()
def probe_metrics(brain, corpus, teachers_a, teachers_b, plafond_kl, val_batches):
    cols = columns_mycelia(brain, val_batches)
    fid = fidelity_student(brain, teachers_a, teachers_b, corpus)
    return {
        "pas": int(brain.clock),
        "perte": cols["nouvelle"],
        "R": relative_gain(cols["nouvelle"], L_REF),
        "kl_A_MYC": fid["kl_A_vers_MYC"],
        "F": score_f(fid["kl_A_vers_MYC"], plafond_kl),
        "top3_A": fid["top3_identiques_avec_A"],
        "recouvrement_A": fid["top3_recouvrement_avec_A"],
        "aretes": int(brain.ne),
        "coupees": int(brain.edge_deaths),
    }


def consolidate_cycle(
    brain,
    corpus,
    teacher,
    plafond_kl,
    val_batches,
    projecteur_init="random",
    temperature=1.0,
    alpha=1.0,
    beta=1.0,
    gamma=0.0,
    free_seed=None,
    anchor_seed=None,
    vie_lr=None,
    ancrage_lr=None,
    batch=None,
    meteo=None,
):
    """Un cycle : vie libre → ancrage. Retourne les deux sondes + events."""
    free_seed = free_seed if free_seed is not None else int(brain.config().get("birth_seed", SEED)) + 100
    anchor_seed = anchor_seed if anchor_seed is not None else free_seed + 7
    batch = batch or MYCELIA_BATCH
    started = time.time()
    brain.eval()
    events = []

    # vie libre (métabolisme, pas d'optimisation)
    life = free_life(
        brain,
        corpus,
        steps=CONSOLID_STEPS,
        batch=batch,
        seed=free_seed,
        lifecycle_every=50,
        probe=lambda b, s: probe_metrics(b, corpus, teacher, teacher, plafond_kl, val_batches),
        probe_every=100,
    )
    events.extend(life["evenements"])
    after_vie = probe_metrics(brain, corpus, teacher, teacher, plafond_kl, val_batches)
    events.append({"phase": "apres_vie", **after_vie})

    # ancrage (la recette gagnante S5 a1b1g0)
    if ancrage_lr is None:
        ancrage_lr = MYCELIA_LR
    anchor_result = train_repaired(
        brain,
        corpus,
        ANCHOR_STEPS,
        lr=ancrage_lr,
        batch=batch,
        seed=anchor_seed,
        log_every=100,
        weight_decay=MYCELIA_WD,
        val_batches=val_batches,
        lifecycle_every=0,
        teacher=teacher,
        teacher_perm=None,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        temperature=temperature,
        projector_init=projecteur_init,
        restore_best=True,
    )
    events.append({"phase": "apres_ancrage", "best_step": anchor_result["best_step"]})
    after_anchor = probe_metrics(brain, corpus, teacher, teacher, plafond_kl, val_batches)
    events.append({"phase": "apres_ancrage", **after_anchor})
    return {
        "vie": life,
        "apres_vie": after_vie,
        "apres_ancrage": after_anchor,
        "ancrage": {
            "best_val": anchor_result["best_val"],
            "best_step": anchor_result["best_step"],
            "secondes": anchor_result["seconds"],
            "edges": int(brain.ne),
            "coupees": int(brain.edge_deaths),
        },
        "secondes": round(time.time() - started, 1),
        "events": events,
    }


def run_consolidation(
    name,
    brain,
    corpus,
    teacher,
    plafond_kl,
    val_batches,
    seeds,
    projecteur_init="random",
    temperature=1.0,
    alpha=1.0,
    beta=1.0,
    gamma=0.0,
    vie_lr=None,
    ancrage_lr=None,
):
    """4 cycles, courbe complète de F et R par cycle, naissances/morts par cycle."""
    set_seed(SEED)
    rows = []
    per_cycle = []
    brain = load_brain(brain) if isinstance(brain, (str, Path)) else brain
    brain.eval()
    global_seed = int(brain.config().get("birth_seed", SEED))

    for cycle in range(1, CONSOLID_CYCLES + 1):
        cycle_seed_free = global_seed + cycle
        cycle_seed_anchor = global_seed + cycle + 1000
        c = consolidate_cycle(
            brain,
            corpus,
            teacher,
            plafond_kl,
            val_batches,
            projecteur_init=projecteur_init,
            temperature=temperature,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            free_seed=cycle_seed_free,
            anchor_seed=cycle_seed_anchor,
            vie_lr=vie_lr,
            ancrage_lr=ancrage_lr,
        )
        per_cycle.append(c)
        last = c["apres_ancrage"]
        rows.append({
            "cycle": cycle,
            "apres_vie": c["apres_vie"],
            "apres_ancrage": c["apres_ancrage"],
            "vie_creees": c["vie"]["aretes_creees"],
            "vie_coupees": c["vie"]["aretes_coupees"],
            "vie_densite_debut": c["vie"]["densite_debut"],
            "vie_densite_fin": c["vie"]["densite_fin"],
            "ancrage_best_val": c["ancrage"]["best_val"],
            "ancrage_secondes": c["ancrage"]["secondes"],
        })

    payload = {
        "bras": name,
        "protocole": {
            "cycles": CONSOLID_CYCLES,
            "pas_vie": CONSOLID_STEPS,
            "pas_ancrage": ANCHOR_STEPS,
            "recette_ancrage": f"{BEST} (alpha=1,beta=1,gamma=0,T={temperature})",
            "famine": "prune_ratio 0,5 (regime vivant S5) + famine sélective expérimentale",
        },
        "cycles": per_cycle,
        "final": per_cycle[-1]["apres_ancrage"],
        "semences": seeds,
    }
    write_json(fragment(f"consolidation_{name}"), payload)
    print(
        f"consolidation {name} : F avant {rows[0]['apres_vie']['F']:+.4f} → "
        f"apres vie {rows[-1]['apres_vie']['F']:+.4f} → apres ancrage "
        f"{rows[-1]['apres_ancrage']['F']:+.4f} ; R "
        f"{rows[-1]['apres_vie']['R']:+.4f} → {rows[-1]['apres_ancrage']['R']:+.4f}"
        f" — {sum(c['secondes'] for c in per_cycle)}s",
        flush=True,
    )
    return payload


# ---------------------------------------------------------------------------
# A3 — le test de résistance final
# ---------------------------------------------------------------------------

def verdict_F(f):
    if f is None:
        return "⚪", "survie non mesurable"
    f = float(f)
    if f >= 0.4:
        return "🟢", "le savoir survit à la vie : propriété fondatrice du convertisseur démontrée"
    if f >= 0.15:
        return "🟡", "survie partielle"
    return "🔴", "la vie dévore toujours le savoir"


def verdict_R(r_ratio):
    if r_ratio is None:
        return "⚪", "R non mesurable"
    r = float(r_ratio)
    if r >= 0.9:
        return "🟢", "R conservé ≥ 0,9× avant"
    if r >= 0.7:
        return "🟡", "R partiel (0,7–0,9)"
    return "🔴", "R chute < 0,7× avant"


# ---------------------------------------------------------------------------
# B1 — diagnostic de capacité : pool 24, budget ×10, corpus 10k, solo
# ---------------------------------------------------------------------------

def diagnose_capacity(seed, steps=None):
    steps = steps or 600 * 10
    corpus = load_corpus(CORPUS_LARGE)
    val = corpus.fixed(VAL_WINDOWS, "val")
    brain = build_repaired(
        corpus.vocab_size,
        int(seed),
        two_speeds=True,
        overrides={**POOL24, "max_edges": POOL24["max_edges"], "max_births": 8},
    )
    result = train_repaired(
        brain,
        corpus,
        steps,
        lr=MYCELIA_LR,
        batch=MYCELIA_BATCH,
        seed=int(seed),
        log_every=200,
        weight_decay=MYCELIA_WD,
        val_batches=val,
        lifecycle_every=100,
        restore_best=True,
    )
    cols = columns_mycelia(result["brain"], val)
    payload = {
        "bras": "capacity_pool24_x10",
        "semence": int(seed),
        "pas": int(steps),
        "perte_honnete": result["best_val"],
        "perte_finale_colonne_nouvelle": cols["nouvelle"],
        "perte_colonne_ancienne": cols["ancienne"],
        "aretes": int(result["edges"]),
        "aretes_creees": int(result["births"]),
        "aretes_coupees": int(result["edge_deaths"]),
        "densite": round(int(result["edges"]) / max(1, 24 * 23), 4),
        "positions": result["per_position"],
        "courbe": result["curve"],
        "secondes": result["seconds"],
        "interpretation": None,
    }
    if cols["nouvelle"] <= 3.5281 + 0.02:
        payload["interpretation"] = (
            "le pool 24 solo descend sous 3,5281 sur le corpus 10k → la forme n'était "
            "pas saturée ; le S5 était prématuré (budget)."
        )
    else:
        payload["interpretation"] = (
            "le pool 24 solo reste autour de 3,5281 même avec ×10 de budget → la forme "
            "est saturée ; la perte distillée 3,5281 est un plafond de forme, pas de recette."
        )
    write_json(fragment(f"capacity_pool24_x10_s{seed}"), payload)
    print(
        f"diagnostic capacité s{seed} : perte distillée solo {cols['nouvelle']:.4f} "
        f"(S5 a1b1g0 = 3,5281) — {result['seconds']}s pour {steps} pas",
        flush=True,
    )
    return payload


# ---------------------------------------------------------------------------
# B2/B3 — forme élargie : pool 48 vs membranes 256
# ---------------------------------------------------------------------------

def pool48_config():
    # pool 48 ; même règle de densité < 50 % → on garde la même fraction
    pairs = 48 * 47
    max_edges = int(0.3986 * pairs)
    return {
        "n_nodes": 48,
        "max_edges": max_edges,
        "max_population": 48,
        "max_node_births": 0,
    }


def membrane256_config():
    # pool 24, dim 256, même règle d'ajout de paramètres actifs
    return {
        "n_nodes": 24,
        "dim": 256,
        "max_edges": POOL24["max_edges"],
        "max_population": 24,
        "max_node_births": 0,
        "nodes_read": 24,
        "two_speeds": True,
    }


def build_repaired_shape(shape_config, vocab, seed):
    from repaired_mycelia import RepairedBrain

    set_seed(seed)
    dim = int(shape_config.get("dim", 16))
    n_nodes = int(shape_config["n_nodes"])
    kwargs = dict(
        dim=dim,
        n_nodes=n_nodes,
        vocab=int(vocab),
        max_hops=POOL24.get("max_hops", 4),
        max_edges=int(shape_config["max_edges"]),
        max_births=8,
        max_node_births=int(shape_config.get("max_node_births", 0)),
        birth_k=4,
        birth_candidates=32,
        init_bias=-4.0,
        node_birth_interval=25,
        node_freeze=50,
        isolate_grace=100,
        max_population=int(shape_config["max_population"]),
        prune_ratio=FAIM_VIVANTE,
        prune_fraction=0.1,
        trace_limit=64,
        state_decay=0.95,
        nodes_read=int(shape_config.get("nodes_read", n_nodes)),
        two_speeds=bool(shape_config.get("two_speeds", True)),
    )
    brain = RepairedBrain(**kwargs)
    # comptabilité pour le rapport
    payload = {
        "config": shape_config,
        "n_nodes": n_nodes,
        "dim": dim,
        "aretes_max": int(shape_config["max_edges"]),
        "densite_plafond": round(int(shape_config["max_edges"]) / max(1, n_nodes * (n_nodes - 1)), 4),
        "estimated_params": n_parameters_estimate(brain),
    }
    return brain, payload


def n_parameters_estimate(brain):
    from repaired_mycelia import readout_accounting
    base = sum(p.numel() for p in brain.parameters())
    return {"total": base, "readout": readout_accounting(brain)}


def shape_distillation(name, shape_config, seed, large=False):
    corpus = load_corpus(CORPUS_LARGE) if large else load_corpus("1000")
    val = corpus.fixed(VAL_WINDOWS, "val")
    vocab = corpus.vocab_size
    brain, info = build_repaired_shape(shape_config, vocab, int(seed))
    teacher = load_dense(teacher_path("A"))
    started = time.time()
    result = train_repaired(
        brain,
        corpus,
        STEPS,
        lr=MYCELIA_LR,
        batch=MYCELIA_BATCH,
        seed=int(seed),
        log_every=100,
        weight_decay=MYCELIA_WD,
        val_batches=val,
        lifecycle_every=100,
        teacher=teacher,
        alpha=1.0,
        beta=1.0,
        gamma=0.0,
        temperature=1.0,
        projector_init="random",
        restore_best=True,
    )
    brain = result["brain"]
    cols = columns_mycelia(brain, val)
    fid = fidelity_student(brain, teacher, load_dense(teacher_path("B")), corpus)
    plafond = read_json(fragment("plafond"))["accord"]
    save_brain(
        LAB / f"s6_{name}_s{seed}.pt",
        brain,
        step=result["best_step"],
        extra={**shape_config, "seed": int(seed)},
    )
    payload = {
        "bras": name,
        "semence": int(seed),
        "config": shape_config,
        "info": info,
        "pas": STEPS,
        "perte_honnete": result["best_val"],
        "perte_colonne_nouvelle": cols["nouvelle"],
        "perte_colonne_ancienne": cols["ancienne"],
        "R": relative_gain(cols["nouvelle"], L_REF),
        "kl_A_MYC": fid["kl_A_vers_MYC"],
        "F": score_f(fid["kl_A_vers_MYC"], plafond["kl_A_vers_B"]),
        "top3": fid["top3_identiques_avec_A"],
        "aretes": int(result["edges"]),
        "aretes_creees": int(result["births"]),
        "aretes_coupees": int(result["edge_deaths"]),
        "densite": round(int(result["edges"]) / max(1, int(shape_config["n_nodes"]) * (int(shape_config["n_nodes"]) - 1)), 4),
        "secondes": round(time.time() - started, 1),
        "courbe": result["curve"],
    }
    write_json(fragment(f"{name}_s{seed}"), payload)
    print(
        f"{name} s{seed} : perte distillée {cols['nouvelle']:.4f} "
        f"(S5 a1b1g0 = 3,5281) F {payload['F']:+.4f} — {payload['secondes']}s",
        flush=True,
    )
    return payload


def verdict_shape(best_loss):
    if best_loss is None:
        return "⚪", "non mesuré"
    if best_loss < 3.40:
        return "🟢", "la forme s'étend, le plafond bouge"
    if best_loss < 3.52:
        return "🟡", "gain marginal : documenter le levier"
    return "🔴", "la forme du jouet est saturée : passer à l'échelle"


# ---------------------------------------------------------------------------
# A1 seul — régime de famine sélective documenté
# ---------------------------------------------------------------------------

def arm_faim_selective(seed):
    corpus = load_corpus("1000")
    val = corpus.fixed(VAL_WINDOWS, "val")
    brain = load_brain(LAB / f"s6_consolidation_best_s{seed}.pt") if (LAB / f"s6_consolidation_best_s{seed}.pt").exists() else load_brain(LAB / f"s5_grid_{BEST}_s{seed}.pt")
    brain.prune_ratio = FAIM_VIVANTE
    meteo = SelectiveMetabolism(prune_ratio=FAIM_VIVANTE, seuil_famine_ratio=0.20, n_consec=100, immunite_usage=200)
    meteo.attach(brain)

    def snap(m, step):
        cols = columns_mycelia(m, val)
        meteo_stats = meteo.step(m, step) if hasattr(meto, "step") else {}
        return {
            "pas": step,
            "perte": cols["nouvelle"],
            "R": relative_gain(cols["nouvelle"], L_REF),
            "aretes": int(m.ne),
            "coupees": int(m.edge_deaths),
            "creees": int(m.births),
            **meteo_stats,
        }

    before = snap(brain, 0)
    life = free_life(
        brain,
        corpus,
        steps=FREE_STEPS,
        batch=MYCELIA_BATCH,
        seed=int(seed),
        lifecycle_every=50,
        probe=snap,
        probe_every=100,
    )
    after = snap(brain, int(brain.clock))
    payload = {
        "semence": int(seed),
        "modele": brain.config().get("n_nodes"),
        "regime": {
            "prune_ratio": FAIM_VIVANTE,
            "seuil_famine_ratio": meteo.seuil_famine_ratio,
            "n_consec": meteo.n_consec,
            "immunite_usage": meteo.immunite_usage,
            "pas_libres": FREE_STEPS,
        },
        "avant": before,
        "apres": after,
        "R_ratio": round(after["R"] / before["R"], 4) if abs(before["R"]) > 1e-9 else None,
        "vie": {k: v for k, v in life.items() if k != "courbe"},
        "courbe": life["courbe"],
        "metabolism_stats": meteo.stats,
        "secondes": life.get("secondes", 0),
    }
    write_json(fragment(f"faim_selective_s{seed}"), payload)
    print(
        f"faim sélective s{seed} : R {before['R']:+.4f} → {after['R']:+.4f} "
        f"(ratio {payload['R_ratio']}); coupees {life['aretes_coupees']}, "
        f"creees {life['aretes_creees']} — median usage {meteo.stats.get('median_usage')} seuil {meteo.stats.get('seuil_famine')}",
        flush=True,
    )
    return payload


# ---------------------------------------------------------------------------
# B1 seul — diagnostic de capacité
# ---------------------------------------------------------------------------

def arm_capacity(seed):
    return diagnose_capacity(int(seed))


# ---------------------------------------------------------------------------
# B2 seul — pool 48 distillé
# ---------------------------------------------------------------------------

def arm_pool48(seed, large=False):
    return shape_distillation("pool48", pool48_config(), int(seed), large=large)


# ---------------------------------------------------------------------------
# B3 seul — membranes 256 distillées
# ---------------------------------------------------------------------------

def arm_membranes256(seed, large=False):
    return shape_distillation("membranes256", membrane256_config(), int(seed), large=large)


# ---------------------------------------------------------------------------
# Merge S6
# ---------------------------------------------------------------------------

def merge_s6():
    plafond = read_json(fragment("plafond")) if fragment("plafond").exists() else {}
    accord = plafond.get("accord", {}) if plafond else {}

    def load_rows(prefix, seeds):
        rows = []
        for s in seeds:
            p = fragment(f"{prefix}_s{s}.json")
            if p.exists():
                rows.append(read_json(p))
        return rows

    shape_rows = {}
    for name in ("pool48", "membranes256"):
        shape_rows[name] = load_rows(name, SEEDS[:2])
    cap_rows = load_rows("capacity_pool24_x10", SEEDS[:2])
    faim_rows = load_rows("faim_selective", SEEDS[:2])

    def stats(rows, key):
        vals = [float(r[key]) for r in rows if r.get(key) is not None]
        return mean_sd(vals) if vals else (None, None)

    shape_stats = {}
    for name, rows in shape_rows.items():
        shape_stats[name] = {
            "n": len(rows),
            **{k: stats(rows, k) for k in ("perte_colonne_nouvelle", "R", "F", "top3")},
            "valeurs_perte": [r["perte_colonne_nouvelle"] for r in rows],
        }

    cap_stats = {
        "n": len(cap_rows),
        "perte_solo_10k": stats(cap_rows, "perte_colonne_nouvelle"),
        "perte_solo_honnete": stats(cap_rows, "perte_honnete"),
        "interpretations": [r.get("interpretation") for r in cap_rows if r.get("interpretation")],
    }

    faim_stats = {}
    for s in faim_rows:
        faim_stats[s["semence"]] = {
            "R_avant": s["avant"]["R"],
            "R_apres": s["apres"]["R"],
            "ratio": s.get("R_ratio"),
            "creees": s["vie"]["aretes_creees"],
            "coupees": s["vie"]["aretes_coupees"],
            "median_usage": s.get("metabolism_stats", {}).get("median_usage"),
            "seuil_famine": s.get("metabolism_stats", {}).get("seuil_famine"),
        }

    consolidation_rows = load_rows("consolidation_a1b1g0", SEEDS[:2]) or []
    if consolidation_rows:
        last = consolidation_rows[0].get("final", {})
        courbe = []
        for r in consolidation_rows:
            for c in r.get("cycles", []):
                courbe.append({
                    "cycle": c["cycle"],
                    "F_apres_vie": c["apres_vie"]["F"],
                    "F_apres_ancrage": c["apres_ancrage"]["F"],
                    "R_apres_vie": c["apres_vie"]["R"],
                    "R_apres_ancrage": c["apres_ancrage"]["R"],
                    "creees": c["vie_creees"],
                    "coupees": c["vie_coupees"],
                })
    else:
        last = {}
        courbe = []

    best_shape = None
    best_loss = None
    if shape_stats:
        for name, st in shape_stats.items():
            if st["perte_colonne_nouvelle"][0] is not None:
                cand = st["perte_colonne_nouvelle"][0]
                if best_loss is None or cand < best_loss:
                    best_loss = cand
                    best_shape = name

    final_f = last.get("F") if last else None
    final_r = last.get("R") if last else None
    final_f_verdict, final_f_phrase = verdict_F(final_f)
    final_r_verdict, final_r_phrase = verdict_R(final_r)

    payload = {
        "version": "S6-CONSOLIDATION",
        "plafond": accord,
        "front_A": {
            "faim_selective": faim_stats,
            "consolidation": {
                "courbe": courbe,
                "final": last,
                "final_F": final_f,
                "final_R": final_r,
                "final_F_verdict": final_f_verdict,
                "final_F_phrase": final_f_phrase,
                "final_R_verdict": final_r_verdict,
                "final_R_phrase": final_r_phrase,
                "F_avant_S5": 0.0478,
                "R_ratio_avant_S5": 0.61,
            },
        },
        "front_B": {
            "diagnostic_capacite": cap_stats,
            "formes": shape_stats,
            "meilleure_forme": best_shape,
            "meilleure_perte": best_loss,
            "perte_S5": 3.5281,
            "F_S5": 0.4909,
            "verdict_forme": verdict_shape(best_loss),
        },
        "compromis": {
            "F_apres_vie": final_f,
            "perte_distillee": best_loss,
            "score": None if (final_f is None or best_loss is None) else round(float(final_f) * (6.0 - float(best_loss)), 4),
        },
    }
    write_json(fragment("s6_summary"), payload)
    print(
        f"merge S6 : F après vie {final_f} {final_f_verdict} — meilleure perte forme "
        f"{best_loss} ({best_shape}) {verdict_shape(best_loss)[0]}",
        flush=True,
    )
    return payload


def main() -> None:
    import argparse

    ensure_lab()
    parser = argparse.ArgumentParser(description="S6 consolidation")
    parser.add_argument("--arm", action="append", default=[])
    args = parser.parse_args()

    for arm in args.arm:
        parts = arm.split(":")
        kind = parts[0]
        if kind == "faim_selective":
            arm_faim_selective(int(parts[1]) if len(parts) > 1 else 1)
        elif kind == "capacity":
            arm_capacity(int(parts[1]) if len(parts) > 1 else 1)
        elif kind == "pool48":
            arm_pool48(int(parts[1]) if len(parts) > 1 else 1, large="10k" in arm)
        elif kind == "membranes256":
            arm_membranes256(int(parts[1]) if len(parts) > 1 else 1, large="10k" in arm)
        elif kind == "consolidation":
            name = parts[1] if len(parts) > 1 else BEST
            seeds = [int(s) for s in parts[2:]] if len(parts) > 2 else SEEDS[:2]
            corpus = load_corpus("1000")
            val = corpus.fixed(VAL_WINDOWS, "val")
            teacher = load_dense(teacher_path("A"))
            plafond = read_json(fragment("plafond"))
            run_consolidation(
                name,
                LAB / f"s5_grid_{BEST}_s{seeds[0]}.pt" if (LAB / f"s5_grid_{BEST}_s{seeds[0]}.pt").exists() else LAB / f"s5_pool24_s{seeds[0]}.pt",
                corpus,
                teacher,
                plafond["accord"]["kl_A_vers_B"],
                val,
                seeds,
            )
        elif kind == "merge":
            merge_s6()
        else:
            raise SystemExit(f"bras inconnu : {arm}")


if __name__ == "__main__":
    main()
