"""S2 — conversion dans les deux sens sur les jumeaux egaux, x 3 semences.

Protocole de S3, inchange : amorcage de 50 pas, greffe, 100 pas d'affinage, et
le meme budget pour un controle parti exactement du meme point. Deux choses
changent, et ce sont les corrections demandees :

1. **l'alignement** ne porte plus sur des familles figees mais sur les
   activations brutes des 2000 stimuli : le score d'une paire est la
   correlation de Pearson de deux profils, et l'appariement est optimal
   (hongrois) au lieu d'etre mutuel. Un concept n'est *traduit* que si sa
   correlation depasse 0,6 ; sinon il est *greffe* — la table separe les deux.
2. **la mesure** ne porte plus sur une seule execution : trois semences par
   sens, un test t apparie, et un gain n'est retenu que si ``p < 0,05`` **et**
   ``gain > +1 %``. En dessous, c'est du bruit de regularisation.

Le test de langue passe a 40 phrases hors corpus et compte l'exactitude en
TOP-3 : sur des micro-cerveaux a 312 mots, le top-1 a 0 % ne distingue plus
rien, alors que « le mot juste est dans les trois propositions » reste une
mesure honnete de ce qui a ete appris.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from lab_common import (
    LAB,
    STIMULI,
    ToyCorpus,
    align_correlation,
    dense_loss,
    ensure_lab,
    kmeans,
    mycelia_loss,
    paired_t,
    profiles,
    read_json,
    set_seed,
    write_json,
)
from micro_dense import n_parameters as dense_parameters
from micro_dense import train_dense
from micro_mycelia import n_parameters as mycelia_parameters
from micro_mycelia import train_mycelia
from translate_lib import (
    arm_dense,
    arm_mycelia,
    batches_loss_dense,
    batches_loss_mycelia,
    concept_direction,
    dense_clone,
    dense_directions,
    heldout_sentences,
    implant_node,
    make_batches,
    mycelia_clone,
    mycelia_directions,
    projection,
    scan_dense,
    scan_mycelia,
    summary,
    test_dense,
    test_mycelia,
)
from twins import (
    SEQ,
    load_dense_twin,
    load_mycelia_twin,
)

AMORCE = 50
STEPS = 100
BATCH = 16
LR = 1e-3
K = 8
SEUIL = 0.6
S2_DIR = LAB / "s2"

_CACHE: dict[int, dict] = {}


def stimuli_texts() -> list[str]:
    """Les 2000 stimuli, categorie retiree (le scan ne lit que du texte)."""
    out = []
    for line in STIMULI.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        out.append(line.split("|", 1)[1] if "|" in line else line)
    return out


def units_of(model) -> int:
    """Nombre d'unites cachees du dense : 2 blocs x MLP."""
    return 2 * int(model.blocks[0].fc1.out_features)


def family_raw(matrix: np.ndarray, labels: np.ndarray, k: int) -> np.ndarray:
    """Profil **brut** d'une famille : moyenne des activations de ses membres.

    Pas de normalisation ici : la correlation de Pearson centre et reduit
    elle-meme chaque profil, donc l'amplitude moyenne d'une famille n'entre pas
    dans le score. Ce qui compte est la forme du profil sur les 2000 stimuli.
    """
    rows = []
    for family in range(int(k)):
        members = np.nonzero(labels == family)[0]
        if members.size == 0:
            rows.append(np.zeros(matrix.shape[0], dtype=np.float32))
            continue
        rows.append(matrix[:, members].mean(axis=1))
    return np.stack(rows).astype(np.float32)


def scans_for_seed(seed: int, corpus: ToyCorpus, texts: list[str]) -> dict:
    """Scan des deux jumeaux d'une semence, calcule une fois pour les deux sens."""
    if seed in _CACHE:
        return _CACHE[seed]
    started = time.time()
    dense = load_dense_twin(seed)
    brain = load_mycelia_twin(seed)
    dscan = scan_dense(dense, corpus, texts)
    mscan = scan_mycelia(brain, corpus, texts)
    alive = brain.node_alive.cpu().numpy()
    nodes = np.nonzero(alive)[0]
    block = mscan[:, nodes]
    keep = block.max(axis=0) > 0
    nodes = nodes[keep]
    _CACHE[seed] = {
        "dense_scan": dscan,
        "mycelia_scan": mscan[:, nodes],
        "mycelia_nodes": nodes,
        "seconds": round(time.time() - started, 1),
    }
    print(
        f"  scan semence {seed} : dense {dscan.shape}, "
        f"NUMENIA {len(nodes)} noeuds actifs ({_CACHE[seed]['seconds']} s)"
    )
    return _CACHE[seed]


def source_block(scan: dict, direction: str, dense_units: int):
    """(matrice des unites, directions) du cote source."""
    if direction == "dense_to_mycelia":
        return scan["dense_scan"][:, :dense_units], "dense"
    return scan["mycelia_scan"], "mycelia"


def receiver_block(scan: dict, direction: str, dense_units: int):
    if direction == "dense_to_mycelia":
        return scan["mycelia_scan"], "mycelia"
    return scan["dense_scan"][:, :dense_units], "dense"


def directions_of(kind: str, model_or_brain, units: int) -> np.ndarray:
    if kind == "dense":
        return dense_directions(model_or_brain)[:units]
    return mycelia_directions(model_or_brain)


def top_shared(prof_s: np.ndarray, prof_r: np.ndarray, texts: list[str], top: int = 3) -> list[dict]:
    both = np.minimum(prof_s, prof_r)
    order = np.argsort(-both)[: int(top)]
    return [
        {"texte": texts[int(i)], "force": round(float(both[int(i)]), 4)}
        for i in order
        if both[int(i)] > 0
    ]


def run_direction(direction: str, seed: int, corpus: ToyCorpus, texts: list[str], sentences: list[str]) -> dict:
    set_seed(seed)
    started = time.time()
    dense = load_dense_twin(seed)
    brain = load_mycelia_twin(seed)
    scan = scans_for_seed(seed, corpus, texts)
    dense_units = units_of(dense)

    s_matrix, s_kind = source_block(scan, direction, dense_units)
    r_matrix, r_kind = receiver_block(scan, direction, dense_units)

    # ---- 1. cartographie de la source -----------------------------------
    labels_s, _, inertia_s = kmeans(profiles(s_matrix), K, seed=11)
    prof_s_raw = family_raw(s_matrix, labels_s, K)
    dirs_s = directions_of(s_kind, dense if s_kind == "dense" else brain, dense_units)

    # ---- 2. cartographie du receveur ------------------------------------
    labels_r, _, inertia_r = kmeans(profiles(r_matrix), K, seed=11)
    prof_r_raw = family_raw(r_matrix, labels_r, K)
    dirs_r = directions_of(r_kind, dense if r_kind == "dense" else brain, dense_units)
    members_s = {i: np.nonzero(labels_s == i)[0] for i in range(K)}
    members_r = {j: np.nonzero(labels_r == j)[0] for j in range(K)}

    # ---- 3. alignement par correlation, appariement optimal --------------
    matching = align_correlation(prof_s_raw.T, prof_r_raw.T, threshold=SEUIL)
    translated = list(matching["traduits"])
    grafted_concepts = list(matching["greffes"])
    pairs = list(matching["pairs"])

    table = []
    for concept in range(K):
        proposal = matching["proposition"].get(concept)
        partner, score = (int(proposal[0]), float(proposal[1])) if proposal else (-1, 0.0)
        vide = int(members_s[concept].size) == 0
        table.append(
            {
                "concept_source": int(concept),
                "unites_source": int(members_s[concept].size),
                "partenaire_propose": int(partner),
                "correlation": round(score, 4),
                "meilleure_correlation": round(float(matching["best"][concept]), 4),
                "statut": "vide" if vide else ("traduit" if concept in translated else "greffe"),
                "unites_receveuses": int(members_r[partner].size) if partner in members_r else 0,
                "stimuli_communs": (
                    top_shared(prof_s_raw[concept], prof_r_raw[partner], texts)
                    if partner >= 0
                    else []
                ),
            }
        )

    # ---- projection et transplant ----------------------------------------
    matched_s = [i for i, _, _ in pairs]
    matched_r = [j for _, j, _ in pairs]
    src_dirs = (
        np.stack([concept_direction(dirs_s, members_s[i]) for i in matched_s])
        if len(matched_s) >= 2
        else np.zeros((0, dirs_s.shape[1]))
    )
    dst_dirs = (
        np.stack([concept_direction(dirs_r, members_r[j]) for j in matched_r])
        if len(matched_r) >= 2
        else np.zeros((0, dirs_r.shape[1]))
    )
    W = (
        projection(src_dirs, dst_dirs)
        if src_dirs.shape[0] >= 2
        else np.eye(dirs_s.shape[1], dirs_r.shape[1], dtype=np.float32)
    )
    with np.errstate(invalid="ignore", divide="ignore"):
        co = np.nan_to_num(np.clip(np.corrcoef(prof_s_raw), -1.0, 1.0), nan=0.0)

    # ---- 4. amorcage du receveur (50 pas), puis greffe / controle --------
    val_batches = corpus.fixed(24, "val")
    batches = make_batches(corpus, STEPS, BATCH, seed=1000 + seed)

    if r_kind == "mycelia":
        warm = train_mycelia(
            mycelia_clone(brain), corpus, AMORCE, lr=LR, batch=BATCH, seed=seed,
            log_every=AMORCE, patience=0, weight_decay=0.05,
        )
        base = warm.pop("brain")
        before = {
            "val": round(float(mycelia_loss(base, val_batches)), 4),
            "train": round(float(batches_loss_mycelia(base, batches[:10])), 4),
        }
        grafted = mycelia_clone(base)
    else:
        # ``train_dense`` modifie le modele en place et le restaure sur sa
        # meilleure validation : on garde donc la reference du clone.
        base = dense_clone(dense)
        train_dense(
            base, corpus, AMORCE, lr=LR, batch=BATCH, seed=seed,
            log_every=AMORCE, patience=0,
        )
        before = {
            "val": round(float(dense_loss(base, val_batches)), 4),
            "train": round(float(batches_loss_dense(base, batches[:10])), 4),
        }
        grafted = dense_clone(base)

    implanted = []
    for concept in grafted_concepts:
        if members_s[concept].size == 0:
            # Famille vide : il n'y a rien a greffer, et projeter un vecteur nul
            # ne ferait qu'ajouter un noeud muet. Le statut « vide » de la table
            # le dit deja ; on ne fabrique pas un transplant fantome.
            continue
        vec = W @ concept_direction(dirs_s, members_s[concept])
        partners = []
        for other, (i, j, _score) in enumerate(pairs):
            if float(co[concept, i]) >= 0.5:
                partners.extend(int(n) for n in members_r[j][:2])
        if r_kind == "mycelia":
            node = implant_node(grafted, vec, partners=partners)
            implanted.append(
                {
                    "concept": int(concept),
                    "noeud": int(node),
                    "partenaires": int(len(partners)),
                    "norme_projetee": round(float(np.linalg.norm(vec)), 4),
                }
            )
        else:
            width = int(grafted.blocks[0].fc1.out_features)
            grafted.grow_mlp(0, torch.as_tensor(vec))
            implanted.append(
                {
                    "concept": int(concept),
                    "neurone": width,
                    "largeur_apres": int(grafted.blocks[0].fc1.out_features),
                    "norme_projetee": round(float(np.linalg.norm(vec)), 4),
                }
            )

    control = mycelia_clone(base) if r_kind == "mycelia" else dense_clone(base)
    if r_kind == "mycelia":
        graft_arm = arm_mycelia(grafted, batches, val_batches, lr=LR)
        ctrl_arm = arm_mycelia(control, batches, val_batches, lr=LR)
        final_graft = graft_arm["brain"]
        final_ctrl = ctrl_arm["brain"]
    else:
        graft_arm = arm_dense(grafted, batches, val_batches, lr=LR)
        ctrl_arm = arm_dense(control, batches, val_batches, lr=LR)
        final_graft = graft_arm["model"]
        final_ctrl = ctrl_arm["model"]

    # ---- 5. test de langue, 40 phrases, top-3 ----------------------------
    def language(model_or_brain, kind: str) -> dict:
        if kind == "dense":
            return test_dense(model_or_brain, corpus, sentences)
        return test_mycelia(model_or_brain, corpus, sentences)

    twin_ref = language(dense if r_kind == "dense" else brain, r_kind)
    after_amorce = language(base, r_kind)
    lang_graft = language(final_graft, r_kind)
    lang_ctrl = language(final_ctrl, r_kind)

    verdict = summary(before, graft_arm, ctrl_arm)
    verdict.update(
        {
            "traduits": len(translated),
            "greffes": len(grafted_concepts),
            "top3": {
                "jumeau_reference": twin_ref["exactitude_top3"],
                "apres_amorce": after_amorce["exactitude_top3"],
                "greffe": lang_graft["exactitude_top3"],
                "controle": lang_ctrl["exactitude_top3"],
            },
            "top1": {
                "jumeau_reference": twin_ref["exactitude"],
                "apres_amorce": after_amorce["exactitude"],
                "greffe": lang_graft["exactitude"],
                "controle": lang_ctrl["exactitude"],
            },
            "perte_langue": {
                "jumeau_reference": twin_ref["perte"],
                "apres_amorce": after_amorce["perte"],
                "greffe": lang_graft["perte"],
                "controle": lang_ctrl["perte"],
            },
            "phrases_gagnees_vs_controle": int(
                sum(1 for a, b in zip(lang_graft["exemples"], lang_ctrl["exemples"]) if a["perte"] < b["perte"])
            ),
            "phrases": len(sentences),
        }
    )

    payload = {
        "sens": "dense -> numenia" if direction == "dense_to_mycelia" else "numenia -> dense",
        "semence": int(seed),
        "meta": {
            "stimuli": len(texts),
            "k": K,
            "seuil": SEUIL,
            "amorce": AMORCE,
            "steps": STEPS,
            "batch": BATCH,
            "lr": LR,
            "parametres_dense": int(dense_parameters(dense)),
            "parametres_mycelia": int(mycelia_parameters(brain)),
            "unites_source": int(s_matrix.shape[1]),
            "unites_receveuses": int(r_matrix.shape[1]),
            "inertie_source": round(float(inertia_s), 4),
            "inertie_receveuse": round(float(inertia_r), 4),
            "secondes": round(time.time() - started, 1),
        },
        "table_alignement": table,
        "concepts_traduits": [int(i) for i in translated],
        "concepts_greffes": [int(i) for i in grafted_concepts],
        "transplants": implanted,
        "verdict": verdict,
        "courbes": {"greffe": graft_arm["curve"], "controle": ctrl_arm["curve"]},
        "predictions_greffe": lang_graft["exemples"],
        "predictions_controle": lang_ctrl["exemples"],
    }
    write_json(S2_DIR / f"{direction}_s{seed}.json", payload)
    print(
        f"  {payload['sens']} semence {seed} : {len(translated)}/8 traduits, "
        f"{len(grafted_concepts)} greffes — avant {verdict['avant']:.4f} → "
        f"greffe {verdict['apres_greffe']:.4f} / controle {verdict['controle']:.4f} "
        f"({verdict['gain_vs_controle']:+.2%}) top3 greffe "
        f"{lang_graft['exactitude_top3']:.0%} ({payload['meta']['secondes']} s)"
    )
    return payload


def aggregate(seeds: list[int]) -> dict:
    """Moyennes par sens et test t apparie sur les semences disponibles."""
    out = {"semences": [], "sens": {}, "decision": {}}
    found = {seed: {} for seed in seeds}
    for direction in ("dense_to_mycelia", "mycelia_to_dense"):
        rows = []
        for seed in seeds:
            path = S2_DIR / f"{direction}_s{seed}.json"
            if not path.exists():
                continue
            rows.append(read_json(path))
            found[seed][direction] = True
        if not rows:
            continue
        grafts = [row["verdict"]["apres_greffe"] for row in rows]
        controls = [row["verdict"]["controle"] for row in rows]
        befores = [row["verdict"]["avant"] for row in rows]
        test = paired_t(grafts, controls)
        top3 = [row["verdict"]["top3"] for row in rows]
        mean = lambda values: round(float(np.mean(values)), 4)  # noqa: E731
        decision = {
            "gain_significatif": bool(test["significatif"]),
            "gain_moyen": test["gain_moyen"],
            "p": test["p"],
            "traduits_moyenne": round(float(np.mean([row["verdict"]["traduits"] for row in rows])), 2),
            "top3_greffe_moyenne": round(float(np.mean([row["greffe"] for row in top3])), 4),
            "top3_controle_moyenne": round(float(np.mean([row["controle"] for row in top3])), 4),
        }
        out["sens"][direction] = {
            "n": len(rows),
            "avant_moyen": mean(befores),
            "greffe_moyen": mean(grafts),
            "controle_moyen": mean(controls),
            "greffe_par_semence": grafts,
            "controle_par_semence": controls,
            "avant_par_semence": befores,
            "test_t": test,
            "decision": decision,
            "top3_par_semence": top3,
            "traduits_par_semence": [row["verdict"]["traduits"] for row in rows],
        }
    out["semences"] = sorted(seed for seed in seeds if found[seed])
    best_translated = max(
        (item["decision"]["traduits_moyenne"] for item in out["sens"].values()), default=0.0
    )
    significant = any(item["decision"]["gain_significatif"] for item in out["sens"].values())
    top3_best = max(
        (item["decision"]["top3_greffe_moyenne"] for item in out["sens"].values()), default=0.0
    )
    if best_translated >= 5 and significant and top3_best > 0.20:
        verdict = "TRADUCTEUR ÉTABLI"
    elif best_translated < 3:
        verdict = "MUR CONFIRMÉ"
    else:
        verdict = "PARTIEL"
    out["decision"] = {
        "traduits_max_moyenne": best_translated,
        "gain_significatif_un_sens": significant,
        "top3_greffe_max": top3_best,
        "verdict": verdict,
    }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="conversion S2 : jumeaux egaux, 3 semences")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--sens", default="both", choices=["both", "dense_to_mycelia", "mycelia_to_dense"])
    parser.add_argument("--aggreger-seulement", action="store_true")
    args = parser.parse_args()

    ensure_lab()
    S2_DIR.mkdir(parents=True, exist_ok=True)
    corpus = ToyCorpus(seq_len=SEQ)
    texts = stimuli_texts()
    sentences = heldout_sentences(corpus, 40)
    print(f"stimuli {len(texts)}, phrases hors corpus {len(sentences)}")

    if not args.aggreger_seulement:
        directions = (
            ["dense_to_mycelia", "mycelia_to_dense"]
            if args.sens == "both"
            else [args.sens]
        )
        for seed in args.seeds:
            for direction in directions:
                run_direction(direction, int(seed), corpus, texts, sentences)

    stats = aggregate([int(seed) for seed in args.seeds])
    write_json(LAB / "s2_stats.json", stats)
    for direction, item in stats["sens"].items():
        print(
            f"{direction:20s} greffe {item['greffe_moyen']:.4f} / controle "
            f"{item['controle_moyen']:.4f} ({item['test_t']['gain_moyen']:+.2%}, "
            f"t={item['test_t']['t']}, p={item['test_t']['p']}, "
            f"{'RETENU' if item['test_t']['significatif'] else 'REJETE'})"
        )
    print(f"verdict : {stats['decision']}")


if __name__ == "__main__":
    main()
