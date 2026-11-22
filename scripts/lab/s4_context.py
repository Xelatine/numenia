"""S4 R3/R4 — le test de contexte et les sondes, avant et apres reparation.

Copie locale du protocole D2 de S3-BIS (memes trois conditions, memes sondes,
meme budget), appliquee deux fois : au NUMENIA de S2 (avant) et au NUMENIA
repare (apres). La seule difference assumee est le nombre de fenetres de
validation : 58, c'est-a-dire **toutes** celles du flux tenu a part (D2 en
prenait 64 sur 58 disponibles).

Trois conditions, memes cibles, meme position, seul le prefixe change :

* ``normal``  — le prefixe reel du flux ;
* ``melange`` — le meme prefixe, permute (le sac de mots est identique) ;
* ``vide``    — un prefixe neutre.

Objectif chiffre de la mission : l'ecart normal/melange doit depasser **0,5 nat**
apres reparation ; sinon la tete n'exploite pas la memoire et la reparation est
cosmetique.

Sondes identiques partout (meme softmax, meme budget, meme regularisation) :

1. ``noeud1``      — la membrane du noeud 1 (16 dims), l'ancienne lecture ;
2. ``milieu``      — les 6 noeuds du milieu (96 dims) ;
3. ``tous``        — les 8 membranes (128 dims) ;
4. ``lecture``     — le vecteur de lecture du modele repare (16 dims) : ce que
   la tete utilise vraiment, et non ce qui est disponible ;
5. ``embeddings``  — les embeddings bruts du prefixe (temoin de largeur).

Chaque sonde recoit en plus l'indice de position (7 colonnes) : sans lui la meme
cible tombe dans un bloc de caracteristiques different selon ``t``.

Ecrit ``data/lab/s4_context.json``.
"""

from __future__ import annotations

import random

import numpy as np
import torch
import torch.nn.functional as F

from lab_common import LAB, ToyCorpus, read_json, write_json
from micro_mycelia import load_mycelia
from repaired_mycelia import SEQ, aligned_loss, load_brain, per_position_loss, shifted_loss

VAL_WINDOWS = 58
TRAIN_WINDOWS = 400
CONDITIONS = ("normal", "melange", "vide")
SEED = 20261120


def condition_rows(batch, condition: str, rng: random.Random):
    """Lignes ``[prefixe(t), pad...]`` pour t = 1..7, et leurs cibles."""
    rows: list[list[int]] = []
    targets: list[int] = []
    positions: list[int] = []
    for x, _y in batch:
        tokens = [int(v) for v in x[0].tolist()]
        for t in range(1, SEQ):
            prefix = tokens[:t]
            if condition == "melange":
                prefix = list(prefix)
                rng.shuffle(prefix)
            elif condition == "vide":
                prefix = [0] * t
            rows.append(prefix + [0] * (SEQ - t))
            targets.append(tokens[t])
            positions.append(t)
    return (
        torch.tensor(rows, dtype=torch.long),
        torch.tensor(targets, dtype=torch.long),
        torch.tensor(positions, dtype=torch.long),
    )


@torch.no_grad()
def condition_loss(brain, batch, condition: str, rng: random.Random) -> dict:
    inputs, targets, positions = condition_rows(batch, condition, rng)
    logits = brain(inputs, grow=False, trace=False)[0]
    picked = logits[torch.arange(inputs.shape[0]), positions]
    loss = float(F.cross_entropy(picked, targets))
    acc = float((picked.argmax(-1) == targets).float().mean())
    entropy = float(
        -(torch.softmax(picked, -1) * torch.log_softmax(picked, -1)).sum(-1).mean()
    )
    return {
        "perte": round(loss, 4),
        "exactitude": round(acc, 4),
        "entropie_predite": round(entropy, 4),
        "logits": picked,
        "cibles": targets,
    }


def _position_onehot(t: int) -> np.ndarray:
    row = np.zeros(SEQ - 1, dtype=np.float64)
    row[t - 1] = 1.0
    return row


@torch.no_grad()
def collect_states(brain, batch):
    """Etats de tous les noeuds avant chaque token (t = 1..7) + position."""
    states: list[np.ndarray] = []
    next_ids: list[int] = []
    for x, _y in batch:
        state = None
        tokens = [int(v) for v in x[0].tolist()]
        for t in range(SEQ):
            if t >= 1:
                row = state[:, 0, :].clone().numpy().reshape(-1)
                states.append(np.concatenate([row, _position_onehot(t)]))
                next_ids.append(tokens[t])
            _, state, _ = brain(
                torch.tensor([[tokens[t]]]), state=state, grow=False, trace=False,
                return_state=True,
            )
    return np.asarray(states, dtype=np.float64), np.asarray(next_ids, dtype=np.int64)


@torch.no_grad()
def collect_readings(brain, batch):
    """Le vecteur de lecture du modele repare (ce que la tete utilise)."""
    rows: list[np.ndarray] = []
    labels: list[int] = []
    for x, _y in batch:
        logits, state, _ = brain(x, grow=False, trace=False, return_state=True)
        readings = getattr(brain, "readout_rows", lambda: torch.zeros(0))()
        tokens = [int(v) for v in x[0].tolist()]
        for t in range(1, SEQ):
            if readings.numel() == 0:
                rows.append(np.concatenate([np.zeros(int(brain.dim)), _position_onehot(t)]))
            else:
                rows.append(
                    np.concatenate(
                        [readings[0, t].numpy().reshape(-1), _position_onehot(t)]
                    )
                )
            labels.append(tokens[t])
    return np.asarray(rows, dtype=np.float64), np.asarray(labels, dtype=np.int64)


@torch.no_grad()
def collect_embeddings(brain, batch):
    rows: list[np.ndarray] = []
    for x, _y in batch:
        embedded = brain.embedding(x[0]).numpy()
        for t in range(1, SEQ):
            prefix = np.zeros((SEQ, embedded.shape[1]), dtype=np.float64)
            prefix[:t] = embedded[:t]
            rows.append(np.concatenate([prefix.reshape(-1), _position_onehot(t)]))
    return np.asarray(rows, dtype=np.float64)


def softmax_probe(
    Xtr, ytr, Xva, yva, vocab: int, steps: int = 400, lr: float = 0.02,
    weight_decay: float = 1e-2,
):
    """Sonde lineaire entrainee (softmax), calibration correcte de la perte."""
    mu = Xtr.mean(0)
    sd = Xtr.std(0) + 1e-8
    A = torch.tensor((Xtr - mu) / sd, dtype=torch.float32)
    B = torch.tensor((Xva - mu) / sd, dtype=torch.float32)
    W = torch.zeros(A.shape[1], vocab, requires_grad=True)
    b = torch.zeros(vocab, requires_grad=True)
    opt = torch.optim.Adam([W, b], lr=lr, weight_decay=weight_decay)
    ta = torch.tensor(ytr, dtype=torch.long)
    tb = torch.tensor(yva, dtype=torch.long)
    with torch.enable_grad():
        for _ in range(int(steps)):
            loss = F.cross_entropy(A @ W + b, ta)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    with torch.no_grad():
        logits = B @ W + b
        ce = float(F.cross_entropy(logits, tb))
        acc = float((logits.argmax(-1) == tb).float().mean())
    return {"perte": round(ce, 4), "exactitude": round(acc, 4), "dims": int(A.shape[1])}


def analyze(brain, corpus: ToyCorpus, label: str, floors: dict) -> dict:
    """Le protocole complet sur un cerveau (trois conditions, sondes, routage)."""
    vocab = corpus.vocab_size
    val = corpus.fixed(VAL_WINDOWS, "val")
    results = {}
    for index, condition in enumerate(CONDITIONS):
        rng = random.Random(SEED + index)
        results[condition] = condition_loss(brain, val, condition, rng)
    normal, shuffle, empty = (results[c] for c in CONDITIONS)

    def kl(first, second) -> float:
        p = torch.softmax(first["logits"], -1)
        q = torch.log_softmax(second["logits"], -1)
        return float((p * (torch.log(p.clamp_min(1e-12)) - q)).sum(-1).mean())

    kl_shuffle = kl(normal, shuffle)
    kl_empty = kl(normal, empty)

    train_fixed = corpus.fixed(TRAIN_WINDOWS, "train")
    Xtr, ntr = collect_states(brain, train_fixed)
    Xva, nva = collect_states(brain, val)
    Etr = collect_embeddings(brain, train_fixed)
    Eva = collect_embeddings(brain, val)
    Rtr, rtr = collect_readings(brain, train_fixed)
    Rva, rva = collect_readings(brain, val)

    def subset(X, start: int, stop: int):
        return np.hstack([X[:, start:stop], X[:, -7:]])

    probes = {
        "noeud1_16d": softmax_probe(subset(Xtr, 0, 16), ntr, subset(Xva, 0, 16), nva, vocab),
        "milieu_96d": softmax_probe(subset(Xtr, 16, 112), ntr, subset(Xva, 16, 112), nva, vocab),
        "tous_128d": softmax_probe(subset(Xtr, 0, 128), ntr, subset(Xva, 0, 128), nva, vocab),
        "lecture": softmax_probe(Rtr, rtr, Rva, rva, vocab),
        "embeddings_128d": softmax_probe(Etr, ntr, Eva, nva, vocab),
    }

    # ---- routage : energie, portes, identifiabilite du dernier token -----
    x0 = val[0][0]
    brain(x0, grow=False, trace=True)
    energy = [round(float(v), 4) for v in brain.last_energy_by_hop]
    ids = torch.arange(brain.n_nodes)
    state = None
    for t in range(SEQ):
        _, state, _ = brain(
            torch.tensor([[int(x0[0, t])]]), state=state, grow=False, trace=False,
            return_state=True,
        )
    _out, gate = brain.bank.forward_rows(state, ids)
    gates = [round(float(g), 3) for g in gate.flatten()]
    silent = [i for i, g in enumerate(gates) if g < 0.1]

    def decodability(repeats: int, rng: random.Random) -> float:
        rows: list[np.ndarray] = []
        labels: list[int] = []
        for token in range(vocab):
            for _ in range(3):
                s = None
                for j in range(repeats):
                    tok = token if j == repeats - 1 else rng.randrange(vocab)
                    _, s, _ = brain(
                        torch.tensor([[int(tok)]]), state=s, grow=False, trace=False,
                        return_state=True,
                    )
                rows.append(s[:, 0, :].numpy().reshape(-1))
                labels.append(token)
        X = np.asarray(rows, dtype=np.float64)
        y = np.asarray(labels, dtype=np.int64)
        mask = np.arange(len(y)) % 3 != 2
        return softmax_probe(
            X[mask], y[mask], X[~mask], y[~mask], vocab, steps=300
        )["exactitude"]

    rng = random.Random(SEED + 7)
    decode_2 = decodability(2, rng)
    decode_7 = decodability(7, rng)

    spread = max(
        abs(normal["perte"] - shuffle["perte"]),
        abs(normal["perte"] - empty["perte"]),
    )
    return {
        "nom": label,
        "fenetres_validation": len(val),
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
        "kl_normal_vs_melange": round(kl_shuffle, 5),
        "kl_normal_vs_vide": round(kl_empty, 5),
        "perte_par_position": per_position_loss(brain, val),
        "sondes": probes,
        "routage": {
            "energie_par_saut": energy,
            "portes": gates,
            "noeuds_muets": silent,
            "decodage_dernier_token_2": decode_2,
            "decodage_dernier_token_7": decode_7,
            "aretes": int(brain.ne),
        },
        "honnete": round(aligned_loss(brain, val), 4),
        "decalee": round(shifted_loss(brain, val), 4),
        "contexte_utilise": bool(spread >= 0.5 and kl_shuffle >= 0.02),
        "planchers": floors,
    }


def main() -> None:
    torch.set_grad_enabled(False)
    corpus = ToyCorpus(seq_len=SEQ)
    floors = read_json(LAB / "s3bis_floors.json")
    reference = {
        "unigramme": floors["unigramme"]["entropie_lissee"],
        "bigramme": floors["bigramme"]["entropie_lissee"],
        "trigramme": floors["trigramme"]["entropie_lissee"],
    }

    old = load_mycelia(str(LAB / "s2_mycelia_s1.pt"))
    print("— avant : NUMENIA de S2 (lecture du noeud 1)")
    before = analyze(old, corpus, "avant (S2)", reference)
    print(f"  conditions : {before['conditions']} — ecart {before['ecart_max_conditions']}")
    print(f"  sondes : { {k: v['perte'] for k, v in before['sondes'].items()} }")

    repaired = load_brain(LAB / "s4_1000_deux_vitesses_s1.pt")
    print("— apres : NUMENIA repare (deux vitesses)")
    after = analyze(repaired, corpus, "apres (S4)", reference)
    print(f"  conditions : {after['conditions']} — ecart {after['ecart_max_conditions']}")
    print(f"  sondes : { {k: v['perte'] for k, v in after['sondes'].items()} }")

    payload = {
        "protocole": {
            "fenetres_validation": VAL_WINDOWS,
            "fenetres_sonde_entrainement": TRAIN_WINDOWS,
            "conditions": list(CONDITIONS),
            "seuil_ecart_contexte": 0.5,
        },
        "avant": before,
        "apres": after,
        "verdict": {
            "ecart_avant": before["ecart_max_conditions"],
            "ecart_apres": after["ecart_max_conditions"],
            "ecart_large": bool(after["ecart_max_conditions"] > 0.5),
            "noeud1_avant": before["sondes"]["noeud1_16d"]["perte"],
            "noeud1_apres": after["sondes"]["noeud1_16d"]["perte"],
            "tous_avant": before["sondes"]["tous_128d"]["perte"],
            "tous_apres": after["sondes"]["tous_128d"]["perte"],
        },
    }
    write_json(LAB / "s4_context.json", payload)
    print("verdict contexte :", payload["verdict"])
    print("ecrit : data/lab/s4_context.json")


if __name__ == "__main__":
    main()
