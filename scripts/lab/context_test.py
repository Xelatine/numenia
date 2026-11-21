"""S3-bis D2 — le test de contexte (inference seulement).

MYCELIA jouet de S2 (semence 1) atteint 5,078 de validation, exactement
l'entropie unigramme du corpus (D1). Deux lectures possibles : le modele ignore
le contexte, ou il l'utilise si mal que le gain est nul. Ce script separe les
deux et, si le contexte est perdu, designe le composant fautif.

Trois conditions, **memes cibles**, meme position, seul le prefixe change :

* ``normal``   — le prefixe reel du flux ;
* ``melange``  — le meme prefixe, permute aleatoirement (le sac de mots est
  identique, l'ordre est detruit) ;
* ``vide``     — un prefixe neutre (``<pad>``), donc aucune information.

Pour chaque fenetre de validation et chaque position ``t`` de 1 a 7, on construit
une ligne ``[prefixe(t), <pad> ...]`` et on lit les logits a l'indice ``t`` : la
prediction du token ``t`` ne voit jamais le token ``t``, seulement son prefixe.
Les cibles sont les memes dans les trois conditions, donc les pertes sont
comparables terme a terme.

Puis le diagnostic, avec des sondes **identiques** partout (meme softmax, meme
budget, meme regularisation) :

1. ``noeud1`` — la membrane du noeud 1, la seule que la tete de lecture lit
   (16 dims) ;
2. ``milieu`` — les 6 noeuds du milieu (96 dims) ;
3. ``tous``   — les 8 membranes (128 dims) ;
4. ``oracle`` — les embeddings bruts du prefixe, 8 positions x 16 dims (128
   dims). C'est le **temoin de largeur** : a largeur egale et tache egale, si
   lui atteint le bigramme et pas les membranes, la largeur n'est pas le
   probleme, c'est le contenu de l'etat.

Chaque sonde recoit en plus l'indice de position (7 colonnes) : sans lui, la
meme cible tombe dans un bloc de caracteristiques different selon ``t`` et
aucune sonde lineaire ne peut la suivre — c'etait un defaut de mesure de la
premiere version, corrige ici et identique pour les quatre sondes.

Enfin le routage : energie par saut, portes ouvertes/fermees, et identifiabilite
du dernier token quand l'historique s'allonge (prefixe aleatoire, evaluation sur
des prefixes jamais vus). Si l'etat cesse d'identifier le dernier mot, c'est
l'agregation qui la dilue ; s'il l'identifie encore, c'est la lecture.

Ecrit ``data/lab/s3bis_context.json``.
"""

from __future__ import annotations

import random

import numpy as np
import torch
import torch.nn.functional as F

from lab_common import LAB, ToyCorpus, read_json, write_json
from micro_mycelia import load_mycelia

SEQ = 8
VAL_WINDOWS = 64
TRAIN_WINDOWS = 400
CONDITIONS = ("normal", "melange", "vide")
SEED = 20261120


def condition_rows(batch, condition: str, rng: random.Random):
    """Lignes ``[prefixe(t), pad...]`` pour t = 1..7, et leurs cibles.

    Rend ``(entrees [B*7, 8], cibles [B*7], positions [B*7])``. La position de
    lecture est ``t`` : le modele y predit le token ``t`` avant de le lire, donc
    il n'a vu que le prefixe.
    """
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


@torch.no_grad()
def per_position_loss(brain, batch) -> list[float]:
    """Perte par position sur le prefixe reel (position 0 = sans contexte)."""
    total = torch.zeros(SEQ)
    count = 0
    for x, _y in batch:
        logits = brain(x, grow=False, trace=False)[0]
        for t in range(SEQ):
            total[t] += float(F.cross_entropy(logits[:, t], x[:, t]))
        count += 1
    return [round(float(v) / max(1, count), 4) for v in total]


def _position_onehot(t: int) -> np.ndarray:
    row = np.zeros(SEQ - 1, dtype=np.float64)
    row[t - 1] = 1.0
    return row


@torch.no_grad()
def collect_states(brain, batch):
    """Etats de tous les noeuds avant chaque token (t = 1..7) + position."""
    states: list[np.ndarray] = []
    next_ids: list[int] = []
    last_ids: list[int] = []
    for x, _y in batch:
        state = None
        tokens = [int(v) for v in x[0].tolist()]
        for t in range(SEQ):
            if t >= 1:
                row = state[:, 0, :].clone().numpy().reshape(-1)
                states.append(np.concatenate([row, _position_onehot(t)]))
                next_ids.append(tokens[t])
                last_ids.append(tokens[t - 1])
            _, state, _ = brain(
                torch.tensor([[tokens[t]]]), state=state, grow=False, trace=False,
                return_state=True,
            )
    return (
        np.asarray(states, dtype=np.float64),
        np.asarray(next_ids, dtype=np.int64),
        np.asarray(last_ids, dtype=np.int64),
    )


@torch.no_grad()
def collect_embeddings(brain, batch):
    """Temoin de largeur : les embeddings bruts du prefixe, 8 x 16 dims."""
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
    """Sonde lineaire entrainee (softmax), calibration correcte de la perte.

    Pourquoi pas une regression ridge : le ridge sur cibles one-hot est mal
    calibre et surestime la perte (mesure : 24,7 % d'exactitude pour 5,64 de
    perte). Une softmax entrainee donne une perte comparable a celle du cerveau.
    """
    mu = Xtr.mean(0)
    sd = Xtr.std(0) + 1e-8
    A = torch.tensor((Xtr - mu) / sd, dtype=torch.float32)
    B = torch.tensor((Xva - mu) / sd, dtype=torch.float32)
    W = torch.zeros(A.shape[1], vocab, requires_grad=True)
    b = torch.zeros(vocab, requires_grad=True)
    opt = torch.optim.Adam([W, b], lr=lr, weight_decay=weight_decay)
    ta = torch.tensor(ytr, dtype=torch.long)
    tb = torch.tensor(yva, dtype=torch.long)
    # Le script tourne en inference (grad desactive globalement) : la sonde est
    # le seul endroit qui apprend, elle reactive le grad pour elle seule.
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


def main() -> None:
    torch.set_grad_enabled(False)
    corpus = ToyCorpus(seq_len=SEQ)
    vocab = corpus.vocab_size
    brain = load_mycelia(str(LAB / "s2_mycelia_s1.pt"))
    val = corpus.fixed(VAL_WINDOWS, "val")
    floors = read_json(LAB / "s3bis_floors.json")

    # ---- 1. les trois conditions ----------------------------------------
    results = {}
    for name, condition in enumerate(CONDITIONS):
        rng = random.Random(SEED + name)
        results[condition] = condition_loss(brain, val, condition, rng)
    normal, shuffle, empty = (results[c] for c in CONDITIONS)

    def kl(first, second) -> float:
        p = torch.softmax(first["logits"], -1)
        q = torch.log_softmax(second["logits"], -1)
        return float((p * (torch.log(p.clamp_min(1e-12)) - q)).sum(-1).mean())

    kl_shuffle = kl(normal, shuffle)
    kl_empty = kl(normal, empty)

    # ---- 2. perte par position ------------------------------------------
    positions = per_position_loss(brain, val)

    # ---- 3. sondes : ou vit l'information ? ------------------------------
    Xtr, ntr, _ltr = collect_states(brain, corpus.fixed(TRAIN_WINDOWS, "train"))
    Xva, nva, _lva = collect_states(brain, val)
    Etr = collect_embeddings(brain, corpus.fixed(TRAIN_WINDOWS, "train"))
    Eva = collect_embeddings(brain, val)

    def subset(X, start: int, stop: int):
        """Colonnes d'un groupe de noeuds + les 7 colonnes de position."""
        return np.hstack([X[:, start:stop], X[:, -7:]])

    probe_node1 = softmax_probe(subset(Xtr, 0, 16), ntr, subset(Xva, 0, 16), nva, vocab)
    probe_mids = softmax_probe(subset(Xtr, 16, 112), ntr, subset(Xva, 16, 112), nva, vocab)
    probe_all = softmax_probe(subset(Xtr, 0, 128), ntr, subset(Xva, 0, 128), nva, vocab)
    probe_oracle = softmax_probe(Etr, ntr, Eva, nva, vocab)

    # ---- 4. routage : energie, portes, identifiabilite -------------------
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
        """Le dernier token est-il encore identifiable quand l'historique grandit ?

        Trois prefixes **aleatoires** par mot (le mot vise est toujours le
        dernier), puis apprentissage sur deux prefixes et test sur le troisieme :
        toutes les classes sont vues a l'entrainement, les echantillons de test
        sont neufs. Une split par *mot* (premiere version) rendait 0 % : une
        sonde lineaire ne peut pas deviner une classe jamais vue.
        """
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
        train = np.arange(len(y)) % 3 != 2
        return softmax_probe(
            X[train], y[train], X[~train], y[~train], vocab, steps=300
        )["exactitude"]

    # 2 tokens et non 1 : a un seul token les trois prefixes sont identiques et
    # la sonde memorise (mesure : 100 % trivial, sans information). Des deux
    # tokens, les prefixes varient et le test redevient honnete.
    rng = random.Random(SEED + 7)
    decode_2 = decodability(2, rng)
    decode_7 = decodability(7, rng)

    # ---- verdict ---------------------------------------------------------
    spread = max(
        abs(normal["perte"] - shuffle["perte"]),
        abs(normal["perte"] - empty["perte"]),
    )
    context_used = spread >= 0.1 and kl_shuffle >= 0.02
    # Ce que la colonie perd en n'agregeant que des messages (etat vs bigramme),
    # et ce que la tete perd en ne lisant qu'un noeud sur huit.
    gain_etat = round(float(normal["perte"]) - float(probe_all["perte"]), 4)
    perte_lecture = round(float(probe_node1["perte"]) - float(probe_all["perte"]), 4)
    if context_used:
        suspect = "aucun"
    elif gain_etat < 0.5:
        # Meme a 128 dims, l'etat ne retrouve pas l'ordre : c'est l'agregation.
        suspect = "agregation"
    elif perte_lecture > 0.3:
        suspect = "lecture"
    else:
        suspect = "routage"

    payload = {
        "modele": "data/lab/s2_mycelia_s1.pt",
        "fenetres_validation": VAL_WINDOWS,
        "conditions": {
            name: {
                "perte": results[name]["perte"],
                "exactitude": results[name]["exactitude"],
                "entropie_predite": results[name]["entropie_predite"],
            }
            for name in CONDITIONS
        },
        "ecart_max_conditions": round(spread, 4),
        "kl_normal_vs_melange": round(kl_shuffle, 5),
        "kl_normal_vs_vide": round(kl_empty, 5),
        "perte_par_position": positions,
        "sondes": {
            "noeud1_16d": probe_node1,
            "noeuds_milieu_96d": probe_mids,
            "tous_noeuds_128d": probe_all,
            "embeddings_prefixe_128d": probe_oracle,
        },
        "routage": {
            "energie_par_saut": energy,
            "portes": gates,
            "noeuds_muets": silent,
            "decodage_dernier_token_2": decode_2,
            "decodage_dernier_token_7": decode_7,
            "aretes": int(brain.ne),
        },
        "planchers": {
            "unigramme": floors["unigramme"]["entropie_lissee"],
            "bigramme": floors["bigramme"]["entropie_lissee"],
            "trigramme": floors["trigramme"]["entropie_lissee"],
        },
        "contexte_utilise": bool(context_used),
        "gain_etat_vs_modele": gain_etat,
        "perte_lecture_16d": perte_lecture,
        "suspect": suspect,
    }
    write_json(LAB / "s3bis_context.json", payload)

    print("condition     perte   exact.  entropie predite")
    for name in CONDITIONS:
        row = results[name]
        print(f"{name:9s}  {row['perte']:.4f}  {row['exactitude']:.4f}  {row['entropie_predite']:.4f}")
    print(f"ecart max entre conditions : {spread:.4f}")
    print(f"KL(normal || melange)      : {kl_shuffle:.5f}")
    print(f"KL(normal || vide)         : {kl_empty:.5f}")
    print(f"perte par position         : {positions}")
    for label, probe in (
        ("noeud 1 (16 dims)", probe_node1),
        ("milieu (96 dims)", probe_mids),
        ("tous noeuds (128 d)", probe_all),
        ("embeddings prefixe", probe_oracle),
    ):
        print(f"sonde {label:20s}: {probe['perte']:.4f} (exact. {probe['exactitude']:.3f}, {probe['dims']} dims)")
    print(f"decodage dernier token 2/7 : {decode_2:.3f} / {decode_7:.3f}")
    print(f"gain de l'etat sur le modele : {gain_etat:+.4f} ; cout de la lecture 16d : {perte_lecture:+.4f}")
    print(f"energie par saut           : {energy}")
    print(f"portes                     : {gates}  muets {silent}")
    print(f"contexte utilise           : {context_used} — suspect : {suspect}")
    print("ecrit : data/lab/s3bis_context.json")


if __name__ == "__main__":
    main()
