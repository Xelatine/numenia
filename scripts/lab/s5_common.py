"""S5-MAX — les briques communes : pool 24, jumeau equitable, colonnes, plafond.

Trois choses seulement, partagees par les trois scripts de la mission.

**Le pool 24 vivant.** ``POOL24`` garde la population figee (24 noeuds) pour que
le jumeau equitable reste defini pendant tout l'entrainement, et laisse le
*CABLAGE* vivre : naissances d'aretes payees par le trafic et coupe des
synapses faminees (``edge_deaths``). Le plafond de 220 aretes vaut 39,9 % des
552 paires ordonnees du pool 24 : la mission exige moins de 50 % de densite, et
un plafond bas est ce qui laisse des slots libres a la naissance suivante.
``POOL32`` sert au re-test demande si le graphe sature (396 aretes = 39,9 % des
992 paires).

**Le jumeau equitable.** On compte les parametres *actifs par token* du cerveau
repare (sans routage, tout le pool diffuse a chaque token : le total est
l'actif), puis on choisit la taille MLP du dense telle que
``12792 + 66 * mlp >= actifs``. Le dense est donc **au moins aussi gros** que
NUMENIA : la comparaison reste defavorable a NUMENIA, comme en S4.

**Les deux colonnes de mesure.** ``colonne_nouvelle`` predit le token courant
(``x[t]``) a partir du prefixe, c'est la tache reelle du cerveau ;
``colonne_ancienne`` predit le token suivant (``y[t]``), la metrique publiee de
S2/S3-BIS. Meme modeles, memes fenetres, seule la cible change.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from lab_common import LAB, ToyCorpus, read_json
from micro_dense import DenseTransformer, n_parameters
from repaired_mycelia import SEQ

# ---------------------------------------------------------------------------
# Configurations
# ---------------------------------------------------------------------------
POOL24 = {
    "n_nodes": 24,
    "max_edges": 220,
    "max_population": 24,
    "max_node_births": 0,
}
POOL32 = {
    "n_nodes": 32,
    "max_edges": 396,
    "max_population": 32,
    "max_node_births": 0,
}
STEPS = 600
SEEDS = (1, 2, 3)
L_REF = 5.082888  # unigramme lisse du corpus 1000 (data/lab/s3bis_floors.json)
DIM, LAYERS, HEADS = 16, 2, 2
MLP_TWIN = 768
FAIM_DEFAUT = 0.1   # le reglage du cerveau (KW["prune_ratio"])
FAIM_VIVANTE = 0.5  # le seul reglage qui fasse mourir des synapses (mesure S5)

CORPORA = {
    "1000": {"path": None, "val": 58},
    "10k": {"path": "corpus_toy10k.txt", "val": 100},
}


def load_corpus(name: str) -> ToyCorpus:
    entry = CORPORA[name]
    path = None if entry["path"] is None else LAB / entry["path"]
    return ToyCorpus(seq_len=SEQ, path=path)


# ---------------------------------------------------------------------------
# Le dense, a la taille demandee
# ---------------------------------------------------------------------------
def build_dense(vocab: int, seed: int, mlp: int = MLP_TWIN) -> DenseTransformer:
    torch.manual_seed(int(seed))
    return DenseTransformer(
        int(vocab), dim=DIM, layers=LAYERS, heads=HEADS, mlp=int(mlp), seq=SEQ
    )


def dense_parameters(mlp: int) -> int:
    """Compte exact du dense (verifie par ``n_parameters`` a la construction)."""
    return 12_792 + 66 * int(mlp)


def save_dense(model: DenseTransformer, path, extra: dict | None = None) -> None:
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": {
                "dim": DIM, "layers": LAYERS, "heads": HEADS,
                "mlp": int(model.blocks[0].fc1.out_features),
                "seq": SEQ,
            },
            "vocab_size": int(model.vocab),
            "extra": extra or {},
        },
        str(path),
    )


def load_dense(path) -> DenseTransformer:
    blob = torch.load(str(path), weights_only=False)
    cfg = blob["config"]
    model = DenseTransformer(
        int(blob["vocab_size"]), dim=cfg["dim"], layers=cfg["layers"],
        heads=cfg["heads"], mlp=cfg["mlp"], seq=cfg["seq"],
    )
    model.load_state_dict(blob["state_dict"])
    model.eval()
    return model


def dense_params_of(path) -> int:
    return n_parameters(load_dense(path))


# ---------------------------------------------------------------------------
# Les deux colonnes, pour les deux familles de modeles
# ---------------------------------------------------------------------------
@torch.no_grad()
def columns_mycelia(brain, batches) -> dict:
    """Colonnes d'un NUMENIA : nouvelle = x[t], ancienne = y[t].

    La position 0 est exclue de la colonne nouvelle : elle n'a aucun contexte,
    et aucun dense ne peut y predire quoi que ce soit. La colonne ancienne la
    garde (c'est ce que publiait S2/S3-BIS).
    """
    new_total = 0.0
    new_count = 0
    old_total = 0.0
    old_count = 0
    full_total = 0.0
    full_count = 0
    for x, y in batches:
        logits = brain(x, grow=False, trace=False)[0]
        for t in range(SEQ):
            old_total += float(F.cross_entropy(logits[:, t], y[:, t]))
            old_count += 1
            full_total += float(F.cross_entropy(logits[:, t], x[:, t]))
            full_count += 1
            if t >= 1:
                new_total += float(F.cross_entropy(logits[:, t], x[:, t]))
                new_count += 1
    return {
        "nouvelle": round(new_total / max(1, new_count), 4),
        "ancienne": round(old_total / max(1, old_count), 4),
        "nouvelle_0_7": round(full_total / max(1, full_count), 4),
        "contextes_nouvelle": int(new_count),
        "contextes_ancienne": int(old_count),
    }


@torch.no_grad()
def columns_dense(model, batches) -> dict:
    """Colonnes du dense : nouvelle = la prediction du token courant.

    Le dense predit ``y[t] = x[t+1]`` a la position ``t``. Pour predire ``x[t]``
    a partir du meme prefixe, c'est sa position ``t-1`` qu'il faut lire : meme
    token, meme information disponible que MYCELIA.
    """
    new_total = 0.0
    new_count = 0
    old_total = 0.0
    old_count = 0
    for x, y in batches:
        logits = model(x)
        for t in range(SEQ):
            old_total += float(F.cross_entropy(logits[:, t], y[:, t]))
            old_count += 1
            if t >= 1:
                new_total += float(F.cross_entropy(logits[:, t - 1], x[:, t]))
                new_count += 1
    return {
        "nouvelle": round(new_total / max(1, new_count), 4),
        "ancienne": round(old_total / max(1, old_count), 4),
        "contextes_nouvelle": int(new_count),
        "contextes_ancienne": int(old_count),
    }


# ---------------------------------------------------------------------------
# R, KL, top-3 : les chiffres du plafond
# ---------------------------------------------------------------------------
def relative_gain(loss: float, l_prof: float, l_ref: float = L_REF) -> float:
    """R = (L_ref - L) / (L_ref - L_prof) : la part du chemin parcourue."""
    denom = float(l_ref) - float(l_prof)
    if abs(denom) < 1e-9:
        return 0.0
    return round((float(l_ref) - float(loss)) / denom, 4)


@torch.no_grad()
def kl_direction(logits_p, logits_q, temperature: float = 1.0) -> float:
    """KL(P || Q) moyenne par contexte, en nats (T^2 si temperature)."""
    t = float(temperature)
    logp = F.log_softmax(logits_p / t, dim=-1)
    logq = F.log_softmax(logits_q / t, dim=-1)
    return float((logp.exp() * (logp - logq)).sum(-1).mean()) * (t * t)


@torch.no_grad()
def topk_sets(logits, k: int = 3) -> torch.Tensor:
    return torch.topk(logits, int(k), dim=-1).indices


def topk_overlap(logits_a, logits_b, k: int = 3) -> dict:
    """Accord top-k : part de contextes itentiques et recouvrement moyen."""
    a = topk_sets(logits_a, k)
    b = topk_sets(logits_b, k)
    same = (a == b).all(dim=-1).float().mean()
    cover = torch.zeros(a.shape[0], dtype=torch.float32)
    for row in range(a.shape[0]):
        cover[row] = len(set(a[row].tolist()) & set(b[row].tolist())) / float(k)
    return {"identiques": round(float(same), 4), "recouvrement": round(float(cover.mean()), 4)}


def mean_sd(values) -> tuple[float, float]:
    values = [float(v) for v in values]
    if not values:
        return 0.0, 0.0
    mean = sum(values) / len(values)
    sd = 0.0 if len(values) < 2 else (
        sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    ) ** 0.5
    return round(mean, 4), round(sd, 4)


__all__ = [
    "CORPORA",
    "DIM",
    "FAIM_DEFAUT",
    "FAIM_VIVANTE",
    "HEADS",
    "LAYERS",
    "L_REF",
    "MLP_TWIN",
    "POOL24",
    "POOL32",
    "SEEDS",
    "STEPS",
    "build_dense",
    "columns_dense",
    "columns_mycelia",
    "dense_parameters",
    "dense_params_of",
    "kl_direction",
    "load_corpus",
    "load_dense",
    "mean_sd",
    "relative_gain",
    "save_dense",
    "topk_overlap",
]
