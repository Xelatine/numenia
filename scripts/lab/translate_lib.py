"""Briques du traducteur : scan des deux bacs, projection, affinage, tests.

La regle du traducteur universel tient en une phrase : **on aligne des
fonctions, pas des poids**. Deux cerveaux de tailles differentes n'ont aucun axe
comparable ; mais presentes aux memes 500 phrases, ils produisent des profils
d'activation comparables, et deux familles qui s'allument ensemble sont la meme
idee. C'est cette correspondance-la qui est transplantee.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from lab_common import ToyCorpus, dense_loss, mycelia_loss, profiles

SEQ = 8
MLP = 64


# ---------------------------------------------------------------------------
# Scan des deux bacs (meme protocole que MYCELI-SCAN)
# ---------------------------------------------------------------------------
@torch.no_grad()
def scan_dense(model, corpus: ToyCorpus, texts: list[str]) -> np.ndarray:
    """[n_textes, L*MLP + dim] : activations des unites cachees."""
    rows = []
    for text in texts:
        ids = corpus.encode(text)[:SEQ]
        ids = [0] * (SEQ - len(ids)) + ids
        rows.append(model.unit_activations(torch.tensor([ids], dtype=torch.long))[0])
    return torch.stack(rows).numpy().astype(np.float32)


@torch.no_grad()
def scan_mycelia(brain, corpus: ToyCorpus, texts: list[str]) -> np.ndarray:
    """[n_textes, n_noeuds] : norme maximale de la membrane, comme en S1."""
    rows = []
    for text in texts:
        ids = corpus.encode(text)[:SEQ]
        state = None
        best = None
        for token in ids or [1]:
            _, state, _ = brain(
                torch.tensor([[int(token)]]),
                state=state,
                grow=False,
                trace=False,
                return_state=True,
            )
            reading = state.norm(dim=-1).amax(dim=1)
            best = reading if best is None else torch.maximum(best, reading)
        rows.append(best.cpu().numpy())
    return np.stack(rows).astype(np.float32)


def sandbox_texts(corpus: ToyCorpus, count: int = 500) -> list[str]:
    """Les memes 500 phrases pour les deux cerveaux du bac.

    Les stimuli de Nyra sont du francais reel (tokenizer BPE de 10 000 tokens) ;
    les bacs jouets ont un vocabulaire de quelques centaines de mots. Pour que
    l'alignement porte sur les **memes entrees**, le scan du bac utilise 500
    phrases du corpus jouet — celles que les deux modeles savent lire.
    """
    texts = corpus.lines[: int(count)]
    if len(texts) < int(count):
        raise ValueError(f"corpus trop court : {len(texts)} phrases")
    return texts


# ---------------------------------------------------------------------------
# Directions (ce qu'une unite ecoute, ce qu'un noeud adresse)
# ---------------------------------------------------------------------------
def dense_directions(model) -> np.ndarray:
    """[L*MLP, dim] : la ligne d'entree de chaque neurone cache.

    Un neurone MLP ecoute une direction de l'espace residuel : c'est sa ligne
    dans ``fc1``. C'est l'equivalent exact du vecteur d'adresse d'un noeud.
    """
    rows = []
    for block in model.blocks:
        rows.append(block.fc1.weight.detach().cpu().numpy())
    return np.concatenate(rows, axis=0).astype(np.float32)


def mycelia_directions(brain) -> np.ndarray:
    """[n_noeuds, dim] : le biais de cle, c'est-a-dire l'adresse du noeud.

    Pourquoi ``k_b`` et pas ``mean_out`` : ``mean_out`` n'est rempli que pendant
    un forward de croissance, donc reste nul pour un cerveau evalue sans
    croissance. ``k_b`` est ecrit par l'optimiseur pour tout noeud vivant, et
    c'est la seule quantite qui decide *qui entend qui*.
    """
    return brain.bank.k_b.detach().cpu().numpy().astype(np.float32)


def concept_direction(directions: np.ndarray, members) -> np.ndarray:
    """Direction moyenne d'une famille (centre de gravite de ce qu'elle ecoute).

    Une famille vide rend un vecteur nul, jamais un ``nan`` : en S2 le k-means
    peut demander 8 familles a un pool qui n'a que 7 unites actives, et la
    moyenne d'un tableau vide empoisonnait toute la projection (greffe a val
    ``nan`` mesuree au premier essai).
    """
    index = np.asarray(members, dtype=int)
    if index.size == 0:
        return np.zeros(directions.shape[1], dtype=np.float32)
    return directions[index].mean(axis=0)


def projection(source: np.ndarray, target: np.ndarray, lam: float = 1e-2) -> np.ndarray:
    """Projection lineaire minimale source -> receveur, au sens des moindres carres.

    Ridge ferme : ``W = (X'X + lam I)^-1 X'Y``. Aucun entrainement, aucune
    dependance : c'est la transformation lineaire la plus simple qui envoie les
    concepts apparies sur les familles receveuses. Les concepts non apparies
    sont ensuite projetes par ce meme W — c'est le transplant.
    """
    X = np.asarray(source, dtype=np.float64)
    Y = np.asarray(target, dtype=np.float64)
    if X.shape[0] < 2:
        return np.eye(X.shape[1], Y.shape[1], dtype=np.float32)
    gram = X.T @ X + float(lam) * np.eye(X.shape[1])
    W = np.linalg.solve(gram, X.T @ Y)
    return W.astype(np.float32)


# ---------------------------------------------------------------------------
# Affinage identique pour les deux bras (meme lots, meme budget)
# ---------------------------------------------------------------------------
def make_batches(corpus: ToyCorpus, count: int, batch: int, seed: int):
    """Les lots sont tires **une fois** : les deux bras lisent la meme chose."""
    generator = torch.Generator().manual_seed(int(seed))
    return [corpus.batch(batch, generator=generator) for _ in range(int(count))]


def fine_tune_dense(model, batches, lr: float = 1e-3, weight_decay: float = 0.01) -> list[float]:
    """100 pas (ou moins) sans croissance : seule la greffe differe entre les bras."""
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    losses = []
    model.train()
    for x, y in batches:
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimiser.step()
        losses.append(float(loss.detach()))
    return losses


def fine_tune_mycelia(brain, batches, lr: float = 1e-3, weight_decay: float = 0.05) -> list[float]:
    """Meme chose cote MYCELIA. ``grow=False`` : le metabolisme est en pause.

    Pourquoi geler la croissance : le verdict compare deux bras au meme budget
    de pas. Si la naissance de synapses continuait, un bras pourrait gagner par
    du cablage aleatoire et non par la greffe ; la greffe est le seul ecart.
    """
    optimiser = torch.optim.AdamW(brain.parameters(), lr=lr, weight_decay=weight_decay)
    losses = []
    brain.train()
    for x, _ in batches:
        logits = brain(x, grow=False, trace=False)[0]
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), x.reshape(-1))
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        brain.freeze_newborn_gradients()
        torch.nn.utils.clip_grad_norm_(brain.parameters(), 1.0)
        optimiser.step()
        losses.append(float(loss.detach()))
    return losses


# ---------------------------------------------------------------------------
# Test de langue : 20 phrases hors corpus
# ---------------------------------------------------------------------------
def heldout_sentences(corpus: ToyCorpus, count: int = 40) -> list[str]:
    """Phrases nouvelles, absentes du corpus et du split de validation.

    S3 en demandait 20 ; S2 en demande 40 (et compte le top-3). Les phrases sont
    fabriquees a partir des mots du corpus jouet : aucun ``<unk>`` ne vient
    fausser la mesure.
    """
    from make_lab_data import ADJ, DET, NOUN, PREP, PRON, QW, build_sentences

    known = set(corpus.lines)
    rng_out: list[str] = []
    seed = 991
    while len(rng_out) < count:
        seed += 1
        for text in build_sentences(60, seed=seed):
            if text not in known and text not in rng_out:
                rng_out.append(text)
            if len(rng_out) >= count:
                break
        if seed > 1100:
            raise RuntimeError(f"impossible de trouver {count} phrases hors corpus")
    return rng_out[:count]


@torch.no_grad()
def test_dense(model, corpus: ToyCorpus, sentences: list[str]) -> dict:
    """Perte et exactitude du mot suivant, phrase par phrase."""
    model.eval()
    rows = []
    total = 0.0
    hits = 0
    hits3 = 0
    for text in sentences:
        ids = corpus.encode(text)
        if len(ids) < 2:
            continue
        ids = ids[: SEQ + 1]
        x = torch.tensor([ids[:-1]])
        target = torch.tensor([ids[1:]])
        logits = model(x)
        loss = float(F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target.reshape(-1)))
        guess = logits[0, -1].argmax().item()
        wanted = int(target[0, -1])
        top3 = torch.topk(logits[0, -1], 3).indices.tolist()
        rows.append(
            {
                "phrase": text,
                "perte": round(loss, 4),
                "predit": corpus.vocab[guess],
                "attendu": corpus.vocab[wanted],
                "juste": bool(guess == wanted),
                "top3": [corpus.vocab[int(i)] for i in top3],
                "juste_top3": bool(wanted in top3),
            }
        )
        total += loss
        hits += int(guess == wanted)
        hits3 += int(wanted in top3)
    return {
        "perte": round(total / max(1, len(rows)), 4),
        "exactitude": round(hits / max(1, len(rows)), 4),
        "exactitude_top3": round(hits3 / max(1, len(rows)), 4),
        "exemples": rows,
    }


@torch.no_grad()
def test_mycelia(brain, corpus: ToyCorpus, sentences: list[str]) -> dict:
    """Meme test, avec l'ecriture MYCELIA (il predit le token lu)."""
    brain.eval()
    rows = []
    total = 0.0
    hits = 0
    hits3 = 0
    for text in sentences:
        ids = corpus.encode(text)[: SEQ + 1]
        if len(ids) < 2:
            continue
        state = None
        logits = None
        for token in ids[:-1]:
            logits, state, _ = brain(
                torch.tensor([[int(token)]]),
                state=state,
                grow=False,
                trace=False,
                return_state=True,
            )
        logits = logits[0, 0]
        wanted = int(ids[-1])
        loss = float(F.cross_entropy(logits.unsqueeze(0), torch.tensor(wanted).unsqueeze(0)))
        guess = int(logits.argmax())
        top3 = torch.topk(logits, 3).indices.tolist()
        rows.append(
            {
                "phrase": text,
                "perte": round(loss, 4),
                "predit": corpus.vocab[guess],
                "attendu": corpus.vocab[wanted],
                "juste": bool(guess == wanted),
                "top3": [corpus.vocab[int(i)] for i in top3],
                "juste_top3": bool(wanted in top3),
            }
        )
        total += loss
        hits += int(guess == wanted)
        hits3 += int(wanted in top3)
    return {
        "perte": round(total / max(1, len(rows)), 4),
        "exactitude": round(hits / max(1, len(rows)), 4),
        "exactitude_top3": round(hits3 / max(1, len(rows)), 4),
        "exemples": rows,
    }


# ---------------------------------------------------------------------------
# Reprise du vivier : copies independantes pour les deux bras
# ---------------------------------------------------------------------------
def mycelia_clone(brain):
    """Un cerveau independant, identique au bit pres (les deux bras partent d'ici)."""
    from mycelia.brain import Brain

    config = dict(brain.config())
    config["n_nodes"] = int(brain.n_nodes)
    clone = Brain(**config)
    clone.load_state_dict(
        {key: value.detach().clone() for key, value in brain.state_dict().items()}
    )
    clone.ne = int(brain.ne)
    clone.clock = int(brain.clock)
    clone.births = int(brain.births)
    clone.node_births = int(brain.node_births)
    clone.edge_deaths = int(brain.edge_deaths)
    clone.node_deaths = int(brain.node_deaths)
    clone.cemetery = {k: {n: t.clone() for n, t in v.items()} for k, v in brain.cemetery.items()}
    clone._rebuild_edge_keys()
    return clone


def dense_clone(model):
    """Copie independante du temoin dense (memes poids, autre objet).

    La largeur cachee est lue sur le modele, pas supposee : les jumeaux de S2 ont
    un MLP de 166, le temoin de S1 en a 64, et une copie qui se trompe de largeur
    ne charge pas ses poids (lecon apprise a la premiere execution de S2).
    """
    from micro_dense import DenseTransformer

    first = model.blocks[0]
    clone = DenseTransformer(
        model.vocab,
        dim=model.dim,
        layers=len(model.blocks),
        heads=int(first.heads),
        mlp=int(first.fc1.out_features),
        seq=int(model.seq),
    )
    clone.load_state_dict({k: v.detach().clone() for k, v in model.state_dict().items()})
    return clone


def summary(before: dict, grafted: dict, control: dict) -> dict:
    """Le verdict chiffre, calcule une fois pour les deux sens.

    Deux lectures, parce qu'elles ne disent pas la meme chose : la perte
    d'entrainement mesure la **vitesse d'apprentissage** (la greffe apporte-t-elle
    de la capacite utile ?), la validation mesure la **generalisation** (cette
    capacite ne fait-elle pas que memoriser ?). Le verdict porte sur les deux.
    """
    x, y, c = float(before["val"]), float(grafted["val"]), float(control["val"])
    xt = float(before.get("train", x))
    yt = float(grafted.get("train", grafted["curve"][-1]["train"]))
    ct = float(control.get("train", control["curve"][-1]["train"]))
    return {
        "avant": round(x, 4),
        "apres_greffe": round(y, 4),
        "controle": round(c, 4),
        "delta_brut": round(y - x, 4),
        "delta_vs_controle": round(y - c, 4),
        "gain_vs_avant": round((x - y) / max(1e-9, x), 4),
        "gain_vs_controle": round((c - y) / max(1e-9, c), 4),
        "train_avant": round(xt, 4),
        "train_greffe": round(yt, 4),
        "train_controle": round(ct, 4),
        "train_gain_vs_controle": round((ct - yt) / max(1e-9, ct), 4),
        "greffe_gagne": bool(y < x),
        "greffe_gagne_sur_controle": bool(y < c),
        "greffe_gagne_entrainement": bool(yt < ct),
    }


@torch.no_grad()
def batches_loss_mycelia(brain, batches) -> float:
    total = 0.0
    for x, _ in batches:
        logits = brain(x, grow=False, trace=False)[0]
        total += float(F.cross_entropy(logits.reshape(-1, logits.shape[-1]), x.reshape(-1)))
    return total / max(1, len(batches))


@torch.no_grad()
def batches_loss_dense(model, batches) -> float:
    total = 0.0
    for x, y in batches:
        logits = model(x)
        total += float(F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1)))
    return total / max(1, len(batches))


# ---------------------------------------------------------------------------
# Greffe : un noeud-porte qui porte le concept
# ---------------------------------------------------------------------------
@torch.no_grad()
def implant_node(brain, vector, partners=()) -> int:
    """Fait naitre un noeud dont l'**adresse** est le concept projete.

    Le noeud ne calcule rien : sa porte est ouverte (sigmoide 2,0 = 0,88), sa
    transformation est l'identite (b1 et b2 a zero, LayerNorm neutre), et son
    adresse (``k_b``) *est* le concept. Il relaie donc ce qu'il recoit vers
    quiconque adresse ce concept, et se fait adresser par ceux qui le portent.
    C'est le transplant minimal : on n'importe pas des poids qui n'ont pas de
    sens ici, on importe une **fonction**.
    """
    node = int(brain.n_nodes)
    if node >= int(brain.max_population):
        # Le vivier du bac etait au plafond (96) : une greffe n'a pas de place.
        # On l'agrandit ici et on reconstruit les cles d'aretes, dont la
        # stride est le plafond du pool.
        brain.max_population = node + 32
        brain._init_kwargs["max_population"] = brain.max_population
        brain._rebuild_edge_keys()
    brain.grow_population(node + 1, step=brain.clock)
    brain.node_alive[node] = True
    brain.node_born_step[node] = brain.clock
    # Pas de gel : le nouveau-ne doit apprendre dans les 100 pas du verdict.
    brain.node_frozen_until[node] = brain.clock
    dim = int(brain.dim)
    vec = torch.as_tensor(np.asarray(vector, dtype=np.float32))
    bank = brain.bank
    bank.gate_w[node].fill_(0.02)
    bank.gate_b[node].fill_(2.0)
    for name in ("b1_w", "b1_b", "b2_w", "b2_b"):
        getattr(bank, name)[node].zero_()
    bank.ln_w[node].fill_(1.0)
    bank.ln_b[node].zero_()
    bank.q_w[node].copy_(torch.eye(dim))
    bank.q_b[node].copy_(vec)
    bank.k_w[node].zero_()
    bank.k_b[node].copy_(vec)
    brain.mean_out[node].copy_(vec)
    brain.add_edge(0, node, count=False)
    brain.add_edge(node, 1, count=False)
    for partner in partners:
        partner = int(partner)
        if 2 <= partner < node and partner != node:
            brain.add_edge(node, partner, count=False)
            brain.add_edge(partner, node, count=False)
    brain.births += 1
    brain.node_births += 1
    return node


# ---------------------------------------------------------------------------
# Bras d'affinage : 100 pas, meme lots, courbe de validation relevee
# ---------------------------------------------------------------------------
def arm_mycelia(brain, batches, val_batches, lr: float = 1e-3,
                weight_decay: float = 0.05, every: int = 10) -> dict:
    """Un bras complet : perte d'entrainement et validation tous les ``every`` pas."""
    optimiser = torch.optim.AdamW(brain.parameters(), lr=lr, weight_decay=weight_decay)
    curve = []
    window: list[float] = []
    brain.train()
    for step, (x, _) in enumerate(batches, 1):
        logits = brain(x, grow=False, trace=False)[0]
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), x.reshape(-1))
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        brain.freeze_newborn_gradients()
        torch.nn.utils.clip_grad_norm_(brain.parameters(), 1.0)
        optimiser.step()
        window.append(float(loss.detach()))
        if step % int(every) == 0:
            brain.eval()
            value = float(mycelia_loss(brain, val_batches))
            brain.train()
            curve.append(
                {
                    "step": step,
                    "train": round(float(np.mean(window)), 4),
                    "val": round(value, 4),
                }
            )
            window = []
    brain.eval()
    final = float(mycelia_loss(brain, val_batches))
    return {
        "curve": curve,
        "val": round(final, 4),
        "val_debut": curve[0]["val"] if curve else None,
        "pool": int(brain.n_nodes),
        "aretes": int(brain.ne),
        "brain": brain,
    }


def arm_dense(model, batches, val_batches, lr: float = 1e-3,
              weight_decay: float = 0.01, every: int = 10) -> dict:
    """Le meme bras, cote dense (memes lots, meme budget)."""
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    curve = []
    window: list[float] = []
    model.train()
    for step, (x, y) in enumerate(batches, 1):
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimiser.step()
        window.append(float(loss.detach()))
        if step % int(every) == 0:
            model.eval()
            value = float(dense_loss(model, val_batches))
            model.train()
            curve.append(
                {
                    "step": step,
                    "train": round(float(np.mean(window)), 4),
                    "val": round(value, 4),
                }
            )
            window = []
    model.eval()
    final = float(dense_loss(model, val_batches))
    return {
        "curve": curve,
        "val": round(final, 4),
        "val_debut": curve[0]["val"] if curve else None,
        "mlp": [int(block.fc1.out_features) for block in model.blocks],
        "model": model,
    }
