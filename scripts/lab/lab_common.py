"""Briques partagees du labo S : chemins, vocabulaire, k-means, scan, alignement.

Le labo est un banc d'essai **separe** : il lit ``data/genome.pt`` sans jamais
l'ecrire, il range tout sons ses propres fichiers dans ``data/lab/``, et il
n'importe rien d'exterieur a torch/numpy. Le k-means et l'appariement sont
codes ici a la main : aucune bibliotheque de clustering n'est installee, et
l'accord interdit le ``pip install``.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
LAB = ROOT / "data" / "lab"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SEED = 20261120
STIMULI = LAB / "stimuli.txt"
STIMULI_NEW = LAB / "stimuli_new.txt"
CORPUS_TOY = LAB / "corpus_toy.txt"


def set_seed(seed: int = SEED) -> None:
    """Meme graine partout : torch, numpy, random."""
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))


def ensure_lab() -> Path:
    LAB.mkdir(parents=True, exist_ok=True)
    return LAB


def to_jsonable(value):
    """Convertit tensors/numpy en types JSON, recursivement."""
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, torch.Tensor):
        return to_jsonable(value.detach().cpu())
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, float):
        return round(float(value), 6)
    return value


def write_json(path: Path | str, payload) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(to_jsonable(payload), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return path


def read_json(path: Path | str):
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Corpus jouet commun aux deux bacs a sable
# ---------------------------------------------------------------------------
class ToyCorpus:
    """Le corpus jouet comme un flux continu, avec 10 % tenus a part.

    Les deux bacs lisent **le meme** flux et tirent leurs lots avec le meme
    generateur : sans ca, une difference de perte entre dense et mycelia
    pourrait venir d'un autre ordre de lecture, pas de l'architecture. Pour le
    dense, ``y[t]`` est le mot suivant du flux ; pour MYCELIA, la cible est la
    fenetre elle-meme, parce que son forward predit le token lu *avant* de
    l'injecter (meme tache, ecriture decalee).
    """

    def __init__(self, seq_len: int = 8, val_fraction: float = 0.1) -> None:
        meta = read_json(LAB / "vocab.json")
        self.vocab: list[str] = list(meta["vocab"])
        self.stoi = {word: i for i, word in enumerate(self.vocab)}
        self.seq_len = int(seq_len)
        lines = [
            line.strip()
            for line in CORPUS_TOY.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        stream: list[int] = []
        for line in lines:
            stream.extend(self.stoi.get(word, 1) for word in line.split())
        tokens = torch.tensor(stream, dtype=torch.long)
        cut = max(self.seq_len + 1, int(tokens.numel() * (1.0 - float(val_fraction))))
        self.train = tokens[:cut]
        self.val = tokens[cut - self.seq_len - 1 :]
        self.lines = lines

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def _windows(self, split: str) -> int:
        stream = self.train if split == "train" else self.val
        return max(0, (stream.numel() - 1) // self.seq_len)

    def batch(self, size: int, split: str = "train", generator=None):
        """(x, y) de forme [size, seq_len] ; y = x decale d'un token."""
        stream = self.train if split == "train" else self.val
        windows = self._windows(split)
        if windows == 0:
            raise ValueError(f"split {split} plus court qu'une fenetre")
        starts = torch.randint(0, windows, (int(size),), generator=generator) * self.seq_len
        idx = starts.unsqueeze(1) + torch.arange(self.seq_len).unsqueeze(0)
        x = stream[idx]
        y = stream[idx + 1]
        return x, y

    def fixed(self, count: int, split: str = "val"):
        """Lots deterministes, pour comparer deux bras sur les memes mots."""
        stream = self.train if split == "train" else self.val
        windows = self._windows(split)
        out = []
        for w in range(min(int(count), windows)):
            start = w * self.seq_len
            idx = start + torch.arange(self.seq_len)
            x = stream[idx].unsqueeze(0)
            out.append((x, stream[idx + 1].unsqueeze(0)))
        return out

    def targets(self, x: torch.Tensor) -> torch.Tensor:
        """Cible MYCELIA : la fenetre elle-meme (il lit avant d'injecter)."""
        return x

    def encode(self, text: str) -> list[int]:
        return [self.stoi.get(word, 1) for word in text.split()] or [1]

    def decode(self, ids) -> str:
        return " ".join(
            self.vocab[int(i)] if 0 <= int(i) < len(self.vocab) else "<unk>" for i in ids
        )


# ---------------------------------------------------------------------------
# K-means, ecrit a la main (k-means++ puis Lloyd)
# ---------------------------------------------------------------------------
def kmeans(X, k: int, iters: int = 80, seed: int = SEED) -> tuple[np.ndarray, np.ndarray, float]:
    """Clusterise les lignes de ``X`` en ``k`` familles.

    k-means++ pour l'initialisation (un tirage uniforme laisse deux centres dans
    la meme famille sur un jeu de 500 points), puis Lloyd jusqu'a stabilite.
    Un cluster vide reprend le point le plus eloigne : sans ca, ``k`` familles
    demandees en rendent ``k-1`` sans le dire.
    """
    X = torch.as_tensor(np.asarray(X), dtype=torch.float32)
    n, d = X.shape
    k = max(1, min(int(k), n))
    if k == 1:
        return np.zeros(n, dtype=np.int64), X.mean(0, keepdim=True).numpy(), float(
            ((X - X.mean(0)) ** 2).sum()
        )
    generator = torch.Generator().manual_seed(int(seed))
    first = int(torch.randint(0, n, (1,), generator=generator))
    centres = [X[first]]
    dist2 = ((X - X[first]) ** 2).sum(1)
    for _ in range(1, k):
        total = float(dist2.sum())
        if total <= 0:
            centres.append(X[int(torch.randint(0, n, (1,), generator=generator))])
            continue
        pick = int(torch.multinomial(dist2.clamp_min(0), 1, generator=generator))
        centres.append(X[pick])
        dist2 = torch.minimum(dist2, ((X - X[pick]) ** 2).sum(1))
    C = torch.stack(centres)
    labels = torch.full((n,), -1, dtype=torch.long)
    for _ in range(int(iters)):
        dist = torch.cdist(X, C)
        assign = dist.argmin(1)
        if bool((assign == labels).all()):
            break
        labels = assign
        for j in range(k):
            mask = labels == j
            if bool(mask.any()):
                C[j] = X[mask].mean(0)
            else:
                far = int(torch.argmax(dist.min(1).values))
                C[j] = X[far]
    inertia = float(((X - C[labels]) ** 2).sum())
    return labels.numpy().astype(np.int64), C.numpy(), inertia


# ---------------------------------------------------------------------------
# Scan d'activation
# ---------------------------------------------------------------------------
@torch.no_grad()
def scan_brain(brain, texts, tokenizer, max_tokens: int = 32) -> np.ndarray:
    """Active le cerveau token par token et rend [n_textes, n_noeuds].

    L'activation d'un noeud pour un stimulus est la **norme L2 maximale de sa
    membrane** sur les tokens du stimulus. Pourquoi le max et pas la moyenne :
    un mot isole (``chat``) occupe un seul token ; une moyenne sur huit tokens
    diluerait un pic de reconnaissance par huit et rangerait tous les mots
    isoles dans la meme famille "faible". Pourquoi la norme : la membrane est un
    vecteur de dimension ``dim``, et son amplitude est la seule lecture qui ne
    depende pas de l'orientation du signal.
    """
    rows: list[np.ndarray] = []
    for text in texts:
        ids = (tokenizer.encode(text).ids or [0])[: int(max_tokens)]
        state = None
        best = None
        for token in ids:
            _, state, _ = brain(
                torch.tensor([[int(token)]]),
                state=state,
                grow=False,
                trace=False,
                return_state=True,
            )
            reading = state.norm(dim=-1).amax(dim=1)
            best = reading if best is None else torch.maximum(best, reading)
        rows.append(best.cpu().numpy().astype(np.float32))
    return np.stack(rows)


@torch.no_grad()
def film_brain(brain, text: str, tokenizer, max_tokens: int = 24) -> dict:
    """Film d'activation : un vecteur par token, pour l'onglet du dashboard."""
    tokens = tokenizer.encode(text).ids or [0]
    tokens = tokens[: int(max_tokens)]
    state = None
    frames: list[list[float]] = []
    labels: list[str] = []
    for token in tokens:
        _, state, _ = brain(
            torch.tensor([[int(token)]]),
            state=state,
            grow=False,
            trace=False,
            return_state=True,
        )
        frames.append(state.norm(dim=-1).amax(dim=1).cpu().numpy().astype(np.float32))
        labels.append(tokenizer.decode([int(token)]))
    return {
        "texte": text,
        "tokens": labels,
        "frames": np.stack(frames) if frames else np.zeros((1, int(brain.n_nodes))),
    }


# ---------------------------------------------------------------------------
# Profils, familles, appariement fonctionnel
# ---------------------------------------------------------------------------
def profiles(matrix: np.ndarray) -> np.ndarray:
    """Normalise chaque colonne (unites/noeuds) en profil sur les stimuli.

    L'alignement du traducteur se fait sur des *profils* et non sur des poids :
    deux cerveaux de tailles differentes n'ont pas d'axes comparables, mais ils
    repondent aux memes 500 phrases. C'est la regle du traducteur universel.
    """
    X = np.asarray(matrix, dtype=np.float32).T  # [unites, stimuli]
    X = np.maximum(X, 0.0)
    norm = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.clip(norm, 1e-8, None)


def cosine_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Similarite cosinus entre deux jeux de profils normalises."""
    return np.clip(np.asarray(A, dtype=np.float32) @ np.asarray(B, dtype=np.float32).T, -1.0, 1.0)


def align(source: np.ndarray, receiver: np.ndarray, threshold: float = 0.35) -> dict:
    """Apparie des concepts source a des familles receveuses, par profil.

    Appariement **mutuel** : une paire ne compte que si chacun est le meilleur
    partenaire de l'autre, et si la similarite depasse le seuil. Le glouton
    simple avait ete essaye d'abord et appariait 6 concepts sur 6 avec un seuil
    de 0,55 alors qu'aucune ligne ne depassait 0,57 qu'une fois : plusieurs
    colonnes passaient le seuil pour le meme concept. Sous le seuil, la paire
    reste non appariee — c'est la liste des "concepts sans equivalent" que le
    transplant doit greffer.
    """
    sim = cosine_matrix(source, receiver)
    best_receiver = sim.argmax(axis=1)
    best_source = sim.argmax(axis=0)
    pairs: list[tuple[int, int, float]] = []
    for i in range(sim.shape[0]):
        j = int(best_receiver[i])
        score = float(sim[i, j])
        if int(best_source[j]) != i or score < float(threshold):
            continue
        pairs.append((i, j, score))
    used_s = {i for i, _, _ in pairs}
    used_r = {j for _, j, _ in pairs}
    return {
        "pairs": pairs,
        "source_unmatched": [i for i in range(sim.shape[0]) if i not in used_s],
        "receiver_unmatched": [j for j in range(sim.shape[1]) if j not in used_r],
        "similarity": sim,
    }


# ---------------------------------------------------------------------------
# Perte d'un cerveau sur un corpus mot-a-mot
# ---------------------------------------------------------------------------
@torch.no_grad()
def mycelia_loss(brain, batches) -> float:
    """Perte moyenne (cross-entropy) du micro-MYCELIA sur des lots fixes."""
    total = 0.0
    count = 0
    for x, y in batches:
        logits = brain(x, grow=False, trace=False)[0]
        total += float(
            torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), y.reshape(-1)
            )
        )
        count += 1
    return total / max(1, count)


@torch.no_grad()
def dense_loss(model, batches) -> float:
    """Perte moyenne du micro-transformer dense sur des lots fixes."""
    total = 0.0
    count = 0
    for x, y in batches:
        logits = model(x)
        total += float(
            torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), y.reshape(-1)
            )
        )
        count += 1
    return total / max(1, count)
