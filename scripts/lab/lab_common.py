"""Briques partagees du labo S : chemins, vocabulaire, k-means, scan, alignement.

Le labo est un banc d'essai **separe** : il lit ``data/genome.pt`` sans jamais
l'ecrire, il range tout sons ses propres fichiers dans ``data/lab/``, et il
n'importe rien d'exterieur a torch/numpy. Le k-means et l'appariement sont
codes ici a la main : aucune bibliotheque de clustering n'est installee, et
l'accord interdit le ``pip install``.
"""

from __future__ import annotations

import json
import math
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

    def __init__(
        self, seq_len: int = 8, val_fraction: float = 0.1, path: Path | str | None = None
    ) -> None:
        meta = read_json(LAB / "vocab.json")
        self.vocab: list[str] = list(meta["vocab"])
        self.stoi = {word: i for i, word in enumerate(self.vocab)}
        self.seq_len = int(seq_len)
        # S3-bis D6 : meme grammaire, corpus 10x plus grand. Le vocabulaire ne
        # change pas (les deux cerveaux gardent leur embedding de 312 mots), seul
        # le flux de phrases change. Par defaut : le corpus de S1/S2, inchange.
        source = Path(path) if path else CORPUS_TOY
        lines = [
            line.strip()
            for line in source.read_text(encoding="utf-8").splitlines()
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
    """Perte moyenne (cross-entropy) du micro-MYCELIA sur des lots fixes.

    S5-MAX M1 : la cible est le token **courant** (``x``), pas le suivant.
    ``Brain.forward`` lit le token avant de l'injecter, donc a la position ``t``
    il predit ``x[t]`` ; c'est aussi la cible de son entrainement
    (``corpus.targets``) et la tache du dense. L'ancienne version comparait au
    couple ``y`` (token suivant), soit un pas de plus que ce que le cerveau
    apprend : sur un modele au plancher les deux se valent (5,1484 contre
    5,1509), sur un modele qui predit l'ecart devient enorme (4,06 contre 6,64
    au meme instantane). La fonction d'origine est gardee, intacte, sous
    ``mycelia_loss_legacy`` pour la tracabilite des chiffres publies.
    """
    total = 0.0
    count = 0
    for x, _y in batches:
        logits = brain(x, grow=False, trace=False)[0]
        total += float(
            torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), x.reshape(-1)
            )
        )
        count += 1
    return total / max(1, count)


@torch.no_grad()
def mycelia_loss_legacy(brain, batches) -> float:
    """La metrique publiee de S2/S3-BIS, **inchangee** : cible ``y`` (suivant).

    Conservee telle quelle (meme corps, meme resultat) pour que chaque chiffre
    deja publie reste reproductible. Ce n'est pas la mesure honnete : elle
    demande au cerveau un token de plus que ce qu'il apprend.
    """
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


# ---------------------------------------------------------------------------
# S2 : correlation des activations brutes et appariement optimal
# ---------------------------------------------------------------------------
def pearson_columns(A, B) -> np.ndarray:
    """Correlation de Pearson entre chaque colonne de ``A`` et de ``B``.

    Pourquoi la correlation et pas le cosinus de S1 : le cosinus compare deux
    profils *positifs* (une activation ne descend jamais sous zero) et recompense
    donc les unites qui s'allument souvent ; la correlation centre chaque profil
    sur sa moyenne et demande si les deux montent *et descendent ensemble*. Un
    concept de S2 est une famille dont le profil suit les memes 2000 stimuli, pas
    une famille qui a la meme intensite moyenne.
    """
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    if A.ndim == 1:
        A = A[:, None]
    if B.ndim == 1:
        B = B[:, None]
    Ac = A - A.mean(axis=0, keepdims=True)
    Bc = B - B.mean(axis=0, keepdims=True)
    An = np.clip(np.linalg.norm(Ac, axis=0, keepdims=True), 1e-12, None)
    Bn = np.clip(np.linalg.norm(Bc, axis=0, keepdims=True), 1e-12, None)
    return np.clip((Ac / An).T @ (Bc / Bn), -1.0, 1.0)


def hungarian(cost) -> list[tuple[int, int]]:
    """Appariement optimal d'une matrice de cout, minimise le cout total.

    Algorithme hongrois (Kuhn-Munkres) dans sa version a potentiels : a chaque
    ligne on cherche un chemin augmentant par plus courtes distances, et les
    potentiels ``u``/``v`` gardent les couts reduits positifs. Complexite
    O(n^2 m), ecrit a la main pour ne rien installer.

    Rectangulaire : il faut ``n <= m`` (moins de lignes que de colonnes).
    Rend la liste des couples ``(ligne, colonne)`` retenus. Pour MAXIMISER un
    score, passer ``-score``.
    """
    cost = np.asarray(cost, dtype=np.float64)
    n, m = cost.shape
    if n == 0 or m == 0:
        return []
    if n > m:
        raise ValueError("hungarian attend n <= m : transposer d'abord")
    big = float(cost.max() - cost.min() + 1.0) * (n + 1) + 1.0
    u = np.zeros(n + 1)
    v = np.zeros(m + 1)
    p = np.zeros(m + 1, dtype=np.int64)
    way = np.zeros(m + 1, dtype=np.int64)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = np.full(m + 1, big)
        used = np.zeros(m + 1, dtype=bool)
        while True:
            used[j0] = True
            i0 = int(p[j0])
            delta = big
            j1 = -1
            for j in range(1, m + 1):
                if used[j]:
                    continue
                cur = cost[i0 - 1, j - 1] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j
            if j1 < 0:
                break
            for j in range(m + 1):
                if used[j]:
                    u[int(p[j])] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while j0:
            j1 = int(way[j0])
            p[j0] = p[j1]
            j0 = j1
    return [(int(p[j]) - 1, j - 1) for j in range(1, m + 1) if p[j] != 0]


def align_correlation(source, receiver, threshold: float = 0.6) -> dict:
    """Appariement des concepts par correlation, au plus proche global.

    Chaque concept source est mis en face d'une **seule** famille receveuse, et
    l'ensemble maximise la somme des correlations (hongrois). La difference avec
    l'appariement mutuel de S1 : un concept n'est pas perdu parce qu'une autre
    ligne lui volait son meilleur partenaire — c'est un choix global.

    Seuil : une paire n'est *traduite* que si sa correlation depasse
    ``threshold``. Les autres concepts n'ont pas d'equivalent : ils seront
    greffes, et la table doit le dire.
    """
    sim = pearson_columns(source, receiver)
    if sim.size == 0:
        return {
            "pairs": [],
            "traduits": [],
            "greffes": [],
            "source_unmatched": [],
            "receiver_unmatched": [],
            "similarity": sim,
            "best": np.zeros((0,), dtype=np.float32),
        }
    transposed = sim.shape[0] > sim.shape[1]
    cost = -sim.T if transposed else -sim
    raw = []
    for line, col in hungarian(cost):
        i, j = (col, line) if transposed else (line, col)
        raw.append((int(i), int(j), float(sim[i, j])))
    pairs = [(i, j, score) for i, j, score in raw if score > float(threshold)]
    translated = [i for i, _, _ in pairs]
    used_r = {j for _, j, _ in pairs}
    all_source = set(range(sim.shape[0]))
    return {
        "pairs": pairs,
        "traduits": translated,
        "greffes": sorted(all_source - set(translated)),
        "source_unmatched": sorted(all_source - set(translated)),
        "receiver_unmatched": [j for j in range(sim.shape[1]) if j not in used_r],
        "similarity": sim,
        "best": sim.max(axis=1),
        "proposition": {i: (j, score) for i, j, score in raw},
    }


# ---------------------------------------------------------------------------
# S2 : test t apparie, a la main (loi de Student, beta incomplete)
# ---------------------------------------------------------------------------
def _betacf(a: float, b: float, x: float, itmax: int = 200, eps: float = 3e-12) -> float:
    """Fraction continue de la beta incomplete (Numerical Recipes, Lentz)."""
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > 1e-30 else 1e-30)
    h = d
    for m in range(1, itmax + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > 1e-30 else 1e-30)
        c = 1.0 + aa / c
        c = c if abs(c) > 1e-30 else 1e-30
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > 1e-30 else 1e-30)
        c = 1.0 + aa / c
        c = c if abs(c) > 1e-30 else 1e-30
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def _betai(a: float, b: float, x: float) -> float:
    """Beta incomplete regularisee I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lgamma = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    front = math.exp(lgamma + a * math.log(x) + b * math.log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def student_t_p(t: float, df: int) -> float:
    """p bilateral de la loi de Student, sans scipy."""
    if df <= 0:
        return 1.0
    t = float(t)
    if not math.isfinite(t):
        return 0.0
    return float(_betai(df / 2.0, 0.5, df / (df + t * t)))


def paired_t(grafted, control, alpha: float = 0.05, min_gain: float = 0.01) -> dict:
    """Test t apparie sur les semences : le gain relatif de la greffe.

    ``d_i = (controle_i - greffe_i) / controle_i`` : un gain POSITIF veut dire
    que la greffe a une validation plus basse que le controle de la meme
    semence. Le t classique, ``t = moyenne(d) / (ecart-type(d) / sqrt(n))``,
    avec ``n - 1`` degres de liberte.

    Une traduction n'est retenue que si ``p < alpha`` ET ``gain > min_gain`` :
    sur trois semences, un gain de 0,2 % peut passer le test sans rien dire ;
    c'est du bruit de regularisation, et le rapport doit pouvoir l'ecrire.
    """
    g = np.asarray(grafted, dtype=np.float64)
    c = np.asarray(control, dtype=np.float64)
    if g.size != c.size or g.size == 0:
        raise ValueError("paired_t attend deux listes de meme longueur, non vides")
    d = (c - g) / np.clip(np.abs(c), 1e-9, None)
    n = int(d.size)
    mean = float(d.mean())
    sd = float(d.std(ddof=1)) if n > 1 else 0.0
    if sd <= 1e-12:
        t = 0.0 if abs(mean) <= 1e-12 else math.copysign(float("inf"), mean)
    else:
        t = mean / (sd / math.sqrt(n))
    p = student_t_p(t, n - 1)
    return {
        "n": n,
        "gain_moyen": round(mean, 6),
        "ecart_type": round(sd, 6),
        "t": round(float(t), 4) if math.isfinite(t) else str(t),
        "ddl": n - 1,
        "p": round(p, 6),
        "alpha": float(alpha),
        "gain_minimum": float(min_gain),
        "significatif": bool(p < float(alpha) and mean > float(min_gain)),
    }
