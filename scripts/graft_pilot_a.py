"""Pilote A — encodeur distillé 896 → 32, mêmes seuils, consignés AVANT.

Le pilote linéaire a échoué (Spearman 0,35-0,60 pour un seuil de 0,70). Ici la
traduction n'est plus une projection forcée : un MLP apprend à préserver les
similarités cosinus des phrases de Qwen dans 32 dimensions.

Seuils inchangés, écrits avant le calcul :
  S1 Spearman des cosinus par paires (20 phrases, 190 paires) ≥ 0,70
  S2 erreur cosinus moyenne ≤ 0,15
  S3 retrieval hit@1 ≥ 4/5 et pas plus d'un point sous Qwen

Entraînement : 1 200 phrases du corpus, paires aléatoires, cible = cosinus des
embeddings Qwen (moyenne des embeddings d'entrée, même représentation que le
pilote linéaire). Aucune phrase d'évaluation n'est utilisée pour l'entraînement.

Si VERT : les poids du MLP deviennent l'encodeur d'entrée de la greffe 1
(transplant de la FONCTION apprise). Si ROUGE : consigné, la greffe 1 reste
fermée.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mycelia.gguf_reader import read_embedding_rows, read_inventory  # noqa: E402

MODEL = ROOT / "data" / "models" / "qwen2.5-0.5b-instruct-q8_0.gguf"
OUT = ROOT / "data" / "mem_report" / "greffe_pilot_a.json"
WEIGHTS = ROOT / "data" / "graft_encoder_a.pt"
DIM = 32
SEED = 20261115
THRESHOLDS = {"spearman_min": 0.70, "cos_error_max": 0.15, "hit_at_1_min": 4}

SENTENCES = [
    "Je m'appelle Théo et je suis allergique à la pénicilline.",
    "Ma couleur préférée est le bleu cobalt.",
    "Je travaille au laboratoire de biologie du campus.",
    "Mon chat s'appelle Fermion et il est noir.",
    "Je déteste qu'on me coupe la parole.",
    "Le train pour Lyon part à huit heures du matin.",
    "La bibliothèque ferme exceptionnellement à dix-neuf heures.",
    "J'ai perdu mon portefeuille dans le métro hier soir.",
    "Mon frère habite à Montréal depuis trois ans.",
    "Le musée expose des peintures de la Renaissance.",
    "Il pleut depuis deux jours sans interruption.",
    "Le café du coin sert un excellent chocolat chaud.",
    "Je dois rendre mon rapport avant vendredi.",
    "Les enfants jouent dans le parc après l'école.",
    "Mon ordinateur portable a besoin d'une nouvelle batterie.",
    "La réunion est reportée à la semaine prochaine.",
    "J'apprends le piano depuis six mois.",
    "Le marché du samedi vend des légumes de saison.",
    "Cette ville est connue pour ses ponts anciens.",
    "Je préfère lire le soir avant de dormir.",
]

PAIRS = [
    ("Je suis allergique à la pénicilline.", "quelle est mon allergie ?", "pénicilline"),
    ("Ma couleur préférée est le bleu.", "quelle couleur j'aime ?", "bleu"),
    ("Je travaille dans un laboratoire.", "où est-ce que je travaille ?", "laboratoire"),
    ("Mon chat s'appelle Fermion.", "comment s'appelle mon chat ?", "Fermion"),
    ("Je déteste qu'on me coupe la parole.", "qu'est-ce qui m'énerve ?", "parole"),
]


def ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    result = np.empty_like(order, dtype=np.float64)
    result[order] = np.arange(len(values))
    return result


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra, rb = ranks(a), ranks(b)
    ra -= ra.mean()
    rb -= rb.mean()
    return float((ra @ rb) / (np.linalg.norm(ra) * np.linalg.norm(rb) + 1e-12))


def pairwise_cos(vectors: np.ndarray) -> np.ndarray:
    vectors = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-9)
    matrix = vectors @ vectors.T
    upper = np.triu_indices(len(vectors), k=1)
    return matrix[upper]


def corpus_sentences(limit: int = 1200) -> list[str]:
    raw = (ROOT / "data" / "corpus_fr.txt").read_text(encoding="utf-8", errors="replace")
    out = []
    banned = set(SENTENCES)
    for chunk in re.split(r"(?<=[.!?])\s+", raw):
        text = " ".join(chunk.split())
        if 25 <= len(text) <= 240 and text not in banned:
            out.append(text)
        if len(out) >= limit:
            break
    return out


class Distiller(nn.Module):
    def __init__(self, dim_in: int = 896, dim_out: int = DIM, hidden: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_in, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1)


def main() -> int:
    from llama_cpp import Llama

    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    llm = Llama(model_path=str(MODEL), n_ctx=256, n_threads=2, verbose=False)
    _metadata, tensors = read_inventory(MODEL)
    emb = {t.name: t for t in tensors}["token_embd.weight"]

    def qwen_embeddings(texts: list[str]) -> np.ndarray:
        token_ids = [llm.tokenize(t.encode("utf-8"), add_bos=False)[:128] for t in texts]
        flat = sorted({i for ids in token_ids for i in ids})
        table = dict(zip(flat, read_embedding_rows(MODEL, emb, flat)))
        rows = []
        for ids in token_ids:
            matrix = np.stack([table[i] for i in ids]) if ids else np.zeros((1, emb.shape[0]))
            rows.append(matrix.mean(axis=0))
        return np.asarray(rows, dtype=np.float32)

    train_texts = corpus_sentences()
    train = qwen_embeddings(train_texts)
    test = qwen_embeddings(SENTENCES)
    print(f"corpus {len(train_texts)} phrases | eval {len(SENTENCES)}", flush=True)

    model = Distiller()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    xt = torch.from_numpy(train)
    steps = 500
    batch = 512
    for step in range(steps):
        idx = torch.from_numpy(rng.integers(0, len(train), size=(batch * 2,)))
        x = xt[idx]
        target = F.normalize(x, dim=-1) @ F.normalize(x, dim=-1).T
        z = model(x)
        pred = z @ z.T
        loss = F.mse_loss(pred, target)
        # VICReg-lite : empêche l'effondrement de l'encodeur appris.
        std = z.std(dim=0)
        loss = loss + 0.05 * F.relu(0.05 - std).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (step + 1) % 100 == 0:
            print(f"step {step + 1} | loss {float(loss):.4f}", flush=True)

    with torch.no_grad():
        mapped_test = model(torch.from_numpy(test)).numpy()
        mapped_train = model(xt).numpy()

    base_cos = pairwise_cos(test)
    mapped_cos = pairwise_cos(mapped_test)
    rho = spearman(base_cos, mapped_cos)
    error = float(np.abs(base_cos - mapped_cos).mean())

    def retrieval(project) -> int:
        facts = project(qwen_embeddings([p[0] for p in PAIRS]))
        queries = project(qwen_embeddings([p[1] for p in PAIRS]))
        facts = facts / (np.linalg.norm(facts, axis=1, keepdims=True) + 1e-9)
        queries = queries / (np.linalg.norm(queries, axis=1, keepdims=True) + 1e-9)
        hits = 0
        for index, (_fact, _query, expected) in enumerate(PAIRS):
            best = int(np.argmax(facts @ queries[index]))
            hits += int(expected.lower() in PAIRS[best][0].lower())
        return hits

    with torch.no_grad():
        def project(x: np.ndarray) -> np.ndarray:
            return model(torch.from_numpy(np.asarray(x, dtype=np.float32))).numpy()
        base_hits = retrieval(lambda x: x)
        learned_hits = retrieval(project)

    result = {
        "thresholds": THRESHOLDS,
        "train_sentences": len(train_texts),
        "steps": steps,
        "mapping": "MLP 896->256->32, cosine distillation",
        "spearman": round(rho, 4),
        "cos_error_mean": round(error, 4),
        "pass_s1": bool(rho >= THRESHOLDS["spearman_min"]),
        "pass_s2": bool(error <= THRESHOLDS["cos_error_max"]),
        "retrieval": {
            "qwen": base_hits,
            "learned": learned_hits,
            "pass_s3": bool(
                learned_hits >= THRESHOLDS["hit_at_1_min"] and learned_hits >= base_hits - 1
            ),
        },
    }
    result["verdict"] = (
        "VERT" if (result["pass_s1"] and result["pass_s2"] and result["retrieval"]["pass_s3"])
        else "ROUGE"
    )
    if result["verdict"] == "VERT":
        torch.save({"state_dict": model.state_dict(), "dim_in": 896, "dim_out": DIM}, WEIGHTS)
        result["weights"] = str(WEIGHTS.relative_to(ROOT))
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print("écrit", OUT)
    return 0 if result["verdict"] == "VERT" else 1


if __name__ == "__main__":
    raise SystemExit(main())
