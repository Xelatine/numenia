"""S4 — le controle de capacite : un bigramme de rang 16, seul.

Le MYCELIA repare et le dense jumeau convergent tous les deux vers ~3,8 nats sur
1000 phrases. Est-ce la force du dense, ou le plafond de **toute** lecture de
dimension 16 ? Ce script entraine le plus petit modele possible qui reponde :
`embedding(312x16) -> LayerNorm -> lineaire 16x16 -> LayerNorm -> tete(16x312)`,
c'est-a-dire exactement la voie rapide du MYCELIA repare, sans graphe, sans
contexte au-dela d'un mot.

Si ce modele atteint ~3,8, alors la barre du dense n'est pas « le dense » : c'est
le rang de la lecture. Le corriger demande d'elargir la dimension de lecture,
pas de changer l'architecture du graphe.

Ecrit ``data/lab/s4_rank16.json``.
"""

from __future__ import annotations

import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from lab_common import LAB, ToyCorpus, ensure_lab, set_seed, write_json
from repaired_mycelia import SEQ, aligned_loss

STEPS = 600
BATCH = 32
LR = 3e-3
WEIGHT_DECAY = 0.01
SEED = 1


class Rank16Bigram(nn.Module):
    """Un bigramme de rang 16 : la meme largeur de lecture que le jouet."""

    def __init__(self, vocab: int, dim: int = 16) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab, dim)
        self.ln1 = nn.LayerNorm(dim)
        self.mix = nn.Linear(dim, dim)
        self.ln2 = nn.LayerNorm(dim)
        self.output_head = nn.Linear(dim, vocab)

    def forward(self, tokens: torch.Tensor, grow=None, trace=None) -> torch.Tensor:
        # Meme signature que le cerveau (grow/trace ignores) pour pouvoir etre
        # mesure par `aligned_loss` sans chemin special.
        # Le token precedent, et un neutre pour la position 0 (aucun contexte).
        prev = torch.cat([torch.zeros_like(tokens[:, :1]), tokens[:, :-1]], dim=1)
        return self.output_head(self.ln2(self.mix(self.ln1(self.embedding(prev)))))


def parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def main() -> None:
    ensure_lab()
    corpus = ToyCorpus(seq_len=SEQ)
    val = corpus.fixed(58, "val")
    set_seed(SEED)
    model = Rank16Bigram(corpus.vocab_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    generator = torch.Generator().manual_seed(SEED)
    best = {"val": float("inf"), "step": 0, "state": None}
    started = time.time()
    for step in range(1, STEPS + 1):
        x, _y = corpus.batch(BATCH, generator=generator)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), x.reshape(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % 25 == 0 or step == 1:
            model.eval()
            value = aligned_loss(model, val)
            model.train()
            if value < best["val"]:
                best = {"val": value, "step": step,
                        "state": {k: v.detach().clone() for k, v in model.state_dict().items()}}
    model.load_state_dict(best["state"])
    model.eval()
    payload = {
        "modele": "bigramme de rang 16 (embedding 312x16 -> 16 -> tete)",
        "parametres": parameters(model),
        "best_val": round(float(best["val"]), 4),
        "best_step": int(best["step"]),
        "pas": STEPS,
        "val_finale": round(aligned_loss(model, val), 4),
        "secondes": round(time.time() - started, 1),
        "reference": {"dense": 3.7921, "bigramme_lisse": 3.0466, "unigramme": 5.0829},
    }
    write_json(LAB / "s4_rank16.json", payload)
    print(f"bigramme rang 16 : {payload['best_val']:.4f} au pas {payload['best_step']} "
          f"({payload['parametres']} parametres, {payload['secondes']}s)")
    print("ecrit : data/lab/s4_rank16.json")


if __name__ == "__main__":
    main()
