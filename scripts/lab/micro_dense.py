"""Bac a sable 1 : micro-transformer dense (2 couches, 16 dims, tache mot suivant).

Le but n'est pas d'avoir un bon modele, c'est d'avoir un **temoin dense** de la
meme taille que le micro-MYCELIA, entraine sur exactement le meme flux de mots,
pour que la comparaison des deux et la conversion aient un sens. Tout est
reimplemente a la main au-dessus de torch : aucune dependance ajoutee.

Ce que le modele fait : lire 8 mots, predire le mot suivant a chaque position
(masque causal). Le scan du traducteur lit ensuite les unites cachees, qui sont
ses "noeuds" a lui.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from lab_common import LAB, SEED, ToyCorpus, ensure_lab, set_seed, write_json

DIM = 16
LAYERS = 2
HEADS = 2
MLP = 64
SEQ = 8


class Block(nn.Module):
    """Pre-LN : attention causale puis MLP, deux residus."""

    def __init__(self, dim: int = DIM, heads: int = HEADS, mlp: int = MLP) -> None:
        super().__init__()
        self.heads = int(heads)
        self.dim = int(dim)
        self.ln1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.ln2 = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, mlp)
        self.fc2 = nn.Linear(mlp, dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch, seq, dim = x.shape
        h = self.ln1(x)
        q, k, v = self.qkv(h).split(dim, dim=-1)
        head_dim = dim // self.heads
        shape = (batch, seq, self.heads, head_dim)
        q, k, v = (t.view(shape).transpose(1, 2) for t in (q, k, v))
        scores = (q @ k.transpose(-1, -2)) / math.sqrt(head_dim) + mask
        attn = torch.softmax(scores, dim=-1)
        mixed = (attn @ v).transpose(1, 2).reshape(batch, seq, dim)
        x = x + self.proj(mixed)
        x = x + self.fc2(F.gelu(self.fc1(self.ln2(x))))
        return x


class DenseTransformer(nn.Module):
    """Le temoin dense : embedding + position + 2 blocs + tete de lecture."""

    def __init__(self, vocab: int, dim: int = DIM, layers: int = LAYERS,
                 heads: int = HEADS, mlp: int = MLP, seq: int = SEQ) -> None:
        super().__init__()
        self.vocab = int(vocab)
        self.dim = int(dim)
        self.seq = int(seq)
        self.embed = nn.Embedding(vocab, dim)
        self.pos = nn.Embedding(seq, dim)
        self.blocks = nn.ModuleList(Block(dim, heads, mlp) for _ in range(layers))
        self.ln_f = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab)
        mask = torch.triu(torch.full((seq, seq), float("-inf")), diagonal=1)
        self.register_buffer("mask", mask)

    def _encode(self, tokens: torch.Tensor) -> torch.Tensor:
        batch, seq = tokens.shape
        x = self.embed(tokens) + self.pos(torch.arange(seq, device=tokens.device)).unsqueeze(0)
        # Masque decoupe a la longueur lue : une prediction sur un prefixe de
        # trois mots ne doit pas exiger huit colonnes (bug mesure au premier
        # essai de `predict_dense`).
        mask = self.mask[:seq, :seq]
        for block in self.blocks:
            x = block(x, mask)
        return self.ln_f(x)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.head(self._encode(tokens))

    def parts(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Etat cache final et logits : ce que la distillation doit aligner."""
        hidden = self._encode(tokens)
        return hidden, self.head(hidden)

    # ------------------------------------------------------------------
    # Scan : les unites cachees sont les "noeuds" du temoin dense
    # ------------------------------------------------------------------
    @torch.no_grad()
    def unit_activations(self, tokens: torch.Tensor) -> torch.Tensor:
        """[batch, L * MLP + dim] : sorties MLP + residu final, max sur les positions.

        Pourquoi les sorties MLP et pas les embeddings : ce sont les seules
        unites qui *calculent* quelque chose de partage entre positions, comme
        un noeud de la colonie. Pourquoi un max sur les positions : meme
        reduction que NUMENIA-SCAN (S1) sur la membrane d'un noeud, sans quoi
        les deux profils d'activation ne seraient pas comparables.
        """
        batch, seq = tokens.shape
        x = self.embed(tokens) + self.pos(torch.arange(seq, device=tokens.device)).unsqueeze(0)
        rows: list[torch.Tensor] = []
        for block in self.blocks:
            h = block.ln1(x)
            q, k, v = block.qkv(h).split(self.dim, dim=-1)
            head_dim = self.dim // block.heads
            shape = (batch, seq, block.heads, head_dim)
            q, k, v = (t.view(shape).transpose(1, 2) for t in (q, k, v))
            attn = torch.softmax(
                (q @ k.transpose(-1, -2)) / math.sqrt(head_dim) + self.mask[:seq, :seq],
                dim=-1,
            )
            x = x + block.proj((attn @ v).transpose(1, 2).reshape(batch, seq, self.dim))
            hidden = F.gelu(block.fc1(block.ln2(x)))
            rows.append(hidden.amax(dim=1))
            x = x + block.fc2(hidden)
        rows.append(self.ln_f(x).amax(dim=1))
        return torch.cat(rows, dim=-1)

    # ------------------------------------------------------------------
    # Croissance, pour la traduction inverse (numenia -> dense)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def grow_mlp(self, block_index: int, incoming: torch.Tensor) -> int:
        """Ajoute un neurone cache au bloc vise, sans changer sa fonction.

        La ligne ajoutee a ``fc1`` est la direction du concept projete (ce que
        le neurone ecoute) ; la colonne ajoutee a ``fc2`` est a zero, donc le
        neurone ne parle pas encore : la fonction du modele est exactement
        preservee jusqu'au premier pas d'affinage. C'est l'equivalent dense
        d'un noeud qui nait muet.
        """
        block = self.blocks[int(block_index)]
        old_in, old_out = block.fc1.out_features, block.fc2.in_features
        weight1 = block.fc1.weight.detach().clone()
        bias1 = block.fc1.bias.detach().clone()
        weight2 = block.fc2.weight.detach().clone()
        new1 = torch.zeros(old_in + 1, weight1.shape[1])
        new1[:old_in] = weight1
        new1[old_in] = incoming.detach().reshape(-1)
        new2 = torch.zeros(weight2.shape[0], old_out + 1)
        new2[:, :old_out] = weight2
        block.fc1 = nn.Linear(weight1.shape[1], old_in + 1)
        block.fc2 = nn.Linear(old_out + 1, weight2.shape[0])
        with torch.no_grad():
            block.fc1.weight.copy_(new1)
            block.fc1.bias.copy_(torch.cat([bias1, bias1.new_zeros(1)]))
            block.fc2.weight.copy_(new2)
        return old_in + 1


def build_model(vocab: int, seed: int = SEED, mlp: int = MLP) -> DenseTransformer:
    torch.manual_seed(int(seed))
    return DenseTransformer(vocab, mlp=int(mlp))


def n_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def train_dense(
    model: DenseTransformer,
    corpus: ToyCorpus,
    steps: int,
    lr: float = 3e-3,
    batch: int = 32,
    seed: int = SEED,
    log_every: int = 100,
    val_batches=None,
    weight_decay: float = 0.01,
    patience: int = 0,
    stop_at: float | None = None,
) -> dict:
    """Entraine le temoin dense et rend la courbe de perte.

    ``patience`` > 0 garde l'etat de la **meilleure validation** et s'arrete
    quand elle ne s'ameliore plus (meme definition de la convergence que pour
    le micro-MYCELIA, sinon les deux courbes ne sont pas comparables).

    ``stop_at`` arrete des que la validation passe sous cette valeur : c'est ce
    qui permet aux jumeaux de S2 de s'arreter au **meme savoir** et pas au meme
    nombre de pas.
    """
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    best = {"val": float("inf"), "step": 0, "state": None}
    stale = 0
    generator = torch.Generator().manual_seed(int(seed))
    val = val_batches if val_batches is not None else corpus.fixed(16, "val")
    curve: list[dict] = []
    model.train()
    started = time.time()
    for step in range(1, int(steps) + 1):
        x, y = corpus.batch(batch, generator=generator)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % int(log_every) == 0 or step == 1:
            model.eval()
            with torch.no_grad():
                total = 0.0
                for vx, vy in val:
                    out = model(vx)
                    total += float(
                        F.cross_entropy(out.reshape(-1, out.shape[-1]), vy.reshape(-1))
                    )
            model.train()
            value = total / max(1, len(val))
            if value < best["val"]:
                best = {
                    "val": value,
                    "step": step,
                    "state": {k: v.detach().clone() for k, v in model.state_dict().items()},
                }
                stale = 0
            else:
                stale += 1
            curve.append(
                {
                    "step": step,
                    "loss": round(float(loss.detach()), 4),
                    "val": round(float(value), 4),
                }
            )
            if stop_at is not None and value <= float(stop_at):
                curve[-1]["cible_atteinte"] = True
                break
            if patience and stale >= int(patience):
                break
    if best["state"] is not None:
        model.load_state_dict(best["state"])
    return {
        "curve": curve,
        "seconds": round(time.time() - started, 1),
        "final_val": round(float(best["val"]), 4),
        "best_step": int(best["step"]),
    }


@torch.no_grad()
def predict_dense(model: DenseTransformer, corpus: ToyCorpus, prefix: str, count: int = 10) -> list[dict]:
    """10 predictions exemples : le mot suivant propose pour un prefixe."""
    model.eval()
    out = []
    for text in prefix.split("|"):
        ids = corpus.encode(text)[-SEQ:]
        # Remplissage a gauche : la position d'un mot reste la meme que pendant
        # l'entrainement, pour un prefixe plus court qu'une fenetre.
        ids = [0] * (SEQ - len(ids)) + ids
        x = torch.tensor([ids], dtype=torch.long)
        logits = model(x)[0, -1]
        top = torch.topk(logits, 3).indices.tolist()
        out.append(
            {
                "prefixe": text,
                "predit": [corpus.vocab[i] for i in top],
                "attendu": "",
            }
        )
    return out[:count]


def main() -> None:
    parser = argparse.ArgumentParser(description="micro-transformer dense (bac a sable 1)")
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--patience", type=int, default=4)
    args = parser.parse_args()

    ensure_lab()
    set_seed(args.seed)
    corpus = ToyCorpus(seq_len=SEQ)
    model = build_model(corpus.vocab_size, args.seed)
    print(f"micro-dense : {n_parameters(model)} parametres, vocab {corpus.vocab_size}")
    result = train_dense(
        model,
        corpus,
        args.steps,
        lr=args.lr,
        batch=args.batch,
        seed=args.seed,
        patience=args.patience,
    )
    torch.save(
        {
            "state_dict": model.state_dict(),
            "vocab": corpus.vocab,
            "config": {"dim": DIM, "layers": LAYERS, "heads": HEADS, "mlp": MLP, "seq": SEQ},
        },
        LAB / "micro_dense.pt",
    )
    samples = predict_dense(
        model,
        corpus,
        "le chat mange|la fille dort|je ne veux pas|pourquoi le chien court|il y a trois",
    )
    write_json(
        LAB / "micro_dense_result.json",
        {"params": n_parameters(model), "vocab": corpus.vocab_size, **result, "samples": samples},
    )
    for row in result["curve"][-8:]:
        print(f"step {row['step']:5d}  loss {row['loss']:.4f}  val {row['val']:.4f}")
    for row in samples:
        print(f"  {row['prefixe']:26s} -> {' '.join(row['predit'])}")
    print(f"val finale {result['final_val']:.4f} en {result['seconds']}s")


if __name__ == "__main__":
    main()
