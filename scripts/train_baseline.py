"""The number without which no result is interpretable: a small Transformer.

Same corpus (LeCarnet), same tokenizer, same budget, same seed, same sequence
length as the text run of the colony. It is a conventional 2-layer, 128-wide,
4-head decoder (~2 M parameters, embeddings tied), trained with the same
AdamW/step budget. Its validation perplexity is written to
``baseline_ppl.txt``; the colony has to be read against that number, not in a
vacuum.

    python scripts/train_baseline.py --steps 10000
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from mycelia.config import load_config  # noqa: E402
from mycelia.train import get_text_dataset  # noqa: E402


class Block(nn.Module):
    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        length = x.shape[1]
        causal = torch.triu(torch.ones(length, length, device=x.device, dtype=torch.bool), diagonal=1)
        attended, _ = self.attn(x, x, x, attn_mask=causal, need_weights=False)
        x = self.norm1(x + attended)
        return self.norm2(x + self.mlp(x))


class TinyTransformer(nn.Module):
    def __init__(self, vocab: int, dim: int = 128, heads: int = 4, layers: int = 2, seq: int = 128):
        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.position = nn.Embedding(seq, dim)
        self.blocks = nn.ModuleList(Block(dim, heads) for _ in range(layers))
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab, bias=False)
        self.head.weight = self.embed.weight  # tied: keeps the budget ~2 M
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """GPT-2-style init: the first loss must be ln(V), not twice it.

        The default init draws the embeddings from N(0, 1) and the head is tied
        to them, so the logits of an untrained model have a standard deviation
        near 11 on a 128-wide readout. The first loss was 21.9 where "knowing
        nothing" is ln(10 000) = 9.21: that is a broken scale, not a hard start,
        and it turns the first hundred steps into a race back toward uniform.
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.position.weight, mean=0.0, std=0.01)
        # Residual branches are scaled so that depth does not change the scale.
        scale = 0.02 / math.sqrt(2 * len(self.blocks))
        for block in self.blocks:
            nn.init.normal_(block.attn.out_proj.weight, mean=0.0, std=scale)
            nn.init.normal_(block.mlp[2].weight, mean=0.0, std=scale)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        x = self.embed(tokens) + self.position(positions)
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x))


def main() -> None:
    parser = argparse.ArgumentParser(description="train the transformer baseline")
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report-every", type=int, default=200)
    parser.add_argument("--out", type=Path, default=ROOT / "baseline_ppl.txt")
    parser.add_argument("--device", default=None, help="cpu or cuda; default: cuda if available")
    parser.add_argument("--save", type=Path, default=None, help="keep the weights (inference bench)")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    dataset = get_text_dataset(args.seq_len)
    vocab = dataset.tokenizer.get_vocab_size()
    model = TinyTransformer(vocab, seq=args.seq_len).to(device)
    print(f"baseline: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M parameters | {device}")
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr)
    generator = torch.Generator().manual_seed(args.seed)
    # The first loss is a measurement, not a hope: it says whether the scale of
    # the readout is right (ln 10 000 = 9.21) before any step is spent on it.
    probe = torch.Generator().manual_seed(args.seed + 1)
    with torch.no_grad():
        probe_x, probe_y = dataset.sample_batch(args.batch_size, generator=probe)
        probe_logits = model(probe_x.to(device))
        initial = float(
            F.cross_entropy(
                probe_logits[:, :-1].reshape(-1, vocab), probe_y[:, 1:].reshape(-1).to(device)
            )
        )
    print(f"init: perte au pas 0 {initial:.4f} | uniforme ln(V) {math.log(vocab):.4f}")

    started = time.time()
    for step in range(1, args.steps + 1):
        x, y = dataset.sample_batch(args.batch_size, generator=generator)
        x, y = x.to(device), y.to(device)
        logits = model(x)
        # The shift the colony does not need. ``TextDataset`` returns (x, x)
        # because the brain reads its prediction *before* injecting token t, so
        # position t never sees its own target. A causal Transformer is not in
        # that position: with attn_mask = triu(diagonal=1), position t still
        # attends to itself, so training against x would teach the identity (and
        # it did: loss 0.0000, validation perplexity 1.000 at step 50). The
        # baseline therefore predicts x[t+1] from positions <= t, which is the
        # usual next-token shift.
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, vocab), y[:, 1:].reshape(-1))
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimiser.step()
        if step % args.report_every == 0:
            print(
                f"step {step:6d} | perte {loss.item():.4f} | ppl {math.exp(min(loss.item(), 20)):8.2f} "
                f"| {(time.time() - started) / step * 1000:.0f} ms/step"
            )

    model.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for x, y in dataset.windows(8, "val"):
            logits = model(x.to(device))
            total += float(
                F.cross_entropy(
                    logits[:, :-1].reshape(-1, vocab), y[:, 1:].reshape(-1).to(device)
                )
            )
            count += 1
    ppl = math.exp(min(total / max(1, count), 20))
    args.out.write_text(
        f"transformer baseline | {sum(p.numel() for p in model.parameters())} params | "
        f"steps {args.steps} | seed {args.seed} | validation perplexity {ppl:.3f}\n",
        encoding="utf-8",
    )
    if args.save is not None:
        torch.save(
            {
                "model": model.state_dict(),
                "vocab": vocab,
                "dim": 128,
                "heads": 4,
                "layers": 2,
                "seq": args.seq_len,
                "steps": args.steps,
                "val_ppl": ppl,
            },
            args.save,
        )
        print(f"weights: {args.save}")
    print(f"validation perplexity: {ppl:.3f} -> {args.out}")


if __name__ == "__main__":
    main()
