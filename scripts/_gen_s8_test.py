#!/usr/bin/env python3
import sys, os
sys.path.insert(0, "scripts")
sys.path.insert(0, "scripts/lab")
from pathlib import Path
from micro_dense import DenseTransformer
from lab_common import LAB, ToyCorpus
corpus = ToyCorpus(seq_len=8, path=LAB / "corpus_toy.txt")
m = DenseTransformer(corpus.vocab_size, dim=16, layers=2, heads=2, mlp=166, seq=8)
DONORS = LAB / "s8" / "donors"
DONORS.mkdir(parents=True, exist_ok=True)
pt = DONORS / "A.pt"
import torch
torch.save({
  "state_dict": m.state_dict(),
  "config": {"type":"dense","dim":16,"layers":2,"heads":2,"mlp":166,"seq":8},
  "vocab_size": corpus.vocab_size,
}, str(pt))
print("wrote", pt, "vocab=", corpus.vocab_size, "bytes=", pt.stat().st_size)
