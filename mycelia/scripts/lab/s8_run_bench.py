#!/usr/bin/env python3
"""S8 bench runner — 4 donneurs × 3 semences = 12 traductions (sans vie).

La version simplifiée ne fait pas la vie 2000 pas (trop lente, ~4 min/traduction).
Elle mesure juste F avant vie : KL entre le donneur et son receveur après
distillation NUMENIA pool 24 + entraînement 600 pas.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

import importlib.util
import sys
from pathlib import Path

import torch

_here = Path(__file__).resolve().parent
_REPO = _here.parent.parent
_ROOT_FOR_CODE = _REPO
# Data lives one level above the repo root (../data), resolve it explicitly.
_DATA_ROOT = (_REPO.parent / "data").resolve()
if not _DATA_ROOT.exists():
    _DATA_ROOT = (_REPO / "data").resolve()
if not _DATA_ROOT.exists():
    _DATA_ROOT = Path.cwd().parent.resolve() / "data"
if not _DATA_ROOT.exists():
    _DATA_ROOT = Path.cwd() / "data"
ROOT = _REPO

MODULES = {}


def _import_file(name: str, rel: str):
    path = _ROOT_FOR_CODE / rel
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    # Register numenia package so intra-project imports like
    # `from mycelia.lifecycle import ...` resolve.
    pkg = sys.modules.setdefault("mycelia", importlib.util.module_from_spec(
        importlib.util.spec_from_file_location("mycelia", ROOT / "__init__.py")
    ))
    if not getattr(pkg, "__path__", None):
        pkg.__path__ = [str(ROOT)]
    spec.loader.exec_module(mod)
    return mod


_brain = _import_file("brain", "brain.py")
_checkpoint = _import_file("checkpoint", "checkpoint.py")
_data = _import_file("data", "data.py")
_lifecycle = _import_file("lifecycle", "lifecycle.py")
_growth = _import_file("growth", "growth.py")

sys.modules.setdefault("mycelia.brain", _brain)
sys.modules.setdefault("mycelia.checkpoint", _checkpoint)
sys.modules.setdefault("mycelia.data", _data)
sys.modules.setdefault("mycelia.lifecycle", _lifecycle)
sys.modules.setdefault("mycelia.growth", _growth)

Brain = _brain.Brain
save_genome = _checkpoint.save_genome
load_genome = _checkpoint.load_genome
get_batch = _data.get_batch
apply_lifecycle = _lifecycle.apply_lifecycle
GrowthManager = _growth.GrowthManager

# ---------------------------------------------------------------------------
# Config S8
# ---------------------------------------------------------------------------
SEQ = 8
VOCAB = 312
TRAIN_STEPS = 600
BATCH_SIZE = 16
MYCELIA_LR = 2e-3
DONORS_DIR = _DATA_ROOT / "lab" / "s8" / "donors"
BENCH_DIR = _DATA_ROOT / "lab" / "s8"
DIVERGENCE_PATH = BENCH_DIR / "divergence.json"

CONFIG = {
    "brain": {
        "n_nodes": 512,
        "dim": 32,
        "vocab": VOCAB,
        "max_hops": 8,
        "dissipation": 0.9,
        "state_decay": 0.995,
        "max_edges": 20000,
        "max_births": 8,
        "max_node_births": 2,
        "birth_k": 4,
        "birth_candidates": 64,
        "init_bias": -8.0,
        "edge_mask_init": 0.5,
        "birth_strength": 0.5,
        "weak_noise": 0.02,
        "node_margin": 32,
        "node_birth_interval": 500,
        "trace_limit": 2000,
        "edge_chunk_size": 512,
        "birth_mode": "traffic",
        "birth_seed": 0,
        "activation": "gelu",
        "output_sparse": False,
    },
    "lifecycle": {
        "prune_every": 100,
        "prune_ratio": 0.1,
        "prune_fraction": 0.1,
        "prune_threshold": 0.02,
        "isolate_grace": 300,
        "identity_weight": 0.01,
        "identity_cosine": 0.999,
        "redundancy_weight": 0.02,
        "redundancy_cosine": 0.5,
        "redundancy_sample": 64,
    },
    "training": {
        "lr": MYCELIA_LR,
        "clip": 1.0,
    },
    "growth": {
        "check_every": 500,
        "grow_threshold": 0.01,
        "optimize_threshold": 0.005,
        "reallocate": 256,
        "suspend_births": 5000,
        "node_freeze": 200,
    },
}


def load_donor(name: str):
    path = DONORS_DIR / f"{name}.pt"
    blob = torch.load(path, map_location="cpu", weights_only=False)
    cfg = blob.get("config", {})
    kind = cfg.get("type", "brain")
    if kind == "dense":
        return _load_dense_donor(name, blob)
    if kind == "brain":
        return _load_brain_donor(name, blob)
    raise ValueError(f"donor {name} type {kind!r} inconnu")


def _load_brain_donor(name: str, blob: dict) -> Brain:
    cfg = dict(blob.get("config", {}))
    for k in ("type", "vocab_size", "nodes_read", "two_speeds"):
        cfg.pop(k, None)
    brain = Brain(**cfg)
    brain.load_state_dict(blob.get("state_dict", {}))
    if "ne" in blob:
        brain.ne = int(blob["ne"])
        brain._rebuild_edge_keys()
    if "clock" in blob:
        brain.clock = int(blob["clock"])
    if "births" in blob:
        brain.births = int(blob["births"])
    if "node_births" in blob:
        brain.node_births = int(blob["node_births"])
    if "edge_deaths" in blob:
        brain.edge_deaths = int(blob["edge_deaths"])
    if "node_deaths" in blob:
        brain.node_deaths = int(blob["node_deaths"])
    if "cemetery" in blob:
        brain.cemetery = {
            int(k): {n: t.clone() for n, t in v.items()}
            for k, v in blob["cemetery"].items()
        }
    return brain


def _load_dense_donor(name: str, blob: dict):
    """Dense donor : on ne peut pas le charger dans un Brain.

    On renvoie un objet minimal qui répond à .forward(x, grow=False)
    avec les logits du modèle dense, et à .to(device).
    """
    from mycelia.brain import Brain as B
    cfg = dict(blob.get("config", {}))
    cls_name = {"dense": "DenseDonor"}.get(cfg.get("type", "dense"), "DenseDonor")
    if cls_name == "DenseDonor":
        return _DenseDonor(blob)
    raise ValueError(f"dense donor {name} : type {cfg.get('type')!r} non géré")


class _DenseDonor:
    """Wrapper minimal pour un donneur dense sérialisé."""

    def __init__(self, blob: dict):
        self.blob = blob
        self.cfg = blob.get("config", {})
        self.state_dict_blob = blob.get("state_dict", {})
        self._model = None

    def to(self, device):
        self.device = device
        return self

    def forward(self, x, grow=False):
        if self._model is None:
            self._build()
        if x.dim() == 2:
            x = x.unsqueeze(1)
        return self._model(x)

    def __call__(self, x, grow=False):
        return self.forward(x, grow=grow)

    def _build(self):
        import torch.nn as nn
        cfg = self.cfg
        dim = int(cfg.get("dim", 16))
        layers = int(cfg.get("layers", 2))
        heads = int(cfg.get("heads", 2))
        mlp = int(cfg.get("mlp", 768))
        vocab = int(self.blob.get("vocab_size", 312))
        class _DenseBlockFixed(nn.Module):
            def __init__(self, dim, heads, mlp):
                super().__init__()
                self.ln1 = nn.LayerNorm(dim)
                self.qkv = nn.Linear(dim, dim * 3, bias=True)
                self.proj = nn.Linear(dim, dim, bias=True)
                self.ln2 = nn.LayerNorm(dim)
                self.fc1 = nn.Linear(dim, mlp, bias=True)
                self.fc2 = nn.Linear(mlp, dim, bias=True)
            def forward(self, h):
                B, T, C = h.shape
                x = self.ln1(h)
                qkv = self.qkv(x).view(B, T, 3, C).permute(2, 0, 1, 3)
                qq = qkv[0]
                kk = qkv[1]
                vv = qkv[2]
                attn = (qq @ kk.transpose(-2, -1)) * (1.0 / C**0.5)
                attn = attn.softmax(dim=-1)
                # qq, kk, vv shapes (B, T, C)
                h2 = (attn @ vv).transpose(1, 2).reshape(B, T, C)
                h = h + self.proj(h2)
                h = h + self.fc2(nn.functional.gelu(self.fc1(self.ln2(h))))
                return h
        class _Model(nn.Module):
            def __init__(self, vocab, dim, layers, heads, mlp):
                super().__init__()
                self.embed = nn.Embedding(vocab, dim)
                for i in range(layers):
                    setattr(self, f"blocks.{i}", _DenseBlockFixed(dim, heads, mlp))
                self.ln_f = nn.LayerNorm(dim)
                self.head = nn.Linear(dim, vocab)
            def forward(self, x):
                h = self.embed(x)
                for i in range(layers):
                    h = getattr(self, f"blocks.{i}")(h)
                return self.head(self.ln_f(h))
        self._model = _Model(vocab, dim, layers, heads, mlp)
        with torch.no_grad():
            for k, v in self._model.state_dict().items():
                if k in self.state_dict_blob:
                    src = self.state_dict_blob[k]
                    if v.shape == src.shape:
                        v.copy_(src)
        self._model.to(self.device)
        self._model.eval()
        self.device = self._model.embed.weight.device

    def eval(self):
        if self._model is not None:
            self._model.eval()
        return self



def build_vierge_pool24(seed: int) -> Brain:
    """NUMENIA pool 24 vierge — cerveau receveur de la traduction."""
    cfg = dict(CONFIG["brain"])
    cfg["n_nodes"] = 24
    cfg["birth_seed"] = seed
    cfg["vocab"] = VOCAB
    brain = Brain(**cfg)
    return brain


def distill_and_train(donor: Brain, recipient: Brain, seed: int) -> dict:
    """Distillation a1b1g0 + entraînement 600 pas sur le receveur.

    Retourne les métriques : perte, KL, best_step, temps.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    donor = donor.to(device)
    recipient = recipient.to(device)
    recipient.train()

    optimizer = torch.optim.AdamW(recipient.parameters(), lr=MYCELIA_LR)
    generator = torch.Generator().manual_seed(seed)

    best_loss = float("inf")
    best_step = 0
    last_loss = 0.0
    start = time.time()

    for step in range(1, TRAIN_STEPS + 1):
        x, y = get_batch(BATCH_SIZE, SEQ, generator=generator)
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad(set_to_none=True)

        with torch.no_grad():
            donor_logits, _ = donor(x, grow=False)

        recipient_logits, _ = recipient(x, grow=False)

        # Distillation : KL(recipient || donor)
        recipient_log_probs = F.log_softmax(recipient_logits, dim=-1)
        donor_probs = F.softmax(donor_logits, dim=-1)
        kl = F.kl_div(recipient_log_probs, donor_probs, reduction="batchmean", log_target=False)
        loss = kl

        loss.backward()
        torch.nn.utils.clip_grad_norm_(recipient.parameters(), 1.0)
        optimizer.step()

        last_loss = float(loss.detach())
        if last_loss < best_loss:
            best_loss = last_loss
            best_step = step

        if step % 100 == 0:
            print(f"  step {step:4d} | KL {last_loss:.4f} | best {best_loss:.4f} (pas {best_step})")

    elapsed = time.time() - start

    # Mesure finale KL sur un batch de validation
    recipient.eval()
    with torch.no_grad():
        x_val, _ = get_batch(BATCH_SIZE, SEQ, generator=torch.Generator().manual_seed(seed + 999))
        x_val = x_val.to(device)
        d_logits, _ = donor(x_val, grow=False)
        r_logits, _ = recipient(x_val, grow=False)
        r_lp = F.log_softmax(r_logits, dim=-1)
        d_p = F.softmax(d_logits, dim=-1)
        final_kl = float(F.kl_div(r_lp, d_p, reduction="batchmean", log_target=False).detach().cpu())

    return {
        "perte": round(last_loss, 4),
        "best_step": best_step,
        "f_avant": round(final_kl, 4),
        "temps": round(elapsed, 1),
        "steps": TRAIN_STEPS,
        "batch_size": BATCH_SIZE,
        "seq": SEQ,
        "lr": MYCELIA_LR,
        "pool": 24,
    }


def compute_top3(recipient: Brain, donor: Brain, seed: int) -> dict:
    """Top-3 contextes identiques vs neufs sur 5000 contextes (version light)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    donor = donor.to(device)
    recipient = recipient.to(device)

    gen = torch.Generator().manual_seed(seed)
    n_ctx = 5000
    identical = 0
    neuf = 0

    with torch.no_grad():
        for i in range(n_ctx):
            x, _ = get_batch(1, SEQ, generator=gen)
            x = x.to(device)
            d_logits, _ = donor(x, grow=False)
            r_logits, _ = recipient(x, grow=False)
            d_pred = d_logits.argmax(dim=-1).item()
            r_pred = r_logits.argmax(dim=-1).item()
            if d_pred == r_pred:
                identical += 1
            # "neuf" = le receveur prédit quelque chose de différent
            neuf += 1 if d_pred != r_pred else 0

    return {
        "top3_identiques": round(identical / n_ctx, 4),
        "top3_neuf": round(neuf / n_ctx, 4),
        "n_contextes": n_ctx,
    }


def run_translation(donor_name: str, seed: int) -> dict:
    """Une traduction complète : donneur → receveur NUMENIA pool 24."""
    print(f"[S8] traduction {donor_name}/seed{seed} — démarrage")
    donor = load_donor(donor_name)
    recipient = build_vierge_pool24(seed)
    recipient.seed = seed

    metrics = distill_and_train(donor, recipient, seed)

    # Top-3 (mesure rapide sur 5000 contextes)
    top3 = compute_top3(recipient, donor, seed)
    metrics.update(top3)

    # Sauvegarde du receveur
    recipient_path = BENCH_DIR / f"recipient_{donor_name}_seed{seed}.pt"
    save_genome(
        recipient_path,
        recipient,
        step=TRAIN_STEPS,
        optimizer=None,
        growth=None,
        tokenizer=None,
        history=None,
        extra={
            "donor": donor_name,
            "seed": seed,
            "s8_bench": True,
            "f_avant": metrics["f_avant"],
        },
    )

    # Sauvegarde du bench
    bench_path = BENCH_DIR / f"bench_{donor_name}_seed{seed}.json"
    bench_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[S8] {donor_name}/seed{seed} terminé — F_avant={metrics['f_avant']} | top3_neuf={metrics['top3_neuf']} | {metrics['temps']}s")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="S8 bench — 4 donneurs × 3 semences")
    parser.add_argument("--donors", nargs="+", default=["A", "B", "C", "D"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--all", action="store_true", help="lancer les 12 traductions")
    args = parser.parse_args()

    if args.all:
        donors = ["A", "B", "C", "D"]
        seeds = [1, 2, 3]
    else:
        donors = args.donors
        seeds = args.seeds

    results = {}
    for donor_name in donors:
        for seed in seeds:
            key = f"{donor_name}_seed{seed}"
            bench_path = BENCH_DIR / f"bench_{donor_name}_seed{seed}.json"
            if bench_path.exists():
                print(f"[S8] {key} déjà fait — skip")
                continue
            try:
                metrics = run_translation(donor_name, seed)
                results[key] = metrics
            except Exception as e:
                print(f"[S8] ERREUR {donor_name}/seed{seed}: {e}")
                results[key] = {"erreur": str(e)}

    # Résumé
    print("\n=== RÉSULTATS S8 ===")
    for key, m in results.items():
        if "erreur" in m:
            print(f"  {key}: ERREUR {m['erreur']}")
        else:
            verdict = "VERT" if m.get("f_avant", 0) >= 0.5 else "ROUGE"
            print(f"  {key}: F_avant={m.get('f_avant')} | top3_neuf={m.get('top3_neuf')} | {verdict}")

    # Rapport synthétique
    report_path = BENCH_DIR / "bench_synthetique.json"
    report = {
        "donneurs": donors,
        "seeds": seeds,
        "resultats": results,
        "resume": {
            k: v.get("f_avant")
            for k, v in results.items()
            if "erreur" not in v
        },
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nRapport synthétique : {report_path}")


if __name__ == "__main__":
    main()
