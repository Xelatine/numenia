"""Bac a sable 2 : micro-MYCELIA — le vrai `Brain`, en petit.

Ce n'est pas une reimplementation : c'est ``mycelia.brain.Brain`` avec un pool
de 24 noeuds, ``dim=16`` et le meme vocabulaire que le temoin dense, entraine
sur le meme flux de mots avec la meme tache. Le metabolisme minimal est celui
du projet : naissances payees par le trafic (``_birth_edges``), adresses
inconnues honorees entre deux pas (``apply_pending_node_births``), puis coupe
des synapses faminees et enterrement des noeuds isoles (``apply_lifecycle``).

Aucun fichier du cerveau reel n'est touche : ce bac ecrit ses propres saves
dans ``data/lab/`` et son propre ``metrics_lab.json``.
"""

from __future__ import annotations

import argparse
import time

import torch
import torch.nn.functional as F

from lab_common import LAB, SEED, ToyCorpus, ensure_lab, mycelia_loss, set_seed, write_json
from mycelia.brain import Brain
from mycelia.checkpoint import (
    brain_from_genome,
    capture_optimizer_state,
    load_genome,
    optimizer_for,
    restore_optimizer_state,
    save_genome,
)
from mycelia.growth import apply_pending_node_births
from mycelia.lifecycle import apply_lifecycle

SEQ = 8
POOL = 24
KW = {
    "dim": 16,
    "max_hops": 4,
    "max_edges": 800,
    "max_births": 8,
    "max_node_births": 2,
    "birth_k": 4,
    "birth_candidates": 32,
    "init_bias": -4.0,
    "node_birth_interval": 25,
    "node_freeze": 50,
    "isolate_grace": 100,
    "max_population": 96,
    "prune_ratio": 0.1,
    "prune_fraction": 0.1,
    "trace_limit": 64,
    "state_decay": 0.95,
}


def build_mycelia(vocab: int, seed: int = SEED, overrides: dict | None = None) -> Brain:
    set_seed(seed)
    kwargs = dict(KW)
    kwargs.update(overrides or {})
    return Brain(n_nodes=int(kwargs.pop("n_nodes", POOL)), vocab=int(vocab), **kwargs)


def snapshot(brain: Brain) -> dict:
    """Sauvegarde en memoire de l'etat complet (poids + compteurs + pointeur)."""
    return {
        "state_dict": {key: value.detach().clone() for key, value in brain.state_dict().items()},
        "n_nodes": int(brain.n_nodes),
        "ne": int(brain.ne),
        "clock": int(brain.clock),
        "births": int(brain.births),
        "node_births": int(brain.node_births),
        "edge_deaths": int(brain.edge_deaths),
        "node_deaths": int(brain.node_deaths),
        "cemetery": {k: {n: t.clone() for n, t in v.items()} for k, v in brain.cemetery.items()},
    }


def restore(brain: Brain, snap: dict) -> Brain:
    """Revient au point de convergence mesure, en **reconstruisant** le cerveau.

    Un pool ne retrecit pas (``grow_population`` refuse ``new_n <= old_n``) :
    apres un arret sur patience, le cerveau courant est plus gros que le point
    garde. On rebatit donc un ``Brain`` a la taille de la sauvegarde et on y
    recharge les poids — c'est la meme operation que ``brain_from_genome``.
    """
    config = dict(brain.config())
    config["n_nodes"] = int(snap["n_nodes"])
    fresh = Brain(**config)
    fresh.load_state_dict(snap["state_dict"])
    fresh.ne = int(snap["ne"])
    fresh.clock = int(snap["clock"])
    fresh.births = int(snap["births"])
    fresh.node_births = int(snap["node_births"])
    fresh.edge_deaths = int(snap["edge_deaths"])
    fresh.node_deaths = int(snap["node_deaths"])
    fresh.cemetery = {
        k: {n: t.clone() for n, t in v.items()} for k, v in snap["cemetery"].items()
    }
    fresh._rebuild_edge_keys()
    fresh.eval()
    return fresh


def n_parameters(brain: Brain) -> int:
    return sum(p.numel() for p in brain.parameters())


def _swap_population(brain: Brain, optimizer, lr: float):
    """Le pool a grandi : les parametres sont neufs, l'optimiseur doit suivre.

    ``bank.grow`` remplace des ``nn.Parameter`` : l'optimiseur garde les anciens
    objets et n'apprendrait plus. On capture les moments par *nom* (la lecon du
    checkpoint) et on les re-attache ; les lignes nouvelles repartent a zero.
    """
    moments = capture_optimizer_state(brain, optimizer)
    fresh = optimizer_for(brain, lr)
    restore_optimizer_state(brain, fresh, moments)
    return fresh


def train_mycelia(
    brain: Brain,
    corpus: ToyCorpus,
    steps: int,
    lr: float = 2e-3,
    batch: int = 16,
    seed: int = SEED,
    log_every: int = 50,
    lifecycle_every: int = 100,
    val_batches=None,
    weight_decay: float = 0.0,
    patience: int = 0,
    stop_at: float | None = None,
    restore_best: bool = False,
    stop_births_at: int | None = None,
) -> dict:
    """Entraine le micro-MYCELIA et rend la courbe + les evenements vitaux.

    ``patience`` > 0 active l'arret sur la **meilleure validation** : sur 1000
    phrases, les deux bacs memorisent en quelques centaines de pas et la
    validation remonte. Converger veut dire ici "la validation ne s'ameliore
    plus" ; l'etat garde est celui de ce minimum, pas le dernier pas.

    ``stop_at`` arrete des que la validation passe sous cette valeur : les
    jumeaux de S2 s'arretent au meme **savoir**, pas au meme nombre de pas.

    ``restore_best`` (S3-bis) : quand ``patience`` vaut 0 (courbe longue, aucun
    arret anticipe), le cerveau rendu reste au dernier pas — or la validation
    remonte apres le minimum. Le drapeau rend l'etat du **meilleur** pas, ce que
    ``best_val`` annonce deja ; sans lui, le chiffre et le modele divergent.
    """
    optimizer = torch.optim.AdamW(brain.parameters(), lr=lr, weight_decay=weight_decay)
    best = {"val": float("inf"), "step": 0, "snap": snapshot(brain)}
    stale = 0
    generator = torch.Generator().manual_seed(int(seed))
    val = val_batches if val_batches is not None else corpus.fixed(16, "val")
    curve: list[dict] = []
    events: list[dict] = []
    pool_size = int(brain.n_nodes)
    last_n = int(brain.n_nodes)
    started = time.time()
    reached = False
    brain.train()
    for step in range(1, int(steps) + 1):
        if stop_births_at is not None and step >= int(stop_births_at) and brain.max_births:
            # Ablation S3-bis D4(ii) : metabolisme gele. Le cablage atteint au
            # pas N reste tel quel ; plus aucune synapse ne nait.
            brain.max_births = 0
        x, _ = corpus.batch(batch, generator=generator)
        target = corpus.targets(x)
        brain.clock = step
        logits = brain(x, grow=True, trace=False)[0]
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target.reshape(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        brain.freeze_newborn_gradients()
        torch.nn.utils.clip_grad_norm_(brain.parameters(), 1.0)
        optimizer.step()

        if step % 25 == 0:
            born = apply_pending_node_births(brain)
            if born:
                events.append({"step": step, "naissances": len(born), "pool": int(brain.n_nodes)})
        if brain.n_nodes != last_n:
            optimizer = _swap_population(brain, optimizer, lr)
            last_n = int(brain.n_nodes)
        # lifecycle_every = 0 desactive la mort cellulaire (ablation S3-bis D4).
        if lifecycle_every and step % int(lifecycle_every) == 0:
            report = apply_lifecycle(brain, step=step)
            if report["edge_deaths"] or report["node_deaths"]:
                events.append({"step": step, **report, "pool": int(brain.n_nodes)})

        reached = False
        if step % int(log_every) == 0 or step == 1:
            brain.eval()
            value = float(mycelia_loss(brain, val))
            brain.train()
            if value < best["val"]:
                best = {"val": value, "step": step, "snap": snapshot(brain)}
                stale = 0
            else:
                stale += 1
            curve.append(
                {
                    "step": step,
                    "loss": round(float(loss.detach()), 4),
                    "val": round(float(value), 4),
                    "pool": int(brain.n_nodes),
                    "edges": int(brain.ne),
                }
            )
            if stop_at is not None and value <= float(stop_at):
                curve[-1]["cible_atteinte"] = True
                best = {"val": value, "step": step, "snap": snapshot(brain)}
                reached = True
        if reached:
            events.append({"step": step, "arret": "cible", "best_step": best["step"]})
            brain = restore(brain, best["snap"])
            curve.append(
                {
                    "step": int(best["step"]),
                    "loss": curve[-1]["loss"],
                    "val": round(float(best["val"]), 4),
                    "pool": int(brain.n_nodes),
                    "edges": int(brain.ne),
                    "restaure": True,
                }
            )
            break
    patience_stop = bool(patience) and stale >= int(patience)
    if not reached and patience_stop:
        events.append({"step": step, "arret": "patience", "best_step": best["step"]})
        brain = restore(brain, best["snap"])
        curve.append(
            {
                "step": int(best["step"]),
                "loss": curve[-1]["loss"],
                "val": round(float(best["val"]), 4),
                "pool": int(brain.n_nodes),
                "edges": int(brain.ne),
                "restaure": True,
            }
        )
    elif restore_best and best["step"]:
        # Courbe longue : on mesure jusqu'au dernier pas, mais on rend le modele
        # du minimum de validation (aucun arret anticipe, aucune triche).
        events.append({"step": step, "arret": "meilleur", "best_step": best["step"]})
        brain = restore(brain, best["snap"])
    return {
        "brain": brain,
        "best_val": round(float(best["val"]), 4),
        "best_step": int(best["step"]),
        "curve": curve,
        "events": events,
        "seconds": round(time.time() - started, 1),
        "final_val": round(float(best["val"]), 4) if curve else None,
        "pool": int(brain.n_nodes),
        "pool_start": pool_size,
        "edges": int(brain.ne),
        "node_births": int(brain.node_births),
        "node_deaths": int(brain.node_deaths),
        "edge_deaths": int(brain.edge_deaths),
    }


@torch.no_grad()
def predict_mycelia(brain: Brain, corpus: ToyCorpus, prefix: str, count: int = 10) -> list[dict]:
    """Predictions exemples : mot suivant propose apres un prefixe."""
    brain.eval()
    out = []
    for text in prefix.split("|"):
        ids = corpus.encode(text)[-SEQ:]
        state = None
        logits = None
        for token in ids:
            logits, state, _ = brain(
                torch.tensor([[token]]), state=state, grow=False, trace=False, return_state=True
            )
        top = torch.topk(logits[0, 0], 3).indices.tolist()
        out.append(
            {
                "prefixe": text,
                "predit": [corpus.vocab[i] for i in top],
                "attendu": "",
            }
        )
    return out[:count]


@torch.no_grad()
def load_mycelia(path: str | None = None) -> Brain:
    path = path or str(LAB / "micro_mycelia.pt")
    brain = brain_from_genome(load_genome(path))
    brain.eval()
    return brain


def main() -> None:
    parser = argparse.ArgumentParser(description="micro-MYCELIA (bac a sable 2)")
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    args = parser.parse_args()

    ensure_lab()
    set_seed(args.seed)
    corpus = ToyCorpus(seq_len=SEQ)
    brain = build_mycelia(corpus.vocab_size, args.seed)
    print(
        f"micro-mycelia : {n_parameters(brain)} parametres, pool {brain.n_nodes}, "
        f"vocab {corpus.vocab_size}"
    )
    result = train_mycelia(
        brain,
        corpus,
        args.steps,
        lr=args.lr,
        batch=args.batch,
        seed=args.seed,
        weight_decay=args.weight_decay,
        patience=args.patience,
    )
    brain = result.pop("brain")
    save_genome(
        LAB / "micro_mycelia.pt", brain, step=int(result.get("best_step") or args.steps),
        history=result["curve"],
    )
    samples = predict_mycelia(
        brain,
        corpus,
        "le chat mange|la fille dort|je ne veux pas|pourquoi le chien court|il y a trois",
    )
    write_json(
        LAB / "micro_mycelia_result.json",
        {"params": n_parameters(brain), "vocab": corpus.vocab_size, **result, "samples": samples},
    )
    for row in result["curve"][-8:]:
        print(
            f"step {row['step']:5d}  loss {row['loss']:.4f}  val {row['val']:.4f}  "
            f"pool {row['pool']}  aretes {row['edges']}"
        )
    for row in samples:
        print(f"  {row['prefixe']:26s} -> {' '.join(row['predit'])}")
    print(
        f"val finale {result['final_val']:.4f} en {result['seconds']}s — "
        f"pool {result['pool_start']}->{result['pool']}, naissances {result['node_births']}, "
        f"morts {result['node_deaths']}, aretes mortes {result['edge_deaths']}"
    )


if __name__ == "__main__":
    main()
