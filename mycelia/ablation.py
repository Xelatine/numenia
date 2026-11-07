"""Ablation: a fixed random graph, to ask whether growth itself earns its keep.

The question the ablation answers is not "does a bigger graph learn better" --
of course it does. It is: **a colony that started with that graph, wired at
random and then frozen, would it have done as well?** If yes, the births are
decoration; if no, the births found a topology worth keeping.

Nothing in this module is imported by the training path unless
``growth.mode: frozen_random`` is set in the config, so a real run cannot be
frozen by accident.
"""

from __future__ import annotations

import torch

from mycelia.brain import Brain


def freeze_random_graph(
    brain: Brain,
    edges: int,
    seed: int = 0,
    min_in_degree: int = 1,
) -> dict:
    """Replace the wiring with a random sparse graph of ``edges`` synapses.

    The seed path (0 -> 2..7 -> 1) is kept: without it the door and the reader
    are disconnected and the arm would measure a mute brain, not a random one.
    Every node is given at least ``min_in_degree`` incoming synapses, so no
    node is unreachable by construction -- a grown graph does not contain
    orphans either, since a node only exists because a message reached it.

    Births and deaths are switched off afterwards: this arm must not grow back.
    """
    n = brain.n_nodes
    edges = max(12, min(int(edges), brain.max_edges))
    generator = torch.Generator().manual_seed(int(seed))

    pairs: list[tuple[int, int]] = list(brain.seed_pairs)
    seen = {s * brain.max_population + t for s, t in pairs}

    # One incoming edge per node first: reachability is not what is being tested.
    targets = torch.arange(n).repeat_interleave(int(min_in_degree))
    sources = torch.randint(0, n, (targets.numel(),), generator=generator)
    for source, target in zip(sources.tolist(), targets.tolist()):
        source = int(source)
        target = int(target)
        if source == target:
            continue
        key = source * brain.max_population + target
        if key in seen:
            continue
        seen.add(key)
        pairs.append((source, target))

    while len(pairs) < edges:
        source = int(torch.randint(1, n, (1,), generator=generator))
        target = int(torch.randint(0, n, (1,), generator=generator))
        if source == target:
            continue
        key = source * brain.max_population + target
        if key in seen:
            continue
        seen.add(key)
        pairs.append((source, target))

    tensor = torch.tensor(pairs[:edges], dtype=torch.long).t().contiguous()
    brain.set_edges(tensor)
    # set_edges refresh the counter from the table size; here nothing was born.
    brain.births = 0
    brain.max_births = 0
    brain.max_node_births = 0
    brain.allow_node_growth = False
    brain.node_alive.fill_(True)
    return {"nodes": n, "edges": brain.ne, "seed": int(seed)}
