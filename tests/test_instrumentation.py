"""Two instruments that must not lie: the initial loss, and the living count.

The first test pins the readout's starting point: a model that begins *worse*
than the uniform distribution hides an early regression inside a warm start.
The second test pins the closure of the node counters, after a run where the
arithmetic 512 + births - deaths gave 456 while the log displayed 520: the
missing term was the revivals, which nothing printed.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from mycelia.brain import Brain
from mycelia.lifecycle import kill_isolated_nodes, revive_node

VOCAB = 10_000


def test_initial_loss_near_uniform() -> None:
    """At initialisation the loss must be ln(vocab), not above it."""
    torch.manual_seed(0)
    brain = Brain(n_nodes=16, vocab=VOCAB, max_hops=4, max_births=0, max_edges=256)
    tokens = torch.randint(0, VOCAB, (2, 8))
    with torch.no_grad():
        logits, _ = brain(tokens, grow=False)
    loss = float(F.cross_entropy(logits.reshape(-1, VOCAB), tokens.reshape(-1)))
    uniform = math.log(VOCAB)

    assert loss < uniform * 1.01, f"initial loss {loss:.4f} > ln(V) {uniform:.4f} + 1 %"
    assert loss > uniform * 0.99, f"initial loss {loss:.4f} is below uniform: a dead readout"


def test_node_counters_close() -> None:
    """alive == slots - deaths + revivals, with the three terms measured."""
    torch.manual_seed(0)
    brain = Brain(n_nodes=16, max_hops=2, max_births=0, max_edges=256)
    # Only 0 -> 2..7 -> 1 exists: nodes 8..15 are isolated from the first beat.
    dead = kill_isolated_nodes(brain, step=0, grace=0)
    assert dead, "the seed wiring leaves nodes 8..15 isolated"
    alive_after_death = int(brain.node_alive.sum())
    assert alive_after_death == brain.n_nodes - brain.node_deaths

    revived = [node_id for node_id in dead[:2] if revive_node(brain, node_id)]
    assert len(revived) == len(dead[:2])
    assert int(brain.node_alive.sum()) == (
        brain.n_nodes - brain.node_deaths + brain.node_revivals
    )
