"""The third arm of the ablation: same calendar, identities drawn at random.

The grown arm lets a synapse be born where a message actually passed; this arm
must honour the *same* calendar while ignoring the traffic entirely. Two things
have to be true, or the arm measures something else than what it claims:

(a) the calendar is the grown arm's -- the same per-step budget, honoured every
    step, so both graphs end the run the same size;
(b) the identities do not depend on the data -- the same seed must draw the
    same wiring even when the tokens differ, which is exactly what the grown
    arm cannot promise.
"""

from __future__ import annotations

import torch

from mycelia.brain import Brain


def _brain(mode: str, seed: int = 42) -> Brain:
    return Brain(
        n_nodes=64,
        vocab=100,
        max_hops=3,
        max_edges=4096,
        max_births=8,
        max_node_births=0,
        birth_mode=mode,
        birth_seed=seed,
    )


def test_schedule_honours_the_same_calendar() -> None:
    """Exactly ``max_births`` new edges per forward, as in the grown arm."""
    brain = _brain("schedule")
    brain.allow_node_growth = False
    x = torch.randint(2, 99, (2, 8))
    brain.train()
    counts = []
    previous = brain.ne
    for _ in range(10):
        brain(x, grow=True)
        counts.append(brain.ne - previous)
        previous = brain.ne

    assert counts == [8] * 10, counts
    assert brain.births == 80
    assert brain.n_nodes == 64 and brain.node_births == 0


def test_traffic_arm_respects_the_same_budget() -> None:
    """The grown arm spends the same budget; it just occasionally misses a pair.

    The comparison is made on the *budget*, not on the coincidences: a traffic
    birth can land on a pair that already exists, so its own per-step count may
    sit just below 8. What must never happen is a step above it.
    """
    brain = _brain("traffic")
    brain.allow_node_growth = False
    x = torch.randint(2, 99, (2, 8))
    brain.train()
    per_step = []
    previous = brain.ne
    for _ in range(10):
        brain(x, grow=True)
        per_step.append(brain.ne - previous)
        previous = brain.ne

    assert max(per_step) <= 8, per_step
    assert sum(per_step) > 0


def test_identities_are_independent_of_the_traffic() -> None:
    """Same birth seed, different data: the same wiring must come out.

    The last assertion is what gives this test teeth: the grown arm, given the
    same two data sequences, produces *different* wires -- so the equality above
    is a property of the schedule, not of the test being trivial.
    """
    first = torch.randint(2, 99, (2, 8), generator=torch.Generator().manual_seed(1))
    second = torch.randint(2, 99, (2, 8), generator=torch.Generator().manual_seed(2))

    def edges_after(tokens: torch.Tensor, mode: str) -> set[tuple[int, int]]:
        brain = _brain(mode)
        brain.allow_node_growth = False
        brain.train()
        for _ in range(12):
            brain(tokens, grow=True)
        return {tuple(pair) for pair in brain.edge_index[:, : brain.ne].t().tolist()}

    scheduled_a = edges_after(first, "schedule")
    scheduled_b = edges_after(second, "schedule")
    assert scheduled_a == scheduled_b

    traffic_a = edges_after(first, "traffic")
    traffic_b = edges_after(second, "traffic")
    assert traffic_a != traffic_b
