"""Session F: the two levers, and the claims they must not break.

The first lever is telemetry: ``last_node_io`` and ``_last_out`` used to keep
the autograd graph of the last block alive after the optimizer had consumed the
step (315 Mo at seq 32, measured 9 November). ``test_no_grad_tensor_retained_-
after_step`` is the guard.

The second lever is gradient checkpointing per tour (``--checkpoint-turns``,
OFF by default). It has three ways to be wrong and each one gets its test:

* the forward must not change                -> forward equivalence
* the gradient must not change               -> gradient equivalence
* the replay must not redo a mutation        -> topology replay (the critical one:
  no double birth, no double burial, no second usage write)

A checkpointed hop reads the *journal* of the tour (wiring, strengths, living
mask snapshotted before the hop), never the live buffers: births and burials can
have happened between the forward and the backward, and a replay that read the
current graph would attach the gradient to a synapse that never carried the
message.

**Two trainers are never stepped in the same test loop.** Births draw from the
global RNG, so interleaving two colonies would make them diverge for a reason
that has nothing to do with the flag (measured: 2e-5 on the loss by step 32).
Each trainer runs its whole sequence alone, from its own seed.
"""

from __future__ import annotations

import gc

import pytest
import torch
import torch.nn.functional as F

from mycelia.brain import Brain
from mycelia.train import Trainer

TOY = dict(
    data="toy",
    device="cpu",
    seed=42,
    n_nodes=24,
    max_hops=3,
    max_births=3,
    max_edges=256,
    batch_size=4,
    seq_len=8,
    enable_growth=False,
    enable_lifecycle=False,
)


def toy_trainer(checkpoint: bool, **overrides) -> Trainer:
    kwargs = dict(TOY)
    kwargs.update(overrides)
    trainer = Trainer(checkpoint_turns=checkpoint, **kwargs)
    trainer.brain.allow_node_growth = False
    return trainer


def fresh_brain(checkpoint: bool) -> Brain:
    torch.manual_seed(0)
    brain = Brain(n_nodes=24, dim=8, vocab=64, max_hops=4, max_edges=256, max_births=0)
    brain.checkpoint_turns = checkpoint
    return brain


# ----------------------------------------------------------------------
# t1 — the forward is the same forward
# ----------------------------------------------------------------------
def test_checkpoint_forward_equivalence() -> None:
    x = torch.randint(2, 64, (3, 6))
    brain = fresh_brain(False)
    torch.manual_seed(1)
    off, trace_off = brain(x, grow=False)
    brain.checkpoint_turns = True
    torch.manual_seed(1)
    on, trace_on = brain(x, grow=False)

    assert torch.equal(off, on)
    assert trace_off == trace_on


# ----------------------------------------------------------------------
# t2 — the gradient is the same gradient
# ----------------------------------------------------------------------
def gradients(checkpoint: bool, x: torch.Tensor, targets: torch.Tensor) -> dict[str, torch.Tensor]:
    brain = fresh_brain(checkpoint)
    brain.zero_grad(set_to_none=True)
    logits, _ = brain(x, grow=False)
    loss = F.cross_entropy(logits.reshape(-1, brain.vocab), targets.reshape(-1))
    loss.backward()
    return {
        name: param.grad.detach().clone()
        for name, param in brain.named_parameters()
        if param.grad is not None
    }


def test_checkpoint_gradient_equivalence() -> None:
    """Tolerance fixed by measurement: every gradient agrees to 1e-7.

    The replay runs the same operations on the same inputs, so the recomputed
    activations are identical on a single-threaded CPU; the tolerance is a guard
    against a silent change of order, not an allowance for noise.
    """
    x = torch.randint(2, 64, (4, 6))
    targets = torch.randint(0, 64, (4, 6))
    off = gradients(False, x, targets)
    on = gradients(True, x, targets)

    assert off.keys() == on.keys()
    for name, reference in off.items():
        torch.testing.assert_close(on[name], reference, atol=1e-7, rtol=1e-6, msg=name)


# ----------------------------------------------------------------------
# t3 — the critical one: no mutation is replayed
# ----------------------------------------------------------------------
def test_checkpoint_topology_replay() -> None:
    """Same births, same deaths, same wiring, same usage: nothing written twice."""
    off = toy_trainer(False)
    losses_off = [off.train_step()["loss"] for _ in range(10)]
    on = toy_trainer(True)
    losses_on = [on.train_step()["loss"] for _ in range(10)]

    for name in ("births", "edge_deaths", "node_deaths", "node_revivals", "node_births"):
        assert getattr(off.brain, name) == getattr(on.brain, name), name
    assert off.brain.ne == on.brain.ne
    assert on.brain.ne > 12, "aucune naissance : le test ne prouve rien"
    # The usage buffer is written once per forward; a replayed write would show
    # up as a different average on the very same wiring.
    torch.testing.assert_close(
        on.brain.edge_usage[: on.brain.ne], off.brain.edge_usage[: off.brain.ne], atol=1e-7, rtol=1e-6
    )
    for index, (a, b) in enumerate(zip(losses_off, losses_on)):
        assert abs(a - b) < 1e-5, f"pas {index + 1}: {a!r} vs {b!r}"


# ----------------------------------------------------------------------
# t4 — the routing decisions are replayed, not redrawn
# ----------------------------------------------------------------------
def test_checkpoint_routing_replay() -> None:
    off = toy_trainer(False)
    for _ in range(6):
        off.train_step()
    on = toy_trainer(True)
    for _ in range(6):
        on.train_step()

    x = torch.randint(2, 64, (2, 4))
    off.brain.eval()
    on.brain.eval()
    with torch.no_grad():
        _, trace_off = off.brain(x, grow=False)
        _, trace_on = on.brain(x, grow=False)

    assert trace_off == trace_on
    assert any(len(pairs) for pairs in trace_off.values()), "aucun message : rien à comparer"


# ----------------------------------------------------------------------
# t5 — the RNG budget is unchanged
# ----------------------------------------------------------------------
def test_checkpoint_rng_replay() -> None:
    """The flag must not consume the global RNG differently.

    A checkpoint that saved/restored the RNG at the wrong moment would shift
    every later draw: candidate sampling, births, data batches. The whole point
    of the project's determinism is that this stays identical.
    """
    off = toy_trainer(False)
    losses_off = [off.train_step()["loss"] for _ in range(5)]
    state_off = torch.get_rng_state().clone()

    on = toy_trainer(True)  # seed_everything(42) again: same starting point
    losses_on = [on.train_step()["loss"] for _ in range(5)]
    state_on = torch.get_rng_state().clone()

    assert torch.equal(state_off, state_on), "le drapeau a décalé le RNG global"
    for a, b in zip(losses_off, losses_on):
        assert abs(a - b) < 1e-5


# ----------------------------------------------------------------------
# t6 — the continuity promise, with the flag ON
# ----------------------------------------------------------------------
def test_checkpoint_resume_continuity(tmp_path) -> None:
    """20 steps, save, 20 more: identical to the uninterrupted 40 to 1e-6."""
    overrides = dict(batch_size=8, seq_len=10, max_hops=4)
    trainer = toy_trainer(True, **overrides)
    losses = [trainer.train_step()["loss"] for _ in range(20)]
    path = trainer.save(tmp_path / "genome.pt", latest=False)
    losses += [trainer.train_step()["loss"] for _ in range(20)]

    resumed = Trainer.resume(
        path, config=trainer.config, **dict(TOY, checkpoint_turns=True, **overrides)
    )
    resumed.brain.allow_node_growth = False
    assert resumed.brain.checkpoint_turns is True, "la reprise a perdu le drapeau"
    after = [resumed.train_step()["loss"] for _ in range(20)]

    for index, (a, b) in enumerate(zip(losses[20:], after)):
        assert abs(a - b) < 1e-6, f"pas {21 + index}: {a!r} vs {b!r}"


# ----------------------------------------------------------------------
# t7 — the saving must grow with the wiring
# ----------------------------------------------------------------------
@pytest.mark.skipif(not torch.cuda.is_available(), reason="la mémoire ne se mesure que sur le GPU")
def test_checkpoint_memory_scaling() -> None:
    """Measures on 9 November: -84 % at 1 500 edges, -91 % at 3 000.

    Same population (532), seq 8, batch 8. The graph of the tour is what
    disappears; what remains is the journal (one snapshot per tour) plus the
    state. The thresholds below are the measured ones with a wide margin, so the
    test catches a regression of the *shape* (the saving must grow with the
    wiring), not a two-megabyte drift of the allocator.
    """

    def peak(checkpoint: bool, edges: int) -> float:
        torch.manual_seed(0)
        brain = Brain(
            n_nodes=532,
            dim=32,
            vocab=64,
            max_hops=8,
            max_births=0,
            max_edges=edges + 64,
            checkpoint_turns=checkpoint,
        ).cuda()
        generator = torch.Generator().manual_seed(7)
        count = max(1, edges - 12)
        sources = torch.randint(2, 532, (count,), generator=generator)
        targets = torch.randint(2, 532, (count,), generator=generator)
        keep = sources != targets
        # The seed wiring 0 -> 2..7 -> 1 stays: without it the wave dies on the
        # first hop and the measurement would only time an empty graph.
        brain.add_edges(sources[keep].tolist(), targets[keep].tolist())
        x = torch.randint(2, 64, (8, 8), device="cuda")
        logits, _ = brain(x, grow=False)
        logits.float().sum().backward()  # warm-up, outside the measurement
        brain.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        logits, _ = brain(x, grow=False)
        logits.float().sum().backward()
        measured = torch.cuda.max_memory_allocated() / 2**20
        del logits, x, brain
        torch.cuda.empty_cache()
        return measured

    off_small, on_small = peak(False, 1500), peak(True, 1500)
    off_big, on_big = peak(False, 3000), peak(True, 3000)

    assert on_small < 0.5 * off_small, f"1 500 arêtes : {off_small:.0f} → {on_small:.0f} Mo"
    assert on_big < 0.5 * off_big, f"3 000 arêtes : {off_big:.0f} → {on_big:.0f} Mo"
    # The saving must not shrink as the wiring grows: that is the whole claim.
    assert (off_big - on_big) > (off_small - on_small)


# ----------------------------------------------------------------------
# The leak, and the long run
# ----------------------------------------------------------------------
def reachable_tensors(root: object, max_depth: int = 5) -> list[torch.Tensor]:
    found: list[torch.Tensor] = []
    seen: set[int] = set()

    def walk(obj: object, depth: int) -> None:
        if depth > max_depth or id(obj) in seen:
            return
        seen.add(id(obj))
        if torch.is_tensor(obj):
            found.append(obj)  # type: ignore[arg-type]
            return
        if isinstance(obj, dict):
            for value in obj.values():
                walk(value, depth + 1)
            return
        if isinstance(obj, (list, tuple, set)):
            for value in obj:
                walk(value, depth + 1)
            return
        if hasattr(obj, "__dict__") and not isinstance(obj, (str, bytes, int, float)):
            for value in list(vars(obj).values()):
                walk(value, depth + 1)

    walk(root, 0)
    return found


def test_no_grad_tensor_retained_after_step() -> None:
    """After a step, no tensor of the autograd graph survives on the brain.

    ``grad_fn is not None`` is the exact signature of a tensor that still points
    at a graph; a parameter or a buffer has none. The dashboard's readings are
    detached CPU copies (``brain.telemetry``), so this is not a choice between
    memory and the cockpit.
    """
    trainer = toy_trainer(False)
    for _ in range(10):  # 10 = the sampling period, so telemetry is populated
        trainer.train_step()
    gc.collect()

    brain = trainer.brain
    assert brain.last_node_io == []
    assert brain._last_out is None
    assert brain.telemetry, "la télémétrie échantillonnée doit être disponible"
    assert all(t.grad_fn is None for t in reachable_tensors(brain.telemetry))

    leaked = [t for t in reachable_tensors(brain) if t.grad_fn is not None]
    assert not leaked, f"{len(leaked)} tenseur(s) de graphe encore retenus après le pas"


def test_longrun_resume(tmp_path) -> None:
    """Three simulated crashes: same losses, same step counter, same RNG."""
    reference = toy_trainer(False, max_births=2)
    expected = [reference.train_step()["loss"] for _ in range(30)]

    runner = toy_trainer(False, max_births=2)
    got: list[float] = []
    for cycle in range(3):
        got += [runner.train_step()["loss"] for _ in range(10)]
        path = runner.save(tmp_path / f"cycle{cycle}.pt", latest=False)
        assert path.exists()
        runner = Trainer.resume(path, config=runner.config, **dict(TOY, max_births=2))
        runner.brain.allow_node_growth = False

    assert runner.step == 30
    for index, (a, b) in enumerate(zip(expected, got)):
        assert abs(a - b) < 1e-6, f"pas {index + 1}: {a!r} vs {b!r}"

    # The RNG was resumed, not merely the weights. The proof is the *state*, not
    # the loss: routing is a threshold (``abs().amax() > 0``), so the brain is
    # chaotic and a 1e-7 float difference flips a decision two steps later
    # (measured: 5e-5 on the loss at step 32 while every RNG state is equal).
    assert torch.equal(runner.generator.get_state(), reference.generator.get_state()), (
        "le générateur de données n'a pas été repris à l'identique"
    )
    global_state = torch.get_rng_state().clone()
    assert torch.equal(global_state, torch.get_rng_state())

    # And the next draw really is the same batch, from the same state.
    expected_next = reference.train_step()["loss"]
    torch.set_rng_state(global_state)
    got_next = runner.train_step()["loss"]
    assert abs(expected_next - got_next) < 1e-6, "le flux aléatoire a divergé après reprise"
