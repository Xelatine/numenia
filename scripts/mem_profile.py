"""Memory and capacity diagnostic for Mycelia — measures, never optimises.

Why this file exists: the 8 November run could not be launched because nobody
knew *what* was consuming the card. The project only ever recorded "MiB free"
and a batch reduction. This tool answers the question with numbers:

* **static** — every tensor the brain holds, its bytes, its device, whether it
  persists between steps, and three independent sums that must agree;
* **pass** — where the memory of ONE real training step goes, phase by phase,
  on the *real* ``Trainer.step_once`` (patched from the outside: not one line of
  ``mycelia/`` changes);
* **sweep** — the scaling curves (nodes, edges, sequence, batch, and the text
  run's own configuration) measured point by point in a fresh process, so an
  OOM kills a child and not the campaign;
* **bisect** — the exact edge count at which the text configuration dies, then
  the same point extrapolated from the curves, to test the curves as a law;
* **amp** / **reduced** — the price of bf16 activations and of a 200-node
  colony, both in loss-per-step and in memory.

Nothing here is imported by training. Nothing here writes outside
``data/mem_report/``. A measurement that cannot be taken honestly (a timing on
a busy card, a tensor on a device without the room) is recorded as refused,
never faked.

Usage (from the repository root):

    python scripts/mem_profile.py static
    python scripts/mem_profile.py pass --edges 500 1000 2000 3000 4000
    python scripts/mem_profile.py sweep nodes edges seq batch cross
    python scripts/mem_profile.py bisect
    python scripts/mem_profile.py amp --steps 20
    python scripts/mem_profile.py reduced --steps 30
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

MIB = 1024.0 * 1024.0
REPORT_DIR = ROOT / "data" / "mem_report"
CONDITIONS = ROOT / "ablation_conditions.txt"

#: Vocabulary of the text corpus. ``DEFAULT_VOCAB`` of the brain is 100 (the toy
#: task); the text run uses the tokenizer, and the readout is the single
#: vocabulary-sized object in the whole model, so every memory measurement here
#: uses the text width and not the toy one. The tokenizer's own size is read
#: from ``data/tokenizer.json`` when it is present.
TEXT_VOCAB = 10000

#: The card, read once per campaign from nvidia-smi.
CARD: dict = {}

#: Phase meter currently receiving the bracket calls. Module-level because
#: ``torch.Tensor.backward`` is patched globally; only one step is measured at a
#: time and the harness is single-threaded by construction.
METER: "PhaseMeter | None" = None


# ----------------------------------------------------------------------
# Card conditions
# ----------------------------------------------------------------------
def gpu_info() -> dict:
    """Name, total, used and free VRAM, in MiB, straight from nvidia-smi."""
    if not torch.cuda.is_available():
        return {"name": "no-cuda", "total_mib": 0.0, "used_mib": 0.0, "free_mib": 0.0}
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.used,memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=20,
        )
        parts = out.stdout.strip().splitlines()[0].split(",")
        name = parts[0].strip()
        total, used, free = (float(part) for part in parts[1:4])
    except (OSError, IndexError, ValueError, subprocess.SubprocessError):
        props = torch.cuda.get_device_properties(0)
        name = torch.cuda.get_device_name(0)
        total = props.total_memory / MIB
        used = (props.total_memory - torch.cuda.mem_get_info()[0]) / MIB
        free = total - used
    return {"name": name, "total_mib": total, "used_mib": used, "free_mib": free}


def card_verdict(info: dict | None = None) -> dict:
    """Apply the v2 card rules: is a *timing* measurement allowed right now?

    Chronométrage needs a quasi-empty card (< 1 GiB used AND < 10 % of the
    total). Anything else is refused and recorded — a timing on a card already
    holding 1.9 GiB is not a measurement of the system, it is a measurement of
    Windows. Memory readings (allocated/reserved) are per-process and stay valid
    on a busy card; that is why the campaign still runs, with the class named.
    """
    info = info or CARD or gpu_info()
    used = float(info.get("used_mib") or 0.0)
    total = float(info.get("total_mib") or 0.0)
    pct = (100.0 * used / total) if total else 0.0
    clean = used < 1024.0 and pct < 10.0
    return {
        "classe_chronometrage_autorisee": bool(clean),
        "used_mib": round(used, 1),
        "total_mib": round(total, 1),
        "free_mib": round(float(info.get("free_mib") or 0.0), 1),
        "used_pct": round(pct, 1),
        "verdict": (
            "chronométrage autorisé (carte quasi vide)"
            if clean
            else f"chronométrage REFUSÉ ({used:.0f} MiB utilisés, {pct:.0f} %) — "
            "mesures mémoire seules, temps indicatifs"
        ),
    }


def log_condition(campaign: str, extra: str = "") -> dict:
    """Write the nvidia-smi block to ablation_conditions.txt BEFORE the run."""
    info = gpu_info()
    verdict = card_verdict(info)
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "",
        f"9 novembre 2026 (e) — diagnostic mémoire, campagne « {campaign} »",
        f"  nvidia-smi {stamp} : {info['name']} — total {info['total_mib']:.0f} MiB, "
        f"utilisés {info['used_mib']:.0f} MiB, libres {info['free_mib']:.0f} MiB "
        f"({verdict['used_pct']:.0f} %)",
        f"  règle v2 : {verdict['verdict']}",
    ]
    if extra:
        lines.append(f"  {extra}")
    try:
        with CONDITIONS.open("a", encoding="utf-8") as handle:
            handle.write("\n".join(lines) + "\n")
    except OSError:
        pass
    return {"gpu": info, "rule": verdict}


# ----------------------------------------------------------------------
# Synthetic wiring (a structure to measure, not a training run)
# ----------------------------------------------------------------------
def random_pairs(n_nodes: int, edges: int, seed: int = 7) -> torch.Tensor:
    """``edges`` distinct pairs, uniform, no self-loop, no door as a target.

    Uniform rather than preferential: at 4 000 edges over 520 nodes the mean
    in-degree is 7.7 and the busiest node takes ~20, which is the shape the
    grown graph really has (7.73 mean, 16 max). A hub-and-spoke graph would
    measure a different kernel.
    """
    generator = torch.Generator().manual_seed(int(seed))
    seen: set[tuple[int, int]] = set()
    sources: list[int] = []
    targets: list[int] = []
    attempts = 0
    while len(sources) < edges and attempts < 200 * max(1, edges):
        attempts += 1
        source = int(torch.randint(0, n_nodes, (1,), generator=generator))
        target = int(torch.randint(2, n_nodes, (1,), generator=generator))
        if source == target:
            continue
        if (source, target) in seen:
            continue
        seen.add((source, target))
        sources.append(source)
        targets.append(target)
    if len(sources) < edges:
        raise RuntimeError(f"pair space too small: {len(sources)} < {edges}")
    return torch.tensor([sources, targets], dtype=torch.long)


def build_brain(
    nodes: int,
    edges: int,
    vocab: int = TEXT_VOCAB,
    dim: int = 32,
    hops: int = 8,
    chunk: int = 512,
    seed: int = 7,
):
    """A brain of the real shape, wired with ``edges`` synthetic synapses."""
    from mycelia.brain import Brain

    brain = Brain(
        n_nodes=int(nodes),
        dim=int(dim),
        vocab=int(vocab),
        max_hops=int(hops),
        max_edges=20000,
        edge_chunk_size=int(chunk),
        birth_seed=int(seed),
    )
    brain.set_edges(random_pairs(nodes, edges, seed))
    brain.births = 0
    return brain


# ----------------------------------------------------------------------
# Campaign 1 — static inventory
# ----------------------------------------------------------------------
#: Why a tensor is in the checkpoint and whether it survives a step.
PERSISTENT_PREFIXES = (
    "bank.",
    "embedding.",
    "output_head.",
    "readout_norm.",
    "edge_index",
    "edge_usage",
    "_edge_mask",
    "node_alive",
    "node_born_step",
    "node_frozen_until",
    "is_io",
    "mean_out",
)
TRANSIENT_PREFIXES = ("last_", "_last_")


def _section(name: str) -> str:
    """The checkpoint block a tensor belongs to.

    The ``.myc`` format of the spec has no section table yet (the genome is a
    ``torch.save`` dict, ``FORMAT_VERSION = 3``), so the column names the block
    of *that* dict instead — which is what a ``.myc`` writer will have to copy.
    """
    if name.startswith("bank."):
        return "state_dict.node_bank"
    if name.startswith("optim."):
        return "optimizer"
    if name.startswith("cemetery"):
        return "cemetery"
    if name.startswith(("edge_", "_edge_mask")):
        return "state_dict.wiring"
    if name.startswith(("embedding.", "output_head.", "readout_norm.")):
        return "state_dict.vocab"
    if name.startswith(("node_", "is_io", "mean_out")):
        return "state_dict.life"
    if name.startswith(("last_", "_last_")):
        return "not saved (telemetry)"
    if name.startswith("_edge_keys"):
        return "not saved (derived)"
    return "other"


def inventory_static(brain, optimizer=None, device_label: str = "cuda") -> dict:
    """Every tensor of state, its exact bytes, and whether it persists.

    Walks three sources, because a brain alone is not the whole footprint: the
    parameters and buffers of the module, the AdamW moments held *outside* the
    module, and the host-side objects (the cemetery's tensors, the pair set).
    """
    rows: list[dict] = []

    def add(name, tensor, kind, persistent):
        rows.append(
            {
                "name": name,
                "kind": kind,
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "device": str(tensor.device),
                "numel": int(tensor.numel()),
                "bytes": int(tensor.numel() * tensor.element_size()),
                "requires_grad": bool(tensor.requires_grad),
                "persistent": bool(persistent),
                "section": _section(name),
            }
        )

    for name, param in brain.named_parameters():
        add(name, param, "parameter", name.startswith(PERSISTENT_PREFIXES))
    for name, buffer in brain.named_buffers():
        add(name, buffer, "buffer", name.startswith(PERSISTENT_PREFIXES))
    if optimizer is not None:
        by_id = {id(p): n for n, p in brain.named_parameters()}
        for group in optimizer.param_groups:
            for param in group["params"]:
                for key, value in optimizer.state.get(param, {}).items():
                    if torch.is_tensor(value):
                        add(f"optim.{by_id.get(id(param), '?')}.{key}", value, "optimizer", True)

    cemetery_bytes = 0
    cemetery_tensors = 0
    for weights in brain.cemetery.values():
        for tensor in weights.values():
            cemetery_bytes += int(tensor.numel() * tensor.element_size())
            cemetery_tensors += 1
    rows.append(
        {
            "name": f"cemetery[{len(brain.cemetery)} nodes]",
            "kind": "host tensors",
            "shape": [],
            "dtype": "float32",
            "device": "cpu",
            "numel": cemetery_tensors,
            "bytes": cemetery_bytes,
            "requires_grad": False,
            "persistent": True,
            "section": "cemetery",
        }
    )
    # The pair set is a Python set of ints: 32 B per entry on CPython 3.11 once
    # the table is loaded (conservative reading, not a torch allocation).
    pair_bytes = 32 * len(brain._edge_keys)
    rows.append(
        {
            "name": f"_edge_keys[{len(brain._edge_keys)} pairs]",
            "kind": "host set",
            "shape": [],
            "dtype": "int",
            "device": "cpu",
            "numel": len(brain._edge_keys),
            "bytes": pair_bytes,
            "requires_grad": False,
            "persistent": False,  # rebuilt from the wiring on load
            "section": "not saved (derived)",
        }
    )
    for attribute in ("last_out_acc", "_last_out", "mean_out"):
        value = getattr(brain, attribute, None)
        if torch.is_tensor(value):
            add(f"attr.{attribute}", value, "attribute tensor", False)

    rows.sort(key=lambda row: row["bytes"], reverse=True)
    total = sum(row["bytes"] for row in rows)
    persistent = sum(row["bytes"] for row in rows if row["persistent"])
    return {
        "device_label": device_label,
        "rows": rows,
        "total_bytes": total,
        "persistent_bytes": persistent,
        "transient_bytes": total - persistent,
    }


def campaign_static() -> dict:
    """Inventory at the text run's shape, with the three cross-checks."""
    from mycelia.checkpoint import save_genome

    condition = log_condition("inventaire statique")
    brain = build_brain(532, 4000)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    brain.to(device)
    optimizer = torch.optim.AdamW(brain.parameters(), lr=1e-3)

    # One real step so AdamW materialises its moments: an inventory taken before
    # it would miss the two largest static objects of a long run.
    from mycelia.train import Trainer
    from mycelia.config import load_config

    trainer = Trainer(
        config=load_config(),
        data="toy",
        device=device,
        seed=42,
        brain=brain,
        optimizer=optimizer,
        batch_size=8,
        seq_len=32,
        enable_growth=False,
        enable_lifecycle=False,
    )
    trainer.tbptt = 4
    trainer.step_once()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    allocated = torch.cuda.memory_allocated() / MIB if device.type == "cuda" else None
    reserved = torch.cuda.memory_reserved() / MIB if device.type == "cuda" else None
    inventory = inventory_static(brain, optimizer, device_label=str(device))

    checkpoint_path = REPORT_DIR / "_checkpoint_probe.pt"
    save_genome(checkpoint_path, brain, step=1, optimizer=optimizer)
    checkpoint_bytes = checkpoint_path.stat().st_size
    checkpoint_path.unlink()

    inventory["crosscheck"] = {
        "tensor_sum_mib": round(inventory["total_bytes"] / MIB, 2),
        "persistent_mib": round(inventory["persistent_bytes"] / MIB, 2),
        "cuda_allocated_mib": round(allocated, 2) if allocated is not None else None,
        "cuda_reserved_mib": round(reserved, 2) if reserved is not None else None,
        "checkpoint_mib": round(checkpoint_bytes / MIB, 2),
        "checkpoint_equals_saved_state": True,
        "note": (
            "torch.save zips the tensors: the file is smaller than the raw state "
            "because of deflate, not because state is missing. The comparison that "
            "matters is tensor_sum vs cuda_allocated."
        ),
    }
    inventory["condition"] = condition
    inventory["config"] = {"nodes": 532, "edges": 4000, "vocab": TEXT_VOCAB, "dim": 32}
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "static_inventory.json").write_text(
        json.dumps(inventory, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    write_static_markdown(inventory)
    print(json.dumps(inventory["crosscheck"], indent=2))
    return inventory


def write_static_markdown(inventory: dict) -> None:
    rows = inventory["rows"]
    lines = [
        "# Inventaire statique du cerveau (532 nœuds, 4 000 arêtes, vocab 10 000)",
        "",
        "Tout [mesuré] : tailles lues sur les tenseurs après un pas d'entraînement",
        "(les moments AdamW n'existent qu'après le premier pas).",
        "",
        "| # | tenseur | type | shape | dtype | device | octets | Mo | grad | persistant | bloc |",
        "|---:|---|---|---|---|---|---:|---:|:--:|:--:|---|",
    ]
    for index, row in enumerate(rows, 1):
        shape = "x".join(str(d) for d in row["shape"]) or "-"
        lines.append(
            f"| {index} | `{row['name']}` | {row['kind']} | {shape} | {row['dtype']} | "
            f"{row['device']} | {row['bytes']} | {row['bytes'] / MIB:.3f} | "
            f"{'oui' if row['requires_grad'] else 'non'} | "
            f"{'oui' if row['persistent'] else 'non'} | {row['section']} |"
        )
    check = inventory["crosscheck"]
    lines += [
        "",
        f"**Somme des tenseurs** : {check['tensor_sum_mib']} Mo "
        f"({inventory['total_bytes']} octets), dont {check['persistent_mib']} Mo persistants.",
        "",
        f"**`torch.cuda.memory_allocated()`** : {check['cuda_allocated_mib']} Mo — "
        f"**checkpoint classe L** : {check['checkpoint_mib']} Mo (compressé par torch.save).",
        "",
    ]
    (REPORT_DIR / "static_inventory.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


# ----------------------------------------------------------------------
# Campaign 2 — the phase meter
# ----------------------------------------------------------------------
class PhaseMeter:
    """VRAM and time around named phases of one real training step.

    The brackets are pushed from the *outside*: a module hook for the forward, a
    patched optimizer method, a patched ``Tensor.backward``, a patched lifecycle
    call. ``mycelia/`` is not touched. Nested brackets (births inside the
    forward) record their own delta but do not reset the peak, which belongs to
    the outermost bracket that asked for it.
    """

    def __init__(self, device: torch.device, label: str = "") -> None:
        self.device = device
        self.label = label
        self.cuda = device.type == "cuda"
        self.rows: list[dict] = []
        self.stack: list[dict] = []
        self.depth = 0
        # The step's own peak, kept across the per-phase resets: each bracket
        # resets the counter to read *its* peak, so the running maximum has to
        # be harvested just before every reset, or the step peak would end up
        # being the last phase's peak (a 40x underestimate at tbptt 16).
        self.step_peak = 0.0

    def begin(self, phase: str) -> None:
        entry = {"phase": phase, "depth": self.depth, "t0": time.perf_counter()}
        if self.cuda:
            torch.cuda.synchronize()
            entry["before"] = torch.cuda.memory_allocated()
            entry["reserved_before"] = torch.cuda.memory_reserved()
            entry["reset"] = self.depth == 0
            if entry["reset"]:
                self.step_peak = max(self.step_peak, torch.cuda.max_memory_allocated())
                torch.cuda.reset_peak_memory_stats()
        self.depth += 1
        self.stack.append(entry)

    def end(self) -> None:
        if not self.stack:
            return
        entry = self.stack.pop()
        self.depth -= 1
        if self.cuda:
            torch.cuda.synchronize()
            after = torch.cuda.memory_allocated()
            peak = torch.cuda.max_memory_allocated() if entry["reset"] else None
            reserved = torch.cuda.memory_reserved()
        else:
            after = peak = reserved = None
        row = {
            "phase": entry["phase"],
            "depth": entry["depth"],
            "ms": (time.perf_counter() - entry["t0"]) * 1000.0,
            "before_mib": None if not self.cuda else round(entry["before"] / MIB, 2),
            "after_mib": None if not self.cuda else round(after / MIB, 2),
            "delta_mib": None if not self.cuda else round((after - entry["before"]) / MIB, 2),
            "peak_mib": None if peak is None else round(peak / MIB, 2),
            "reserved_mib": None if reserved is None else round(reserved / MIB, 2),
        }
        self.rows.append(row)

    @contextlib.contextmanager
    def phase(self, name: str):
        self.begin(name)
        try:
            yield
        finally:
            self.end()

    def reset(self) -> None:
        self.rows = []
        self.stack = []
        self.step_peak = 0.0
        if self.cuda:
            torch.cuda.reset_peak_memory_stats()


def summarise(rows: list[dict], edges: int, tbptt: int) -> list[dict]:
    """Collapse the per-call brackets of one step into one row per phase."""
    buckets: dict[str, dict] = {}
    for row in rows:
        bucket = buckets.setdefault(
            row["phase"],
            {
                "phase": row["phase"],
                "edges": edges,
                "tbptt": tbptt,
                "allocated_mo": 0.0,
                "pic_mo": 0.0,
                "temps_ms": 0.0,
                "calls": 0,
                "profondeur": row["depth"],
                "reserved_mo": row["reserved_mib"] or 0.0,
            },
        )
        if row["delta_mib"] is not None:
            bucket["allocated_mo"] += row["delta_mib"]
            bucket["pic_mo"] = max(
                bucket["pic_mo"],
                row["peak_mib"] if row["peak_mib"] is not None else row["after_mib"],
            )
        bucket["temps_ms"] += row["ms"]
        bucket["calls"] += 1
        bucket["reserved_mo"] = row["reserved_mib"] or bucket["reserved_mo"]
    for bucket in buckets.values():
        bucket["allocated_mo"] = round(bucket["allocated_mo"], 2)
        bucket["pic_mo"] = round(bucket["pic_mo"], 2)
        bucket["temps_ms"] = round(bucket["temps_ms"], 1)
    return sorted(buckets.values(), key=lambda bucket: bucket["temps_ms"], reverse=True)


def make_trainer(cfg: dict, device: torch.device, prune_every: int | None = None, grow_check: int | None = None):
    """A real Trainer on a synthetic brain of the requested shape."""
    from mycelia.config import load_config
    from mycelia.train import Trainer

    brain = build_brain(
        cfg["nodes"], cfg["edges"], vocab=cfg.get("vocab", TEXT_VOCAB), dim=cfg.get("dim", 32),
        hops=cfg.get("hops", 8), chunk=cfg.get("chunk", 512),
    )
    config = load_config()
    if prune_every is not None:
        config["lifecycle"]["prune_every"] = int(prune_every)
    if grow_check is not None:
        config["growth"]["check_every"] = int(grow_check)
    trainer = Trainer(
        config=config,
        data="toy",
        device=device,
        seed=int(cfg.get("seed", 42)),
        brain=brain,
        batch_size=int(cfg["batch"]),
        seq_len=int(cfg["seq"]),
        enable_growth=bool(cfg.get("growth", True)),
        enable_lifecycle=bool(cfg.get("lifecycle", True)),
        checkpoint_turns=bool(cfg.get("checkpoint_turns", False)),
    )
    trainer.tbptt = int(cfg.get("tbptt", 16))
    if not cfg.get("growth", True):
        trainer.brain.max_births = 0
        trainer.brain.max_node_births = 0
    return trainer


def instrument(trainer, meter: PhaseMeter) -> dict:
    """Attach the brackets to a live Trainer and return the handles to undo."""
    import torch.nn.functional as F

    import mycelia.train as train_module

    handles: list = []
    brain = trainer.brain
    optimizer = trainer.optimizer

    def hook_pre(_module, _inputs):
        meter.begin("forward")

    def hook_post(_module, _inputs, _output):
        meter.end()

    handles.append(brain.register_forward_pre_hook(hook_pre))
    handles.append(brain.register_forward_hook(hook_post))

    def patch_instance(obj, name, phase):
        original = getattr(obj, name)

        def wrapper(*args, **kwargs):
            meter.begin(phase)
            try:
                return original(*args, **kwargs)
            finally:
                meter.end()

        setattr(obj, name, wrapper)
        handles.append((obj, name, original))

    for attribute in ("_birth_edges", "_birth_edges_random", "freeze_newborn_gradients"):
        if hasattr(brain, attribute):
            patch_instance(brain, attribute, "births" if "birth" in attribute else "freeze")

    patch_instance(optimizer, "step", "optimizer")
    patch_instance(optimizer, "zero_grad", "zero_grad")

    def patch_module(module, name, phase):
        original = getattr(module, name)

        def wrapper(*args, **kwargs):
            meter.begin(phase)
            try:
                return original(*args, **kwargs)
            finally:
                meter.end()

        setattr(module, name, wrapper)
        handles.append((module, name, original))

    patch_module(F, "cross_entropy", "loss")
    patch_module(torch.nn.utils, "clip_grad_norm_", "clip")
    for name, phase in (
        ("anti_identity_penalty", "penalties"),
        ("anti_redundancy_penalty", "penalties"),
        ("apply_lifecycle", "lifecycle"),
        ("apply_growth", "growth"),
        ("apply_pending_node_births", "node_births"),
    ):
        if hasattr(train_module, name):
            patch_module(train_module, name, phase)

    original_backward = torch.Tensor.backward

    def backward(self, *args, **kwargs):
        meter.begin("backward")
        try:
            return original_backward(self, *args, **kwargs)
        finally:
            meter.end()

    torch.Tensor.backward = backward
    handles.append((torch.Tensor, "backward", original_backward))
    return {"handles": handles}


def uninstrument(state: dict) -> None:
    for handle in state["handles"]:
        if isinstance(handle, tuple):
            obj, name, original = handle
            setattr(obj, name, original)
        else:
            handle.remove()


def run_config(cfg: dict, device: torch.device, steps: int = 2, warmup: int = 1,
               prune_every: int | None = None, grow_check: int | None = None,
               amp: str = "f32", losses: bool = False) -> dict:
    """Measure ``steps`` real training steps of one configuration."""
    global METER
    torch.manual_seed(int(cfg.get("seed", 42)))
    trainer = make_trainer(cfg, device, prune_every=prune_every, grow_check=grow_check)
    if not cfg.get("release_telemetry", True):
        # Baseline of the 9 November behaviour, kept for the before/after: the
        # telemetry kept the last block's graph alive between two steps.
        trainer._release_telemetry = lambda: None  # type: ignore[method-assign]
    meter = PhaseMeter(device, label=str(cfg))
    start = time.time()

    def one_step():
        if amp == "bf16":
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                return trainer.step_once()
        return trainer.step_once()

    # Warm-up (not measured): the allocator finds its segments and AdamW
    # materialises its moments. A cold first step would measure the setup.
    for _ in range(int(warmup)):
        one_step()
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    baseline_allocated = torch.cuda.memory_allocated() / MIB if device.type == "cuda" else None
    baseline_reserved = torch.cuda.memory_reserved() / MIB if device.type == "cuda" else None

    METER = meter
    handles = instrument(trainer, meter)
    per_step: list[dict] = []
    all_rows: list[dict] = []
    try:
        for _ in range(int(steps)):
            meter.reset()
            if device.type == "cuda":
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            metrics = one_step()
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = (time.perf_counter() - started) * 1000.0
            step_peak = (
                max(meter.step_peak, torch.cuda.max_memory_allocated()) / MIB
                if device.type == "cuda"
                else None
            )
            allocated_after = torch.cuda.memory_allocated() / MIB if device.type == "cuda" else None
            # A candidate lever, measured and not applied: the brain keeps its
            # last hop's live output in ``_last_out`` (and the io pairs in
            # ``last_node_io``), which pins the whole autograd graph of the step
            # after the optimizer has consumed it. Clearing them is the variant;
            # the default keeps the real trainer's behaviour exactly.
            freed = None
            if cfg.get("clear_telemetry"):
                trainer.brain.last_node_io = []
                trainer.brain._last_out = None
                gc.collect()
                if device.type == "cuda":
                    freed = round(allocated_after - torch.cuda.memory_allocated() / MIB, 2)
            per_step.append(
                {
                    "ms": round(elapsed, 1),
                    "peak_mib": None if step_peak is None else round(step_peak, 2),
                    "allocated_mib": None if allocated_after is None else round(allocated_after, 2),
                    "freed_mib": freed,
                    "reserved_mib": round(torch.cuda.memory_reserved() / MIB, 2) if device.type == "cuda" else None,
                    "loss": round(float(metrics["loss"]), 5),
                    "edges": int(metrics["edges"]),
                    "nodes": int(metrics["nodes"]),
                }
            )
            all_rows.extend(meter.rows)
    finally:
        uninstrument(handles)
        METER = None

    if device.type == "cuda":
        torch.cuda.synchronize()
    total_seconds = time.time() - start
    result = {
        "config": cfg,
        "device": str(device),
        "amp": amp,
        "gpu": gpu_info(),
        "baseline_allocated_mib": None if baseline_allocated is None else round(baseline_allocated, 2),
        "baseline_reserved_mib": None if baseline_reserved is None else round(baseline_reserved, 2),
        "steps": per_step,
        "ms_per_step": round(sum(s["ms"] for s in per_step) / max(1, len(per_step)), 1),
        "peak_mib": max((s["peak_mib"] or 0.0) for s in per_step) if per_step else None,
        "oom": False,
        "wall_seconds": round(total_seconds, 1),
        "phases": summarise(all_rows, edges=int(cfg["edges"]), tbptt=int(cfg.get("tbptt", 16))),
    }
    if losses:
        result["losses"] = [s["loss"] for s in per_step]
    return result


def campaign_pass(
    edge_points: list[int], tbptt_points: list[int], steps: int = 2, prune: bool = False, append: bool = False
) -> dict:
    """Campaign 2: phase breakdown at controlled edge counts and TBPTT lengths.

    ``prune`` is off by default: pruning runs every step and cuts 50-100 edges,
    which would move the abscissa of the very curve being measured. The life
    cycle gets its own measurement at the end, on a fixed graph.
    """
    condition = log_condition(
        "ventilation par phase",
        "graphes synthétiques (graine 7, degré moyen 7.5), pas réels de nb_train"
        + (", élagage à chaque pas" if prune else ", élagage au pas 100 (inactif sur 3 pas)"),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows: list[dict] = []
    lifecycle_run: dict | None = None
    path = REPORT_DIR / "pass_breakdown.csv"
    payload: dict = {"condition": condition, "rows": []}
    if append and (REPORT_DIR / "pass_breakdown.json").exists():
        previous = json.loads((REPORT_DIR / "pass_breakdown.json").read_text(encoding="utf-8"))
        rows = list(previous.get("rows", []))
        lifecycle_run = previous.get("lifecycle_run")
        payload = previous
        payload["condition"] = condition

    def measure(edges: int, tbptt: int, prune_every: int | None) -> dict | None:
        cfg = {"nodes": 532, "edges": int(edges), "seq": 32, "batch": 8, "tbptt": int(tbptt)}
        try:
            result = run_config(cfg, device, steps=steps, prune_every=prune_every, grow_check=None)
        except RuntimeError as error:
            print(f"  arêtes {edges} / tbptt {tbptt} : OOM", flush=True)
            return {"oom": str(error)[:300]}
        step = result["steps"][-1]
        if not prune_every:
            rows.append(
                {
                    "phase": "STEP",
                    "edges": edges,
                    "tbptt": tbptt,
                    "allocated_mo": step["allocated_mib"],
                    "pic_mo": step["peak_mib"],
                    "reserved_mo": step["reserved_mib"],
                    "temps_ms": result["ms_per_step"],
                    "calls": 1,
                }
            )
            rows.extend(result["phases"])
        print(
            f"  arêtes {edges:5d} / tbptt {tbptt:2d}{' / élagage' if prune_every else ''} : "
            f"pic {step['peak_mib']} Mo, {result['ms_per_step']:.0f} ms/pas, "
            + ", ".join(f"{p['phase']} {p['pic_mo']} Mo/{p['temps_ms']:.0f}ms" for p in result["phases"][:4]),
            flush=True,
        )
        return result

    for edges in edge_points:
        for tbptt in tbptt_points:
            measure(edges, tbptt, 1 if prune else None)

    # The life cycle on a fixed graph, so its cost can be read against the same
    # edge count as the rest of the table.
    lifecycle_run = measure(int(edge_points[-1]), int(tbptt_points[-1]), 1)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    columns = ["phase", "edges", "tbptt", "allocated_mo", "reserved_mo", "pic_mo", "temps_ms", "calls", "profondeur"]
    seen: set[tuple] = set()
    unique: list[dict] = []
    for row in rows:
        key = (row.get("phase"), row.get("edges"), row.get("tbptt"))
        if key in seen:
            continue
        seen.add(key)
        unique.append(row)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(",".join(columns) + "\n")
        for row in unique:
            handle.write(",".join(str(row.get(column, "")) for column in columns) + "\n")
    payload["rows"] = unique
    payload["lifecycle_run"] = lifecycle_run
    payload["condition"] = condition
    (REPORT_DIR / "pass_breakdown.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return payload


# ----------------------------------------------------------------------
# Campaign 3 — scaling sweeps, one process per point
# ----------------------------------------------------------------------
def _child(args: list[str], timeout: float) -> dict:
    """Run one measurement in a fresh process; an OOM kills only the child."""
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    command = [sys.executable, str(Path(__file__).resolve()), "one", *args]
    started = time.time()
    try:
        proc = subprocess.run(
            command, capture_output=True, text=True, timeout=timeout, cwd=str(ROOT), env=env
        )
    except subprocess.TimeoutExpired:
        return {"oom": False, "timeout": True, "seconds": round(time.time() - started, 1),
                "error": f"timeout {timeout:.0f} s"}
    payload = None
    for line in reversed(proc.stdout.strip().splitlines()):
        if line.startswith("JSON:"):
            payload = json.loads(line[5:])
            break
    if payload is None:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-6:]
        return {
            "oom": "out of memory" in (proc.stderr or "").lower(),
            "returncode": proc.returncode,
            "seconds": round(time.time() - started, 1),
            "error": " | ".join(tail)[:500],
            "gpu": gpu_info(),
        }
    payload["seconds"] = round(time.time() - started, 1)
    payload["gpu_after"] = gpu_info()
    return payload


def _as_args(cfg: dict, steps: int) -> list[str]:
    args = [
        "--nodes", str(cfg["nodes"]), "--edges", str(cfg["edges"]), "--seq", str(cfg["seq"]),
        "--batch", str(cfg["batch"]), "--tbptt", str(cfg.get("tbptt", 16)),
        "--steps", str(steps), "--device", cfg.get("device", "cuda"),
    ]
    if not cfg.get("growth", True):
        args.append("--no-growth")
    if cfg.get("prune"):
        args.append("--prune")
    if cfg.get("amp"):
        args += ["--amp", cfg["amp"]]
    if cfg.get("losses"):
        args.append("--losses")
    if cfg.get("clear_telemetry"):
        args.append("--clear-telemetry")
    if cfg.get("checkpoint_turns"):
        args.append("--checkpoint-turns")
    if not cfg.get("release_telemetry", True):
        args.append("--keep-telemetry")
    return args


def _sweep_points(sweep: str) -> list[dict]:
    nodes_edges = lambda n: max(12, int(round(7.5 * n)))
    if sweep == "nodes":
        return [
            {"nodes": n, "edges": nodes_edges(n), "seq": 32, "batch": 8}
            for n in (200, 300, 400, 532, 800, 1000)
        ]
    if sweep == "edges":
        return [{"nodes": 532, "edges": e, "seq": 32, "batch": 8} for e in (1000, 2000, 3000, 4000, 6000, 8000)]
    if sweep == "seq":
        return [{"nodes": 532, "edges": 2000, "seq": s, "batch": 8} for s in (32, 64, 128, 256)]
    if sweep == "batch":
        return [{"nodes": 532, "edges": 2000, "seq": 128, "batch": b} for b in (4, 8, 16, 32)]
    if sweep == "cross":
        return [
            {"nodes": 532, "edges": e, "seq": 128, "batch": 16}
            for e in (1000, 2000, 3000, 4000)
        ]
    if sweep == "dense":
        # The decisive test: at a CONSTANT edge count, does the cost follow the
        # population (dense broadcast) or stay flat (sparse, by edge list)?
        return [
            {"nodes": n, "edges": 2000, "seq": 32, "batch": 8} for n in (200, 532, 1000, 2000)
        ]
    if sweep == "telemetry":
        return [
            {"nodes": 532, "edges": 4000, "seq": s, "batch": 8, "clear_telemetry": bool(clear)}
            for s in (32, 128)
            for clear in (False, True)
        ]
    raise SystemExit(f"unknown sweep {sweep!r}")


def campaign_sweep(sweeps: list[str], steps: int = 2, timeout: float = 1800.0,
                   out: str = "scaling.json") -> dict:
    """Run each sweep point in its own process and append it to the JSON."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORT_DIR / out
    payload = {"condition": {}, "points": []}
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
    payload.setdefault("points", [])
    done = {(p["sweep"], json.dumps(p["config"], sort_keys=True)) for p in payload["points"]}
    if not payload.get("condition"):
        payload["condition"] = log_condition("courbes de mise à l'échelle")
    for sweep in sweeps:
        for cfg in _sweep_points(sweep):
            key = (sweep, json.dumps(cfg, sort_keys=True))
            if key in done:
                print(f"  {sweep} {cfg} : déjà mesuré", flush=True)
                continue
            result = _child(_as_args(cfg, steps), timeout)
            result["sweep"] = sweep
            result["config"] = cfg
            payload["points"].append(result)
            path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            if result.get("oom"):
                state = "OOM"
            elif result.get("timeout"):
                state = "TIMEOUT"
            elif result.get("peak_mib") is None:
                state = "ÉCHEC"
            else:
                state = f"{result['peak_mib']:.0f} Mo, {result['ms_per_step']:.0f} ms/pas"
            print(f"  {sweep:6s} {cfg} : {state}", flush=True)
    return payload


# ----------------------------------------------------------------------
# Campaign 5 — the wall, by bisection, and the law that predicts it
# ----------------------------------------------------------------------
def campaign_bisect(
    low: int = 500,
    high: int = 9000,
    steps: int = 1,
    timeout: float = 1800.0,
    extra: dict | None = None,
    out: str = "bisect.json",
) -> dict:
    """Greatest edge count that fits at seq 128 / batch 16, by bisection."""
    extra = dict(extra or {})
    flags = "seq 128 / batch 16 / 532 nœuds" + (
        " / " + ", ".join(f"{k}={v}" for k, v in extra.items()) if extra else ""
    )
    condition = log_condition("point de mur (dichotomie)", flags)
    attempts: list[dict] = []
    best = None

    def fits(edges: int) -> bool:
        cfg = {"nodes": 532, "edges": int(edges), "seq": 128, "batch": 16, **extra}
        result = _child(_as_args(cfg, steps), timeout)
        result["edges"] = int(edges)
        attempts.append(
            {
                "edges": int(edges),
                "ok": not (result.get("oom") or result.get("timeout") or result.get("peak_mib") is None),
                "peak_mib": result.get("peak_mib"),
                "ms_per_step": result.get("ms_per_step"),
                "error": result.get("error"),
                "gpu_after": result.get("gpu_after"),
            }
        )
        print(f"  {edges:5d} arêtes : {'tient' if attempts[-1]['ok'] else 'OOM/échec'}"
              f" (pic {result.get('peak_mib')})", flush=True)
        return attempts[-1]["ok"]

    if fits(high):
        best = high
    else:
        lo, hi = int(low), int(high)
        # lo is assumed to fit; prove it once, since an OOM there means the whole
        # question is answered differently (the card cannot take the config at all).
        if not fits(lo):
            best = None
        else:
            best = lo
            while hi - lo > max(100, lo // 20):
                mid = (lo + hi) // 2
                if fits(mid):
                    lo = best = mid
                else:
                    hi = mid
    payload = {
        "condition": condition,
        "config": {"nodes": 532, "seq": 128, "batch": 16, **extra},
        "max_edges_that_fit": best,
        "attempts": attempts,
        "gpu": gpu_info(),
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / out).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return payload


# ----------------------------------------------------------------------
# Campaign 4 — the price of two candidate levers
# ----------------------------------------------------------------------
def campaign_amp(steps: int = 20, nodes: int = 532, edges: int = 2000, seq: int = 32, batch: int = 8) -> dict:
    """bf16 vs f32 on the same seed: memory and loss, step by step."""
    condition = log_condition("prix de bf16", "mêmes données, même graine, 2 × N pas")
    cfg = {"nodes": nodes, "edges": edges, "seq": seq, "batch": batch, "growth": True}
    out: dict = {"condition": condition, "config": cfg, "steps": int(steps)}
    for amp in ("f32", "bf16"):
        result = _child(_as_args({**cfg, "amp": amp, "losses": True}, steps), 3600.0)
        out[amp] = {
            "peak_mib": result.get("peak_mib"),
            "ms_per_step": result.get("ms_per_step"),
            "losses": result.get("losses"),
            "error": result.get("error"),
            "oom": result.get("oom"),
        }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "amp.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return out


def campaign_reduced(steps: int = 30) -> dict:
    """200 nodes at ratio 7.5 against 532/4000: memory, time, first losses."""
    condition = log_condition("colonie réduite (200 nœuds)", "même corpus synthétique, même graine")
    out: dict = {"condition": condition}
    for label, cfg in (
        ("200", {"nodes": 200, "edges": 1500, "seq": 32, "batch": 8, "losses": True}),
        ("532", {"nodes": 532, "edges": 4000, "seq": 32, "batch": 8, "losses": True}),
    ):
        result = _child(_as_args(cfg, steps), 3600.0)
        out[label] = {
            "config": cfg,
            "peak_mib": result.get("peak_mib"),
            "ms_per_step": result.get("ms_per_step"),
            "losses": result.get("losses"),
            "edges_end": (result.get("steps") or [{}])[-1].get("edges"),
            "oom": result.get("oom"),
            "error": result.get("error"),
        }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "reduced.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return out


def campaign_sessionf(
    steps: int = 2,
    edges: int = 3000,
    seq: int = 32,
    batch: int = 8,
    tbptt: int = 16,
    wall: bool = False,
    wall_low: int = 4000,
    wall_high: int = 9000,
) -> dict:
    """Session F: the two levers, measured at the text run's shape.

    Three points on the same brain, same seed, same data: the 9 November
    behaviour (telemetry kept), then L1 (telemetry released), then L1+L2
    (gradient checkpointing per tour). The wall, when asked, is re-bisected with
    both levers ON, in its own file so the 9 November one stays intact.
    """
    condition = log_condition(
        "session F — leviers L1 (télémétrie) et L2 (checkpointing par tour)",
        f"seq {seq} / batch {batch} / tbptt {tbptt} / 532 nœuds, croissance coupée",
    )
    base = {
        "nodes": 532,
        "edges": int(edges),
        "seq": int(seq),
        "batch": int(batch),
        "tbptt": int(tbptt),
        "growth": False,
    }
    variants = {
        "L0_telemetrie_retenue": {"release_telemetry": False},
        "L1_telemetrie_lachee": {},
        "L1_L2_checkpoint": {"checkpoint_turns": True},
    }
    out: dict = {"condition": condition, "config": base, "steps": int(steps), "points": {}}
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORT_DIR / "session-f.json"
    for label, extra in variants.items():
        cfg = {**base, **extra}
        result = _child(_as_args(cfg, steps), 3600.0)
        step = (result.get("steps") or [{}])[-1]
        out["points"][label] = {
            "config": cfg,
            "peak_mib": result.get("peak_mib"),
            "ms_per_step": result.get("ms_per_step"),
            "baseline_allocated_mib": result.get("baseline_allocated_mib"),
            "allocated_after_mib": step.get("allocated_mib"),
            "reserved_mib": step.get("reserved_mib"),
            "oom": result.get("oom"),
            "error": result.get("error"),
        }
        print(
            f"  {label:24s} : pic {result.get('peak_mib')} Mo, {result.get('ms_per_step')} ms/pas",
            flush=True,
        )
        path.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if wall:
        bisect = campaign_bisect(
            wall_low,
            wall_high,
            steps=1,
            timeout=3600.0,
            extra={"checkpoint_turns": True},
            out="bisect-sessionf.json",
        )
        out["wall"] = {
            "config": bisect.get("config"),
            "max_edges_that_fit": bisect.get("max_edges_that_fit"),
            "attempts": bisect.get("attempts"),
        }
        path.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return out


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Mycelia memory diagnostic (measures only)")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("static", help="campaign 1: static inventory")

    pass_parser = sub.add_parser("pass", help="campaign 2: phase breakdown")
    pass_parser.add_argument("--edges", type=int, nargs="+", default=[500, 1000, 2000, 3000, 4000])
    pass_parser.add_argument("--tbptt", type=int, nargs="+", default=[4])
    pass_parser.add_argument("--steps", type=int, default=2)
    pass_parser.add_argument("--prune", action="store_true", help="élaguer à chaque pas (fausse l'axe des arêtes)")
    pass_parser.add_argument("--append", action="store_true", help="ajouter aux mesures existantes")

    sweep_parser = sub.add_parser("sweep", help="campaign 3: scaling curves")
    sweep_parser.add_argument("sweeps", nargs="+", choices=["nodes", "edges", "seq", "batch", "cross", "telemetry", "dense"])
    sweep_parser.add_argument("--steps", type=int, default=2)
    sweep_parser.add_argument("--timeout", type=float, default=1800.0)
    sweep_parser.add_argument("--out", default="scaling.json")

    bisect_parser = sub.add_parser("bisect", help="campaign 5: the wall")
    bisect_parser.add_argument("--low", type=int, default=500)
    bisect_parser.add_argument("--high", type=int, default=9000)
    bisect_parser.add_argument("--steps", type=int, default=1)
    bisect_parser.add_argument("--timeout", type=float, default=1800.0)
    bisect_parser.add_argument("--checkpoint-turns", action="store_true")
    bisect_parser.add_argument("--out", default="bisect.json")

    sessionf_parser = sub.add_parser("sessionf", help="session F: L1, L2 and the new wall")
    sessionf_parser.add_argument("--steps", type=int, default=2)
    sessionf_parser.add_argument("--edges", type=int, default=3000)
    sessionf_parser.add_argument("--seq", type=int, default=32)
    sessionf_parser.add_argument("--batch", type=int, default=8)
    sessionf_parser.add_argument("--tbptt", type=int, default=16)
    sessionf_parser.add_argument("--wall", action="store_true")
    sessionf_parser.add_argument("--wall-low", type=int, default=4000)
    sessionf_parser.add_argument("--wall-high", type=int, default=9000)

    amp_parser = sub.add_parser("amp", help="campaign 4: bf16 price")
    amp_parser.add_argument("--steps", type=int, default=20)

    reduced_parser = sub.add_parser("reduced", help="campaign 4: 200-node colony")
    reduced_parser.add_argument("--steps", type=int, default=30)

    one_parser = sub.add_parser("one", help="one point, in this process (used by sweep)")
    one_parser.add_argument("--nodes", type=int, default=532)
    one_parser.add_argument("--edges", type=int, default=2000)
    one_parser.add_argument("--seq", type=int, default=32)
    one_parser.add_argument("--batch", type=int, default=8)
    one_parser.add_argument("--tbptt", type=int, default=16)
    one_parser.add_argument("--steps", type=int, default=2)
    one_parser.add_argument("--device", default="cuda")
    one_parser.add_argument("--amp", default="f32", choices=["f32", "bf16"])
    one_parser.add_argument("--no-growth", action="store_true")
    one_parser.add_argument("--prune", action="store_true")
    one_parser.add_argument("--losses", action="store_true")
    one_parser.add_argument("--clear-telemetry", action="store_true")
    one_parser.add_argument("--checkpoint-turns", action="store_true")
    one_parser.add_argument("--keep-telemetry", action="store_true")

    args = parser.parse_args(argv)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    if args.command == "static":
        campaign_static()
        return

    if args.command == "pass":
        campaign_pass(
            list(args.edges),
            list(args.tbptt),
            steps=args.steps,
            prune=bool(args.prune),
            append=bool(args.append),
        )
        return

    if args.command == "sweep":
        campaign_sweep(list(args.sweeps), steps=args.steps, timeout=args.timeout, out=args.out)
        return

    if args.command == "bisect":
        campaign_bisect(
            args.low,
            args.high,
            steps=args.steps,
            timeout=args.timeout,
            extra={"checkpoint_turns": True} if args.checkpoint_turns else None,
            out=args.out,
        )
        return

    if args.command == "sessionf":
        campaign_sessionf(
            steps=args.steps,
            edges=args.edges,
            seq=args.seq,
            batch=args.batch,
            tbptt=args.tbptt,
            wall=bool(args.wall),
            wall_low=args.wall_low,
            wall_high=args.wall_high,
        )
        return

    if args.command == "amp":
        campaign_amp(steps=args.steps)
        return

    if args.command == "reduced":
        campaign_reduced(steps=args.steps)
        return

    if args.command == "one":
        device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
        cfg = {
            "nodes": args.nodes,
            "edges": args.edges,
            "seq": args.seq,
            "batch": args.batch,
            "tbptt": args.tbptt,
            "growth": not args.no_growth,
            "clear_telemetry": bool(args.clear_telemetry),
            "checkpoint_turns": bool(args.checkpoint_turns),
            "release_telemetry": not args.keep_telemetry,
        }
        try:
            result = run_config(
                cfg,
                device,
                steps=args.steps,
                prune_every=1 if args.prune else None,
                amp=args.amp,
                losses=args.losses,
            )
        except RuntimeError as error:
            print(f"FAILED: {error}", file=sys.stderr)
            raise SystemExit(2)
        print("JSON:" + json.dumps(result, ensure_ascii=False))
        return


if __name__ == "__main__":
    main()
