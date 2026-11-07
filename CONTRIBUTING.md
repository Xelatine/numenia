# Contributing

This file holds what a contributor (human or agent) must know before touching the
code. The project rules that shape the architecture are in [README.md](README.md);
what follows is process and traps.

## The rules that are not negotiable

1. **Measured ≠ estimated ≠ supposed.** Every claim in a document carries either a
   log line, a code line, or the word "hypothesis". A prediction that turns out
   wrong is corrected in place, visibly — see §10 of
   `docs/DIAGNOSTIC-memoire-05-11.md`.
2. **`test_resume_continuity` stays at 1e-6.** A gap is a signal, not a nuance.
   Never loosen it to make a change pass.
3. **Complete deliveries, never conversational patches.** A fix ships as a dated,
   complete ZIP (code + tests + docs), never as a diff in a message.
4. **Floating-point equivalence uses 1e-6, never `torch.equal`.** Additions are
   not associative; the chunked scatter (D2) reorders them on purpose.
5. **`pid_alive` uses `OpenProcess`, never `os.kill(pid, 0)`** (Windows).
6. Only a local `pytest` counts as evidence. Anything not reproduced on the
   machine is "unverified", including the empty cells of the README.

## The traps, from the real debugging log

These are failures that already cost a run. They are general; the shortest
useful version:

1. **Test `ne`, not the buffer's shape.** A preallocated edge buffer has 20 000
   columns whatever the wiring. Every test asserts on the used count.
2. **Never write in place into a view saved by autograd.** Clone the source and
   target views before a forward can add edges; append through the free slot;
   compact only between two steps.
3. **Retry OOM outside the `except`.** Inside it, the exception still pins the
   graph that failed; release, `gc.collect()`, `empty_cache()`, *then* retry with
   a smaller batch. Never fall back to a dense graph.
4. **Group by node, not by (sample, node).** Copying weights per sample is the
   most expensive mistake of the first version (~12.7 GB). The backward's saved
   activations, not the forward, are the real cost.
5. **Budget births and cool the start.** 8 births per *step*, `init_bias` −8,
   20 000-edge cap, messages only from active sources. Otherwise the targets
   become active, create more targets, and the graph explodes.
6. **No `.item()` in the diffusion loop.** Values stay tensors; readings happen
   after the loop.
7. **Never loosen `test_resume_continuity`.** A 3.3e-6 gap once suggested the
   tolerance was too tight. The cause was AdamW's scalar `step` being rejected by
   a shape check, so no moment was restored. The test caught a real bug; it was
   right to stay at 1e-6.
8. **Parameter initialisation must not consume the training RNG.** A new
   `normal_()` on the global generator shifts every later draw (candidate
   sampling included) and silently changes the trajectory of existing runs and
   tests. Draw from a dedicated `torch.Generator`.
9. **Ship complete fixes, not conversational patches.** The repository is the
   source of truth; the ZIP is complete (code + tests + docs).
10. **Do not send a shortcut or a half-written file.** It will be truncated.

## Running things

```bash
pip install -r requirements.txt
pytest -q                    # collection is limited to tests/ by pytest.ini
python mycelia/train.py      # toy task, 2 000 steps, scale 0
python scripts/bench_step.py --batch 16 --seq 10 --edges 2200 --only step
```

`archive/` holds frozen prototypes and is not collected, not imported and not
shipped in the ZIPs: their `Config` class no longer exists. Read them as history.

## Documentation languages

`README.md` (English) and `docs/README.fr.md` (French) are authoritative. The
six other translations are frozen snapshots: they describe the project as it was
on 5 November 2026 and are not updated. Where they disagree with the English
text, the English text wins.
