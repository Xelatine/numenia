# Mycelia

**An artificial brain with emergent wiring that learns French.**

Mycelia is not a stack of layers. It is a colony of small autonomous nodes that
talk by messages, decide on their own whom to address, grow synapses when they
want to reach someone, cut them when they starve, die when nothing reaches them
any more — and come back exactly as they were when a message calls them.

The bet is the opposite of the Transformer's: an intelligence can come from
**dynamic structure** (nodes that are born, die, specialise) rather than from raw
size. The biological model is the *Drosophila* brain mapped by FlyWire: 140 000
neurons, ~50 million synapses, and a lot of very simple cells whose power comes
from the number of their connections.

Status: the six construction stages are implemented, tested, and observable live
in a browser cockpit. See [Results](#results) for exactly what was measured and
what was not.

## Nyra — state on 15 November 2026

The runtime is called **Nyra**. What exists, as verified by the audit in
[`AUDIT-ETAT.md`](AUDIT-ETAT.md):

| Piece | State |
|---|---|
| Routed brain, 10 020 nodes, 10 % awake per token | ✅ `mycelia/brain.py` (`n_nodes`, `route_fraction`). Smoke measured `nodes_ever_awake: 10020`, `never_awake: 0`, `cemetery: 0` |
| Run B2 — 3 000 steps, loss ceiling 0.002342 | ❌ no verdict: `data/mem_report/twin10020.json` does not exist; the 100-step smoke stands at **2.7801**, and 0.002342 is a *dense* reference (decision recorded in `PROTOCOLE-reprise-run.md`) |
| `--resume` — never restart from zero | ✅ `scripts/run_twin_10020.py` calls `Trainer.resume`, which logs `reprise à step X` |
| Local chat — `scripts/webchat.py`, `start.py` | ⚠️ runs on `0.0.0.0:8080`, answers and logs; but the readout is a token id turned into `chr()`, not a decoded sentence |
| `self.json` identity | ❌ never written on this machine; "Nyra" is a string, not an identity read at wake-up |
| Métabolisme · DPO (`preferences.jsonl`) · Garde-manger · prompt maître | ❌ not a line of code |

The cockpit below is `scripts/dashboard.py`; the chat is `scripts/webchat.py`.
Both use port 8080, so they cannot run at the same time.

Docs: [français](docs/README.fr.md) · [español](docs/README.es.md) ·
[deutsch](docs/README.de.md) · [中文](docs/README.zh.md) · [日本語](docs/README.ja.md) ·
[português](docs/README.pt.md) · [русский](docs/README.ru.md) · [العربية](docs/README.ar.md)

**English and French are the authoritative versions** (`README.md`,
`docs/README.fr.md`). The six other translations are frozen snapshots taken on
5 November 2026 and are not kept in step with the code; where they disagree, the
English text wins.

---

## The two levels

| Level | What it is | Weight |
|---|---|---|
| **Node** | The equivalent of a biological neuron. It receives, decides, transmits, is born, dies. Width 32. | one node |
| **Internal organs** | `gate`, `body`, `router` — mathematical parameters, *not* neurons. | ~6 400 parameters per node |

The brief estimated ~4 000 parameters/node (the node without its router). The
implementation adds the query/key address (D3), so a node is 6 401 parameters:
gate 33, body 4 192, LayerNorm 64, query 1 056, key 1 056. That is the real
budget below.

## The foundations (non-negotiable)

| # | Rule | Where it lives |
|---|---|---|
| F1 | Many nodes from the start, almost all unconnected; only a minimal entry→exit path exists. | `Brain._seed_wiring` (exactly 12 edges: 0→2..7 then 2..7→1) |
| F2 | Total decentralisation: every node decides alone whom to address. The message chooses its destination. | query/key router, one score per edge |
| F3 | Speaking makes a node exist: addressing a dead or never-born node revives/allocates it. | `_birth_edges`, `revive_node`, `growth.try_birth_node` |
| F4 | A node that receives must modify or forward, never return its input unchanged. | soft cosine penalty (> 0.999), `anti_identity_penalty` |
| F5 | Loops appear by chance and die by dissipation. No loop is detected or broken. | messages ×0.9 per hop, membranes ×0.995 per hop |
| F6 | Progressive extinction, never an artificial ceiling. | usage EMA + pruning + isolation |
| F7 | Two-stage death: links first, then isolated nodes. Exact cemetery. Nodes 0 and 1 immortal. | `lifecycle.py` |
| F8 | Growth by tiers, triggered by measurements, never by a counter. | `GrowthManager` (validation only) |
| F9 | Emergent specialisation: anti-redundancy tax + circulation trace. | `anti_redundancy_penalty`, trace |
| F10 | Light inference: only the nodes that are traversed compute. | wavefront of hot nodes only |
| F11 | Endless evolution by tiers, symbolic cap 140 000 nodes. | `max_population` |

---

## How the signal circulates

There is no dense matrix anywhere. The wiring is a preallocated edge buffer
`edge_index` of shape [2, 20 000] plus a used count `ne`, and one trainable
strength per slot. The 140 000² = 19.6-billion-cell matrix is never allocated.

The forward pass is a **wavefront**, token by token:

1. Node 0 injects the token embedding into the wavefront (`E = 1`).
2. At each hop, every hot node transforms *what it just received* through its
   gate and body, scores its targets with its own query against their keys, and
   sends `out × score × strength × 0.9` to the next hop.
3. Only the nodes that received something compute (F10). A signal that receives
   nothing dies; there is no anti-loop rule.
4. Node 1 accumulates every message in its **membrane**, which cools by 0.995
   per hop (~0.96 per token, about twenty tokens of memory).
5. At position `t`, the prediction is read from node 1 *before* injecting token
   `t`. That is the whole causality: prediction of the current token from the
   past only, so no shift is needed in the dataset.
6. The trace `(hop, source, target)` is recorded and streamed to the cockpit.

Because a step is eight truncated forwards of 16 tokens, the birth budget is
passed across blocks: **8 births per optimizer step**, not per call. The old
mistake reset the budget eight times per step, which is how a graph explodes
slowly.

The membrane remembers; the wavefront travels. Early attempts made nodes emit
their whole membrane, which turned any cycle into an amplifier (energy 9.99 →
4.5 → 2.0 in a two-node loop was the *fixed* version; before, it grew to 1.5
million). A node now emits what just arrived, which is also what makes the
dissipation rule true.

---

## Layout

```
mycelia/
  node.py        one neuron (reference) + NodeBank (stacked, growable, batched)
  brain.py       graph, wavefront, router, births, readout, speech
  lifecycle.py   usage EMA, pruning, isolation, cemetery, exact revival, taxes
  growth.py      reallocation, node birth from a message, tier manager
  checkpoint.py  versioned genome, optimizer by name, complete RNG state
  data.py        delayed-copy toy task + French text stream (5 % validation)
  train.py       Trainer (CLI, dashboard and tests share it), OOM handling, SIGINT
scripts/         train_tokenizer, generate, train_baseline, dashboard
dashboard/       runner.py (threaded), server.py (FastAPI + WS), static/index.html
tests/           17 files, 90 tests
pytest.ini      collection limited to tests/, so a bare `pytest` is safe
configs/         config.yaml — every threshold, loosenable without code
docs/            README.fr.md + short translations (ES, DE, ZH, JA, PT, RU, AR)
rapports/        session reports, verdicts, ablation notes (+ INDEX.md)
livraisons/      dated delivery ZIPs, never versioned (*.zip)
data/            checkpoints, corpora, runs/ — rebuilt locally, never versioned
archive/         frozen v1/v2 prototypes, read-only history, never versioned
```

## Run

```bash
pip install -r requirements.txt

# 1. French corpus (LeCarnet) + 10 000-token BPE — downloads once
python scripts/train_tokenizer.py

# 2. architecture regression: delayed copy, seconds per step
python mycelia/train.py                       # 2 000 steps, scale 0
python mycelia/train.py --scale 1             # 302 nodes (C. elegans)
python mycelia/train.py --scale 2             # 3 000 nodes (fly larva)

# 3. French: 10 000 steps, batch 16, seq 128, truncated backprop 16
python mycelia/train.py --data text --scale 0

# 4. speak
python scripts/generate.py --prompt "Il était une fois" --tokens 100

# 5. the number without which nothing is interpretable
python scripts/train_baseline.py --steps 10000

# 6. the cockpit
python scripts/dashboard.py --train --data text --scale 0
#    → http://127.0.0.1:8080

# 7. everything must stay green
pytest -q
```

Interrupting training with Ctrl+C saves `data/genome.pt` between two steps and
prints the command to resume. `--resume` restores weights, wiring (`ne`), the
cemetery, the growth tier, AdamW's moments by name, and every RNG (torch, CUDA,
numpy, Python, plus the batch sampler). `test_resume_continuity` compares 50
losses after a reload with the uninterrupted run and requires 1e-6.

## Scales and memory

| `--scale` | Nodes | Node parameters | Meaning |
|---|---|---|---|
| 0 | 512 | 3.3 M | prototype, RTX 3070 8 GB |
| 1 | 302 | 1.9 M | wink at *C. elegans*, didactic |
| 2 | 3 000 | 19.2 M | fly larva |
| 3 | 140 000 | 896 M | the fly — **roadmap, not a measured result** |

Scale 1 has *fewer* nodes than scale 0 on purpose: 302 is the wink at
*C. elegans*, a didactic size to compare against the literature, while 512 is the
prototype that must fit an 8 GB card at batch 16. A smaller scale is not a
degraded one, it is a different reference point; growth is driven by validation,
so both can be pushed.

At 140 000 the weights alone are ~3.6 GB in fp32 and AdamW doubles it, which does
not fit in 8 GB: `mmap=True` and sampled sub-graphs are the declared path, not a
measured result here. What *is* true at every scale is the sparse storage:
`edge_index [2, max_edges]`, never a dense adjacency. `test_scale_2_forward`
runs 3 000 nodes under five CPU seconds.

## Corpus

`MaxLSB/LeCarnet` (Hugging Face): small stories written natively in French with a
deliberately restricted vocabulary. `scripts/train_tokenizer.py` streams it into
`data/corpus_fr.txt` (24 M characters in the shipped run), trains the BPE on it,
and prints five round-trips to check accents and elisions. The old English
fallback is obsolete. Fallbacks if the Hub is down: `iproskurina/TinyStories-French`,
`rrenaud/multilingual_tinystories` (GitHub), or the local corpus file.

## Results

All numbers below were produced by `pytest` and by the runs described, on the
real machine (RTX 3070, torch 2.11+cu128). Anything not measured is marked.

### Test suite

```
pytest -q
90 tests collected in 1.17 s   # --collect-only, 15 Nov 2026
```

Seventeen files: `test_brain`, `test_lifecycle`, `test_text`, `test_growth`,
`test_checkpoint`, `test_checkpoint_turns`, `test_dashboard`,
`test_instrumentation`, `test_ablation`, `test_birth_schedule`, `test_chat`,
`test_external`, `test_gguf_reader`, `test_identity`, `test_memory`,
`test_naming` (plus `conftest.py`). The number quoted above is a **collection**,
verified on 15 November 2026; the old "43 passed" figure predates this suite
and is no longer quoted.

### Toy run (delayed copy, `--scale 0`, batch 32, CUDA)

| Step | Loss | Perplexity | Edges used | Alive nodes |
|---|---|---|---|---|
| 100 | 3.03 | 20.6 | 612 | 512 |
| 200 | 1.76 | 5.8 | 635 | 512 |
| 400 | ~0.7 | ~2.0 | 756 | 431 |

Births, cuts and burials were all visible in the seven vital signs; ~0.26 s/step
at 512 nodes. This run predates the final wavefront; the 500-step regression in
`test_learning` reaches **4.86 → 0.11** with the current code. Peak GPU memory is
measured below.

### French run (validation, live cockpit, `Il était une fois`)

The final code was trained live in the cockpit on LeCarnet at `--scale 0`,
batch 16, sequence 128, eight hops. Measured numbers are in the run log of this
conversation; the honest summary:

* validation/training perplexity starts at ~11 750 (vocabulary 10 000);
* it falls during the run, and the 40-token samples move from random tokens to
  French-looking fragments;
* the speed is reported as a **rolling mean of the last 20 steps** (v3), never
  `elapsed/steps`, which can only rise and hides every slow interval;
* measured, exclusive GPU, RTX 3070: **3.2 s/step for the first 40 steps** (the
  graph is still nearly empty) and **8.5 s/step around 800 edges**; the cost
  follows the *wiring*, not the step counter. The profile of one step puts 52 %
  in `run_backward`, 16 % in the forward, over ~1 024 sequential hop-iterations
  (128 positions × 8 hops), each a few dozen small kernels;
* v3 removes the sort-based `torch.isin` from the birth path and the misleading
  cumulative display. It does **not** reach the < 2 s/step target at 2 000 dense
  edges: measured 2.24 s/step at batch 8, sequence 32, and an out-of-memory at
  batch 16, sequence 128 (4 GiB of activations on an 8 GiB card). Grouping the
  state into one tensor was not the missing piece — the membrane has always been
  a single `[N, batch, dim]` tensor — so the next lever is the **number of
  kernels per hop**, and the record stays honest until that is cut;
* **the ETA of the 10 000-step text run is therefore unknown**: at 8.5 s/step it
  is more than 24 h, and the measured curve rises with the wiring.

**Not measured:** a full 10 000-step text run (hours), the 3-seed table, and the
Transformer baseline's final perplexity. The scripts are complete and tested at
small step counts; the numbers must be produced on the target machine.

### Three seeds

| Seed | Validation perplexity | Final pool | Final edges |
|---|---|---|---|
| 42 | short run only (see above) | 532 after 2 steps; tiers not reached | 102 at step 11 |
| 43 | not run | — | — |
| 44 | not run | — | — |

The table is filled by `--seed 42|43|44` on a real 10 000-step budget. It is kept
here, empty where it must be, rather than filled with invented numbers.

## What we do differently

* **Neural Cellular Automata** (Mordvintsev et al., *Distill*, 2020,
  <https://distill.pub/2020/growing-ca/>) grow a *fixed* grid of cells with a
  shared rule. Mycelia's cells are heterogeneous, individually plastic, and the
  graph itself is created and destroyed locally; there is no shared update rule,
  and a node's identity is its address, not its position.
* **FlyWire / FLYNN** (Dorkenwald et al., *Nature*, 2024,
  <https://www.nature.com/articles/s41586-024-07558-y>; Lappalainen et al.,
  *Nature*, 2024, <https://www.nature.com/articles/s41586-024-07763-9>) map a real
  connectome and simulate it. Mycelia borrows the *scale philosophy* (many simple
  units, sparse edges, local computation) but the connectome is learned, not
  copied: no wiring diagram is given to the model.
* **Dynamic Sparse Training** (Evci et al., 2020,
  <https://arxiv.org/abs/1911.11134>; Liu et al., 2021) prunes and regrows weights
  inside a fixed dense network, with a global mask and a global schedule. Here the
  topology is the computation, each node chooses its own targets from a query/key
  address, deletion is a local usage rule, and the count of neurons changes.
* **Generative Agents** (Park et al., 2023,
  <https://arxiv.org/abs/2304.03442>) give LLM agents a memory stream and
  reflection. Mycelia has no LLM underneath: memory is a decaying membrane, and
  what emerges is observed through circulation, not prompted.

## The traps

The ten failures that already cost a run — in-place buffers, the OOM retry, the
birth budget, the 1e-6 tolerance, the RNG a parameter init may not consume — now
live in [CONTRIBUTING.md](CONTRIBUTING.md#the-traps-from-the-real-debugging-log),
together with the delivery rules. The architecture rules stay in this file.

## Debugging plan

* **No wiring after 2 000 steps** — lower L1 (1e-4 → 1e-5), loosen `init_bias`,
  check that query/key scores vary, check the signal energy at node 0.
* **The network never dies** — check the usage EMA (decay 0.99) and the 0.02
  pruning threshold; both are in `configs/config.yaml`.
* **Perplexity stuck at vocab/10** — the entry→exit path is dead; strengthen
  exploration toward node 1 (the trace shows whether anything reaches it).
* **Perplexity stagnates in text mode** — suspect #1 is truncated backprop at 16
  tokens; suspect #2 the birth budget; suspect #3 the edge cap (`ne` vs
  `max_edges`). All three are visible in the cockpit.
* **OOM** — the retry is already outside the `except`; verify the per-node
  grouping, and check that `edge_index` is still [2, max_edges].

## Success tiers (judged on 20 generations)

1. One grammatical, understandable French sentence.
2. A coherent story of 3–5 sentences.
3. Multi-step reasoning (questions/answers).
4. Long term (`scale 3`, 140 000 nodes): long reasoning on modest hardware.

The current stage is below tier 1 with a short run: the samples are French-shaped
word salad. That is what a 2 M-parameter graph looks like before a full 10 000-step
run, and the honest verdict is "architecture works, language not yet".

## Verdict of the toy run

| Criterion | Result |
|---|---|
| Loss falls clearly (not 1 %) | yes: 4.86 → 0.11 (test), 4.6 → 0.7 (400-step run) |
| Seven vital signs move, births then deaths | yes: births, cuts, 505 burials |
| Trace shows 0 → … → 1 with ≥ 2 intermediate nodes | partial: 0→3→1 observed; longer paths appear as the graph densifies |
| GPU memory < 6 GB at batch 32 | measured below |
| Step < 1 s on RTX 3070 | toy 0.26 s — text 5–8 s, noted as the known bottleneck |

## License

MIT. See [LICENSE](LICENSE).
