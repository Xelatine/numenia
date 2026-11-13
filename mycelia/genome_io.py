"""Format disque ``genome.myc/`` : le cerveau persistant, mmap-ready.

Le format n'est accepté que s'il préserve le programme : un forward sur le
cerveau chargé depuis le ``.pt`` et un forward sur le cerveau rechargé depuis
``genome.myc/`` doivent produire les mêmes logits (tolérance fp16 explicite,
``atol 1e-3``). Une conversion qui change silencieusement le résultat est
rejetée — c'est la règle de la session R.

Contenu du répertoire :

* ``meta.json``      — version, dimensions, config, offsets, md5 par fichier,
                       hash global, date ;
* ``topology.csr``   — indptr int32 [N+1], indices int32 [E], poids fp16 [E] ;
* ``embeddings.bin`` — les organes des nœuds, [N, K] fp32, aligné 4 Ko ;
* ``token_embedding.bin`` — [vocab, dim] fp32 ;
* ``output_head.bin``     — [vocab, dim+1] fp32 (poids + biais) ;
* ``misc.bin``       — readout_norm, mean_out, alive, born_step, usage ;
* ``energy.bin`` / ``salience.bin`` — fp16 [N] (mémoire épisodique) ;
* ``cemetery.bin``   — nœuds morts (torch.save, hors mmap) ;
* ``memories.npz``   — copie de la mémoire deux poids, si fournie.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from mycelia.brain import Brain
from mycelia.node import BANK_SHAPES

FORMAT = "myc-1"
ALIGN = 4096


def md5_file(path: str | Path) -> str:
    digest = hashlib.md5()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _align(handle, align: int = ALIGN) -> int:
    pos = handle.tell()
    pad = (-pos) % align
    if pad:
        handle.write(b"\x00" * pad)
    return handle.tell()


def _write_array(handle, array: np.ndarray) -> dict:
    _align(handle)
    offset = handle.tell()
    handle.write(np.ascontiguousarray(array).tobytes())
    return {"offset": offset, "bytes": int(array.nbytes), "dtype": str(array.dtype), "shape": list(array.shape)}


def _read_array(handle, spec: dict) -> np.ndarray:
    handle.seek(int(spec["offset"]))
    dtype = np.dtype(spec["dtype"])
    count = int(np.prod(spec["shape"])) if spec["shape"] else 1
    return np.frombuffer(handle.read(int(spec["bytes"])), dtype=dtype, count=count).reshape(spec["shape"])


def _bank_layout(dim: int) -> list[tuple[str, tuple[int, ...], int]]:
    """(nom, forme par nœud, nombre de valeurs) dans l'ordre du bloc."""
    layout = []
    for name in BANK_SHAPES:
        shape = {
            "gate_w": (1, dim), "gate_b": (1,),
            "b1_w": (2 * dim, dim), "b1_b": (2 * dim,),
            "b2_w": (dim, 2 * dim), "b2_b": (dim,),
            "ln_w": (dim,), "ln_b": (dim,),
            "q_w": (dim, dim), "q_b": (dim,),
            "k_w": (dim, dim), "k_b": (dim,),
        }[name]
        layout.append((name, shape, int(np.prod(shape))))
    return layout


def save_myc(path: str | Path, brain: Brain, source: str | Path | None = None,
             memories: str | Path | None = None, weight_dtype=np.float32) -> dict:
    path = Path(path)
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)
    n, ne, dim, vocab = int(brain.n_nodes), int(brain.ne), int(brain.dim), int(brain.vocab)
    layout = _bank_layout(dim)

    # Organes des nœuds, concaténés en [N, K] fp32.
    blocks = [getattr(brain.bank, name).detach().cpu().reshape(n, size).numpy() for name, _, size in layout]
    nodes = np.concatenate(blocks, axis=1).astype(np.float32)

    src = brain.edge_index[0, :ne].detach().cpu().numpy().astype(np.int64)
    tgt = brain.edge_index[1, :ne].detach().cpu().numpy().astype(np.int64)
    # fp16 a été mesuré : l'équité forward casse (5,9e-3 > atol 1e-3). Le
    # format garde donc fp32 par défaut ; fp16 reste disponible pour mesurer.
    weights = brain._edge_mask.detach().cpu()[:ne].numpy().astype(weight_dtype)
    order = np.argsort(src, kind="stable")
    src_sorted = src[order]
    indptr = np.zeros(n + 1, dtype=np.int32)
    np.add.at(indptr, src_sorted + 1, 1)
    indptr = np.cumsum(indptr).astype(np.int32)
    csr_tgt = tgt[order].astype(np.int32)
    csr_w = weights[order]

    files: dict[str, dict] = {}
    with (path / "topology.csr").open("wb") as handle:
        files["topology.csr"] = {
            "indptr": _write_array(handle, indptr),
            "indices": _write_array(handle, csr_tgt),
            "weights": _write_array(handle, csr_w),
        }
    with (path / "embeddings.bin").open("wb") as handle:
        spec = _write_array(handle, nodes)
        spec["layout"] = [
            {"name": name, "shape": list(shape), "size": size} for name, shape, size in layout
        ]
        files["embeddings.bin"] = spec
    with (path / "token_embedding.bin").open("wb") as handle:
        files["token_embedding.bin"] = _write_array(handle, brain.embedding.weight.detach().cpu().numpy().astype(np.float32))
    head = np.concatenate(
        [brain.output_head.weight.detach().cpu().numpy().astype(np.float32),
         brain.output_head.bias.detach().cpu().numpy().astype(np.float32)[:, None]],
        axis=1,
    )
    with (path / "output_head.bin").open("wb") as handle:
        files["output_head.bin"] = _write_array(handle, head)
    mean_out = brain.mean_out.detach().cpu().numpy().astype(np.float32)
    misc = np.concatenate([
        brain.readout_norm.weight.detach().cpu().numpy().astype(np.float32),
        brain.readout_norm.bias.detach().cpu().numpy().astype(np.float32),
        mean_out.reshape(-1),
        brain.node_alive.detach().cpu().numpy().astype(np.uint8),
        brain.node_born_step.detach().cpu().numpy().astype(np.int64),
        brain.edge_usage[:ne].detach().cpu().numpy().astype(np.float32),
    ])
    with (path / "misc.bin").open("wb") as handle:
        spec = _write_array(handle, misc)
        spec["mean_out_shape"] = list(mean_out.shape)
        files["misc.bin"] = spec
    zeros = np.zeros(n, dtype=np.float16)
    with (path / "energy.bin").open("wb") as handle:
        files["energy.bin"] = _write_array(handle, zeros)
    with (path / "salience.bin").open("wb") as handle:
        files["salience.bin"] = _write_array(handle, zeros)
    torch.save(brain.cemetery, path / "cemetery.bin")
    if memories is not None and Path(memories).exists():
        shutil.copy2(memories, path / "memories.npz")

    hashes = {name: md5_file(path / name) for name in files}
    hashes["cemetery.bin"] = md5_file(path / "cemetery.bin")
    global_hash = hashlib.md5("".join(sorted(hashes.values())).encode()).hexdigest()
    meta = {
        "format": FORMAT,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "n_nodes": n, "ne": ne, "dim": dim, "vocab": vocab,
        "config": dict(brain._init_kwargs, n_nodes=n),
        "counters": {
            "clock": int(brain.clock), "births": int(brain.births),
            "node_births": int(brain.node_births),
            "edge_deaths": int(brain.edge_deaths), "node_deaths": int(brain.node_deaths),
        },
        "files": files, "md5": hashes, "hash_global": global_hash,
        "source": str(source) if source else None,
        "source_md5": md5_file(source) if source and Path(source).exists() else None,
    }
    (path / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return {"path": str(path), "hash_global": global_hash, "md5": hashes,
            "bytes": sum((path / name).stat().st_size for name in hashes)}


def load_myc(path: str | Path) -> tuple[Brain, dict]:
    path = Path(path)
    meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
    if meta.get("format") != FORMAT:
        raise ValueError(f"format {meta.get('format')!r} inconnu (attendu {FORMAT})")
    for name, expected in meta["md5"].items():
        found = md5_file(path / name)
        if found != expected:
            raise ValueError(f"md5 de {name} : {found} != {expected}")
    brain = Brain(**meta["config"])
    n, ne, dim = int(meta["n_nodes"]), int(meta["ne"]), int(meta["dim"])
    brain.ne = ne
    brain.clock = int(meta["counters"]["clock"])
    brain.births = int(meta["counters"]["births"])
    brain.node_births = int(meta["counters"]["node_births"])
    brain.edge_deaths = int(meta["counters"]["edge_deaths"])
    brain.node_deaths = int(meta["counters"]["node_deaths"])

    with (path / "topology.csr").open("rb") as handle:
        indptr = _read_array(handle, meta["files"]["topology.csr"]["indptr"])
        indices = _read_array(handle, meta["files"]["topology.csr"]["indices"])
        weights = _read_array(handle, meta["files"]["topology.csr"]["weights"])
    src = np.repeat(np.arange(n, dtype=np.int64), np.diff(indptr).astype(np.int64))
    brain.edge_index[0, :ne] = torch.from_numpy(src)
    brain.edge_index[1, :ne] = torch.from_numpy(indices.astype(np.int64))
    brain._edge_mask.data[:ne] = torch.from_numpy(weights.astype(np.float32))
    brain._rebuild_edge_keys()

    with (path / "embeddings.bin").open("rb") as handle:
        nodes = _read_array(handle, meta["files"]["embeddings.bin"])
    col = 0
    for spec in meta["files"]["embeddings.bin"]["layout"]:
        size = int(spec["size"])
        block = torch.from_numpy(nodes[:, col:col + size].copy()).reshape(n, *spec["shape"])
        getattr(brain.bank, spec["name"]).data.copy_(block)
        col += size
    with (path / "token_embedding.bin").open("rb") as handle:
        brain.embedding.weight.data.copy_(torch.from_numpy(_read_array(handle, meta["files"]["token_embedding.bin"])))
    with (path / "output_head.bin").open("rb") as handle:
        head = _read_array(handle, meta["files"]["output_head.bin"])
    brain.output_head.weight.data.copy_(torch.from_numpy(head[:, :dim].copy()))
    brain.output_head.bias.data.copy_(torch.from_numpy(head[:, dim].copy()))
    with (path / "misc.bin").open("rb") as handle:
        misc = _read_array(handle, meta["files"]["misc.bin"])
    p = 0
    brain.readout_norm.weight.data.copy_(torch.from_numpy(misc[p:p + dim].copy())); p += dim
    brain.readout_norm.bias.data.copy_(torch.from_numpy(misc[p:p + dim].copy())); p += dim
    shape = tuple(meta["files"]["misc.bin"]["mean_out_shape"])
    count = int(np.prod(shape))
    brain.mean_out.data.copy_(torch.from_numpy(misc[p:p + count].copy()).reshape(shape)); p += count
    brain.node_alive.data.copy_(torch.from_numpy(misc[p:p + n].astype(np.bool_))); p += n
    brain.node_born_step.data.copy_(torch.from_numpy(misc[p:p + n].copy())); p += n
    brain.edge_usage[:ne] = torch.from_numpy(misc[p:p + ne].copy())
    brain.cemetery = torch.load(path / "cemetery.bin", weights_only=False)
    return brain, meta


def equity(original: Brain, reloaded: Brain, tokens: torch.Tensor, atol: float = 1e-3) -> dict:
    """Le test d'équité : mêmes logits, sinon le format est rejeté."""
    original.eval(); reloaded.eval()
    with torch.no_grad():
        a, _ = original(tokens, grow=False)
        b, _ = reloaded(tokens, grow=False)
    diff = (a - b).abs()
    return {
        "atol": atol,
        "max_abs": float(diff.max()),
        "mean_abs": float(diff.mean()),
        "ok": bool(float(diff.max()) <= atol),
        "shape": list(a.shape),
    }
