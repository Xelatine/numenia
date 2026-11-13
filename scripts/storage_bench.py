"""Banc stockage out-of-core : int8, mmap froid, réorganisation, dimensionnement.

Tout ce qui suit est mesuré sur BIO-1, sauf le graphe de 200 000 nœuds qui est
une SIMULATION à communautés plantées (jamais présentée comme le graphe réel).
Les lectures froides utilisent POSIX_FADV_DONTNEED : sans cela, on mesurerait
le page cache, pas le disque.
"""

from __future__ import annotations

import json
import mmap
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mycelia.checkpoint import brain_from_genome, load_genome  # noqa: E402
from mycelia.node import BANK_SHAPES  # noqa: E402
from mycelia.genome_io import _bank_layout, md5_file  # noqa: E402

OUT = ROOT / "data" / "mem_report" / "storage_bench.json"
DIM = 32
ROW_FLOATS = sum(size for _, _, size in _bank_layout(DIM))  # 6 401 à dim 32


def fadvise_drop(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)


def sizes_table() -> dict:
    rows = {}
    for n in (10_000, 100_000, 1_000_000):
        topology = n * 20 * (4 + 4 + 4)  # indptr/indices int32 + poids fp32
        rows[n] = {
            "organes_fp32_go": round(n * ROW_FLOATS * 4 / 1e9, 2),
            "organes_int8_go": round(n * ROW_FLOATS * 1 / 1e9, 2),
            "topologie_go": round(topology / 1e9, 3),
            "ram_topk5_fp32_mo": round(0.05 * n * ROW_FLOATS * 4 / 1e6, 1),
            "ram_topk5_int8_mo": round(0.05 * n * ROW_FLOATS * 1 / 1e6, 1),
        }
    return rows


def int8_equity() -> dict:
    """Quantification int8 par ligne de nœud, puis forward d'équité."""
    brain = brain_from_genome(load_genome(ROOT / "data" / "genome_run3000.pt"))
    tokens = torch.tensor([[2, 5, 9, 14, 3, 7, 11, 4, 8, 6, 12, 15, 2, 9, 5, 3]] * 4)
    brain.eval()
    with torch.no_grad():
        base, _ = brain(tokens, grow=False)
    stats = {}
    with torch.no_grad():
        for name in BANK_SHAPES:
            param = getattr(brain.bank, name)
            rows = param.data.reshape(param.shape[0], -1)
            scale = rows.abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / 127.0
            quant = torch.round(rows / scale).clamp(-127, 127).to(torch.int8)
            dequant = quant.float() * scale
            param.data.copy_(dequant.reshape(param.shape))
            stats[name] = {
                "max_abs_row": float((dequant - rows).abs().max()),
                "bytes_fp32": int(rows.numel() * 4),
                "bytes_int8": int(rows.numel()),
            }
    with torch.no_grad():
        after, _ = brain(tokens, grow=False)
    diff = (after - base).abs()
    return {
        "max_abs_logits": float(diff.max()),
        "mean_abs_logits": float(diff.mean()),
        "relative_mean": float(diff.mean() / base.abs().mean()),
        "seuil": 0.05,
        "ok": bool(float(diff.mean() / base.abs().mean()) < 0.05),
        "detail": stats,
    }


def golden_int8() -> dict:
    """HIT@1 lexical avant/après quantification des souvenirs (int8)."""
    sys.path.insert(0, str(ROOT / "scripts"))
    import test_retrieval as tr

    mem_fp = tr.EpisodicMemory(dim=tr.DIM)
    vecs = {}
    for content, _q, _e in tr.GOLDEN:
        vecs[content] = tr.lexical_embed(content)
        mem_fp.add(vecs[content], salience=0.9, text=content)
    mem_q = tr.EpisodicMemory(dim=tr.DIM)
    for content, _q, _e in tr.GOLDEN:
        v = vecs[content]
        scale = max(float(np.abs(v).max()) / 127.0, 1e-8)
        q = np.round(v / scale).astype(np.int8)
        mem_q.add(q.astype(np.float32) * scale, salience=0.9, text=content)

    def hit(memory):
        hits = 0
        for content, query, expect in tr.GOLDEN:
            res = memory.retrieve(tr.lexical_embed(query), top_k=1)
            hits += int(bool(res) and expect.lower() in memory.text[res[0][0]].lower())
        return hits

    return {"fp32": f"{hit(mem_fp)}/5", "int8": f"{hit(mem_q)}/5"}


def simulated_graph(n: int, communities: int, per_node: int = 10, seed: int = 0) -> dict:
    """Graphe SIMULÉ à communautés plantées (pas le graphe réel)."""
    rng = np.random.default_rng(seed)
    size = n // communities
    comm = np.repeat(np.arange(communities), size)
    if comm.size < n:
        comm = np.concatenate([comm, np.full(n - comm.size, communities - 1)])
    src = np.repeat(np.arange(n, dtype=np.int64), per_node)
    start = comm[src] * size
    tgt = start + rng.integers(0, size, size=src.size, dtype=np.int64)
    tgt = np.minimum(tgt, n - 1)
    cross = rng.random(src.size) < 0.1
    tgt[cross] = rng.integers(0, n, size=int(cross.sum()), dtype=np.int64)
    keep = src != tgt
    return {"n": n, "src": src[keep], "tgt": tgt[keep], "community": comm,
            "communities": communities, "per_node": per_node}


def write_rows(path: Path, n: int, row_bytes: int, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    block = 20_000
    with path.open("wb") as handle:
        for start in range(0, n, block):
            stop = min(start + block, n)
            rows = rng.integers(0, 256, size=(stop - start, row_bytes), dtype=np.uint8)
            handle.write(rows.tobytes())
    fd = os.open(path, os.O_RDONLY)
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)


def cold_random(mmap_path: Path, n: int, row_bytes: int, nodes: np.ndarray, repeats: int = 1) -> dict:
    """Lecture froide de lignes choisies, via mmap (une page par ligne)."""
    fadvise_drop(mmap_path)
    handle = mmap_path.open("rb")
    mapped = mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ)
    started = time.perf_counter()
    checksum = 0
    for _ in range(repeats):
        for node in nodes:
            offset = int(node) * row_bytes
            checksum += mapped[offset]
    elapsed = time.perf_counter() - started
    mapped.close(); handle.close()
    return {"seconds": round(elapsed, 4), "rows": int(nodes.size * repeats),
            "us_per_row": round(1e6 * elapsed / max(1, nodes.size * repeats), 2),
            "checksum": int(checksum)}


def reorder_gain(n: int = 50_000, row_bytes: int = ROW_FLOATS) -> dict:
    """Réorganisation physique par communauté : même tirage, ordre séquentiel."""
    graph = simulated_graph(n, communities=max(1, n // 400))
    comm = graph["community"]
    order = np.argsort(comm, kind="stable")
    rank = np.empty(n, dtype=np.int64)
    rank[order] = np.arange(n)
    path = Path("/tmp/sim_rows.bin")
    write_rows(path, n, row_bytes)
    rng = np.random.default_rng(1)
    hot = rng.choice(n, size=max(8, n // 20), replace=False)  # top-k 5 %
    # Avant : ordre d'origine = aléatoire (la communauté est dispersée).
    before_random = cold_random(path, n, row_bytes, np.sort(hot))
    # Après : les mêmes nœuds, triés par leur nouvelle position = régions jointives.
    after = cold_random(path, n, row_bytes, np.sort(rank[hot]))
    path.unlink(missing_ok=True)
    gain = 1.0 - after["seconds"] / max(1e-9, before_random["seconds"])
    return {
        "n": n, "rows_hot": int(hot.size), "row_bytes": row_bytes,
        "cold_random_order_seconds": before_random["seconds"],
        "cold_community_order_seconds": after["seconds"],
        "gain": round(gain, 4),
        "note": "SIMULATION à communautés plantées ; gain = borne supérieure",
    }


def rss_mb() -> float:
    with open("/proc/self/status", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("VmRSS"):
                return int(line.split()[1]) / 1024.0
    return 0.0


def mmap_threshold(n: int = 200_000, row_bytes: int = ROW_FLOATS) -> dict:
    """Chargement complet vs mmap froid à 200 k nœuds (int8, 1,28 Go)."""
    path = Path("/tmp/sim_200k.bin")
    started = time.perf_counter()
    write_rows(path, n, row_bytes)
    write_s = time.perf_counter() - started
    before = rss_mb()
    started = time.perf_counter()
    full = np.fromfile(path, dtype=np.uint8)
    load_s = time.perf_counter() - started
    after = rss_mb()
    rng = np.random.default_rng(2)
    hot = rng.choice(n, size=max(8, n // 20), replace=False)
    cold = cold_random(path, n, row_bytes, np.sort(hot))
    del full
    path.unlink(missing_ok=True)
    return {
        "n": n, "row_bytes": row_bytes, "file_go": round(n * row_bytes / 1e9, 3),
        "write_s": round(write_s, 2), "full_load_s": round(load_s, 2),
        "full_load_rss_mb": round(after - before, 1),
        "mmap_cold_us_per_row": cold["us_per_row"],
        "mmap_rows": cold["rows"],
        "seuil_projete_noeuds_fp32": int(5.3e9 / (row_bytes * 4)),
        "seuil_projete_noeuds_int8": int(5.3e9 / row_bytes),
    }


def main() -> None:
    result = {"sizes": sizes_table(), "int8": int8_equity(), "golden_int8": golden_int8()}
    print("int8:", result["int8"]["max_abs_logits"], "rel", round(result["int8"]["relative_mean"], 5),
          "ok", result["int8"]["ok"])
    print("golden int8:", result["golden_int8"])
    result["reorder"] = reorder_gain()
    print("reorder:", result["reorder"])
    result["mmap"] = mmap_threshold()
    print("mmap:", result["mmap"])
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("écrit", OUT, "| md5", md5_file(OUT))


if __name__ == "__main__":
    main()
