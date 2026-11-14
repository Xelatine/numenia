"""Golden retrieval avec l'encodeur dédié EmbeddingGemma-300M.

Deux chiffres seulement : hit@1 sur les 5 paires, et la marge entre le bon
souvenir et le meilleur faux. La RAM est mesurée (getrusage), pas lue sur fiche.
"""

from __future__ import annotations

import json
import resource
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from mycelia.memory import EpisodicMemory  # noqa: E402
from test_retrieval import GOLDEN  # noqa: E402


def main() -> None:
    from gemma_embed import GemmaEncoder

    encoder = GemmaEncoder()
    memory = EpisodicMemory(dim=encoder.dim)
    for content, _query, _expect in GOLDEN:
        memory.add(encoder(content, kind="doc"), salience=0.9, text=content)

    hits = 0
    margins = []
    for content, query, expect in GOLDEN:
        result = memory.retrieve(encoder(query, kind="query"), top_k=len(GOLDEN))
        best = result[0] if result else None
        ok = bool(best) and expect.lower() in memory.text[best[0]].lower()
        hits += int(ok)
        shown = memory.text[best[0]][:48] if best else "AUCUN"
        if best is not None:
            good = [s for i, s in result if expect.lower() in memory.text[i].lower()]
            bad = [s for i, s in result if expect.lower() not in memory.text[i].lower()]
            margins.append((max(good) if good else float("nan"),
                            max(bad) if bad else float("nan")))
        print(f"{'OK ' if ok else 'KO '} [{expect}] → {shown}")

    goods = [g for g, _ in margins if g == g]
    bads = [b for _, b in margins if b == b]
    if goods:
        print(f"cos bon  : min {min(goods):.4f} | moy {sum(goods)/len(goods):.4f}")
    if bads:
        print(f"cos faux : max {max(bads):.4f} | moy {sum(bads)/len(bads):.4f}")
    hwm_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    print(f"HIT@1 = {hits}/5 | dim {encoder.dim} | HWM {hwm_mb:.1f} Mo")
    report = {
        "encoder": "embeddinggemma-300m-q8_0",
        "md5": "29ebd01c9362065f36405ef185c7755d",
        "dim": encoder.dim,
        "hits": hits,
        "total": len(GOLDEN),
        "hwm_mb": round(hwm_mb, 1),
        "margins": [{"good": g, "bad": b} for g, b in margins],
    }
    (ROOT / "data" / "mem_report" / "golden_gemma.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if hits < len(GOLDEN):
        print("GOLDEN ROUGE — ne pas activer les niveaux 3-4, ne pas ingérer")
        raise SystemExit(1)
    print("GOLDEN 5/5 — l'encodeur est vert")


if __name__ == "__main__":
    main()
