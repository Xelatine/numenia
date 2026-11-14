"""Greffe paramétrique — étape 1 : extraction de l'inventaire Qwen.

Ne modifie aucun poids. Lit le GGUF et consigne :
  * métadonnées d'architecture,
  * liste complète des tenseurs (nom, forme, type, octets),
  * les tenseurs utiles à la greffe : embedding, attentions, MLP, norms.

Sortie : data/mem_report/qwen_tensors.json (+ résumé terminal).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mycelia.gguf_reader import read_inventory, summary  # noqa: E402

MODEL = ROOT / "data" / "models" / "qwen2.5-0.5b-instruct-q8_0.gguf"
OUT = ROOT / "data" / "mem_report" / "qwen_tensors.json"


def main() -> int:
    if not MODEL.exists():
        print(f"modèle absent : {MODEL}")
        return 1
    metadata, tensors = read_inventory(MODEL)
    report = summary(metadata, tensors)
    report["file"] = {"path": str(MODEL.relative_to(ROOT)), "bytes": MODEL.stat().st_size}
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"fichier   : {MODEL.name} ({MODEL.stat().st_size} o)")
    print(f"architecture : {report['architecture']} | couches {report['n_layer']}"
          f" | dim {report['n_embd']} | têtes {report['n_head']}"
          f" (kv {report['n_head_kv']}) | ffn {report['n_fft']} | vocab {report['n_vocab']}")
    print(f"tenseurs  : {report['tensor_count']} | types {report['types']}")
    emb = report["global"].get("token_embd.weight")
    if emb:
        print(f"embedding : {emb['shape']} {emb['type_name']} ({emb['n_bytes'] / 1048576:.1f} Mo)")
    print("couche 0  :")
    for name, info in report["layers"].get(0, {}).items():
        print(f"  {name:22s} {info['shape']} {info['type']}")
    print("écrit", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
