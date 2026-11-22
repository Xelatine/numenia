#!/usr/bin/env python3
import sys, os, time
from pathlib import Path

# Réduction minimale : on part des helpers lab sans importer numenia
sys.path.insert(0, "scripts")
sys.path.insert(0, "scripts/lab")

from flask import Flask, jsonify, request

try:
    from lab_common import ToyCorpus, CORPUS_TOY, SEQ, LAB
except Exception as e:
    print("lab_common import failed:", e)
    raise

app = Flask(__name__)

DONORS_DIR = LAB / "s8" / "donors"
FAMILY_JSON = LAB / "s8" / "family.json"
BENCH_JSON = LAB / "s8" / "bench.json"

DONORS_DIR.mkdir(parents=True, exist_ok=True)

TAG_LABELS = {
    "A": "dense (lignee S5)",
    "B": "dense variant (graine+arch)",
    "C": "dense ordre melange (contextualite)",
    "D": "NUMENIA sparse entrainee (test symbolique)",
    "trad_A": "NUMENIA traduite d'A (a1b1g0)",
    "trad_B": "NUMENIA traduite de B (a1b1g0)",
    "trad_C": "NUMENIA traduite de C (a1b1g0)",
    "trad_D": "NUMENIA traduite de D (a1b1g0)",
}

ALLOWED_TAGS = list(TAG_LABELS.keys())

def _read_json(path):
    try:
        return __import__("json").loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}

@app.route("/api/lab/s8", methods=["GET"])
def api_lab_s8():
    corpus = ToyCorpus(seq_len=SEQ, path=CORPUS_TOY)
    family = _read_json(FAMILY_JSON) if FAMILY_JSON.exists() else {}
    bench = _read_json(BENCH_JSON) if BENCH_JSON.exists() else {}
    donors = []
    for tag in ALLOWED_TAGS:
        pt = DONORS_DIR / (tag + ".pt")
        entry = {"tag": tag, "label": TAG_LABELS[tag], "pret": pt.exists()}
        if pt.exists():
            import torch
            blob = torch.load(str(pt), weights_only=False)
            cfg = blob.get("config", {})
            entry["type"] = cfg.get("type", "unknown")
            entry["config_brut"] = {
                k: v for k, v in cfg.items()
                if k in ("dim", "layers", "heads", "mlp", "seq", "n_nodes", "two_speeds")
            }
            entry["extra"] = blob.get("extra", {})
            try:
                entry["vocab_size"] = int(blob.get("vocab_size", len(blob.get("vocab", corpus.vocab))))
            except Exception:
                entry["vocab_size"] = int(corpus.vocab_size)
            try:
                entry["n_parameters"] = int(sum(v.numel() for v in blob["state_dict"].values()))
            except Exception:
                entry["n_parameters"] = None
        donors.append(entry)
    import datetime
    return jsonify({
        "date": datetime.datetime.now().isoformat(timespec="seconds"),
        "donors_dir": str(DONORS_DIR.relative_to(Path.cwd())),
        "corpus": {"path": str(CORPUS_TOY.relative_to(Path.cwd())), "vocab": int(corpus.vocab_size), "val_windows": len(corpus.fixed(16, "val"))},
        "donors": donors,
        "family": family,
        "bench": bench,
        "ready": all((DONORS_DIR / (t + ".pt")).exists() for t in ["A", "B", "C", "D"]),
    })

@app.route("/api/lab/s8/chat", methods=["POST"])
def api_lab_s8_chat():
    body = request.get_json(silent=True) or {}
    tag = str(body.get("tag", "A")).strip()
    if not tag or tag not in ALLOWED_TAGS:
        return jsonify({"ok": False, "error": "tag inconnu : " + tag}), 400
    prefixes = body.get("prefixes") or ["le chat", "je ne veux pas", "la fille", "pourquoi le"]
    if not isinstance(prefixes, list) or not prefixes:
        return jsonify({"ok": False, "error": "prefixes vide"}), 400
    prefixes = [str(p) for p in prefixes]
    temperature = float(body.get("temperature", 1.0))
    top_k = int(body.get("top_k", 3))
    if top_k < 1:
        top_k = 3
    pt = DONORS_DIR / (tag + ".pt")
    if not pt.exists():
        return jsonify({"ok": False, "error": "donneur " + tag + " non present (pas de .pt)"}), 404
    corpus = None
    rows = None
    try:
        import torch
        from micro_dense import DenseTransformer
        corpus = ToyCorpus(seq_len=SEQ, path=CORPUS_TOY)
        blob = torch.load(str(pt), weights_only=False)
        cfg = blob.get("config", {})
        if cfg.get("type") != "dense":
            return jsonify({"ok": False, "error": "type " + str(cfg.get("type")) + " non gere dans ce serveur test"}), 501
        m = DenseTransformer(
            int(blob.get("vocab_size", len(blob.get("vocab", corpus.vocab)))),
            dim=cfg.get("dim", 16), layers=cfg.get("layers", 2),
            heads=cfg.get("heads", 2), mlp=cfg.get("mlp", 768),
            seq=cfg.get("seq", SEQ),
        )
        m.load_state_dict(blob["state_dict"])
        m.eval()
        rows = []
        for text in prefixes[:8]:
            ids = corpus.encode(text)
            ids = ids[-SEQ:] if len(ids) >= SEQ else [0] * (SEQ - len(ids)) + ids
            x = torch.tensor([ids], dtype=torch.long)
            logits = m(x)[0, -1]
            logp = torch.nn.functional.log_softmax(logits / max(1e-3, float(temperature)), dim=-1)
            p = logp.exp()
            ent = float(-(p * logp).sum())
            top = torch.topk(p, top_k).indices.tolist()
            rows.append({
                "prefixe": text,
                "top%d" % top_k: [corpus.vocab[i] for i in top],
                "entropie": round(ent, 4),
            })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"ok": False, "error": "predict failed: " + str(e)[:160]}), 500
    return jsonify({
        "ok": True,
        "tag": tag,
        "label": TAG_LABELS[tag],
        "temperature": round(temperature, 3),
        "top_k": top_k,
        "predictions": rows,
    })

if __name__ == "__main__":
    print("S8 server minimal : /api/lab/s8 + /api/lab/s8/chat")
    app.run(host="127.0.0.1", port=8090, debug=False)
