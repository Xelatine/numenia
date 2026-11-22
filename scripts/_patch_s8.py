#!/usr/bin/env python3
from pathlib import Path

p = Path("scripts/webchat.py")
s = p.read_text(encoding="utf-8")

anchor = "\n\n# -- Temps reel, historique, eloquence, configuration ----------------------\nGPU_CACHE"
i = s.find(anchor)
if i == -1:
    raise SystemExit("ancre Temps reel introuvable")

block = r'''
# -- S8 — banc multi-donneurs : chat donneur / traduit ---------------
# ═══════════════════════════════════════════════════════════════════════
DONORS_DIR = None
FAMILY_JSON = None
BENCH_JSON = None
S8_LAB_PATH = None
S8_CORPUS_PATH = None
S8_SEQ = 8


def _s8_paths():
    global DONORS_DIR, FAMILY_JSON, BENCH_JSON, S8_LAB_PATH, S8_CORPUS_PATH, S8_SEQ
    if S8_LAB_PATH is not None:
        return
    import sys, os
    _root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if _root not in sys.path:
        sys.path.insert(0, _root)
    if os.path.join(_root, "scripts") not in sys.path:
        sys.path.insert(0, os.path.join(_root, "scripts"))
    if os.path.join(_root, "scripts", "lab") not in sys.path:
        sys.path.insert(0, os.path.join(_root, "scripts", "lab"))
    from lab_common import CORPUS_TOY as _C, LAB as _L, SEQ as _SEQ
    S8_LAB_PATH = _L
    S8_CORPUS_PATH = _C
    S8_SEQ = int(_SEQ)
    DONORS_DIR = _L / "s8" / "donors"
    FAMILY_JSON = _L / "s8" / "family.json"
    BENCH_JSON = _L / "s8" / "bench.json"


def _label_for(tag: str) -> str:
    _s8_paths()
    m = {
        "A": "dense (lignee S5)",
        "B": "dense variant (graine+arch)",
        "C": "dense ordre melange (contextualite)",
        "D": "NUMENIA sparse entrainee (test symbolique)",
        "trad_A": "NUMENIA traduite d'A (a1b1g0)",
        "trad_B": "NUMENIA traduite de B (a1b1g0)",
        "trad_C": "NUMENIA traduite de C (a1b1g0)",
        "trad_D": "NUMENIA traduite de D (a1b1g0)",
    }
    return m.get(tag, tag)


def _predict_brain(brain, texts, temperature=1.0, top_k=3):
    """Retourne [(prefixe, top_textes, entropie_nats)] pour un Brain/RepairedBrain."""
    corpus = _s8_corpus()
    brain.eval()
    rows = []
    for text in texts:
        ids = corpus.encode(text)
        ids = ids[-S8_SEQ:] if len(ids) >= S8_SEQ else [0] * (S8_SEQ - len(ids)) + ids
        state = None
        last = None
        for token in ids:
            inp = torch.tensor([[token]], dtype=torch.long)
            if state is not None:
                logits, state, _ = brain(inp, state=state, grow=False, trace=False, return_state=True)
                last = logits[0, 0]
            else:
                out = brain(inp, grow=False, trace=False)
                last = out[0, 0] if out.dim() == 3 else (out[0] if out.dim() == 2 else out)
        if last is None:
            continue
        logp = F.log_softmax(last / max(1e-3, float(temperature)), dim=-1)
        p = logp.exp()
        ent = float(-(p * logp).sum())
        top = torch.topk(p, int(top_k)).indices.tolist()
        rows.append({
            "prefixe": text,
            "top%d" % int(top_k): [corpus.vocab[i] for i in top],
            "entropie": round(ent, 4),
        })
    return rows


def _predict_dense(model, texts, temperature=1.0, top_k=3):
    """Retourne [(prefixe, top_textes, entropie_nats)] pour un DenseTransformer."""
    corpus = _s8_corpus()
    model.eval()
    rows = []
    for text in texts:
        ids = corpus.encode(text)
        ids = ids[-S8_SEQ:] if len(ids) >= S8_SEQ else [0] * (S8_SEQ - len(ids)) + ids
        x = torch.tensor([ids], dtype=torch.long)
        logits = model(x)[0, -1]
        logp = F.log_softmax(logits / max(1e-3, float(temperature)), dim=-1)
        p = logp.exp()
        ent = float(-(p * logp).sum())
        top = torch.topk(p, int(top_k)).indices.tolist()
        rows.append({
            "prefixe": text,
            "top%d" % int(top_k): [corpus.vocab[i] for i in top],
            "entropie": round(ent, 4),
        })
    return rows


def _s8_corpus():
    _s8_paths()
    from lab_common import ToyCorpus
    return ToyCorpus(seq_len=S8_SEQ, path=S8_CORPUS_PATH)


def _s8_predict_tag(tag, texts, temperature, top_k):
    """Prediction unifiee : retourne rows ou None."""
    rows = None
    try:
        _s8_paths()
    except Exception:
        return rows
    if not DONORS_DIR.exists():
        return rows
    pt = DONORS_DIR / (tag + ".pt")
    if not pt.exists():
        return rows
    try:
        corpus = _s8_corpus()
    except Exception:
        return rows
    try:
        blob = torch.load(str(pt), weights_only=False)
    except Exception:
        return rows
    cfg = blob.get("config", {})
    try:
        if cfg.get("type") == "dense":
            from micro_dense import DenseTransformer
            m = DenseTransformer(
                int(blob.get("vocab_size", len(blob.get("vocab", corpus.vocab)))),
                dim=cfg.get("dim", 16), layers=cfg.get("layers", 2),
                heads=cfg.get("heads", 2), mlp=cfg.get("mlp", 768),
                seq=cfg.get("seq", S8_SEQ),
            )
            m.load_state_dict(blob["state_dict"])
            m.eval()
            rows = _predict_dense(m, texts, temperature, top_k)
        elif cfg.get("type") == "brain":
            from repaired_mycelia import load_brain
            brain = load_brain(pt)
            rows = _predict_brain(brain, texts, temperature, top_k)
        else:
            rows = None
    except Exception:
        rows = None
    return rows


@app.route("/api/lab/s8", methods=["GET"])
def api_lab_s8():
    """Statut du banc S8 : famille + bench."""
    _s8_paths()
    corpus = _s8_corpus()
    family = {}
    if FAMILY_JSON.exists():
        try:
            family = read_json(FAMILY_JSON)
        except Exception:
            family = {}
    bench = {}
    if BENCH_JSON.exists():
        try:
            bench = read_json(BENCH_JSON)
        except Exception:
            bench = {}
    tags = ["A", "B", "C", "D", "trad_A", "trad_B", "trad_C", "trad_D"]
    donors = []
    for tag in tags:
        pt = DONORS_DIR / (tag + ".pt")
        entry = {"tag": tag, "label": _label_for(tag), "pret": pt.exists()}
        if pt.exists():
            try:
                blob = torch.load(str(pt), weights_only=False)
            except Exception:
                entry["type"] = "unknown"
                entry["config_brut"] = {}
                entry["extra"] = {}
                entry["vocab_size"] = int(corpus.vocab_size)
                entry["n_parameters"] = None
            else:
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
    return jsonify({
        "date": datetime.now().isoformat(timespec="seconds"),
        "donors_dir": str(DONORS_DIR.relative_to(ROOT)),
        "corpus": {"path": str(S8_CORPUS_PATH.relative_to(ROOT)), "vocab": int(corpus.vocab_size), "val_windows": len(corpus.fixed(16, "val"))},
        "donors": donors,
        "family": family,
        "bench": bench,
        "ready": all((DONORS_DIR / (t + ".pt")).exists() for t in ["A", "B", "C", "D"]),
    })


@app.route("/api/lab/s8/chat", methods=["POST"])
def api_lab_s8_chat():
    """Chat token par token : donneur ou traduit complete un prefixe.

    Corps attendu :
        tag: "A"|"B"|"C"|"D"|"trad_A"|...
        prefixes: ["le chat", ...], jusqu'a 8
        temperature: 1.0
        top_k: 3
    Resultat : { tag, label, temperature, top_k, predictions: [ {prefixe, top3, entropie} ] }
    """
    _s8_paths()
    body = request.get_json(silent=True) or {}
    tag = str(body.get("tag", "A")).strip()
    allowed = ["A", "B", "C", "D", "trad_A", "trad_B", "trad_C", "trad_D"]
    if not tag or tag not in allowed:
        return jsonify({"ok": False, "error": "tag inconnu : " + tag}), 400
    prefixes = body.get("prefixes") or ["le chat", "je ne veux pas", "la fille", "pourquoi le"]
    if not isinstance(prefixes, list) or not prefixes:
        return jsonify({"ok": False, "error": "prefixes vide"}), 400
    prefixes = [str(p) for p in prefixes]
    temperature = float(body.get("temperature", 1.0))
    top_k = int(body.get("top_k", 3))
    if top_k < 1:
        top_k = 3
    rows = _s8_predict_tag(tag, prefixes[:8], temperature, top_k)
    if rows is None:
        return jsonify({"ok": False, "error": "donneur " + tag + " non present (pas de .pt)"}), 404
    return jsonify({
        "ok": True,
        "tag": tag,
        "label": _label_for(tag),
        "temperature": round(temperature, 3),
        "top_k": top_k,
        "predictions": rows,
    })

'''

s2 = s[:i] + block + s[i:]
p.write_text(s2, encoding="utf-8")
print("patched; s8 block lines inserted:", block.count("\n"))
PY
