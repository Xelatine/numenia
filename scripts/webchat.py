#!/usr/bin/env python3



"""Nyra - interface web locale (Flask, 0.0.0.0:8080).







Reseau local ONLY : aucune sortie internet, pas de redirection de port.



La voix utilise le moteur local NUMENIA (data/genome.pt + data/tokenizer.json),



jamais un modele externe. Ne modifie jamais les poids.



"""







from __future__ import annotations







import json



import subprocess



import sys



import time



from datetime import datetime, timezone



from html import escape



from pathlib import Path







from flask import Flask, jsonify, request

# torch/F sont utilises par le banc S8 (_predict_brain, _predict_dense,
# _s8_predict_tag). Ils n'etaient importes que dans une seule fonction, si bien
# que tout le bloc S8 levait NameError -- c'est la vraie cause du 404 du chat,
# pas la memoire (corrige le 2026-11-22, S8-REPAIR-AD).
import torch
import torch.nn.functional as F







ROOT = Path(__file__).resolve().parent.parent



sys.path.insert(0, str(ROOT))



sys.path.insert(0, str(ROOT / "scripts"))







try:
    from numenia import nyra_control as nc
except Exception:
    nc = None
try:
    from mycelia.brain import Brain
except Exception:
    Brain = None
try:
    from mycelia.identity import Identity
except Exception:
    Identity = None
try:
    from mycelia.nyra_control import count_mots_fr
except Exception:
    count_mots_fr = None
try:
    from mycelia.train import Trainer
except Exception:
    Trainer = None
try:
    from mycelia.voice_mycelia import respond
    VOICE_READY = True
except Exception:
    respond = None
    VOICE_READY = False


try:
    from nyra_panel import PANEL_SCRIPT, PANEL_SECTIONS, PANEL_STYLE, PANEL_TABS
except Exception:
    PANEL_SCRIPT = PANEL_SECTIONS = PANEL_STYLE = PANEL_TABS = ""








# -- Configuration ---------------------------------------------------------



FLASK_HOST = "0.0.0.0"



FLASK_PORT = 8080



ACCESS_LOG = ROOT / "data" / "webchat" / "access.log"



GENOME_PATH = ROOT / "data" / "genome.pt"



METRICS_PATH = ROOT / "data" / "metrics.json"



NEXT_MILESTONE = "10 020 nœuds — serveur B2"



# Jalons de perte du panel 📈 : le prochain est le plus grand encore au-dessus



# de la perte courante.



MILESTONES = (7.0, 5.0, 3.5)



NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0



MAX_CHATLOG = 50







START_TIME = time.time()







# Identite - chargement depuis self.json (pas de fallback, Phase 3.3)



def load_identity():



    path = ROOT / "data" / "genome_myc" / "self.json"



    if not path.exists():



        # refus d'eveil - PAS de fallback



        return None



    return Identity.load(path)











_identity = load_identity()



IDENTITY_NONE = _identity is None



if not IDENTITY_NONE:



    identity = _identity











# -- Brain initialise une seule fois (CPU, routing actif) ------------------



def _init_brain() -> Brain:



    trainer = Trainer(



        config={},



        data="toy",



        device="cpu",



        scale=4,



        batch_size=1,



        seq_len=32,



        seed=42,



        n_nodes=10_020,



        max_node_births=0,



        route_fraction=0.10,



        checkpoint_turns=True,



    )



    trainer.brain.checkpoint_turns = True



    trainer.report_every = 100



    return trainer.brain











brain = _init_brain()











# -- Point de generation vocale NUMENIA ------------------------------------



def _voice_chat(user_prompt: str) -> dict:
    """Genere une reponse via le moteur NUMENIA local (aucun modele externe)."""
    if not VOICE_READY or respond is None:
        return {
            "réponse": "",
            "voix_config": {"log": "voix locale NUMENIA indisponible"},
            "succès": False,
        }
    return respond(user_prompt)











# -- Lecture des donnees reelles -------------------------------------------



def _status_data() -> dict:



    """Etat reel lu sur le checkpoint genome.pt (taille lue, pas codee en dur)."""



    nodes = "N/A"



    alive = "N/A"



    step = "N/A"



    try:



        import torch







        if GENOME_PATH.exists():



            g = torch.load(GENOME_PATH, map_location="cpu", weights_only=False)



            cfg = g.get("config", {}) if isinstance(g, dict) else {}



            # Le pool reel est le n_nodes de premier niveau (brain.n_nodes),



            # pas celui du config de construction : apres des naissances les



            # deux divergent (532 vs 512 au step 25 de la session texte).



            nodes = g.get("n_nodes", cfg.get("n_nodes", "N/A"))



            step = g.get("step", cfg.get("step", "N/A"))



            # La croissance se lit dans le genome : le pool (slots alloues) et



            # les vivants (ceux que le genome n'a pas enterres) sont deux



            # nombres differents, et c'est leur ecart qui raconte la vie.



            buffer = (g.get("state_dict") or {}).get("node_alive")



            if buffer is not None:



                alive = int(buffer.sum())



    except Exception:



        pass



    return {



        "awake": not IDENTITY_NONE,



        "genome": "data/genome.pt",



        "nodes": nodes,



        "nodes_pool": nodes,



        "nodes_alive": alive,



        "step": step,



        "uptime": round(time.time() - START_TIME, 1),



        "voice": "numenia",



    }











def _chatlog_data(limit: int = MAX_CHATLOG) -> list:



    """Derniers echanges lus dans data/webchat/access.log."""



    entries: list = []



    if not ACCESS_LOG.exists():



        return entries



    try:



        with open(ACCESS_LOG, "r", encoding="utf-8", errors="replace") as f:



            lines = f.readlines()[-limit:]



    except OSError:



        return entries



    for line in lines:



        line = line.strip()



        if not line or " chat:" not in line:



            continue



        ts_part, rest = line.split(" chat:", 1)



        if " réponse:" not in rest:



            continue



        prompt, tail = rest.split(" réponse:", 1)



        # tail = "<reponse> voice=numenia genome=... nodes=... step=..."



        reponse = tail



        for marker in (" voice=numenia", " voice="):



            if marker in reponse:



                reponse = reponse.split(marker, 1)[0]



                break



        entries.append(



            {"ts": ts_part.strip(), "prompt": prompt.strip(), "réponse": reponse.strip()}



        )



    return entries











def _metrics_data() -> list:



    """Historique par session, lu dans data/metrics.json (aucun chiffre invente).







    Sans fichier, on affiche l'etat courant et une perte inconnue (N/A) : une



    courbe vide vaut mieux qu'une perte plausible sortie de nulle part.



    """



    if METRICS_PATH.exists():



        try:



            rows = json.loads(METRICS_PATH.read_text(encoding="utf-8"))



            if isinstance(rows, list) and rows:



                return rows



        except (OSError, ValueError):



            pass



    status = _status_data()



    return [



        {



            "date": datetime.now().strftime("%Y-%m-%d %H:%M"),



            "nodes": status["nodes"],



            "step": status["step"],



            "perte": "N/A",



        }



    ]











def _commits(n: int = 5) -> list:



    """Timeline des derniers commits du depot local (git, aucun reseau)."""



    try:



        out = subprocess.run(



            ["git", "log", "--oneline", "-n", str(n)],



            cwd=str(ROOT),



            capture_output=True,



            timeout=5,



        )



        text = out.stdout.decode("utf-8", errors="replace")



        return [ln.strip() for ln in text.splitlines() if ln.strip()]



    except Exception:



        return []











# -- Routes Flask ----------------------------------------------------------



app = Flask(__name__)











@app.route("/awake", methods=["GET"])



def awake():



    """Sante du systeme : 200 si l'identite est presente."""



    if IDENTITY_NONE:



        return "ÉVEIL REFUSÉ: data/genome_myc/self.json absent", 403



    return "200 OK"











@app.route("/chat", methods=["POST"])



def chat():



    """Genere une reponse de Nyra via la voix locale NUMENIA."""



    data = request.get_json(silent=True) or {}



    user_msg = data.get("message", "")



    if not user_msg:



        return jsonify({"error": "message manquant"}), 400







    try:



        result = _voice_chat(user_msg)



        reponse = result.get("réponse", "")



        log_entry = result.get("voix_config", {}).get("log", "")



        # Z8 : le compteur de mots francais s'applique ici, sur la reponse



        # telle qu'elle part ; le lanceur en fait la somme par session dans



        # metrics.json (champ "mots_fr").



        mots_fr, mots_total = count_mots_fr(reponse)



        try:



            nc.append_eloquence(



                {



                    "ts": nc.now_iso(),



                    "session": (nc.read_state() or {}).get("session"),



                    "prompt": user_msg,



                    "reponse": reponse,



                    "mots_fr": mots_fr,



                    "mots_total": mots_total,



                }



            )



        except OSError:



            pass



        ACCESS_LOG.parent.mkdir(parents=True, exist_ok=True)



        with open(ACCESS_LOG, "a", encoding="utf-8") as f:



            timestamp = datetime.now(timezone.utc).isoformat()



            f.write(



                timestamp



                + " chat:"



                + user_msg



                + " réponse:"



                + reponse



                + " "



                + log_entry



                + "\n"



            )



        return jsonify(



            {



                "réponse": reponse,



                "voix_log": log_entry,



                "succès": True,



                "mots_fr": mots_fr,



                "mots_total": mots_total,



            }



        )



    except Exception as e:



        return (



            jsonify(



                {



                    "erreur": "Erreur génération vocale: " + str(e)[:200],



                    "succès": False,



                }



            ),



            500,



        )











@app.route("/api/status", methods=["GET"])



def api_status():



    return jsonify(_status_data())











@app.route("/api/chatlog", methods=["GET"])



def api_chatlog():



    return jsonify(_chatlog_data())











@app.route("/api/metrics", methods=["GET"])



def api_metrics():



    return jsonify(_metrics_data())











# -- Labo S : atlas, film, conversions -------------------------------------



LAB_DIR = ROOT / "data" / "lab"











def _lab_json(name: str, default):



    """Lit un artefact du labo ; un labo vide n'est pas une erreur serveur."""



    try:



        return json.loads((LAB_DIR / name).read_text(encoding="utf-8"))



    except (OSError, ValueError):



        return default











@app.route("/api/lab/atlas", methods=["GET"])



def api_lab_atlas():



    return jsonify(_lab_json("atlas_nyra.json", {}))











@app.route("/api/lab/film", methods=["GET"])



def api_lab_film():



    return jsonify(_lab_json("film_nyra.json", {"noeuds": [], "stimuli": []}))











@app.route("/api/lab/scan-history", methods=["GET"])



def api_lab_scan_history():



    """C4 : serie des scans recurrents, un point par relance de Nyra."""



    return jsonify(_lab_json("scan_history.json", {"scans": []}))











@app.route("/api/lab/s3bis", methods=["GET"])



def api_lab_s3bis():



    """S3-bis : courbe longue, verdict A/B, planchers, suspect D2."""



    return jsonify(



        {



            "summary": _lab_json("s3bis_summary.json", {}),



            "long": _lab_json("s3bis_long.json", {}),



            "ablations": _lab_json("s3bis_ablations.json", {}),



            "context": _lab_json("s3bis_context.json", {}),



            "floors": _lab_json("s3bis_floors.json", {}),



        }



    )











@app.route("/api/lab/s4", methods=["GET"])



def api_lab_s4():



    """S4 : lecture reparee, validation, distillation, survie au metabolisme."""



    return jsonify(



        {



            "summary": _lab_json("s4_summary.json", {}),



            "validation": _lab_json("s4_validation.json", {}),



            "context": _lab_json("s4_context.json", {}),



            "distill": _lab_json("s4_distill.json", {}),



        }



    )











@app.route("/api/lab/s5", methods=["GET"])



@app.route("/api/lab/s5/", methods=["GET"])



def api_lab_s5():



    """S5-max : mesure corrigee, plafond inter-donneurs, score F, survie du receveur."""



    return jsonify(



        {



            "summary": _lab_json("s5_summary.json", {}),



            "mesure": _lab_json("s5_measure.json", {}),



            "pool24": _lab_json("s5_pool24.json", {}),



            "distill": _lab_json("s5_distill.json", {}),



            "plafond": _lab_json("s5_plafond.json", {}),



        }



    )











@app.route("/api/lab/conversions", methods=["GET"])



def api_lab_conversions():



    return jsonify(



        {



            "dense_to_mycelia": _lab_json("conversion_dense_to_mycelia.json", {}),



            "mycelia_to_dense": _lab_json("conversion_mycelia_to_dense.json", {}),



        }



    )











@app.route("/api/lab/s6", methods=["GET"])



def api_lab_s6():



    """S6-consolidation : faim sélective, consolidation cycle vie/ancrage, verdict F/R."""



    return jsonify(



        {



            "summary": _lab_json("s6_summary.json", {}),



            "faim": _lab_json("s6_faim_selective_s1.json", {}),



            "consolidation": _lab_json("s6_consolidation_a1b1g0_s1.json", {}),



        }



    )











@app.route("/api/lab/s7", methods=["GET"])



def api_lab_s7():



    """S7-traducteur : professeur, receveur, traduction, test aveugle, vie, verdict."""



    return jsonify(



        {



            "demo": _lab_json("s7_demo.json", {}),



            "donor": _lab_json("s7_donor.json", {}),



            "test_chain": _lab_json("test_chain.json", {}),



        }



    )











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
    sys.path.insert(0, os.path.join(_root, "scripts"))
    sys.path.insert(0, os.path.join(_root, "scripts", "lab"))
    from lab_common import CORPUS_TOY10K as _C, LAB as _L
    _SEQ = 8  # SEQ n'existe pas dans lab_common
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



    """Retourne [(prefixe, top3_textes, entropie_nats)] pour un Brain/RepairedBrain."""



    corpus = _s8_corpus()



    brain.eval()



    rows = []



    for text in texts:



        ids = corpus.encode(text)



        ids = ids[-S8_SEQ:] if len(ids) >= S8_SEQ else [0] * (S8_SEQ - len(ids)) + ids



        state = brain._fresh_state(1) if hasattr(brain, "_fresh_state") else None



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



    """Retourne [(prefixe, top3_textes, entropie_nats)] pour un DenseTransformer."""



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


    """Corpus 10k piece par piece, sans garder une reference globale dans l'espace des noms du serveur."""


    _s8_paths()


    from lab_common import ToyCorpus, CORPUS_TOY10K as PATH10K


    return ToyCorpus(seq_len=S8_SEQ, path=PATH10K)







def _s8_predict_tag(tag, texts, temperature, top_k):
    """Prédiction unifiée : retourne rows ou None."""
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
        blob = torch.load(str(pt), map_location="cpu", weights_only=False, mmap=True)
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
    from lab_common import CORPUS_TOY as _C, LAB as _L
    _SEQ = 8  # SEQ n'existe pas dans lab_common
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
    """Retourne [(prefixe, top_textes, entropie_nats)] pour un Brain/RepairedBrain.

    Meme convention que le banc et la demo : etat neuf, on porte l'etat le long
    du prompt, et on lit la derniere position -- la distribution du dernier token
    du prefixe, qui est la tache du cerveau (``ToyCorpus.targets(x) = x``).
    ``Brain.forward`` rend **toujours** un tuple : lire ``out.dim()`` leverait
    une AttributeError et faisait repondre 404 a tous les tags cerveau.
    """
    corpus = _s8_corpus()
    brain.eval()
    rows = []
    for text in texts:
        ids = corpus.encode(text)
        ids = ids[-S8_SEQ:] if len(ids) >= S8_SEQ else [0] * (S8_SEQ - len(ids)) + ids
        state = brain._fresh_state(1) if hasattr(brain, "_fresh_state") else None
        logits = None
        for token in ids:
            inp = torch.tensor([[token]], dtype=torch.long)
            out = brain(inp, state=state, grow=False, trace=False, return_state=True)
            logits = out[0] if isinstance(out, tuple) else out
            state = out[1] if isinstance(out, tuple) and len(out) > 1 else state
        if logits is None:
            continue
        logp = F.log_softmax(logits[0, 0] / max(1e-3, float(temperature)), dim=-1)
        prob = logp.exp()
        ent = float(-(prob * logp).sum())
        top = torch.topk(prob, int(top_k)).indices.tolist()
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


def _s8_predict_tag(tag, texts, temperature, top_k, seed=1):
    """Prediction unifiee : (rows, source) ou (None, motif d'absence).

    Chargement paresseux : **un seul** fichier est lu par requete, puis relache.
    Les tags ``trad_X`` servent le traduit du banc -- ``recipient_X_seed{seed}.pt``,
    cerveau sparse de ~272 Ko -- sans jamais charger le donneur correspondant.
    Un donneur ``X`` lit ``donors/X.pt`` comme avant.
    """
    try:
        _s8_paths()
    except Exception:
        return None, "chemins S8 indisponibles"
    if tag.startswith("trad_"):
        base = tag[len("trad_"):]
        pt = S8_LAB_PATH / "s8" / ("recipient_%s_seed%d.pt" % (base, int(seed)))
    else:
        pt = DONORS_DIR / (tag + ".pt")
    if not pt.exists():
        return None, "fichier absent : " + pt.name
    if not pt.is_file():
        return None, "chemin invalide : " + pt.name
    try:
        corpus = _s8_corpus()
    except Exception:
        return None, "corpus indisponible"
    try:
        blob = torch.load(str(pt), map_location="cpu", weights_only=False)
    except Exception as exc:
        return None, "lecture impossible (%s)" % type(exc).__name__
    cfg = dict(blob.get("config", {}) or {})
    try:
        if cfg.get("type") == "dense":
            from micro_dense import DenseTransformer
            m = DenseTransformer(
                int(blob.get("vocab_size", len(corpus.vocab))),
                dim=cfg.get("dim", 16), layers=cfg.get("layers", 2),
                heads=cfg.get("heads", 2), mlp=cfg.get("mlp", 768),
                seq=cfg.get("seq", S8_SEQ),
            )
            m.load_state_dict(blob["state_dict"])
            m.eval()
            rows = _predict_dense(m, texts, temperature, top_k)
            del m
        elif cfg.get("type") == "brain" or cfg.get("n_nodes") is not None:
            from repaired_mycelia import load_brain
            blob = None  # load_brain relit le fichier : on relache d'abord
            brain = load_brain(pt)
            rows = _predict_brain(brain, texts, temperature, top_k)
            del brain
        else:
            return None, "type de modele inconnu"
    except Exception as exc:
        return None, "chargement impossible (%s)" % type(exc).__name__
    blob = None
    return rows, pt.name


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
            entry["type"] = "voir fichiers"  # torch.load consomme trop de memoire; les infos sont dans les .pt
            entry["config_brut"] = {}
            entry["extra"] = {}
            entry["vocab_size"] = int(corpus.vocab_size)
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
    try:
        seed = int(body.get("seed", 1))
    except Exception:
        seed = 1
    rows, source = _s8_predict_tag(tag, prefixes[:8], temperature, top_k, seed)
    if rows is None:
        return jsonify({"ok": False, "error": tag + " indisponible : " + str(source)}), 404
    return jsonify({
        "ok": True,
        "tag": tag,
        "label": _label_for(tag),
        "fichier": source,
        "seed": int(seed) if tag.startswith("trad_") else None,
        "temperature": round(temperature, 3),
        "top_k": top_k,
        "predictions": rows,
    })



# -- Temps reel, historique, eloquence, configuration ----------------------
GPU_CACHE: dict = {"at": 0.0, "value": {}}











# -- Temps reel, historique, eloquence, configuration ----------------------



GPU_CACHE: dict = {"at": 0.0, "value": {}}











def _gpu_stats() -> dict:



    """Temperature, VRAM et occupation du GPU, cachees 5 s.







    Une seule requete ``nvidia-smi`` : les compteurs par processus repondent



    ``[N/A]`` sous WDDM sur ce portable, donc la VRAM des autres se deduit du



    total utilise moins celle du lanceur (voir scripts/nyra_launcher.py).



    """



    now = time.time()



    if now - float(GPU_CACHE["at"]) < 5.0:



        return GPU_CACHE["value"]



    value: dict = {}



    try:



        out = subprocess.run(



            [



                "nvidia-smi",



                "--query-gpu=temperature.gpu,memory.used,memory.total,"



                "utilization.gpu",



                "--format=csv,noheader,nounits",



            ],



            capture_output=True,



            text=True,



            timeout=4,



            creationflags=NO_WINDOW,



        )



        parts = [



            p.strip() for p in (out.stdout or "").strip().splitlines()[0].split(",")



        ]



        value = {



            "temp_c": float(parts[0]),



            "gpu_used_mb": float(parts[1]),



            "gpu_total_mb": float(parts[2]),



            "gpu_pct": float(parts[3]),



        }



    except (OSError, IndexError, ValueError, subprocess.SubprocessError):



        value = {}



    GPU_CACHE.update({"at": now, "value": value})



    return value











def _live_data() -> dict:



    """Etat du lanceur (training / paused / cooling) et mesures du moment."""



    state = nc.read_state() or {}



    try:



        fresh = (time.time() - float(state.get("updated_ts"))) < 120.0



    except (TypeError, ValueError):



        fresh = False



    if state and fresh:



        data = dict(state)



        data["source"] = "lanceur"



    else:



        status = _status_data()



        data = {



            "status": "idle",



            "reason": "lanceur arrete" if not state else "etat du lanceur perime",



            "step": status["step"],



            "session": state.get("session"),



            "session_minutes_left": state.get("session_minutes_left"),



            "loss": state.get("loss"),



            "tok_s": state.get("tok_s"),



            "vram_mb": state.get("vram_mb"),



            "vram_reserved_mb": state.get("vram_reserved_mb"),



            "ram_mb": state.get("ram_mb"),



            "cumulative_training_min": state.get("cumulative_training_min"),



            "vram_guard_hits": state.get("vram_guard_hits"),



            "device": state.get("device"),



            "source": "serveur",



        }



    gpu = _gpu_stats()



    if gpu:



        data.update(



            {



                "temp_c": gpu["temp_c"],



                "gpu_pct": gpu["gpu_pct"],



                "gpu_used_mb": gpu["gpu_used_mb"],



                "gpu_total_mb": gpu["gpu_total_mb"],



            }



        )



        reserved = data.get("vram_reserved_mb") if fresh else None



        data["other_vram_mb"] = round(



            max(0.0, gpu["gpu_used_mb"] - float(reserved or 0.0)), 1



        )



    return data











def _next_milestone(rows: list) -> dict:



    """Prochain jalon de perte et son ETA, d'apres les pas reellement faits."""



    points: list = []



    for row in rows:



        try:



            points.append((float(row["step"]), float(row["perte"])))



        except (KeyError, TypeError, ValueError):



            continue



    if not points:



        return {"cible": None, "note": "aucune perte mesurée pour l'instant"}



    step_now, loss_now = points[-1]



    targets = [m for m in MILESTONES if m < loss_now]



    if not targets:



        return {



            "cible": None,



            "perte_actuelle": round(loss_now, 4),



            "note": "tous les jalons (7 / 5 / 3,5) sont derrière elle",



        }



    target = max(targets)



    rate = None



    window = points[-5:]



    if len(window) >= 2:



        (s0, l0), (s1, l1) = window[0], window[-1]



        if s1 > s0 and l0 > l1:



            rate = (l0 - l1) / (s1 - s0)



    if rate is None:



        return {



            "cible": target,



            "perte_actuelle": round(loss_now, 4),



            "note": "pas encore de pente mesurée (il faut deux sessions en baisse)",



        }



    cfg = nc.read_config()



    state = nc.read_state() or {}



    tok_s = state.get("session_tok_s") or state.get("tok_s")



    per_step_tokens = float(cfg["batch_size"]) * float(cfg["seq_len"])



    needed = (loss_now - target) / rate



    heures = None



    if tok_s and float(tok_s) > 0:



        heures = needed * per_step_tokens / float(tok_s) / 3600.0



    return {



        "cible": target,



        "perte_actuelle": round(loss_now, 4),



        "pas_restants": round(needed, 0),



        "heures": None if heures is None else round(heures, 1),



        "note": (



            f"pente {rate * 1000:.3f} nats / 1000 pas"



            + (f", {float(tok_s):.0f} tok/s mesurés" if tok_s else ", tok/s inconnu")



        ),



    }











def _history_data() -> dict:



    rows = _metrics_data()



    cumul = 0.0



    for row in rows:



        try:



            cumul += float(row.get("duree_min") or 0.0)



        except (TypeError, ValueError):



            continue



    return {



        "rows": rows,



        "cumul_min": round(cumul, 1),



        "next_milestone": _next_milestone(rows),



    }











def _eloquence_data() -> dict:



    entries = nc.read_eloquence()



    series = [



        {



            "date": row.get("date"),



            "session": row.get("session"),



            "mots_fr": row.get("mots_fr"),



        }



        for row in nc.read_metrics()



        if isinstance(row.get("mots_fr"), (int, float))



    ]



    verdict = "premier point"



    if len(series) >= 2:



        last, previous = float(series[-1]["mots_fr"]), float(series[-2]["mots_fr"])



        verdict = "mieux" if last > previous else ("pire" if last < previous else "pareil")



    return {



        "entries": entries,



        "last": entries[-1] if entries else None,



        "series": series,



        "verdict": verdict,



    }











@app.route("/api/live", methods=["GET"])



def api_live():



    return jsonify(_live_data())











@app.route("/api/history", methods=["GET"])



def api_history():



    return jsonify(_history_data())











@app.route("/api/eloquence", methods=["GET"])



def api_eloquence():



    return jsonify(_eloquence_data())











@app.route("/api/config", methods=["GET", "POST"])



def api_config():



    if request.method == "POST":



        patch = request.get_json(silent=True) or {}



        if not isinstance(patch, dict):



            return jsonify({"ok": False, "error": "objet JSON attendu"}), 400



        return jsonify({"ok": True, "config": nc.write_config(patch)})



    return jsonify(nc.read_config())











# -- Dashboard -------------------------------------------------------------



# Chaîne brute : le JS qu'elle contient garde ses échappements (``\n`` dans un



# join, ``\u2019``) au lieu que Python les convertisse en vrais caractères —



# c'est ce qui cassait tout le bloc <script> du dashboard.



DASHBOARD_HTML = r"""<!DOCTYPE html>



<html lang="fr">



<head>



<meta charset="UTF-8">



<meta name="viewport" content="width=device-width, initial-scale=1.0">



<title>Nyra — Dashboard NUMENIA</title>



<style>



  :root { color-scheme: dark; }



  * { box-sizing: border-box; }



  body { margin: 0; padding: 24px; background: #08090b; color: #dfe3e8;



         font: 14px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; }



  header { max-width: 1080px; margin: 0 auto 20px; display: flex;



           align-items: baseline; gap: 12px; flex-wrap: wrap; }



  h1 { margin: 0; font-size: 20px; letter-spacing: .2px; color: #f2f4f7; }



  h1 span { color: #6ee7a8; }



  .sub { color: #7c848f; font-size: 12px; }



  main { max-width: 1080px; margin: 0 auto; display: grid; gap: 16px;



         grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); }



  .card { background: #101317; border: 1px solid #1e242c; border-radius: 10px;



          padding: 16px 18px; }



  .card.wide { grid-column: 1 / -1; }



  h2 { margin: 0 0 12px; font-size: 13px; font-weight: 600; letter-spacing: .08em;



       text-transform: uppercase; color: #8b939e; }



  dl { display: grid; grid-template-columns: auto 1fr; gap: 6px 14px; margin: 0; }



  dt { color: #7c848f; }



  dd { margin: 0; color: #e9edf2; font-variant-numeric: tabular-nums; }



  .ok { color: #6ee7a8; } .warn { color: #f5a86b; } .muted { color: #7c848f; }



  .log { margin: 0; padding: 12px; background: #0a0c0f; border: 1px solid #1a1f26;



         border-radius: 8px; white-space: pre-wrap; word-break: break-word;



         font: 12px/1.6 ui-monospace, "Cascadia Code", Consolas, monospace;



         color: #b8c1cc; }



  .scroll { max-height: 260px; overflow-y: auto; }



  .row { display: flex; gap: 10px; padding: 6px 0; border-bottom: 1px solid #161b21; }



  .row:last-child { border-bottom: 0; }



  .ts { color: #6b737e; flex: 0 0 62px; font-variant-numeric: tabular-nums; }



  .q { color: #9fb6d4; } .a { color: #c9d3de; }



  table { width: 100%; border-collapse: collapse; font-size: 13px; }



  th, td { text-align: left; padding: 7px 8px; border-bottom: 1px solid #1a1f26; }



  th { color: #7c848f; font-weight: 500; font-size: 12px; }



  td { font-variant-numeric: tabular-nums; }



  ul.commits { list-style: none; margin: 0; padding: 0; font-size: 12.5px; }



  ul.commits li { padding: 5px 0; border-bottom: 1px solid #161b21; color: #aab3bd;



                  font-family: ui-monospace, Consolas, monospace; }



  ul.commits li:last-child { border-bottom: 0; }



  .milestone { margin-top: 12px; padding: 10px 12px; border-radius: 8px;



               border: 1px dashed #2c4a3a; background: #0c1611; color: #8ee0ac;



               font-size: 12.5px; }



  svg.chart { display: block; width: 100%; height: 130px; }



  footer { max-width: 1080px; margin: 18px auto 0; color: #5d656f; font-size: 11.5px;



           display: flex; justify-content: space-between; gap: 12px; flex-wrap: wrap; }



  a { color: #6ee7a8; }



__PANEL_STYLE__



</style>



</head>



<body>



<header>



  <h1>Nyra <span>·</span> Dashboard NUMENIA</h1>



  <div class="sub">voix locale · genome=data/genome.pt · aucun modèle externe</div>



  __PANEL_TABS__



</header>







<main>



  <section class="card vis" data-panel="home">



    <h2>🫀 État</h2>



    <dl>



      <dt>Awake</dt><dd id="s-awake">__AWAKE__</dd>



      <dt>Uptime</dt><dd id="s-uptime">__UPTIME__ s</dd>



      <dt>Voix</dt><dd id="s-voice">__VOICE__</dd>



      <dt>Genome</dt><dd id="s-genome">__GENOME__</dd>



    </dl>



  </section>







  <section class="card vis" data-panel="home">



    <h2>🧠 Cerveau</h2>



    <dl>



      <dt>Nodes</dt><dd id="s-nodes">pool __NODES_POOL__ · vivants __NODES_ALIVE__</dd>



      <dt>Step</dt><dd id="s-step">__STEP__</dd>



      <dt>Perte</dt><dd id="s-perte">non mesurée</dd>



    </dl>



  </section>







  <section class="card wide vis" data-panel="home">



    <h2>🗣️ Dernières paroles</h2>



    <div id="last-words" class="log">__LAST_WORDS__</div>



  </section>







  <section class="card wide vis" data-panel="home">



    <h2>💬 Conversations</h2>



    <div id="conversations" class="log scroll">__CONVERSATIONS__</div>



  </section>







  <section class="card wide vis" data-panel="home">



    <h2>📈 Croissance</h2>



    __GROWTH_SVG__



    <table>



      <thead><tr><th>Date</th><th>Nodes</th><th>Step</th><th>Perte</th></tr></thead>



      <tbody id="metrics-body">__METRICS_ROWS__</tbody>



    </table>



    <h2 style="margin-top:16px">Timeline des commits</h2>



    <ul class="commits">__COMMITS__</ul>



    <div class="milestone">Jalon suivant : __NEXT_MILESTONE__</div>



  </section>



__PANEL_SECTIONS__



</main>







<footer>



  <span>Rafraîchissement auto toutes les 10 s (fetch, pas de websocket).</span>



  <span><a href="/dashboard">Rafraîchir maintenant</a> · <a href="/">Accueil</a></span>



</footer>







<script>



function fmt(sec) { return (sec * 1).toFixed(1); }



function esc(t) { return String(t).replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c])); }



function hhmm(ts) { return ts && ts.length > 16 ? ts.substring(11, 16) : ts; }







async function refresh() {



  try {



    const st = await (await fetch('/api/status')).json();



    document.getElementById('s-awake').textContent = st.awake ? 'éveillée' : 'refus d\u2019éveil';



    document.getElementById('s-awake').className = st.awake ? 'ok' : 'warn';



    document.getElementById('s-uptime').textContent = fmt(st.uptime) + ' s';



    document.getElementById('s-voice').textContent = st.voice;



    document.getElementById('s-genome').textContent = st.genome;



    document.getElementById('s-nodes').textContent = 'pool '



      + (st.nodes_pool !== undefined ? st.nodes_pool : st.nodes)



      + ' · vivants '



      + (st.nodes_alive !== undefined ? st.nodes_alive : 'N/A');



    document.getElementById('s-step').textContent = st.step;







    const log = await (await fetch('/api/chatlog')).json();



    const tail = log.slice(-5);



    document.getElementById('last-words').textContent = tail.length



      ? tail.map(e => hhmm(e.ts) + '  ' + e.prompt + '  ->  ' + e['réponse']).join('\n')



      : 'aucune parole enregistrée pour l\u2019instant';



    document.getElementById('conversations').innerHTML = log.length



      ? log.map(e => '<div class="row"><span class="ts">' + hhmm(e.ts) + '</span>'



          + '<span><span class="q">' + esc(e.prompt) + '</span><br>'



          + '<span class="a">' + esc(e['réponse']) + '</span></span></div>').join('')



      : '<span class="muted">aucune conversation</span>';







    const m = await (await fetch('/api/metrics')).json();



    document.getElementById('metrics-body').innerHTML = m.map(r =>



      '<tr><td>' + esc(r.date) + '</td><td>' + esc(r.nodes) + '</td><td>'



      + esc(r.step) + '</td><td>' + esc(r.perte) + '</td></tr>').join('');



  } catch (err) {



    console.error('refresh failed', err);



  }



}



refresh();



setInterval(refresh, 10000);



__PANEL_SCRIPT__



</script>



</body>



</html>



"""











def _growth_svg(nodes_value) -> str:



    """Graphique SVG dessiné à la main : barre du checkpoint + jalon 10 020."""



    try:



        n = int(nodes_value)



    except (TypeError, ValueError):



        n = 0



    target = 10020



    width, height = 640, 130



    left, right, top, bottom = 48, 16, 16, 34



    plot_w = width - left - right



    plot_h = height - top - bottom



    ratio = 0.0 if target <= 0 else max(0.0, min(1.0, n / target))



    bar_w = max(3, int(plot_w * ratio))



    bar_h = int(plot_h * 0.62)



    y_bar = top + plot_h - bar_h







    ticks = ""



    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):



        x = left + plot_w * frac



        val = int(target * frac)



        ticks += (



            f'<line x1="{x:.0f}" y1="{top}" x2="{x:.0f}" y2="{top + plot_h}" '



            f'stroke="#1a1f26" stroke-width="1"/>'



            f'<text x="{x:.0f}" y="{height - 12}" fill="#5d656f" font-size="10" '



            f'text-anchor="middle">{val}</text>'



        )







    return (



        f'<svg class="chart" viewBox="0 0 {width} {height}" '



        f'preserveAspectRatio="none" role="img" '



        f'aria-label="Croissance des nœuds : {n} sur {target}">'



        f"{ticks}"



        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" '



        f'y2="{top + plot_h}" stroke="#2a323c" stroke-width="1"/>'



        f'<rect x="{left}" y="{y_bar}" width="{bar_w}" height="{bar_h}" rx="3" '



        f'fill="#2f6b4a"/>'



        f'<rect x="{left}" y="{y_bar}" width="{bar_w}" height="4" fill="#6ee7a8"/>'



        f'<text x="{left + bar_w + 8}" y="{y_bar + 14}" fill="#c9d3de" '



        f'font-size="11">{n} nœuds</text>'



        f'<text x="{left + plot_w - 4}" y="{top + 12}" fill="#5d656f" '



        f'font-size="10" text-anchor="end">cible {target}</text>'



        f"</svg>"



    )











@app.route("/dashboard", methods=["GET"])



def dashboard():



    """Page dashboard embarquee : etat, voix, conversations, croissance."""



    status = _status_data()



    chatlog = _chatlog_data()



    metrics = _metrics_data()







    last_words = "\n".join(



        f"{e['ts'][11:16]}  {e['prompt']}  ->  {e['réponse']}"



        for e in chatlog[-5:]



    ) or "aucune parole enregistrée pour l’instant"







    conversations = "".join(



        '<div class="row"><span class="ts">'



        + escape(e["ts"][11:16] or e["ts"])



        + '</span><span><span class="q">'



        + escape(e["prompt"])



        + '</span><br><span class="a">'



        + escape(e["réponse"])



        + "</span></span></div>"



        for e in chatlog



    ) or '<span class="muted">aucune conversation</span>'







    metrics_rows = "".join(



        "<tr><td>"



        + escape(str(m["date"]))



        + "</td><td>"



        + escape(str(m["nodes"]))



        + "</td><td>"



        + escape(str(m["step"]))



        + "</td><td>"



        + escape(str(m["perte"]))



        + "</td></tr>"



        for m in metrics



    )







    commits = "".join(f"<li>{escape(c)}</li>" for c in _commits(5))







    html = DASHBOARD_HTML



    html = html.replace("__AWAKE__", "éveillée" if status["awake"] else "refus d’éveil")



    html = html.replace("__UPTIME__", str(status["uptime"]))



    html = html.replace("__VOICE__", escape(str(status["voice"])))



    html = html.replace("__GENOME__", escape(str(status["genome"])))



    html = html.replace("__NODES_POOL__", escape(str(status["nodes_pool"])))



    html = html.replace("__NODES_ALIVE__", escape(str(status["nodes_alive"])))



    html = html.replace("__STEP__", escape(str(status["step"])))



    html = html.replace("__LAST_WORDS__", escape(last_words))



    html = html.replace("__CONVERSATIONS__", conversations)



    html = html.replace("__GROWTH_SVG__", _growth_svg(status["nodes"]))



    html = html.replace("__METRICS_ROWS__", metrics_rows)



    html = html.replace("__COMMITS__", commits or "<li>historique indisponible</li>")



    html = html.replace("__NEXT_MILESTONE__", escape(NEXT_MILESTONE))



    html = html.replace("__PANEL_STYLE__", PANEL_STYLE)



    html = html.replace("__PANEL_TABS__", PANEL_TABS)



    html = html.replace("__PANEL_SECTIONS__", PANEL_SECTIONS)



    html = html.replace("__PANEL_SCRIPT__", PANEL_SCRIPT)



    return html











@app.route("/", methods=["GET"])



def index():



    return (



        "<!DOCTYPE html><html lang=\"fr\"><head><meta charset=\"UTF-8\">"



        "<title>Nyra — Voix NUMENIA locale</title>"



        "<style>body{background:#08090b;color:#dfe3e8;font:14px/1.6 system-ui,sans-serif;"



        "padding:40px;max-width:640px;margin:0 auto}h1{font-size:20px}"



        "a{color:#6ee7a8}code{background:#101317;border:1px solid #1e242c;"



        "border-radius:6px;padding:2px 6px;font-size:12.5px}</style></head><body>"



        "<h1>Nyra — Voix NUMENIA locale</h1>"



        "<p><a href=\"/dashboard\">📊 Ouvrir le dashboard</a></p>"



        "<p>POST <code>/chat</code> avec JSON <code>{\"message\": \"votre question\"}</code></p>"



        "<p><code>/awake</code> pour la vérification de santé.</p>"



        "</body></html>"



    )











# -- Lancement -------------------------------------------------------------



if __name__ == "__main__":



    print("Demarrage Nyra - voix locale NUMENIA")



    print("  /awake     -> verification sante (200 ou erreur)")



    print("  /chat      -> POST JSON {\"message\": \"...\"}")



    print("  /api/status|chatlog|metrics -> JSON")



    print("  /api/live|history|eloquence|config -> JSON (panel)")



    print("  /dashboard -> page HTML")



    print("Host: 0.0.0.0:8080 (local ONLY, pas de redirection)")



    app.run(host=FLASK_HOST, port=FLASK_PORT)



