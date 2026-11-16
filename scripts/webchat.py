#!/usr/bin/env python3
"""Nyra - interface web locale (Flask, 0.0.0.0:8080).

Reseau local ONLY : aucune sortie internet, pas de redirection de port.
La voix utilise le moteur local MYCELIA (data/genome.pt + data/tokenizer.json),
jamais un modele externe. Ne modifie jamais les poids.
"""

from __future__ import annotations

import subprocess
import sys
import time
from datetime import datetime, timezone
from html import escape
from pathlib import Path

from flask import Flask, jsonify, request

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mycelia.brain import Brain
from mycelia.identity import Identity
from mycelia.train import Trainer
from mycelia.voice_mycelia import respond

# -- Configuration ---------------------------------------------------------
FLASK_HOST = "0.0.0.0"
FLASK_PORT = 8080
ACCESS_LOG = ROOT / "data" / "webchat" / "access.log"
GENOME_PATH = ROOT / "data" / "genome.pt"
NEXT_MILESTONE = "10 020 nœuds — serveur B2"
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


# -- Point de generation vocale MYCELIA ------------------------------------
def _voice_chat(user_prompt: str) -> dict:
    """Genere une reponse via le moteur MYCELIA local (aucun modele externe)."""
    return respond(user_prompt)


# -- Lecture des donnees reelles -------------------------------------------
def _status_data() -> dict:
    """Etat reel lu sur le checkpoint genome.pt (taille lue, pas codee en dur)."""
    nodes = "N/A"
    step = "N/A"
    try:
        import torch

        if GENOME_PATH.exists():
            g = torch.load(GENOME_PATH, map_location="cpu", weights_only=False)
            cfg = g.get("config", {}) if isinstance(g, dict) else {}
            nodes = cfg.get("n_nodes", "N/A")
            step = cfg.get("step", "N/A")
    except Exception:
        pass
    return {
        "awake": not IDENTITY_NONE,
        "genome": "data/genome.pt",
        "nodes": nodes,
        "step": step,
        "uptime": round(time.time() - START_TIME, 1),
        "voice": "mycelia",
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
        # tail = "<reponse> voice=mycelia genome=... nodes=... step=..."
        reponse = tail
        for marker in (" voice=mycelia", " voice="):
            if marker in reponse:
                reponse = reponse.split(marker, 1)[0]
                break
        entries.append(
            {"ts": ts_part.strip(), "prompt": prompt.strip(), "réponse": reponse.strip()}
        )
    return entries


def _metrics_data() -> list:
    """Historique par session. La perte n'est pas stockee : N/A, on n'invente pas."""
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
    """Genere une reponse de Nyra via la voix locale MYCELIA."""
    data = request.get_json(silent=True) or {}
    user_msg = data.get("message", "")
    if not user_msg:
        return jsonify({"error": "message manquant"}), 400

    try:
        result = _voice_chat(user_msg)
        reponse = result.get("réponse", "")
        log_entry = result.get("voix_config", {}).get("log", "")
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
            {"réponse": reponse, "voix_log": log_entry, "succès": True}
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


# -- Dashboard -------------------------------------------------------------
DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Nyra — Dashboard MYCELIA</title>
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
</style>
</head>
<body>
<header>
  <h1>Nyra <span>·</span> Dashboard MYCELIA</h1>
  <div class="sub">voix locale · genome=data/genome.pt · aucun modèle externe</div>
</header>

<main>
  <section class="card">
    <h2>🫀 État</h2>
    <dl>
      <dt>Awake</dt><dd id="s-awake">__AWAKE__</dd>
      <dt>Uptime</dt><dd id="s-uptime">__UPTIME__ s</dd>
      <dt>Voix</dt><dd id="s-voice">__VOICE__</dd>
      <dt>Genome</dt><dd id="s-genome">__GENOME__</dd>
    </dl>
  </section>

  <section class="card">
    <h2>🧠 Cerveau</h2>
    <dl>
      <dt>Nodes</dt><dd id="s-nodes">__NODES__</dd>
      <dt>Step</dt><dd id="s-step">__STEP__</dd>
      <dt>Perte</dt><dd id="s-perte">non mesurée</dd>
    </dl>
  </section>

  <section class="card wide">
    <h2>🗣️ Dernières paroles</h2>
    <div id="last-words" class="log">__LAST_WORDS__</div>
  </section>

  <section class="card wide">
    <h2>💬 Conversations</h2>
    <div id="conversations" class="log scroll">__CONVERSATIONS__</div>
  </section>

  <section class="card wide">
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
    document.getElementById('s-nodes').textContent = st.nodes;
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
    html = html.replace("__NODES__", escape(str(status["nodes"])))
    html = html.replace("__STEP__", escape(str(status["step"])))
    html = html.replace("__LAST_WORDS__", escape(last_words))
    html = html.replace("__CONVERSATIONS__", conversations)
    html = html.replace("__GROWTH_SVG__", _growth_svg(status["nodes"]))
    html = html.replace("__METRICS_ROWS__", metrics_rows)
    html = html.replace("__COMMITS__", commits or "<li>historique indisponible</li>")
    html = html.replace("__NEXT_MILESTONE__", escape(NEXT_MILESTONE))
    return html


@app.route("/", methods=["GET"])
def index():
    return (
        "<!DOCTYPE html><html lang=\"fr\"><head><meta charset=\"UTF-8\">"
        "<title>Nyra — Voix MYCELIA locale</title>"
        "<style>body{background:#08090b;color:#dfe3e8;font:14px/1.6 system-ui,sans-serif;"
        "padding:40px;max-width:640px;margin:0 auto}h1{font-size:20px}"
        "a{color:#6ee7a8}code{background:#101317;border:1px solid #1e242c;"
        "border-radius:6px;padding:2px 6px;font-size:12.5px}</style></head><body>"
        "<h1>Nyra — Voix MYCELIA locale</h1>"
        "<p><a href=\"/dashboard\">📊 Ouvrir le dashboard</a></p>"
        "<p>POST <code>/chat</code> avec JSON <code>{\"message\": \"votre question\"}</code></p>"
        "<p><code>/awake</code> pour la vérification de santé.</p>"
        "</body></html>"
    )


# -- Lancement -------------------------------------------------------------
if __name__ == "__main__":
    print("Demarrage Nyra - voix locale MYCELIA")
    print("  /awake     -> verification sante (200 ou erreur)")
    print("  /chat      -> POST JSON {\"message\": \"...\"}")
    print("  /api/status|chatlog|metrics -> JSON")
    print("  /dashboard -> page HTML")
    print("Host: 0.0.0.0:8080 (local ONLY, pas de redirection)")
    app.run(host=FLASK_HOST, port=FLASK_PORT)
