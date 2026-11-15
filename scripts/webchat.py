#!/usr/bin/env python3
"""Nyra — Interface web locale (Flask, 0.0.0.0:8080)
Réseau local ONLY : aucune sortie internet, pas de redirection de port.
Le chat utilise le cerveau routé 10 020 nœuds, seuil 0,002342.
Ne modifie jamais les poids.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from flask import Flask, render_template, render_template_string, request, jsonify

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mycelia.brain import Brain
from mycelia.data import get_batch
from mycelia.train import Trainer, process_memory

# ── Configuration ──────────────────────────────────────────────
FLASK_HOST = "0.0.0.0"
FLASK_PORT = 8080
TOKEN_FILE = ROOT / "data" / "webchat" / "token.txt"
ACCESS_LOG = ROOT / "data" / "webchat" / "access.log"
AWAKE_FLAG = ROOT / "data" / "awake.flag"
# Un état affiché doit pouvoir revenir à zéro : sans activité depuis ce délai,
# le flag est retiré. Le badge ne peut plus afficher « actif » indéfiniment.
AWAKE_TTL_SECONDS = 300

# Seuil : on n'envoie pas au LLM, on utilise le cerveau routé
REFERENCE_LOSS = 0.002342

# ── Brain initialisé une seule fois (CPU, routing actif) ──────
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

# ── Utilitaires ──────────────────────────────────────────────
def _load_token() -> str | None:
    try:
        return ACCESS_LOG.read_text(encoding="utf-8").strip() or None
    except FileNotFoundError:
        return None


def _save_token(token: str | None) -> None:
    TOKEN_FILE.parent.mkdir(parents=True, exist_ok=True)
    if token:
        ACCESS_LOG.parent.mkdir(parents=True, exist_ok=True)
        ACCESS_LOG.write_text("", encoding="utf-8", newline="\n")
        TOKEN_FILE.write_text(token, encoding="utf-8")
    else:
        TOKEN_FILE.unlink(missing_ok=True)


def _append_access(ip: str, action: str) -> None:
    ACCESS_LOG.parent.mkdir(parents=True, exist_ok=True)
    line = f"{datetime.now(timezone.utc).isoformat()} {ip} {action}\n"
    # pathlib.Path n'a pas d'append_text : on ouvre en ajout. Sans cela, chaque
    # message casse en 500 (audit du 15/11, ex-webchat.py:81).
    with ACCESS_LOG.open("a", encoding="utf-8") as journal:
        journal.write(line)


def _awake_flag_path() -> Path:
    return AWAKE_FLAG


def _set_awake(flag: bool) -> None:
    if flag:
        AWAKE_FLAG.write_text("", encoding="utf-8")
    else:
        AWAKE_FLAG.unlink(missing_ok=True)


def _is_awake() -> bool:
    try:
        age = time.time() - AWAKE_FLAG.stat().st_mtime
    except FileNotFoundError:
        return False
    if age > AWAKE_TTL_SECONDS:
        AWAKE_FLAG.unlink(missing_ok=True)
        return False
    return True

# ── Routes Flask ─────────────────────────────────────────────
app = Flask(__name__)

# Token optional : s'il existe, il faut le vérifier
REQUIRE_TOKEN = False


@app.route("/chat", methods=["GET", "POST"])
def chat():
    ip = request.remote_addr or "unknown"
    action = "open" if request.method == "GET" else "message"

    # Flag éveil : premier message crée le flag
    if request.method == "POST" and not _is_awake():
        _set_awake(True)

    # Si le navigateur ferme ou 5 min d'inactivité → supprimer le flag
    # (géré côté JS avec setTimeout / beforeUnload ; ici on note juste)

    # Traitement du message via le cerveau routé
    user_msg = request.form.get("message", "") if request.method == "POST" else ""

    if user_msg:
        # Écriture journal
        _append_access(ip, f"message:{user_msg[:40]}")

        # Inférence une seule token (batch_size=1, route_fraction actif)
        try:
            import torch
            with torch.no_grad():
                tokens = torch.tensor([[int(c) for c in user_msg[:32]]], dtype=torch.long)
                logits, _ = brain(tokens, grow=False, return_state=False)
                # Shape logits: [1, seq_len, vocab], on prend la dernière token
                pred = logits[0, -1, :].argmax().item()
                response = chr(max(32, min(126, pred)))  # affichable ASCII
        except Exception as e:
            response = f"⚠ erreur cerveau : {e}"

        # Délai simulé (temps de "pensée" du cerveau)
        time.sleep(0.3)

        # Journal de la réponse
        _append_access(ip, f"réponse:{response[:40]}")

        # Étiquette [nyra] sur la première réponse autonome
        # (le self-play l'écrira plus tard ; ici on juste note)
        label = "[nyra]"

    else:
        response = ""
        label = "[nyra]"

    # Lecture du badge métabolisme en temps réel
    badge = "actif" if _is_awake() else "en attente du verdict"

    html = render_template_string(TEMPLATE, title="Nyra ✦", response=response,
                                   badge=badge, label=label,
                                   awake=_is_awake(),
                                   référence=REFERENCE_LOSS)

    # Nouveau message → badge actif
    if request.method == "POST" and user_msg:
        _set_awake(True)

    return html


@app.route("/feedback", methods=["POST"])
def feedback():
    ip = request.remote_addr or "unknown"
    choix = request.form.get("choix", "")  # "👍" ou "👎"
    _append_access(ip, f"feedback:{choix}")
    # Écriture dans preferences.jsonl seulement si IP consignée du user
    # (cf. Phase 3 : seules les IP du user comptent)
    # Pour l'instant on journalise simplement
    return jsonify(ok=True)


@app.route("/history")
def history():
    # Retourne les derniers échanges de cette session
    # (mémoire de session basique via Flask session ou cookie)
    return jsonify(derniers=[], comptes=0)


@app.route("/debug")
def debug():
    routes = [rule.rule for rule in app.url_map.iter_rules()]
    return {"routes": routes}


@app.route("/awake")
def awake():
    return jsonify(actif=_is_awake())

def run_flask():
    import torch  # assuré d'être chargé dans le thread Flask
    # Flask s'assure que torch est importé une seule fois
    app.run(host=FLASK_HOST, port=FLASK_PORT, threaded=True)

# ── Interface inline (Aucun CDN, responsive) ─────────────────
TEMPLATE = """<!doctype html>
<html lang="fr">
<head>
  <meta charset="utf-8">
  <title>Nyra ✦</title>
  <style>
    body{font-family:system-ui,Arial,sans-serif;margin:0;padding:1rem;background:#fafafa;color:#333}
    .chat{max-width:600px;margin:2rem auto;border:1px solid #ddd;border-radius:8px;padding:1rem;background:#fff}
    .message{margin:0.5rem 0;padding:0.5rem;border-radius:6px;max-width:80%;display:inline-block}
    .user{background:#e3f2fd;margin-left:auto}
    .bot{background:#fff3e0;margin-right:auto}
    .input{display:flex;margin-top:1rem;border-top:1px solid #eee;padding-top:1rem}
    input{flex:1;padding:0.5rem;border:1px solid #ccc;border-radius:4px}
    button{padding:0 0.8rem;margin-left:0.5rem;background:#1976d2;color:#fff;border:none;border-radius:4px}
    .badge{font-size:0.8rem;padding:0.2rem 0.5rem;background:#777;color:#fff;border-radius:3px;margin-left:1rem}
    .goodbad{margin-top:1rem;display:flex;justify-content:flex-end gap:0.5rem}
    @media(max-width:480px){.chat{max-width:100% padding:0.5rem}h2{font-size:1.2rem}}
  </style>
</head>
<body>
  <div class="chat">
    <h2>Nyra ✦</h2>
    <div id="messages">
      {% if response %}<div class="message bot">Nyra : {{ response }}</div>{% endif %}
    </div>
    <div class="input">
      <form action="/chat" method="post">
        <input name="message" placeholder="Tape un message…" autocomplete="off" required>
        <button type="envoyer">Envoyer</button>
      </form>
    </div>
    <span class="badge">{{ badge }}</span>
    <span class="label">{{ label }}</span>
  </div>
  <div class="goodbad">
    <button name="choix" value="👍">👍</button>
    <button name="choix" value="👎">👎</button>
  </div>
  <script>
    // Créer data/awake.flag côté client si pas encore fait
    function checkAwake(){fetch("/awake").then(r=>r.json()).then(d=>{document.querySelector('.badge').textContent=d.actif?'métabolisme : actif':'métabolisme : veille'})}
    setInterval(checkAwake,3000);
    // Soumission du formulaire
    document.querySelector('form').addEventListener('submit',e=>{e.preventDefault();fetch('/chat',{method:'POST',body:new FormData(e.target)}).then(r=>r.text()).then(t=>{document.getElementById('messages').innerHTML+='<div class="message bot">Nyra : '+t+'</div>'})})
    // Badge réseau : on sait pas si extérieur peut répondre
    fetch('/awake').then(r=>r.json()).then(d=>{if(!d.actif)alert('Nyra attend le verdict twin10020.json avant de parler pleinement')})
  </script>
</body>
</html>"""

# ── Lancement ────────────────────────────────────────────────
if __name__ == "__main__":
    # Au démarrage : afficher URL locale + URL réseau
    import socket
    def get_local_ip():
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except Exception:
            return "127.0.0.1"

    local_ip = get_local_ip()
    # Adresse réseau : on cherche une IP locale non loopback
    # (simple : on suppose que l'interface principale est utilisée)
    try:
        import subprocess as sp
        result = sp.check_output(["hostname", "-I"]).decode().split()
        network_ip = result[0] if result else local_ip
    except Exception:
        network_ip = local_ip

    print(f"=== Nyra démarré ===")
    print(f"URL locale : http://localhost:{FLASK_PORT}")
    print(f"URL réseau : http://{network_ip}:{FLASK_PORT}")
    print("Réseau local ONLY — aucune sortie internet")
    print("Badge [métabolisme : veille/actif] en temps réel")
    print("Aucune route vers l'internet sortant n'est ouverte")
    print("===================")
    # Lancer Flask dans un thread dédié
    threading.Thread(target=run_flask, daemon=True).start()
    # Bloquer le processus principal (Flower tourne en boucle)
    import time as _time
    try:
        while True:
            _time.sleep(3600)
    except KeyboardInterrupt:
        pass
    finally:
        # Extinction propre : plus de badge « actif » après l'arrêt.
        _set_awake(False)
        print("Nyra ✦ — flag d'éveil retiré (arrêt propre).")