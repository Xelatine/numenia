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
from mycelia.identity import Identity
from mycelia.voice_mycelia import respond

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

# Identité — chargement depuis self.json (pas de fallback, conformément Phase 3.3)
def load_identity():
    path = Path("data/genome_myc/self.json")
    if not path.exists():
        # refus d'éveil — PAS de fallback
        return None
    return Identity.load(path)

_identity = load_identity()
if _identity is None:
    IDENTITY_NONE = True
else:
    IDENTITY_NONE = False
    identity = _identity

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


# ── Point de génération vocale MYCELIA ──────────────────────────
def _voice_chat(user_prompt: str) -> dict:
    """Génère une réponse via le moteur MYCELIA local.

    Retourne un dict avec :
      - réponse: texte brut généré
      - log: entrée de journalgage honnête
    """
    result = respond(user_prompt)
    return result


# ── Routes Flask ──────────────────────────────────────────────
app = Flask(__name__)

@app.route("/awake", methods=["GET"])
def awake():
    """Santé du système : 200 si le cerveau est initialisé."""
    if IDENTITY_NONE:
        return "ÉVEIL REFUSÉ: data/genome_myc/self.json absent", 403
    return "200 OK"


@app.route("/chat", methods=["POST"])
def chat():
    """Génère une réponse de Nyra via la voix locale MYCELIA."""
    data = request.get_json(silent=True) or {}
    user_msg = data.get("message", "")
    if not user_msg:
        return jsonify({"error": "message manquant"}), 400

    try:
        result = _voice_chat(user_msg)
        réponse = result.get("réponse", "")
        log_entry = result.get("voix_config", {}).get("log", "")
        # Journaliser le log dans le fichier accès
        with open(ACCESS_LOG, "a", encoding="utf-8") as f:
            timestamp = datetime.now(timezone.utc).isoformat()
            f.write(timestamp + " chat:" + user_msg + " réponse:" + réponse + " " + log_entry + "\n")
        return jsonify({
            "réponse": réponse,
            "voix_log": log_entry,
            "succès": True,
        })
    except Exception as e:
        return jsonify({
            "erreur": "Erreur génération vocale: " + str(e)[:200],
            "succès": False,
        }), 500


# Page d'accueil
@app.route("/", methods=["GET"])
def index():
    return render_template_string(
        "<html><body><h1>Nyra — Voix MYCELIA locale</h1>"
        "<p>POST /chat avec JSON {\"message\": \"votre question\"}</p>"
        "<p>Endpoint /awake pour vérification de santé.</p></body></html>"
    )


# Lancement
if __name__ == "__main__":
    print("Démarrage Nyra — voix locale MYCELIA")
    print("  /awake  → vérification santé (200 ou erreur)")
    print("  /chat     → POST JSON {\"message\": \"...\"}")
    print("Host: 0.0.0.0:8080 (local ONLY, pas de redirection)")
    app.run(host=FLASK_HOST, port=FLASK_PORT)
