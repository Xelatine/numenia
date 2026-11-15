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
