#!/usr/bin/env bash
# Lanceur Nyra — version serveur Linux (SERVEUR-LAN).
# Le même rôle que start.py sur le PC : refuser un double lancement,
# choisir l'interpréteur qui a torch, lancer et journaliser.
#
# Usage :  bash scripts/start_nyra.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PORT=8080
LOG="$ROOT/data/webchat/server.log"

# Le venv qui contient réellement torch (l'autre, .venv, est en Python 3.14 sans torch).
if [ -x "$ROOT/.venv-cu126/bin/python" ]; then
    PY="$ROOT/.venv-cu126/bin/python"
else
    PY="$(command -v python3)"
fi

# Double lancement : on teste une vraie écoute, pas un bind.
if ss -ltn 2>/dev/null | grep -q ":$PORT "; then
    echo "Nyra tourne déjà sur le port $PORT — rien à lancer."
    echo "URL : http://localhost:$PORT/chat"
    exit 0
fi

mkdir -p "$ROOT/data/webchat"
setsid "$PY" -u "$ROOT/scripts/webchat.py" >> "$LOG" 2>&1 &
PID=$!
sleep 3

if kill -0 "$PID" 2>/dev/null; then
    IP="$(hostname -I 2>/dev/null | awk '{print $1}')"
    echo "Nyra ✦ lancé (PID $PID, $("$PY" -V 2>&1))"
    echo "URL locale : http://localhost:$PORT/chat"
    [ -n "${IP:-}" ] && echo "URL réseau : http://$IP:$PORT/chat"
    echo "Journal    : $LOG"
else
    echo "Échec du lancement — dernières lignes du journal :"
    tail -20 "$LOG" 2>/dev/null || true
    exit 1
fi
