"""Lanceur Nyra — démarre le chat web local (scripts/webchat.py).

Usage :
    python start.py

Comportement :
  * refuse de démarrer une seconde instance si le port 8080 est déjà pris ;
  * vérifie que flask et torch sont importables dans l'interpréteur courant ;
  * lance le serveur et recopie sa sortie ici + dans data/webchat/server.log ;
  * Ctrl+C arrête proprement le serveur (jamais de kill global).

Réseau local uniquement : aucune sortie internet n'est ouverte.
"""

from __future__ import annotations

import socket
import subprocess
import sys
from pathlib import Path

# Console Windows en cp1252 : forcer l'UTF-8 pour les accents et le signe ✦.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

ROOT = Path(__file__).resolve().parent
SCRIPT = ROOT / "scripts" / "webchat.py"
PORT = 8080
LOG = ROOT / "data" / "webchat" / "server.log"


def port_busy(port: int) -> bool:
    """Vrai si quelque chose écoute déjà sur le port.

    Un bind ne suffit pas sous Windows : SO_REUSEADDR permet de réserver un port
    déjà écouté. On teste donc une vraie connexion sur la boucle locale.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1.0)
        try:
            sock.connect(("127.0.0.1", port))
        except OSError:
            return False
    return True


def missing_modules() -> list[str]:
    missing = []
    for module in ("flask", "torch"):
        try:
            __import__(module)
        except ImportError:
            missing.append(module)
    return missing


def main() -> int:
    if not SCRIPT.exists():
        print(f"ERREUR : {SCRIPT} introuvable.")
        return 1

    if port_busy(PORT):
        print(f"Nyra tourne déjà sur le port {PORT} — rien à lancer.")
        print(f"URL : http://localhost:{PORT}/chat")
        return 0

    missing = missing_modules()
    if missing:
        print(f"ERREUR : module(s) manquant(s) pour {sys.executable} : {', '.join(missing)}.")
        print("Le chat tourne avec le Python système (flask + torch), pas avec .venv.")
        return 1

    LOG.parent.mkdir(parents=True, exist_ok=True)
    print(f"Nyra ✦ — lancement de {SCRIPT.name} (python {sys.version.split()[0]})")

    process = subprocess.Popen(
        [sys.executable, "-u", str(SCRIPT)],
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )

    try:
        assert process.stdout is not None
        with LOG.open("a", encoding="utf-8") as journal:
            journal.write(f"\n=== lancement {process.pid} ===\n")
            for line in process.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                journal.write(line)
                journal.flush()
    except KeyboardInterrupt:
        print("\nArrêt demandé — extinction du serveur.")
        process.terminate()
    finally:
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()

    print("Nyra ✦ — serveur arrêté.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
