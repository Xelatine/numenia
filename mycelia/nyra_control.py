"""Controle du centre de commandement Nyra : config, etat, eloquence.

Ces trois fichiers JSON sont la seule interface entre le lanceur
(``scripts/nyra_launcher.py``) et le panel (``scripts/webchat.py``) : le
lanceur ecrit l'etat, le panel le lit, et la configuration va dans l'autre
sens. Aucun seuil n'est code en dur ailleurs : la config est relue avant
chaque session, jamais capturee une fois pour toutes.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
CONFIG_PATH = DATA / "nyra_config.json"
STATE_PATH = DATA / "nyra_state.json"
ELOQUENCE_PATH = DATA / "nyra_eloquence.json"
SESSIONS_DIR = DATA / "runs" / "nyra_sessions"
GENOME_PATH = DATA / "genome.pt"
METRICS_PATH = DATA / "metrics.json"

# Les neuf champs du panel, puis les reglages internes : ils n'ont pas besoin
# d'un bouton, mais ils obeissent a la meme regle (rien en dur dans le code
# qui decide d'une pause ou d'une duree).
DEFAULTS: dict = {
    "session_min": 60,
    "batch_size": 8,
    "warmup": 1000,
    "lr_max": 0.0003,
    "seuil_vram": 1900,
    "temp_pause": 85,
    "temp_stop": 90,
    "mode_247": True,
    "pauses_thermiques": True,
    "stop_after_session": False,
    "seq_len": 128,
    "save_min": 5,
    "game_vram_mb": 2048,
    "pause_thermique_min": 5,
    "intervalle_thermique_min": 60,
}

LIMITS: dict = {
    "session_min": (1, 10080),
    "batch_size": (1, 64),
    "warmup": (0, 200000),
    "lr_max": (0.0, 1.0),
    "seuil_vram": (100, 8000),
    "temp_pause": (40, 100),
    "temp_stop": (40, 110),
    "seq_len": (16, 1024),
    "save_min": (1, 240),
    "game_vram_mb": (256, 8000),
    "pause_thermique_min": (1, 120),
    "intervalle_thermique_min": (1, 1440),
}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _truthy(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "on", "oui", "yes", "vrai"}
    return False


def coerce(key: str, value) -> object:
    """Ranger une valeur dans le type (et l'intervalle) de son defaut."""
    default = DEFAULTS[key]
    if isinstance(default, bool):
        return _truthy(value)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if isinstance(default, int):
        number = int(round(number))
    low, high = LIMITS.get(key, (None, None))
    if low is not None:
        number = max(low, min(high, number))
    return int(number) if isinstance(default, int) else float(number)


def read_config(path: Path | None = None) -> dict:
    """Defauts, puis le fichier : un champ absent reste sur son defaut."""
    config = dict(DEFAULTS)
    path = Path(path) if path is not None else CONFIG_PATH
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return config
    if isinstance(raw, dict):
        for key in DEFAULTS:
            if key in raw:
                config[key] = coerce(key, raw[key])
    return config


def _atomic_write(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    os.replace(tmp, path)


def write_config(patch: dict, path: Path | None = None) -> dict:
    """Fusionne un patch valide dans le fichier et renvoie la config relue."""
    path = Path(path) if path is not None else CONFIG_PATH
    config = read_config(path)
    if isinstance(patch, dict):
        for key, value in patch.items():
            if key in DEFAULTS:
                config[key] = coerce(key, value)
    config["updated"] = now_iso()
    _atomic_write(path, config)
    return config


def read_state() -> dict | None:
    """Etat publie par le lanceur, ou None s'il n'y a jamais eu de lanceur."""
    try:
        state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return state if isinstance(state, dict) else None


def write_state(state: dict) -> None:
    state = dict(state)
    state["updated"] = now_iso()
    state["updated_ts"] = datetime.now(timezone.utc).timestamp()
    try:
        _atomic_write(STATE_PATH, state)
    except OSError:
        # Un etat qu'on ne sait pas ecrire ne doit jamais tuer l'entrainement.
        pass


# Dictionnaire de controle de l'eloquence : des mots-outils francais, pas une
# liste de vocabulaire savant. Il ne mesure pas le sens, il mesure si la langue
# francaise apparait dans ce que Nyra repond (panel 🗣️).
MOTS_FR = frozenset(
    """
    le la les l un une des de du d au aux et ou est sont etre ete etait etaient
    bonjour merci salut qui que quoi dont je tu il elle on nous vous ils elles
    me te se ce cet cette ces mon ma mes ton ta tes son sa ses notre nos votre vos
    leur leurs pour par pas ne plus dans sur avec sans sous entre vers chez mais
    donc or ni car si comme tout tous toute toutes tres bien oui non ca cela ceci
    y en ai as a avons avez ont suis es sommes etes fait faire dit dire peu
    beaucoup aussi encore deja toujours jamais quand comment pourquoi quel quelle
    quels quelles moi toi lui eux chaque autre meme ici maintenant alors puis
    enfin peut voici voila aujourd hui demain hier
    """.split()
)

_WORD = re.compile(r"[a-zà-öø-ÿ]+(?:-[a-zà-öø-ÿ]+)?", re.IGNORECASE)


def count_mots_fr(text: str) -> tuple[int, int]:
    """(mots francais, mots au total) dans une reponse."""
    if not text:
        return 0, 0
    words = _WORD.findall(text.lower())
    french = sum(1 for word in words if word in MOTS_FR)
    return french, len(words)


def read_eloquence() -> list:
    try:
        rows = json.loads(ELOQUENCE_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    return rows if isinstance(rows, list) else []


def append_eloquence(entry: dict, limit: int = 500) -> None:
    rows = read_eloquence()
    rows.append(entry)
    _atomic_write(ELOQUENCE_PATH, rows[-limit:])


def read_metrics() -> list:
    try:
        rows = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    return rows if isinstance(rows, list) else []


def append_metrics(row: dict) -> list:
    rows = read_metrics()
    rows.append(row)
    _atomic_write(METRICS_PATH, rows)
    return rows
