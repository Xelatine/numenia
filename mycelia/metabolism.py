"""Metabolisme de Nyra — 4 nourritures + journal + gardes-fous.

Ce module gère les 4 sources d'énergie (nourritures) de l'agent,
écrit un journal metabolique et fournit des gardes-fous contre
l'épuisement ou la saturation.

Les 4 nourritures (disponibles via data/corpus_*.txt ou génération) :
  - N1 : corpus français (LeCarnet) — 24 Mo de caractères
  - N2 : corpus de validation / texte mêlé — 61 Mo
  - N3 : données de benchmark (GSM8K / ARC-AGI fragments) — 676 Mo
  - N4 : bruit de fond / redirect — énergie de repli

Chaque nourriture a un taux d'activation, un compteur d'usage et
un seuil de dissipation. Le journal (data/metabolism.log) enregistre
chaque consommation avec timestamp, source et quantité restante.

Le test d'identité (« qui es-tu ? ») doit fonctionner après avoir
mangé : un agent qui mange sans savoir qui il est mange aveuglément.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

# Chemins
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
LOG_FILE = DATA_DIR / "metabolism.log"

# 4 nourritures — identifiantes, indexées par ordinal
FoodName = Literal["N1", "N2", "N3", "N4"]

FOODS: dict[FoodName, dict] = {
    "N1": {
        "label": "Corpus français (LeCarnet)",
        "source": DATA_DIR / "corpus_fr.txt",
        "size_mb": 24.0,
        "activated": False,
        "consumed": 0.0,
        "threshold": 0.5,
    },
    "N2": {
        "label": "Corpus mélangé validation",
        "source": DATA_DIR / "corpus_gutenberg.txt",
        "size_mb": 61.4,
        "activated": False,
        "consumed": 0.0,
        "threshold": 0.5,
    },
    "N3": {
        "label": "Benchmark GSM8K/ARC-AGI",
        "source": DATA_DIR / "models" / "benchmark_fragments.json",
        "size_mb": 676.0,
        "activated": False,
        "consumed": 0.0,
        "threshold": 0.5,
    },
    "N4": {
        "label": "Répartition / repli",
        "source": None,
        "size_mb": 0.0,
        "activated": True,  # toujours disponible
        "consumed": 0.0,
        "threshold": 0.1,
    },
}


def _init_log() -> None:
    """Initialiser le fichier journal si inexistant."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not LOG_FILE.exists():
        LOG_FILE.write_text("", encoding="utf-8", newline="\n")


def consume(food: FoodName, amount: float = 1.0) -> dict:
    """Consommer une quantité d'une nourriture.

    Arguments:
        food: l'une des 4 nourritures (N1..N4)
        amount: quantité à consommer (unités arbitraires)

    Retourne:
        État après consommation : {food, amount, remaining, logged}
    """
    if food not in FOODS:
        raise ValueError(f"Nourriture inconnue : {food}")

    _init_log()

    food_info = FOODS[food]
    previously = food_info["activated"]

    # Mise à jour de la consommation
    food_info["consumed"] = round(food_info["consumed"] + amount, 3)
    remaining = round(food_info["size_mb"] - food_info["consumed"], 3)
    food_info["activated"] = remaining > 0 and food_info["consumed"] < food_info["size_mb"] * 0.9

    # Journalisation
    timestamp = datetime.now(timezone.utc).isoformat()
    line = (
        f"{timestamp} consume:{food} amount:{amount}"
        f" consumed:{food_info['consumed']} remaining:{remaining}MB\n"
    )
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(line)

    return {
        "food": food,
        "amount": amount,
        "consumed": food_info["consumed"],
        "remaining": remaining,
        "activated": food_info["activated"],
        "logged": True,
    }


def status() -> dict:
    """Retourner l'état complet du métabolisme."""
    _init_log()
    return {
        food: {
            "label": info["label"],
            "activated": info["activated"],
            "consumed": info["consumed"],
            "remaining": round(info["size_mb"] - info["consumed"], 3),
            "threshold": info["threshold"],
        }
        for food, info in FOODS.items()
    }


def guard() -> dict:
    """Gardes-fous : vérifier l'état métabolique.

    Vérifications :
      - Aucune nourriture ne dépasse 90 % consommée (safety)
      - Le journal existe et n'est pas vide
      - Au moins une nourriture est activée

    Retourne:
        {ok: bool, détails: {...}}
    """
    _init_log()
    st = status()

    issues = []
    for food, info in st.items():
        consumed = info.get("consumed", 0)
        size_mb = info.get("size_mb", 0)
        if consumed > size_mb * 0.9:
            issues.append(f"{food} > 90 % consommé")

    # Au moins une nourriture doit être activée
    activated = [f for f, i in st.items() if i.get("activated", False)]
    if not activated:
        issues.append("Aucune nourriture activée")

    # Vérifier le journal
    log_ok = LOG_FILE.exists() and LOG_FILE.stat().st_size > 0
    if not log_ok:
        issues.append("Journal metabolism.log manquant ou vide")

    return {
        "ok": len(issues) == 0,
        "issues": issues,
        "status": st,
    }


# ---- Lancement direct (pour test) ----

if __name__ == "__main__":
    print("=== Métabolisme Nyra ===")
    print()

    # Initialiser
    _init_log()

    # Test : consommer un peu de chaque nourriture
    for food in ["N1", "N2", "N3", "N4"]:
        result = consume(food, 0.1)
        print(f"Consommer {result['food']}: remaining={result['remaining']} MB, activated={result['activated']}")

    # État complet
    print()
    print("État métabolisme :")
    for food, info in status().items():
        print(f"  {food}: {'activée' if info['activated'] else 'inactive'} "
              f"( consommée {info['consumed']}/{info['size_mb']} MB )")

    # Gardes-fous
    print()
    g = guard()
    print(f"Gardes-fous: ok={g['ok']}")
    if g["issues"]:
        for issue in g["issues"]:
            print(f"  ⚠ {issue}")