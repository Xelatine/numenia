"""Metabolisme de Nyra — 4 nourritures + journal + gardes-fous + états temporels.

Ce module gère les 4 sources d'énergie (nourritures) de l'agent,
écrit un journal metabolique, suit l'état éveil/faim/dormant via une
valeur d'énergie E, et fournit des gardes-fous.

Système d'énergie :
  - E ∈ [0, 100], décroît de -1/min quand éveillé, stable sinon
  - consume(AUGMENTE E) selon la nourriture (N1:+15, N2:+10, N3:+5, N4:+1)
  - États : éveillé (E > 70) / affamé (30 < E ≤ 70) / dormant (E ≤ 30)
  - metabolism.log journalise les transitions de seuil avec timestamp

Les 4 nourritures (disponibles via data/corpus_*.txt ou génération) :
  - N1 : corpus français (LeCarnet) — 24 Mo de caractères
  - N2 : corpus de validation / texte mêlé — 61 Mo
  - N3 : données de benchmark (GSM8K / ARC-AGI fragments) — 676 Mo
  - N4 : bruit de fond / redirect — énergie de repli (toujours disponible)

Le test d'identité (« qui es-tu ? ») doit fonctionner après avoir mangé :
un agent qui mange sans savoir qui il est mange aveuglément.
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

# État énergétique global de l'agent
energy_E: float = 0.0  # Valeur en [0, 100]

FoodName = Literal["N1", "N2", "N3", "N4"]

FOODS: dict[FoodName, dict] = {
    "N1": {
        "label": "Corpus français (LeCarnet)",
        "source": DATA_DIR / "corpus_fr.txt",
        "size_mb": 24.0,
        "activated": False,
        "consumed": 0.0,
        "threshold": 0.5,
        "energy_increment": 15,  # ajouté à E quand consommée
    },
    "N2": {
        "label": "Corpus mélangé validation",
        "source": DATA_DIR / "corpus_gutenberg.txt",
        "size_mb": 61.4,
        "activated": False,
        "consumed": 0.0,
        "threshold": 0.5,
        "energy_increment": 10,
    },
    "N3": {
        "label": "Benchmark GSM8K/ARC-AGI",
        "source": DATA_DIR / "models" / "benchmark_fragments.json",
        "size_mb": 676.0,
        "activated": False,
        "consumed": 0.0,
        "threshold": 0.5,
        "energy_increment": 5,
    },
    "N4": {
        "label": "Répartition / repli",
        "source": None,
        "size_mb": 0.0,
        "activated": True,  # toujours disponible
        "consumed": 0.0,
        "threshold": 0.1,
        "energy_increment": 1,
    },
}


def _decay_energy() -> float:
    """Décroissance naturelle de l'énergie quand éveillé.

    -1 par minute de temps éveillé. Appelée à chaque cycle de vérification.
    """
    global energy_E
    # Si l'agent est éveillé (E > 70), décroissance de 1/min
    if energy_E > 70:
        energy_E = max(0, energy_E - 1.0)
    # Sinon, énergie stable (pas de décroissance dans le sommeil/affamage)
    return energy_E


def _check_threshold_crossing(old_E: float, new_E: float) -> dict | None:
    """Vérifier si un seuil d'état a été franchi.

    Retourne un dict de journalisation si un état a changé, None sinon.
    """
    old_state = _state_from_E(old_E)
    new_state = _state_from_E(new_E)

    if old_state != new_state:
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "old_state": old_state,
            "new_state": new_state,
            "energy": new_E,
            "source": "metabolism_decay",
        }
    return None


def _state_from_E(E: float) -> str:
    """Convertir la valeur d'énergie en état textuel."""
    if E > 70:
        return "éveillé"
    elif E > 30:
        return "affamé"
    else:
        return "dormant"


def _init_log() -> None:
    """Initialiser le fichier journal si inexistant."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not LOG_FILE.exists():
        LOG_FILE.write_text("", encoding="utf-8", newline="\n")


def consume(food: FoodName, amount: float = 1.0) -> dict:
    global energy_E
    """Consommer une quantité d'une nourriture.

    Arguments:
        food: l'une des 4 nourritures (N1..N4)
        amount: quantité à consommer (unités arbitraires)

    Retourne:
        État après consommation : {food, amount, remaining, energy_E, logged}
    """
    if food not in FOODS:
        raise ValueError(f"Nourriture inconnue : {food}")

    _init_log()

    food_info = FOODS[food]

    # Applicationgarde-fou : ne pas consommer si en état dormant
    if energy_E <= 30:
        return {
            "food": food,
            "amount": amount,
            "error": "Consommation impossible : agent en état dormant (E ≤ 30)",
            "energy_E": energy_E,
            "logged": False,
        }

    # Augmentation de l'énergie
    increment = food_info["energy_increment"]
    energy_E = min(100, energy_E + increment * amount)

    # Mise à jour de la consommation
    food_info["consumed"] = round(food_info["consumed"] + amount, 3)
    remaining = round(food_info["size_mb"] - food_info["consumed"], 3)
    food_info["activated"] = remaining > 0 and food_info["consumed"] < food_info["size_mb"] * 0.9

    # Décay after consumption (simuler le passage du temps)
    _decay_energy()

    # Journaliser les transitions de seuil
    crossing = _check_threshold_crossing(energy_E - increment * amount, energy_E)
    if crossing:
        line = (
            f"{crossing['timestamp']} threshold_crossing:"
            f"{crossing['old_state']}->{crossing['new_state']}"
            f" energy={crossing['energy']} source={crossing['source']}\\n"
        )
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with LOG_FILE.open("a", encoding="utf-8") as f:
            f.write(line)

    remaining = round(food_info["size_mb"] - food_info["consumed"], 3)
    return {
        "food": food,
        "amount": amount,
        "consumed": food_info["consumed"],
        "remaining": remaining,
        "energy_E": energy_E,
        "activated": food_info["activated"],
        "logged": True,
        "state": _state_from_E(energy_E),
    }


def status() -> dict:
    """Retourner l'état complet du métabolise."""
    _init_log()
    return {
        "energy_E": round(energy_E, 1),
        "state": _state_from_E(energy_E),
        "foods": {
            food: {
                "label": info["label"],
                "activated": info["activated"],
                "consumed": info["consumed"],
                "remaining": round(info["size_mb"] - info["consumed"], 3),
                "threshold": info["threshold"],
            }
            for food, info in FOODS.items()
        },
    }


def guard() -> dict:
    """Gardes-fous : vérifier l'état métabolique.

    Vérifications :
      - Aucune nourriture ne dépasse 90 % consommée (safety)
      - Le journal existe et n'est pas vide
      - Au moins une nourriture est activée
      - Énergie dans [0, 100]

    Retourne:
        {ok: bool, détails: {...}}
    """
    _init_log()
    st = status()

    issues = []
    for food, info in st["foods"].items():
        consumed = info.get("consumed", 0)
        size_mb = info.get("size_mb", 0)
        if consumed > size_mb * 0.9:
            issues.append(f"{food} > 90 % consommé")

    # Au moins une nourriture doit être activée
    activated = [f for f, i in st["foods"].items() if i.get("activated", False)]
    if not activated:
        issues.append("Aucune nourriture activée")

    # Vérifier le journal
    log_ok = LOG_FILE.exists() and LOG_FILE.stat().st_size > 0
    if not log_ok:
        issues.append("Journal metabolism.log manquant ou vide")

    # Vérifier l'énergie
    if st["energy_E"] < 0 or st["energy_E"] > 100:
        issues.append(f"Énergie hors plage: E={st['energy_E']}")

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

    # Simuler 3 cycles de consommation et décroissance
    print("=== Cycle 1 : consommation N1 ===")
    result = consume("N1", 1.0)
    print(f"  Energie E: {result['energy_E']}, État: {result['state']}")

    print("\n=== Cycle 2 : consommation N2 ===")
    result = consume("N2", 1.0)
    print(f"  Energie E: {result['energy_E']}, État: {result['state']}")

    print("\n=== Cycle 3 : nouvelle consommation N1 ===")
    result = consume("N1", 1.0)
    print(f"  Energie E: {result['energy_E']}, État: {result['state']}")

    # État complet
    print()
    print("État métabolisme :")
    s = status()
    for k, v in s.items():
        print(f"  {k}: {v}")

    # Gardes-fous
    print()
    g = guard()
    print(f"Gardes-fous: ok={g['ok']}")
    if g["issues"]:
        for issue in g["issues"]:
            print(f"  ⚠ {issue}")
    else:
        print("  ✓ Tout ok")