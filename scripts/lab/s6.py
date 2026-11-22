"""S6-CONSOLIDATION — le script principal du labo.

Front A (savoir résiste à la vie) :
  0. (re)mesurer le régime de famine sélective à partir du meilleur élève S5.
  1. consolidation cycle vie/ancrage ×4 (a1b1g0) sur 2 semences.
  2. verdict F/R final.

Front B (forme de l'élève) :
  B1. diagnostic de capacité pool 24 ×10 sur 10k.
  B2. pool 48 distillé (a1b1g0).
  B3. membranes 256 distillées, comparé à pool 48 à paramètres égaux.

Résultats dans data/lab/s6_*.json ; rapport dans data/lab/RAPPORT-S6.md.

    python scripts/lab/s6.py --front A --front B

Un front peut être exécuté seul ; si un front échoue, l'autre est livré.
"""

from __future__ import annotations

import argparse
import sys

from lab_common import ensure_lab
from s6_core import (
    arm_capacity,
    arm_consolidation,
    arm_faim_selective,
    arm_membranes256,
    arm_pool48,
    merge_s6,
)


def run_front_A():
    ensure_lab()
    print("== FRONT A — le savoir résiste à la vie ==")
    # A1 — régime de famine sélective documenté (référence destruction S5 + nouveau régime)
    print("— A1 : régime de famine sélective")
    try:
        arm_faim_selective(1)
    except Exception as e:
        print(f"  ⚠ A1 échoué : {e}", file=sys.stderr)
    # A2 — consolidation cycle vie/ancrage ×4
    print("— A2 : consolidation (4 cycles vie 500 + ancrage 100, a1b1g0)")
    try:
        for seed in (1, 2):
            arm_consolidation("a1b1g0", seed=seed)
    except Exception as e:
        print(f"  ⚠ A2 échoué : {e}", file=sys.stderr)
    # A3 — verdict
    print("— A3 : verdict final")
    try:
        merge_s6()
    except Exception as e:
        print(f"  ⚠ A3 (merge) échoué : {e}", file=sys.stderr)


def run_front_B():
    ensure_lab()
    print("== FRONT B — la forme de l'élève gagne de la capacité ==")
    print("— B1 : diagnostic de capacité (pool 24 ×10, corpus 10k)")
    try:
        for seed in (1, 2):
            arm_capacity(seed)
    except Exception as e:
        print(f"  ⚠ B1 échoué : {e}", file=sys.stderr)
    print("— B2 : pool 48 distillé (a1b1g0)")
    try:
        for seed in (1, 2):
            arm_pool48(seed)
    except Exception as e:
        print(f"  ⚠ B2 échoué : {e}", file=sys.stderr)
    print("— B3 : membranes 256 distillées")
    try:
        for seed in (1, 2):
            arm_membranes256(seed)
    except Exception as e:
        print(f"  ⚠ B3 échoué : {e}", file=sys.stderr)
    print("— B4 : verdict forme")
    try:
        merge_s6()
    except Exception as e:
        print(f"  ⚠ B4 (merge) échoué : {e}", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description="S6-CONSOLIDATION")
    parser.add_argument("--front", action="append", default=[])
    args = parser.parse_args()
    fronts = set(args.front or ["A", "B"])
    if "A" in fronts:
        run_front_A()
    if "B" in fronts:
        run_front_B()


if __name__ == "__main__":
    main()
