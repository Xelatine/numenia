"""Courbe de perte du run 3 000 pas, en SVG pur (aucune dépendance).

Lit data/mem_report/run3000_trace.txt et écrit data/mem_report/run3000_loss.svg.
Marque verticalement le pas 500 (couture checkpoint_turns) pour qu'aucun lecteur
ne cherche une anomalie là où il y a un changement de régime documenté.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TRACE = ROOT / "data" / "mem_report" / "run3000_trace.txt"
OUT = ROOT / "data" / "mem_report" / "run3000_loss.svg"

PATTERN = re.compile(r"TRACE\s+(\d+).*perte\s+([0-9.]+)")

W, H = 960, 460
LEFT, RIGHT, TOP, BOTTOM = 78, 30, 30, 62


def main() -> None:
    points = [
        (int(step), float(loss))
        for step, loss in PATTERN.findall(TRACE.read_text(encoding="utf-8"))
    ]
    if not points:
        raise SystemExit("aucun point TRACE")
    steps = [s for s, _ in points]
    losses = [v for _, v in points]
    x0, x1 = 0, max(steps)
    lo = math.floor(math.log10(min(losses)))
    hi = math.ceil(math.log10(max(losses)))
    if hi - lo < 2:
        hi = lo + 2

    def sx(step: float) -> float:
        return LEFT + (step - x0) / (x1 - x0) * (W - LEFT - RIGHT)

    def sy(loss: float) -> float:
        value = math.log10(max(loss, 10**lo))
        return TOP + (hi - value) / (hi - lo) * (H - TOP - BOTTOM)

    path = " ".join(
        f"{'M' if index == 0 else 'L'}{sx(step):.1f},{sy(loss):.1f}"
        for index, (step, loss) in enumerate(points)
    )

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
        f'viewBox="0 0 {W} {H}" font-family="system-ui, sans-serif">',
        f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
        f'<text x="{W/2}" y="20" text-anchor="middle" font-size="15" '
        f'font-weight="600">MYCELIA — run 3 000 pas, perte (échelle log)</text>',
    ]
    for power in range(lo, hi + 1):
        y = sy(10**power)
        parts.append(
            f'<line x1="{LEFT}" y1="{y:.1f}" x2="{W-RIGHT}" y2="{y:.1f}" '
            f'stroke="#e6e6e6"/>'
        )
        parts.append(
            f'<text x="{LEFT-8}" y="{y+4:.1f}" text-anchor="end" font-size="11" '
            f'fill="#555">1e{power}</text>'
        )
    for step in range(0, x1 + 1, 500):
        x = sx(step)
        parts.append(
            f'<line x1="{x:.1f}" y1="{TOP}" x2="{x:.1f}" y2="{H-BOTTOM}" '
            f'stroke="#f0f0f0"/>'
        )
        parts.append(
            f'<text x="{x:.1f}" y="{H-BOTTOM+18}" text-anchor="middle" '
            f'font-size="11" fill="#555">{step}</text>'
        )
    x500 = sx(500)
    parts.append(
        f'<line x1="{x500:.1f}" y1="{TOP}" x2="{x500:.1f}" y2="{H-BOTTOM}" '
        f'stroke="#c0392b" stroke-dasharray="6 4"/>'
    )
    parts.append(
        f'<text x="{x500+6:.1f}" y="{TOP+16}" font-size="11" fill="#c0392b">'
        f'pas 500 — checkpoint_turns ON (reprise)</text>'
    )
    parts.append(
        f'<path d="{path}" fill="none" stroke="#1f6feb" stroke-width="2.2"/>'
    )
    parts.append(
        f'<text x="{LEFT}" y="{H-14}" font-size="11" fill="#555">'
        f'points : {len(points)} relevés TRACE (tous les 100 pas) — '
        f'deux régimes : 0-500 sans recalcul, 500-3000 avec</text>'
    )
    parts.append("</svg>")
    OUT.write_text("\n".join(parts) + "\n", encoding="utf-8")
    print(f"ecrit {OUT} ({OUT.stat().st_size} o, {len(points)} points)")


if __name__ == "__main__":
    main()
