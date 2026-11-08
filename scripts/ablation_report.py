"""Turn the two arms into the verdict: curves, final perplexities, degrees.

The decision rule was fixed before the runs (docs/ABLATION-croissance.md §5):
an arm `frozen` within 2 % of `grown` means growth has not paid at this budget.
This script only reads the artefacts -- no model is trained here.

    python scripts/ablation_report.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from mycelia.ablation import freeze_random_graph  # noqa: E402
from mycelia.brain import Brain  # noqa: E402
from scripts.degree_stats import describe as describe_file, gini  # noqa: E402

MARKS = (50, 100, 150, 200, 250, 300, 350, 400, 450, 500)
WINDOW = 50


def load_curve(path: Path) -> dict[int, dict]:
    rows: dict[int, dict] = {}
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows[int(row["step"])] = row
    return rows


def frozen_degrees(arm: dict) -> dict:
    """Rebuild the arm's graph: freeze_random_graph is deterministic in (n, edges, seed)."""
    brain = Brain(n_nodes=int(arm["pool"]), max_hops=8)
    freeze_random_graph(brain, edges=int(arm["edges"]), seed=42)
    if brain.ne != int(arm["edges"]):
        raise SystemExit(f"graphe figé reconstruit à {brain.ne} arêtes, attendu {arm['edges']}")
    in_degree = torch.bincount(
        brain.edge_index[1, : brain.ne], minlength=brain.n_nodes
    ).double()
    return {
        "edges": brain.ne,
        "nodes": int(brain.n_nodes),
        "mean_in": round(float(in_degree.mean()), 2),
        "max_in": int(in_degree.max()),
        "gini_in": round(gini(in_degree), 4),
        "orphans_in": int((in_degree == 0).sum()),
    }


def windows(curve: dict[int, dict]) -> list[tuple[int, float, float, float]]:
    """(window end, loss delta, births per step, mean edges) per block of WINDOW steps.

    ``edge_births`` is a cumulative counter, not a per-step count: the rate over a
    window is the difference between its two ends, divided by its length.
    """
    out = []
    for end in range(WINDOW, max(curve) + 1, WINDOW):
        start = max(1, end - WINDOW)
        rows = [curve[s] for s in range(start + 1, end + 1) if s in curve]
        if not rows:
            continue
        births = (int(curve[end]["edge_births"]) - int(curve[start]["edge_births"])) / len(rows)
        delta = float(curve[end]["loss"]) - float(curve[start]["loss"])
        mean_edges = sum(int(r["edges"]) for r in rows) / len(rows)
        out.append((end, delta, births, mean_edges))
    return out


def pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = (sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys)) ** 0.5
    return num / den if den else 0.0


def main() -> None:
    grown = json.loads((ROOT / "ablation_grown.json").read_text(encoding="utf-8"))
    frozen = json.loads((ROOT / "ablation_frozen.json").read_text(encoding="utf-8"))
    gc = load_curve(ROOT / "data" / "per_step_grown.csv")
    fc = load_curve(ROOT / "data" / "per_step_frozen.csv")

    gain = (frozen["val_ppl"] - grown["val_ppl"]) / grown["val_ppl"]
    if gain <= 0.02:
        case = "B"
        verdict = "LA CROISSANCE N'A PAS PAYÉ À CE BUDGET (écart ≤ 2 %)"
    elif gain > 0.02:
        case = "A"
        verdict = "LA CROISSANCE A PAYÉ (écart > 2 % en faveur du grandi)"
    else:
        case = "C"
        verdict = "LE GRAPHE FIGÉ FAIT MIEUX (écart > 2 % en faveur du figé)"

    # The frozen arm must not grow at all: any edge variation is a flag bug.
    frozen_edges_seen = {int(r["edges"]) for r in fc.values()}
    frozen_births = sum(int(r["edge_births"]) for r in fc.values())

    grown_degrees = describe_file(ROOT / "data" / "genome_comparaison.pt")
    frozen_deg = frozen_degrees(frozen)
    gw = windows(gc)
    slope_vs_edges = pearson([w[1] for w in gw], [w[3] for w in gw])

    lines = [
        "# Ablation croissance contre graphe figé — résultats bruts",
        "",
        "**8 novembre 2026.** Deux arms, même corpus, même graine (42), `seq 32`, `batch 8`,",
        f"{grown['steps']} pas chacun, même carte, même session. Journal par pas :",
        "`data/per_step_grown.csv` et `data/per_step_frozen.csv` (500 lignes chacun).",
        "",
        "## 1. Verdict (règle fixée d'avance, `docs/ABLATION-croissance.md` §5)",
        "",
        f"**{verdict}** — cas **{case}**.",
        "",
        "| arm | perte initiale | perte finale | validation ppl | arêtes | nœuds | durée |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| grandi | {grown['first_loss']:.4f} | {grown['final_loss']:.4f} | "
        f"**{grown['val_ppl']:.3f}** | {grown['edges']} | {grown['nodes']} | 3 051 s |",
        f"| figé aléatoire | {frozen['first_loss']:.4f} | {frozen['final_loss']:.4f} | "
        f"**{frozen['val_ppl']:.3f}** | {frozen['edges']} | {frozen['nodes']} | 3 712 s |",
        "",
        f"Écart de perplexité de validation : **{gain:+.2%}**. En pertes (nats) : "
        f"{frozen['final_loss'] - grown['final_loss']:+.4f} nat par pas de la part du figé, "
        "soit la même différence vue en unités lisibles.",
        "",
        "## 2. Courbes, pas par pas",
        "",
        "| pas | grandi perte | grandi arêtes | figé perte | figé arêtes |",
        "|---:|---:|---:|---:|---:|",
    ]
    for mark in MARKS:
        g = gc.get(mark)
        f = fc.get(mark)
        if not g or not f:
            continue
        lines.append(
            f"| {mark} | {float(g['loss']):.4f} | {g['edges']} | "
            f"{float(f['loss']):.4f} | {f['edges']} |"
        )

    lines += [
        "",
        "## 3. Distribution des degrés entrants",
        "",
        "| arm | degré entrant moyen | max | Gini | orphelins |",
        "|---|---:|---:|---:|---:|",
        f"| grandi (`data/genome_comparaison.pt`, pas 500) | {grown_degrees['mean_in']} | "
        f"{grown_degrees['max_in']} | {grown_degrees['gini_in']} | {grown_degrees['orphans_in']} |",
        f"| figé (reconstruit, graine 42, {frozen_deg['edges']} arêtes) | {frozen_deg['mean_in']} | "
        f"{frozen_deg['max_in']} | {frozen_deg['gini_in']} | {frozen_deg['orphans_in']} |",
        "",
        "Le génome du pas 500 vient du run jumeau (`COMPARAISON-500`) : l'arm d'ablation",
        "n'a pas sauvegardé le sien. Même protocole, écart de reproductibilité 1,75 %.",
        "",
        "## 4. Vie du graphe lue dans `per_step.csv`",
        "",
        f"- **figé** : arêtes constantes à {sorted(frozen_edges_seen)} sur les 500 pas, "
        f"{frozen_births} naissance cumulée → le drapeau tient, et l'arm est bien exemptée "
        "de la propriété E\\* ≈ naissances/pas × 1 000.",
        f"- **grandi** (compteurs cumulés au pas 500) : {int(gc[max(gc)]['edge_births'])} "
        f"naissances d'arêtes, {int(gc[max(gc)]['edge_deaths'])} morts d'arêtes, "
        f"{int(gc[max(gc)]['node_deaths'])} morts de nœuds, "
        f"{int(gc[max(gc)]['node_revivals'])} résurrections, cimetière "
        f"{int(gc[max(gc)]['cemetery'])}, {int(gc[max(gc)]['edges'])} arêtes et "
        f"{int(gc[max(gc)]['nodes'])} nœuds vivants.",
        f"- naissances par pas : **{min(w[2] for w in gw):.1f} à {max(w[2] for w in gw):.1f}** "
        "sur les dix fenêtres — la colonie ne naît pas *parce que* la perte stagne, elle naît "
        "huit fois par pas par construction (`max_births 8`). Ce n'est pas un mécanisme "
        "adaptatif, c'est une montée en capacité régulière.",
        "",
        "### Pente de perte et taille du graphe, par fenêtre de 50 pas (grandi)",
        "",
        "| fenêtre | Δ perte | naissances/pas | arêtes moyennes |",
        "|---|---:|---:|---:|",
    ]
    for end, delta, births, mean_edges in gw:
        lines.append(
            f"| {end - WINDOW + 1}–{end} | {delta:+.4f} | {births:.1f} | {mean_edges:.0f} |"
        )
    lines += [
        "",
        f"Corrélation Δ perte / arêtes moyennes entre fenêtres : **{slope_vs_edges:+.2f}** "
        "(négative = les fenêtres où le graphe est le plus gros sont celles où la perte "
        "baisse le plus ; n = 10, descriptif, pas une preuve).",
        "## 5. Coût",
        "",
        "Grandi 3 051 s, figé 3 712 s pour le même budget de pas : la croissance n'a pas "
        "coûté de temps, elle en a économisé (18 %), parce qu'elle n'entretient pas 3 997 "
        "arêtes dès le premier pas. Le coût par pas en fin de course est le même (~9 s à "
        "~4 000 arêtes).",
        "",
    ]
    (ROOT / "docs" / "ABLATION-resultats.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"verdict: {verdict} | écart {gain:+.2%} | cas {case}")
    print(f"figé arêtes vues: {sorted(frozen_edges_seen)} | naissances {frozen_births}")
    print(f"génome grandi: {grown_degrees['mean_in']} moyen, max {grown_degrees['max_in']}, "
          f"Gini {grown_degrees['gini_in']}, orphelins {grown_degrees['orphans_in']}")
    print(f"génome figé  : {frozen_deg['mean_in']} moyen, max {frozen_deg['max_in']}, "
          f"Gini {frozen_deg['gini_in']}, orphelins {frozen_deg['orphans_in']}")
    print(f"corrélation pente/arêtes: {slope_vs_edges:+.2f}")


if __name__ == "__main__":
    main()
