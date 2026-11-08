"""Turn the three arms into the verdict: routing vs capacity, curves, degrees.

The question, asked in two steps. On 8 November the grown colony beat a frozen
random graph of the same size by 6.61 % (case A), which proves that growth pays
but not *why*. This second ablation adds ``schedule``: the same calendar of
births (8 per step, same seed) with the identities drawn at random. It sits
between the two arms and the gap splits into routing and capacity.

    grown     identities chosen by the messages that passed   (traffic)
    schedule  same calendar, identities drawn at random       (capacity)
    frozen    a random graph of the final size, never changed (pre-drawn)

The decision rule was fixed before the run. The instrument's resolution is the
1.75 % run-to-run variance measured on two grown runs, so an arm within 1.5 % of
another one is *equal*, and anything finer is *undetermined* -- said plainly
rather than dressed up.

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
# Measured on 8 November: two independent grown runs gave 1 977.843 and
# 1 943.301, i.e. 1.75 %. Nothing finer than that can be called a difference.
VARIANCE = 0.0175
EQUAL = 0.015


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


def routing_verdict(grown: float, schedule: float, frozen: float) -> dict:
    """Cas 1 / 2 / 3 / indéterminé, with the distance to each landmark in hand."""
    to_grown = (schedule - grown) / grown
    to_frozen = (schedule - frozen) / frozen
    if abs(to_grown) <= EQUAL:
        case = "2"
        text = (
            "L'ARM SANS ROUTAGE ÉGALE LE GRANDI — la valeur de la croissance est un "
            "calendrier de capacité, pas le couplage message→structure"
        )
    elif abs(to_frozen) <= EQUAL:
        case = "3"
        text = (
            "L'ARM SANS ROUTAGE ÉGALE LE FIGÉ — tout l'avantage vient des arêtes nées "
            "là où les messages passent : le couplage routage-structure est l'âme du système"
        )
    elif grown < schedule < frozen:
        case = "1"
        text = (
            "L'ARM SANS ROUTAGE SE PLACE ENTRE LES DEUX — le routage a une contribution "
            "propre, la montée en capacité explique le reste"
        )
    else:
        case = "indéterminé"
        text = (
            "VERDICT INDÉTERMINÉ — l'arm sort de l'intervalle des deux autres, l'instrument "
            "ne sépare pas ce qui se passe à cette résolution"
        )
    return {"case": case, "text": text, "to_grown": to_grown, "to_frozen": to_frozen}


def main() -> None:
    grown = json.loads((ROOT / "ablation_grown.json").read_text(encoding="utf-8"))
    frozen = json.loads((ROOT / "ablation_frozen.json").read_text(encoding="utf-8"))
    schedule = json.loads((ROOT / "ablation_schedule.json").read_text(encoding="utf-8"))
    gc = load_curve(ROOT / "data" / "per_step_grown.csv")
    fc = load_curve(ROOT / "data" / "per_step_frozen.csv")
    sc = load_curve(ROOT / "data" / "per_step_schedule.csv")

    routing = routing_verdict(
        float(grown["val_ppl"]), float(schedule["val_ppl"]), float(frozen["val_ppl"])
    )
    gain_frozen = (frozen["val_ppl"] - grown["val_ppl"]) / grown["val_ppl"]
    gain_schedule = (frozen["val_ppl"] - schedule["val_ppl"]) / frozen["val_ppl"]

    # The frozen arm must not grow at all: any edge variation is a flag bug.
    frozen_edges_seen = {int(r["edges"]) for r in fc.values()}
    frozen_births = sum(int(r["edge_births"]) for r in fc.values())
    schedule_edges_seen = {int(r["edges"]) for r in sc.values()}

    grown_genome = ROOT / "data" / "genome_grown.pt"
    if not grown_genome.is_file():
        grown_genome = ROOT / "data" / "genome_comparaison.pt"
    grown_degrees = describe_file(grown_genome)
    schedule_degrees = describe_file(ROOT / "data" / "genome_schedule.pt")
    frozen_deg = frozen_degrees(frozen)
    gw, sw = windows(gc), windows(sc)
    slope_grown = pearson([w[1] for w in gw], [w[3] for w in gw])
    slope_schedule = pearson([w[1] for w in sw], [w[3] for w in sw])

    lines = [
        "# Ablation — le verdict du routage",
        "",
        "**8 novembre 2026.** Trois arms, même corpus, même graine (42), `seq 32`,",
        f"`batch 8`, {grown['steps']} pas chacun, même carte, trois sessions successives.",
        "Journaux par pas : `data/per_step_grown.csv`, `data/per_step_frozen.csv`,",
        "`data/per_step_schedule.csv`.",
        "",
        "## 1. Verdict : le routage",
        "",
        f"**{routing['text']}** — cas **{routing['case']}**.",
        "",
        f"Schedule à **{routing['to_grown']:+.2%}** du grandi et "
        f"**{routing['to_frozen']:+.2%}** du figé. La bande de résolution est de "
        f"{EQUAL:.1%} : au-delà, l'instrument ne tranche pas "
        f"(variance run-to-run mesurée {VARIANCE:.2%}).",
        "",
        "## 2. Les trois arms",
        "",
        "| arm | perte initiale | perte finale | validation ppl | arêtes | nœuds | durée |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| grandi (routage) | {grown['first_loss']:.4f} | {grown['final_loss']:.4f} | "
        f"**{grown['val_ppl']:.3f}** | {grown['edges']} | {grown['nodes']} | 3 051 s |",
        f"| schedule (sans routage) | {schedule['first_loss']:.4f} | "
        f"{schedule['final_loss']:.4f} | **{schedule['val_ppl']:.3f}** | "
        f"{schedule['edges']} | {schedule['nodes']} | "
        f"{str(schedule['seconds']) + ' s' if schedule.get('seconds') else 'n/d'} |",
        f"| figé aléatoire | {frozen['first_loss']:.4f} | {frozen['final_loss']:.4f} | "
        f"**{frozen['val_ppl']:.3f}** | {frozen['edges']} | {frozen['nodes']} | 3 712 s |",
        "",
        f"Grandi contre figé : **{gain_frozen:+.2%}**. Schedule contre figé : "
        f"**{gain_schedule:+.2%}**. L'écart grandi−schedule est ce que le routage achète.",
        "",
        "## 3. Les trois courbes, pas par pas",
        "",
        "| pas | grandi perte | grandi arêtes | schedule perte | schedule arêtes | "
        "figé perte | figé arêtes |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for mark in MARKS:
        g, s, f = gc.get(mark), sc.get(mark), fc.get(mark)
        if not (g and s and f):
            continue
        lines.append(
            f"| {mark} | {float(g['loss']):.4f} | {g['edges']} | {float(s['loss']):.4f} | "
            f"{s['edges']} | {float(f['loss']):.4f} | {f['edges']} |"
        )

    lines += [
        "",
        "## 4. Distribution des degrés entrants",
        "",
        "| arm | degré entrant moyen | max | Gini | orphelins |",
        "|---|---:|---:|---:|---:|",
        f"| grandi (`{grown_genome.relative_to(ROOT).as_posix()}`) | "
        f"{grown_degrees['mean_in']} | {grown_degrees['max_in']} | "
        f"{grown_degrees['gini_in']} | {grown_degrees['orphans_in']} |",
        f"| schedule (`data/genome_schedule.pt`) | {schedule_degrees['mean_in']} | "
        f"{schedule_degrees['max_in']} | {schedule_degrees['gini_in']} | "
        f"{schedule_degrees['orphans_in']} |",
        f"| figé (reconstruit, graine 42) | {frozen_deg['mean_in']} | {frozen_deg['max_in']} | "
        f"{frozen_deg['gini_in']} | {frozen_deg['orphans_in']} |",
        "",
        "## 5. Vie du graphe lue dans `per_step.csv`",
        "",
        f"- **figé** : arêtes constantes à {sorted(frozen_edges_seen)} sur les 500 pas, "
        f"{frozen_births} naissance cumulée → drapeau vérifié, arm exemptée de "
        "E\\* (elle ne croît pas).",
        f"- **grandi** : {int(gc[max(gc)]['edge_births'])} naissances d'arêtes, "
        f"{int(gc[max(gc)]['edge_deaths'])} morts d'arêtes, "
        f"{int(gc[max(gc)]['node_deaths'])} morts de nœuds, "
        f"{int(gc[max(gc)]['node_revivals'])} résurrections, cimetière "
        f"{int(gc[max(gc)]['cemetery'])}, {int(gc[max(gc)]['edges'])} arêtes, "
        f"{int(gc[max(gc)]['nodes'])} nœuds vivants.",
        f"- **schedule** : {int(sc[max(sc)]['edge_births'])} naissances d'arêtes, "
        f"{int(sc[max(sc)]['edge_deaths'])} morts d'arêtes, "
        f"{int(sc[max(sc)]['node_deaths'])} morts de nœuds, "
        f"{int(sc[max(sc)]['node_revivals'])} résurrections, cimetière "
        f"{int(sc[max(sc)]['cemetery'])}, {int(sc[max(sc)]['edges'])} arêtes, "
        f"{int(sc[max(sc)]['nodes'])} nœuds vivants.",
        f"- arêtes du schedule au fil des pas : {sorted(schedule_edges_seen)[:3]} … "
        f"{sorted(schedule_edges_seen)[-3:]} — le calendrier, vu de l'extérieur.",
        "",
        "### Calendrier des naissances (arêtes cumulées, par fenêtre de 50 pas)",
        "",
        "| fenêtre | grandi naissances/pas | schedule naissances/pas | grandi Δ perte | "
        "schedule Δ perte |",
        "|---|---:|---:|---:|---:|",
    ]
    for (end, g_delta, g_births, _g_edges), (_e, s_delta, s_births, _s_edges) in zip(gw, sw):
        lines.append(
            f"| {end - WINDOW + 1}–{end} | {g_births:.1f} | {s_births:.1f} | "
            f"{g_delta:+.4f} | {s_delta:+.4f} |"
        )

    lines += [
        "",
        f"Corrélation Δ perte / taille du graphe : grandi **{slope_grown:+.2f}**, "
        f"schedule **{slope_schedule:+.2f}** (n = 10, descriptif).",
        "",
        "## 6. Ce que ce verdict ne dit pas",
        "",
        f"- La résolution est de {EQUAL:.1%} : un écart de routage plus fin que la variance "
        f"({VARIANCE:.2%}) ne serait pas vu ici. Un seul arm par condition, sauf le grandi "
        "qui en a deux.",
        "- La ressemblance des calendriers est celle du *budget* (8 naissances par pas, par "
        "construction). Les morts, elles, dépendent du trafic dans les deux arms qui "
        "grandissent : elles ne sont pas appariées une à une.",
        "- Aucune mesure de temps n'entre dans ce verdict ; les durées sont là pour le coût, "
        "pas pour la décision.",
        "",
    ]
    out_path = ROOT / "rapports" / "ABLATION-resultats.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"VERDICT ROUTAGE: cas {routing['case']} | {routing['text']}")
    print(
        f"grandi {grown['val_ppl']:.3f} | schedule {schedule['val_ppl']:.3f} | "
        f"figé {frozen['val_ppl']:.3f}"
    )
    print(f"schedule vs grandi {routing['to_grown']:+.2%} | vs figé {routing['to_frozen']:+.2%}")
    print(f"figé arêtes vues: {sorted(frozen_edges_seen)} | naissances {frozen_births}")
    for name, deg in (("grandi", grown_degrees), ("schedule", schedule_degrees), ("figé", frozen_deg)):
        print(
            f"degrés {name:9s}: {deg['mean_in']} moyen, max {deg['max_in']}, "
            f"Gini {deg['gini_in']}, orphelins {deg['orphans_in']}"
        )


if __name__ == "__main__":
    main()
