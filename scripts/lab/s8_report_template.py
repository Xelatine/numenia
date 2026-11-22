#!/usr/bin/env python3
"""
S8 — Rapport banc multi-donneurs
=================================

Ce script génère le rapport S8 à partir des résultats du banc.
Il lit les fichiers JSON produits par s8_train_donors.py et s8_run_bench.py.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))
sys.path.insert(0, str(Path(__file__).parent))

from lab_common import LAB, write_json, read_json


def load_json(path):
    """Charger un JSON, retourner None si introuvable."""
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


def generate_report(donors_dir, bench_dir, output_path):
    """Générer le rapport S8 complet."""

    print("=" * 60)
    print("S8 — Génération du rapport banc multi-donneurs")
    print("=" * 60)

    # 1. Carte d'identité des donneurs
    print("\n[1] Carte d'identité des donneurs...")
    identities = {}
    for tag in ["A", "B", "C", "D"]:
        identity_file = bench_dir / f"identity_{tag}.json"
        if identity_file.exists():
            identities[tag] = json.loads(identity_file.read_text(encoding="utf-8"))
            print(f"  [{tag}] {identities[tag]['label']}:"
                  f" perte_nouvelle={identities[tag]['perte_nouvelle']}, "
                  f"perte_ancienne={identities[tag]['perte_ancienne']}, "
                  f"params={identities[tag]['n_parameters']}")
        else:
            print(f"  [{tag}] Introuvable")

    # 2. Matrice de divergence
    print("\n[2] Matrice de divergence...")
    divergence_file = bench_dir / "divergence.json"
    divergence = load_json(divergence_file) if divergence_file.exists() else None

    if divergence:
        print("  Matrice KL:")
        tags = divergence.get("donneurs", [])
        for i, tag_a in enumerate(tags):
            for j, tag_b in enumerate(tags):
                if i >= j:
                    continue
                key = f"{tag_a}_vers_{tag_b}"
                kl = divergence.get("kl_matrix", {}).get(key, "N/A")
                top3 = divergence.get("top3_matrix", {}).get(key, {})
                print(f"    KL({tag_a}‖{tag_b}) = {kl}")
                print(f"      top3_identiques = {top3.get('identiques', 'N/A')}")
                print(f"      top3_recouvrement = {top3.get('recouvrement', 'N/A')}")
    else:
        print("  Pas de matrice de divergence (fichier introuvable)")

    # 3. Table maîtresse
    print("\n[3] Table maîtresse (12 traductions)...")
    table_file = bench_dir / "table_maitresse.json"
    table_data = load_json(table_file) if table_file.exists() else None

    if table_data:
        table = table_data.get("table", [])
        summary = table_data.get("resume", {})

        print(f"  Total: {summary.get('total', len(table))} traductions")
        print(f"  Par donneur: {summary.get('par_donneur', {})}")
        print(f"  Voyance par donneur: {summary.get('voyance_par_donneur', {})}")
        print("\n  Détail:")
        for row in table:
            print(f"    [{row['donneur']}/seed{row['seed_receveur']}] "
                  f"F_avant={row.get('F_avant', 'N/A')}, "
                  f"F_apres={row.get('F_apres', 'N/A')}, "
                  f"top3_avant={row.get('top3_avant', 'N/A')}, "
                  f"top3_apres={row.get('top3_apres', 'N/A')} → {row.get('voyance', '?')}")
    else:
        print("  Pas de table maîtresse (fichier introuvable)")

    # 4. Rédiger le rapport final
    print("\n[4] Rédaction du rapport...")
    rapport = {
        "date": Path(".").absolute().stat().st_mtime,
        "donors_dir": str(donors_dir),
        "bench_dir": str(bench_dir),
        "identites": identities,
        "divergence": divergence,
        "table_maitresse": table_data,
        "conclusions": {
            "voyance_universelle": None,
            "donneur_traduisable": None,
            "donneur_non_traduisable": None,
        }
    }

    # Analyser les résultats
    if table:
        voyance_counts = {}
        for row in table:
            v = row.get("voyance", "?")
            voyance_counts[v] = voyance_counts.get(v, 0) + 1

        total = sum(voyance_counts.values())
        vert_count = voyance_counts.get("VERT", 0)

        if vert_count == total:
            rapport["conclusions"]["voyance_universelle"] = True
            rapport["conclusions"]["voyance"] = "VERT — le traducteur fonctionne universellement"
        elif vert_count > 0:
            rapport["conclusions"]["voyance_universelle"] = False
            rapport["conclusions"]["voyance"] = f"PARTIEL — {vert_count}/{total} traductions VERT"
        else:
            rapport["conclusions"]["voyance_universelle"] = False
            rapport["conclusions"]["voyance"] = f"ROUGE — aucune traduction VERT ({total}/{total})"

        # Identifier les donneurs traduisibles et non traduisibles
        par_donneur = {}
        for tag in identities.keys():
            rows_tag = [row for row in table if row["donneur"] == tag]
            if not rows_tag:
                continue
            vert_tag = sum(1 for row in rows_tag if row.get("voyance") == "VERT")
            total_tag = len(rows_tag)
            par_donneur[tag] = {
                "voyance": "VERT" if vert_tag == total_tag else "JAUNE" if vert_tag > 0 else "ROUGE",
                "vert": vert_tag,
                "total": total_tag,
            }

        if par_donneur:
            vert_tags = [tag for tag, v in par_donneur.items() if v["voyance"] == "VERT"]
            rouge_tags = [tag for tag, v in par_donneur.items() if v["voyance"] == "ROUGE"]

            if vert_tags:
                rapport["conclusions"]["donneur_traduisable"] = vert_tags
            if rouge_tags:
                rapport["conclusions"]["donneur_non_traduisable"] = rouge_tags

    write_json(output_path, rapport)
    print(f"\nRapport écrit dans: {output_path}")
    print(f"  Voyance: {rapport['conclusions'].get('voyance', 'N/A')}")
    if rapport["conclusions"]["donneur_traduisable"]:
        print(f"  Donneurs traduisibles: {rapport['conclusions']['donneur_traduisable']}")
    if rapport["conclusions"]["donneur_non_traduisable"]:
        print(f"  Donneurs non traduisibles: {rapport['conclusions']['donneur_non_traduisable']}")

    return rapport


def main():
    parser = argparse.ArgumentParser(description="S8 : générer le rapport banc multi-donneurs")
    parser.add_argument("--donors-dir", type=Path, default=LAB / "s8" / "donors",
                        help="Répertoire des donateurs")
    parser.add_argument("--bench-dir", type=Path, default=LAB / "s8",
                        help="Répertoire du banc")
    parser.add_argument("--output", type=Path, default=LAB / "s8" / "rapport.json",
                        help="Fichier de sortie du rapport")
    args = parser.parse_args()

    rapport = generate_report(args.donors_dir, args.bench_dir, args.output)

    # Afficher quelques statistiques clés
    print("\n" + "=" * 60)
    print("Statistiques clés")
    print("=" * 60)

    if rapport.get("table_maitresse"):
        table = rapport["table_maitresse"]["table"]
        print(f"\nOntology:")
        print(f"  Total traductions: {len(table)}")
        print(f"  F moyen avant vie: {sum(r.get('F_avant', 0) or 0 for r in table) / len(table):.4f}")
        print(f"  F moyen après vie: {sum(r.get('F_apres', 0) or 0 for r in table) / len(table):.4f}")
        print(f"  Top-3 moyen avant vie: {sum(r.get('top3_avant', 0) or 0 for r in table) / len(table):.4f}")
        print(f"  Top-3 moyen après vie: {sum(r.get('top3_apres', 0) or 0 for r in table) / len(table):.4f}")

        print(f"\nVoyance distribution:")
        for v in ["VERT", "JAUNE", "ROUGE"]:
            count = sum(1 for r in table if r.get("voyance") == v)
            print(f"  {v}: {count}/{len(table)} ({100*count/len(table):.1f}%)")

        print(f"\nPerformance par donneur:")
        for tag in rapport["identities"].keys():
            rows = [r for r in table if r["donneur"] == tag]
            if not rows:
                continue
            identite = rapport["identities"][tag]
            print(f"\n  [{tag}] {identite['label']}")
            print(f"    Type: {identite['type']}")
            print(f"    Perte nouvelle: {identite['perte_nouvelle']}")
            print(f"    Parametres: {identite['n_parameters']}")
            print(f"    Traductions: {len(rows)}")
            print(f"    F moyen avant: {sum(r.get('F_avant', 0) or 0 for r in rows) / len(rows):.4f}")
            print(f"    F moyen après: {sum(r.get('F_apres', 0) or 0 for r in rows) / len(rows):.4f}")
            print(f"    Top-3 moyen avant: {sum(r.get('top3_avant', 0) or 0 for r in rows) / len(rows):.4f}")
            print(f"    Top-3 moyen après: {sum(r.get('top3_apres', 0) or 0 for r in rows) / len(rows):.4f}")
            print(f"    Voyance: {rapport['conclusions']['donneur_traduisable'] and tag in rapport['conclusions']['donneur_traduisable'] and 'VERT' or (rapport['conclusions']['donneur_non_traduisable'] and tag in rapport['conclusions']['donneur_non_traduisable'] and 'ROUGE' or 'N/A')}")


if __name__ == "__main__":
    main()
