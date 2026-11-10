"""Session G — mesurer le coût temps des trois modes de checkpointing.

Un point = un processus (``mem_profile.py one``) : un OOM ou un pic ne
contamine pas le point suivant. Les résultats sont fusionnés dans
``data/mem_report/session-g.json`` et réécrits après CHAQUE point, de sorte
qu'une campagne interrompue garde ses mesures déjà payées.

Les trois modes, sur le même graphe et la même graine :
  off        : aucun checkpoint (la référence de temps)
  on         : checkpoint de tout le tour (session F, ×1,91 à 3 000 arêtes)
  selective  : checkpoint du routage seul (session G)
  off_keep   : off mais télémétrie conservée — le L0 de la session F, qui
               manquait aux grandes tailles
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "mem_report" / "session-g.json"
PROFILE = ROOT / "scripts" / "mem_profile.py"

MODES = {
    "off": [],
    "on": ["--checkpoint-turns"],
    "selective": ["--checkpoint-turns", "--checkpoint-selective"],
    "off_keep": ["--keep-telemetry"],
}

# (arêtes, seq, batch, tbptt, modes)
GROUPS = {
    "1000": [(1000, 32, 8, 16, ("off", "on", "selective"))],
    "2000": [(2000, 32, 8, 16, ("off", "on", "selective"))],
    "3000": [(3000, 32, 8, 16, ("off", "on", "selective"))],
    "texte": [(3000, 128, 16, 16, ("on", "selective"))],
    "l0": [(2000, 32, 8, 16, ("off_keep",)), (3000, 32, 8, 16, ("off_keep",))],
}


def load() -> dict:
    if OUT.exists():
        return json.loads(OUT.read_text(encoding="utf-8"))
    return {"session": "G", "mesures": []}


def save(payload: dict) -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def run_point(edges: int, seq: int, batch: int, tbptt: int, mode: str, steps: int) -> dict:
    command = [
        sys.executable, str(PROFILE), "one",
        "--nodes", "532", "--edges", str(edges),
        "--seq", str(seq), "--batch", str(batch), "--tbptt", str(tbptt),
        "--steps", str(steps), "--no-growth", "--device", "cuda",
        *MODES[mode],
    ]
    started = time.time()
    proc = subprocess.run(command, capture_output=True, text=True, cwd=str(ROOT))
    payload = None
    for line in reversed(proc.stdout.strip().splitlines()):
        if line.startswith("JSON:"):
            payload = json.loads(line[5:])
            break
    row: dict = {
        "edges": edges, "seq": seq, "batch": batch, "tbptt": tbptt, "mode": mode,
        "steps": steps, "wall_s": round(time.time() - started, 1),
    }
    if payload is None:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-4:]
        row.update({"oom": "out of memory" in (proc.stderr or "").lower(), "error": " | ".join(tail)[:400]})
        return row
    row.update(
        {
            "ms_per_step": payload["ms_per_step"],
            "peak_mib": payload["peak_mib"],
            "baseline_allocated_mib": payload.get("baseline_allocated_mib"),
            "gpu_used_mib_at_start": (payload.get("gpu") or {}).get("used_mib"),
            "losses": [s["loss"] for s in payload.get("steps", [])],
        }
    )
    return row


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="session G: three modes, same graph")
    parser.add_argument("groups", nargs="+", choices=sorted(GROUPS))
    parser.add_argument("--steps", type=int, default=2)
    args = parser.parse_args(argv)

    payload = load()
    done = {(m["edges"], m["seq"], m["batch"], m["mode"]) for m in payload["mesures"] if "ms_per_step" in m}
    for group in args.groups:
        for edges, seq, batch, tbptt, modes in GROUPS[group]:
            for mode in modes:
                key = (edges, seq, batch, mode)
                if key in done:
                    print(f"  {group}/{mode} arêtes {edges} : déjà mesuré, ignoré", flush=True)
                    continue
                row = run_point(edges, seq, batch, tbptt, mode, args.steps)
                payload["mesures"].append(row)
                save(payload)
                if row.get("ms_per_step") is None:
                    print(f"  {group}/{mode} arêtes {edges} : ECHEC {row.get('error', '')}", flush=True)
                else:
                    print(
                        f"  {group}/{mode} arêtes {edges} seq {seq}/b{batch} : "
                        f"{row['ms_per_step']:.0f} ms/pas, pic {row['peak_mib']:.1f} Mo, "
                        f"carte {row['gpu_used_mib_at_start']} Mio, {row['wall_s']:.0f} s",
                        flush=True,
                    )


if __name__ == "__main__":
    main()
