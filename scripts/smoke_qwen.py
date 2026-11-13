"""Smoke-test du GGUF Qwen2.5-0.5B-Instruct Q8_0 (à lancer APRÈS le run 3000).

Une seule génération de 20 tokens, prompt fixe, seed fixe, latence mesurée.
Aucune boucle, aucun entraînement : c'est la première parole de l'organisme et
une donnée de dimensionnement (s/token).

Exécution reportée au FIN du run 3 000 pas — le CPU lui appartient jusque-là.

Usage :
    .venv/bin/python scripts/smoke_qwen.py
    .venv/bin/python scripts/smoke_qwen.py --tokens 20 --threads 4
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / "data" / "models" / "qwen2.5-0.5b-instruct-q8_0.gguf"
PROMPT = "Bonjour, je m'appelle Mycelia. "
EXPECTED_MD5 = "a49598f88d6157e900ae2aac730873c9"


def md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="first utterance of the organism")
    parser.add_argument("--tokens", type=int, default=20)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--context", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    if not MODEL.exists():
        print(f"modèle absent : {MODEL}")
        return 1
    actual = md5(MODEL)
    print(f"modele {MODEL.name} | {MODEL.stat().st_size} o | md5 {actual}")
    if actual != EXPECTED_MD5:
        print(f"ATTENTION md5 inattendu (attendu {EXPECTED_MD5})")
        return 2

    try:
        from llama_cpp import Llama
    except ImportError:
        print(
            "llama-cpp-python absent.\n"
            "Installer dans le venv du serveur :\n"
            "  .venv/bin/pip install llama-cpp-python"
        )
        return 3

    started = time.perf_counter()
    llm = Llama(
        model_path=str(MODEL),
        n_ctx=int(args.context),
        n_threads=int(args.threads),
        seed=int(args.seed),
        verbose=False,
    )
    loaded = time.perf_counter() - started
    started = time.perf_counter()
    result = llm(
        PROMPT,
        max_tokens=int(args.tokens),
        temperature=0.0,
        seed=int(args.seed),
        echo=False,
    )
    elapsed = time.perf_counter() - started
    text = result["choices"][0]["text"]
    produced = int(result["usage"]["completion_tokens"])
    print(f"chargement : {loaded:.2f} s")
    print(f"prompt     : {PROMPT!r}")
    print(f"sortie     : {text!r}")
    print(
        f"{produced} tokens en {elapsed:.2f} s"
        f" | {elapsed / max(1, produced):.3f} s/token"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
