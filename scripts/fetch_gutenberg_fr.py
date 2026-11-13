"""Construire data/corpus_gutenberg.txt depuis le top 100 français de Gutenberg.

Source : API Gutendex (métadonnées Gutenberg), filtre langue=fr, tri par
download_count décroissant. Télécharge le texte UTF-8 de chaque livre, retire
le boilerplate Project Gutenberg (marqueurs START/END), concatène.

Sortie : data/corpus_gutenberg.txt + journal ligne par livre sur stdout.
Aucun entraînement, aucune inférence.

Usage : python scripts/fetch_gutenberg_fr.py [--limit 100]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "corpus_gutenberg.txt"
API = "https://gutendex.com/books/?languages=fr&sort=popular"
UA = {"User-Agent": "Mozilla/5.0 (mycelia-corpus-fetch)"}

START = re.compile(
    r"\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*",
    re.S | re.I,
)
END = re.compile(
    r"\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*",
    re.S | re.I,
)


def fetch(url: str, timeout: int = 120, retries: int = 3) -> bytes | None:
    for attempt in range(1, retries + 1):
        try:
            request = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except Exception as error:  # noqa: BLE001 - on journalise et on réessaie
            print(f"  retry {attempt}/{retries} {url} : {error}", flush=True)
            time.sleep(2 * attempt)
    return None


def catalogue(max_books: int = 128) -> list[dict]:
    books: list[dict] = []
    url: str | None = API
    while url and len(books) < max_books:
        raw = fetch(url)
        if raw is None:
            break
        page = json.loads(raw)
        books.extend(page["results"])
        url = page.get("next")
        print(f"catalogue : {len(books)} livres ({url or 'fin'})", flush=True)
    return books


def strip_boilerplate(text: str) -> tuple[str, bool]:
    start = START.search(text)
    if start:
        text = text[start.end() :]
    end = END.search(text)
    if end:
        text = text[: end.start()]
    return text.strip(), bool(start and end)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="fetch the French Gutenberg top")
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args(argv)

    picked: list[dict] = []
    seen: set[int] = set()
    for book in catalogue():
        url = book.get("formats", {}).get("text/plain; charset=utf-8")
        if not url or book["id"] in seen:
            continue
        seen.add(book["id"])
        picked.append(book)
        if len(picked) >= args.limit:
            break
    print(f"livres retenus : {len(picked)}", flush=True)

    total = 0
    with OUT.open("w", encoding="utf-8", newline="\n") as output:
        for index, book in enumerate(picked, 1):
            url = book["formats"]["text/plain; charset=utf-8"]
            raw = fetch(url)
            if raw is None:
                print(f"{index:3d} ECHEC {book['id']} {book['title'][:60]}", flush=True)
                continue
            body, had_markers = strip_boilerplate(raw.decode("utf-8", "replace"))
            if len(body) < 1000:
                print(f"{index:3d} IGNORE (court) {book['id']} {book['title'][:50]}", flush=True)
                continue
            output.write(body)
            output.write("\n\n")
            total += len(body)
            print(
                f"{index:3d} ok {book['id']} dl={book.get('download_count', 0)}"
                f" chars={len(body)} markers={int(had_markers)}"
                f" | {book['title'][:60]}",
                flush=True,
            )

    size = OUT.stat().st_size
    print(f"FIN | {len(picked)} livres | {total} caracteres | {size} octets", flush=True)
    sys.exit(0 if size > 0 else 1)


if __name__ == "__main__":
    main()
