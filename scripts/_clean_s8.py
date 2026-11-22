#!/usr/bin/env python3
"""Nettoie le bloc _s8_predict_tag : supprime les lignes orphelines apres return rows."""
from pathlib import Path
p = Path("scripts/webchat.py")
lines = p.read_text(encoding="utf-8").splitlines(keepends=True)
out = []
i = 0
n = len(lines)
while i < n:
    line = lines[i]
    out.append(line)
    if line.startswith("    return rows\r\n") or line.startswith("    return rows\n"):
        # saute les lignes vides puis le return None orphelin jusqu'au prochain def/début de fonction
        i += 1
        while i < n and lines[i].strip() == "":
            i += 1
        if i < n and lines[i].lstrip().startswith("return None"):
            i += 1
            while i < n and lines[i].strip() == "":
                i += 1
        continue
    i += 1
p.write_text("".join(out), encoding="utf-8")
print("cleaned lines up to:", len(out))
