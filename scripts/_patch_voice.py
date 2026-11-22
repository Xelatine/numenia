#!/usr/bin/env python3
from pathlib import Path
p = Path("scripts/webchat.py")
s = p.read_text(encoding="utf-8", errors="replace")

# 1. Rendre l'import non fatal
old_import = "from mycelia.voice_mycelia import respond\n"
if old_import not in s:
    raise SystemExit("import voice introuvable")
new_import = ("try:\n    from mycelia.voice_mycelia import respond\n"
              "    VOICE_READY = True\nexcept Exception as _e:\n"
              "    respond = None\n    VOICE_READY = False\n")
s = s.replace(old_import, new_import, 1)

# 2. Protéger /chat quand respond est None (si la route existe)
head = "\ndef chat():\n    \"\"\"Genere une reponse de Nyra via la voix locale MYCELIA.\"\"\"\n"
i = s.find(head)
if i != -1:
    j = s.find("\r\n\r\n@app.route", i)
    if j == -1:
        j = s.find("\n@app.route", i)
    if j != -1:
        block = s[i:j]
        if "respond(user_prompt)" in block:
            s = s[:i] + (
                "\ndef chat():\n"
                "    \"\"\"Genere une reponse de Nyra via la voix locale MYCELIA.\"\"\"\n"
                "    if not VOICE_READY or respond is None:\n"
                "        return jsonify({\"erreur\": \"voix locale MYCELIA indisponible\", \"succes\": False}), 503\n"
                + block[block.find("\n    data = request.get_json"):]
            ) + s[j:]

p.write_text(s, encoding="utf-8")
print("patched voice import; respond guard inserted:", "if not VOICE_READY" in s)
