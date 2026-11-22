
import sys
sys.path.insert(0, "scripts")
exec(open("scripts/webchat.py", encoding="utf-8", errors="replace").read().split("if __name__")[0])
print("Routes:", len(app.url_map))
print("S8 routes present:", any("s8" in rule.rule for rule in app.url_map.iter_rules()))
