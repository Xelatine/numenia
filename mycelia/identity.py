"""self.json : l'identité du corps, offline par nature.

L'âge est une MESURE (now − birth_ts), jamais un chiffre décrété. Le sexe est
une DÉCLARATION libre, jamais déduite. Le nom vient de l'utilisateur ou de
l'organisme au premier Sommeil — jamais du hasard, jamais du réseau.

Le test qui compte : débrancher le câble et demander « qui es-tu ? » doit
fonctionner. Ce module ne fait aucun appel réseau et ne lit aucune variable
d'environnement distante.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_SEX = "indéterminé"
UNNAMED = "sans-nom (à nommer au premier Sommeil)"


class Identity:
    def __init__(
        self,
        name: str | None = None,
        birth_ts: str | None = None,
        declared_sex: str = DEFAULT_SEX,
        version_genome: str = "",
        cpu_seconds: float = 0.0,
        stable: bool = True,
    ) -> None:
        self.name = name or UNNAMED
        self.birth_ts = birth_ts or datetime.now(timezone.utc).isoformat()
        self.declared_sex = declared_sex or DEFAULT_SEX
        self.version_genome = version_genome
        self.cpu_seconds = float(cpu_seconds)
        self.stable = bool(stable)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "birth_ts": self.birth_ts,
            "declared_sex": self.declared_sex,
            "declared_age_fn": "now() - birth_ts",
            "version_genome": self.version_genome,
            "cpu_seconds": round(self.cpu_seconds, 1),
            "stable": self.stable,
        }

    def age(self, now: datetime | None = None) -> dict:
        now = now or datetime.now(timezone.utc)
        born = datetime.fromisoformat(self.birth_ts)
        if born.tzinfo is None:
            born = born.replace(tzinfo=timezone.utc)
        seconds = max(0.0, (now - born).total_seconds())
        return {
            "humain_jours": round(seconds / 86400.0, 3),
            "cpu_jours": round(self.cpu_seconds / 86400.0, 4),
            "secondes": round(seconds, 1),
        }

    def who_am_i(self, now: datetime | None = None) -> str:
        age = self.age(now)
        return (
            f"Je suis {self.name}. Né le {self.birth_ts[:10]}, "
            f"âge mesuré {age['humain_jours']:.1f} jours humains "
            f"({age['cpu_jours']:.2f} jour CPU). Sexe déclaré : {self.declared_sex}. "
            f"Génome {self.version_genome or 'non versionné'}."
        )

    def save(self, path: str | Path) -> str:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        return str(path)

    @classmethod
    def load(cls, path: str | Path) -> "Identity":
        blob = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            name=blob.get("name"),
            birth_ts=blob.get("birth_ts"),
            declared_sex=blob.get("declared_sex", DEFAULT_SEX),
            version_genome=blob.get("version_genome", ""),
            cpu_seconds=blob.get("cpu_seconds", 0.0),
            stable=blob.get("stable", True),
        )

    @classmethod
    def birth(cls, genome_dir: str | Path, name: str | None = None,
              declared_sex: str = DEFAULT_SEX, version_genome: str = "") -> "Identity":
        identity = cls(name=name, declared_sex=declared_sex, version_genome=version_genome)
        identity.save(Path(genome_dir) / "self.json")
        return identity


def speak_offline(identity: Identity, question: str = "qui es-tu ?") -> str:
    """Répond à la question identitaire sans réseau. Test de débranchement."""
    return identity.who_am_i()
