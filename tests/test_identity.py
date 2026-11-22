"""L'identité : offline, mesurée, stable à travers sauvegarde et compaction."""

from __future__ import annotations

import shutil
from datetime import datetime, timedelta, timezone

from mycelia.identity import Identity, speak_offline


def test_birth_and_reload_offline(tmp_path) -> None:
    identity = Identity.birth(tmp_path, name="Numenia", declared_sex="indéterminé",
                              version_genome="run3000")
    assert (tmp_path / "self.json").exists()
    reloaded = Identity.load(tmp_path / "self.json")
    assert reloaded.name == "Numenia"
    assert reloaded.birth_ts == identity.birth_ts
    text = speak_offline(reloaded)
    assert "Numenia" in text and "jours" in text


def test_age_is_a_measure_not_a_number(tmp_path) -> None:
    identity = Identity(name="Numenia")
    born = datetime.fromisoformat(identity.birth_ts)
    later = born + timedelta(days=30)
    age = identity.age(later)
    assert 29.9 < age["humain_jours"] < 30.1


def test_identity_survives_copy(tmp_path) -> None:
    identity = Identity.birth(tmp_path, name="Numenia")
    clone = tmp_path / "compact"
    shutil.copytree(tmp_path, clone)
    assert Identity.load(clone / "self.json").to_dict() == identity.to_dict()


def test_name_never_random() -> None:
    blank = Identity()
    assert "sans-nom" in blank.name
    assert blank.declared_sex == "indéterminé"
