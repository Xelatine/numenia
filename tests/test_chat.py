"""Le chat : identité offline, mémoire qui se souvient, refus sans réseau."""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from scripts.chat import ChatEngine, load_identity


class FakeEncoder:
    """Vecteur déterministe : deux textes identiques ont le même, sans modèle."""

    dim = 32

    def __call__(self, text: str, kind: str = "doc") -> np.ndarray:
        seed = int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "little")
        vector = np.random.default_rng(seed).standard_normal(self.dim).astype(np.float32)
        return vector / np.linalg.norm(vector)


def fake_generator(question: str) -> str:
    return f"réponse locale à {question[:20]}"


def engine(tmp_path, name="Test") -> ChatEngine:
    from mycelia.identity import Identity

    identity = Identity.birth(tmp_path, name=name, version_genome="myc-test")
    return ChatEngine(
        encoder=FakeEncoder(),
        generator=fake_generator,
        identity=identity,
        memory_path=tmp_path / "memory.npz",
    )


def test_identity_is_offline(tmp_path) -> None:
    chat = engine(tmp_path, name="Soline")
    level, answer = chat.ask("qui es-tu ?")
    assert level == "identite"
    assert "Soline" in answer
    assert "jours" in answer


def test_same_exchange_is_remembered(tmp_path) -> None:
    chat = engine(tmp_path)
    first_level, first_answer = chat.ask("quelle est ma couleur préférée ?")
    assert first_level == "qwen"
    second_level, second_answer = chat.ask("quelle est ma couleur préférée ?")
    assert second_level == "memoire"
    assert second_answer == first_answer
    assert len(chat.memory) == 2  # les deux échanges sont écrits


def test_memory_survives_a_new_process(tmp_path) -> None:
    chat = engine(tmp_path)
    first_level, first_answer = chat.ask("mon chat s'appelle Fermion")
    assert first_level == "qwen"
    reopened = ChatEngine(
        encoder=FakeEncoder(),
        generator=fake_generator,
        identity=chat.identity,
        memory_path=tmp_path / "memory.npz",
    )
    level, answer = reopened.ask("mon chat s'appelle Fermion")
    assert level == "memoire"
    assert answer == first_answer


def test_levels_3_4_refuse_without_network(tmp_path, monkeypatch) -> None:
    import socket

    def forbidden(*_a, **_k):
        raise AssertionError("aucun appel réseau ne doit partir")

    monkeypatch.setattr(socket, "socket", forbidden)
    chat = engine(tmp_path)
    chat.organs.local_model = lambda _q: ""  # force l'escalade au-delà du niveau 2
    escalation = chat.organs.escalate("quelle heure est-il à Tokyo ?")
    assert escalation.blocked is True
    assert escalation.external_called is False
    assert "détecteur" in escalation.reason


def test_empty_question_is_a_clean_refusal(tmp_path) -> None:
    chat = engine(tmp_path)
    level, answer = chat.ask("   ")
    assert level == "refus"
    assert "vide" in answer
    assert len(chat.memory) == 0


def test_load_identity_creates_self_json(tmp_path) -> None:
    identity = load_identity(tmp_path / "genome" / "self.json")
    assert identity.name.startswith("sans-nom")
    assert (tmp_path / "genome" / "self.json").exists()
