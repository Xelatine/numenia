"""La hiérarchie d'escalade : stubbée sans détecteur, jamais d'appel réel."""

from __future__ import annotations

from mycelia.organs.external import ExternalOrgans


def test_level3_blocked_without_detector() -> None:
    calls = []
    organs = ExternalOrgans(gemini_client=lambda q: calls.append(q) or "réponse")
    result = organs.escalate("quelle heure est-il à Tokyo ?")
    assert result.blocked is True
    assert result.level == 3
    assert "détecteur d'ignorance" in result.reason
    assert calls == []  # aucun appel réel, jamais


def test_level1_memory_answers_first() -> None:
    class Hit:
        confidence = 0.9
        answer = "souvenir local"

    organs = ExternalOrgans(memory_retrieve=lambda q: Hit())
    result = organs.escalate("question")
    assert result.level == 1 and result.answer == "souvenir local"
    assert result.external_called is False


def test_detector_enables_level3_with_sanitized_question() -> None:
    seen = []
    organs = ExternalOrgans(
        gemini_client=lambda q: seen.append(q) or "réponse externe",
        ignorance_detector=lambda: True,
    )
    result = organs.escalate("  quelle   est la capitale du Japon ?  ", memory_context="secret local")
    assert result.level == 3 and result.external_called is True
    assert seen == ["quelle est la capitale du Japon ?"]  # contexte local absent


def test_web_result_is_an_input_never_raw_answer() -> None:
    organs = ExternalOrgans(
        web_search=lambda q: "fait trouvé",
        ignorance_detector=lambda: True,
    )
    result = organs.escalate("actualité récente")
    assert result.level == 4
    assert "input" in result.reason
