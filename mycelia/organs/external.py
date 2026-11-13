"""Hiérarchie d'escalade à quatre niveaux — STUBBÉE aux niveaux 3-4.

Niveau 1 : mémoire locale (MYCELIA + souvenirs), toujours d'abord.
Niveau 2 : modèle local (Qwen GGUF), sans sortie réseau.
Niveau 3 : Gemini API — seulement si un détecteur d'ignorance existe.
Niveau 4 : recherche web — même condition, et le résultat devient un INPUT.

La condition d'activation est le golden retrieval 5/5 : sans lui, l'organisme ne
sait pas dire « je ne sais pas », et toute escalade serait aveugle. Les niveaux
3-4 renvoient donc un refus explicite, jamais une erreur bloquante.

Vie privée : le contexte local (souvenirs, génome) n'est JAMAIS transmis. Seule
une question décontextualisée peut sortir, après passage par ``sanitize``.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Escalation:
    level: int
    answer: str | None
    blocked: bool = False
    reason: str = ""
    external_called: bool = False


@dataclass
class ExternalOrgans:
    memory_retrieve: callable | None = None
    local_model: callable | None = None
    gemini_client: callable | None = None
    web_search: callable | None = None
    ignorance_detector: callable | None = None
    confidence_threshold: float = 0.55
    history: list = field(default_factory=list)

    def _has_detector(self) -> bool:
        return self.ignorance_detector is not None and bool(self.ignorance_detector())

    def sanitize(self, question: str) -> str:
        """Aucune mémoire, aucun génome : une question nue, bornée en longueur."""
        return " ".join(str(question).split())[:500]

    def escalate(self, question: str, memory_context: str | None = None) -> Escalation:
        # Niveau 1 : mémoire locale.
        if self.memory_retrieve is not None:
            hits = self.memory_retrieve(question)
            confidence = float(getattr(hits, "confidence", 0.0)) if hits is not None else 0.0
            if hits and confidence >= self.confidence_threshold:
                return self._record(Escalation(1, str(getattr(hits, "answer", hits))))
        # Niveau 2 : modèle local.
        if self.local_model is not None:
            answer = self.local_model(self.sanitize(question))
            if answer:
                return self._record(Escalation(2, str(answer)))
        # Niveau 3 : sortie de la machine — condition stricte.
        if not self._has_detector():
            return self._record(Escalation(
                3, None, blocked=True,
                reason="escalade bloquée : pas de détecteur d'ignorance (golden 5/5 non atteint)",
            ))
        if self.gemini_client is not None:
            answer = self.gemini_client(self.sanitize(question))
            return self._record(Escalation(3, str(answer), external_called=True))
        # Niveau 4 : recherche web, résultat ramené comme input, jamais affiché brut.
        if self.web_search is not None:
            found = self.web_search(self.sanitize(question))
            return self._record(Escalation(
                4, str(found), external_called=True,
                reason="résultat web ramené comme input à mémoriser, pas comme réponse finale",
            ))
        return self._record(Escalation(4, None, blocked=True, reason="aucun organe externe disponible"))

    def _record(self, escalation: Escalation) -> Escalation:
        self.history.append(escalation)
        return escalation
