"""Premier chat avec MYCELIA — identité offline, mémoire, Qwen local.

Chaque ligne est une question. La réponse dit **par quel étage** elle est passée :

    [identite]  self.json, aucun réseau
    [memoire]   retrieval local (EmbeddingGemma, seuil 0,30)
    [qwen]      Qwen2.5-0.5B GGUF local, privé
    [refus]     niveaux 3-4 : aucun détecteur câblé, donc refus explicite

Chaque échange est encodé et écrit dans la mémoire épisodique : la conversation
d'aujourd'hui doit être retrouvable demain par le retrieval. Aucun client réseau
n'est instancié ici.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mycelia.identity import Identity, speak_offline  # noqa: E402
from mycelia.memory import EpisodicMemory  # noqa: E402
from mycelia.organs.external import ExternalOrgans  # noqa: E402

SELF_PATH = ROOT / "data" / "genome_myc" / "self.json"
MEMORY_PATH = ROOT / "data" / "memory_chat.npz"
CONFIDENCE = 0.30  # calibré sur Gemma (connus min 0,322 / inconnus max 0,260)


@dataclass
class RetrievalHit:
    answer: str
    confidence: float


class ChatEngine:
    """Le REPL, sans I/O : testable avec un faux encodeur et un faux générateur."""

    def __init__(
        self,
        encoder,
        generator,
        identity: Identity,
        memory_path: Path | str = MEMORY_PATH,
        confidence: float = CONFIDENCE,
        memory: EpisodicMemory | None = None,
    ) -> None:
        self.encoder = encoder
        self.generator = generator
        self.identity = identity
        self.memory_path = Path(memory_path)
        self.confidence = float(confidence)
        if memory is not None:
            self.memory = memory
        elif self.memory_path.exists():
            self.memory = EpisodicMemory.load(self.memory_path)
        else:
            self.memory = EpisodicMemory(dim=int(getattr(encoder, "dim", 32)))
        self.organs = ExternalOrgans(
            memory_retrieve=self._retrieve,
            local_model=self._local,
            ignorance_detector=None,
            confidence_threshold=self.confidence,
        )

    # -- niveau 1 -------------------------------------------------------
    def _retrieve(self, question: str):
        if len(self.memory) == 0:
            return None
        hits = self.memory.retrieve(self.encoder(question, kind="query"), top_k=1)
        if not hits:
            return None
        index, score = hits[0]
        if score < self.confidence:
            return None
        return RetrievalHit(answer=self._answer_of(self.memory.text[index]), confidence=score)

    # -- niveau 2 -------------------------------------------------------
    def _local(self, question: str) -> str:
        return str(self.generator(question)).strip()

    @staticmethod
    def _answer_of(record: str) -> str:
        marker = "\nR: "
        return record.split(marker, 1)[1] if marker in record else record

    # -- boucle ---------------------------------------------------------
    def ask(self, message: str) -> tuple[str, str]:
        """Renvoie (niveau, réponse). Écrit TOUJOURS l'échange en mémoire."""
        text = " ".join(str(message).split())
        if not text:
            return "refus", "question vide"
        if text.lower().startswith("qui es-tu"):
            return "identite", speak_offline(self.identity)

        escalation = self.organs.escalate(text)
        if escalation.blocked:
            level, answer = "refus", escalation.reason
        elif escalation.level == 1:
            level, answer = "memoire", str(escalation.answer)
        elif escalation.level == 2:
            level, answer = "qwen", str(escalation.answer)
        else:
            level, answer = "refus", escalation.reason or "étage indisponible"

        record = f"Q: {text}\nR: {answer}"
        # L'index porte la QUESTION (c'est elle qu'on retrouvera) ; le texte
        # stocké porte l'échange complet, réponse et niveau compris.
        self.memory.add(
            self.encoder(text, kind="doc"), salience=0.9, text=record
        )
        self.memory.save(self.memory_path)
        return level, answer


def load_identity(path: Path = SELF_PATH) -> Identity:
    if Path(path).exists():
        return Identity.load(path)
    return Identity.birth(Path(path).parent, version_genome="myc-1")


def main() -> int:
    sys.path.insert(0, str(ROOT / "scripts"))
    from gemma_embed import GemmaEncoder
    from llama_cpp import Llama

    encoder = GemmaEncoder()
    llm = Llama(
        model_path=str(ROOT / "data" / "models" / "qwen2.5-0.5b-instruct-q8_0.gguf"),
        n_ctx=1024,
        n_threads=4,
        seed=42,
        verbose=False,
    )

    def generator(question: str) -> str:
        result = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": (
                    "Tu es MYCELIA, un organisme local. Réponds en français, "
                    "brièvement, sans prétendre avoir fait une action."
                )},
                {"role": "user", "content": question},
            ],
            max_tokens=80,
            temperature=0.6,
        )
        return result["choices"][0]["message"]["content"]

    engine = ChatEngine(encoder=encoder, generator=generator, identity=load_identity())
    print("MYCELIA — chat. (ligne vide pour quitter)")
    print(f"(mémoire : {len(engine.memory)} souvenirs | seuil {CONFIDENCE})")
    while True:
        try:
            message = input("toi> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not message:
            break
        level, answer = engine.ask(message)
        print(f"mycelia> [{level}] {answer}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
