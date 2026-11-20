"""Genere les donnees du labo : corpus jouet (1000 phrases) et 500 stimuli FR.

Tout est deterministe (graine fixe) : relancer ce script redonne exactement les
memes fichiers, donc les mesures du rapport sont reproductibles. Les listes de
mots sont volontairement courtes — un vocabulaire de quelques centaines de mots
suffit pour une tache de prediction du mot suivant et garde l'entrainement en
circuit court.
"""

from __future__ import annotations

import random

from lab_common import CORPUS_TOY, LAB, SEED, STIMULI, STIMULI_NEW, ensure_lab, write_json

DET = ["le", "la", "les", "un", "une", "mon", "ma", "son", "sa", "ce", "cette"]
PRON = ["je", "tu", "il", "elle", "nous", "vous", "ils", "on"]
VERB = [
    "mange", "dort", "court", "parle", "regarde", "écoute", "joue", "chante",
    "travaille", "arrive", "part", "tombe", "monte", "descend", "ouvre", "ferme",
    "cherche", "trouve", "donne", "prend", "lit", "écrit", "dessine", "compte",
    "marche", "saute", "nage", "vole", "crie", "rit", "pleure", "attend",
    "répond", "demande", "pense", "crois", "sait", "veut", "peut", "doit",
    "aime", "déteste", "voit", "entend", "sent", "goûte", "touche", "range",
    "lave", "cuit",
]
NOUN = [
    "chat", "chien", "oiseau", "poisson", "cheval", "souris", "lapin", "vache",
    "poule", "loup", "renard", "ours", "singe", "canard", "grenouille",
    "fille", "garçon", "femme", "homme", "enfant", "mère", "père", "frère",
    "sœur", "ami", "voisine", "boulanger", "maîtresse", "médecin", "facteur",
    "table", "chaise", "livre", "porte", "fenêtre", "maison", "jardin", "école",
    "route", "voiture", "vélo", "ballon", "gâteau", "pain", "pomme", "fleur",
    "arbre", "rivière", "montagne", "soleil", "lune", "étoile", "pluie", "neige",
    "vent", "mer", "forêt", "ville", "village", "chambre",
]
ADJ = [
    "petit", "grand", "beau", "joli", "content", "triste", "fatigué", "rapide",
    "lent", "chaud", "froid", "doux", "fort", "calme", "noir", "blanc", "rouge",
    "vert", "bleu", "jaune", "vieux", "jeune", "bon", "mauvais", "propre",
    "sale", "plein", "vide", "lourd", "léger", "heureux", "inquiet", "curieux",
    "gentil", "méchant",
]
ADV = [
    "vite", "lentement", "bien", "mal", "beaucoup", "peu", "toujours", "jamais",
    "souvent", "ici", "là", "maintenant", "demain", "hier", "encore", "aussi",
    "presque", "trop", "assez", "ensemble",
]
QW = ["pourquoi", "comment", "quand", "où", "qui", "combien", "quel", "quelle"]
PREP = ["dans", "sur", "sous", "avec", "sans", "pour", "vers", "chez", "près"]
CONJ = ["et", "ou", "mais", "car", "donc"]
NEG = ["ne", "pas", "rien", "jamais", "plus"]
NUM = ["zéro", "un", "deux", "trois", "quatre", "cinq", "six", "sept", "huit",
       "neuf", "dix", "onze", "douze", "treize", "quatorze", "quinze", "seize",
       "vingt", "trente", "cent", "mille"]
EMO = ["peur", "joie", "colère", "tristesse", "surprise", "faim", "soif",
       "sommeil", "mal", "espoir", "honte", "fierté"]

#: Chaque nom appartient a un groupe, et chaque groupe a ses verbes. Sans cette
#: structure, un corpus de 1000 phrases tirees au hasard n'apprend rien : la
#: valence d'un verbe est libre, donc le mot suivant reste uniforme sur 370
#: mots et la perte ne descend pas sous ln(370) = 5,91. Avec elle, "le chat"
#: appelle "mange" et la tache a une reponse.
GROUPS = {
    "animal": [
        "chat", "chien", "oiseau", "poisson", "cheval", "souris", "lapin",
        "vache", "poule", "loup", "renard", "ours", "singe", "canard", "grenouille",
    ],
    "personne": [
        "fille", "garçon", "femme", "homme", "enfant", "mère", "père", "frère",
        "sœur", "ami", "voisine", "boulanger", "maîtresse", "médecin", "facteur",
    ],
    "chose": [
        "table", "chaise", "livre", "porte", "fenêtre", "maison", "jardin", "école",
        "route", "voiture", "vélo", "ballon", "gâteau", "pain", "pomme", "fleur",
        "arbre", "rivière", "montagne", "soleil", "lune", "étoile", "pluie", "neige",
        "vent", "mer", "forêt", "ville", "village", "chambre",
    ],
}
VERBS_BY_GROUP = {
    "animal": ["mange", "dort", "court", "saute", "nage", "vole", "crie", "attend"],
    "personne": ["parle", "regarde", "écoute", "travaille", "arrive", "part", "lit", "écrit", "dessine", "compte", "rit", "attend"],
    "chose": ["tombe", "reste", "brille", "s'ouvre", "se ferme", "pousse", "coule", "tourne"],
}
ADJ_BY_GROUP = {
    "animal": ["petit", "grand", "rapide", "lent", "doux", "fort", "jeune", "vieux"],
    "personne": ["content", "triste", "fatigué", "calme", "heureux", "gentil", "curieux", "inquiet"],
    "chose": ["joli", "beau", "chaud", "froid", "propre", "sale", "lourd", "léger"],
}
GROUPE_DE = {word: group for group, words in GROUPS.items() for word in words}


def pick_verb(noun: str, rng: random.Random) -> str:
    return rng.choice(VERBS_BY_GROUP[GROUPE_DE[noun]])


def pick_adj(noun: str, rng: random.Random) -> str:
    return rng.choice(ADJ_BY_GROUP[GROUPE_DE[noun]])


SPECIALS = ["<pad>", "<unk>"]


def build_sentences(count: int = 1000, seed: int = SEED) -> list[str]:
    """Fabrique un corpus de phrases simples, sans doublon."""
    rng = random.Random(seed)
    seen: set[str] = set()
    out: list[str] = []

    def add(text: str) -> None:
        text = " ".join(text.split())
        if text in seen:
            return
        seen.add(text)
        out.append(text)

    while len(out) < count:
        kind = rng.randrange(12)
        det = rng.choice(DET)
        noun = rng.choice(NOUN)
        verb = pick_verb(noun, rng)
        if kind == 0:
            add(f"{det} {noun} {verb}.")
        elif kind == 1:
            other = rng.choice(NOUN)
            add(f"{det} {noun} {verb} {rng.choice(DET)} {other}.")
        elif kind == 2:
            add(f"{det} {noun} {pick_adj(noun, rng)} {verb}.")
        elif kind == 3:
            add(f"je ne {verb} pas.")
        elif kind == 4:
            add(f"je ne {verb} pas {rng.choice(DET)} {rng.choice(NOUN)}.")
        elif kind == 5:
            add(f"{rng.choice(QW)} {verb} {det} {noun} ?")
        elif kind == 6:
            add(f"{det} {noun} est {pick_adj(noun, rng)}.")
        elif kind == 7:
            add(f"il y a {rng.choice(NUM)} {noun}.")
        elif kind == 8:
            add(f"{noun} et {noun} {verb} {rng.choice(ADV)}.")
        elif kind == 9:
            add(f"je suis {rng.choice(ADJ)}.")
        elif kind == 10:
            add(f"tu {verb} {det} {noun}.")
        else:
            add(f"{det} {noun} {verb} {rng.choice(PREP)} {det} {rng.choice(NOUN)}.")
    return out


def build_stimuli(seed: int = SEED) -> list[tuple[str, str]]:
    """500 stimuli etiquetes par categorie (phrase, question, negation, ...)."""
    rng = random.Random(seed + 1)
    plan = [
        ("phrase", 120),
        ("question", 90),
        ("negation", 80),
        ("mot", 70),
        ("emotion", 70),
        ("nombre", 70),
    ]
    seen: set[str] = set()
    rows: list[tuple[str, str]] = []

    def add(cat: str, text: str) -> bool:
        text = " ".join(text.split())
        if text in seen:
            return False
        seen.add(text)
        rows.append((cat, text))
        return True

    def make(cat: str) -> str:
        det = rng.choice(DET)
        noun = rng.choice(NOUN)
        verb = pick_verb(noun, rng)
        if cat == "phrase":
            return rng.choice(
                [
                    f"{det} {noun} {verb}.",
                    f"{det} {noun} {pick_adj(noun, rng)} {verb}.",
                    f"{det} {noun} {verb} {rng.choice(DET)} {rng.choice(NOUN)}.",
                    f"{rng.choice(PRON)} {verb} {rng.choice(ADV)}.",
                    f"{det} {noun} est {pick_adj(noun, rng)}.",
                ]
            )
        if cat == "question":
            return rng.choice(
                [
                    f"est-ce que {det} {noun} {verb} ?",
                    f"{rng.choice(QW)} {verb} {det} {noun} ?",
                    f"{rng.choice(QW)} est {det} {noun} ?",
                    f"où est {det} {noun} ?",
                ]
            )
        if cat == "negation":
            return rng.choice(
                [
                    f"je ne {verb} pas.",
                    f"je ne {verb} pas {det} {noun}.",
                    f"il n'y a pas {det} {noun}.",
                    f"{rng.choice(PRON)} ne {verb} jamais {det} {noun}.",
                ]
            )
        if cat == "mot":
            return rng.choice([noun, verb, rng.choice(ADJ), rng.choice(ADV)])
        if cat == "emotion":
            return rng.choice(
                [
                    f"j'ai {rng.choice(EMO)}.",
                    f"je suis {rng.choice(ADJ)}.",
                    f"{rng.choice(PRON)} a peur {rng.choice(PREP)} {rng.choice(NOUN)}.",
                    f"je me sens {rng.choice(ADJ)} aujourd'hui.",
                ]
            )
        return rng.choice(
            [
                f"il y a {rng.choice(NUM)} {noun}.",
                f"j'ai {rng.choice(NUM)} ans.",
                f"{rng.choice(NUM)} plus {rng.choice(NUM)} font {rng.choice(NUM)}.",
                f"{det} {noun} a {rng.choice(NUM)} pattes.",
            ]
        )

    for cat, need in plan:
        made = 0
        guard = 0
        while made < need and guard < 20000:
            guard += 1
            if add(cat, make(cat)):
                made += 1
        if made < need:
            raise RuntimeError(f"categorie {cat} : {made}/{need} stimuli uniques")
    return rows


def build_new_stimuli(corpus: list[str], seen: list[str], count: int = 20) -> list[str]:
    """20 stimuli hors corpus, pour le test de langue (S3)."""
    rng = random.Random(SEED + 2)
    known = set(corpus) | set(seen)
    out: list[str] = []
    guard = 0
    while len(out) < count and guard < 5000:
        guard += 1
        det, noun = rng.choice(DET), rng.choice(NOUN)
        verb = pick_verb(noun, rng)
        text = rng.choice(
            [
                f"{det} {noun} {verb} {rng.choice(PREP)} {rng.choice(DET)} {rng.choice(NOUN)}.",
                f"{rng.choice(PRON)} ne {verb} pas {det} {noun}.",
                f"{rng.choice(QW)} est {det} {noun} {rng.choice(ADJ)} ?",
                f"j'ai vu {det} {noun} {rng.choice(ADJ)}.",
                f"{det} {noun} {verb} et {det} {noun} {verb}.",
            ]
        )
        if text in known:
            continue
        known.add(text)
        out.append(text)
    return out


def vocab_from(sentences: list[str]) -> list[str]:
    words = sorted({word for line in sentences for word in line.split()})
    return SPECIALS + words


def main() -> None:
    ensure_lab()
    corpus = build_sentences(1000)
    CORPUS_TOY.write_text("\n".join(corpus) + "\n", encoding="utf-8")

    rows = build_stimuli()
    STIMULI.write_text(
        "\n".join(f"{cat}|{text}" for cat, text in rows) + "\n", encoding="utf-8"
    )

    fresh = build_new_stimuli(corpus, [text for _, text in rows], 20)
    STIMULI_NEW.write_text("\n".join(fresh) + "\n", encoding="utf-8")

    vocab = vocab_from(corpus)
    write_json(
        LAB / "vocab.json",
        {"vocab": vocab, "size": len(vocab), "corpus": len(corpus), "stimuli": len(rows)},
    )
    print(f"corpus_toy.txt      : {len(corpus)} phrases")
    print(f"stimuli.txt         : {len(rows)} stimuli")
    print(f"stimuli_new.txt     : {len(fresh)} phrases hors corpus")
    print(f"vocabulaire         : {len(vocab)} mots (dont {len(SPECIALS)} speciaux)")
    counts: dict[str, int] = {}
    for cat, _ in rows:
        counts[cat] = counts.get(cat, 0) + 1
    print(f"categories          : {counts}")


if __name__ == "__main__":
    main()
