"""S3-bis D1 — les planchers n-gram du corpus jouet.

La question que ce script tranche avant toute autre : NUMENIA plafonne a ~5,1 de
perte. Est-ce un *plafond d'apprentissage*, ou simplement le niveau qu'atteint
un modele qui ne regarde pas le contexte ? Un modele qui predit chaque mot sans
regarder ce qui precede ne peut pas descendre sous l'entropie **unigramme** du
corpus. Si cette borne vaut deja 5,0-5,1, alors NUMENIA n'a pas de plafond
structurel : il predit exactement comme un sac de mots, et le probleme est
l'usage du contexte (bug de routage/agregation/lecture), pas la capacite.

Trois bornes, en nats par token, sur ``data/lab/corpus_toy.txt`` :

* **unigramme** — p(w), aucune memoire ;
* **bigramme** — p(w | mot precedent) ;
* **trigramme** — p(w | deux mots precedents).

Lissage documente (le meme aux trois ordres) : un **prior de Dirichlet
hierarchique** d'ordre inferieur. Pour un contexte ``h`` de compte total
``N_h`` et un prior ``p_prior`` (l'unigramme pour le bigramme, le bigramme pour
le trigramme) :

    p(w | h) = (c(h, w) + alpha * N_h * p_prior(w)) / (N_h * (1 + alpha))

avec ``alpha = 0.5``. Pourquoi ce lissage et pas un add-1 : avec 312 mots et
1000 phrases, un add-1 uniforme ajoute 312 pseudo-occurrences a chaque contexte,
ce qui ecrase les contextes rares et *surestime* la perte du bigramme (le
plancher deviendrait trop haut, et on accuserait NUMENIA a tort). Le prior
d'ordre inferieur ne rajoute que la masse d'un demi-contexte virtuel, et il est
le seul choix qui rende les trois ordres comparables : chacun est un modele
"ordre n + repli sur ordre n-1".

On rapporte aussi l'entropie **empirique** (MLE, sans lissage) : c'est la
quantite d'information reellement contenue dans le corpus, independante du
lissage. Les deux chiffres sont utiles : la MLE dit ce que le corpus contient,
la lissee dit ce qu'un modele a cette taille peut esperer.

Ecrit ``data/lab/s3bis_floors.json`` et imprime le verdict D1.
"""

from __future__ import annotations

import math

from lab_common import CORPUS_TOY, LAB, read_json, write_json

ALPHA = 0.5
BOS = "<s>"


def corpus_tokens() -> tuple[list[str], list[list[str]]]:
    """Le corpus en phrases, et le flux de tokens (ponctuation comprise)."""
    lines = [
        line.strip()
        for line in CORPUS_TOY.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return lines, [line.split() for line in lines]


def vocab_size() -> int:
    meta = read_json(LAB / "vocab.json")
    return int(meta["size"])


def unigram_floor(tokens: list[str], vocab: int, alpha: float = ALPHA) -> dict:
    """Entropie unigramme : MLE (le contenu du corpus) et lissee (la borne)."""
    counts: dict[str, int] = {}
    for word in tokens:
        counts[word] = counts.get(word, 0) + 1
    total = sum(counts.values())
    mle = -sum((c / total) * math.log(c / total) for c in counts.values())
    # Chaque token du corpus, lu par le modele lisse.
    penalised = 0.0
    for word, count in counts.items():
        p = (count + alpha) / (total + alpha * vocab)
        penalised -= count * math.log(p)
    return {
        "tokens": total,
        "types": len(counts),
        "entropie_mle": round(mle, 6),
        "entropie_lissee": round(penalised / total, 6),
        "alpha": alpha,
    }


def _hierarchical(counts: dict[tuple, int], prior, contexts: int, vocab: int,
                  alpha: float = ALPHA) -> tuple[float, float]:
    """Perte lissee et entropie MLE d'un ordre n, prior d'ordre n-1.

    ``counts`` associe ``(contexte, mot)`` a son compte ; ``prior(mot,
    contexte)`` rend la probabilite de repli. On rend la perte moyenne en nats
    par token, et l'entropie conditionnelle empirique (MLE).
    """
    by_context: dict[tuple, int] = {}
    for (context, _word), count in counts.items():
        by_context[context] = by_context.get(context, 0) + count
    token_total = sum(counts.values())
    cross = 0.0
    mle = 0.0
    for (context, word), count in counts.items():
        n_h = by_context[context]
        p = (count + alpha * n_h * prior(word, context)) / (n_h * (1.0 + alpha))
        cross -= count * math.log(max(p, 1e-12))
        mle -= count * math.log(count / n_h)
    return cross / token_total, mle / token_total


def ngram_floors(sentences: list[list[str]], vocab: int, alpha: float = ALPHA) -> dict:
    """Bigramme et trigramme, avec repli hierarchique documente."""
    uni_counts: dict[str, int] = {}
    for sentence in sentences:
        for word in sentence:
            uni_counts[word] = uni_counts.get(word, 0) + 1
    uni_total = sum(uni_counts.values())

    def p_uni(word: str, _context=None) -> float:
        return (uni_counts.get(word, 0) + alpha) / (uni_total + alpha * vocab)

    bigram: dict[tuple, int] = {}
    trigram: dict[tuple, int] = {}
    for sentence in sentences:
        padded = [BOS, BOS] + list(sentence)
        for i in range(2, len(padded)):
            word = padded[i]
            h1, h2 = padded[i - 1], padded[i - 2]
            bigram[(h1, word)] = bigram.get((h1, word), 0) + 1
            trigram[(h2, h1, word)] = trigram.get((h2, h1, word), 0) + 1

    bi_by_context: dict[tuple, int] = {}
    for (context, _word), count in bigram.items():
        bi_by_context[context] = bi_by_context.get(context, 0) + count

    def p_bi(word: str, context) -> float:
        h1 = context[0]
        n_h = bi_by_context.get((h1,), 0)
        c = bigram.get((h1, word), 0)
        prior = p_uni(word)
        if n_h == 0:
            return prior
        return (c + alpha * n_h * prior) / (n_h * (1.0 + alpha))

    bi_cross, bi_mle = _hierarchical(bigram, p_uni, len(bi_by_context), vocab, alpha)
    # Les cles du trigramme sont (h2, h1, w) : contexte = (h2, h1).
    tri_keys = {((h2, h1), w): c for (h2, h1, w), c in trigram.items()}
    tri_cross, tri_mle = _hierarchical(tri_keys, p_bi, len(trigram), vocab, alpha)
    return {
        "bigramme": {
            "contextes": len(bi_by_context),
            "paires": len(bigram),
            "entropie_mle": round(bi_mle, 6),
            "entropie_lissee": round(bi_cross, 6),
        },
        "trigramme": {
            "contextes": len(tri_keys),
            "triplets": len(trigram),
            "entropie_mle": round(tri_mle, 6),
            "entropie_lissee": round(tri_cross, 6),
        },
    }


def main() -> None:
    lines, sentences = corpus_tokens()
    vocab = vocab_size()
    tokens = [word for sentence in sentences for word in sentence]
    uni = unigram_floor(tokens, vocab)
    higher = ngram_floors(sentences, vocab)

    uniform = round(math.log(vocab), 6)
    floor = uni["entropie_lissee"]
    numenia = 5.1
    # Le seuil du brief : unigramme dans 5,0-5,1 -> pas de plafond, bug d'usage.
    if 5.0 <= floor <= 5.15:
        verdict = "BUG"
        phrase = (
            "l'unigramme vaut déjà le niveau de NUMENIA : il prédit comme un sac "
            "de mots, ce n'est PAS un plafond d'apprentissage mais un bug d'usage "
            "du contexte"
        )
    elif floor < 4.5:
        verdict = "APPRENTISSAGE"
        phrase = (
            "le plancher sans contexte est bien plus bas que 5,1 : NUMENIA laisse "
            "de l'information contextuelle sur la table, le plafond est un vrai "
            "problème d'apprentissage"
        )
    else:
        verdict = "ZONE"
        phrase = (
            "l'unigramme tombe entre 4,5 et 5,0 : le plafond de NUMENIA n'est ni "
            "expliqué ni absous par le sans-contexte, la zone reste ambiguë"
        )

    payload = {
        "corpus": str(CORPUS_TOY.name),
        "phrases": len(lines),
        "vocabulaire": vocab,
        "uniforme_ln_v": uniform,
        "alpha": ALPHA,
        "unigramme": uni,
        "bigramme": higher["bigramme"],
        "trigramme": higher["trigramme"],
        "L_ref": floor,
        "mycelia_s2": numenia,
        "ecart_mycelia_unigramme": round(numenia - floor, 6),
        "verdict": verdict,
        "phrase": phrase,
    }
    write_json(LAB / "s3bis_floors.json", payload)

    print(f"corpus            : {len(lines)} phrases, {uni['tokens']} tokens, vocab {vocab}")
    print(f"uniforme ln(V)    : {uniform:.4f}")
    print(f"unigramme  MLE    : {uni['entropie_mle']:.4f}   lisse a=0,5 : {floor:.4f}")
    print(
        f"bigramme   MLE    : {higher['bigramme']['entropie_mle']:.4f}   "
        f"lisse : {higher['bigramme']['entropie_lissee']:.4f}"
    )
    print(
        f"trigramme  MLE    : {higher['trigramme']['entropie_mle']:.4f}   "
        f"lisse : {higher['trigramme']['entropie_lissee']:.4f}"
    )
    print(f"NUMENIA S2        : {numenia:.4f}  (ecart a l'unigramme : {numenia - floor:+.4f})")
    print(f"VERDICT D1        : {verdict} — {phrase}")
    print("ecrit : data/lab/s3bis_floors.json")


if __name__ == "__main__":
    main()
