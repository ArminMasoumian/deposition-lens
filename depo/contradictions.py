"""Very conservative candidate detection for opposite answers to similar questions.

The score is a lexical question similarity, not a probability of contradiction.
"""
from __future__ import annotations
from dataclasses import dataclass
from difflib import SequenceMatcher
from .parser import Exchange, Transcript
from .textutil import content_tokens, normalize
from .verifier import Citation

NEGATIVE_STARTS = ('no', 'never', 'i did not', 'i didn t', 'i do not', 'i don t', 'i was not', 'i wasn t')


def polarity(answer: str) -> int:
    text = normalize(answer)
    if not text:
        return 0
    if text.split()[0] in ('yes', 'yeah', 'correct', 'right'):
        return 1
    if any(text == s or text.startswith(s + ' ') for s in NEGATIVE_STARTS):
        return -1
    return 0

@dataclass
class Contradiction:
    first: Citation
    first_answer: str
    second: Citation
    second_answer: str
    similarity: float
    first_question: str = ''
    second_question: str = ''


def _jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


def _cite(ex: Exchange) -> Citation | None:
    span = ex.full_span()
    return Citation(span[0], span[1], span[2]) if span else None


def find_contradictions(transcript: Transcript, min_similarity: float = .5) -> list[Contradiction]:
    items = [(ex, set(content_tokens(ex.question_text)), polarity(ex.answer_text)) for ex in transcript.exchanges]
    found = []
    for i, (a, ta, pa) in enumerate(items):
        if not ta or not pa:
            continue
        for b, tb, pb in items[i+1:]:
            if not tb or not pb or pa == pb:
                continue
            # Generic one-word overlap is not enough evidence for a contradiction.
            overlap = ta & tb
            if len(overlap) < 2:
                continue
            similarity = 0.5 * _jaccard(ta, tb) + 0.5 * SequenceMatcher(None, normalize(a.question_text), normalize(b.question_text)).ratio()
            if similarity < min_similarity:
                continue
            ca, cb = _cite(a), _cite(b)
            if ca and cb:
                found.append(Contradiction(ca, a.answer_text, cb, b.answer_text, similarity, a.question_text, b.question_text))
    return sorted(found, key=lambda c: -c.similarity)
