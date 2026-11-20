"""Conservative, source-linked UI summaries; no generative model is involved."""
from __future__ import annotations

import re
from dataclasses import dataclass

from .parser import Transcript, Exchange
from .textutil import content_tokens
from .verifier import Citation

# Heuristic labels convey *why selected*, not a legal interpretation.
TOPICS = {
    'Work and experience': ('work', 'employ', 'job', 'research', 'consult', 'expert', 'experience', 'career'),
    'Observations and events': ('saw', 'see', 'observ', 'happen', 'incident', 'accident', 'fell', 'visit'),
    'Documents and evidence': ('record', 'document', 'report', 'photo', 'review', 'research', 'publication'),
    'Dates and chronology': ('when', 'before', 'after', 'time', 'year', 'date', 'first'),
    'Uncertainty and limits': ('remember', 'recall', 'unsure', 'certain', 'know', 'not sure'),
}

@dataclass
class Highlight:
    category: str
    question: str
    answer: str
    citations: list[Citation]


def _snippet(text: str, max_chars: int = 340) -> str:
    cleaned = re.sub(r'\s+', ' ', text).strip()
    # OCR sometimes folds the next examiner's question into an answer.
    cleaned = re.split(r'(?<![A-Za-z])(?:\.Q\.|Q[•])\s*', cleaned, maxsplit=1)[0].strip()
    if len(cleaned) <= max_chars:
        return cleaned
    # Don't pretend the clipped text is the entire quotation.
    cut = cleaned.rfind(' ', 0, max_chars)
    return cleaned[:cut if cut > max_chars // 2 else max_chars].rstrip(' ,;') + '…'


def display_excerpt(text: str, max_chars: int = 340) -> str:
    return _snippet(text, max_chars)


def select_highlights(tr: Transcript, limit: int = 8) -> list[Highlight]:
    candidates = []
    for index, ex in enumerate(tr.exchanges):
        answer = ex.answer_text.strip()
        question = ex.question_text.strip()
        spans = ex.answer_spans()
        if not spans or len(content_tokens(answer)) < 5 or not question:
            continue
        if len(answer) > 1800:  # OCR-mangled answers frequently have multiple exchanges joined
            continue
        tokens = set(content_tokens(question + ' ' + answer))
        categories = [(sum(t in tokens for t in content_tokens(' '.join(keys))), label) for label, keys in TOPICS.items()]
        strength, category = max(categories, key=lambda pair: pair[0])
        # Downrank overly long material, e.g. pages of unrelated case names.
        short_score = min(len(content_tokens(answer)), 34) / 12 - max(0, len(answer) - 500) / 100
        procedural = bool(re.search(r'\b(?:case number|civil action|supreme court|court of|et al|number (?:one|two|three))\b', answer, re.I))
        score = strength * 2 + short_score - (10 if procedural else 0)
        candidates.append((score, index, category, ex, spans))
    candidates.sort(key=lambda x: (-x[0], x[1]))
    chosen = sorted(candidates[:limit], key=lambda x: x[1])
    return [Highlight(category, ex.question_text, ex.answer_text,
                      [Citation(p, a, b, tr.span_text(p, a, b)) for p, a, b in spans])
            for _, _, category, ex, spans in chosen]
