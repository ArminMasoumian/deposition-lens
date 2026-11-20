"""Source-linked date references, not verified historic events or legal findings.

Keep full words and date-specific context. Avoid duplicate partial dates from
OCR and never infer a missing year from a nearby, unrelated deposition.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from .parser import Transcript
from .verifier import Citation

MONTHS = 'January February March April May June July August September October November December'.split()
_MONTH_RE = '|'.join(MONTHS)
# Prefer the longest date: 28 October 1998 / October 28, 1998 / February 1999.
DATE_RE = re.compile(
    rf'\b(?:{_MONTH_RE})\s+\d{{1,2}}(?:st|nd|rd|th)?(?:,?\s+(?:19|20)\d{{2}})?\b'
    rf'|\b(?:{_MONTH_RE})\s+(?:19|20)\d{{2}}\b'
    rf'|\b(?:19|20)\d{{2}}\b', re.I)
YEAR_RE = re.compile(r'\b(?:19|20)\d{2}\b')
TIME_RE = re.compile(r'\b\d{1,2}:\d{2}\s*(?:a\.m\.|p\.m\.|am|pm)(?=\s|$)', re.I)

@dataclass
class TimelineEvent:
    when: str
    text: str
    citation: Citation
    sort_key: tuple = (9999, 12, 32)


def _date_key(date: str) -> tuple:
    yr = YEAR_RE.search(date)
    month = next((i for i, m in enumerate(MONTHS, 1) if m.lower() in date.lower()), 12)
    day = re.search(rf'\b(?:{_MONTH_RE})\s+(\d{{1,2}})\b', date, re.I)
    return (int(yr.group()) if yr else 9999, month, int(day.group(1)) if day else 32)


def _sentence_context(text: str, start: int, end: int) -> str:
    """Choose a complete nearby clause; don't slice in the middle of words."""
    left = max(text.rfind('. ', 0, start), text.rfind('? ', 0, start), text.rfind('! ', 0, start))
    left = left + 2 if left >= 0 else 0
    right_candidates = [n + 1 for pattern in ('. ', '? ', '! ') if (n := text.find(pattern, end)) >= 0]
    right = min(right_candidates) if right_candidates else len(text)
    fragment = ' '.join(text[left:right].split())
    # Long OCR paragraphs: crop by full words around the date, marking truncation.
    if len(fragment) > 285:
        local_start = max(0, start - left - 105)
        local_end = min(len(fragment), end - left + 145)
        # For OCR weird whitespace, locate occurrence again inside normalized text.
        date_text = ' '.join(text[start:end].split())
        pos = fragment.lower().find(date_text.lower())
        if pos >= 0:
            local_start = max(0, pos - 105)
            local_end = min(len(fragment), pos + len(date_text) + 145)
        if local_start:
            gap = fragment.find(' ', local_start)
            local_start = gap + 1 if gap >= 0 else 0
        if local_end < len(fragment):
            gap = fragment.rfind(' ', 0, local_end)
            local_end = gap if gap > local_start else len(fragment)
        fragment = ('… ' if local_start else '') + fragment[local_start:local_end].strip() + (' …' if local_end < len(fragment) else '')
    return fragment


def _useful(context: str, date: str) -> bool:
    words = re.findall(r'\b[a-zA-Z]{3,}\b', context)
    return len(words) >= 5 and len(context) >= len(date) + 22


def build_timeline(tr: Transcript) -> list[TimelineEvent]:
    events = []
    seen = set()
    for ex in tr.exchanges:
        if not ex.answer or not ex.answer_text.strip() or len(ex.answer_text) > 1100:
            continue
        # OCR can embed the following examiner question in the witness answer.
        text = re.split(r'(?<![A-Za-z])(?:\.Q\.|Q[•])\s*', ex.answer_text, maxsplit=1)[0]
        matches = sorted([*DATE_RE.finditer(text), *TIME_RE.finditer(text)], key=lambda m: m.start())
        for match in matches:
            when = match.group().strip(' ,')
            # Year embedded in an already matched full month/day/year date.
            if when.isdigit() and any(m.start() < match.start() < m.end() for m in matches):
                continue
            # Avoid an OCR duplicate such as 'October 28 -- ... October 28, 1998'
            # in the same short passage. Keep only the more complete date.
            if not YEAR_RE.search(when):
                same_prefix = re.sub(r'\s+', ' ', when.lower()).strip()
                if any(m.start() != match.start() and abs(m.start() - match.start()) <= 100
                       and YEAR_RE.search(m.group()) and m.group().lower().startswith(same_prefix)
                       for m in matches):
                    continue
            ctx = _sentence_context(text, match.start(), match.end())
            if not _useful(ctx, when):
                continue
            # A bare year repeated in a nearby full date is not its own event.
            if when.isdigit() and any(m.start() != match.start() and abs(m.start() - match.start()) <= 55
                                      and YEAR_RE.search(m.group()) and len(m.group()) > 4
                                      for m in matches):
                continue
            # Find the physical source line rather than inferring from a character offset.
            source_line = next((ln for ln in ex.answer if when.lower() in ln.text.lower()), None)
            if source_line is None:
                # A date can straddle two physical OCR lines; use the containing
                # answer as a fallback and label it by its source page.
                span = ex.answer_span()
                if span is None: continue
                citation = Citation(*span)
            else:
                citation = Citation(source_line.page, source_line.number, source_line.number)
            # Same event and date is reported once, including duplicate OCR lines.
            key = (when.lower(), citation.page, ctx.casefold())
            if key in seen: continue
            seen.add(key)
            events.append(TimelineEvent(when, ctx, citation, _date_key(when)))
    return sorted(events, key=lambda e: (e.sort_key, e.citation.page, e.citation.line_start))
