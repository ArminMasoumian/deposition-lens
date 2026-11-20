"""Structured, self-contained reports that never suggest legal certification."""
from __future__ import annotations
import json
from datetime import datetime, timezone
from .parser import Transcript
from .verifier import Verdict


def json_report(transcript: Transcript, verdicts: list[Verdict], title: str = 'Analysis') -> str:
    return json.dumps({
        'title': title,
        'generated_utc': datetime.now(timezone.utc).isoformat(),
        'document': {'pages': transcript.page_count, 'numbered_lines': len(transcript.lines), 'exchanges': len(transcript.exchanges)},
        'disclaimer': 'Research prototype. Citation checks do not establish factual truth, legal relevance, or reliability. Attorney review required.',
        'claims': [{
            'statement': v.claim.statement,
            'citation_check': v.status,
            'lexical_support': round(v.support, 3),
            'reason': v.detail,
            'citations': [{'page': c.page, 'start': c.line_start, 'end': c.line_end,
                           'quote': c.quote, 'source_lines': transcript.span_text(c.page, c.line_start, c.line_end)}
                          for c in v.claim.citations],
        } for v in verdicts]
    }, ensure_ascii=False, indent=2)


def markdown_report(transcript: Transcript, verdicts: list[Verdict], title: str = 'Analysis') -> str:
    out = [f'# {title}', '', '> Research prototype. Lexical citation checks do not establish factual or legal correctness. Attorney review required.', '']
    for i, v in enumerate(verdicts, 1):
        out += [f'## {i}. {v.claim.statement}', '', f'**Citation check:** `{v.status}`', '']
        if v.detail:
            out += [f'**Reason:** {v.detail}', '']
        for c in v.claim.citations:
            source = transcript.span_text(c.page, c.line_start, c.line_end)
            out += [f'- **{c.label()}** — {source or "MISSING"}']
        out.append('')
    return '\n'.join(out)
