"""Produce claims (statements with page:line citations) from a transcript.

Two modes:
  baseline  rule-based extraction, no model needed; each claim cites the answer it came from
  llm       a model writes the claims; the verifier then checks every one of them
"""

from __future__ import annotations

import json

from .llm import LLM
from .parser import Transcript
from .textutil import content_tokens
from .verifier import Citation, Claim

SYSTEM_PROMPT = (
    "You are a litigation support assistant. Use ONLY the transcript provided. "
    "Every claim must cite the exact page and line range that supports it and include a short "
    "verbatim quote (under 25 words) copied from those lines. If the transcript does not say "
    "something, do not state it. Return ONLY a JSON array, no other text, where each item is "
    '{"statement": str, "page": int, "line_start": int, "line_end": int, "quote": str}.'
)


def _informativeness(text: str) -> float:
    """Cheap score: longer answers with numbers and names are more informative."""
    tokens = content_tokens(text)
    words = text.split()
    numbers = sum(1 for w in words if any(ch.isdigit() for ch in w))
    names = sum(1 for w in words[1:] if w[:1].isupper())
    return len(tokens) + 4 * min(numbers, 3) + 2 * min(names, 4)


def baseline_summary(transcript: Transcript, max_claims: int = 12) -> list[Claim]:
    scored = []
    for index, ex in enumerate(transcript.exchanges):
        if len(ex.answer_text.split()) < 6:
            continue
        spans = ex.answer_spans()
        if not spans:
            continue
        scored.append((_informativeness(ex.answer_text), index, ex, spans))
    scored.sort(key=lambda item: (-item[0], item[1]))
    chosen = sorted(scored[:max_claims], key=lambda item: item[1])  # back to transcript order
    return [
        Claim(
            f"Witness testified: {ex.answer_text}",
            [
                Citation(page, start, end, transcript.span_text(page, start, end))
                for page, start, end in spans
            ],
        )
        for _, _, ex, spans in chosen
    ]


def parse_claims_json(text: str) -> list[Claim]:
    """Parse the model's JSON reply into claims. Raises ValueError if it is not usable."""
    start, end = text.find("["), text.rfind("]")
    if start == -1 or end <= start:
        raise ValueError("model reply did not contain a JSON array")
    try:
        items = json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        raise ValueError(f"model reply was not valid JSON: {exc}") from exc

    claims = []
    for item in items:
        if not isinstance(item, dict) or "statement" not in item:
            continue
        cites = item.get("citations")
        if cites is None:
            cites = [item]
        parsed = []
        for cite in cites:
            try:
                page = int(cite["page"])
                line_start = int(cite["line_start"])
                line_end = int(cite.get("line_end", line_start))
            except (KeyError, TypeError, ValueError):
                continue
            quote = cite.get("quote") or None
            parsed.append(Citation(page, line_start, line_end, quote))
        claims.append(Claim(str(item["statement"]), parsed))
    return claims


def llm_summary(transcript: Transcript, llm: LLM, max_claims: int = 12) -> list[Claim]:
    user = (
        f"Transcript (each line is prefixed with its page and line):\n\n{transcript.render()}\n\n"
        f"Summarize the testimony in at most {max_claims} claims."
    )
    return parse_claims_json(llm.complete(SYSTEM_PROMPT, user))


def summarize(transcript: Transcript, llm: LLM | None = None, max_claims: int = 12) -> list[Claim]:
    if llm is None:
        return baseline_summary(transcript, max_claims)
    return llm_summary(transcript, llm, max_claims)

