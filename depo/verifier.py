"""The citation checker: confirm every claim is backed by the lines it cites.

A claim is only "verified" when
  1. every cited page/line exists in the transcript,
  2. any quoted text really appears in the cited lines, and
  3. the claim's content words are mostly present in the cited lines.

The checks are deterministic code, so a model cannot talk its way past them.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from .parser import Transcript
from .textutil import normalize, support

VERIFIED = "verified"
NO_CITATION = "no_citation"
OUT_OF_RANGE = "out_of_range"
QUOTE_NOT_FOUND = "quote_not_found"
WEAK_SUPPORT = "weak_support"

# Statuses that mean the model cited something that is not in the transcript.
FABRICATED = {OUT_OF_RANGE, QUOTE_NOT_FOUND}


@dataclass(frozen=True)
class Citation:
    page: int
    line_start: int
    line_end: int
    quote: str | None = None

    def label(self) -> str:
        if self.line_start == self.line_end:
            return f"p.{self.page}:{self.line_start}"
        return f"p.{self.page}:{self.line_start}-{self.line_end}"


@dataclass
class Claim:
    statement: str
    citations: list[Citation] = field(default_factory=list)


@dataclass
class Verdict:
    claim: Claim
    status: str
    detail: str = ""
    support: float = 0.0

    @property
    def ok(self) -> bool:
        return self.status == VERIFIED


def verify_claim(transcript: Transcript, claim: Claim, min_support: float = 0.34) -> Verdict:
    if not claim.citations:
        return Verdict(claim, NO_CITATION, "claim has no citation")

    spans: list[str] = []
    for cite in claim.citations:
        span = transcript.span_text(cite.page, cite.line_start, cite.line_end)
        if span is None:
            return Verdict(claim, OUT_OF_RANGE, f"{cite.label()} does not exist in the transcript")
        if cite.quote and normalize(cite.quote) not in normalize(span):
            where = transcript.find_quote(cite.quote)
            hint = ""
            if where:
                hint = f"; the quote appears at p.{where[0]}:{where[1]}-{where[2]}"
            return Verdict(
                claim, QUOTE_NOT_FOUND, f"quoted text is not in {cite.label()}{hint}"
            )
        spans.append(span)

    score = support(claim.statement, " ".join(spans))
    if score < min_support:
        return Verdict(
            claim,
            WEAK_SUPPORT,
            f"only {score:.0%} of the claim's key words appear in the cited lines",
            score,
        )
    return Verdict(claim, VERIFIED, "", score)


def verify_claims(
    transcript: Transcript, claims: list[Claim], min_support: float = 0.34
) -> list[Verdict]:
    return [verify_claim(transcript, claim, min_support) for claim in claims]


def summarize_verdicts(verdicts: list[Verdict]) -> dict[str, int]:
    counts = Counter(v.status for v in verdicts)
    counts["total"] = len(verdicts)
    return dict(counts)
