"""Measure how good the output is against hand-labeled facts.

Gold file format (JSON list):
    [{"id": "F1", "description": "...", "page": 2, "line_start": 4, "line_end": 6}, ...]

Metrics
  citation_accuracy  share of claims that pass every verifier check
  fabricated         claims citing lines or quotes that are not in the transcript
  key_fact_recall    share of gold facts covered by at least one VERIFIED claim
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from .parser import Transcript
from .verifier import FABRICATED, Claim, verify_claims


@dataclass
class EvalResult:
    claims: int
    verified: int
    fabricated: int
    facts: int
    facts_covered: int
    missed: list[str]

    @property
    def citation_accuracy(self) -> float:
        return self.verified / self.claims if self.claims else 0.0

    @property
    def key_fact_recall(self) -> float:
        return self.facts_covered / self.facts if self.facts else 0.0


def load_gold(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _overlaps(cite, fact: dict) -> bool:
    return (
        cite.page == fact["page"]
        and cite.line_start <= fact["line_end"]
        and cite.line_end >= fact["line_start"]
    )


def evaluate(transcript: Transcript, gold: list[dict], claims: list[Claim]) -> EvalResult:
    verdicts = verify_claims(transcript, claims)
    verified = [v for v in verdicts if v.ok]
    covered, missed = 0, []
    for fact in gold:
        hit = any(_overlaps(cite, fact) for v in verified for cite in v.claim.citations)
        if hit:
            covered += 1
        else:
            missed.append(f"{fact['id']}: {fact['description']}")
    return EvalResult(
        claims=len(claims),
        verified=len(verified),
        fabricated=sum(1 for v in verdicts if v.status in FABRICATED),
        facts=len(gold),
        facts_covered=covered,
        missed=missed,
    )
