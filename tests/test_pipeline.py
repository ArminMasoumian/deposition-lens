import json
from pathlib import Path

import pytest

from depo import analyzer, qa
from depo.contradictions import find_contradictions, polarity
from depo.evaluate import evaluate, load_gold
from depo.parser import load_transcript, parse_transcript
from depo.timeline import build_timeline
from depo.verifier import (
    OUT_OF_RANGE,
    QUOTE_NOT_FOUND,
    VERIFIED,
    WEAK_SUPPORT,
    Citation,
    Claim,
    verify_claim,
    verify_claims,
)

DATA = Path(__file__).resolve().parent.parent / "data"


@pytest.fixture(scope="module")
def transcript():
    return load_transcript(str(DATA / "sample_deposition.txt"))


# ---------- parser ----------


def test_parses_pages_lines_and_exchanges(transcript):
    assert transcript.page_count == 4
    assert len(transcript.lines) == 76
    first = transcript.exchanges[0]
    assert first.question_text == "Please state your name for the record."
    assert first.answer_text == "Maria Lopez."
    assert first.answer_span() == (1, 2, 2)


def test_colloquy_is_not_part_of_an_exchange(transcript):
    assert all("Objection" not in ex.answer_text for ex in transcript.exchanges)


def test_answer_crossing_a_page_gets_one_span_per_page(transcript):
    ex = next(e for e in transcript.exchanges if "three feet across" in e.answer_text)
    spans = ex.answer_spans()
    assert [s[0] for s in spans] == [1, 2]


def test_span_text_missing_line_returns_none(transcript):
    assert transcript.span_text(1, 1, 99) is None
    assert transcript.span_text(9, 1, 1) is None


# ---------- verifier ----------


def test_verified_claim(transcript):
    text = transcript.span_text(1, 4, 5)
    claim = Claim(
        "She is a warehouse supervisor at Coastline Moving Company.",
        [Citation(1, 4, 5, "warehouse supervisor at Coastline")],
    )
    verdict = verify_claim(transcript, claim)
    assert verdict.status == VERIFIED, (verdict, text)


def test_out_of_range_citation_is_rejected(transcript):
    claim = Claim("She saw a forklift.", [Citation(7, 1, 3, "forklift")])
    assert verify_claim(transcript, claim).status == OUT_OF_RANGE


def test_invented_quote_is_rejected(transcript):
    claim = Claim(
        "She saw a forklift hit the shelf.",
        [Citation(1, 4, 5, "a forklift hit the shelf")],
    )
    assert verify_claim(transcript, claim).status == QUOTE_NOT_FOUND


def test_misplaced_quote_points_to_real_location(transcript):
    claim = Claim(
        "Maybe three feet across.",
        [Citation(1, 4, 5, "Maybe three feet across")],
    )
    verdict = verify_claim(transcript, claim)
    assert verdict.status == QUOTE_NOT_FOUND
    assert "p.1:25" in verdict.detail


def test_claim_unsupported_by_cited_lines_is_flagged(transcript):
    claim = Claim(
        "The witness admitted she disabled the safety cameras and lied to investigators.",
        [Citation(1, 4, 5)],
    )
    assert verify_claim(transcript, claim).status == WEAK_SUPPORT


def test_claim_without_citation_is_flagged(transcript):
    assert verify_claim(transcript, Claim("Something.", [])).status == "no_citation"


# ---------- baseline summary ----------


def test_baseline_claims_are_all_verified(transcript):
    claims = analyzer.baseline_summary(transcript, max_claims=12)
    assert len(claims) == 12
    verdicts = verify_claims(transcript, claims)
    assert all(v.ok for v in verdicts), [v.detail for v in verdicts if not v.ok]


# ---------- LLM path (fake model, no network) ----------


class FakeLLM:
    def __init__(self, reply: str):
        self.reply = reply

    def complete(self, system: str, user: str) -> str:
        return self.reply


def test_llm_hallucinations_are_caught(transcript):
    reply = json.dumps(
        [
            {
                "statement": "The witness works as a warehouse supervisor.",
                "page": 1,
                "line_start": 4,
                "line_end": 5,
                "quote": "warehouse supervisor at Coastline Moving Company",
            },
            {
                "statement": "The witness said a forklift caused the accident.",
                "page": 3,
                "line_start": 1,
                "line_end": 2,
                "quote": "a forklift caused the accident",
            },
            {
                "statement": "The witness testified on page 12.",
                "page": 12,
                "line_start": 1,
                "line_end": 1,
                "quote": "page twelve",
            },
        ]
    )
    claims = analyzer.summarize(transcript, FakeLLM("Here you go:\n" + reply + "\nDone."))
    statuses = [v.status for v in verify_claims(transcript, claims)]
    assert statuses == [VERIFIED, QUOTE_NOT_FOUND, OUT_OF_RANGE]


def test_unparseable_model_reply_raises(transcript):
    with pytest.raises(ValueError):
        analyzer.summarize(transcript, FakeLLM("I cannot do that."))


# ---------- question answering ----------


def test_qa_returns_cited_answer(transcript):
    claims = qa.answer(transcript, "Was there a wet floor sign?")
    assert claims
    verdicts = verify_claims(transcript, claims)
    assert verdicts[0].ok
    assert "did not see any sign" in claims[0].statement


def test_qa_declines_when_transcript_is_silent(transcript):
    assert qa.answer(transcript, "What is the capital of France?") == []


# ---------- timeline and contradictions ----------


def test_timeline_finds_dates_and_times(transcript):
    whens = " | ".join(e.when for e in build_timeline(transcript))
    assert "March 2016" in whens
    assert "2:15" not in whens  # time is in a question, only answers are scanned
    assert "7:00 a.m." in whens
    assert "November 2024" in whens


def test_polarity():
    assert polarity("Yes, I saw it.") == 1
    assert polarity("No, I did not.") == -1
    assert polarity("I believe so.") == 0


def test_finds_the_planted_contradiction(transcript):
    found = find_contradictions(transcript)
    assert len(found) == 1
    item = found[0]
    assert "No, I did not" in item.first_answer
    assert "Yes, I saw it" in item.second_answer


def test_no_contradiction_in_clean_transcript():
    text = (
        "[Page 1]\n"
        " 1  Q. Did you see the car?\n"
        " 2  A. Yes, it was red.\n"
        " 3  Q. Did you see the truck?\n"
        " 4  A. No, I did not.\n"
    )
    assert find_contradictions(parse_transcript(text)) == []


# ---------- evaluation ----------


def test_evaluation_reports_real_numbers(transcript):
    gold = load_gold(str(DATA / "sample_gold.json"))
    result = evaluate(transcript, gold, analyzer.baseline_summary(transcript, 12))
    assert result.claims == 12
    assert result.citation_accuracy == 1.0
    assert result.fabricated == 0
    assert 0 < result.key_fact_recall <= 1
