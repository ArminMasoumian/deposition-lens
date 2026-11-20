"""Command line interface: python -m depo <command> ..."""

from __future__ import annotations

import argparse
import sys

from . import analyzer, qa
from .contradictions import find_contradictions
from .evaluate import evaluate, load_gold
from .llm import get_backend
from .parser import load_transcript
from .timeline import build_timeline
from .verifier import Verdict, summarize_verdicts, verify_claims


def _print_verdicts(verdicts: list[Verdict]) -> None:
    for verdict in verdicts:
        cites = ", ".join(c.label() for c in verdict.claim.citations) or "no citation"
        if verdict.ok:
            print(f"[SOURCE CHECK PASSED] {verdict.claim.statement}\n           ({cites})\n")
        else:
            print(
                f"[UNVERIFIED: {verdict.status}] {verdict.claim.statement}\n"
                f"           ({cites}) {verdict.detail}\n"
            )
    counts = summarize_verdicts(verdicts)
    ok = counts.get("verified", 0)
    print(f"{ok} of {counts['total']} source-reference checks passed (not semantic validation).")
    print("Review by an attorney is required before relying on any output.")


def _build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--llm",
        choices=["none", "openai", "anthropic"],
        default="none",
        help="model backend (default: none, rule-based baseline)",
    )

    parser = argparse.ArgumentParser(prog="depo", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("summarize", parents=[common], help="cited summary of the testimony")
    p.add_argument("transcript")
    p.add_argument("--max-claims", type=int, default=12)

    p = sub.add_parser("ask", parents=[common], help="find related testimony with page/line citations")
    p.add_argument("transcript")
    p.add_argument("question")

    p = sub.add_parser("timeline", help="dates and times mentioned, with citations")
    p.add_argument("transcript")

    p = sub.add_parser("contradictions", help="opposite answers to near-identical questions")
    p.add_argument("transcript")
    p.add_argument("--min-similarity", type=float, default=0.5)

    p = sub.add_parser("evaluate", parents=[common], help="score the summary against gold labels")
    p.add_argument("transcript")
    p.add_argument("gold")
    p.add_argument("--max-claims", type=int, default=12)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    transcript = load_transcript(args.transcript)
    if not transcript.exchanges:
        print("No question/answer exchanges found. Check the transcript format (see README).")
        return 1

    if args.cmd == "summarize":
        claims = analyzer.summarize(transcript, get_backend(args.llm), args.max_claims)
        _print_verdicts(verify_claims(transcript, claims))

    elif args.cmd == "ask":
        claims = qa.answer(transcript, args.question, get_backend(args.llm))
        if not claims:
            print("No sufficiently close keyword match; the transcript may still contain relevant testimony.")
            return 0
        _print_verdicts(verify_claims(transcript, claims))

    elif args.cmd == "timeline":
        for event in build_timeline(transcript):
            print(f"{event.when}  ({event.citation.label()})\n  {event.text}\n")

    elif args.cmd == "contradictions":
        found = find_contradictions(transcript, args.min_similarity)
        if not found:
            print("No contradictions found by the heuristic.")
        for item in found:
            print(
                f"Possible contradiction (question similarity {item.similarity:.0%}):\n"
                f"  {item.first.label()}: {item.first_answer}\n"
                f"  {item.second.label()}: {item.second_answer}\n"
            )

    elif args.cmd == "evaluate":
        claims = analyzer.summarize(transcript, get_backend(args.llm), args.max_claims)
        result = evaluate(transcript, load_gold(args.gold), claims)
        print(f"claims produced:        {result.claims}")
        print(f"citation accuracy:      {result.citation_accuracy:.0%} ({result.verified}/{result.claims})")
        print(f"fabricated citations:   {result.fabricated}")
        print(
            f"key-fact recall:        {result.key_fact_recall:.0%} "
            f"({result.facts_covered}/{result.facts})"
        )
        for line in result.missed:
            print(f"  missed {line}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
