"""Parse a page:line formatted transcript into lines and question/answer exchanges.

Expected format (see data/sample_deposition.txt):

    [Page 3]
     1  Q. Where were you at about 2:15 p.m.?
     2  A. Near Loading Dock 4.

Each numbered line keeps its page and line number so every later claim can cite it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .textutil import normalize

PAGE_RE = re.compile(r"^\[Page (\d+)\]\s*$")
LINE_RE = re.compile(r"^\s*(\d{1,2})\s{2}(.*)$")
SPEAKER_RE = re.compile(r"^(Q\.|A\.|[A-Z][A-Z .'\-]{2,}:)\s*(.*)$")

Span = tuple[int, int, int]  # (page, line_start, line_end)


@dataclass(frozen=True)
class Line:
    page: int
    number: int
    text: str  # without the Q./A. prefix
    speaker: str | None  # "Q", "A", or a colloquy speaker such as "MR. DIAZ"
    raw: str


def _page_span(lines: list[Line]) -> Span | None:
    """Span of the lines on the first page they appear (a citation covers one page)."""
    if not lines:
        return None
    page = lines[0].page
    numbers = [ln.number for ln in lines if ln.page == page]
    return (page, min(numbers), max(numbers))


def _page_spans(lines: list[Line]) -> list[Span]:
    """One span per page the lines touch, so a citation never straddles a page break."""
    spans: list[Span] = []
    for page in dict.fromkeys(ln.page for ln in lines):
        numbers = [ln.number for ln in lines if ln.page == page]
        spans.append((page, min(numbers), max(numbers)))
    return spans


@dataclass
class Exchange:
    question: list[Line] = field(default_factory=list)
    answer: list[Line] = field(default_factory=list)

    @property
    def question_text(self) -> str:
        return " ".join(ln.text for ln in self.question)

    @property
    def answer_text(self) -> str:
        return " ".join(ln.text for ln in self.answer)

    def answer_span(self) -> Span | None:
        return _page_span(self.answer)

    def answer_spans(self) -> list[Span]:
        return _page_spans(self.answer)

    def full_span(self) -> Span | None:
        return _page_span(self.question + self.answer)


class Transcript:
    def __init__(self, lines: list[Line], exchanges: list[Exchange], header: list[str]):
        self.lines = lines
        self.exchanges = exchanges
        self.header = header
        self._index = {(ln.page, ln.number): ln for ln in lines}

    def span_text(self, page: int, start: int, end: int) -> str | None:
        """Text of lines start..end on a page, or None if any of those lines does not exist."""
        if start > end:
            return None
        parts = []
        for number in range(start, end + 1):
            line = self._index.get((page, number))
            if line is None:
                return None
            parts.append(line.text)
        return " ".join(parts)

    def find_quote(self, quote: str, max_lines: int = 6) -> Span | None:
        """Find where a quote really appears (used to explain a bad citation)."""
        target = normalize(quote)
        if not target:
            return None
        pages: dict[int, list[Line]] = {}
        for ln in self.lines:
            pages.setdefault(ln.page, []).append(ln)
        # Try the shortest windows first so the reported location is the tightest match.
        for size in range(max_lines):
            for page, plines in pages.items():
                for i in range(len(plines) - size):
                    window = " ".join(ln.text for ln in plines[i : i + size + 1])
                    if target in normalize(window):
                        return (page, plines[i].number, plines[i + size].number)
        return None

    def render(self) -> str:
        """Transcript as '[p.3:12] text' lines, used inside LLM prompts."""
        return "\n".join(f"[p.{ln.page}:{ln.number}] {ln.raw}" for ln in self.lines)

    @property
    def page_count(self) -> int:
        return len({ln.page for ln in self.lines})


def parse_transcript(text: str) -> Transcript:
    header: list[str] = []
    lines: list[Line] = []
    exchanges: list[Exchange] = []
    page: int | None = None
    current: Exchange | None = None
    mode: str | None = None
    last_speaker: str | None = None

    for raw in text.splitlines():
        page_match = PAGE_RE.match(raw.strip())
        if page_match:
            page = int(page_match.group(1))
            continue
        if page is None:
            header.append(raw)
            continue
        line_match = LINE_RE.match(raw)
        if not line_match:
            continue

        number = int(line_match.group(1))
        body = line_match.group(2).strip()
        speaker_match = SPEAKER_RE.match(body)
        if speaker_match:
            tag, body_text = speaker_match.group(1), speaker_match.group(2)
            last_speaker = tag.rstrip(":.")
        else:
            body_text = body

        line = Line(page, number, body_text, last_speaker, body)
        lines.append(line)

        if last_speaker == "Q":
            if speaker_match and mode != "Q":
                current = Exchange()
                exchanges.append(current)
            if current is None:
                current = Exchange()
                exchanges.append(current)
            mode = "Q"
            current.question.append(line)
        elif last_speaker == "A":
            if current is None:
                current = Exchange()
                exchanges.append(current)
            mode = "A"
            current.answer.append(line)
        else:  # colloquy between attorneys ends the current exchange
            mode = None
            current = None

    exchanges = [ex for ex in exchanges if ex.answer]
    return Transcript(lines, exchanges, header)


def load_transcript(path: str) -> Transcript:
    with open(path, encoding="utf-8") as handle:
        return parse_transcript(handle.read())
