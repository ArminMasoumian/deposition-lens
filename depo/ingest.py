"""Conservative input handling for transcripts.

PDF page coordinates are never silently represented as deposition page:line numbers.
Only existing numbered transcript lines are accepted. Image-only scans need OCR
and manual validation outside this demo.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .parser import Transcript, parse_transcript

MAX_BYTES = 12 * 1024 * 1024
MAX_PAGES = 250
_NUMBERED = re.compile(r'^\s*(\d{1,2})\s+(.+\S)\s*$')
_PAGE_HEADER = re.compile(r'^\s*(?:\[Page\s+\d+\]|Page\s+\d+)\s*$', re.I)


class IngestionError(ValueError):
    pass


@dataclass
class LoadedDocument:
    transcript: Transcript
    format: str
    warnings: list[str]


def _validate(transcript: Transcript) -> None:
    if len(transcript.lines) < 3 or not transcript.exchanges:
        raise IngestionError(
            'No usable numbered Q/A testimony found. This demo requires deposition page headers '
            'and existing numbered lines (e.g. "1  Q. ..."). Do not fabricate line numbers.'
        )
    keys = [(line.page, line.number) for line in transcript.lines]
    if len(set(keys)) != len(keys):
        raise IngestionError('Duplicate page/line identifiers found. Correct the source before analysis.')


def from_text(text: str) -> LoadedDocument:
    if not text.strip():
        raise IngestionError('The transcript is empty.')
    if len(text.encode('utf-8')) > MAX_BYTES:
        raise IngestionError('The transcript exceeds the 12 MB demo limit.')
    transcript = parse_transcript(text)
    _validate(transcript)
    return LoadedDocument(transcript, 'TXT', [])


def from_pdf_bytes(data: bytes) -> LoadedDocument:
    if len(data) > MAX_BYTES:
        raise IngestionError('The PDF exceeds the 12 MB demo limit.')
    try:
        import fitz
    except ImportError as exc:
        raise IngestionError('PDF support requires pymupdf (pip install pymupdf).') from exc
    try:
        document = fitz.open(stream=data, filetype='pdf')
    except Exception as exc:
        raise IngestionError('Could not open this PDF.') from exc
    try:
        if document.needs_pass:
            raise IngestionError('Password-protected PDFs are not supported.')
        if not 0 < len(document) <= MAX_PAGES:
            raise IngestionError('PDF must have 1–250 pages.')
        lines = []
        numbered_counts = []
        for page_index, page in enumerate(document, 1):
            text = page.get_text('text', sort=True)
            raw = text.splitlines()
            # PDF physical page number is NOT automatically the deposition page number.
            # A transcript's printed page label may differ from its PDF offset.
            declared_page = None
            for raw_line in raw[:12]:
                hit = _PAGE_HEADER.fullmatch(raw_line.strip())
                if hit:
                    declared_page = int(re.search(r'\d+', hit.group()).group())
                    break
            if declared_page is None:
                raise IngestionError(
                    f'PDF page {page_index} has no recognizable deposition page header. '
                    'Export a numbered transcript with explicit "Page N" headers; '
                    'PDF page order alone is not a safe legal citation.'
                )
            numbered = []
            for raw_line in raw:
                match = _NUMBERED.match(raw_line)
                if match and 1 <= int(match.group(1)) <= 25:
                    # PDF extraction frequently collapses the two delimiter spaces.
                    numbered.append(f'{match.group(1)}  {match.group(2)}')
            numbered_counts.append(len(numbered))
            lines.append(f'[Page {declared_page}]')
            lines.extend(numbered)
        transcript = parse_transcript('\n'.join(lines))
        _validate(transcript)
        if any(count < 1 for count in numbered_counts):
            raise IngestionError('Some PDF pages have no extractable numbered lines. Check for scanned pages or unusual layouts.')
        return LoadedDocument(transcript, 'PDF', [
            'PDF text extraction is layout-dependent. Compare the displayed lines with the original PDF before relying on citations.',
            'Scanned or unnumbered PDFs are not supported; no OCR or invented line labels are used.',
        ])
    finally:
        document.close()


def load_upload(name: str, data: bytes) -> LoadedDocument:
    if name.lower().endswith('.pdf'):
        return from_pdf_bytes(data)
    if name.lower().endswith('.txt'):
        try:
            return from_text(data.decode('utf-8-sig'))
        except UnicodeDecodeError as exc:
            raise IngestionError('Text files must be UTF-8 encoded.') from exc
    raise IngestionError('Upload a .txt or .pdf file.')
