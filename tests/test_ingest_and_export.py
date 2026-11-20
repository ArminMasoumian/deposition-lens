import io
import json
from pathlib import Path

import pytest

from depo.ingest import IngestionError, from_pdf_bytes, from_text, load_upload
from depo.export import json_report, markdown_report
from depo.analyzer import summarize
from depo.verifier import verify_claims

SAMPLE = Path(__file__).resolve().parents[1] / 'data' / 'sample_deposition.txt'


def test_load_text_and_report():
    doc = from_text(SAMPLE.read_text())
    verdicts = verify_claims(doc.transcript, summarize(doc.transcript))
    report = json.loads(json_report(doc.transcript, verdicts))
    assert len(report['claims']) == len(verdicts)
    assert 'not establish factual truth' in report['disclaimer']
    assert 'attorney review' in markdown_report(doc.transcript, verdicts).lower()


def test_unknown_extension():
    with pytest.raises(IngestionError, match='Upload'):
        load_upload('report.docx', b'hello')


def test_unnumbered_transcript_fails_closed():
    with pytest.raises(IngestionError, match='numbered'):
        from_text('[Page 1]\nQ. hello\nA. world')


def test_duplicate_line_rejected():
    with pytest.raises(IngestionError, match='Duplicate'):
        from_text('[Page 1]\n 1  Q. Hello?\n 2  A. This answer is long enough.\n 2  A. Another answer is here.')


def _sample_pdf(page_label=True):
    from reportlab.pdfgen import canvas
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer)
    y = 760
    if page_label:
        c.drawString(40, y, 'Page 27')
    lines = [' 1  Q. Did you see the sign?', ' 2  A. No, the sign was not present.',
             ' 3  Q. What time was it?', ' 4  A. About 4:00 PM that afternoon.']
    for offset, line in enumerate(lines, 1):
        c.drawString(40, y - 20 * offset, line)
    c.save()
    return buffer.getvalue()


def test_pdf_uses_transcript_page_label_not_pdf_offset():
    doc = from_pdf_bytes(_sample_pdf())
    assert doc.transcript.page_count == 1
    assert doc.transcript.span_text(27, 2, 2) == 'No, the sign was not present.'
    assert doc.transcript.span_text(1, 2, 2) is None


def test_pdf_without_printed_page_rejected():
    with pytest.raises(IngestionError, match='header'):
        from_pdf_bytes(_sample_pdf(page_label=False))


def test_bad_pdf_rejected():
    with pytest.raises(IngestionError, match='Could not open'):
        from_pdf_bytes(b'not a real PDF')


def test_bundled_demo_pdf():
    demo = SAMPLE.with_suffix('.pdf')
    doc = from_pdf_bytes(demo.read_bytes())
    assert doc.transcript.page_count == 4
    assert doc.transcript.span_text(2, 3, 4) == 'No. I did not see any sign in that part of the aisle.'
