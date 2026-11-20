from pathlib import Path
from depo.ingest import from_text, from_pdf_bytes
from depo.review import select_highlights, display_excerpt
from depo.timeline import build_timeline
from depo.contradictions import find_contradictions
from depo.qa import answer

DATA = Path(__file__).resolve().parent.parent / 'data'

def test_highlights_are_usable_and_not_case_lists():
    tr=from_text((DATA/'sample_deposition.txt').read_text()).transcript
    highlights=select_highlights(tr,6)
    assert len(highlights)==6
    assert all(h.question and h.answer and h.citations for h in highlights)
    assert any('pallet jack' in h.answer for h in highlights)


def test_ocr_question_is_not_displayed_as_answer():
    text='Spoke with toxicologists. .Q. What about other research?'
    assert display_excerpt(text) == 'Spoke with toxicologists.'


def test_timeline_individual_date_mentions_sorted():
    tr=from_text((DATA/'sample_deposition.txt').read_text()).transcript
    events=build_timeline(tr)
    assert any(x.when=='March 2016' for x in events)
    assert any(x.when=='November 2024' for x in events)
    assert all(x.when.count(',') == 0 for x in events)
    assert all(len(x.text)>len(x.when)+20 for x in events)


def test_conflict_scores_are_lexical_and_filterable():
    tr=from_text((DATA/'sample_deposition.txt').read_text()).transcript
    candidates=find_contradictions(tr,0.3)
    assert candidates
    assert any('pallet jack' in c.first_question for c in candidates)
    assert all(c.first_question and c.second_question for c in candidates)
    assert all(c.similarity<1 for c in candidates)
    assert len([c for c in candidates if c.similarity>=0.9]) < len(candidates)


def test_sample_pdf_works():
    doc=from_pdf_bytes((DATA/'sample_deposition.pdf').read_bytes())
    assert doc.transcript.page_count==4
    assert len(doc.transcript.exchanges)>0
