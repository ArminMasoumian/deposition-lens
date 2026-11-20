"""Regression cases for evidence retrieval and date presentation."""
from pathlib import Path
from depo.parser import parse_transcript
from depo.qa import answer
from depo.timeline import build_timeline

ROOT = Path(__file__).resolve().parent.parent

def test_answer_side_retrieval_and_citation():
    tr = parse_transcript('''[Page 1]
 1  Q. Why do you say your first deposition was later?
 2  A. I left Battelle in 1989 and went to work for Failure Analysis Associates.
 3  Q. Did you contact Battelle in Europe?
 4  A. Yes.
''')
    claims = answer(tr, 'When did the witness leave Battelle and where did they work afterward?')
    assert claims and 'Failure Analysis Associates' in claims[0].statement
    assert claims[0].citations[0].page == 1
    assert 'Yes.' not in claims[0].statement


def test_document_question_prefers_informative_answer():
    tr = parse_transcript('''[Page 1]
 1  Q. Did you rely on medical records?
 2  A. Yes.
 3  Q. Which documents did you bring?
 4  A. I reviewed medical records of Mary Smith and depositions of John Smith and Alice Smith.
''')
    claims = answer(tr, 'What medical records and depositions did the witness review?')
    assert claims and 'Mary Smith' in claims[0].statement


def test_time_references_are_preserved():
    text = (ROOT / 'data/sample_deposition.txt').read_text()
    items = build_timeline(parse_transcript(text))
    assert '7:00 a.m.' in [item.when for item in items]


def test_date_dedup_does_not_infer_year_or_cut_words():
    tr = parse_transcript('''[Page 1]
 1  Q. What did you review?
 2  A. I reviewed a file on October 28, and correcting myself,
 3  October 28, 1998 was the date of the deposition by Mr. Smith.
 4  I also reviewed notes from January 27 of an unknown year.
''')
    items = build_timeline(tr)
    dates = [item.when for item in items]
    assert dates.count('October 28, 1998') == 1
    assert 'October 28' not in dates
    assert 'January 27' in dates
    assert all(not x.text.endswith('deposi') for x in items)


def test_ocr_followup_question_does_not_become_timeline_event():
    tr = parse_transcript('''[Page 1]
 1  Q. What happened next?
 2  A. I met colleagues in 1989. Q• Did you say 1997?
''')
    dates = [e.when for e in build_timeline(tr)]
    assert '1997' not in dates
