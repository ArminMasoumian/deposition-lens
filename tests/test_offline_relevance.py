from pathlib import Path
from depo.parser import parse_transcript
from depo.qa import answer
from depo.ingest import from_text

DATA = Path(__file__).resolve().parent.parent / 'data'

def test_does_not_answer_complex_question_with_person_name_only():
    tr = parse_transcript('''[Page 1]\n 1  Q. State your name.\n 2  A. Bruce Kelman.\n 3  Q. What is your job?\n 4  A. Scientist.\n''')
    assert answer(tr, 'What consulting research did Bruce Kelman do for tobacco companies?') == []

def test_finds_matching_simple_question():
    tr = from_text((DATA / 'sample_deposition.txt').read_text()).transcript
    found = answer(tr, 'What was the witness doing near Loading Dock 4?')
    assert found and any('manifest' in c.statement for c in found)

def test_still_abstains_when_silent():
    tr = from_text((DATA / 'sample_deposition.txt').read_text()).transcript
    assert answer(tr, 'What consulting did the witness provide to a tobacco company?') == []

def test_app_explicit_load_and_unique_download_keys():
    source = (DATA.parent / 'app.py').read_text()
    assert "st.button('Load Deposition'" in source
    assert 'if load_clicked and uploaded is not None:' in source
    assert 'if doc is None:' in source
    assert "key=f'md_{key}'" in source
    assert "key=f'json_{key}'" in source
    assert 'No paid model APIs' in source


def test_no_auto_tab_analysis_or_sample_button():
    source = (DATA.parent / 'app.py').read_text()
    assert 'Load fictional example' not in source
    assert 'PUBLIC DEMO:' not in source
    assert "if st.button('Generate Timeline'" in source
    assert "if st.button('Find Possible Conflicts'" in source
    assert "if st.button('Run Evaluation'" in source
