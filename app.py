"""Deposition Lens: offline, source-linked transcript review.

Start with: streamlit run app.py
No external AI API is used, and no uploaded file is saved by this application.
"""
from __future__ import annotations

from pathlib import Path
import hashlib
import html

import streamlit as st

from depo.analyzer import summarize
from depo.contradictions import find_contradictions
from depo.evaluate import evaluate, load_gold
from depo.export import json_report, markdown_report
from depo.ingest import IngestionError, load_upload
from depo.qa import answer
from depo.review import select_highlights, display_excerpt
from depo.timeline import build_timeline
from depo.verifier import Claim, Citation, verify_claims

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title='Deposition Lens | Transcript Review', page_icon='⚖️', layout='wide')
st.markdown('''<style>
.block-container {max-width: 1320px; padding-top: 2rem}
.hero {padding:22px 28px;border:1px solid #2D405F;border-radius:18px;background:linear-gradient(115deg,#182A44,#111C31);margin-bottom:16px}
.eyebrow {color:#91B5F9;font-weight:700;letter-spacing:.12em;font-size:.75rem}
.hero h1 {margin:.35rem 0;color:#F3F7FF;font-size:2.4rem}
.hero p {color:#C0CEDD;margin:0}
[data-testid="stMetricValue"] {font-size:1.55rem}
.source-lines {white-space:pre-wrap;overflow-wrap:anywhere;font-family:monospace;line-height:1.65;padding:13px;border-radius:9px;background:#14223a;color:#eef3fe}
.source-line-selected {background:#38537d;display:block}
</style>''', unsafe_allow_html=True)
st.markdown('''<div class="hero"><div class="eyebrow">LEGAL AI · EVIDENCE-FIRST WORKFLOWS</div>
<h1>Deposition Lens</h1><p>Explore testimony, find relevant statements, and check page-and-line references against the source.</p></div>''', unsafe_allow_html=True)
st.caption('Document review tool · Not legal advice · Review results against the original transcript')


def clear_results():
    for key in ('overview_results','qa_result','timeline_results','conflict_candidates','evaluation_result'):
        st.session_state.pop(key, None)


with st.sidebar:
    st.header('Workspace')
    uploaded = st.file_uploader('Transcript', type=['txt','pdf'], help='Select an existing numbered transcript (TXT or text-based PDF).')
    load_clicked = st.button('Load Deposition', type='primary', use_container_width=True, disabled=uploaded is None)
    if st.session_state.get('loaded_doc') is not None:
        if st.button('Clear workspace', use_container_width=True):
            for key in ('loaded_doc','loaded_name','loaded_id','source_page','source_range'):
                st.session_state.pop(key, None)
            clear_results()
            st.rerun()
    st.divider()
    st.subheader('Analysis engine')
    st.caption('Offline text extraction and retrieval. No paid model APIs.')
    st.caption('Citation checks confirm source text and line references, not legal conclusions.')

if load_clicked and uploaded is not None:
    try:
        raw = uploaded.getvalue()
        document = load_upload(uploaded.name, raw)
        fingerprint = hashlib.sha256(raw).hexdigest()
        st.session_state['loaded_doc'] = document
        st.session_state['loaded_name'] = uploaded.name
        st.session_state['loaded_id'] = fingerprint
        st.session_state['source_page'] = min(ln.page for ln in document.transcript.lines)
        st.session_state.pop('source_range', None)
        clear_results()
    except IngestionError as exc:
        st.error(str(exc))
    except Exception as exc:
        st.error(f'Unable to read transcript: {exc}')

doc = st.session_state.get('loaded_doc')
if doc is None:
    st.info('Select a TXT or PDF transcript, then click **Load Deposition**.')
    st.stop()

tr = doc.transcript
source_label = st.session_state['loaded_name']
doc_id = st.session_state['loaded_id']
st.subheader(source_label)
metrics = st.columns(4)
for col, label, value in zip(metrics,
        ['Deposition pages','Numbered lines','Q/A exchanges','Processing mode'],
        [tr.page_count,len(tr.lines),len(tr.exchanges),'Offline']):
    col.metric(label,value)
for warning in doc.warnings:
    st.warning(warning)


def source_label_text(cite: Citation) -> str:
    return f'{cite.label()} (source text)'


def source_widget(cite: Citation, unique: str):
    with st.expander(f'View source · {source_label_text(cite)}'):
        lines = [ln for ln in tr.lines if ln.page == cite.page and cite.line_start <= ln.number <= cite.line_end]
        if lines:
            st.markdown('<div class="source-lines">' + '\n'.join(
                html.escape(f'{ln.number:>2}  {ln.raw}') for ln in lines) + '</div>', unsafe_allow_html=True)
            if st.button('Jump to page in Source viewer', key=f'jump_{unique}'):
                st.session_state['source_page'] = cite.page
                st.session_state['source_range'] = (cite.page, cite.line_start, cite.line_end)
                st.info('Page selected. Open the **Source viewer** tab to see the highlighted lines.')
        else:
            st.error('These lines were not found in the loaded transcript.')


def export_controls(claims: list[Claim], title: str, key: str):
    verdicts = verify_claims(tr, claims)
    a,b = st.columns(2)
    a.download_button('Download Markdown report', markdown_report(tr,verdicts,title),
                      file_name=f'{key}_report.md',mime='text/markdown',use_container_width=True,key=f'md_{key}')
    b.download_button('Download JSON report', json_report(tr,verdicts,title),
                      file_name=f'{key}_report.json',mime='application/json',use_container_width=True,key=f'json_{key}')


def format_question_answer(text: str):
    if text.startswith('Related transcript excerpt — Q: ') and ' | A: ' in text:
        question, response = text.removeprefix('Related transcript excerpt — Q: ').split(' | A: ',1)
        return question, response
    return '', text


tabs = st.tabs(['Overview','Ask the transcript','Timeline','Possible conflicts','Source viewer','Evaluation'])

with tabs[0]:
    st.markdown('#### Testimony highlights')
    st.caption('Selected witness answers, not a complete summary. Categories are keyword-based.')
    n = st.slider('Maximum highlights',min_value=4,max_value=20,value=8,key='claimcount')
    if st.button('Generate highlights',type='primary'):
        st.session_state['overview_results'] = (doc_id,n,select_highlights(tr,n))
    stored = st.session_state.get('overview_results')
    if not stored or stored[:2] != (doc_id,n):
        st.info('Click **Generate highlights** to review selected witness statements.')
    else:
        highlights = stored[2]
        if not highlights:
            st.info('No suitable highlights were found in this transcript.')
        for i,h in enumerate(highlights,1):
            with st.container(border=True):
                st.markdown(f'**{i}. {h.category}**')
                st.write('Question: ' + display_excerpt(h.question,220))
                st.write('Witness answer (excerpt): ' + display_excerpt(h.answer))
                st.caption('Selected by keyword-based topical scoring; read the complete source before drawing conclusions.')
                for j,cite in enumerate(h.citations):
                    source_widget(cite,f'h_{i}_{j}')
        if highlights:
            export_controls([Claim(h.answer,h.citations) for h in highlights], 'Selected testimony excerpts','highlights')

with tabs[1]:
    st.markdown('#### Search related testimony')
    st.caption('Returns potentially related verbatim Q/A exchanges, not a generated legal answer.')
    question = st.text_input('Question', placeholder='What did the witness see near the loading dock?')
    if st.button('Search testimony',type='primary',disabled=not question.strip()):
        st.session_state['qa_result']=(doc_id,question,answer(tr,question))
    stored=st.session_state.get('qa_result')
    if stored and stored[:2] == (doc_id,question):
        results=stored[2]
        if not results:
            st.info('No sufficiently relevant keyword matches were found. This does not prove the testimony is silent.')
        else:
            for i,claim in enumerate(results,1):
                q,a=format_question_answer(claim.statement)
                with st.container(border=True):
                    st.markdown(f'**{i}. Related Q/A exchange**')
                    if q: st.write('Question: '+display_excerpt(q,360))
                    st.write('Witness answer: '+display_excerpt(a,460))
                    st.caption('Related text only. Source reference checked separately from topical relevance.')
                    for j,cite in enumerate(claim.citations):
                        source_widget(cite,f'qa_{i}_{j}')
            export_controls(results,f'Related testimony: {question}','search')

with tabs[2]:
    st.markdown('#### Timeline of date references')
    st.caption('Dates are extracted from answers. An entry is a date mention, not a verified historical event.')
    if st.button('Generate Timeline',type='primary'):
        st.session_state['timeline_results']=(doc_id,build_timeline(tr))
    stored=st.session_state.get('timeline_results')
    if not stored or stored[0]!=doc_id:
        st.info('Click **Generate Timeline** to extract dated testimony.')
    elif not stored[1]:
        st.info('No dated statements with enough context were found.')
    else:
        events=stored[1]
        st.caption(f'{len(events)} date mentions · Ordered by date where possible; uncertain dates appear last.')
        for i,e in enumerate(events,1):
            with st.container(border=True):
                st.markdown(f'**{e.when}** · `{e.citation.label()}`')
                st.write(display_excerpt(e.text,300))
                source_widget(e.citation,f't_{i}')

with tabs[3]:
    st.markdown('#### Possible conflicting statements')
    st.caption('Lexical candidates only. A similarity score is NOT the probability that a contradiction exists.')
    threshold=st.slider('Question similarity threshold',min_value=0.3,max_value=0.95,value=0.5,step=0.01,key='conflict_threshold',format='%.2f')
    if st.button('Find Possible Conflicts',type='primary'):
        st.session_state['conflict_candidates']=(doc_id,find_contradictions(tr,min_similarity=0.3))
    stored=st.session_state.get('conflict_candidates')
    if not stored or stored[0]!=doc_id:
        st.info('Click **Find Possible Conflicts** to compare answers to similar questions.')
    else:
        candidates=stored[1]
        matches=[c for c in candidates if c.similarity+1e-9 >= threshold]
        st.caption(f'{len(matches)} of {len(candidates)} candidates meet the selected threshold.')
        if not matches:
            st.info('No candidate comparisons meet this threshold.')
        for i,c in enumerate(matches,1):
            with st.container(border=True):
                st.markdown(f'**Candidate comparison · question similarity {c.similarity:.0%}**')
                left,right=st.columns(2)
                for col, question_text, ans, cite, side in (
                    (left,c.first_question,c.first_answer,c.first,'a'),
                    (right,c.second_question,c.second_answer,c.second,'b')):
                    with col:
                        st.caption(cite.label())
                        st.write('Question: '+display_excerpt(question_text,230))
                        st.write('Answer: '+display_excerpt(ans,240))
                        source_widget(cite,f'c_{i}_{side}')

with tabs[4]:
    st.markdown('#### Source viewer')
    st.caption('Lines shown below come from the uploaded text extraction. OCR-derived labels may not match the certified court transcript.')
    pages=sorted({ln.page for ln in tr.lines})
    if st.session_state.get('source_page') not in pages:
        st.session_state['source_page']=pages[0]
    page=st.number_input('Jump to page',min_value=min(pages),max_value=max(pages),step=1,key='source_page')
    if page not in pages:
        st.warning('This page number was not found in the transcript.')
    else:
        selected=st.session_state.get('source_range')
        contents=[]
        for line in tr.lines:
            if line.page==page:
                raw=html.escape(f'{line.number:>2}  {line.raw}')
                if selected and selected[0]==page and selected[1]<=line.number<=selected[2]:
                    contents.append(f'<span class="source-line-selected">{raw}</span>')
                else:
                    contents.append(raw)
        st.markdown('<div class="source-lines">'+'\n'.join(contents)+'</div>',unsafe_allow_html=True)

with tabs[5]:
    st.markdown('#### Reproducible evaluation')
    st.caption('Accuracy metrics require manually labeled reference answers; source-reference checks alone cannot establish semantic correctness.')
    sample_bytes=(ROOT/'data'/'sample_deposition.txt').read_bytes()
    sample_pdf_hash = hashlib.sha256((ROOT/'data'/'sample_deposition.pdf').read_bytes()).hexdigest()
    is_sample = doc_id in (hashlib.sha256(sample_bytes).hexdigest(),sample_pdf_hash)
    if not is_sample:
        st.info('No validated answer key is available for this document. Evaluation scores cannot be calculated from an unlabelled deposition.')
    else:
        if st.button('Run Evaluation',type='primary'):
            gold=load_gold(str(ROOT/'data'/'sample_gold.json'))
            claims=summarize(tr,max_claims=12)
            st.session_state['evaluation_result']=(doc_id,evaluate(tr,gold,claims))
        stored=st.session_state.get('evaluation_result')
        if not stored or stored[0]!=doc_id:
            st.info('Click **Run Evaluation** to check the included sample against its annotated reference locations.')
        else:
            result=stored[1]
            a,b=st.columns(2)
            a.metric('Source checks passed',f'{result.verified}/{result.claims}')
            b.metric('Annotated locations found',f'{result.facts_covered}/{result.facts}')
            st.caption('These are results on one fictional sample, NOT real-world legal accuracy or answer correctness.')
            with st.expander('Annotated locations not retrieved'):
                for missed in result.missed: st.write(missed)
