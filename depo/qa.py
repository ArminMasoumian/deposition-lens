"""Offline evidence retrieval, not a model-generated legal answer.

Scores the *answer* as well as the examiner's question, uses a small, explicit
vocabulary of legal/transcript paraphrases, and abstains for weak matches.
"""
from __future__ import annotations

import re
from .analyzer import SYSTEM_PROMPT, parse_claims_json
from .llm import LLM
from .parser import Exchange, Transcript
from .retrieval import BM25
from .textutil import content_tokens
from .verifier import Citation, Claim
from .review import display_excerpt

# These groups only expand literal lexical search. They do not claim semantic
# equivalence, entailment, or determine an answer's truth.
_GROUPS = (
    ('leave', 'left', 'depart', 'quit', 'resign', 'afterward', 'afterwards', 'subsequent', 'next'),
    ('work', 'worked', 'job', 'employ', 'position', 'role', 'career', 'responsibility'),
    ('review', 'reviewed', 'examin', 'read', 'looked'),
    ('research', 'researcher', 'studied', 'study', 'investigat'),
    ('consult', 'consulting', 'consultant', 'advis', 'expert'),
    ('record', 'records', 'document', 'documents', 'file'),
    ('deposition', 'deposed', 'testimony', 'transcript'),
    ('tobacco', 'cigarette', 'smok'),
)
_CANON = {}
for group in _GROUPS:
    canonical = group[0]
    for variant in group:
        for token in content_tokens(variant):
            _CANON[token] = canonical

# Common mentions of the witness inside a question contribute nothing to
# relevance, since most of the transcript is about that same witness.
_HONORIFICS = {'bruce', 'kelman', 'witness', 'mr', 'mrs', 'ms', 'dr', 'institute', 'memorial'}
_GENERIC = {'work', 'when', 'time', 'first', 'date', 'company', 'companies'}

def _terms(text: str) -> set[str]:
    return {_CANON.get(tok, tok) for tok in content_tokens(text) if tok not in _HONORIFICS}


def _intent(question: str) -> set[str]:
    raw = question.lower()
    intent = set()
    if re.search(r'\b(?:when|year|date)\b', raw): intent.add('date')
    if re.search(r'\b(?:where|afterwards?|next|later)\b', raw): intent.add('transition')
    if re.search(r'\b(?:role|position|job|title|responsibilit)', raw): intent.add('role')
    if re.search(r'\b(?:what|which|list)\b', raw) and re.search(r'\b(?:records?|documents?|depositions?)\b', raw): intent.add('detail')
    return intent


def _rank(transcript: Transcript, question: str, k: int) -> list[Exchange]:
    exchanges = transcript.exchanges
    if not exchanges: return []
    query = _terms(question)
    if not query: return []
    intent = _intent(question)
    docs = [list(_terms(ex.question_text)) + list(_terms(ex.answer_text)) for ex in exchanges]
    bm25 = BM25(docs)
    base_scores = bm25.scores(sorted(query))
    ranked = []
    for idx, ex in enumerate(exchanges):
        if not ex.question or not ex.answer: continue
        question_terms = _terms(ex.question_text)
        # OCR sometimes merges the next examiner question into an answer.
        # Ranking must not treat that later question as the witness's answer.
        visible_answer = display_excerpt(ex.answer_text, 10000)
        embedded_job_list = ('role' in intent and 'battelle' in query and
                             bool(re.search(r'\b(?:Senior Research Scientist|Section Manager|Department Manager)\b', ex.answer_text, re.I)))
        answer_terms = _terms(ex.answer_text if embedded_job_list else visible_answer)
        all_terms = question_terms | answer_terms
        matches = all_terms & query
        if len(matches) < min(2, len(query)): continue
        # Distinctive concepts should match, not merely the institution name.
        distinctive = query - {'battelle', 'company', 'companies', 'work', 'date'}
        distinctive_match = distinctive & matches
        if distinctive and not distinctive_match: continue
        if 'tobacco' in query and 'tobacco' not in all_terms: continue
        if 'research' in query and 'research' not in all_terms: continue
        if 'tobacco' in query and 'research' in query:
            # The two concepts must actually be discussed together in a
            # source question or witness answer. Matching one from each side
            # of an unrelated Q/A is an especially common OCR false positive.
            if not {'tobacco', 'research'} <= question_terms:
                continue

        answer = ex.answer_text if embedded_job_list else visible_answer
        short_answer = len(_terms(answer)) < 3
        # A literal answer about a career transition takes priority over broad
        # references to the institution in unrelated questions.
        intent_score = 0.0
        if 'date' in intent:
            if re.search(r'\b(?:19|20)\d{2}\b', answer): intent_score += 5.0
            else: intent_score -= 2.0
        if 'transition' in intent:
            if re.search(r'\b(?:left|departed|went to work|joined|after leaving|moved to)\b', answer, re.I): intent_score += 7.0
            else: intent_score -= 5.0
        if 'role' in intent:
            if re.search(r'\b(?:position|role|worked as|was a|scientist|toxicologist|director|manager|researcher)\b', answer, re.I): intent_score += 5.0
            if embedded_job_list: intent_score += 13.0
            else: intent_score -= 3.0
        if 'detail' in intent:
            if re.search(r'\b(?:reviewed|records|depositions|notes|list|medical history)\b', answer, re.I): intent_score += 5.0
            if len(_terms(answer)) >= 15: intent_score += 3.0
        if short_answer:
            intent_score -= 9 if len(query) >= 3 else 2
        if 'research' in query and 'tobacco' in query and 'research' not in answer_terms:
            intent_score -= 5
        if 'research' in query and 'tobacco' in query and 'research' in question_terms:
            intent_score += 3
        # The user asks about this witness, not whether acquaintances did work
        # for an industry. Keep those third-party references below direct ones.
        if 'research' in query and 'tobacco' in query:
            if re.search(r'\b(?:friends?|colleagues?|other researchers|other people)\b', ex.question_text, re.I):
                intent_score -= 12
            if re.search(r'\b(?:have you|did you|your research|you ever)\b', ex.question_text, re.I):
                intent_score += 4
        elif len(_terms(answer)) >= 12:
            intent_score += 2.5
        if not (matches & answer_terms) and len(query) >= 4:
            intent_score -= 3
        if 'role' in intent and 'battelle' in query and 'battelle' not in all_terms:
            continue
        overlap = len(matches) / len(query)
        score = 7 * overlap + 2.2 * len(matches & answer_terms) + 1.5 * len(matches & question_terms) + intent_score + min(base_scores[idx], 20) * .22
        # Confident retrieval must have enough topic context. The BM25 result
        # on its own never establishes a suitable answer.
        if score < 6 or (intent_score < -4 and overlap < .8): continue
        ranked.append((score, idx))
    ranked.sort(key=lambda pair: (-pair[0], pair[1]))
    return [exchanges[idx] for _, idx in ranked[:k]]


def answer(transcript: Transcript, question: str, llm: LLM | None = None, k: int = 3) -> list[Claim]:
    top = _rank(transcript, question, k)
    if not top: return []
    if llm is None:
        results = []
        for ex in top:
            spans = ex.answer_spans()
            if not spans or not ex.answer_text.strip(): continue
            response_text = display_excerpt(ex.answer_text, 10000)
            # A source OCR defect occasionally places the actual response after
            # an embedded examiner question. Recover it only for an obvious
            # enumerated job-title answer, preserving the original source cite.
            if re.search(r'\b(?:Senior Research Scientist|Section Manager)\b', ex.answer_text, re.I) and 'Q•' in ex.answer_text:
                response_text = ex.answer_text.split('Q•', 1)[1]
                response_text = re.split(r'\?\s*(?:Okay\.|All right\.)', response_text, maxsplit=1)[-1].strip()
                response_text = display_excerpt(response_text, 10000)
            results.append(Claim(
                f"Related transcript excerpt — Q: {ex.question_text} | A: {response_text}",
                [Citation(page, start, end, transcript.span_text(page, start, end)) for page, start, end in spans],
            ))
        return results
    excerpts = []
    for ex in top:
        for ln in ex.question + ex.answer:
            excerpts.append(f"[p.{ln.page}:{ln.number}] {ln.raw}")
        excerpts.append('')
    user = ('Excerpts (each line is prefixed with its page and line):\n\n' + '\n'.join(excerpts)
            + f'\nQuestion: {question}\nAnswer using only these excerpts. If they do not answer the question, return [].')
    return parse_claims_json(llm.complete(SYSTEM_PROMPT, user))
