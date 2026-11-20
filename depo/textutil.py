"""Small text helpers shared by the verifier, retrieval and contradiction checks."""

from __future__ import annotations

import re

STOPWORDS = set(
    """a an the and or but if of to in on at by for with from as is are was were be been being it
    its this that these those i you he she we they them his her their our your me my not no yes do
    did does done have has had will would could should can may might shall so then than there here
    which who whom what when where why how about into over under up down out off also just very any
    some one two witness testified said say""".split()
)

_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def normalize(text: str) -> str:
    """Lowercase, drop punctuation, collapse whitespace."""
    return _NON_ALNUM.sub(" ", text.lower()).strip()


def stem(token: str) -> str:
    """Very light suffix stripping so 'mopped' and 'mop' or 'signs' and 'sign' can match."""
    if len(token) > 5 and token.endswith("ing"):
        return token[:-3]
    if len(token) > 4 and token.endswith("ed"):
        return token[:-2]
    if len(token) > 4 and token.endswith("es"):
        return token[:-2]
    if len(token) > 3 and token.endswith("s"):
        return token[:-1]
    return token


def content_tokens(text: str) -> list[str]:
    """Stemmed tokens with stopwords removed. Digits are kept (dates and amounts matter)."""
    out = []
    for tok in normalize(text).split():
        if tok in STOPWORDS:
            continue
        if len(tok) < 2 and not tok.isdigit():
            continue
        out.append(stem(tok))
    return out


def support(statement: str, source: str) -> float:
    """Share of the statement's content tokens that also appear in the source text (0 to 1)."""
    wanted = set(content_tokens(statement))
    if not wanted:
        return 1.0
    have = set(content_tokens(source))
    return len(wanted & have) / len(wanted)
