"""Tiny BM25 ranker so retrieval works with no external services or packages."""

from __future__ import annotations

import math
from collections import Counter


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.n = len(docs)
        self.avgdl = (sum(len(d) for d in docs) / self.n) if self.n else 1.0
        self.avgdl = self.avgdl or 1.0
        self.lengths = [len(d) for d in docs]
        self.tfs = [Counter(d) for d in docs]
        self.df: Counter = Counter()
        for d in docs:
            for term in set(d):
                self.df[term] += 1

    def _idf(self, term: str) -> float:
        n = self.df.get(term, 0)
        return math.log(1 + (self.n - n + 0.5) / (n + 0.5))

    def scores(self, query: list[str]) -> list[float]:
        out = []
        for tf, length in zip(self.tfs, self.lengths):
            score = 0.0
            for term in query:
                freq = tf.get(term, 0)
                if not freq:
                    continue
                denom = freq + self.k1 * (1 - self.b + self.b * length / self.avgdl)
                score += self._idf(term) * freq * (self.k1 + 1) / denom
            out.append(score)
        return out

    def top(self, query: list[str], k: int = 3) -> list[int]:
        scored = [(s, i) for i, s in enumerate(self.scores(query)) if s > 0]
        scored.sort(key=lambda pair: (-pair[0], pair[1]))
        return [i for _, i in scored[:k]]
