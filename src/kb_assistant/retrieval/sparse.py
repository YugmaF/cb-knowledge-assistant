"""BM25 as sparse vectors.

Dense embeddings find meaning ("card payments failing" ~ "transactions declined"); BM25 finds exact
tokens that embeddings blur: incident ids, service names, error codes ("INC-2025-014",
"payments-ledger", "HikariPool"). Encoding BM25 as a sparse vector lets Pinecone score both in one
query: the document side carries the BM25 term-frequency weight, the query side carries IDF, and
their dot product is the BM25 score.
"""

from __future__ import annotations

import json
import math
import re
import zlib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

_TOKEN = re.compile(r"[a-z0-9]+(?:[-_.][a-z0-9]+)*")
_STOPWORDS = frozenset(
    ["a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from", "has", "have", "how", "i", "if", "in", "into", "is", "it", "its", "of", "on", "or", "our", "so", "that", "the", "their", "then", "there", "these", "they", "this", "to", "was", "we", "were", "what", "when", "where", "which", "who", "why", "will", "with", "you", "your", "all", "any", "can", "did", "do", "does", "about", "after", "before", "over", "under"]
)


def tokenize(text: str) -> list[str]:
    tokens = _TOKEN.findall(text.lower())
    out: list[str] = []
    for tok in tokens:
        if tok in _STOPWORDS or len(tok) < 2:
            continue
        out.append(tok)
        # "payments-ledger" also indexes "payments" and "ledger" so partial queries still match.
        if "-" in tok or "_" in tok:
            out.extend(p for p in re.split(r"[-_.]", tok) if len(p) > 1 and p not in _STOPWORDS)
    return out


def token_index(token: str) -> int:
    """Stable 31-bit hash, so indices are identical across processes and machines."""
    return zlib.crc32(token.encode()) & 0x7FFFFFFF


@dataclass
class SparseVector:
    indices: list[int]
    values: list[float]

    def to_pinecone(self) -> dict:
        return {"indices": self.indices, "values": self.values}

    def dot(self, other: SparseVector) -> float:
        lookup = dict(zip(other.indices, other.values, strict=True))
        return sum(v * lookup.get(i, 0.0) for i, v in zip(self.indices, self.values, strict=True))

    def scaled(self, factor: float) -> SparseVector:
        return SparseVector(list(self.indices), [v * factor for v in self.values])


class BM25Encoder:
    def __init__(self, k1: float = 1.2, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.doc_freq: dict[int, int] = {}
        self.n_docs = 0
        self.avg_len = 1.0

    def fit(self, texts: list[str]) -> BM25Encoder:
        df: Counter[int] = Counter()
        total = 0
        for text in texts:
            tokens = tokenize(text)
            total += len(tokens)
            df.update({token_index(t) for t in tokens})
        self.doc_freq = dict(df)
        self.n_docs = len(texts)
        self.avg_len = total / max(1, len(texts))
        return self

    def encode_document(self, text: str) -> SparseVector:
        tokens = tokenize(text)
        tf = Counter(token_index(t) for t in tokens)
        norm = self.k1 * (1 - self.b + self.b * len(tokens) / self.avg_len)
        indices = list(tf)
        values = [tf[i] * (self.k1 + 1) / (tf[i] + norm) for i in indices]
        return SparseVector(indices, values)

    def encode_query(self, text: str) -> SparseVector:
        counts = Counter(token_index(t) for t in tokenize(text))
        indices: list[int] = []
        values: list[float] = []
        for idx in counts:
            df = self.doc_freq.get(idx, 0)
            idf = math.log(1 + (self.n_docs - df + 0.5) / (df + 0.5))
            indices.append(idx)
            values.append(idf)
        # Unit-normalise the query side so sparse scores sit on a scale comparable to cosine.
        norm = math.sqrt(sum(v * v for v in values)) or 1.0
        return SparseVector(indices, [v / norm for v in values])

    def save(self, path: Path) -> None:
        path.write_text(json.dumps({
            "k1": self.k1, "b": self.b, "n_docs": self.n_docs, "avg_len": self.avg_len,
            "doc_freq": self.doc_freq,
        }))

    @classmethod
    def load(cls, path: Path) -> BM25Encoder:
        data = json.loads(path.read_text())
        enc = cls(data["k1"], data["b"])
        enc.n_docs = data["n_docs"]
        enc.avg_len = data["avg_len"]
        enc.doc_freq = {int(k): v for k, v in data["doc_freq"].items()}
        return enc
