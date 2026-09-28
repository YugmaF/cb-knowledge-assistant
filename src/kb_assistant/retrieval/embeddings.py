"""Dense embeddings.

Default: BAAI/bge-small-en-v1.5 through fastembed (ONNX, runs locally, 384 dimensions). Local
embeddings keep document text inside the network boundary, cost nothing per query, and cannot be
rate-limited by a vendor. The HashingEmbedder is a deterministic stand-in used by tests and when
the ONNX model cannot be downloaded; it keeps the pipeline runnable, not accurate.
"""

from __future__ import annotations

import asyncio
import hashlib
import math
from typing import Protocol

from kb_assistant.retrieval.sparse import tokenize


class Embedder(Protocol):
    dim: int
    name: str

    async def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    async def embed_query(self, text: str) -> list[float]: ...


class FastEmbedEmbedder:
    def __init__(self, model_name: str, dim: int) -> None:
        from fastembed import TextEmbedding

        self._model = TextEmbedding(model_name=model_name)
        self.dim = dim
        self.name = model_name

    def _embed(self, texts: list[str]) -> list[list[float]]:
        return [vec.tolist() for vec in self._model.embed(texts)]

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # ONNX inference is CPU-bound and synchronous: run it off the event loop.
        return await asyncio.to_thread(self._embed, texts)

    async def embed_query(self, text: str) -> list[float]:
        # bge models were trained with this instruction prefix on the query side only.
        vectors = await asyncio.to_thread(
            self._embed, [f"Represent this sentence for searching relevant passages: {text}"]
        )
        return vectors[0]


class HashingEmbedder:
    """Feature-hashing bag of words, L2-normalised. Deterministic and dependency-free."""

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim
        self.name = f"hashing-{dim}"

    def _vector(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for tok in tokenize(text):
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            vec[h % self.dim] += 1.0 if (h >> 64) & 1 else -1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]

    async def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


def build_embedder(model_name: str, dim: int) -> Embedder:
    if model_name.startswith("hashing"):
        return HashingEmbedder(dim)
    return FastEmbedEmbedder(model_name, dim)
