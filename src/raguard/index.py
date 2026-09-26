"""Vector store abstraction and a zero-dependency in-memory implementation.

``VectorStore`` is the seam where Chroma / pgvector / FAISS adapters plug in.
The default ``NumpyVectorStore`` keeps embeddings in a dense matrix and scores
cosine similarity — good enough for evaluation runs and CI, and honest about
its O(n·k) search cost.
"""

from __future__ import annotations

import hashlib
import math
import re
from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray

from raguard.ingest import Chunk

_TOKEN_RE = re.compile(r"[a-z0-9']+")
_STOPWORDS = frozenset(
    """
    a an the and or but if then when at by for with about into through during
    before after above below to from up down in out on off over under again
    further once here there all any both each few more most other some such no
    nor not only own same so than too very can will just should now is are was
    were be been being have has had having do does did doing would could ought
    i you he she it we they them his her its our their this that these those
    am what which who whom as of
    """.split()
)


class Embedder(ABC):
    """Embedding boundary. Swap in sentence-transformers via subclassing."""

    dimension: int

    @abstractmethod
    def embed(self, texts: list[str]) -> NDArray[np.float32]:
        """Embed a batch of texts; returns shape ``(len(texts), dimension)``."""


class HashingEmbedder(Embedder):
    """Deterministic bag-of-hashed-n-grams embedder.

    This is NOT a semantic model. It maps each text into a fixed-size vector
    by hashing character 3-grams into buckets and weighting them with a
    smoothed TF — enough to make lexical retrieval work offline and to make
    the pipeline fully testable without model downloads. Production use
    should subclass ``Embedder`` with a real model; see
    ``examples/`` and the ADR on pluggable embedders.
    """

    def __init__(self, dimension: int = 512) -> None:
        self.dimension = dimension

    def embed(self, texts: list[str]) -> NDArray[np.float32]:
        vectors = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for row, text in enumerate(texts):
            vectors[row] = self._embed_one(text)
        return self._normalize(vectors)

    # ------------------------------------------------------------------
    def _embed_one(self, text: str) -> NDArray[np.float32]:
        vec = np.zeros(self.dimension, dtype=np.float32)
        # stopword filtering matters here: common-word 3-grams otherwise
        # drown out the discriminative tokens that drive retrieval
        tokens = [w for w in _TOKEN_RE.findall(text.lower()) if w not in _STOPWORDS]
        grams = [w[i : i + 3] for w in tokens for i in range(max(0, len(w) - 2))]
        grams += tokens  # whole-word fallback
        if not grams:
            return vec
        counts: dict[int, float] = {}
        for g in grams:
            bucket = int(hashlib.md5(g.encode("utf-8")).hexdigest(), 16) % self.dimension
            counts[bucket] = counts.get(bucket, 0.0) + 1.0
        for bucket, count in counts.items():
            vec[bucket] = 1.0 + math.log(count)  # smoothed TF
        return vec

    @staticmethod
    def _normalize(vectors: NDArray[np.float32]) -> NDArray[np.float32]:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        return vectors / norms


class VectorStore(ABC):
    """Storage seam for chunk embeddings."""

    @abstractmethod
    def add(self, chunks: list[Chunk], embedder: Embedder) -> None:
        """Embed and store ``chunks``."""

    @abstractmethod
    def search(self, query: str, embedder: Embedder, top_k: int = 5) -> list[tuple[Chunk, float]]:
        """Return the ``top_k`` ``(chunk, cosine_score)`` pairs, best first."""

    @abstractmethod
    def get(self, chunk_id: str) -> Chunk | None:
        """Fetch a chunk by id, or None if absent."""

    @abstractmethod
    def __len__(self) -> int:
        """Number of stored chunks."""


class NumpyVectorStore(VectorStore):
    """In-memory store backed by a single dense matrix.

    Cosine similarity is computed as a matrix-vector product because vectors
    are L2-normalized at insert and query time. Exact, dependency-free, and
    deliberately simple — a reference implementation for the ``VectorStore``
    contract rather than a production index.
    """

    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        self._matrix: NDArray[np.float32] | None = None  # (n, dim), normalized

    # ------------------------------------------------------------------
    def add(self, chunks: list[Chunk], embedder: Embedder) -> None:
        if not chunks:
            return
        new = embedder.embed([c.text for c in chunks])
        if self._matrix is None:
            self._matrix = new
            self._chunks = list(chunks)
        else:
            if new.shape[1] != self._matrix.shape[1]:
                raise ValueError(
                    f"embedding dimension mismatch: store has {self._matrix.shape[1]}, "
                    f"got {new.shape[1]}"
                )
            self._matrix = np.vstack([self._matrix, new])
            self._chunks.extend(chunks)

    def search(self, query: str, embedder: Embedder, top_k: int = 5) -> list[tuple[Chunk, float]]:
        if self._matrix is None or not self._chunks:
            return []
        q = embedder.embed([query])  # (1, dim), normalized
        scores = (self._matrix @ q[0]).astype(np.float64)
        k = min(top_k, len(self._chunks))
        # argpartition for O(n) selection, then sort the top-k descending
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [(self._chunks[int(i)], float(scores[int(i)])) for i in top]

    def get(self, chunk_id: str) -> Chunk | None:
        for chunk in self._chunks:
            if chunk.id == chunk_id:
                return chunk
        return None

    def __len__(self) -> int:
        return len(self._chunks)
