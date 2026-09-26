"""Retrieval with an optional rerank hook."""

from __future__ import annotations

from collections.abc import Callable

from raguard.index import Embedder, VectorStore
from raguard.ingest import Chunk

# A reranker takes the query and the raw retrieval hits and returns them
# re-ordered (or filtered) with updated scores. Wire in a cross-encoder here.
Reranker = Callable[[str, list[tuple[Chunk, float]]], list[tuple[Chunk, float]]]


class Retriever:
    """Thin orchestration layer over a ``VectorStore``.

    Args:
        store: The vector store to search.
        embedder: Embedder used for query encoding.
        top_k: How many chunks to fetch per query.
        reranker: Optional post-processing hook applied to raw hits before
            they are returned. Identity by default.
    """

    def __init__(
        self,
        store: VectorStore,
        embedder: Embedder,
        top_k: int = 5,
        reranker: Reranker | None = None,
    ) -> None:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        self.store = store
        self.embedder = embedder
        self.top_k = top_k
        self.reranker = reranker

    def retrieve(self, query: str) -> list[tuple[Chunk, float]]:
        """Return ranked ``(chunk, score)`` hits for ``query``."""
        hits = self.store.search(query, self.embedder, top_k=self.top_k)
        if self.reranker is not None:
            hits = self.reranker(query, hits)
        return hits
