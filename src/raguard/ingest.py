"""Document loading and chunking.

The chunker is deliberately deterministic and dependency-free: the same input
always produces the same chunk ids, which matters when you want stable
references for citations and labelled evaluation sets.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import BaseModel, Field


class Chunk(BaseModel):
    """A single retrievable unit of text."""

    id: str
    text: str
    source: str = "unknown"
    metadata: dict = Field(default_factory=dict)


@dataclass
class Chunker:
    """Fixed-size chunker with overlap.

    Splits on sentence-ish boundaries where possible (``. ! ? \\n``) so that
    chunks stay semantically coherent instead of cutting mid-sentence.

    Args:
        chunk_size: Target maximum characters per chunk.
        overlap: Characters of overlap between consecutive chunks. Must be
            smaller than ``chunk_size``.
        respect_sentence_boundaries: If True, prefer splitting after
            punctuation instead of at exactly ``chunk_size`` characters.
    """

    chunk_size: int = 512
    overlap: int = 64
    respect_sentence_boundaries: bool = True
    _sources: dict = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if self.overlap < 0:
            raise ValueError("overlap must be non-negative")
        if self.overlap >= self.chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def chunk_text(self, text: str, source: str = "inline") -> list[Chunk]:
        """Split ``text`` into overlapping chunks with deterministic ids."""
        spans = self._split_spans(text)
        chunks: list[Chunk] = []
        for i, (start, end) in enumerate(spans):
            chunk_text = text[start:end].strip()
            if not chunk_text:
                continue
            chunk_id = self._stable_id(source, i, chunk_text)
            chunks.append(Chunk(id=chunk_id, text=chunk_text, source=source))
        return chunks

    def chunk_file(self, path: str | Path) -> list[Chunk]:
        """Load a UTF-8 text file and chunk it, using the path as source."""
        p = Path(path)
        return self.chunk_text(p.read_text(encoding="utf-8"), source=str(p))

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------
    def _split_spans(self, text: str) -> list[tuple[int, int]]:
        """Return (start, end) character spans covering ``text``."""
        if len(text) <= self.chunk_size:
            return [(0, len(text))] if text.strip() else []

        spans: list[tuple[int, int]] = []
        start = 0
        n = len(text)
        while start < n:
            hard_end = min(start + self.chunk_size, n)
            end = self._preferred_end(text, start, hard_end)
            if end <= start:  # pragma: no cover - defensive
                end = hard_end
            spans.append((start, end))
            if end >= n:
                break
            start = max(0, end - self.overlap)
        return spans

    def _preferred_end(self, text: str, start: int, hard_end: int) -> int:
        """Pick a split point at or before ``hard_end``.

        With sentence boundaries enabled, walk back from ``hard_end`` to the
        last sentence terminator. If none exists in the window (or the walk
        back would swallow almost the whole chunk), split at ``hard_end``.
        """
        if not self.respect_sentence_boundaries or hard_end >= len(text):
            return hard_end
        window = text[start:hard_end]
        last_term = -1
        for i, ch in enumerate(window):
            if ch in ".!?\n":
                last_term = i
        if last_term <= self.chunk_size // 4:
            return hard_end
        return start + last_term + 1

    @staticmethod
    def _stable_id(source: str, ordinal: int, text: str) -> str:
        digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]
        return f"{source}::chunk-{ordinal}::{digest}"
