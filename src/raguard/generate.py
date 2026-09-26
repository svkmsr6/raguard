"""Answer generation behind a provider boundary.

The pipeline never calls an LLM directly; it calls an ``LLMProvider``.
``MockLLMProvider`` keeps the entire pipeline runnable offline and makes the
evaluation metrics deterministic in tests — it answers strictly from the
retrieved context, which is exactly the behaviour you want when validating
the harness itself.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from raguard.ingest import Chunk


@dataclass
class GenerationResult:
    """What a provider produced for one query."""

    answer: str
    cited_chunk_ids: list[str] = field(default_factory=list)
    provider_name: str = "unknown"


class LLMProvider(ABC):
    """Generation boundary. Implement with OpenAI, Anthropic, a local model…"""

    name: str = "abstract"

    @abstractmethod
    def generate(self, query: str, context: list[Chunk]) -> GenerationResult:
        """Answer ``query`` using only ``context`` chunks."""


class MockLLMProvider(LLMProvider):
    """Deterministic extractive provider for offline runs and tests.

    Behaviour:
      * If the query's most distinctive content words appear in exactly one
        chunk, it extracts the sentence containing the most matches from that
        chunk and cites it.
      * Otherwise it returns a refusal-style fallback and cites nothing —
        mirroring what a guarded system should do when context is ambiguous.
    """

    name = "mock-extractive"
    _STOPWORDS = frozenset(
        "a an the is are was were what who when where which why how can does do "
        "i me my we our you your it its this that of in on for to from with and "
        "or not be been being have has had should could would will about".split()
    )

    def generate(self, query: str, context: list[Chunk]) -> GenerationResult:
        content_words = self._content_words(query)
        scored = [
            (chunk, len(content_words & set(self._content_words(chunk.text)))) for chunk in context
        ]
        scored = [(c, s) for c, s in scored if s > 0]
        scored.sort(key=lambda pair: pair[1], reverse=True)

        if scored and scored[0][1] > 0 and (len(scored) == 1 or scored[0][1] > scored[1][1]):
            best = scored[0][0]
            sentence = self._best_sentence(best.text, content_words)
            return GenerationResult(
                answer=f"Based on the provided material: {sentence}",
                cited_chunk_ids=[best.id],
                provider_name=self.name,
            )
        return GenerationResult(
            answer=("I do not have enough reliable context to answer that confidently."),
            cited_chunk_ids=[],
            provider_name=self.name,
        )

    # ------------------------------------------------------------------
    @classmethod
    def _content_words(cls, text: str) -> set[str]:
        return {
            w.strip(".,!?;:\"'()").lower()
            for w in text.split()
            if len(w) > 2 and w.strip(".,!?;:\"'()").lower() not in cls._STOPWORDS
        }

    @staticmethod
    def _best_sentence(text: str, query_words: set[str]) -> str:
        sentences = [s.strip() for s in text.replace("\n", " ").split(".") if s.strip()]
        if not sentences:
            return text[:300]
        best = max(
            sentences,
            key=lambda s: len(query_words & set(s.lower().split())),
        )
        return best + ("." if not best.endswith(".") else "")
