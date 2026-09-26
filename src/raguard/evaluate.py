"""Deterministic evaluation metrics — the core of raguard.

Every metric here is computed from text and ids alone: no LLM-as-judge, no
external API, same input → same score. That makes them free to run on every
request, reproducible in CI, and impossible to silently drift when a model
provider changes. Where a judgement genuinely requires a model (nuance,
paraphrase, implication), the ``Metric`` protocol is the seam where
LLM-judge wrappers (e.g. ragas) plug in — see ADR 0002.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Protocol

from raguard.generate import GenerationResult
from raguard.ingest import Chunk

_WORD_RE = re.compile(r"[a-z0-9']+")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def content_words(text: str) -> set[str]:
    """Lowercased alphanumeric tokens with stopwords removed."""
    stop = _stopwords()
    return {w for w in _WORD_RE.findall(text.lower()) if w not in stop and len(w) > 2}


@dataclass(frozen=True)
class _Stopwords:
    words: frozenset[str] = frozenset(
        """
        a an the and or but if then else when at by for with about into through
        during before after above below to from up down in out on off over under
        again further once here there all any both each few more most other some
        such no nor not only own same so than too very can will just should now
        is are was were be been being have has had having do does did doing would
        could ought i you he she it we they them his her its our their this that
        these those am what which who whom as of
        """.split()
    )


_STOPWORDS = _Stopwords().words


def _stopwords() -> frozenset[str]:
    return _STOPWORDS


# ----------------------------------------------------------------------
# Metric protocol — the extension point for ragas / LLM judges
# ----------------------------------------------------------------------
class Metric(Protocol):
    """A named scorer over a full run slice.

    Custom metrics (including ragas adapters) implement this protocol and are
    attached to the pipeline via ``Raguard(extra_metrics=[...])``.
    """

    name: str

    def score(self, run: EvalSlice) -> float:
        """Return a score in ``[0, 1]`` (or NaN if not applicable)."""


@dataclass
class EvalSlice:
    """Everything a metric might need from one pipeline run."""

    question: str
    retrieved: list[Chunk] = field(default_factory=list)
    retrieved_scores: list[float] = field(default_factory=list)
    gold_chunk_ids: list[str] = field(default_factory=list)
    generation: GenerationResult | None = None

    @property
    def retrieved_ids(self) -> list[str]:
        return [c.id for c in self.retrieved]


# ----------------------------------------------------------------------
# Built-in deterministic metrics
# ----------------------------------------------------------------------
class RetrievalPrecision:
    """Fraction of retrieved chunks that are gold-relevant.

    ``|retrieved ∩ gold| / |retrieved|`` — measures how much noise the
    generator has to wade through; noise is where hallucinations breed.
    """

    name = "retrieval_precision"

    def score(self, run: EvalSlice) -> float:
        gold = set(run.gold_chunk_ids)
        retrieved = run.retrieved_ids
        if not retrieved:
            return 0.0
        if not gold:
            return float("nan")
        return len(gold & set(retrieved)) / len(retrieved)


class RetrievalRecall:
    """Fraction of gold chunks that retrieval surfaced.

    ``|retrieved ∩ gold| / |gold|`` — a correct answer is impossible when
    the supporting evidence never reached the generator.
    """

    name = "retrieval_recall"

    def score(self, run: EvalSlice) -> float:
        gold = set(run.gold_chunk_ids)
        if not gold:
            return float("nan")
        return len(gold & set(run.retrieved_ids)) / len(gold)


class CitationFaithfulness:
    """Do cited chunks actually support the answer's claims?

    For each answer sentence, we check whether its content words are
    lexically covered by the union of the *cited* chunks (not the whole
    retrieval set — that is the difference from groundedness). Score is the
    fraction of supported sentences. A sentence unsupported by its citation
    is the classic signature of citation drift / fabricated references.
    """

    name = "citation_faithfulness"
    coverage_threshold = 0.5

    def score(self, run: EvalSlice) -> float:
        if run.generation is None:
            return float("nan")
        cited = [c for c in run.retrieved if c.id in set(run.generation.cited_chunk_ids)]
        answer = run.generation.answer
        sentences = [s for s in _SENTENCE_SPLIT_RE.split(answer.strip()) if s.strip()]
        if not sentences:
            return 0.0
        if not cited:
            # The answer cites nothing — only acceptable if it is a refusal.
            return 1.0 if self._is_refusal(answer) else 0.0
        cited_vocab = set().union(*(content_words(c.text) for c in cited))
        supported = sum(1 for s in sentences if self._sentence_supported(s, cited_vocab))
        return supported / len(sentences)

    def _sentence_supported(self, sentence: str, cited_vocab: set[str]) -> bool:
        words = content_words(sentence)
        if not words:
            return True  # punctuation-only fragment; not a claim
        overlap = len(words & cited_vocab) / len(words)
        return overlap >= self.coverage_threshold

    @staticmethod
    def _is_refusal(answer: str) -> bool:
        return any(
            marker in answer.lower()
            for marker in ("do not have enough", "cannot answer", "no reliable context")
        )


class AnswerGroundedness:
    """Is the answer lexically grounded in *any* retrieved chunk?

    Coverage of the answer's content words by the full retrieval set. Low
    groundedness with high citation faithfulness is a red flag worth
    investigating (cited sources fine, answer drawing on nothing).
    """

    name = "answer_groundedness"

    def score(self, run: EvalSlice) -> float:
        if run.generation is None:
            return float("nan")
        answer_words = content_words(run.generation.answer)
        if not answer_words:
            return 0.0
        if not run.retrieved:
            return 0.0
        context_vocab = set().union(*(content_words(c.text) for c in run.retrieved))
        return len(answer_words & context_vocab) / len(answer_words)


class HallucinationRisk:
    """Heuristic composite: 1.0 means clean, 0.0 means high risk.

    Combines the three deterministic signals above into one triage number:

        risk = 1 − (0.4 · groundedness + 0.4 · citation_faithfulness
                    + 0.2 · retrieval_precision)

    It is a prioritisation aid, not a verdict — see docs/hld.md for the
    failure-mode reasoning and docs/adr/0002 for why no LLM judge sits here.
    """

    name = "hallucination_risk"

    def __init__(self) -> None:
        self._groundedness = AnswerGroundedness()
        self._faithfulness = CitationFaithfulness()
        self._precision = RetrievalPrecision()

    def score(self, run: EvalSlice) -> float:
        parts = [
            self._groundedness.score(run),
            self._faithfulness.score(run),
            self._precision.score(run),
        ]
        usable = [p for p in parts if p == p]  # drop NaN
        if not usable:
            return float("nan")
        # Re-weight when gold labels are absent (precision NaN).
        weights = [0.4, 0.4, 0.2][: len(usable)]
        total = sum(weights)
        risk = 1.0 - sum(w * p for w, p in zip(weights, usable, strict=True)) / total
        return min(max(risk, 0.0), 1.0)
