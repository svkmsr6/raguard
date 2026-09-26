"""End-to-end orchestration: ingest → retrieve → guard → generate → evaluate."""

from __future__ import annotations

from dataclasses import dataclass, field

from raguard.evaluate import (
    AnswerGroundedness,
    CitationFaithfulness,
    EvalSlice,
    HallucinationRisk,
    Metric,
    RetrievalPrecision,
    RetrievalRecall,
)
from raguard.generate import GenerationResult, LLMProvider
from raguard.guardrails import GuardrailSuite, GuardrailVerdict
from raguard.index import Embedder, VectorStore
from raguard.ingest import Chunk, Chunker
from raguard.retrieve import Retriever


@dataclass
class RunReport:
    """The scored result of one ``Raguard.run`` call."""

    question: str
    answer: str
    cited_chunk_ids: list[str]
    retrieved: list[Chunk]
    metrics: dict[str, float]
    guardrails: GuardrailVerdict
    blocked: bool = False
    block_reason: str | None = None
    context_snippets: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Serialize to a JSON-friendly dict."""
        return {
            "question": self.question,
            "answer": self.answer,
            "cited_chunk_ids": self.cited_chunk_ids,
            "retrieved_ids": [c.id for c in self.retrieved],
            "metrics": self.metrics,
            "blocked": self.blocked,
            "block_reason": self.block_reason,
            "guardrails": [
                {
                    "detector": f.detector,
                    "severity": f.severity.value,
                    "message": f.message,
                }
                for f in self.guardrails.triggered
                if f.severity.value != "info"
            ],
        }


class Raguard:
    """Composable RAG pipeline with built-in evaluation and guardrails.

    Typical usage::

        pipeline = Raguard.from_documents(docs)
        report = pipeline.run("What is grounding?", gold_chunk_ids=[...])

    Every run returns a ``RunReport`` whose metrics were computed
    deterministically, with no external API calls, alongside any guardrail
    findings. Extra metrics (e.g. a ragas adapter implementing ``Metric``)
    can be supplied via ``extra_metrics``.
    """

    def __init__(
        self,
        retriever: Retriever,
        provider: LLMProvider,
        guardrails: GuardrailSuite | None = None,
        extra_metrics: list[Metric] | None = None,
    ) -> None:
        self.retriever = retriever
        self.provider = provider
        self.guardrails = guardrails or GuardrailSuite()
        self._builtin_metrics: list[Metric] = [
            RetrievalPrecision(),
            RetrievalRecall(),
            CitationFaithfulness(),
            AnswerGroundedness(),
            HallucinationRisk(),
        ]
        self._extra_metrics = list(extra_metrics or [])

    # ------------------------------------------------------------------
    @classmethod
    def from_documents(
        cls,
        documents: list[str],
        *,
        store: VectorStore | None = None,
        embedder: Embedder | None = None,
        chunker: Chunker | None = None,
        provider: LLMProvider | None = None,
        top_k: int = 5,
        guardrails: GuardrailSuite | None = None,
        extra_metrics: list[Metric] | None = None,
    ) -> Raguard:
        """Build a pipeline from raw text strings (sources: doc-0, doc-1, …)."""
        from raguard.index import HashingEmbedder, NumpyVectorStore

        chunker = chunker or Chunker()
        embedder = embedder or HashingEmbedder()
        store = store or NumpyVectorStore()
        for i, doc in enumerate(documents):
            store.add(chunker.chunk_text(doc, source=f"doc-{i}"), embedder)
        retriever = Retriever(store, embedder, top_k=top_k)
        return cls(
            retriever=retriever,
            provider=provider or _default_provider(),
            guardrails=guardrails,
            extra_metrics=extra_metrics,
        )

    # ------------------------------------------------------------------
    def run(self, question: str, gold_chunk_ids: list[str] | None = None) -> RunReport:
        """Answer ``question`` and score the whole run.

        Guardrails run first: a BLOCKed question short-circuits generation
        and returns a report with ``blocked=True`` and no fabricated answer.
        """
        verdict = self.guardrails.check_question(question)
        blocking = next((f for f in verdict.triggered if f.severity.value == "block"), None)
        if blocking is not None:
            return RunReport(
                question=question,
                answer="",
                cited_chunk_ids=[],
                retrieved=[],
                metrics={},
                guardrails=verdict,
                blocked=True,
                block_reason=f"{blocking.detector}: {blocking.message}",
            )

        hits = self.retriever.retrieve(question)
        retrieved = [chunk for chunk, _ in hits]
        generation: GenerationResult = self.provider.generate(question, retrieved)

        output_verdict = self.guardrails.check_output(generation.answer)
        verdict.findings.extend(output_verdict.triggered)

        slice_ = EvalSlice(
            question=question,
            retrieved=retrieved,
            retrieved_scores=[score for _, score in hits],
            gold_chunk_ids=gold_chunk_ids or [],
            generation=generation,
        )
        metrics: dict[str, float] = {}
        for metric in [*self._builtin_metrics, *self._extra_metrics]:
            value = metric.score(slice_)
            if value == value:  # skip NaN so reports stay clean
                metrics[metric.name] = round(float(value), 4)

        return RunReport(
            question=question,
            answer=generation.answer,
            cited_chunk_ids=generation.cited_chunk_ids,
            retrieved=retrieved,
            metrics=metrics,
            guardrails=verdict,
            context_snippets=[c.text[:200] for c in retrieved],
        )


def _default_provider() -> LLMProvider:
    from raguard.generate import MockLLMProvider

    return MockLLMProvider()
