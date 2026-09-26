"""raguard — evaluation-first RAG with deterministic metrics and guardrails.

Public API re-exports for the most common entry points::

    from raguard import Raguard, Chunker, HashingEmbedder, NumpyVectorStore
"""

from raguard.evaluate import (
    AnswerGroundedness,
    CitationFaithfulness,
    EvalSlice,
    HallucinationRisk,
    Metric,
    RetrievalPrecision,
    RetrievalRecall,
)
from raguard.generate import GenerationResult, LLMProvider, MockLLMProvider
from raguard.guardrails import (
    GuardrailFinding,
    GuardrailSuite,
    GuardrailVerdict,
    PIIDetector,
    PromptInjectionFilter,
    SensitiveDomainGuard,
)
from raguard.index import Embedder, HashingEmbedder, NumpyVectorStore, VectorStore
from raguard.ingest import Chunk, Chunker
from raguard.pipeline import Raguard, RunReport
from raguard.retrieve import Retriever

__version__ = "0.1.0"

__all__ = [
    "AnswerGroundedness",
    "Chunk",
    "Chunker",
    "CitationFaithfulness",
    "Embedder",
    "EvalSlice",
    "GenerationResult",
    "GuardrailFinding",
    "GuardrailSuite",
    "GuardrailVerdict",
    "HallucinationRisk",
    "HashingEmbedder",
    "LLMProvider",
    "Metric",
    "MockLLMProvider",
    "NumpyVectorStore",
    "PIIDetector",
    "PromptInjectionFilter",
    "RetrievalPrecision",
    "RetrievalRecall",
    "Retriever",
    "Raguard",
    "RunReport",
    "SensitiveDomainGuard",
    "VectorStore",
    "__version__",
]
