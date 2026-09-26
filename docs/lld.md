# raguard — Low-Level Design

Module-by-module contracts, data structures, and algorithms. Companion to
[hld.md](hld.md).

## 1. Data model

### 1.1 `Chunk` (pydantic model, `ingest.py`)

| Field | Type | Notes |
| --- | --- | --- |
| `id` | `str` | `"{source}::chunk-{ordinal}::{sha1(text)[:10]}"` — content-derived, stable across runs |
| `text` | `str` | chunk body |
| `source` | `str` | provenance label (file path, doc index, corpus name) |
| `metadata` | `dict` | extension bag for adapters |

### 1.2 `EvalSlice` (dataclass, `evaluate.py`)

The single argument every metric receives. Contains the question, retrieved
chunks + scores, gold ids, and the optional `GenerationResult`. Keeps the
`Metric` protocol one-method and future-proof.

### 1.3 `GenerationResult` (dataclass, `generate.py`)

`answer: str`, `cited_chunk_ids: list[str]`, `provider_name: str`. Providers
must cite only ids they were actually given as context — the pipeline checks
this implicitly via faithfulness scoring.

### 1.4 `RunReport` (dataclass, `pipeline.py`)

`question, answer, cited_chunk_ids, retrieved, metrics, guardrails, blocked,
block_reason, context_snippets`. `to_dict()` returns a JSON-safe dict for
logging/observability.

### 1.5 `GuardrailFinding` / `GuardrailVerdict` (`guardrails.py`)

Finding: `detector, severity(INFO|WARN|BLOCK), message, matched`. Verdict:
list of findings + `blocked` property (any BLOCK → true).

## 2. `ingest.py` — chunking algorithm

`Chunker.chunk_text(text, source)`:

1. If `len(text) <= chunk_size`, one span covering the text.
2. Else slide a window of `chunk_size` with `overlap` re-anchoring between
   chunks: `next_start = max(0, end - overlap)`.
3. Within a window, `_preferred_end` walks back from the hard end to the
   last sentence terminator (`.`, `!`, `?`, `\n`). If the walk-back would
   keep less than a quarter of the window, split at the hard end instead —
   this avoids pathological tiny chunks on terminator-free stretches.
4. Each span becomes a `Chunk` with a deterministic id.

Ids are `sha1(text)`-derived, so the same document chunks identically on any
machine — required for citation stability and for labelled eval sets that
reference chunk ids.

## 3. `index.py` — embedding and storage

### 3.1 `HashingEmbedder`

- Character 3-grams of lowercased text + whole words, hashed
  (md5 → int) into `dimension` buckets.
- Weights: smoothed TF `1 + log(count)`.
- Output rows L2-normalized → cosine similarity reduces to a dot product.

Not a semantic model — a deterministic lexical vectorizer that makes the
pipeline runnable offline and the tests hermetic. See ADR 0001.

### 3.2 `NumpyVectorStore`

- State: `list[Chunk]` + a single `(n, dim)` float32 matrix (normalized).
- `add`: embed batch, `vstack` onto matrix; dimension mismatch → `ValueError`.
- `search`: `scores = matrix @ q` (dot product = cosine since both
  normalized), then `argpartition` to select top-k in O(n), sorted
  descending. Returns `(Chunk, float)` pairs.
- Exact search; cost is O(n·dim) per query — acceptable to ~100k chunks.

## 4. `retrieve.py`

`Retriever.retrieve(query)` = `store.search(query, embedder, top_k)` followed
by the optional `Reranker(query, hits) -> hits` hook. The hook type is a
plain `Callable`, so rerankers (cross-encoder, heuristic, rules) compose
without subclassing.

## 5. `generate.py` — provider boundary

`LLMProvider.generate(query, context) -> GenerationResult`.

`MockLLMProvider` (deterministic, offline):

1. Compute query content words (lowercase, stopword-filtered, len > 2).
2. Score each context chunk by content-word overlap.
3. If a unique best chunk exists (strictly greater than runner-up, > 0):
   extract from it the sentence with the highest query-word overlap, prefix
   it with "Based on the provided material: ", and cite that chunk.
4. Else return the refusal sentence and cite nothing.

Step 4 mirrors the desired production behaviour: ambiguous or missing
context → refuse rather than confabulate. This makes the mock useful for
guardrail and faithfulness testing, not just plumbing.

## 6. `evaluate.py` — metric engine

- `Metric` protocol: `name: str`, `score(run: EvalSlice) -> float` in
  `[0,1]` or NaN when not applicable (e.g. no gold labels).
- `content_words`: regex token extraction + stopword filter.
- `CitationFaithfulness`: sentence-split on `(?<=[.!?])\s+`; per-sentence
  coverage of cited-chunk vocabulary ≥ 0.5 → supported. Uncited answers
  score 1.0 iff refusal-shaped, else 0.0.
- `HallucinationRisk`: renormalises its 0.4/0.4/0.2 weights over the
  non-NaN components.

NaN values are dropped from `RunReport.metrics` so reports stay clean when
gold labels are absent.

## 7. `guardrails.py` — detectors

- `PIIDetector`: ordered `(label, compiled_regex, severity)` table; scans
  all patterns, reports the worst severity and up to 5 example matches.
  Extensible via `extra_patterns`.
- `PromptInjectionFilter`: word-boundary patterns targeting instruction
  override, role impersonation, delimiter escape, exfiltration, tool abuse.
  Any hit → BLOCK.
- `SensitiveDomainGuard`: tuple of `SafetyRule(label, compiled_regex,
  guidance, severity)`; each matching rule emits a finding carrying its
  guidance text (the "what to do instead" is part of the output).
- `GuardrailSuite`: facade composing the three; `sensitive_mode` flag
  controls whether the sensitive checklist runs on questions and outputs.

All detectors are pure functions of their input string → trivially testable,
cachable, and safe to run in parallel.

## 8. `pipeline.py` — orchestration

`Raguard.run(question, gold_chunk_ids=None)`:

1. `guardrails.check_question` → BLOCK short-circuits to a blocked report.
2. `retriever.retrieve` → context chunks.
3. `provider.generate` → answer + citations.
4. `guardrails.check_output` → findings merged into the verdict.
5. Build `EvalSlice`; run built-in + extra metrics; drop NaN.
6. Return `RunReport`.

`Raguard.from_documents` is the convenience constructor: chunks raw strings
(`doc-0`, `doc-1`, …), indexes them, wires defaults. All defaults are
replaceable via constructor args — the class never imports concrete
providers at module scope (lazy import inside `_default_provider`).

## 9. Test strategy

| Area | File | Approach |
| --- | --- | --- |
| Chunker | `tests/test_chunker.py` | determinism, size/overlap invariants, sentence boundaries, validation |
| Store | `tests/test_vector_store.py` | ranking correctness, normalization bounds, dimension mismatch, empty store |
| Metrics | `tests/test_metrics.py` | perfect/drift/refusal faithfulness; precision/recall arithmetic; groundedness |
| Guardrails | `tests/test_guardrails.py` | positive/negative cases per detector; suite blocking |
| Pipeline | `tests/test_pipeline.py` | end-to-end scored report, gold metrics, blocking, refusal, rerank hook, JSON serialisation |

Everything runs on the managed runtime with numpy + pydantic + pytest —
no downloads, no API keys.

## 10. Extension recipes

**Real embeddings:** subclass `Embedder` wrapping `sentence_transformers`,
pass to `Raguard.from_documents(embedder=...)`.

**Chroma/pgvector:** implement `VectorStore` (4 methods), pass as `store=`.

**OpenAI generation:** implement `LLMProvider.generate` using the `openai`
package; enforce citation discipline by returning ids from `context`.

**ragas:** wrap each ragas metric in a class with `name` + `score(EvalSlice)`
calling the ragas API; pass via `extra_metrics`.
