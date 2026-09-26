# ADR 0001: Pluggable embedders and stores behind ABCs

- Status: Accepted
- Date: 2026-09-26

## Context

raguard's value is in evaluation and guardrails, not in any specific
embedding model or vector database. But a pipeline that only runs against
one vendor's stack is hard to test, hard to CI, and hard to adopt. The
counter-argument for hard-coding a strong default (e.g. sentence-transformers
+ Chroma) is real: most users want the "good" path out of the box.

## Decision

Define two narrow ABCs — `Embedder` (one method: `embed(list[str]) ->
ndarray`) and `VectorStore` (four methods: `add`, `search`, `get`, `len`) —
and ship honest reference implementations that run anywhere:

- `HashingEmbedder`: character n-gram hashing with smoothed TF weights.
  Lexical, not semantic — its docstring says so.
- `NumpyVectorStore`: dense-matrix exact cosine search. O(n·k), fine to
  ~100k chunks — its docstring says that too.

Optional dependencies (`sentence-transformers`, `chromadb`, `openai`,
`ragas`) stay in extras; nothing in the core imports them.

## Consequences

- The full pipeline — ingest → retrieve → guard → generate → evaluate —
  runs offline with zero downloads, so CI and the example scripts are
  hermetic.
- The evaluation harness is testable without mocking half the internet;
  metric tests pin behaviour, not vendor responses.
- Adapters are ~30 lines each; users bring their own stack behind the same
  contracts.
- Cost: lexical retrieval quality is lower than semantic embeddings, so the
  shipped eval numbers on the hashing embedder understate what a real
  embedder achieves. `evals/run_eval.py` is explicitly the place to measure
  that gap when you swap components.

## Alternatives considered

- **Hard-code sentence-transformers as the default.** Rejected: pulls torch
  into every install, makes CI slow and non-hermetic, and couples the
  project's identity to a vendor.
- **Depend on Chroma as the store.** Rejected for the same reason; the store
  ABC keeps pgvector/FAISS/Redis equally available.
- **No ABCs, duck-typed protocols everywhere.** Rejected: ABCs give early,
  clear errors at construction time, which matters for a library whose
  users will write adapters.
