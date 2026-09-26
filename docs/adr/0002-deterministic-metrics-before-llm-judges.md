# ADR 0002: Deterministic metrics before LLM judges

- Status: Accepted
- Date: 2026-09-26

## Context

Answer-quality evaluation in RAG systems is often done with LLM-as-a-judge:
prompt a strong model to rate faithfulness/groundedness on a scale. Judges
are genuinely better at semantic nuance (paraphrase, implication) than any
lexical heuristic. But they carry costs that matter enormously at the bottom
of the stack:

- **Cost per evaluation** — a judge call per answer is real money at
  production volume.
- **Nondeterminism** — temperature, provider version, and prompt drift make
  scores irreproducible; CI cannot assert on them.
- **Dependency** — your evaluation quality is capped by, and coupled to, a
  third-party model.
- **Circular-trust risk** — using one black-box model to certify another's
  outputs can inherit and launder the first model's blind spots.

## Decision

All built-in metrics are deterministic functions of text and ids: content-word
coverage for faithfulness and groundedness, set arithmetic for retrieval
precision/recall, a weighted composite for hallucination-risk triage. They
run on every request, in CI, with zero marginal cost and exact
reproducibility.

Where judges add value (paraphrase-heavy corpora, implication-rich domains),
they plug in through the `Metric` protocol as *additional* metrics — ragas
adapters included — never as replacements for the deterministic floor.

## Consequences

- `RunReport.metrics` is reproducible: the same run always produces the same
  numbers, so regression tests can assert thresholds.
- Evaluation is free at request time, enabling "score everything, triage
  the tail" instead of "sample a few and judge them."
- Known blind spots are documented rather than hidden: lexical coverage
  misses paraphrase. That trade is stated in the README and in each
  metric's docstring.
- The `EvalSlice` abstraction keeps judge inputs well-defined when they are
  added: question, retrieval, generation — one object, one contract.

## Alternatives considered

- **ragas as the evaluation core.** Rejected: makes the core depend on LLM
  calls for its identity; great tool, wrong layer to build on.
- **Deterministic metrics only, no judge seam.** Rejected: for
  paraphrase-heavy corpora lexical metrics saturate quickly; pretending
  otherwise would overclaim, which this project refuses to do.
- **Hybrid from day one.** Rejected: forces API keys on anyone who runs the
  examples; the seam ships first, the judges opt in.
