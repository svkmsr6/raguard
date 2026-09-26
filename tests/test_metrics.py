"""Deterministic metric tests."""

from raguard import (
    AnswerGroundedness,
    Chunk,
    CitationFaithfulness,
    EvalSlice,
    GenerationResult,
    RetrievalPrecision,
    RetrievalRecall,
)

CHUNK_A = Chunk(id="a", text="Grounding means paying attention to the present moment.")
CHUNK_B = Chunk(id="b", text="The water cycle is powered by the sun.")


def _run(answer: str, cited: list[str], retrieved=None) -> EvalSlice:
    return EvalSlice(
        question="q",
        retrieved=retrieved or [CHUNK_A, CHUNK_B],
        gold_chunk_ids=["a"],
        generation=GenerationResult(answer=answer, cited_chunk_ids=cited),
    )


def _run_with(gold: list[str], retrieved: list[Chunk]) -> EvalSlice:
    return EvalSlice(question="q", retrieved=retrieved, gold_chunk_ids=gold)


def test_retrieval_precision():
    metric = RetrievalPrecision()
    assert metric.score(_run_with(["a"], [CHUNK_A, CHUNK_B])) == 0.5
    assert metric.score(_run_with(["a"], [CHUNK_A])) == 1.0
    assert metric.score(_run_with(["a"], [])) == 0.0


def test_retrieval_recall():
    metric = RetrievalRecall()
    assert metric.score(_run_with(["a", "b"], [CHUNK_A])) == 0.5
    assert metric.score(_run_with(["a"], [CHUNK_A, Chunk(id="z", text="t")])) == 1.0


def test_citation_faithfulness_perfect_when_supported():
    metric = CitationFaithfulness()
    run = _run("Grounding means paying attention to the present moment.", ["a"])
    assert metric.score(run) == 1.0


def test_citation_faithfulness_zero_for_fabricated_citation():
    """An answer about water cycles citing the grounding chunk only."""
    metric = CitationFaithfulness()
    run = _run(
        "The water cycle is powered entirely by the sun.",
        ["a"],  # cites grounding chunk — drift!
        retrieved=[CHUNK_A],
    )
    assert metric.score(run) == 0.0


def test_citation_faithfulness_refusal_with_no_citations_is_clean():
    metric = CitationFaithfulness()
    run = _run("I do not have enough reliable context to answer that confidently.", [])
    assert metric.score(run) == 1.0
    # but an unsupported claim with no citations must score 0
    run2 = _run("The sun powers the water cycle.", [], retrieved=[CHUNK_B])
    assert metric.score(run2) == 0.0


def test_answer_groundedness():
    metric = AnswerGroundedness()
    grounded = _run("Grounding is about the present moment.", ["a"])
    assert metric.score(grounded) == 1.0
    ungrounded = _run("Elephants are the largest land mammals.", ["a"])
    assert metric.score(ungrounded) < 0.5
