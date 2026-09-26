"""Chunker behaviour tests."""

import pytest

from raguard import Chunker


def test_chunk_ids_are_deterministic():
    chunker = Chunker(chunk_size=60, overlap=10)
    text = (
        "Alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu. "
        "Nu xi omicron pi rho sigma tau upsilon phi chi psi omega."
    )
    first = chunker.chunk_text(text, source="doc-a")
    second = chunker.chunk_text(text, source="doc-a")
    assert [c.id for c in first] == [c.id for c in second]
    assert [c.text for c in first] == [c.text for c in second]


def test_chunks_respect_size_and_overlap():
    chunker = Chunker(chunk_size=100, overlap=20)
    text = "Lorem ipsum dolor sit amet. " * 100
    chunks = chunker.chunk_text(text, source="doc-b")
    assert len(chunks) > 1
    joined = chunks[0].text + chunks[1].text
    # consecutive chunks must overlap: the tail of chunk 0 reappears at the
    # start of the re-windowed chunk 1
    tail = chunks[0].text[-20:]
    assert tail.strip() and tail in joined[len(chunks[0].text) - 20 :]
    for chunk in chunks:
        assert len(chunk.text) <= chunker.chunk_size + 8  # sentence slack


def test_short_text_produces_single_chunk():
    chunker = Chunker()
    chunks = chunker.chunk_text("Short and sweet.", source="doc-c")
    assert len(chunks) == 1
    assert chunks[0].source == "doc-c"


def test_invalid_config_rejected():
    with pytest.raises(ValueError):
        Chunker(chunk_size=0)
    with pytest.raises(ValueError):
        Chunker(chunk_size=10, overlap=10)


def test_sentence_boundaries_avoid_midword_cuts():
    chunker = Chunker(chunk_size=80, overlap=0)
    text = (
        "First sentence here is long enough. Second sentence also quite long. "
        "Third sentence finishes the document."
    )
    chunks = chunker.chunk_text(text, source="doc-d")
    for chunk in chunks[:-1]:
        assert chunk.text.rstrip().endswith((".", "!", "?", "\n"))
