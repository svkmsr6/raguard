"""In-memory vector store correctness tests."""

import numpy as np
import pytest

from raguard import Chunk, HashingEmbedder, NumpyVectorStore


@pytest.fixture()
def store() -> NumpyVectorStore:
    return NumpyVectorStore()


@pytest.fixture()
def embedder() -> HashingEmbedder:
    return HashingEmbedder(dimension=128)


def test_add_and_len(store, embedder):
    assert len(store) == 0
    store.add([Chunk(id="c1", text="retrieval augmented generation")], embedder)
    store.add([Chunk(id="c2", text="vector databases and embeddings")], embedder)
    assert len(store) == 2


def test_get_by_id(store, embedder):
    chunk = Chunk(id="c1", text="some text", source="s")
    store.add([chunk], embedder)
    assert store.get("c1") is chunk
    assert store.get("missing") is None


def test_search_returns_cosine_ranking(store, embedder):
    chunks = [
        Chunk(id="vec", text="vector embeddings similarity search index"),
        Chunk(id="cook", text="baking bread flour oven recipe"),
        Chunk(id="db", text="database transactions acid durability"),
    ]
    store.add(chunks, embedder)
    hits = store.search("how do vector similarity searches work", embedder, top_k=2)
    assert len(hits) == 2
    ids = [c.id for c, _ in hits]
    assert ids[0] == "vec"
    scores = [s for _, s in hits]
    assert scores[0] >= scores[1] > 0.0


def test_cosine_scores_are_normalized(store, embedder):
    store.add([Chunk(id="a", text="hello world"), Chunk(id="b", text="goodbye")], embedder)
    hits = store.search("hello", embedder, top_k=2)
    for _, score in hits:
        assert -1.0 <= score <= 1.0


def test_search_empty_store(store, embedder):
    assert store.search("anything", embedder, top_k=3) == []


def test_dimension_mismatch_rejected(store, embedder):
    store.add([Chunk(id="a", text="text one")], embedder)
    other = HashingEmbedder(dimension=64)
    with pytest.raises(ValueError):
        store.add([Chunk(id="b", text="text two")], other)


def test_embeddings_are_l2_normalized():
    emb = HashingEmbedder(dimension=64)
    vectors = emb.embed(["some query text", ""])
    norms = np.linalg.norm(vectors, axis=1)
    assert np.isclose(norms[0], 1.0)
