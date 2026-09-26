"""Evaluate retrieval precision/recall on the labelled eval dataset.

Runs fully offline (hashing embedder + mock provider)::

    python evals/run_eval.py [--dataset evals/eval_dataset.json] [--top-k 3]

The per-question scores come from the same deterministic metrics the
pipeline reports at request time — this script is where you check that the
retrieval half of the system is actually doing its job before you trust any
answer-level scores.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from raguard import (  # noqa: E402
    Chunk,
    HashingEmbedder,
    MockLLMProvider,
    NumpyVectorStore,
    Raguard,
    Retriever,
)

DATASET_PATH = Path(__file__).resolve().parent / "eval_dataset.json"


def load_dataset(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def build_pipeline(dataset: dict, top_k: int) -> Raguard:
    embedder = HashingEmbedder(dimension=512)
    store = NumpyVectorStore()
    store.add(
        [Chunk(id=c["id"], text=c["text"], source=dataset["name"]) for c in dataset["chunks"]],
        embedder,
    )
    return Raguard(
        retriever=Retriever(store, embedder, top_k=top_k),
        provider=MockLLMProvider(),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH)
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    dataset = load_dataset(args.dataset)
    pipeline = build_pipeline(dataset, top_k=args.top_k)

    rows: list[dict] = []
    precisions: list[float] = []
    recalls: list[float] = []
    hits = 0

    for example in dataset["examples"]:
        report = pipeline.run(example["question"], gold_chunk_ids=example["gold_chunk_ids"])
        precision = report.metrics.get("retrieval_precision", float("nan"))
        recall = report.metrics.get("retrieval_recall", float("nan"))
        precisions.append(precision)
        recalls.append(recall)
        top_hit_gold = example["gold_chunk_ids"][0] in report.cited_chunk_ids
        hits += int(top_hit_gold)
        rows.append(
            {
                "question": example["question"],
                "gold": example["gold_chunk_ids"],
                "retrieved": [c.id for c in report.retrieved],
                "cited": report.cited_chunk_ids,
                "precision": round(precision, 3),
                "recall": round(recall, 3),
                "top_hit_gold": top_hit_gold,
            }
        )

    n = len(rows)
    print(f"dataset : {dataset['name']} ({n} examples, top_k={args.top_k})")
    print(f"{'QUESTION':60s} {'PREC':>6s} {'RECALL':>6s} TOP1")
    for row in rows:
        q = row["question"][:58]
        print(
            f"{q:60s} {row['precision']:6.2f} {row['recall']:6.2f} "
            f"{'yes' if row['top_hit_gold'] else 'no'}"
        )
    print("-" * 78)
    print(
        f"mean retrieval precision : {sum(precisions) / n:.3f}\n"
        f"mean retrieval recall    : {sum(recalls) / n:.3f}\n"
        f"top-1 gold citation rate : {hits}/{n}"
    )


if __name__ == "__main__":
    main()
