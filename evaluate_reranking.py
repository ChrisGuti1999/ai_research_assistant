
"""V3: Compare ChromaDB retrieval against cross-encoder reranking."""

import csv
import json
from statistics import mean
from time import perf_counter

from rag import ROOT, search_reports
from reranker import get_reranker, rerank_chunks
from evaluate import calculate_ranking_metrics


SOURCE = (
    "13.3.2.1_Plastic Energy Limited "
    "signed audited FS FY2023.pdf"
)


def first_relevant_rank(chunks, relevant_ids):
    """Find the first relevant passage in the ranked results."""
    return next(
        (
            rank
            for rank, chunk in enumerate(chunks, start=1)
            if chunk["chunk_id"] in relevant_ids
        ),
        None,
    )


def main():
    cases = json.loads(
        (ROOT / "eval_cases_v2.json").read_text(encoding="utf-8")
    )

    # Load the cross-encoder before timing individual questions.
    get_reranker()

    rows = []

    for case in cases:
        # Retrieval metrics need labelled relevant chunks.
        # We'll test abstention questions in the end-to-end evaluation.
        if case["should_abstain"]:
            continue

        question = case["question"]
        relevant = set(case["relevant_chunk_ids"])

        # Stage 1: retrieve 10 candidates with ChromaDB.
        candidates = search_reports(
            question,
            top_k=10,
            source_filter=SOURCE,
        )

        # Original vector-search ranking.
        baseline = candidates[:3]

        # Stage 2: rerank the SAME candidates.
        start = perf_counter()

        reranked = rerank_chunks(
            question,
            candidates,
            top_k=3,
        )

        rerank_seconds = perf_counter() - start

        # Reuse V2's existing metric calculations.
        baseline_metrics = calculate_ranking_metrics(
            baseline, relevant, k=3
        )
        reranked_metrics = calculate_ranking_metrics(
            reranked, relevant, k=3
        )

        # How many relevant chunks were retrieved initially?
        candidate_ids = {c["chunk_id"] for c in candidates}
        candidate_recall = len(candidate_ids & relevant) / len(relevant)

        row = {
            "id": case["id"],
            "question": question,
            "candidate_recall_at_10": candidate_recall,
            "original_rank_at_10": first_relevant_rank(
                candidates, relevant
            ),
            "reranked_rank_at_3": first_relevant_rank(
                reranked, relevant
            ),
            "baseline_hit_at_3": baseline_metrics["chunk_hit_at_3"],
            "reranked_hit_at_3": reranked_metrics["chunk_hit_at_3"],
            "baseline_precision_at_3": baseline_metrics["precision_at_3"],
            "reranked_precision_at_3": reranked_metrics["precision_at_3"],
            "baseline_mrr_at_3": baseline_metrics["reciprocal_rank_at_3"],
            "reranked_mrr_at_3": reranked_metrics["reciprocal_rank_at_3"],
            "rerank_seconds": round(rerank_seconds, 3),
            "baseline_chunk_ids": json.dumps(
                [c["chunk_id"] for c in baseline]
            ),
            "reranked_chunk_ids": json.dumps(
                [c["chunk_id"] for c in reranked]
            ),
        }

        rows.append(row)

        print(
            f"{case['id']}: "
            f"Original rank={row['original_rank_at_10']} | "
            f"Reranked rank={row['reranked_rank_at_3']} | "
            f"Hit@3: {row['baseline_hit_at_3']} "
            f"-> {row['reranked_hit_at_3']}"
        )

    # Save detailed results.
    output = ROOT / "reranking_comparison_v3.csv"

    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    # Summarise results.
    print("\n=== V3 RETRIEVAL EVALUATION ===")

    print(
        f"Candidate Recall@10: "
        f"{mean(r['candidate_recall_at_10'] for r in rows):.1%}"
    )

    for metric in ["hit_at_3", "precision_at_3", "mrr_at_3"]:
        baseline_score = mean(
            r[f"baseline_{metric}"] for r in rows
        )
        reranked_score = mean(
            r[f"reranked_{metric}"] for r in rows
        )

        print(
            f"{metric}: "
            f"Baseline={baseline_score:.3f} | "
            f"Reranked={reranked_score:.3f}"
        )

    print(
        f"Mean reranking time: "
        f"{mean(r['rerank_seconds'] for r in rows):.3f}s"
    )

    print(f"\nSaved results to {output.name}")


if __name__ == "__main__":
    main()
