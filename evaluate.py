
"""V2 RAG evaluation: evidence retrieval, abstention and performance."""

import csv
import json
from statistics import mean
from rag import ROOT, answer_question

import argparse
from agent import agent_answer


def detect_abstention(answer: str) -> bool:
    """Basic heuristic: detect when the model declines to answer."""
    phrases = [
        "i don't know",
        "insufficient",
        "not disclosed",
        "not available",
        "cannot determine",
        "not provided",
    ]
    return any(phrase in answer.lower() for phrase in phrases)


def calculate_ranking_metrics(sources, relevant_chunk_ids, k=3):
    """Calculate fixed @3 and configurable @K retrieval metrics."""


    metrics = (
        "hit_at_1",
        "chunk_hit_at_3",
        "chunk_hit_at_k",
        "precision_at_3",
        "precision_at_k",
        "reciprocal_rank_at_3",
        "reciprocal_rank_at_k",
    )

    if not relevant_chunk_ids:
        return {metric: None for metric in metrics}


    # Preserve retrieval order
    ranked_ids = [
        source["chunk_id"] for source in sources[:k]
    ]

    def relevant_count(n):
        return sum(
            chunk_id in relevant_chunk_ids for chunk_id in ranked_ids[:n]
        )

    def reciprocal_rank(n):
        return next(
            (
                1 / rank
             for rank, chunk_id in enumerate(ranked_ids[:n], start=1)
                                             if chunk_id in relevant_chunk_ids
                                             ),
                                             0.0)

    return {
        "hit_at_1": relevant_count(1) > 0,
        "chunk_hit_at_3": relevant_count(3) > 0,
        "chunk_hit_at_k": relevant_count(k) > 0,
        "precision_at_3": relevant_count(3) / 3,
        "precision_at_k": relevant_count(k) / k,
        "reciprocal_rank_at_3": reciprocal_rank(3),
        "reciprocal_rank_at_k": reciprocal_rank(k),
    }
                                            
def main():


    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        choices=["rag", "agent"],
        default="rag"
    )

    parser.add_argument(
        "--top-k",
        type=int,
        choices=[3, 5],
        default=3,
    )

    args = parser.parse_args()

    # 1. Load the verified ground-truth questions
    cases = json.loads(
        (ROOT / "eval_cases_v2.json").read_text(encoding="utf-8")
    )

    rows = []

    for case in cases:
        # 2. Run the RAG application
        #result = answer_question(case["question"])

        source = ("13.3.2.1_Plastic Energy Limited "
            "signed audited FS FY2023.pdf"
        )

        if args.mode == "rag":
            result = answer_question(
                case["question"],
                top_k=args.top_k,
                source_filter=source
            )
        else:
            result = agent_answer(
                case["question"],
                top_k=args.top_k,
                source_filter=source
            ) 

        retrieved_chunks_ranked = [
        {
            "citation": f"[{i}]",
            "chunk_id": source["chunk_id"],
            "page": source["page"],
        }
        for i, source in enumerate(result["sources"], start=1)
        ]  


        # Compare retrieved chunks against ground-truth chunks
        expected_chunks = set(case.get("relevant_chunk_ids", []))


        if not case["should_abstain"] and not expected_chunks:
            raise ValueError(
                f"{case['id']} has no labelled relevant chunks"
            )

        ranking = calculate_ranking_metrics(
            result["sources"],
            expected_chunks,
            k=args.top_k
        )

        # 3. Compare retrieved pages with expected pages
        expected_pages = set(case["expected_pages"])

        ranked_pages = [
            int(source["page"]) for source in result["sources"]
        ]

        retrieved_pages = set(ranked_pages)

        page_hit_at_3 = (
            bool(expected_pages & set(ranked_pages[:3]))
            if expected_pages else None
        )

        page_hit_at_k = (
            bool(expected_pages & set(ranked_pages[:args.top_k]))
            if expected_pages else None
        )

        # 4. Evaluate abstention behaviour
        abstained = detect_abstention(result["answer"])
        abstention_correct = (
            abstained == case["should_abstain"]
        )

        # 5. Store results for review
        row = {
            "id": case["id"],
            "mode": args.mode,
            "agent_search_query": result.get("search_query", ""),
            "question": case["question"],
            "expected_answer": case["expected_answer"] or "",
            "actual_answer": result["answer"],
            "expected_pages": sorted(expected_pages),
            "retrieved_pages": sorted(retrieved_pages),
            "retrieved_chunks_ranked": json.dumps(retrieved_chunks_ranked),
            "top_k": args.top_k,
            "page_hit_at_3": page_hit_at_3 if page_hit_at_3 is not None else "",
            "page_hit_at_k": page_hit_at_k if page_hit_at_k is not None else "",
            "should_abstain": case["should_abstain"],
            "abstained": abstained,
            "abstention_correct": abstention_correct,
            "answer_correct_manual": "",
            "latency_seconds": result["latency_seconds"],
            "input_tokens": result["input_tokens"],
            "output_tokens": result["output_tokens"],
        }


        # Add ranking metrics to the evaluation row
        row.update({
            key: value if value is not None else ""
            for key, value in ranking.items()
        })


        rows.append(row)
        print(
            f"{case['id']}: Page Hit@3={row['page_hit_at_3']}, "
            f"Abstention correct={abstention_correct}"
        )

    # 6. Export the evaluation results
    output = ROOT / (f"evaluation_results_{args.mode}_k{args.top_k}.csv"
)

    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    # 7. Print summary statistics
    scorable = [
        row for row in rows
        if isinstance(row["page_hit_at_3"], bool)
    ]

    if scorable:
        hits = sum(row["page_hit_at_3"] for row in scorable)
        print(f"\nPage Hit@3: {hits}/{len(scorable)}")


    chunk_scorable = [
        row for row in rows
        if isinstance(row["chunk_hit_at_3"], bool)
    ]

    if chunk_scorable:
        hits = sum(row["chunk_hit_at_3"] for row in chunk_scorable)
        print(f"Chunk Hit@3: {hits}/{len(chunk_scorable)}")



    # Calculate ranking metrics across answerable questions
    ranking_rows = [
        row for row in rows
        if isinstance(row["hit_at_1"], bool)
    ]

    if ranking_rows:
        summary_metrics = [
            ("hit_at_1", "Hit@1", True),
            ("chunk_hit_at_3", "Chunk Hit@3", True),
            ("chunk_hit_at_k", f"Chunk Hit@{args.top_k}", True),
            ("precision_at_3", "Mean Precision@3", True),
            ("precision_at_k", f"Mean Precision@{args.top_k}", True),
            ("reciprocal_rank_at_3", "MRR@3", True),
            ("reciprocal_rank_at_k", f"MRR@{args.top_k}", True),
        ]
           
    for field,label, as_percent in summary_metrics:
        score = mean(row[field] for row in ranking_rows)

        if as_percent:
            print(f"{label}: {score:.1%}")
        else:
            print(f"{label}: {score:.3f}")



    if ranking_rows:
        hit_at_1 = mean(
            row["hit_at_1"] for row in ranking_rows
        )

        precision_at_3 = mean(
            row["precision_at_3"] for row in ranking_rows
        )

        mrr_at_3 = mean(
            row["reciprocal_rank"] for row in ranking_rows
        )

        print(f"Hit@1: {hit_at_1:.1%}")
        print(f"Mean Precision@3: {precision_at_3:.1%}")
        print(f"MRR@3: {mrr_at_3:.3f}")



    correct_abstentions = sum(
        row["abstention_correct"] for row in rows
    )
    print(f"Abstention behaviour: {correct_abstentions}/{len(rows)}")
    print(f"Saved: {output.name}")
    print("Review answer correctness manually in the CSV.")


if __name__ == "__main__":
    main()
