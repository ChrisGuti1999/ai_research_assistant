
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
    """Evaluate the ranking of retrieved chunks."""

    if not relevant_chunk_ids:
        return {
            "hit_at_1": None,
            "precision_at_3": None,
            "reciprocal_rank": None,
        }

    # Preserve retrieval order
    ranked_ids = [
        source["chunk_id"] for source in sources[:k]
    ]

    # Identify the ranks containing relevant evidence
    relevant_ranks = [
        rank
        for rank, chunk_id in enumerate(ranked_ids, start=1)
        if chunk_id in relevant_chunk_ids
    ]

    hit_at_1 = bool(ranked_ids) and ranked_ids[0] in relevant_chunk_ids

    precision_at_3 = (
        len(relevant_ranks) / k
    )

    reciprocal_rank = (
        1 / relevant_ranks[0]
        if relevant_ranks else 0.0
    )

    return {
        "hit_at_1": hit_at_1,
        "precision_at_3": round(precision_at_3, 3),
        "reciprocal_rank": round(reciprocal_rank, 3),
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

        ranking = calculate_ranking_metrics(
            result["sources"],
            expected_chunks,
            k=3
        )

        retrieved_chunks = {
            source["chunk_id"]
            for source in result["sources"]
        }

        chunk_hit_at_3 = (
            bool(expected_chunks & retrieved_chunks)
            if expected_chunks else None
        )



        # 3. Compare retrieved pages with expected pages
        expected_pages = set(case["expected_pages"])
        retrieved_pages = {
            int(source["page"]) for source in result["sources"]
        }

        page_hit = (
            bool(expected_pages & retrieved_pages)
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
            "chunk_hit_at_3": (chunk_hit_at_3 if chunk_hit_at_3 is not None else ""),
            "page_hit_at_3": page_hit if page_hit is not None else "",
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
        if isinstance(row["precision_at_3"], float)
    ]

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
