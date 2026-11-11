
"""V2 RAG evaluation: evidence retrieval, abstention and performance."""

import csv
import json

from rag import ROOT, answer_question


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


def main():
    # 1. Load the verified ground-truth questions
    cases = json.loads(
        (ROOT / "eval_cases_v2.json").read_text(encoding="utf-8")
    )

    rows = []

    for case in cases:
        # 2. Run the RAG application
        result = answer_question(case["question"])

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

        rows.append(row)
        print(
            f"{case['id']}: Page Hit@3={row['page_hit_at_3']}, "
            f"Abstention correct={abstention_correct}"
        )

    # 6. Export the evaluation results
    output = ROOT / "evaluation_results_v2.csv"

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

    correct_abstentions = sum(
        row["abstention_correct"] for row in rows
    )
    print(f"Abstention behaviour: {correct_abstentions}/{len(rows)}")
    print(f"Saved: {output.name}")
    print("Review answer correctness manually in the CSV.")


if __name__ == "__main__":
    main()
