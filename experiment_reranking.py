
"""V3 experiment: compare vector retrieval against cross-encoder reranking."""

from rag import get_collection, search_reports
from reranker import rerank_chunks


question = (
    "What was the average number of employees employed by "
    "Plastic Energy Limited during the year ended 31 December 2023?"
)

# Identify the exact document filename stored in ChromaDB.
collection = get_collection()
metadata = collection.get(include=["metadatas"])["metadatas"]

sources = {
    item["source"] for item in metadata if item and "source" in item
}

matches = [
    source for source in sources
    if "plastic energy" in source.lower() and "2023" in source.lower()
]

if len(matches) != 1:
    raise ValueError(f"Expected one Plastic Energy FY2023 report: {matches}")

source_filter = matches[0]


# STEP 1: Retrieve 10 candidate chunks using embeddings.
candidates = search_reports(
    question,
    top_k=10,
    source_filter=source_filter,
)

print("\n=== ORIGINAL CHROMADB RANKING ===")

for rank, chunk in enumerate(candidates, start=1):
    print(
        f"{rank}. Page {chunk['page']} | "
        f"Distance: {chunk['distance']:.3f} | "
        f"{chunk['chunk_id']}"
    )


# STEP 2: Rerank the same 10 candidates.
reranked = rerank_chunks(
    question,
    candidates,
    top_k=3,
)

print("\n=== CROSS-ENCODER TOP 3 ===")

for rank, chunk in enumerate(reranked, start=1):
    print(
        f"{rank}. Page {chunk['page']} | "
        f"Rerank score: {chunk['rerank_score']:.3f} | "
        f"{chunk['chunk_id']}"
    )


# STEP 3: Check whether the correct chunk reached the top 3.
gold_chunk_suffix = ":p24:c0"

original_rank = next(
    (
        rank for rank, chunk in enumerate(candidates, start=1)
        if chunk["chunk_id"].endswith(gold_chunk_suffix)
    ),
    None,
)

reranked_rank = next(
    (
        rank for rank, chunk in enumerate(reranked, start=1)
        if chunk["chunk_id"].endswith(gold_chunk_suffix)
    ),
    None,
)

print("\n=== RESULT ===")
print(f"Original rank in top 10: {original_rank}")
print(f"Rank after reranking: {reranked_rank}")
print(f"Correct chunk in reranked top 3: {reranked_rank is not None}")
