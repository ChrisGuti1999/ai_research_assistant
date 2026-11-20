
"""Rerank retrieved passages using a pretrained cross-encoder."""

from functools import lru_cache
from sentence_transformers import CrossEncoder


MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def get_reranker():
    """Load the model once and reuse it."""
    return CrossEncoder(MODEL_NAME)


def rerank_chunks(question: str, chunks: list[dict], top_k: int = 3,) -> list[dict]:
    """Rerank candidate chunks and return the top_k most relevant."""

    if top_k < 1:
        raise ValueError("top_k must be at least 1")

    if not chunks:
        return []

    # 1. Pair the question with each candidate passage.
    pairs = [(question, chunk["text"]) for chunk in chunks]

    # 2. Score each question-passage pair.
    scores = get_reranker().predict(pairs)

    # 3. Attach a reranking score, preserving existing metadata.
    scored_chunks = [
        {**chunk, "rerank_score": float(score)}
        for chunk, score in zip(chunks, scores)
    ]

    # 4. Sort by relevance (highest score first).
    scored_chunks.sort(
        key=lambda chunk: chunk["rerank_score"],
        reverse=True,
    )

    # 5. Return only the best passages.
    return scored_chunks[:top_k]


if __name__ == "__main__":
    question = "How many employees did the company have in 2023?"

    sample_chunks = [
        {
            "chunk_id": "demo:p12:c0",
            "text": "The company reported revenue of £4 million in 2023.",
            "source": "demo.pdf",
            "page": 12,
            "distance": 0.4,
        },
        {
            "chunk_id": "demo:p24:c0",
            "text": "The average number of employees in 2023 was 92.",
            "source": "demo.pdf",
            "page": 24,
            "distance": 0.6,
        },
        {
            "chunk_id": "demo:p26:c0",
            "text": "The company reported a loss during the financial year.",
            "source": "demo.pdf",
            "page": 26,
            "distance": 0.5,
        },
    ]

    results = rerank_chunks(question, sample_chunks, top_k=3)

    for rank, chunk in enumerate(results, start=1):
        print(
            f"{rank}. Page {chunk['page']} | "
            f"Score: {chunk['rerank_score']:.3f}"
        )
        print(chunk["text"])
        print()
