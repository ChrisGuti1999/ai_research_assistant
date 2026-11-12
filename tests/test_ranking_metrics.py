# tests/test_ranking_metrics.py
from evaluate import calculate_ranking_metrics


def test_relevant_chunk_at_rank_five():
    sources = [
        {"chunk_id": f"chunk_{i}"}
        for i in range(1, 6)
    ]

    relevant = {"chunk_5"}

    result = calculate_ranking_metrics(
        sources,
        relevant,
        k=5
    )

    assert result["hit_at_1"] is False
    assert result["chunk_hit_at_3"] is False
    assert result["chunk_hit_at_k"] is True
    assert result["precision_at_k"] == 0.2
    assert result["reciprocal_rank_at_k"] == 0.2
