
from rag import get_collection

collection = get_collection()

# Inspect all chunks stored for page xxx
results = collection.get(
    where={"page": 24},
    include=["documents", "metadatas"]
)

for chunk_id, text, metadata in zip(
    results["ids"],
    results["documents"],
    results["metadatas"]
):
    print(f"\nChunk ID: {chunk_id}")
    print(f"Source: {metadata['source']}")
    print(f"Text: {text}\n")
    print("-" * 60)
