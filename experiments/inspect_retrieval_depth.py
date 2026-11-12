
from rag import search_reports, get_collection

source = (
    "13.3.2.1_Plastic Energy Limited "
    "signed audited FS FY2023.pdf"
)

question = (
    "average number of employees 2023 "
    "Plastic Energy Limited"
)

# Check that page 24 is indexed
collection = get_collection()

indexed = collection.get(
    where={
        "$and": [
            {"source": source},
            {"page": 24}
        ]
    },
    include=["documents"]
)

print("\nPAGE 24 INDEXED CHUNKS:")
for chunk_id, text in zip(
    indexed["ids"], indexed["documents"]
):
    print(chunk_id)
    print(text[:1000])

# Retrieve more passages, without changing the index
results = search_reports(
    question,
    top_k=10,
    source_filter=source
)

print("\nTOP 10 RETRIEVED CHUNKS:")

for rank, chunk in enumerate(results, start=1):
    print(
        f"Rank {rank}: "
        f"Page {chunk['page']} | "
        f"Distance {chunk['distance']} | "
        f"{chunk['chunk_id']}"
    )
