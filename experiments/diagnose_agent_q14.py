
from agent import agent_answer

source = (
    "13.3.2.1_Plastic Energy Limited "
    "signed audited FS FY2023.pdf"
)

question = (
    "What was the average number of employees "
    "employed by Plastic Energy Limited during "
    "the year ended 31 December 2023?"
)

result = agent_answer(
    question,
    source_filter=source
)

print("\nAGENT SEARCH QUERY:")
print(result["search_query"])

print("\nRETRIEVED CHUNKS:")
for i, chunk in enumerate(result["sources"], start=1):
    print(
        f"[{i}] Page {chunk['page']} | "
        f"{chunk['chunk_id']}"
    )

print("\nFINAL ANSWER:")
print(result["answer"])

print("\nPERFORMANCE:")
print(f"Latency: {result['latency_seconds']}s")
print(f"Input tokens: {result['input_tokens']}")
print(f"Output tokens: {result['output_tokens']}")
