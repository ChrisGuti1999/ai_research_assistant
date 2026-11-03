# AI Research Assistant — beginner-friendly RAG / agent / evaluation starter

A **learning prototype**, not a production system. All bundled companies and financial statements are **fictional examples**. The code is designed to be readable, so you can explain every component in interviews.

## The learning sequence

1. **LLM API**: Python calls a pre-trained OpenAI model. No model training is required.
2. **RAG**: Extract PDF/text, split into overlapping chunks, create embeddings, persist them in a Chroma vector database, retrieve top-3 similar chunks, ask an LLM to answer using only those chunks.
3. **Agent**: The model requests a `search_reports` function with a chosen query; Python executes it and returns evidence to the model. This is *bounded tool-calling*, not a complex autonomous agent.
4. **Evaluation + monitoring**: Run test questions, record retrieval Hit@3, model abstentions, latency, and token usage. Manually review factual accuracy/citation correctness.
5. **Deployment**: Only after you are comfortable: push code to GitHub (never include `.env`, `.chroma`, or confidential files) and deploy via Streamlit Community Cloud with the API key stored as a secret. A public demo needs usage limits and safety controls.

## Setup (Mac Terminal)

Use Python **3.11 or newer**.

```bash
cd ai_research_assistant_starter
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

Go to https://platform.openai.com/api-keys , create an API key for your own project, and edit the `.env` file. **Never commit or share the API key.** An OpenAI API account is billed separately from your ChatGPT subscription; set a spending limit in the developer platform. See https://developers.openai.com/api/docs/quickstart . The default model is `gpt-5.6-terra` as of October 2026, but you may choose another supported model.

Run the app:

```bash
python -m streamlit run app.py
```

Click **Index documents** in the app sidebar. Indexing calls the OpenAI embeddings API, so it needs your key and may incur a small charge. Then ask:

- What drove Aurelia Analytics revenue growth in 2026?
- What was Harbor Payments' processed payment volume in 2026?
- Who was Aurelia Analytics' CEO in June 2026? (should acknowledge missing evidence)

Explore the retrieved passages under the answer: these are the **RAG context**. Try **Agent (Phase 2)** to see the LLM choose and call the `search_reports` tool. Use `python evaluate.py` for a tiny evaluation run (once the documents are indexed). The evaluation is just a starting point; four questions cannot establish reliability.

## Read the code in this order

- `app.py` — user interface; what the application does.
- `rag.py`: `chunk_text()` → `load_chunks()` → `ingest()` → `search_reports()` → `answer_question()`.
- `agent.py` — model function calling and tool execution.
- `evaluate.py` — small evaluation harness.

## Use real public research material

Replace the three fictional `.txt` files in `documents/` with publicly available PDF annual reports or investor presentations, re-index, then write 10–20 test questions. Verify PDF extraction worked (scanned images may require separate text extraction). Do **not** upload PayPal/client/employer information without authorisation.

## Limits / next steps

This is a real RAG prototype, not a validated production application. In particular, text chunking is character-based rather than token-aware, answer grounding is encouraged by instructions rather than guaranteed, retrieval uses no score threshold or reranking, and the example evaluation is small. The agent makes one tool-call round and only has one tool. Improve it by adding query filters, metadata constraints, stronger citations, prompt-injection defences, human feedback, tests, retries, cost controls and meaningful deployment monitoring.

## CV examples (only after you've actually completed the steps)

After RAG: "Built a Python-based RAG prototype to answer natural-language questions over research documents, using OpenAI embeddings, Chroma vector search and source-grounded LLM responses."

After adding tool calling and evaluation: "Extended a research assistant with LLM function calling and an evaluation harness for retrieval Hit@3, abstention, response latency and token usage."

**Never describe a personal demo as a production deployment or paid work.**

## Official technical references

- Responses API: https://developers.openai.com/api/docs/guides/text
- Embeddings: https://developers.openai.com/api/docs/guides/embeddings
- Function calling: https://developers.openai.com/api/docs/guides/function-calling
- Chroma: https://docs.trychroma.com/docs/collections/manage-collections
- Streamlit: https://docs.streamlit.io/develop/concepts/architecture/run-your-app
