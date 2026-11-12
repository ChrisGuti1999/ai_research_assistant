# AI Research Assistant — RAG, LLM Tool Calling & Evaluation

A Python-based research assistant that answers natural-language questions over financial research documents. It combines **Retrieval-Augmented Generation (RAG)**, **OpenAI embeddings**, **ChromaDB vector search**, and an optional **LLM tool-calling workflow**, with a lightweight **Streamlit** interface and evaluation script.

> **Project status:** Functional local learning prototype, not a production deployment.

**Data sources:** The repository includes fictional company reports for demonstration purposes. V2 evaluation uses real audited FY2023 financial statements from Plastic Energy Limited, indexed locally, to benchmark retrieval quality, answer correctness and system efficiency. The original financial report is not distributed with this repository.


## Application Demo

<table>
  <tr>
    <th align="center">RAG Mode</th>
    <th align="center">Agent Mode</th>
  </tr>
  <tr>
    <td width="50%">
      <img src="images/rag_demo.png" alt="RAG Mode" width="100%">
    </td>
    <td width="50%">
      <img src="images/agent_demo.png" alt="Agent Mode" width="100%">
    </td>
  </tr>
</table>

## What it does

- **Indexes documents:** Extracts text from `.pdf` and `.txt` files, splits it into overlapping chunks, generates embeddings, and stores the text, vectors, and source metadata in a persistent local ChromaDB collection.
- **Answers with RAG:** Embeds the user's question, retrieves the configurable retrieval depth (top_k = 3 or 5) matching chunks with optional source filtering, and supplies them as context to an LLM. The assistant is instructed to answer from the evidence and cite its sources.
- **Supports a bounded agent mode:** An LLM formulates a query using OpenAI function calling; Python runs the `search_reports` tool and passes its results back to the LLM for a final response.
- **Evaluates retrieval and response performance:** Reports response latency, generative-model token usage, and retrieved passages; an evaluation script measures page-level and chunk-level Hit@K, Precision@K, MRR, abstention behaviour, latency and generative token usage.

## Architecture

```text
                  ┌────────────────────────┐
                  │ Streamlit UI (app.py)   │
                  │ Index / Ask / View      │
                  └────────────┬───────────┘
                               │
              ┌────────────────┴────────────────┐
              │                                 │
       Document ingestion                  Question answering
              │                                 │
       PDF / TXT files                  ┌────────┴─────────┐
              │                        │                  │
       Text extraction               RAG mode          Agent mode
              │                        │                  │
       Overlapping chunks       Use user question    LLM generates
              │                 as search query      tool query
       OpenAI embeddings               │                  │
              │                        └────────┬─────────┘
              ▼                                 │
       Local ChromaDB ◄───────────── Embed query + vector search
       (text + vectors                       │
        + metadata)                     Retrieved passages
                                               │
                                          OpenAI LLM
                                               │
                                     Answer + source references
```

Both modes use the **same indexed documents and retrieval function**. The key difference is that basic RAG searches with the original question, whereas agent mode uses a query selected by the LLM through a structured tool call. In this prototype, the agent is **required to call one available search tool**; it is not an autonomous multi-tool system.

## V2 — Financial Statement Evaluation & Benchmarking

### Objective

Evaluate the accuracy, retrieval quality and efficiency of a financial research assistant built with OpenAI embeddings, ChromaDB and the Responses API.

V2 introduces a structured evaluation framework to benchmark RAG against an LLM tool-calling agent using 15 ground-truth questions from Plastic Energy Limited's FY2023 financial statements. The evaluation investigates retrieval quality, source filtering, response latency and token efficiency, with the aim of selecting the simplest architecture that reliably retrieves supporting evidence.

### Evaluation methodology

Created a ground-truth dataset of 15 financial research questions:

- 13 answerable questions covering financial figures, calculations and company disclosures.
- 2 unanswerable questions designed to test abstention behaviour.

Relevant document chunks were manually labelled using their source, page and chunk IDs.

Evaluated four configurations using identical questions, the same document index, source filtering and embedding model.

### Benchmark results — November 2026

The following results were obtained from the latest single-run comparison of four source-filtered configurations, using the same 15-question evaluation dataset.

| Metric | RAG k=3 | RAG k=5 | Agent k=3 | Agent k=5 |
|---|---:|---:|---:|---:|
| Hit@1 | 10/13 | 10/13 | 8/13 | 8/13 |
| Chunk Hit@3 | 12/13 | 12/13 | 11/13 | 11/13 |
| Chunk Hit@K | 12/13 | 13/13 | 11/13 | 13/13 |
| Mean Precision@3 | 48.7% | 48.7% | 38.5% | 41.0% |
| Mean Precision@K | 48.7% | 33.8% | 38.5% | 33.8% |
| MRR@3 | 0.833 | 0.821 | 0.731 | 0.718 |
| MRR@K | 0.833 | 0.840 | 0.731 | 0.753 |
| Correct answer/abstain decision | 14/15 | 15/15 | 13/15 | 15/15 |
| Mean latency | 2.19s | 2.19s | 4.01s | 4.20s |
| Mean generative LLM tokens | 1,072 | 1,642 | 1,358 | 1,980 |


### Key findings

**1. Source filtering substantially improved retrieval quality.**

Initial vector searches across multiple indexed companies retrieved irrelevant financial evidence. Restricting retrieval to Plastic Energy's FY2023 financial statements increased labelled Chunk Hit@3 from 46.2% (6/13) to 92.3% (12/13), demonstrating the importance of document-level metadata filtering.

**2. Increasing retrieval depth improved evidence coverage.**

The system initially failed to answer a question about employee headcount because the correct passage ranked outside the top three retrieved chunks. Increasing retrieval depth to five recovered the relevant evidence and enabled the model to correctly identify 92 employees.

**3. Standard RAG with k=5 was the preferred configuration.**

Both standard RAG and Agent mode achieved 13/13 labelled evidence hits at k=5. However, standard RAG had approximately 48% lower average latency and consumed 17% fewer generative LLM tokens.

For this dataset, the additional LLM-based query formulation in Agent mode did not demonstrate sufficient retrieval improvement to justify its overhead.

These findings favour the simpler architecture, while highlighting the trade-off between retrieval depth, answer quality and computational efficiency.


### Limitations

- **Dataset size:** The benchmark includes 15 questions from one financial report, limiting generalisability.

- **Run variability:** Each configuration was evaluated once in the latest benchmark. Repeated runs are needed to assess consistency.

- **Answer quality:** Retrieval relevance was manually labelled, but factual answer correctness and citation faithfulness have not been fully evaluated.

- **Architecture comparison:** RAG and Agent modes use different generation instructions, so the benchmark compares complete workflows rather than isolating query rewriting.

- **Cost measurement:** Generative LLM token counts exclude embedding usage and do not represent total API cost.

- **Reproducibility:** The Plastic Energy FY2023 PDF was indexed locally but is not included in this repository. Reproduction requires the same document, chunking configuration and ground-truth labels.

This remains a local learning prototype and has not been validated for production or high-stakes financial decision-making.


### Running the benchmark

Activate the project's virtual environment, index the evaluation document and run:

```
python evaluate.py --mode rag --top-k 3
python evaluate.py --mode rag --top-k 5
python evaluate.py --mode agent --top-k 3
python evaluate.py --mode agent --top-k 5
```

Each configuration generates a separate CSV containing retrieved evidence, ranking metrics, abstention behaviour, latency and token consumption.


## Tech stack

| Component | Technology |
| --- | --- |
| Language | Python |
| LLM generation and tool calling | OpenAI Responses API |
| Embeddings | `text-embedding-3-small` |
| Vector database | ChromaDB (`PersistentClient`) |
| Document extraction | pypdf and Python text processing |
| User interface | Streamlit |
| Evaluation | Python, JSON test cases, CSV reporting |

The application uses the OpenAI API directly; **LangChain is not required**.

## Getting started

**Requirements:** Python **3.12 recommended** (the version used for local development), an OpenAI API key, and API billing/credits enabled for your project.

```bash
# Clone the repository
git clone https://github.com/ChrisGuti1999/ai_research_assistant.git
cd ai_research_assistant

# Create and activate a virtual environment (macOS / Linux)
python3.12 -m venv .venv
source .venv/bin/activate

# Install dependencies
python -m pip install -r requirements.txt

# Create a private local configuration file
cp .env.example .env
```

Edit `.env` and replace the placeholder key:

```dotenv
OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL=gpt-4o-mini
```

Never commit `.env` or share your API key. The repository's `.gitignore` excludes `.env`, virtual environments, local ChromaDB data and generated usage logs. **Indexing and asking questions make billable OpenAI API requests.**

Launch the interface:

```bash
python -m streamlit run app.py
```

Open `http://localhost:8501`, click **Index documents**, choose **RAG** or **Agent** mode, and ask a question about the indexed material. The three included `.txt` documents are fictional examples, so you can try the application without downloading external reports.

### Example questions

- What drove Aurelia Analytics' revenue growth in 2026?
- What was Harbor Payments' processed payment volume in 2026?
- Who was Aurelia Analytics' chief executive in June 2026? *(The documents do not supply this information; the assistant should abstain.)*




## Repository structure

```text
app.py                 Streamlit interface
rag.py                 Ingestion, embeddings, retrieval, generation, usage logging
agent.py               Bounded LLM function-calling workflow
evaluate.py            Configurable RAG/Agent benchmark runner
eval_cases.json        Four sample evaluation questions
eval_cases_v2.json     15 labelled financial research evaluation cases
inspect_chunks.py      Inspect indexed chunk
documents/             Fictional sample financial summaries
tests/test_chunking.py Chunking unit tests
requirements.txt       Python dependencies
.env.example           Safe configuration template
```

## Roadmap

The project follows an iterative development approach, progressing from a functional RAG prototype to evaluated retrieval systems, agentic workflows and production engineering practices.

### V2.1 — Evaluation & Application Refinement

Finalise manual answer-correctness scoring and citation-groundedness checks. Repeat benchmarks across multiple runs to measure variability, and improve the Streamlit interface with configurable retrieval depth and document selection.

Improve reproducibility through documented evaluation setup, regression tests and consistent experiment logging.

### V3 — Advanced Retrieval & Reranking

Implement a two-stage retrieval pipeline that retrieves a larger candidate set from ChromaDB and reranks passages according to their relevance to the question.

Evaluate whether reranking improves Hit@3, Precision@3 and answer quality while controlling generation context size, latency and token consumption.

Explore hybrid retrieval combining semantic vector search with lexical or keyword matching, particularly for financial terminology and numerical disclosures.

### V4 — Bounded Iterative Agents

Extend the existing single-step tool-calling workflow into an iterative research agent capable of evaluating retrieved evidence and performing additional searches when necessary.

Introduce explicit stopping criteria, a maximum number of tool calls, structured tool-execution logs and evidence validation.

Benchmark iterative agents against standard RAG on complex questions requiring multiple pieces of evidence.

### V5 — Production Engineering & Deployment

Improve system reliability through automated testing, API error handling, retries, logging and monitoring.

Containerise the application using Docker and explore deployment and CI/CD workflows using GitHub Actions.

Introduce stronger data validation, access controls and safeguards before considering use with sensitive or confidential documents.


## Notes

This is an **independent, AI-assisted learning project**. It does not represent work completed for an employer or client, and it has not been validated for production or high-stakes financial decision-making.
