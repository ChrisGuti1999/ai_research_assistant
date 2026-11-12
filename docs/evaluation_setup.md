# Reproducing the V2 Financial Statement Benchmark

This document explains how to reproduce the **15-question Plastic Energy FY2023** RAG-versus-Agent benchmark. The repository also includes fictional `.txt` documents for demonstration; those are **not** the benchmark's ground-truth financial report.

## 1. Requirements

- Python 3.12 (the version used during development).
- An OpenAI API key with API billing enabled.
- The same **Plastic Energy Limited FY2023 signed audited financial statements** PDF used for the benchmark, obtained from an authorised source. The PDF is **not distributed in this repository**.
- Access to the V2 code (`feature/v2-rag-evaluation`, or `main` after V2 is merged).

Indexing and all four evaluation runs make billable OpenAI API calls.

## 2. Set up the project

```bash
git clone -b feature/v2-rag-evaluation https://github.com/ChrisGuti1999/ai_research_assistant.git
cd ai_research_assistant
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` and set:

```dotenv
OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL=your_chosen_available_model
```

**For a like-for-like comparison, use the exact generation model used for the published benchmark.** The model in `.env.example` is an example, not proof of which model produced earlier results. Record your actual model identifier and package versions when publishing or repeating a benchmark. Never commit `.env`.

## 3. Add the financial statement

Place the source PDF in `documents/` with this **exact filename**:

```text
documents/13.3.2.1_Plastic Energy Limited signed audited FS FY2023.pdf
```

This filename is intentionally required: `evaluate.py` source-filters by that string, and `eval_cases_v2.json` uses it in ground-truth chunk IDs. The PDF is ignored by Git via `.gitignore`.

If you have a different edition, filename, or text-extraction result, the ground-truth chunk IDs and page numbers may not match. Do not silently relabel chunks to make scores look better.

## 4. Build the ChromaDB index

```bash
python -m streamlit run app.py
```

In the Streamlit sidebar, select **Index documents**. This rebuilds the local `.chroma` collection using all supported files in `documents/`, including the fictional demo `.txt` files. The benchmark itself uses a source filter to restrict retrieval to the Plastic Energy PDF.

Current ingestion settings in `rag.py`:

| Setting | Value |
| --- | --- |
| Embedding model | `text-embedding-3-small` |
| Character chunk size | 1,300 |
| Character overlap | 220 |
| Vector store | Local ChromaDB collection `research_documents` |
| Source metadata | Exact filename and 1-based extracted PDF page number |

For a consistent run, do not change the PDF, chunking parameters, embedding model, or indexed collection between the four configurations.

To check that the PDF was indexed, you can run:

```bash
python -c "from rag import get_collection; c=get_collection(); print('Plastic Energy chunks:', len(c.get(where={'source': '13.3.2.1_Plastic Energy Limited signed audited FS FY2023.pdf'})['ids']))"
```

## 5. Run the four configurations

Stop Streamlit if desired, then run from the project root with `.venv` activated:

```bash
python evaluate.py --mode rag --top-k 3
python evaluate.py --mode rag --top-k 5
python evaluate.py --mode agent --top-k 3
python evaluate.py --mode agent --top-k 5
```

The evaluation reads `eval_cases_v2.json`, containing **13 answerable and 2 expected-abstention questions**, and writes the following files in the project root:

```text
evaluation_results_rag_k3.csv
evaluation_results_rag_k5.csv
evaluation_results_agent_k3.csv
evaluation_results_agent_k5.csv
```

**Caution:** A run overwrites its corresponding CSV. Save any manually reviewed version separately before rerunning.

## 6. Understand and review the results

- **Chunk Hit@3 / Hit@K:** Whether at least one labelled relevant chunk appears in the first three / first K retrieved results.
- **Precision@3 / Precision@K:** Proportion of retrieved chunks labelled relevant within the first three / first K results.
- **Reciprocal Rank@3 / @K:** Reciprocal of the rank of the first labelled relevant chunk, or zero when none is found within that cutoff. Averaging these values across answerable questions gives MRR.
- **Page Hit@3 / @K:** Whether an expected evidence page is retrieved within the cutoff.
- **Abstention agreement:** Whether the system answered or abstained as expected; **not** factual answer accuracy.
- **Input/output tokens and latency:** Usage for the executed workflow; token counts do not include embeddings.

The `answer_correct_manual` column is intentionally left blank by `evaluate.py`. Review answers against the source financial statements before making factual-accuracy claims. Check the evidence behind the model's citations separately; a cited chunk is not necessarily relevant or sufficient.

## 7. Interpretation and limitations

The README benchmark numbers describe **observed single runs**, not statistically validated performance differences. Agent-generated queries and answers may change on repetition. The two modes also use somewhat different generation instructions, so this is a comparison of full workflows, not an isolated experiment on query rewriting.

This evaluation covers one document and a small manually labelled set of questions. It is a **local learning prototype**, not a validated financial decision-making system.

If you reproduce the experiment, record the run date, Python and dependency versions, generation model, source-PDF provenance/checksum, indexed document count, and all four result CSVs. This will make differences between runs easier to diagnose.
