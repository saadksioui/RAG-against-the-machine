*This project has been created as part of the 42 curriculum by sksioui.*

# RAG Against The Machine

## Description
This project implements a complete Retrieval-Augmented Generation (RAG) pipeline designed to answer natural language questions about a specific codebase (vLLM). Instead of relying solely on the pre-trained knowledge of a Large Language Model, this system ingests a target repository, creates searchable lexical and semantic indexes, retrieves the most relevant code snippets or documentation based on a user's query, and provides these snippets as context to a local LLM (`Qwen/Qwen3-0.6B`) to generate accurate, grounded answers. 

It fulfills all mandatory requirements of the 42 curriculum, alongside all five bonus features: Semantic Embeddings, Hybrid Retrieval, Incremental Indexing, LLM Response Caching, and a FastAPI-based REST API.

---

## Instructions

### Prerequisites
- Python 3.10+
- `uv` (Fast Python package and project manager)
- `make`

### Installation
Clone the repository and install the dependencies using the provided Makefile:
```bash
git clone <repository_url>
cd rag-against-the-machine
make install
```

### Execution
You can run the pipeline using the command-line interface (CLI) or start the API server.

**Using the CLI:**
```bash
# 1. Index the raw dataset
uv run python -m src index --max_chunk_size 2000

# 2. Search a single query
uv run python -m src search "How do I configure the OpenAI server?" --k 5 --search_mode hybrid

# 3. Generate an answer
uv run python -m src answer "How do I configure the OpenAI server?" --k 5 --search_mode hybrid
```

**Using the API (Bonus):**
```bash
# Start the FastAPI server (runs on port 8000)
uv run python -m src.api
```

---

## System Architecture
The system is composed of several decoupled modules exchanging strongly-typed Pydantic models:
1. **Ingestion & Sync Engine:** Scans `data/raw/`. Uses a SHA-256 file hashing mechanism (`sync_state.json`) to detect new, modified, or deleted files. This enables surgical incremental indexing without rebuilding the entire corpus.
2. **Chunker:** Parses text and Python files, breaking them down into contextually meaningful pieces while preserving exact `first_character_index` and `last_character_index` boundaries.
3. **Retrieval Pipeline:** 
   - **Lexical:** Uses `bm25s` for fast, exact-keyword matching.
   - **Semantic (Bonus):** Uses `ChromaDB` for dense vector embeddings to capture semantic meaning.
   - **Hybrid (Bonus):** Combines both methods using Reciprocal Rank Fusion (RRF).
4. **Generator:** Wraps the HuggingFace Transformers pipeline for `Qwen/Qwen3-0.6B`. It dynamically constructs prompts injecting the retrieved chunks.
5. **Cache Layer (Bonus):** Intercepts queries, hashing the `(query, k, mode)` tuple to immediately serve previously generated LLM answers from disk.

---

## Chunking Strategy
The chunking process is handled dynamically based on file extensions using LangChain's `RecursiveCharacterTextSplitter`.
- **Python Code (`.py`):** Uses language-specific splitting (`Language.PYTHON`). This ensures that structural boundaries like classes and function definitions are kept together rather than being arbitrarily sliced.
- **Markdown/Text (`.md`, `.txt`):** Split using standard text separators (paragraphs, headers).
- **Parameters:** The `max_chunk_size` defaults to 2000 characters, with a `10%` overlap to maintain context between adjacent chunks.
- **Mapping:** To comply with the strict evaluation metrics, a custom sliding window algorithm using `string.find()` maps LangChain's output back to the original text to extract the exact starting and ending character indices.

---

## Retrieval Method
The system supports three retrieval algorithms:
1. **BM25 (Lexical):** Tokenizes the query (handling CamelCase and snake_case splits) and ranks chunks based on term frequency and inverse document frequency. Highly effective for exact code identifiers (e.g., `openai_transcription_client`).
2. **ChromaDB (Semantic):** Maps chunks to a dense vector space, allowing the system to find conceptually related information even if the exact keywords differ.
3. **Hybrid (RRF):** Executes both searches concurrently. It merges the results using the Reciprocal Rank Fusion formula: `RRF_Score = 1 / (60 + Rank)`. The constant 60 prevents small-k bias, ensuring a balanced weighting between lexical and semantic signals.

---

## Performance Analysis
- **Recall@k Validation:** The system utilizes a 1D Intersection-over-Union (IoU) calculation to measure if retrieved chunks overlap (>= 5%) with the ground-truth sources.
- **Accuracy:** Successfully achieves `> 80%` Recall@5 on the documentation dataset and `> 50%` on code datasets.
- **Speed (Hardware Timers):** 
  - **Indexing:** The full corpus processes well under the 5-minute limit (typically < 30 seconds). Incremental updates process in under a second.
  - **Retrieval:** Batch querying processes 200 questions well within the 90-second limit, averaging `< 0.2` seconds per query.

---

## Design Decisions
- **Uv for Dependency Management:** Chosen for its significant speed advantage over standard `pip`, ensuring the environment builds rapidly on campus machines.
- **Pydantic Validation:** Used heavily in `models.py` to enforce strict schema adherence between the CLI, the retrieval engine, and the JSON outputs required by the Moulinette evaluator.
- **Graceful Error Handling:** Instead of hard `exit(1)` calls, the engine raises standard Python exceptions. This allows the CLI to fail gracefully and the FastAPI backend to catch errors and return `400/500` HTTP statuses without crashing the web server.

---

## Challenges Faced
- **Exact Index Mapping:** LangChain's text splitters drop the original character indices of the chunks. *Solution:* Developed a custom index-tracking loop that uses `find()` with a shifting `start_from` window to accurately map chunks back to their raw file spans.
- **Duplication in Vector DB:** Re-indexing a modified file caused duplicate entries in ChromaDB. *Solution:* Integrated a surgical delete operation that removes stale paths from the persistent ChromaDB collection and the BM25 JSON state before re-chunking.
- **Strict Evaluator Paths:** The Moulinette strictly compares paths. *Solution:* Switched all path handling to relative `Path` objects cast to strings, ensuring they exactly match `data/raw/vllm-0.10.1/...` without absolute system paths leaking in.

---

## Example Usage

**1. Batch Retrieval Evaluation:**
```bash
uv run python -m src search_dataset \
  --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json \
  --k 10 \
  --save_directory data/output/search_results/UnansweredQuestions \
  --search_mode hybrid
```

**2. Batch Answer Generation:**
```bash
uv run python -m src answer_dataset \
  --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
  --save_directory data/output/search_results_and_answer/UnansweredQuestions \
  --search_mode hybrid
```

**3. Local Recall Checking:**
```bash
uv run python -m src evaluate \
  --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
  --dataset_path data/datasets/AnsweredQuestions/dataset_docs_public.json
```

---

## Resources
- **Libraries used:** [LangChain Text Splitters](https://python.langchain.com/docs/modules/data_connection/document_transformers/), [ChromaDB](https://docs.trychroma.com/), [BM25s](https://github.com/xhluca/bm25s), [Transformers (HuggingFace)](https://huggingface.co/docs/transformers/index).
- **AI Usage:** Large Language Models (Gemini/ChatGPT) were utilized strictly as a learning accelerator and debugging tool during this project. Specifically, AI was used to:
  - Explain the mathematical theory behind 1-Dimensional Intersection-over-Union (IoU) and Reciprocal Rank Fusion (RRF).
  - Draft the FastAPI boilerplate and help convert CLI `exit()` calls into robust HTTP exceptions to prevent server crashes.
  - Write a Python benchmarking script to verify hardware timer constraints prior to evaluation.