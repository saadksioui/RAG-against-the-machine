# RAG Against the Machine — 20-Day Mastery Roadmap

*A theory-first, practice-second study plan, written the way a senior engineer would onboard a mentee onto this exact project.*

---

## 0. How to use this guide

The subject itself tells you the failure mode to avoid: someone who ships working code but can't explain it in the defense, fails. So this guide is built around one rule:

> **Never write a line for the real project until you can explain, out loud, without notes, why it needs to exist.**

The structure follows the "theory → mini-exercise → real feature" loop for every concept:

1. **Theory** — read/watch just enough to build a correct mental model (not exhaustive reading).
2. **Mini-exercise** — a small, disposable script that isolates *one* concept, away from the pressure of the real vLLM corpus and the moulinette.
3. **Explain it** — say out loud (or write two sentences) what you just built and why it works. If you can't, go back to step 1.
4. **Real feature** — only now port the idea into `src/`.

One note on AI use, in the spirit of the subject's own "AI Instructions" chapter: this guide gives you *signatures, hints, and things to verify* for the mini-exercises, not full solutions. Filling in the blank yourself is the part that builds the muscle you'll need in the oral defense and the recode drill.

---

## 1. Concept Map

Each concept has an ID (so exercises and the roadmap can reference it), a one-line "why it matters here", a mastery bar, and resources.

### FUNDAMENTAL — Python & tooling hygiene
*(You need these before writing any RAG-specific code. Skipping this layer is the #1 cause of losing points on things that have nothing to do with retrieval.)*

**F1 — Modern Python typing (3.10+)**
Why: every data model and function signature in this project needs type hints; `X | Y` unions appear directly in the provided models (`AnsweredQuestion | UnansweredQuestion`).
Mastery bar: you can read/write `list[str]`, `X | Y`, `Optional[X]`, and explain the difference between `X | None` and a field with a default.
- [Docs] Python `typing` module — https://docs.python.org/3/library/typing.html
- [Docs] "What's new in Python 3.10" (union syntax) — https://docs.python.org/3/whatsnew/3.10.html

**F2 — Pydantic v2 data modeling**
Why: the project's entire inter-stage contract (`MinimalSource`, `AnsweredQuestion`, `RagDataset`, etc.) is pydantic. Get this wrong and every downstream command breaks.
Mastery bar: you can explain `BaseModel` vs plain class, what `Field(default_factory=...)` buys you over `Field(default=...)`, the difference between `model_dump()`, `model_dump_json()`, and `model_validate_json()`, and why `AnsweredQuestion(UnansweredQuestion)` inheritance makes sense here.
- [Docs] Pydantic official docs — https://docs.pydantic.dev/latest/
- [Article] Pydantic v2 core tutorial (models, fields, validators) — search "pydantic v2 core tutorial models fields validators"

**F3 — Exceptions & context managers**
Why: the subject explicitly fails you if the program crashes on an unhandled exception during review, and the CLI must survive malformed JSON, missing files, `k=0`, empty queries.
Mastery bar: you can write a function that never raises past its own boundary, using `try/except` with *specific* exception types (not a bare `except:`), and you know when to write a custom context manager (`__enter__`/`__exit__` or `@contextmanager`) vs just using `with open(...)`.
- [Docs] Python `contextlib` — https://docs.python.org/3/library/contextlib.html
- [Docs] Errors and Exceptions tutorial — https://docs.python.org/3/tutorial/errors.html

**F4 — JSON serialization**
Why: every CLI command reads or writes JSON that must validate against a pydantic model.
Mastery bar: you can explain why `json.loads` on untrusted input needs a `try/except json.JSONDecodeError`, and why validating with pydantic *after* parsing catches structural errors that raw `json.loads` won't.
- [Docs] Python `json` module — https://docs.python.org/3/library/json.html

**F5 — CLI construction with Python Fire**
Why: mandated by the subject; every command (`index`, `search`, `search_dataset`, `answer`, `answer_dataset`, `evaluate`) must be Fire-exposed.
Mastery bar: you can turn a class with public methods into a multi-command CLI, and you understand why exposing a *class* (one command per method) fits this project's six commands better than exposing six separate scripts.
- [Docs] The Python Fire Guide — https://google.github.io/python-fire/guide/
- [Repo] google/python-fire (README + examples) — https://github.com/google/python-fire

**F6 — Progress bars with tqdm**
Why: mandatory for long-running operations (indexing thousands of files, batch retrieval/generation).
Mastery bar: you can wrap an iterator with `tqdm` and explain why you'd wrap the *file loop* in indexing but not a single-query search.
- [Repo] tqdm — https://github.com/tqdm/tqdm

**F7 — Static analysis: flake8 & mypy (incl. `--strict`)**
Why: `make lint` and `make lint-strict` are graded gates; all functions must type-check.
Mastery bar: you can run `mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs` locally, read one real error, and fix it without adding a blanket `# type: ignore`.
- [Docs] mypy command line — https://mypy.readthedocs.io/en/stable/command_line.html
- [Docs] flake8 — https://flake8.pycqa.org/en/latest/

**F8 — Project/dependency management with `uv`**
Why: the reviewer and moulinette *only* run `uv sync`. If your `pyproject.toml`/`uv.lock` aren't right, nothing else in this project gets evaluated.
Mastery bar: you can explain the difference between `uv add`, `uv sync`, and `uv run`, and why `uv.lock` must be committed.
- [Docs] uv official docs — https://docs.astral.sh/uv/
- [Docs] uv installation — https://docs.astral.sh/uv/getting-started/installation/

**F9 — Makefile automation & Git hygiene**
Why: `install/run/debug/clean/lint/lint-strict` targets are graded; `.gitignore` keeps model weights and generated data out of your repo (explicitly forbidden by the subject).
Mastery bar: you can write a Makefile target that calls `uv run python -m pdb -m src ...` for `debug`, and you know exactly which paths (`data/processed/`, `.venv/`, `__pycache__/`, `.mypy_cache/`) belong in `.gitignore` and why.
- [Docs] GNU Make manual (targets, phony) — https://www.gnu.org/software/make/manual/make.html

---

### INTERMEDIATE — Information retrieval engineering
*(This is the theoretical core of the "R" in RAG. Recall@5 thresholds live or die here.)*

**I1 — Tokenization for IR (not the same as LLM tokenization)**
Why: BM25/TF-IDF need *words*, not subword tokens; code identifiers (`snake_case`, `camelCase`) tokenize very differently from English prose.
Mastery bar: you can explain why splitting `first_character_index` into `["first", "character", "index"]` (vs. treating it as one token) changes whether a query for "character index" matches it.
- [Article] Pinecone — "Getting Started with Hybrid Search" (sparse tokenization context) — https://www.pinecone.io/learn/hybrid-search-intro/

**I2 — Chunking strategy design & character-offset bookkeeping**
Why: this is the single most bug-prone part of the mandatory section. `first_character_index`/`last_character_index` are compared **verbatim** by the grader — an off-by-one here silently tanks your recall.
Mastery bar: you can explain why a Python file should be chunked around syntactic units (functions/classes, via the `ast` module) rather than fixed-size windows, why Markdown should split on headers/paragraphs, and — critically — you can prove that `original_text[first_idx:last_idx]` reconstructs your chunk *exactly*, on every chunk, every time.
- [Docs] Python `ast` module — https://docs.python.org/3/library/ast.html
- [Docs] Python string slicing semantics — https://docs.python.org/3/library/stdtypes.html#common-sequence-operations

**I3 — Inverted index & vector space model**
Why: this is the data structure underneath both TF-IDF and BM25.
Mastery bar: you can draw (on paper) an inverted index for 3 tiny documents by hand: term → list of (doc_id, position/count).
- [Article] Pinecone hybrid search intro (sparse vs dense framing) — https://www.pinecone.io/learn/hybrid-search-intro/

**I4 — TF-IDF**
Why: one of the two allowed lexical retrieval methods.
Mastery bar: given 3 tiny documents, you can compute term frequency and inverse document frequency by hand for one term, then explain why cosine similarity (not raw dot product) is used to rank documents of different lengths.
- [Docs] scikit-learn `TfidfVectorizer` — https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html

**I5 — BM25**
Why: the second allowed lexical method, and generally stronger than raw TF-IDF for this kind of corpus.
Mastery bar: you can explain, in your own words, what the `k1` parameter (term-frequency saturation — extra occurrences of a word matter less and less) and the `b` parameter (document-length normalization) each do, and why BM25 tends to beat TF-IDF on short exact-keyword queries.
- [Repo] dorianbrown/rank_bm25 (readable reference implementation) — https://github.com/dorianbrown/rank_bm25
- [Docs] `rank_bm25` on PyPI (usage example) — https://pypi.org/project/rank-bm25/

**I6 — Index persistence**
Why: indexing must run once and be queryable in milliseconds afterward; you can't rebuild in-memory structures on every CLI invocation.
Mastery bar: you can explain the trade-off between `pickle` (fast, Python-only, less safe) and a custom JSON/`numpy` serialization for your index, and why you chose one for this project.
- [Docs] Python `pickle` module (and its caveats) — https://docs.python.org/3/library/pickle.html

**I7 — Recall@k and IoU-based span matching**
Why: this *is* the grading metric. A source only counts if it's in the *same file* and its character span overlaps the reference span above an IoU of 0.05.
Mastery bar: you can compute IoU between two character ranges by hand `[a,b]` vs `[c,d]`, and explain why the bar is deliberately low (0.05) — it rewards "found roughly the right region", not exact chunk-boundary matching.
- [Article] Wikipedia — "Precision and recall" (background) — https://en.wikipedia.org/wiki/Precision_and_recall

**I8 — Pipeline data contracts across stages**
Why: `StudentSearchResults` → `StudentSearchResultsAndAnswer` is a producer/consumer chain; every stage must be independently runnable and re-loadable from disk.
Mastery bar: you can explain why `search_dataset` and `answer_dataset` are separate commands rather than one combined command (hint: re-running only generation after tuning your prompt, without re-running retrieval).

---

### ADVANCED — RAG architecture & LLM integration
*(This is where retrieval theory meets a real, small, imperfect model.)*

**A1 — RAG architecture (indexing / retrieving / augmenting / generating)**
Why: this is the whole project, end to end.
Mastery bar: you can explain, without the subject's diagram in front of you, why retraining a model is not the answer to "the model doesn't know about our codebase", and where each of the four RAG stages sits in your `src/` layout.
- [Video] IBM Technology — "What is Retrieval-Augmented Generation (RAG)?" — https://www.ibm.com/think/videos/rag
- [Docs] IBM — "What is RAG?" — https://www.ibm.com/think/topics/retrieval-augmented-generation

**A2 — Local LLM inference with Hugging Face Transformers**
Why: you must load and run `Qwen/Qwen3-0.6B` locally (CPU-friendly bonus machines).
Mastery bar: you can load a tokenizer + model, build a prompt with `apply_chat_template`, and call `.generate()` with a `max_new_tokens` budget — and explain what each of those three steps does.
- [Docs] Qwen/Qwen3-0.6B model card (official usage snippet) — https://huggingface.co/Qwen/Qwen3-0.6B
- [Docs] Qwen quickstart — https://qwen.readthedocs.io/en/latest/getting_started/quickstart.html

> ⚠️ Don't confuse `Qwen/Qwen3-0.6B` (the causal LM you must use for generation) with `Qwen/Qwen3-Embedding-0.6B` (a different, embedding-only model). They are separate checkpoints with separate purposes.

**A3 — Context-window / token budgeting under a hard character cap**
Why: the moulinette rejects any single retrieved source longer than 2000 characters (`max_context_length`); you must also fit *multiple* sources plus the question inside Qwen3-0.6B's context.
Mastery bar: you can explain the difference between "characters" (what the subject's cap is measured in) and "tokens" (what the model's context window is measured in), and why that distinction matters when you decide how many top-k sources to actually pass to the model.

**A4 — Prompt engineering for grounded, hallucination-resistant answers with a small reasoning model**
Why: Qwen3-0.6B is small and has a "thinking" vs "non-thinking" mode that materially changes output style and latency.
Mastery bar: you can explain what `enable_thinking=False` does when building the chat prompt (suppresses the `<think>...</think>` reasoning block) and why you'd want that off for a task that must return clean, parseable JSON quickly, versus when thinking mode might help reasoning-heavy questions.
- [Docs] Qwen/Qwen3-0.6B model card — "Switching Between Thinking and Non-Thinking Mode" section — https://huggingface.co/Qwen/Qwen3-0.6B
- [Blog] Hugging Face — "The 4 Things Qwen-3's Chat Template Teaches Us" — https://huggingface.co/blog/qwen-3-chat-template-deep-dive

**A5 — Structured JSON output from a small LLM**
Why: `answer`/`answer_dataset` must produce JSON conforming to `StudentSearchResultsAndAnswer`, but small instruction-tuned models don't reliably emit clean JSON on the first try.
Mastery bar: you can describe at least two strategies — (1) prompt-with-schema + `json.loads` + regex fallback + retry, and (2) constrained/guided decoding (e.g. the `outlines` library) — and explain the latency/complexity trade-off between them for a 0.6B model on CPU.
- [Repo] dottxt-ai/outlines (constrained generation) — https://github.com/dottxt-ai/outlines
- [Article] "Let's make LLMs generate JSON!" (comparison of tools) — search "profiq let's make llms generate json"

**A6 — Performance engineering**
Why: hard budgets — indexing ≤ 5 minutes for the whole corpus, retrieval ≤ 90 seconds for 200 questions.
Mastery bar: you can profile your own indexing/retrieval code (e.g. with `time.perf_counter()` or `cProfile`), identify the slowest step, and explain why vectorized `numpy`/`scipy.sparse` operations beat a Python `for` loop over every document for every query.
- [Docs] Python `cProfile` — https://docs.python.org/3/library/profile.html

**A7 — Defensive CLI engineering**
Why: the CLI is *tested* with degenerate inputs (empty query, `k=0`, missing files, malformed JSON) and must never crash.
Mastery bar: you can list, from memory, every degenerate input mentioned in the subject and state what your CLI does for each one (not "it will probably work" — the actual behavior).

---

### BONUS — Only after the mandatory part is 100% validated
*(One point each, five total. Pick based on what deepens the concept you found hardest above.)*

**B1 — Semantic embeddings**
Mastery bar: you can explain why "vLLM continuous batching" and "how does vLLM batch requests" might retrieve the same chunk under embeddings but not under BM25.
- [Docs] sentence-transformers/all-MiniLM-L6-v2 model card — https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2

**B2 — Hybrid retrieval**
Mastery bar: you can explain why you can't just add a BM25 score and a cosine-similarity score directly (different scales) and describe one fix (min-max normalization, or rank fusion instead of score fusion).
- [Article] Pinecone — "Getting Started with Hybrid Search" — https://www.pinecone.io/learn/hybrid-search-intro/

**B3 — Incremental indexing**
Mastery bar: you can explain how you'd detect that a file changed (mtime vs content hash) and why content hash is more robust.

**B4 — Caching**
Mastery bar: you can explain what invalidates your cache (a re-index? a changed `k`?) and why silently serving a stale cache is worse than a cold cache.

**B5 — Local HTTP API**
Mastery bar: you can explain why your HTTP request/response bodies should reuse the *same* pydantic models as your CLI, instead of inventing new ones.

---

## 2. The 20-Day Roadmap

**Guiding split:** ~9 days theory-and-toy-exercises, ~8 days building the real mandatory project, ~3 days hardening/buffer. Don't compress the first block to "get to the real code faster" — the toy exercises are what make Days 14-17 fast instead of a debugging nightmare on a 13,000-chunk corpus.

| Day | Phase | Focus | Concepts |
|---|---|---|---|
| 1 | Orientation | Read the whole subject twice (the info-boxes matter). Skim a RAG overview. `git init`, `uv init`, empty `src/`, Makefile stub, `.gitignore`. | F8, F9, A1 |
| 2 | Foundations | Pydantic + typing theory → **Exercise 1** | F1, F2, F4 |
| 3 | Foundations | Fire + tqdm + Makefile + flake8/mypy theory → **Exercise 2** | F5, F6, F7, F9 |
| 4 | Foundations | Exceptions/context managers/robust I/O theory → **Exercise 3** | F3 |
| 5 | IR Theory | Tokenization, inverted index, TF-IDF math (by hand first) | I1, I3, I4 |
| 6 | IR Theory | Finish TF-IDF by hand → **Exercise 4**; then BM25 math (k1, b) | I4, I5 |
| 7 | IR Theory | Chunking strategy design → **Exercise 5** (two chunkers, offset-exact) | I2 |
| 8 | IR Theory | Toy BM25 search engine on your chunked corpus → **Exercise 6** | I3, I5, I6 |
| 9 | IR Theory | Recall@k / IoU theory → **Exercise 7** (your own evaluator, unit-tested) | I7 |
| 10 | LLM Theory | Transformers basics, Qwen3 chat template, thinking mode → **Exercise 8** | A2 |
| 11 | LLM Theory | Prompting & grounding under a context budget → **Exercise 9** | A3, A4 |
| 12 | LLM Theory | Structured JSON output strategies → **Exercise 10** | A5 |
| 13 | LLM Theory | Toy end-to-end RAG (chunk → BM25 → prompt → Qwen3 → parsed answer) → **Exercise 11** | A1, I8, A6 |
| 14 | Build | Scaffold the real repo: `pyproject.toml`, `uv.lock`, `src/` layout, the 6 pydantic models, README skeleton, `data/raw/` in place | F2, F8, F9 |
| 15 | Build | Real indexing pipeline (both chunkers) on the vLLM corpus + `index` command, under 5 minutes, with tqdm | I2, I6, F5, F6, A6 |
| 16 | Build | Real retrieval + `search`/`search_dataset`; run `evaluate` against the public dataset; tune chunk size / method until you clear 80% (docs) / 50% (code) recall@5 | I3, I4/I5, I7 |
| 17 | Build | Real generation + `answer`/`answer_dataset`/`evaluate`; wire degenerate-input handling everywhere | A2, A3, A4, A5, A7 |
| 18 | Harden | `flake8`/`mypy --strict` to zero errors; a handful of `pytest` unit tests on chunkers/evaluator/edge cases; profile against the 5-min/90-sec budgets | F7, A6, A7 |
| 19 | Polish | Write the full README (all required sections); run the *exact* moulinette workflow end-to-end (`index` → `search_dataset` → `moulinette evaluate_student_search_results` → `answer_dataset`) | I8, A7 |
| 20 | Buffer | One bonus if time allows; peer-review your own code out loud (per the subject's own "AI Instructions" chapter); do a timed mock "recode" drill on a random function | B1–B5, all |

---

## 3. Mini-Exercises

Do these **before** touching `data/raw/vllm-0.10.1/` or the moulinette. Each is scoped to fit in a session; each has a "definition of done" you can check yourself against — no answer key, because checking it yourself *is* the exercise.

**Exercise 1 — Model Playground** *(F1, F2, F4)*
Recreate the six pydantic models from the subject in a scratch file, from memory (peek only to check afterward). Instantiate an `AnsweredQuestion`, round-trip it through `model_dump_json()` → `model_validate_json()`, then deliberately pass an `int` where `file_path` expects a `str` and read the `ValidationError`.
*Done when:* you can explain, unprompted, why `AnsweredQuestion` extends `UnansweredQuestion` instead of duplicating its fields, and what problem `default_factory=lambda: str(uuid.uuid4())` solves that `default=str(uuid.uuid4())` would not.

**Exercise 2 — Two-Command CLI** *(F5, F6, F7, F9)*
Build a throwaway Fire CLI with two commands: `greet(name: str) -> str` and `sum_range(n: int) -> int` (wrap the loop in `tqdm` even though it's fast — just to practice the API). Add a Makefile with `install`, `run`, `lint`. Run flake8 and mypy with the exact flags from the subject until both are clean.
*Done when:* `uv run python -m yourpkg greet --name=Ada` works from a cold `uv sync`, and `make lint` exits 0.

**Exercise 3 — Never Crash** *(F3)*
Write `load_json_safely(path: str) -> dict | None` that: file doesn't exist → returns `None`; file isn't valid JSON → returns `None`; file is valid JSON but not a dict (e.g. a list) → returns `None`. No exception should ever propagate out of this function.
*Done when:* you have three quick test calls (missing file, garbage text, a JSON array) and none of them raise.

**Exercise 4 — Hand-Rolled TF-IDF** *(I4)*
Write 5 one-paragraph toy documents (mix in some repeated vocabulary on purpose). By hand, compute TF and IDF for one term that appears in 2 of the 5 docs. Then implement TF-IDF + cosine similarity ranking in plain Python/numpy, and verify your ranking matches `sklearn.feature_extraction.text.TfidfVectorizer` on the same corpus and query.
*Done when:* your hand calculation and your code agree, and you can explain in one sentence why a term appearing in *all* documents gets an IDF near zero.

**Exercise 5 — Two Chunkers** *(I2)*
Write `chunk_python(source: str, max_chunk_size: int) -> list[tuple[str, int, int]]` using the `ast` module to split on function/class boundaries (falling back to fixed windows for anything else), and `chunk_markdown(source: str, max_chunk_size: int) -> list[tuple[str, int, int]]` splitting on headers/paragraphs. Each tuple is `(chunk_text, first_idx, last_idx)`.
*Done when:* for every chunk of every test file, `source[first_idx:last_idx] == chunk_text` exactly — write this as an assertion, not a manual check.

**Exercise 6 — BM25 vs TF-IDF Showdown** *(I3, I5)*
Build a 15-20 "document" toy corpus mixing short Python snippets and short Markdown paragraphs (reuse Exercise 5's chunker). Index it with both TF-IDF (Exercise 4) and BM25 (`rank_bm25` or your own). Run the same 5 queries against both, compare the top-3 results.
*Done when:* you can write 2-3 sentences on a case where BM25's ranking differs from TF-IDF's, and why (usually: BM25 saturates repeated terms, TF-IDF's raw dot product doesn't as cleanly without careful normalization).

**Exercise 7 — Recall@k Judge** *(I7)*
Implement `recall_at_k(predicted: list[MinimalSource], reference: list[MinimalSource], k: int, iou_threshold: float = 0.05) -> float`. Unit-test it against hand-crafted cases: exact span match, partial overlap just above/below the IoU threshold, same range but *different file* (must not count), and `k` larger than your prediction list.
*Done when:* all four hand-crafted cases pass, and you can explain why the "different file, same range" case is a zero, not a partial credit.

**Exercise 8 — Say Hello to Qwen** *(A2, A4)*
Load `Qwen/Qwen3-0.6B` locally. Generate a response to the same short prompt twice: once with `enable_thinking=True`, once with `enable_thinking=False`. Diff the outputs and time both calls.
*Done when:* you can point at the `<think>...</think>` block (or its absence) in the raw decoded text and explain what triggered it.

**Exercise 9 — Grounded or Not** *(A3, A4)*
Fix a 3-snippet context and ask a question the context does **not** answer. Iterate on your prompt (system instructions, phrasing of "answer only using the context below") until the model says something like "the context doesn't cover this" instead of inventing an answer. Log what you tried and what changed the behavior.
*Done when:* you have at least 2 prompt versions logged with their outputs, and one clearly reduces hallucination versus the other.

**Exercise 10 — JSON or Bust** *(A5)*
Run 10 varied (context, question) pairs through Qwen3-0.6B, asking each time for a JSON object matching a tiny 2-field schema (e.g. `{"answer": str, "confidence": str}`). Track how many parse with plain `json.loads()` vs. how many need a "grab the first `{...}` block with regex" fallback vs. how many fail outright.
*Done when:* you have a parse-success rate and can describe the most common failure mode you saw (e.g. trailing commentary after the JSON, or an unescaped quote inside a string value).

**Exercise 11 — Toy End-to-End RAG** *(A1, A6, I8)*
Wire it all together on your toy corpus: chunk → BM25 index → retrieve top-k → build a context-budgeted prompt → Qwen3-0.6B → parse into a `MinimalAnswer`. One `__main__` script, no CLI yet. Time the whole run.
*Done when:* it runs start to finish on a query you invent, producing a valid `MinimalAnswer`, and you know which stage took the most wall-clock time.

**Exercise 12 (optional preview) — Embed a Sentence** *(B1)*
Encode 5 toy sentences with `sentence-transformers/all-MiniLM-L6-v2`, compute the cosine similarity matrix, and check whether two *semantically* similar sentences that share almost no words (e.g. "the car broke down" / "the automobile stopped working") score high — contrast this with what Exercise 6's BM25 would have done with the same pair.
*Done when:* you can state, in one sentence, the complementary weakness BM25 has that embeddings fix (and vice versa — embeddings can miss an exact identifier match that BM25 nails).

---

## 4. Self-Evaluation Checklist

Work through this the way the evaluator will. Anything unchecked two days before your deadline is a real risk, not a "nice to have."

### Tooling & repo hygiene
- [ ] `uv sync` from a clean clone (no manual pip installs) brings up a working environment
- [ ] `pyproject.toml` and `uv.lock` are committed at the repo root
- [ ] `Makefile` has working `install`, `run`, `debug`, `clean`, `lint` (and ideally `lint-strict`) targets
- [ ] `flake8 .` passes with zero errors, including on bonus files if implemented
- [ ] `mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs` passes with zero errors
- [ ] `.gitignore` excludes `__pycache__/`, `.mypy_cache/`, virtual envs, and — critically — model weights and generated data under `data/processed/` and `data/output/`
- [ ] No large data files, model weights, or generated outputs are committed to the repo

### Data models
- [ ] `MinimalSource`, `UnansweredQuestion`, `AnsweredQuestion`, `RagDataset`, `MinimalSearchResults`, `MinimalAnswer`, `StudentSearchResults`, `StudentSearchResultsAndAnswer` are all implemented as pydantic `BaseModel`s
- [ ] Service/orchestration classes (indexer, retriever, pipeline) are allowed to be plain classes — you haven't over-engineered these into pydantic unnecessarily
- [ ] Every function that builds or consumes these models has correct type hints

### Indexing
- [ ] Two distinct chunking strategies exist: one for Python source, one for Markdown/text
- [ ] Chunk size is a CLI argument (`--max_chunk_size`), default 2000, and nothing you produce ever exceeds it
- [ ] For every chunk, `raw_text[first_character_index:last_character_index]` reconstructs the chunk exactly
- [ ] `file_path` in every `MinimalSource` matches the corpus path verbatim (e.g. `data/raw/vllm-0.10.1/...`)
- [ ] Full-corpus indexing completes in under 5 minutes, with a `tqdm` progress bar
- [ ] The index is persisted under `data/processed/` and loads fast on subsequent runs (no re-chunking on every query)

### Retrieval
- [ ] At least one of TF-IDF or BM25 is implemented (both is fine, but one is enough)
- [ ] `search` returns correct top-k results for a single query
- [ ] `search_dataset` runs over a full JSON dataset and writes a valid `StudentSearchResults` file
- [ ] Your own `evaluate` command reports recall@1/3/5/10 and you've checked it against a public dataset
- [ ] You independently clear ≥ 80% recall@5 on docs questions and ≥ 50% recall@5 on code questions before submission
- [ ] Batch retrieval over 200 questions completes in under 90 seconds

### Generation
- [ ] `Qwen/Qwen3-0.6B` is the model actually used (or another model that you've verified everything still works with as a drop-in default)
- [ ] The retrieved context is built within the model's token budget, not just the 2000-character-per-source cap
- [ ] `answer` and `answer_dataset` produce output conforming to `StudentSearchResultsAndAnswer`
- [ ] Output JSON reliably parses — you've measured this over more than a handful of test questions, not just eyeballed one example
- [ ] Answers are grounded (no confidently invented facts) on at least a spot-check of outputs
- [ ] You understand and can explain your own strategy for getting valid JSON out of a 0.6B model

### CLI & output contracts
- [ ] All six commands exist: `index`, `search`, `search_dataset`, `answer`, `answer_dataset`, `evaluate`
- [ ] Every input/output path is a CLI argument — nothing is hard-coded
- [ ] The directory layout matches the subject exactly: `src/`, `data/raw/`, `data/processed/`, `data/datasets/{UnansweredQuestions,AnsweredQuestions}/`, `data/output/search_results/<scope>/`, `data/output/search_results_and_answer/<scope>/`
- [ ] `--save_directory` is scoped by dataset type so you don't silently overwrite one dataset's results with another's

### Robustness (this is graded by deliberately hostile input)
- [ ] Empty query → handled, no crash
- [ ] `k=0` → handled, no crash
- [ ] Nonsensical/gibberish query → handled, no crash
- [ ] Missing input file → handled, no crash, clear message
- [ ] Malformed JSON input → handled, no crash, clear message
- [ ] You never `import` or call the moulinette from your own code

### README
- [ ] First line is the exact italicized attribution sentence with your login(s)
- [ ] Description, Instructions, Resources sections present
- [ ] Resources section documents *how AI was used*, for which tasks, on which parts of the project
- [ ] System architecture, chunking strategy, retrieval method, performance analysis, design decisions, challenges faced, and example usage are all covered
- [ ] Written entirely in English

### Defense readiness
- [ ] You can explain, without notes, why IoU 0.05 is a low bar and what that implies about how exact your chunk boundaries need to be
- [ ] You can explain, without notes, why `file_path` must match exactly and what happens if your prefix differs from the reference
- [ ] You can walk through your chunking code function by function and justify every design choice
- [ ] You've done a timed practice "recode": pick a random function in your own code and give yourself 10 minutes to modify its behavior as if an evaluator asked
- [ ] A peer has reviewed your retrieval and generation code and you've defended your choices to them

### Bonus (only attempt after every box above is checked)
- [ ] Semantic embeddings added alongside the lexical index
- [ ] Hybrid retrieval combines both rankings sensibly (normalized, not just added)
- [ ] Incremental indexing re-indexes only changed files
- [ ] Caching speeds up cold start / repeated queries, with a clear invalidation rule
- [ ] A local HTTP API reuses the same pydantic models as the CLI

---

## 5. Appendix — Cheat Sheet

**TF-IDF**, for term *t* and document *d* in corpus *D*:
`tf(t, d)` = count of *t* in *d* (often normalized by document length)
`idf(t, D)` = log( |D| / number of documents containing *t* )
`tfidf(t, d) = tf(t, d) × idf(t, D)`
Rank documents for a query by cosine similarity between the query's TF-IDF vector and each document's.

**BM25**, informally: like TF-IDF, but term frequency contributes with *diminishing returns* (controlled by `k1`, typically ~1.2-2.0) and is normalized against how long the document is relative to the corpus average (controlled by `b`, typically ~0.75).

**IoU (Intersection over Union)** for two character ranges `[a1, a2]` and `[b1, b2]`:
`intersection = max(0, min(a2, b2) - max(a1, b1))`
`union = (a2 - a1) + (b2 - b1) - intersection`
`IoU = intersection / union`
A source counts as a match only if it's in the same file **and** IoU ≥ 0.05.

**Recall@k**: for a question, (number of reference sources found among your top-k results) / (total number of reference sources for that question), then averaged across all questions.

**Key CLI commands (from the subject, for quick reference):**
```
uv run python -m src index --max_chunk_size 2000
uv run python -m src search "<query>" --k 5
uv run python -m src search_dataset --dataset_path <path> --k 10 --save_directory <dir>
uv run python -m src answer "<query>" --k 5
uv run python -m src answer_dataset --student_search_results_path <path> --save_directory <dir>
uv run python -m src evaluate --student_search_results_path <path> --dataset_path <path>
```

**Hard numeric budgets to keep taped above your monitor:**
- Max chunk size: 2000 characters (never exceed)
- Indexing time: ≤ 5 minutes for the whole corpus
- Retrieval throughput: ≤ 90 seconds for 200 questions
- Recall@5 targets: ≥ 80% docs, ≥ 50% code
- IoU match threshold: 0.05

---

*Good luck. Build the toy versions first — every hour spent on Exercises 1-12 is an hour you won't spend debugging an off-by-one character index inside a 13,000-chunk index at 2am on day 17.*
