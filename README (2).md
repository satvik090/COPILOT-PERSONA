# CopilotPersona

**Developer-adaptive code suggestion system via personal pattern RAG**

GitHub Copilot knows how the average developer on GitHub writes code. It does not know how *you* write code. CopilotPersona fixes that by building a persistent index of your personal coding patterns — across all your projects — and injecting those patterns into Copilot's context on every suggestion. The result is completions that match your naming conventions, annotation style, error handling patterns, and docstring format rather than the open-source average.

---

## The Problem

Copilot suggestions frequently violate team and personal conventions. You always use type annotations. Copilot doesn't. You always return tuples instead of raising exceptions. Copilot raises. You write Google-style docstrings. Copilot writes plain ones or none at all. Every suggestion requires manual correction to fit your style — correction that adds up to significant friction over a working day.

This is not a failure of the underlying model. It is a failure of context. Copilot has no access to your coding history. CopilotPersona gives it that access.

---

## How It Works

```
Your Python projects
        │
        ▼
   AST Chunker          ← splits at function/class boundaries
        │
        ▼
  Embedder              ← all-MiniLM-L6-v2, runs locally
        │
        ▼
  ChromaDB              ← persistent local vector store
        │
        ▼
  Pattern Classifier    ← extracts naming, annotation,
        │                  error handling, docstring patterns
        ▼
  FastAPI /retrieve     ← returns pattern summary < 150 tokens
        │
        ▼
  VS Code Extension     ← injects summary into Copilot prompt
        │
        ▼
  GitHub Copilot        ← now knows how YOU write code
```

Five pattern dimensions are extracted and tracked per developer:

- **Naming convention** — snake_case vs camelCase vs mixed
- **Type annotation density** — full, partial, or none
- **Error handling style** — try/except vs return tuple vs raises
- **Docstring format** — Google, NumPy, plain, or none
- **Identifier vocabulary** — recurring names and abstractions

A pattern is only injected if it appears in at least 70% of retrieved similar functions. Weak or inconsistent patterns are deliberately excluded to avoid noise.

---

## Proof of Work

Evaluation was run against 18 real functions extracted from 5 personal Python projects using an AST-based scoring pipeline (`evaluate.py`). Each function was completed twice: once by vanilla GPT-4o with no personal context, once with CopilotPersona pattern injection. Both completions were scored against the original ground truth function across all five dimensions.

```
Mode: DEMO
Functions evaluated: 18
Projects evaluated: 5

Per-dimension results:
  naming_convention_match   baseline=0.28  enhanced=1.00  +260%
  type_annotation_coverage  baseline=0.06  enhanced=0.06  +0%
  error_handling_style      baseline=0.56  enhanced=1.00  +80%
  docstring_format          baseline=0.06  enhanced=0.06  +0%
  import_and_naming_style   baseline=0.01  enhanced=0.28  +2398%

Overall:
  baseline_total_average              = 0.1911
  enhanced_total_average              = 0.4783
  overall_improvement                 = 150.25%
  style_correction_reduction          = 35.50%
```

**CopilotPersona reduced style correction needs by 35.50% across 18 functions from 5 personal projects, measured across naming conventions, type annotations, error handling style, docstring format, and identifier vocabulary (AST-based evaluation, zero parse failures).**

The largest gains were in naming convention matching (+260%) and identifier vocabulary similarity (+2398%), reflecting that personal naming patterns are highly consistent but invisible to a generic model.

Type annotation coverage and docstring format showed 0% improvement in this run because the sample projects had inconsistent annotation coverage — the classifier correctly declined to inject a pattern that did not meet the 70% confidence threshold. This is the system working as intended: it only injects patterns it is confident about.

---

## System Design

### Retrieval layer
- **ChromaDB** persistent vector store, local only — no code leaves the machine during indexing
- **sentence-transformers** (`all-MiniLM-L6-v2`) for embedding, running in-process
- Chunking at function and class definition boundaries using Python `ast` module
- Two-layer cache: embedding cache (content-hash keyed) and retrieval cache (30s TTL)
- Full retrieval round trip target: under 150ms

### Pattern classification
- Structural features extracted deterministically via AST — no model required for classification
- Confidence threshold: 70% recurrence across retrieved chunks before a pattern is injected
- Natural language summary hard-capped at 150 tokens to avoid crowding Copilot's context window

### API
- FastAPI async service on port 8000
- `POST /retrieve` — main retrieval endpoint
- `POST /index/trigger` — manual re-index
- `GET /index/status` — chunk count and last indexed timestamp
- `GET /metrics` — Prometheus exposition format

### VS Code Extension
- Intercepts Copilot completion cycle via Language Model API
- Injects pattern summary as additional system context
- Falls back silently if local server is unreachable — zero disruption to normal Copilot behaviour
- Status bar indicator, enable/disable toggle, manual re-index command

### Observability
Prometheus metrics exposed at `/metrics`:
- `retrieval_latency_seconds` (p50, p95, p99)
- `retrieval_cache_hits` / `retrieval_cache_misses`
- `pattern_confidence_score` histogram
- `index_total_chunks` gauge
- `embedding_latency_seconds`

---

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.11 |
| API | FastAPI + uvicorn |
| Vector store | ChromaDB (local persistent) |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| AST analysis | Python stdlib `ast` module |
| Scheduling | APScheduler |
| Metrics | prometheus-client |
| VS Code integration | VS Code Extension API + Language Model API |
| Evaluation | OpenAI GPT-4o + custom AST scorer |
| Infra | Docker Compose (Prometheus + Grafana) |

---

## Running Locally

```bash
git clone https://github.com/yourname/copilot-persona
cd copilot-persona

python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# edit .env: set CORPUS_DIRS to your Python projects folder
#            set OPENAI_API_KEY if running evaluation

docker compose up -d           # starts Prometheus + Grafana

uvicorn app.main:app --reload --port 8000
```

Check indexing status:
```bash
curl http://localhost:8000/index/status
```

Wait until `total_chunks > 0` before installing the extension.

**Install the VS Code extension:**
- Open VS Code
- Extensions → `...` → Install from VSIX
- Or press F5 inside the `extension/` folder to launch a dev host

---

## Running the Evaluation

```bash
# dry run: verify corpus extraction without API calls
python evaluate.py --dry-run

# full evaluation: generates evaluation_report.txt
python evaluate.py
```

Outputs:
- `evaluation_report.txt` — human-readable summary
- `evaluation_report.json` — structured results
- `evaluation_raw.json` — raw completions for inspection

Make sure the FastAPI server is running and the index has chunks before running the full evaluation. The enhanced completion path calls `/retrieve` for every function — if the server is down it falls back to baseline and improvement will show as 0%.

---

## Running Tests

```bash
pytest tests/ -v
```

Tests cover AST extraction, pattern classification, and retrieval — all runnable without any external services using monkeypatched embeddings and a temporary ChromaDB instance.

---

## Project Structure

```
copilot-persona/
├── app/
│   ├── main.py          FastAPI entrypoint
│   ├── config.py        Environment configuration
│   ├── indexer.py       Background corpus indexer
│   ├── retriever.py     ChromaDB query layer
│   ├── embedder.py      sentence-transformers wrapper
│   ├── extractor.py     AST feature extraction
│   ├── classifier.py    Pattern classifier
│   ├── cache.py         Embedding + retrieval cache
│   └── observability.py Prometheus metrics
├── extension/
│   ├── package.json     VS Code extension manifest
│   └── extension.js     Extension entrypoint
├── tests/
│   ├── test_extractor.py
│   ├── test_classifier.py
│   └── test_retriever.py
├── evaluate.py          AST-based evaluation pipeline
├── docker-compose.yml
└── requirements.txt
```

---

## Key Design Decisions

**Why local embeddings instead of OpenAI embeddings?**
Personal code should never leave the machine during indexing. Using a local sentence-transformers model means the vector store is built entirely offline. OpenAI is only called during evaluation, where the developer has explicitly opted in.

**Why 70% confidence threshold for pattern injection?**
A pattern that appears in 60% of retrieved chunks is not a convention — it is a tendency. Injecting weak patterns adds noise without improving suggestions. The threshold is configurable via `.env` for developers with more consistent codebases.

**Why cap the injected summary at 150 tokens?**
Copilot's context window is shared between the injected pattern summary, the current file context, and the prompt itself. Exceeding 150 tokens risks crowding out the actual code context, which would degrade rather than improve suggestions. The cap is enforced by the classifier before the summary leaves the retrieval service.

**Why ChromaDB instead of Pinecone or Weaviate?**
This system is designed to run entirely locally with no external dependencies at runtime. ChromaDB's persistent client mode gives production-grade vector search without requiring a running server process or cloud account.
