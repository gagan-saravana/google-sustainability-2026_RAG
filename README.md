# Google 2026 Environmental Report Q&A

A local, citation-oriented retrieval-augmented generation (RAG) demo for asking
questions about Google's 2026 Environmental Report. The project turns a
pre-converted Markdown copy of the report into searchable passages, stores
Gemini embeddings in a local Chroma database, and answers questions from the
most relevant passages through either a terminal command or a NiceGUI chat
interface.

This is a demonstration project, not an official Google product or a
production-ready multi-user service. Answers are model-generated; verify
important figures against the cited report passages.

## What it does

- LlamaParse (from LlamaCloud by LlamaIndex) was used to parse the official pdf into clean markdown (upstream step).
- Cleans and chunks the report while retaining heading breadcrumbs, table
  structure, captions, and source-type metadata.
- Embeds report passages and questions with Google's Gemini API.
- Persists vectors and passage metadata locally with ChromaDB.
- Retrieves the closest passages and asks Gemini to produce a concise,
  report-grounded answer with inline citations.
- Offers both a command-line question interface and a browser chat UI.
- Provides local tests for chunking, retrieval/synthesis, follow-up question
  condensing, and demo limits; these tests do not require API calls.

The source file in `data/` is already-converted Markdown. PDF parsing and
RAGAS evaluation are not part of the current runtime pipeline.

## Architecture

```text
                         OFFLINE INDEXING
  data/google_2026_sustainability.md
                  |
                  v
       src/chunking.py: clean, parse structure,
       create contextual chunks and metadata
          |                         |
          |                         +--> python -m src.chunking
          |                                -> data/chunks.jsonl
          v
       src/index.py: skip existing content-hash IDs,
       embed missing chunks in batches, upsert
          |
          +--> src/embeddings.py --> Gemini embedding API
          |
          +--> ChromaDB PersistentClient
               -> chroma_db/ (local vector index)

                       QUESTION ANSWERING
  Browser -> app.py (NiceGUI) -- follow-up? --> src/condense.py
                    |                               |
                    +-------------------------------+
                                    |
                              standalone question
                                    |
                                    v
                 src/synthesize.py: answer_question()
                         /                    \
                        v                      v
       src/retrieve.py: embed query       src/synthesize.py:
       and search ChromaDB                 build grounded prompt
                |                                  |
                v                                  v
       src/embeddings.py                   Gemini generation API
                |                                  |
                +--> Gemini embedding API          v
                                         Answer + parsed citations
                                                    |
                                    +---------------+---------------+
                                    v                               v
                              NiceGUI chat                    src/ask.py
                                                               (terminal)
```

Indexing is an explicit, offline-from-serving preparation step: the app and
terminal question command read the persisted index, but do not create it. The
Gemini API is used for both embeddings and generated text. ChromaDB itself runs
locally and does not require a hosted vector database.

## Request and development flow

### 1. Prepare the report index

`src.chunking` reads the Markdown document and emits chunks with stable IDs and
metadata. `src.index` then opens or creates the Chroma collection, embeds only
chunks that are not already present, and saves each batch as it goes. The index
can therefore be resumed by running the command again after an interruption.

### 2. Answer a question

The CLI (`src.ask`) and web app (`app.py`) both call the shared
`answer_question()` pipeline:

1. For a follow-up in the web chat, `src.condense` rewrites it as a standalone
   question using recent conversation history. The first question skips this
   extra model call.
2. `src.embeddings` formats and embeds the question.
3. `src.retrieve` queries Chroma for the top passages (six by default),
   excluding endnotes unless requested.
4. `src.synthesize` numbers the passages and sends a grounded prompt to Gemini.
   It parses citation markers from the answer so the UI or CLI can display the
   cited passages.

The NiceGUI event loop delegates the blocking model and database work to a
worker thread. The chat keeps short-lived history and usage counters in server
memory; both are lost when the process restarts.

## Technology stack

| Area | Technology | Role |
| --- | --- | --- |
| Language | Python 3.10+ | Application, CLI tools, and tests. The source uses Python 3.10 type syntax. |
| Generative AI | Google Gen AI SDK (`google-genai`) | Calls Gemini for embedding and text generation. |
| Embedding model | `gemini-embedding-2`, 768 dimensions | Creates vectors for report chunks and questions. |
| Answer model | `gemini-3.5-flash-lite` | Generates grounded answers and follow-up rewrites. |
| Vector store | ChromaDB | Local persistent cosine-similarity index and passage metadata. |
| Web UI | NiceGUI | Python-defined browser chat served locally. |
| Configuration | `python-dotenv` | Loads local settings such as `GEMINI_API_KEY` from `.env`. |
| API resilience | Tenacity | Retries Gemini rate-limit and selected server errors with backoff. |
| Tests | pytest | Runs the local unit tests. |
| Input format | Markdown | Pre-converted report content in `data/`; no PDF parser runs here. |

Dependency lower bounds are listed in `requirements.txt`; exact versions are
not pinned.

## Design choices

### Structure-aware chunks

The report's headings provide useful context that a fixed-size splitter could
discard. Chunks retain a breadcrumb such as `Energy > Energy for our data
centers > Nuclear`, and those breadcrumbs are used as document titles during
embedding and as source labels in answers. Paragraphs are packed to a target
size; oversized paragraphs are split at sentence boundaries with a small
overlap.

Tables are rendered as pipe-separated rows rather than split arbitrarily.
Large tables are split between rows with their header repeated, while figure
captions are attached to following data tables when possible. Parser noise,
table-of-contents material, and standalone page numbers are removed.
Endnotes remain indexable, but are tagged so normal retrieval can exclude them.

### Asymmetric embeddings and explicit vectors

Documents use the format `title: <section breadcrumb> | text: <chunk>`, while
queries use `task: question answering | query: <question>`. This preserves
different document and query roles for retrieval. The model and output
dimension are centralized in `src/config.py`, recorded in Chroma metadata, and
validated when indexing or querying. The app supplies Gemini-generated vectors
to Chroma explicitly, ensuring the same embedding space is used at both ends.

### Local, resumable indexing

Chroma's persistent local directory makes the demo easy to run without an
external database. Stable content-hash IDs and batch-by-batch upserts make
indexing incremental: a rerun embeds missing chunks rather than repeating all
work. A rebuild is available when the source or chunking strategy changes.

### Grounded responses and inspectable sources

The generation prompt restricts answers to retrieved passages, asks the model
to say when the answer is not in the report, and requires inline passage
citations. Citation parsing then lets the UI show the sources actually
referenced, with passage text available for inspection. Endnotes are filtered
out by default to reduce retrieval noise.

### Testable boundaries

Retrieval, generation, and chunking dependencies can be replaced by test
doubles. That allows core behavior to be tested without a Gemini key, network
access, or a live Chroma collection.

### Demo-oriented quota and session controls

The chat limits each browser to three questions by default and enforces a
global daily cap to protect a free-tier API quota. A question slot is reserved
before model calls and refunded on failure; concurrent submissions from the
same browser are blocked. These counters and conversation history are
in-memory, so the application is intended for a single local process rather
than a multi-worker deployment.

## Project structure

```text
project_rag/
├── app.py                         # NiceGUI browser chat and server startup
├── requirements.txt               # Runtime and test dependencies
├── .env.example                   # Template for local environment variables
├── .gitignore                     # Excludes local secrets and generated artifacts
├── README.md                      # Project guide
├── data/
│   ├── google_2026_sustainability.md  # Pre-converted report source
│   └── chunks.jsonl               # Generated, human-readable chunk dump
├── src/
│   ├── config.py                  # Models, paths, chunking and demo settings
│   ├── gemini_client.py           # Shared Gemini client and retry policy
│   ├── chunking.py                # Markdown cleanup, parsing and chunking
│   ├── embeddings.py              # Document/query formatting and embeddings
│   ├── index.py                   # Incremental Chroma indexing CLI
│   ├── retrieve.py                # Vector search and endnote filtering
│   ├── synthesize.py              # Grounded answer prompts and citations
│   ├── condense.py                # Follow-up question rewriting
│   ├── limits.py                  # In-memory browser and daily quota limits
│   └── ask.py                     # Terminal question-answering CLI
├── tests/
│   ├── test_chunking.py           # Chunking and report sanity checks
│   ├── test_synthesize.py         # Retrieval, prompting and citation tests
│   ├── test_condense.py           # Follow-up rewriting tests
│   └── test_limits.py             # Demo quota/session tests
└── chroma_db/                     # Generated local persistent vector index
```

`.venv/`, `.env`, Python bytecode, the chunk dump, and `chroma_db/` are local
artifacts and are excluded by `.gitignore`. `data/chunks.jsonl` is created by
the chunk-inspection command; it is not needed by the runtime once the index
has been built.

## Run locally

### Prerequisites

- Python 3.10 or later.
- A Gemini API key. Create one in [Google AI Studio](https://aistudio.google.com/apikey).
- Network access to the Gemini API for indexing and question answering.

### 1. Create an environment and install dependencies

From the project root, use PowerShell on Windows:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Open `.env` and replace `your-key-here` with your Gemini API key:

```dotenv
GEMINI_API_KEY=your-actual-key
```

Do not commit `.env` or share the key. It is already ignored by `.gitignore`.
The commands below use the environment's Python executable directly, so PowerShell
activation policy changes are not required.

For macOS or Linux, the equivalent setup is:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
cp .env.example .env
```

Then edit `.env` with your Gemini API key. Run the remaining commands with the
environment activated.

### 2. Inspect chunks (optional)

This parses the report locally, prints chunk statistics, and writes the
human-readable JSON Lines dump to `data/chunks.jsonl`. It does not call Gemini:

```powershell
# Windows
.\.venv\Scripts\python.exe -m src.chunking
```

```bash
# macOS / Linux (with .venv activated)
python -m src.chunking
```

### 3. Build the vector index

Start with a no-embedding dry run, then build the complete index:

```powershell
# Windows
.\.venv\Scripts\python.exe -m src.index --dry-run
.\.venv\Scripts\python.exe -m src.index
```

```bash
# macOS / Linux (with .venv activated)
python -m src.index --dry-run
python -m src.index
```

Indexing calls the Gemini embedding API and stores vectors in `chroma_db/`.
It is resumable and safe to rerun. Optional indexing commands in PowerShell on
Windows:

```powershell
.\.venv\Scripts\python.exe -m src.index --limit 20
.\.venv\Scripts\python.exe -m src.index --rebuild
```

On macOS/Linux, use `python -m src.index --limit 20` or
`python -m src.index --rebuild` with the virtual environment activated.

`--limit` intentionally creates a partial index; run the command again without
the limit to index the remaining chunks. `--rebuild` re-embeds the full report
and may use additional API quota. Do not use it unless a clean re-index is
needed.

### 4. Ask from the terminal

```powershell
# Windows
.\.venv\Scripts\python.exe -m src.ask "How much clean energy did Google procure in 2025?"
.\.venv\Scripts\python.exe -m src.ask "What is the data center PUE?" --show-chunks
```

```bash
# macOS / Linux (with .venv activated)
python -m src.ask "How much clean energy did Google procure in 2025?"
python -m src.ask "What is the data center PUE?" --show-chunks
```

Useful options:

```text
--k N                 Retrieve N passages (default: 6)
--include-endnotes    Allow endnote passages into retrieval
--show-chunks         Print all retrieved passages, not only cited sources
```

### 5. Start the browser chat

```powershell
# Windows
.\.venv\Scripts\python.exe app.py
```

```bash
# macOS / Linux (with .venv activated)
python app.py
```

Open [http://localhost:8080](http://localhost:8080). The app checks that the
index exists and reports its chunk count on startup. It uses port `8080` by
default; set `PORT` before launching to choose another port. For example, in
PowerShell:

```powershell
$env:PORT = "8090"
.\.venv\Scripts\python.exe app.py
```

## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `GEMINI_API_KEY` | unset | Required for indexing and any model-backed question. |
| `DEMO_DAILY_CAP` | `150` | Question slots per UTC day for this process; failed requests are refunded. |
| `PORT` | `8080` | Port used by the NiceGUI web server. |
| `STORAGE_SECRET` | Random per server start | Optional NiceGUI browser-cookie signing secret. |

Model names, embedding dimensions, chunking thresholds, retrieval count, and
per-browser limits are configured in `src/config.py`. Keep the embedding model
and dimension unchanged for an existing index unless you rebuild it: vectors
from incompatible embedding spaces cannot be compared meaningfully.

## Tests

Run the local test suite from the project root:

```powershell
# Windows
.\.venv\Scripts\python.exe -m pytest
```

```bash
# macOS / Linux (with .venv activated)
python -m pytest
```

Tests use fakes for model and retrieval calls, so they do not consume API quota
or need a populated Chroma index.

## Scope and operational notes

- PDF-to-Markdown conversion was performed upstream (using LlamaCloud); the repository contains
  the Markdown output, not an ingestion service for the original PDF.
- RAGAS evaluation is intentionally not included. The demo instead exposes
  retrieval passages and citations for manual inspection without additional
  LLM-as-judge calls or evaluation dependencies.
- Browser history, per-browser counters, and the daily counter live in process
  memory and reset on restart. This design is suitable for a local demo, not a
  horizontally scaled public service.
- Questions, selected report passages, and report content sent for embedding
  are sent to the configured Gemini API. Use the service in accordance with
  Google's API terms and your organization's data-handling requirements.
