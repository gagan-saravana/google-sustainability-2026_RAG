# Google 2026 Sustainability Report Q&A

A browser-based question-answering demo for Google's 2026 sustainability
report. It retrieves relevant report passages and uses Gemini to produce
grounded answers with citations to those passages.

## Development flow

The report is prepared and queried in three stages:

1. Parse the source: LlamaParse converts Google's 2026 sustainability
   report PDF into Markdown. This upstream conversion has already been done;
   the Markdown file is in `data/google_2026_sustainability.md`.
2. Build the retrieval index: The pipeline splits the Markdown into
   structure-aware chunks, preserving section headings as breadcrumb context.
   Gemini creates an embedding for each chunk, and ChromaDB stores the vectors
   and chunk metadata locally.
3. Answer questions: The app embeds a question with Gemini, retrieves
   similar chunks from ChromaDB, then asks Gemini to synthesize an answer
   grounded in those passages and cite its sources.

```text
Google sustainability report (PDF)
                 |
                 v
       LlamaParse (upstream)
                 |
                 v
   data/google_2026_sustainability.md
                 |
       header-aware chunking
       + section breadcrumbs
                 |
                 v
       Gemini embeddings
                 |
                 v
        ChromaDB index
                 |
       user question -------> Gemini question embedding
                                      |
                                      v
                            similar report chunks
                                      |
                                      v
                              Gemini synthesis
                                      |
                                      v
                         cited answer in browser
```

## Directory structure

```text
.
├── app.py                       # NiceGUI browser chat
├── requirements.txt             # Python dependencies
├── .env.example                 # Gemini API key template
├── data/
│   ├── google_2026_sustainability.md  # LlamaParse output
│   └── chunks.jsonl                  # Optional chunk inspection output
├── src/
│   ├── chunking.py               # Markdown cleanup and structure-aware chunks
│   ├── config.py                 # Models, paths, and application settings
│   ├── embeddings.py             # Gemini document and query embeddings
│   ├── gemini_client.py          # Shared Gemini API client
│   ├── index.py                  # Create/update the ChromaDB index
│   ├── retrieve.py               # Similarity search against ChromaDB
│   ├── synthesize.py             # Grounded answer generation and citations
│   ├── condense.py               # Follow-up question rewriting
│   └── limits.py                 # In-memory demo usage limits
└── chroma_db/                    # Generated local vector index
```

## Tech stack

| Technology | Role |
| --- | --- |
| Python | Application and indexing pipeline |
| LlamaParse | Upstream PDF-to-Markdown parsing |
| Google Gemini API (`google-genai`) | Chunk and query embeddings; answer generation |
| ChromaDB | Local persistent vector store |
| NiceGUI | Browser-based chat interface |
| `python-dotenv` | Loads the Gemini API key from `.env` |
| Tenacity | Retries selected Gemini API errors |

## Set up and run

### Prerequisites

- Python 3.10 or later
- A Gemini API key from [Google AI Studio](https://aistudio.google.com/apikey)
- Network access to the Gemini API

### 1. Install dependencies

From the project root, run these commands in PowerShell:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Open `.env` and replace the placeholder with your Gemini API key:

```dotenv
GEMINI_API_KEY=your-actual-key
```

Keep `.env` private; do not commit or share your API key.

### 2. Build the local index

From the project root, run:

```powershell
.\.venv\Scripts\python.exe -m src.index
```

Indexing chunks the Markdown, embeds the chunks with Gemini, and stores the
index in `chroma_db/`. It can be rerun to add chunks that are not already
stored. Indexing requires Gemini API access and may use API quota.

### 3. Start the web app

Run:

```powershell
.\.venv\Scripts\python.exe app.py
```

Open [http://localhost:8080](http://localhost:8080) in your browser and ask a
question about the report. The app reads the existing local index; build it
before starting the app.
