# config file for keeping config parameters in one place

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # reads GEMINI_API_KEY from a local .env file (see .env.example)

# ----- paths -----------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "google_2026_sustainability.md"   # LlamaCloud output
CHUNKS_DUMP = ROOT / "data" / "chunks.jsonl"                  # human-readable dump for inspection
CHROMA_DIR = ROOT / "chroma_db"                               # persistent vector store
COLLECTION_NAME = "google_2026_sustainability"

# ----- embeddings (Gemini API) -----------------------------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
EMBEDDING_MODEL = "gemini-embedding-2"   # text-only alternative: "gemini-embedding-001"
# Gemini embeddings are trained with Matryoshka Representation Learning, so
# they can be truncated with little quality loss. 768 dims = 4x less storage
# than the 3072 default. Google lists 768 / 1536 / 3072 as recommended sizes.
EMBEDDING_DIM = 768
EMBED_BATCH_SIZE = 32          # chunks per API call (page doesn't state a limit; start modest)
PAUSE_BETWEEN_BATCHES_S = 1.0  # gentle pacing for free-tier rate limits

# ----- chunking (all sizes in characters; ~4 chars is roughly 1 token) ---------
TARGET_CHARS = 1000      # pack paragraphs of one section up to about this size
MAX_CHARS = 1400         # a single paragraph longer than this is split by sentence
OVERLAP_CHARS = 150      # sentence overlap, used ONLY when splitting an oversized paragraph
MIN_CHARS = 40           # prose chunks shorter than this carry no useful signal
TABLE_MAX_CHARS = 2500   # larger tables are split by rows (header row repeated)

# ----- retrieval + answer synthesis (step 3) -------------------------------------
TOP_K = 6                                  # passages handed to the LLM per question
GENERATION_MODEL = "gemini-3.5-flash-lite" # free-tier model per the pricing page; any other free Flash model works
TEMPERATURE = 0.2                          # low: answers should stick to the context

# ----- chat UI + demo limits (step 5) ----------------------------------------------
MAX_PROMPTS_PER_BROWSER = 3     # questions each browser may ask
# Global safety net for the free Gemini quota. A question costs up to 3 API calls
# (condense follow-up, embed query, generate answer). Tune via DEMO_DAILY_CAP.
DAILY_CAP = int(os.getenv("DEMO_DAILY_CAP", "150"))
MAX_QUESTION_CHARS = 300
MAX_TRACKED_BROWSERS = 5000     # in-memory sessions kept before the oldest is forgotten
STORAGE_SECRET = os.getenv("STORAGE_SECRET")   # signs the browser cookie; random per run if unset
