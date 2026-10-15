from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent

PDF_DIR = BASE_DIR / "data" / "pdf"

CHROMA_DB_PATH = BASE_DIR / "chromaDB"

COLLECTION_NAME = "employee_handbook"

# Embedding Model
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# ==========================
# Chunking
# ==========================

CHUNK_SIZE = 500

CHUNK_OVERLAP = 100

# ==========================
# Retrieval
# ==========================

# Retrieve a wider pool, then let the evaluator select the few chunks that are
# actually relevant. This is deliberately separate from ANSWER_CONTEXT_K.
TOP_K = 8
ANSWER_CONTEXT_K = 4
MAX_REPAIR_ATTEMPTS = 1

# A response must meet all of these evaluator scores to be shown as verified.
MIN_GROUNDEDNESS_SCORE = 0.85
MIN_ANSWER_RELEVANCE_SCORE = 0.75
MIN_COMPLETENESS_SCORE = 0.70

# ==========================
# Gemini
# ==========================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

GEMINI_MODEL = "gemini-2.5-flash"
