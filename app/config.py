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

TOP_K = 3

# ==========================
# Gemini
# ==========================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

GEMINI_MODEL = "gemini-2.5-flash"