from google import genai

from config import GEMINI_API_KEY, GEMINI_MODEL, TOP_K
from embeddings import EmbeddingService
from prompt import build_prompt
from vectordb import VectorDB


class RAGService:
    def __init__(self):
        """Initialize the services shared by every question."""
        if not GEMINI_API_KEY:
            raise RuntimeError("GEMINI_API_KEY is not configured.")

        self.embedding_service = EmbeddingService()
        self.db = VectorDB()
        self.gemini_client = genai.Client(api_key=GEMINI_API_KEY)

    def ask(self, question: str) -> dict:
        """Answer a question using retrieved PDF chunks and return citations."""
        question = question.strip()
        if not question:
            raise ValueError("Question cannot be empty.")

        query_embedding = self.embedding_service.embed_text(question)
        results = self.db.search(query_embedding=query_embedding, top_k=TOP_K)

        if not results:
            return {
                "answer": "I couldn't find that information in the indexed documents.",
                "sources": [],
            }

        prompt = build_prompt(results, question)
        response = self.gemini_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
        )

        sources = []
        for result in results:
            metadata = result["metadata"]
            sources.append(
                {
                    "source": metadata["source"],
                    "document_id": metadata.get("document_id"),
                    "page": metadata["page"],
                    "chunk_index": metadata.get("chunk_index"),
                    "distance": result["distance"],
                }
            )

        return {
            "answer": response.text,
            "sources": sources,
        }
