from google import genai

from config import GEMINI_API_KEY, GEMINI_MODEL, TOP_K
from embeddings import EmbeddingService
from prompt import build_prompt
from tracing import hide_self, summarize_retrieval_output, traceable
from vectordb import VectorDB


class RAGService:
    def __init__(self):
        """Initialize the services shared by every question."""
        if not GEMINI_API_KEY:
            raise RuntimeError("GEMINI_API_KEY is not configured.")

        self.embedding_service = EmbeddingService()
        self.db = VectorDB()
        self.gemini_client = genai.Client(api_key=GEMINI_API_KEY)

    @traceable(
        name="RAG Question Answering",
        run_type="chain",
        process_inputs=hide_self,
    )
    def ask(self, question: str) -> dict:
        """Answer a question using retrieved PDF chunks and return citations."""
        question = question.strip()
        if not question:
            raise ValueError("Question cannot be empty.")

        results = self._retrieve_chunks(question)

        if not results:
            return {
                "answer": "I couldn't find that information in the indexed documents.",
                "sources": [],
            }

        prompt = self._build_grounded_prompt(results, question)
        answer = self._generate_answer(prompt)

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
            "answer": answer,
            "sources": sources,
        }

    @traceable(
        name="Retrieve PDF Chunks",
        run_type="retriever",
        process_inputs=hide_self,
        process_outputs=summarize_retrieval_output,
    )
    def _retrieve_chunks(self, question: str) -> list[dict]:
        query_embedding = self.embedding_service.embed_text(question)
        return self.db.search(query_embedding=query_embedding, top_k=TOP_K)

    @traceable(
        name="Build Grounded Prompt",
        run_type="prompt",
        process_inputs=hide_self,
    )
    def _build_grounded_prompt(self, results: list[dict], question: str) -> str:
        return build_prompt(results, question)

    @traceable(
        name="Generate Gemini Answer",
        run_type="llm",
        process_inputs=hide_self,
    )
    def _generate_answer(self, prompt: str) -> str:
        response = self.gemini_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
        )
        return response.text
