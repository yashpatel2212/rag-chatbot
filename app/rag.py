from google import genai

from config import ANSWER_CONTEXT_K, GEMINI_API_KEY, GEMINI_MODEL, MAX_REPAIR_ATTEMPTS, MIN_ANSWER_RELEVANCE_SCORE, MIN_COMPLETENESS_SCORE, MIN_GROUNDEDNESS_SCORE, TOP_K
from embeddings import EmbeddingService
from evaluation import ABSTENTION_ANSWER, build_answer_grader_prompt, build_query_rewrite_prompt, build_retrieval_grader_prompt, clamp_score, parse_json_response
from prompt import build_prompt
from tracing import hide_self, summarize_retrieval_output, traceable
from vectordb import VectorDB


class RAGService:
    def __init__(self):
        if not GEMINI_API_KEY:
            raise RuntimeError("GEMINI_API_KEY is not configured.")
        self.embedding_service = EmbeddingService()
        self.db = VectorDB()
        self.gemini_client = genai.Client(api_key=GEMINI_API_KEY)

    @traceable(name="RAG Question Answering", run_type="chain", process_inputs=hide_self)
    def ask(self, question: str) -> dict:
        """Answer only when retrieval and answer verification both pass."""
        question = question.strip()
        if not question:
            raise ValueError("Question cannot be empty.")

        attempts, search_query = 0, question
        last_reason = "No supporting content was found in the uploaded documents."
        while attempts <= MAX_REPAIR_ATTEMPTS:
            results = self._retrieve_chunks(search_query)
            selected, retrieval_evaluation = self._select_relevant_context(question, results)
            last_reason = retrieval_evaluation["reason"]

            if selected:
                answer = self._generate_answer(self._build_grounded_prompt(selected, question))
                answer_evaluation = self._evaluate_answer(question, answer, selected)
                if self._answer_is_verified(answer_evaluation):
                    return {
                        "answer": answer,
                        "sources": self._sources_from_results(selected),
                        "evaluation": {
                            "verified": True,
                            "confidence": self._confidence(answer_evaluation),
                            "reason": answer_evaluation["reason"],
                            "attempts": attempts + 1,
                            "retrieval": retrieval_evaluation,
                        },
                    }
                last_reason = answer_evaluation["reason"]

            if attempts == MAX_REPAIR_ATTEMPTS:
                break
            attempts += 1
            search_query = self._rewrite_query(question, last_reason)

        return self._abstention(last_reason, attempts + 1)

    @traceable(name="Retrieve PDF Chunks", run_type="retriever", process_inputs=hide_self, process_outputs=summarize_retrieval_output)
    def _retrieve_chunks(self, question: str) -> list[dict]:
        return self.db.search(query_embedding=self.embedding_service.embed_text(question), top_k=TOP_K)

    @traceable(name="Grade Retrieved Context", run_type="chain", process_inputs=hide_self)
    def _select_relevant_context(self, question: str, results: list[dict]) -> tuple[list[dict], dict]:
        if not results:
            return [], {"answerable": False, "relevance_score": 0.0, "reason": "No indexed chunks matched the question."}
        try:
            grade = self._generate_json(build_retrieval_grader_prompt(question, results))
            indexes = grade.get("relevant_chunk_numbers", [])
            indexes = [i for i in indexes if isinstance(i, int) and 1 <= i <= len(results)]
            selected = [results[i - 1] for i in dict.fromkeys(indexes)][:ANSWER_CONTEXT_K]
            evaluation = {
                "answerable": grade.get("answerable") is True and bool(selected),
                "relevance_score": clamp_score(grade.get("relevance_score")),
                "reason": str(grade.get("reason") or "Retrieved context could not be verified."),
            }
            return (selected if evaluation["answerable"] else []), evaluation
        except Exception:
            # Fail closed: an evaluation failure never permits an answer.
            return [], {"answerable": False, "relevance_score": 0.0, "reason": "Could not verify retrieved context."}

    @traceable(name="Grade Generated Answer", run_type="chain", process_inputs=hide_self)
    def _evaluate_answer(self, question: str, answer: str, results: list[dict]) -> dict:
        try:
            grade = self._generate_json(build_answer_grader_prompt(question, answer, results))
            return {
                "groundedness_score": clamp_score(grade.get("groundedness_score")),
                "answer_relevance_score": clamp_score(grade.get("answer_relevance_score")),
                "completeness_score": clamp_score(grade.get("completeness_score")),
                "citations_valid": grade.get("citations_valid") is True,
                "should_abstain": grade.get("should_abstain") is True,
                "reason": str(grade.get("reason") or "Answer could not be verified."),
            }
        except Exception:
            return {"groundedness_score": 0.0, "answer_relevance_score": 0.0, "completeness_score": 0.0, "citations_valid": False, "should_abstain": True, "reason": "Could not verify the generated answer."}

    def _answer_is_verified(self, evaluation: dict) -> bool:
        return (not evaluation["should_abstain"] and evaluation["citations_valid"] and evaluation["groundedness_score"] >= MIN_GROUNDEDNESS_SCORE and evaluation["answer_relevance_score"] >= MIN_ANSWER_RELEVANCE_SCORE and evaluation["completeness_score"] >= MIN_COMPLETENESS_SCORE)

    def _confidence(self, evaluation: dict) -> float:
        return round((evaluation["groundedness_score"] + evaluation["answer_relevance_score"] + evaluation["completeness_score"]) / 3, 2)

    @traceable(name="Rewrite Query for RAG Repair", run_type="chain", process_inputs=hide_self)
    def _rewrite_query(self, question: str, failure_reason: str) -> str:
        try:
            rewritten = self._generate_json(build_query_rewrite_prompt(question, failure_reason)).get("rewritten_query", "")
            return rewritten.strip() if isinstance(rewritten, str) and rewritten.strip() else question
        except Exception:
            return question

    def _abstention(self, reason: str, attempts: int) -> dict:
        return {"answer": ABSTENTION_ANSWER, "sources": [], "evaluation": {"verified": False, "confidence": 0.0, "reason": reason, "attempts": attempts}}

    def _sources_from_results(self, results: list[dict]) -> list[dict]:
        return [{"source": item["metadata"]["source"], "document_id": item["metadata"].get("document_id"), "page": item["metadata"]["page"], "chunk_index": item["metadata"].get("chunk_index"), "distance": item["distance"]} for item in results]

    @traceable(name="Build Grounded Prompt", run_type="prompt", process_inputs=hide_self)
    def _build_grounded_prompt(self, results: list[dict], question: str) -> str:
        return build_prompt(results, question)

    @traceable(name="Generate Gemini Answer", run_type="llm", process_inputs=hide_self)
    def _generate_answer(self, prompt: str) -> str:
        return self.gemini_client.models.generate_content(model=GEMINI_MODEL, contents=prompt).text

    def _generate_json(self, prompt: str) -> dict:
        response = self.gemini_client.models.generate_content(model=GEMINI_MODEL, contents=prompt, config={"response_mime_type": "application/json"})
        return parse_json_response(response.text)
