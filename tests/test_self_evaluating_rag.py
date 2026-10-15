import sys
import unittest
import os
from pathlib import Path


os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGSMITH_TRACING_V2"] = "false"
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from evaluation import ABSTENTION_ANSWER, candidate_context, parse_json_response
from rag import RAGService


class SelfEvaluatingRAGTests(unittest.TestCase):
    def test_context_keeps_readable_newlines_and_json_fences_parse(self):
        context = candidate_context(
            [{"metadata": {"source": "policy.pdf", "page": 2}, "document": "Leave is 12 days."}]
        )
        self.assertIn("CHUNK 1\nSource: policy.pdf\nPage: 2", context)
        self.assertEqual(parse_json_response("```json\n{\"answerable\": true}\n```"), {"answerable": True})

    def test_unverified_retrieval_abstains_after_one_repair(self):
        service = RAGService.__new__(RAGService)
        service._retrieve_chunks = lambda query: []
        service._rewrite_query = lambda question, reason: question

        result = service.ask("What is the company share price?")

        self.assertEqual(result["answer"], ABSTENTION_ANSWER)
        self.assertFalse(result["evaluation"]["verified"])
        self.assertEqual(result["evaluation"]["attempts"], 2)

    def test_only_a_verified_answer_is_returned_with_sources(self):
        service = RAGService.__new__(RAGService)
        chunk = {
            "document": "Employees receive 12 annual leave days.",
            "metadata": {"source": "policy.pdf", "page": 3, "document_id": "doc-1", "chunk_index": 0},
            "distance": 0.1,
        }
        service._retrieve_chunks = lambda query: [chunk]
        service._select_relevant_context = lambda question, results: (
            results,
            {"answerable": True, "relevance_score": 1.0, "reason": "Direct evidence."},
        )
        service._build_grounded_prompt = lambda results, question: "prompt"
        service._generate_answer = lambda prompt: "Employees receive 12 annual leave days. [Source: policy.pdf, page: 3]"
        service._evaluate_answer = lambda question, answer, results: {
            "groundedness_score": 1.0,
            "answer_relevance_score": 1.0,
            "completeness_score": 1.0,
            "citations_valid": True,
            "should_abstain": False,
            "reason": "Fully supported.",
        }

        result = service.ask("How many annual leave days are available?")

        self.assertTrue(result["evaluation"]["verified"])
        self.assertEqual(result["evaluation"]["confidence"], 1.0)
        self.assertEqual(result["sources"][0]["page"], 3)


if __name__ == "__main__":
    unittest.main()
