"""Fail-closed evaluation helpers for the RAG answer loop."""

import json
import re
from typing import Any


ABSTENTION_ANSWER = "I couldn't verify an answer from the uploaded documents."


def parse_json_response(text: str | None) -> dict[str, Any]:
    """Parse a model JSON response, including an occasional Markdown fence."""
    if not text:
        raise ValueError("Evaluator returned an empty response.")

    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned).strip()
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start < 0 or end < start:
        raise ValueError("Evaluator did not return a JSON object.")
    payload = json.loads(cleaned[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("Evaluator response is not a JSON object.")
    return payload


def clamp_score(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def candidate_context(results: list[dict]) -> str:
    sections = []
    for index, result in enumerate(results, start=1):
        metadata = result["metadata"]
        sections.append(
            f"CHUNK {index}\n"
            f"Source: {metadata['source']}\n"
            f"Page: {metadata['page']}\n"
            f"Text: {result['document']}"
        )
    return "\n\n---\n\n".join(sections)


def build_retrieval_grader_prompt(question: str, results: list[dict]) -> str:
    return f"""You are a strict retrieval evaluator for a document RAG system.
Decide whether the candidate chunks explicitly contain enough evidence to answer
the user's question. Do not treat general knowledge or plausible inference as
evidence. Select only chunks directly useful to answering the question.

Return JSON only in this exact shape:
{{"answerable": true, "relevant_chunk_numbers": [1], "relevance_score": 0.0,
  "reason": "brief explanation"}}

Set answerable to false when the evidence is missing, ambiguous, or only partly
answers a question that needs a complete answer. Scores are 0 to 1.

QUESTION: {question}

CANDIDATE CHUNKS:
{candidate_context(results)}"""


def build_answer_grader_prompt(question: str, answer: str, results: list[dict]) -> str:
    return f"""You are a strict answer verifier for a document RAG system.
Judge the answer solely against the supplied chunks. Every factual claim must
be explicitly supported by them. An answer is not grounded if it adds outside
knowledge, guesses, or makes unsupported implications. Check that citations
refer to the shown source/page information.

Return JSON only in this exact shape:
{{"groundedness_score": 0.0, "answer_relevance_score": 0.0,
  "completeness_score": 0.0, "citations_valid": true,
  "should_abstain": false, "unsupported_claims": [], "reason": "brief explanation"}}

Use should_abstain=true if the answer should not be shown as a verified answer.
Scores are 0 to 1.

QUESTION: {question}

ANSWER TO VERIFY:
{answer}

SUPPORTING CHUNKS:
{candidate_context(results)}"""


def build_query_rewrite_prompt(question: str, failure_reason: str) -> str:
    return f"""You improve a search query for a PDF document retrieval system.
Return JSON only: {{"rewritten_query": "..."}}.
Keep the user's intent, add precise terms or split compound concepts if useful,
and do not add facts not present in the question.

ORIGINAL QUESTION: {question}
RETRIEVAL/EVALUATION ISSUE: {failure_reason}"""
