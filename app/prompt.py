def build_prompt(results: list[dict], question: str) -> str:
    """
    Builds the final prompt for the LLM using retrieved chunks.
    """

    context = ""

    for i, result in enumerate(results, start=1):
        source = result["metadata"]["source"]
        page = result["metadata"]["page"]
        document = result["document"]

        context += (
            f"Result {i}\n"
            f"Source: {source}\n"
            f"Page: {page}\n"
            f"{document}\n\n"
            f"{'-'*80}\n\n"
        )

    prompt = f"""
You are a document question-answering assistant. Answer ONLY using the provided context.

Rules:
1. Answer ONLY from the provided context.
2. Do NOT make up information.
3. Do not use outside knowledge, assumptions, or general advice. Do not infer a
   fact that is not explicitly supported by the context.
4. If the answer is absent or incomplete, reply exactly:
   "I couldn't verify an answer from the uploaded documents."
5. Cite every factual answer using this format:
   [Source: filename, page: number]
6. Be concise but complete.

==================== CONTEXT ====================

{context}

==================== QUESTION ====================

{question}

==================== ANSWER ====================
"""

    return prompt
