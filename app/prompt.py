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
You are an AI assistant that answers questions ONLY using the provided context.

Rules:
1. Answer ONLY from the provided context.
2. Do NOT make up information.
3. If the answer is not present in the context, reply:
   "I couldn't find that information in the provided handbook."
4. Cite every factual answer using this format:
   [Source: filename, page: number]
5. Be concise but complete.

==================== CONTEXT ====================

{context}

==================== QUESTION ====================

{question}

==================== ANSWER ====================
"""

    return prompt