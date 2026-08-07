import os
from functools import wraps
from typing import Any, Callable


DEFAULT_LANGSMITH_PROJECT = "rag-pdf-app"


def configure_langsmith() -> None:
    """Normalize LangSmith env vars so tracing works across SDK versions."""
    tracing = os.getenv("LANGSMITH_TRACING")
    if tracing and not os.getenv("LANGSMITH_TRACING_V2"):
        os.environ["LANGSMITH_TRACING_V2"] = tracing

    if not os.getenv("LANGSMITH_PROJECT"):
        os.environ["LANGSMITH_PROJECT"] = DEFAULT_LANGSMITH_PROJECT


configure_langsmith()

try:
    from langsmith import traceable
except ImportError:

    def traceable(*args: Any, **kwargs: Any) -> Callable:
        """No-op fallback so the app can run without LangSmith installed."""
        if args and callable(args[0]) and len(args) == 1 and not kwargs:
            return args[0]

        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*func_args: Any, **func_kwargs: Any) -> Any:
                return func(*func_args, **func_kwargs)

            return wrapper

        return decorator


def hide_self(inputs: dict) -> dict:
    """Keep service/client objects out of LangSmith inputs."""
    return {key: value for key, value in inputs.items() if key != "self"}


def summarize_ingestion_inputs(inputs: dict) -> dict:
    cleaned = {
        key: value
        for key, value in inputs.items()
        if key not in {"db", "embedding_service"}
    }
    if "pdf_path" in cleaned:
        cleaned["pdf_path"] = str(cleaned["pdf_path"])
    return cleaned


def summarize_retrieval_output(results: list[dict]) -> dict:
    """Store useful retrieval details without dumping unnecessary internals."""
    matches = []
    for result in results:
        metadata = result.get("metadata", {})
        document = result.get("document", "")
        matches.append(
            {
                "source": metadata.get("source"),
                "document_id": metadata.get("document_id"),
                "page": metadata.get("page"),
                "chunk_index": metadata.get("chunk_index"),
                "distance": result.get("distance"),
                "text_preview": document[:700],
            }
        )

    return {
        "match_count": len(matches),
        "matches": matches,
    }


def summarize_ingestion_output(chunk_count: int) -> dict:
    return {"chunk_count": chunk_count}
