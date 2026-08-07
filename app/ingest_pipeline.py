from pathlib import Path

from chunking import chunk_text
from config import PDF_DIR
from embeddings import EmbeddingService
from ingest import extract_pages
from vectordb import VectorDB


def ingest_pdf(
    pdf_path: Path,
    db: VectorDB,
    embedding_service: EmbeddingService,
    document_id: str | None = None,
    source: str | None = None,
) -> int:
    """Extract, embed, and store one PDF. Returns its chunk count."""
    pages = extract_pages(pdf_path)
    if not pages:
        raise ValueError(f"No extractable text found in {pdf_path.name}.")

    document_id = document_id or pdf_path.stem
    source = source or pdf_path.name
    total_chunks = 0

    for page in pages:
        chunks = chunk_text(page["text"])
        if not chunks:
            continue

        embeddings = embedding_service.embed_documents(chunks)
        db.add_documents(
            chunks=chunks,
            embeddings=embeddings,
            source=source,
            document_id=document_id,
            page=page["page_number"],
            stored_filename=pdf_path.name,
        )
        total_chunks += len(chunks)

    return total_chunks


def ingest_directory() -> int:
    """Rebuild the collection from every PDF currently in PDF_DIR."""
    pdf_paths = sorted(PDF_DIR.glob("*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError(f"No PDF files found in: {PDF_DIR}")

    db = VectorDB()
    embedding_service = EmbeddingService()
    db.reset_collection()

    total_chunks = 0
    for pdf_path in pdf_paths:
        chunk_count = ingest_pdf(pdf_path, db, embedding_service)
        total_chunks += chunk_count
        print(f"Indexed {pdf_path.name}: {chunk_count} chunks.")

    print(f"DONE: Indexed {len(pdf_paths)} PDFs and {total_chunks} chunks.")
    return total_chunks


if __name__ == "__main__":
    ingest_directory()
