from contextlib import asynccontextmanager
import mimetypes
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, Request, UploadFile, status
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from config import BASE_DIR, PDF_DIR
from ingest_pipeline import ingest_pdf
from rag import RAGService

mimetypes.add_type("text/javascript", ".jsx")


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4_000)


@asynccontextmanager
async def lifespan(app: FastAPI):
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    app.state.rag_service = RAGService()
    yield


app = FastAPI(title="RAG PDF API", version="1.0.0", lifespan=lifespan)
FRONTEND_DIR = BASE_DIR / "frontend"


def get_rag_service(request: Request) -> RAGService:
    return request.app.state.rag_service


@app.get("/app.jsx", include_in_schema=False)
async def frontend_module():
    return FileResponse(FRONTEND_DIR / "app.jsx", media_type="text/javascript")


@app.post("/chat")
async def chat(request_body: ChatRequest, request: Request):
    rag_service = get_rag_service(request)
    try:
        return await run_in_threadpool(rag_service.ask, request_body.question)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=502, detail="Unable to generate an answer.") from error


@app.post("/upload", status_code=status.HTTP_201_CREATED)
async def upload_pdf(request: Request, file: UploadFile = File(...)):
    original_filename = Path(file.filename or "").name
    if not original_filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    pdf_id = uuid4().hex
    destination = PDF_DIR / original_filename

    if destination.exists():
        raise HTTPException(
            status_code=409,
            detail="A PDF with this filename already exists. Rename it before uploading.",
        )

    try:
        with destination.open("wb") as output_file:
            while content := await file.read(1024 * 1024):
                output_file.write(content)

        if destination.stat().st_size == 0:
            raise HTTPException(status_code=400, detail="Uploaded PDF is empty.")

        rag_service = get_rag_service(request)
        chunk_count = await run_in_threadpool(
            ingest_pdf,
            destination,
            rag_service.db,
            rag_service.embedding_service,
            pdf_id,
            original_filename,
        )
    except HTTPException:
        destination.unlink(missing_ok=True)
        raise
    except ValueError as error:
        destination.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail=str(error)) from error
    except Exception as error:
        rag_service = get_rag_service(request)
        await run_in_threadpool(rag_service.db.delete_document, pdf_id)
        destination.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail="Unable to index the PDF.") from error
    finally:
        await file.close()

    return {
        "pdf_id": pdf_id,
        "source": original_filename,
        "chunk_count": chunk_count,
    }


@app.get("/documents")
async def list_documents(request: Request):
    rag_service = get_rag_service(request)
    documents = await run_in_threadpool(rag_service.db.list_documents)
    return {"documents": documents}


@app.delete("/delete/{pdf_id}")
async def delete_pdf(pdf_id: str, request: Request):
    rag_service = get_rag_service(request)
    metadata = await run_in_threadpool(rag_service.db.delete_document, pdf_id)

    if metadata is None:
        raise HTTPException(status_code=404, detail="Document not found.")

    stored_filename = metadata.get("stored_filename", metadata["source"])
    file_path = PDF_DIR / Path(stored_filename).name
    file_path.unlink(missing_ok=True)

    return {"pdf_id": pdf_id, "deleted": True}


app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
