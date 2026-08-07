# Document Desk

A full-stack Retrieval-Augmented Generation (RAG) application that lets users
upload PDFs, ask questions about them, and receive grounded answers with
document and page-level citations.

## Features

- Upload and index multiple PDFs
- Page-aware text extraction with PyMuPDF
- Semantic search using Sentence Transformers and ChromaDB
- Gemini-generated answers grounded in retrieved document context
- Source, page, and chunk-level metadata
- Document listing and deletion through FastAPI
- React web interface for upload, chat, and document management

## Architecture

PDF Upload
-> PyMuPDF page extraction
-> Recursive text chunking
-> Sentence Transformer embeddings
-> ChromaDB vector storage
-> Similarity search
-> Gemini response with citations

## Tech Stack

- Python, FastAPI, Uvicorn
- PyMuPDF
- Sentence Transformers
- ChromaDB
- Google Gemini API
- React
- HTML/CSS

## Setup

1. Clone the repository.
2. Create and activate a virtual environment.
3. Install dependencies:

```bash
pip install -r requirements.txt
```

## API Endpoints

| Method | Endpoint           | Purpose                           |
|--------|--------------------|-----------------------------------|
| POST   | `/chat`            | Ask a question about indexed PDFs |
| POST   | `/upload`          | Upload and index a PDF            |
| GET    | `/documents`       | List indexed documents            |
| DELETE | `/delete/{pdf_id}` | Delete a document and its vectors |


## Screenshots

### Home Page

![Home](screenshots/home.jpeg)

### Chat Interface

![Chat](screenshots/answer.jpeg)
![Chat](screenshots/answer(2).jpeg)

### Retrieved Context

![search](screenshots/searching.jpeg)
