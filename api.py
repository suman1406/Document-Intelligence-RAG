# api.py

import os
import uuid
import tempfile
from fastapi import FastAPI, File, UploadFile, HTTPException, Path, Response
from pydantic import BaseModel
from typing import Dict, Any, List
import google.generativeai as genai
from tinydb import TinyDB, Query
import fitz  # PyMuPDF
from dotenv import load_dotenv

load_dotenv() 

# --- API Key Configuration ---
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("Error: Please set the GEMINI_API_KEY environment variable before starting the server.")
genai.configure(api_key=api_key)

from rag_system import RAGSystem, PolicySummary

# --- App Initialization & Storage ---
app = FastAPI(
    title="AI Document Adjudicator API",
    description="A production-grade API for analyzing complex documents with a RAG system. Provides document management, proactive summaries, and detailed query analysis.",
    version="2.0.0"
)

db = TinyDB('document_sessions.json')
DocumentQuery = Query()
rag_cache: Dict[str, RAGSystem] = {}
UPLOADS_DIR = "uploaded_documents"
os.makedirs(UPLOADS_DIR, exist_ok=True)

# --- Pydantic Models ---
class QueryRequest(BaseModel):
    query: str

class UploadResponse(BaseModel):
    document_id: str
    filename: str
    message: str
    summary: PolicySummary

class DocumentInfo(BaseModel):
    document_id: str
    filename: str

class DeletionResponse(BaseModel):
    document_id: str
    message: str

# --- API Endpoints ---

@app.post("/documents/", response_model=UploadResponse, status_code=201, tags=["Document Management"], summary="Upload and Process a Document")
async def upload_document(file: UploadFile = File(..., description="The policy document (PDF, DOCX, or EML) to be processed.")):
    """
    Uploads a document, saves it, creates a searchable index, and generates a proactive summary.
    """
    file_id = str(uuid.uuid4())
    file_suffix = os.path.splitext(file.filename)[1]
    persistent_filepath = os.path.join(UPLOADS_DIR, f"{file_id}{file_suffix}")
    
    with open(persistent_filepath, "wb") as f:
        f.write(await file.read())

    try:
        rag_system = RAGSystem()
        rag_system.load_and_process_document(persistent_filepath)
        
        # **UPGRADE:** Generate and include the proactive summary
        summary = rag_system.summarize_document()
        
        rag_cache[file_id] = rag_system
        
        db.insert({
            'document_id': file_id,
            'filepath': persistent_filepath,
            'filename': file.filename
        })
        
        return {
            "document_id": file_id,
            "filename": file.filename,
            "message": "Document processed successfully and is ready to be queried.",
            "summary": summary
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process document: {str(e)}")

@app.get("/documents/", response_model=List[DocumentInfo], tags=["Document Management"], summary="List All Uploaded Documents")
async def list_documents():
    """
    Retrieves a list of all documents that have been uploaded and are available for querying.
    """
    return db.all()

@app.delete("/documents/{document_id}", response_model=DeletionResponse, tags=["Document Management"], summary="Delete a Document")
async def delete_document(document_id: str = Path(..., description="The unique ID of the document to delete.")):
    """
    Deletes a processed document from the system, including its file and index.
    """
    doc_record = db.get(DocumentQuery.document_id == document_id)
    if not doc_record:
        raise HTTPException(status_code=404, detail="Document not found.")
        
    try:
        # Remove from cache, DB, and filesystem
        if document_id in rag_cache:
            del rag_cache[document_id]
        db.remove(DocumentQuery.document_id == document_id)
        os.remove(doc_record['filepath'])
        
        return {"document_id": document_id, "message": "Document and its index have been successfully deleted."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete document: {str(e)}")

@app.post("/documents/{document_id}/query/", tags=["Querying"], summary="Query a Document")
async def query_document(
    query_request: QueryRequest,
    document_id: str = Path(..., description="The unique ID of the uploaded document.")
):
    """
    Asks a question to a specific, previously uploaded document and receives a structured JSON analysis.
    """
    if document_id in rag_cache:
        rag_system = rag_cache[document_id]
    else:
        doc_record = db.get(DocumentQuery.document_id == document_id)
        if not doc_record:
            raise HTTPException(status_code=404, detail="Document not found.")
        
        print(f"Cache miss for {document_id}. Initializing new RAG system.")
        rag_system = RAGSystem()
        rag_system.load_and_process_document(doc_record['filepath'])
        rag_cache[document_id] = rag_system
    
    try:
        response = rag_system.get_response(query_request.query)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred while processing the query: {str(e)}")

@app.get("/documents/{document_id}/page/{page_number}/image", tags=["Source Highlighting"], summary="Get Document Page Image")
async def get_page_image(
    document_id: str = Path(..., description="The unique ID of the document."),
    page_number: int = Path(..., description="The page number to retrieve (1-based index).")
):
    """
    Retrieves a specific page of a PDF document as a PNG image for source highlighting.
    """
    doc_record = db.get(DocumentQuery.document_id == document_id)
    if not doc_record:
        raise HTTPException(status_code=404, detail="Document not found.")
    
    filepath = doc_record['filepath']
    if not filepath.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Page image retrieval is only supported for PDF documents.")

    try:
        doc = fitz.open(filepath)
        if page_number < 1 or page_number > len(doc):
            raise HTTPException(status_code=404, detail="Page number out of range.")
        
        page = doc.load_page(page_number - 1) # fitz is 0-indexed
        pix = page.get_pixmap()
        img_bytes = pix.tobytes("png")
        
        return Response(content=img_bytes, media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to render page image: {str(e)}")

@app.get("/")
def read_root():
    return {"message": "Welcome to the AI Document Adjudicator API. Go to /docs to see the API documentation."}
