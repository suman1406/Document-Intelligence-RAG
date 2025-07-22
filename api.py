# api.py

import os
import uuid
import tempfile
from fastapi import FastAPI, File, UploadFile, HTTPException, Path
from pydantic import BaseModel
from typing import Dict, Any
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

# --- API Key Configuration ---
# **FIX:** Configure the API key at the very top, before any other imports that might use it.
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("Error: Please set the GEMINI_API_KEY environment variable before starting the server.")
genai.configure(api_key=api_key)
os.environ["GOOGLE_API_KEY"] = api_key

# Now that the key is configured, we can safely import our RAG system.
from rag_system import RAGSystem

# --- App Initialization ---
app = FastAPI(
    title="AI Document Adjudicator API",
    description="An API for analyzing complex documents with a RAG system.",
    version="1.0.0"
)

# --- In-Memory Storage ---
# For a hackathon, a simple dictionary to hold active sessions is perfect.
rag_sessions: Dict[str, RAGSystem] = {}

# --- Pydantic Models for Request & Response ---
class QueryRequest(BaseModel):
    query: str

class UploadResponse(BaseModel):
    document_id: str
    filename: str
    message: str

# --- API Endpoints ---

@app.post("/documents/", response_model=UploadResponse, status_code=201)
async def upload_document(file: UploadFile = File(...)):
    """
    Uploads a document, processes it, and prepares it for querying.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp_file:
        tmp_file.write(await file.read())
        tmp_file_path = tmp_file.name

    try:
        document_id = str(uuid.uuid4())
        
        # Now that the API key is configured, this will work correctly.
        rag_system = RAGSystem()
        rag_system.load_and_process_document(tmp_file_path)
        
        rag_sessions[document_id] = rag_system
        
        return {
            "document_id": document_id,
            "filename": file.filename,
            "message": "Document processed successfully and is ready to be queried."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process document: {str(e)}")
    finally:
        os.unlink(tmp_file_path)

@app.post("/documents/{document_id}/query/")
async def query_document(
    query_request: QueryRequest,
    document_id: str = Path(..., description="The unique ID of the uploaded document.")
):
    """
    Asks a question to a specific, previously uploaded document.
    """
    if document_id not in rag_sessions:
        raise HTTPException(status_code=404, detail="Document not found. Please upload the document first.")
    
    rag_system = rag_sessions[document_id]
    
    try:
        response = rag_system.get_response(query_request.query)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred while processing the query: {str(e)}")

@app.get("/")
def read_root():
    return {"message": "Welcome to the AI Document Adjudicator API. Go to /docs to see the API documentation."}
