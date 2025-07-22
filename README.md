AI Document Adjudicator Engine
An advanced Retrieval-Augmented Generation (RAG) system designed to analyze complex legal and policy documents with high accuracy and speed. This project provides a robust API for developers to integrate intelligent document analysis into their applications.

Key Features
High-Speed Indexing: Utilizes FAISS (Facebook AI Similarity Search) for blazingly fast in-memory vector indexing and retrieval.

State-of-the-Art Accuracy: Implements a retrieve-and-rerank architecture with a Cross-Encoder model to ensure the most semantically relevant context is used for generating answers.

Intelligent & Transparent Reasoning: Powered by Gemini 1.5 Pro, the system can handle vague queries, make and state clear assumptions, and provide detailed, auditable justifications for its decisions.

Structured JSON Output: Returns a predictable, multi-layered JSON response that is ideal for downstream automation and integration into other systems.

Robust API: Built with FastAPI, providing an interactive and easy-to-use interface for uploading documents and submitting queries.

Workflow
The system follows a sophisticated RAG pipeline designed to maximize both speed and accuracy.

flowchart TD
    subgraph "User Interaction & Document Indexing"
        A["📤<br><b>1. Upload Document</b><br>(.pdf, .docx, .eml)"] --> B["✂<br><b>2. Document Chunking</b><br>(Semantic Splitting)"]
        B --> C["🤖<br><b>3. Text Embedding</b><br>(Gemini Model)"]
        C --> D["🗄<br><b>4. Vector Storage</b><br>(Index in FAISS)"]
    end

    subgraph "Live Query Processing"
        E["⌨<br><b>5. User Query Input</b>"] --> F["🤖<br><b>6. Query Embedding</b>"]
        D -- Provides Stored Context --> G["🔍<br><b>7. Retrieve & Re-Rank</b><br>(FAISS Search + Cross-Encoder)"]
        F -- Provides Query Vector --> G
        G --> H["📝<br><b>8. Prompt Construction</b><br>(Combine Context + Query)"]
        H --> I["🧠<br><b>9. LLM Synthesis</b><br>(Gemini 1.5 Pro)"]
        I --> J["📨<br><b>10. API Response</b><br>(Structured JSON)"]
    end

    classDef indexing fill:#e6f3ff,stroke:#007bff,stroke-width:2px,color:#000
    classDef inference fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px,color:#000
    classDef io fill:#fff0e6,stroke:#fd7e14,stroke-width:2px,color:#000
    classDef merge fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px,color:#000

    class A,E io
    class B,C,D indexing
    class F,H,I inference
    class G merge
    class J io

Setup and Installation
1. Prerequisites
Python 3.8 or higher

A Google Gemini API Key

2. Installation
Clone the repository:

git clone <your-repo-url>
cd <your-repo-name>



Set up your API Key:
You need to set your Gemini API key as an environment variable.

Windows (Command Prompt):

set GEMINI_API_KEY="YOUR_API_KEY_HERE"



Windows (PowerShell):

$env:GEMINI_API_KEY="YOUR_API_KEY_HERE"



macOS / Linux:

export GEMINI_API_KEY="YOUR_API_KEY_HERE"



Install dependencies:

pip install -r requirements.txt



How to Run the API
Start the API Server:
Run the following command in your terminal from the project's root directory:

uvicorn api:app --reload



The server will start, typically on http://127.0.0.1:8000.

Access the API Documentation:
Open your web browser and navigate to http://127.0.0.1:8000/docs. You will see the interactive FastAPI documentation (Swagger UI), which you can use to test the endpoints directly.

API Workflow Example (using curl)
Upload a Document:
(Replace path/to/your/policy.pdf with the actual file path)

curl -X POST -F "file=@path/to/your/policy.pdf" http://127.0.0.1:8000/documents/



This will return a JSON response with a unique document_id.

Query the Document:
(Replace {document_id} with the ID from the previous step)

curl -X POST -H "Content-Type: application/json" \
-d '{"query": "46M, knee surgery, 3-month policy"}' \
http://127.0.0.1:8000/documents/{document_id}/query/



This will return the final, structured JSON analysis.