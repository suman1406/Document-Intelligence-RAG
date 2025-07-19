# rag_backend.py

import os
from typing import List, Optional

# LangChain Imports
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain.docstore.document import Document
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredEmailLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_google_genai import ChatGoogleGenerativeAI

# New import for Semantic Chunking
from langchain_experimental.text_splitter import SemanticChunker

# Pydantic for Structured Output
from pydantic import BaseModel, Field

# --- Define Pydantic models for structured output ---
class GeneratedQueries(BaseModel):
    """Pydantic model for the structured output of the query generation step."""
    sub_queries: List[str] = Field(description="A list of decomposed, simpler questions.")
    hypothetical_answer: str = Field(description="A detailed, hypothetical answer to the original query.")
    keywords: List[str] = Field(description="A list of important keywords and synonyms from the query.")

class Decision(BaseModel):
    """Pydantic model for the final decision output."""
    summary: str = Field(description="A concise, one-to-two-line summary of the final decision and its primary reason.")
    decision: str = Field(description="The final decision: 'Approved', 'Denied', or 'Further Information Required'.")
    amount: Optional[float] = Field(description="The approved monetary amount, if applicable. Null otherwise.")
    justification: str = Field(description="A detailed explanation for the decision, citing specific clauses.")
    clause_references: List[str] = Field(description="List of source clauses supporting the decision.")

# --- Initialize Models (globally, as they are heavyweight) ---
print("✨ Initializing models...")
try:
    # Corrected to a valid, powerful model name
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0)
    print("Gemini LLM initialized successfully.")
except Exception as e:
    raise RuntimeError(f"Could not initialize Gemini LLM: {e}")

embedding_model = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'}
)
print("Embedding model (all-MiniLM-L6-v2) loaded.")

cross_encoder_model = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
print("Cross-encoder model for re-ranking loaded.")


# --- Define Parsers and Prompts ---
query_parser = JsonOutputParser(pydantic_object=GeneratedQueries)
decision_parser = JsonOutputParser(pydantic_object=Decision)

query_generation_prompt = PromptTemplate(
    template="You are an expert query analyst. Analyze the user's question and generate a set of more effective queries to improve document retrieval.\n1. **Sub-Queries:** Decompose the query into smaller questions.\n2. **Hypothetical Answer (HyDE):** Create an ideal answer.\n3. **Keywords:** Extract key terms and synonyms.\n\n**User Query:** \"{query}\"\n\n{format_instructions}",
    input_variables=["query"],
    partial_variables={"format_instructions": query_parser.get_format_instructions()},
)

# **REASONING UPGRADE 5.0: Prompt to handle specific vs. general definitions**
synthesis_prompt = PromptTemplate(
    template=(
        "You are a senior insurance claims adjudicator. Your task is to provide the most helpful, definitive answer possible based on the provided policy context and a user's query. "
        "Apply general rules and exceptions meticulously. If you find a specific definition (e.g., 'Hospital (for International Cover)'), be aware that a general 'Standard Definition' for the same term might also exist in the context and you should present both. "
        "If multiple definitions exist, present both clearly. "
        "**Crucially, if a user's query is vague (e.g., 'is therapy covered?'), do not simply ask for more information.** Your primary goal is to avoid a 'Further Information Required' decision. "
        "Instead, do the following: "
        "1. **Assume the Best-Case Scenario:** Base your answer on the most comprehensive coverage available in the context (e.g., assume they have the 'Imperial Plus Plan' with international benefits if mentioned). "
        "2. **Provide a Concrete Decision:** Give a clear 'Approved' or 'Denied' decision based on this assumption. "
        "3. **State Your Assumption Clearly:** In the justification, explicitly state the assumptions you made (e.g., 'Assuming you have the Imperial Plus Plan and the treatment is post-hospitalization...'). "
        "4. **Explain Alternatives:** Briefly mention how the decision might change under different circumstances (e.g., '...this would not be covered under a domestic-only plan.'). "
        "Respond *only* in a valid JSON format.\n\n"
        "**Policy Context:**\n---\n{context}\n---\n\n"
        "**User Query:** \"{query}\"\n\n"
        "{format_instructions}"
    ),
    input_variables=["context", "query"],
    partial_variables={"format_instructions": decision_parser.get_format_instructions()},
)


# Grounding Check Prompt
grounding_prompt = PromptTemplate(
    template="You are a meticulous fact-checker. Based *only* on the provided 'Policy Context', determine if the 'Justification' for the decision is fully supported. Answer with a single word: 'Yes' or 'No'.\n\n**Policy Context:**\n---\n{context}\n---\n\n**Justification to Verify:**\n\"{justification}\"\n\n**Is the justification fully supported by the context? (Yes/No):**",
    input_variables=["context", "justification"]
)

# --- Core RAG Logic Functions ---
def load_single_document(file_path: str) -> List[Document]:
    """Detects the file type and uses the appropriate loader."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif ext == ".docx":
        loader = Docx2txtLoader(file_path)
    elif ext in [".eml", ".msg"]:
        loader = UnstructuredEmailLoader(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")
    return loader.load()

def create_rag_retriever(file_path: str):
    """Creates a RAG retriever using Semantic Chunking. Caching is removed."""
    print(f"\n🔧 Building RAG pipeline for: {os.path.basename(file_path)}")
    documents = load_single_document(file_path)
    
    text_splitter = SemanticChunker(embedding_model, breakpoint_threshold_type="percentile")
    chunked_documents = text_splitter.split_documents(documents)
    print(f"    - Loaded and split into {len(chunked_documents)} semantic chunks.")

    vectorstore = Chroma.from_documents(documents=chunked_documents, embedding=embedding_model)
    
    # **ROBUSTNESS FIX: Increased 'k' to retrieve more documents before re-ranking**
    # This increases the chance of capturing all relevant definitions.
    base_retriever = vectorstore.as_retriever(search_kwargs={"k": 15}) 
    compressor = CrossEncoderReranker(model=cross_encoder_model, top_n=5)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=base_retriever
    )
    print(f"    - Vector store and retriever are ready.")

    return compression_retriever, vectorstore

def process_claim_query(query: str, retriever, vectorstore):
    """Processes a query through the full RAG pipeline, now including a grounding check."""
    final_result = {}
    try:
        print(f"\n🚀 Processing Query: '{query}'")

        # 1. ADVANCED QUERY TRANSLATION
        query_generation_chain = query_generation_prompt | llm | query_parser
        generated_queries = query_generation_chain.invoke({"query": query})
        
        # 2. MULTI-FACETED RETRIEVAL
        all_retrieved_docs = []
        text_queries = [query] + generated_queries.get('sub_queries', []) + generated_queries.get('keywords', [])
        for q in text_queries:
            all_retrieved_docs.extend(retriever.invoke(q))

        hypothetical_answer = generated_queries.get('hypothetical_answer', "")
        hyde_docs = vectorstore.similarity_search(hypothetical_answer, k=3)
        all_retrieved_docs.extend(hyde_docs)

        unique_docs = {doc.page_content: doc for doc in all_retrieved_docs}.values()
        context = "\n\n".join([doc.page_content for doc in unique_docs])
        final_result['retrieved_context'] = list(unique_docs) # For UI display
        
        if not unique_docs:
            return {"summary": "Could not find relevant information.", "decision": "Further Information Required", "justification": "No relevant policy clauses found.", "amount": None, "clause_references": [], "grounding_check": "Not Performed"}

        # 3. SYNTHESIS
        synthesis_chain = synthesis_prompt | llm | decision_parser
        decision_output = synthesis_chain.invoke({"context": context, "query": query})
        final_result.update(decision_output)

        # 4. Grounding Check
        print("    - Performing grounding check...")
        grounding_chain = grounding_prompt | llm | StrOutputParser()
        grounding_response = grounding_chain.invoke({
            "context": context,
            "justification": final_result.get('justification', '')
        })
        final_result['grounding_check'] = grounding_response.strip()
        print(f"    - Grounding Check Result: {final_result['grounding_check']}")

        return final_result

    except Exception as e:
        print(f"An error occurred during query processing: {e}")
        return {"summary": "An error occurred during processing.", "decision": "Error", "justification": str(e), "amount": None, "clause_references": [], "grounding_check": "Failed"}
