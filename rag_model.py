# rag_backend.py

import os
import json
import hashlib
import joblib
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

# **PROMPT UPGRADE: New Pydantic models for the detailed, multi-scenario JSON output**
class ClauseInfo(BaseModel):
    """Model for a single clause used in a decision."""
    clause_title: str = Field(description="The title or heading of the relevant clause.")
    clause_text: str = Field(description="The exact text of the clause that was used for the decision.")
    matched_terms: List[str] = Field(description="Specific terms from the clause that match the query.")

class Justification(BaseModel):
    """Model for the detailed justification of a decision."""
    summary: str = Field(description="A brief summary of why the decision was made.")
    assumptions: List[str] = Field(description="A list of assumptions made to reach the decision.")
    clauses_used: List[ClauseInfo] = Field(description="A list of specific clauses that support the decision.")
    alternate_outcomes: List[str] = Field(description="Potential outcomes if the assumptions were different.")

class DecisionItem(BaseModel):
    """Model for a single decision scenario."""
    scenario: str = Field(description="A description of the scenario or inferred case being analyzed.")
    decision: str = Field(description="The decision for this scenario: 'Approved', 'Denied', 'Needs More Info', etc.")
    amount: str = Field(description="The applicable amount, described textually or as a number (e.g., '₹100,000' or 'Up to annual limit of ₹5,00,000').")
    justification: Justification = Field(description="A nested object containing the detailed justification.")

class FinalResponse(BaseModel):
    """The final, top-level response model containing all decisions."""
    decisions: List[DecisionItem] = Field(description="A list of all plausible decisions and scenarios.")


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
# Use the new comprehensive decision model for the parser
decision_parser = JsonOutputParser(pydantic_object=FinalResponse)

query_generation_prompt = PromptTemplate(
    template="You are an expert query analyst. Analyze the user's question and generate a set of more effective queries to improve document retrieval.\n1. **Sub-Queries:** Decompose the query into smaller questions.\n2. **Hypothetical Answer (HyDE):** Create an ideal answer.\n3. **Keywords:** Extract key terms and synonyms.\n\n**User Query:** \"{query}\"\n\n{format_instructions}",
    input_variables=["query"],
    partial_variables={"format_instructions": query_parser.get_format_instructions()},
)

# **PROMPT UPGRADE: New prompt reflecting the detailed adjudicator persona and complex JSON output**
synthesis_prompt = PromptTemplate(
    template=(
        "You are an expert policy and legal adjudicator AI. Your job is to analyze the provided document context and return a structured understanding of how the policy applies to a specific query. "
        "Your responsibilities:\n"
        "1. **Query Interpretation:** Extract structured case facts from the query (age, procedure, location, etc.). If the query is absent or ambiguous, treat it as a request to enumerate all possible covered or excluded cases.\n"
        "2. **Clause Extraction & Mapping:** Retrieve and reference every relevant clause, definition, or rule.\n"
        "3. **Outcome Generation:** Provide a definitive decision when possible. If not, provide all plausible outcomes, each with associated clauses, clear assumptions, and specific decision logic.\n"
        "4. **Total Transparency:** Always list all assumptions, amounts, limits, and waiting periods exactly as stated. Never ignore edge cases.\n\n"
        "Return your answer **strictly in the specified JSON format**.\n\n"
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
            return {"decisions": [{"scenario": "No Information Found", "decision": "Needs More Info", "amount": "N/A", "justification": {"summary": "No relevant policy clauses could be found to address the query.", "assumptions": [], "clauses_used": [], "alternate_outcomes": []}}]}

        # 3. SYNTHESIS
        synthesis_chain = synthesis_prompt | llm | decision_parser
        decision_output = synthesis_chain.invoke({"context": context, "query": query})
        final_result.update(decision_output)

        # 4. Grounding Check (Applied to all justifications)
        print("    - Performing grounding check...")
        grounding_chain = grounding_prompt | llm | StrOutputParser()
        all_justifications = ". ".join([d['justification']['summary'] for d in final_result.get('decisions', [])])
        grounding_response = grounding_chain.invoke({
            "context": context,
            "justification": all_justifications
        })
        final_result['grounding_check'] = grounding_response.strip()
        print(f"    - Grounding Check Result: {final_result['grounding_check']}")

        return final_result

    except Exception as e:
        print(f"An error occurred during query processing: {e}")
        return {"decisions": [{"scenario": "Processing Error", "decision": "Error", "amount": "N/A", "justification": {"summary": str(e), "assumptions": [], "clauses_used": [], "alternate_outcomes": []}}]}
