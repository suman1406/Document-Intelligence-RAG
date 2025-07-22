# rag_system.py

import os
import json
from typing import List, Dict, Any

import google.generativeai as genai
import faiss
import numpy as np

# --- Document Loading & Processing ---
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredEmailLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from sentence_transformers import CrossEncoder
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.docstore.document import Document

# --- Pydantic Models ---
from pydantic import BaseModel, Field

# --- Pydantic Models for Structured Output ---
class PolicySummary(BaseModel):
    policy_name: str = Field(description="The official name of the policy or document.")
    policy_type: str = Field(description="The type of document (e.g., 'Health Insurance Policy', 'Service Agreement').")
    key_monetary_limits: List[str] = Field(description="A list of the most important monetary limits or coverage amounts.")
    major_waiting_periods: List[str] = Field(description="A list of the most significant waiting periods for coverage.")
    key_exclusions: List[str] = Field(description="A list of the top 3-5 most important exclusions.")

class DecomposedQuery(BaseModel):
    decomposed_queries: List[str] = Field(description="A list of 3-5 diverse questions generated from the user's original query.")

class ClauseInfo(BaseModel):
    clause_title: str = Field(description="The title or heading of the relevant clause.")
    clause_text: str = Field(description="The exact text of the clause that was used for the decision.")
    page_number: int = Field(description="The page number where this clause can be found.")
    matched_terms: List[str] = Field(description="Specific terms from the clause that match the query.")

class Justification(BaseModel):
    summary: str = Field(description="A brief summary of why the decision was made.")
    assumptions: List[str] = Field(description="A list of assumptions made to reach the decision.")
    clauses_used: List[ClauseInfo] = Field(description="A list of specific clauses that support the decision.")
    alternate_outcomes: List[str] = Field(description="Potential outcomes if the assumptions were different.")

class DecisionItem(BaseModel):
    scenario: str = Field(description="A description of the scenario or inferred case being analyzed.")
    decision: str = Field(description="The decision for this scenario: 'Approved', 'Denied', 'Needs More Info', etc.")
    amount: str = Field(description="The applicable amount, described textually or as a number.")
    justification: Justification = Field(description="A nested object containing the detailed justification.")

class FinalResponse(BaseModel):
    overall_summary: str = Field(description="A concise, one-sentence summary of the outcome.")
    overall_decision: str = Field(description="The final, high-level decision: 'Approved', 'Denied', or 'Needs More Info'.")
    decisions: List[DecisionItem] = Field(description="A list of all plausible decisions and scenarios.")


class RAGSystem:
    def __init__(self, embedding_model_name='models/embedding-001', reranker_model_name='cross-encoder/ms-marco-MiniLM-L-6-v2'):
        print("Initializing models for RAG session...")
        # **UPGRADE:** Use the Gemini embedding model for all embedding tasks, including chunking.
        self.embedding_model_name = embedding_model_name
        self.gemini_embeddings = GoogleGenerativeAIEmbeddings(model=self.embedding_model_name)
        self.reranker = CrossEncoder(reranker_model_name)
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0)
        self.db = None
        self.documents: List[Document] = []
        print("Models initialized.")

    def _embed_content(self, content: List[str], task_type: str) -> np.ndarray:
        try:
            result = genai.embed_content(
                model=self.embedding_model_name,
                content=content,
                task_type=task_type
            )
            return np.array(result['embedding'], dtype=np.float32)
        except Exception as e:
            print(f"An error occurred during embedding: {e}")
            return np.array([])

    def load_and_process_document(self, file_path: str):
        print(f"Loading document from: {file_path}")
        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".pdf":
            loader = PyPDFLoader(file_path)
        elif ext == ".docx":
            loader = Docx2txtLoader(file_path)
        elif ext in [".eml", ".msg"]:
            loader = UnstructuredEmailLoader(file_path)
        else:
            raise ValueError(f"Unsupported file type: {ext}")
        
        loaded_docs = loader.load()

        # **ACCURACY UPGRADE:** Using SemanticChunker with the Gemini embedding model.
        print("Applying semantic chunking...")
        text_splitter = SemanticChunker(self.gemini_embeddings)
        self.documents = text_splitter.split_documents(loaded_docs)
        
        doc_texts = [doc.page_content for doc in self.documents]
        print(f"Document loaded and split into {len(self.documents)} semantic chunks.")

        print("Creating FAISS index with Gemini embeddings...")
        embeddings = self._embed_content(doc_texts, "retrieval_document")
        if embeddings.size == 0:
            raise ValueError("Failed to create document embeddings.")
            
        embedding_dim = embeddings.shape[1]
        self.db = faiss.IndexFlatL2(embedding_dim)
        self.db.add(embeddings)
        print("FAISS index created successfully.")

    def summarize_document(self) -> Dict[str, Any]:
        print("Generating proactive document summary...")
        summary_keywords = [
            "Table of Benefits", "Sum Insured", "Exclusions", "Waiting Period", 
            "Pre-Existing Disease", "Scope of cover", "Preamble"
        ]
        
        retrieved_indices = set()
        for keyword in summary_keywords:
            keyword_embedding = self._embed_content([keyword], "retrieval_query")
            if keyword_embedding.size > 0:
                _, indices = self.db.search(keyword_embedding, 2)
                retrieved_indices.update(indices[0])

        summary_docs = [self.documents[i] for i in retrieved_indices]
        summary_context = " ".join([doc.page_content for doc in summary_docs])

        parser = JsonOutputParser(pydantic_object=PolicySummary)
        prompt = PromptTemplate(
            template=(
                "You are an expert policy analyst. Based on the following text from a policy document, "
                "extract the key details into a structured JSON object. Focus on identifying the policy's name, type, "
                "major monetary limits, significant waiting periods, and the most important exclusions.\n\n"
                "CONTEXT:\n---\n{context}\n---\n\n{format_instructions}"
            ),
            input_variables=["context"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )
        chain = prompt | self.llm | parser
        summary = chain.invoke({"context": summary_context})
        return summary

    def _decompose_query(self, query: str) -> List[str]:
        parser = JsonOutputParser(pydantic_object=DecomposedQuery)
        prompt = PromptTemplate(
            template=(
                "You are an expert at query analysis. Deconstruct a user's question into 3-5 specific, diverse questions "
                "that cover different facets of the original query to ensure comprehensive document retrieval.\n"
                "QUERY: {query}\n\n{format_instructions}"
            ),
            input_variables=["query"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )
        
        chain = prompt | self.llm | parser
        response = chain.invoke({"query": query})
        return response['decomposed_queries']

    def retrieve_and_rerank(self, query: str, n_retrieve: int = 10, n_final: int = 5) -> List[Document]:
        print("Decomposing query for multi-faceted retrieval...")
        decomposed_queries = self._decompose_query(query)
        all_queries = [query] + decomposed_queries
        print(f"Generated queries: {all_queries}")

        retrieved_indices = set()
        for q in all_queries:
            query_embedding = self._embed_content([q], "retrieval_query")
            if query_embedding.size > 0:
                _, indices = self.db.search(query_embedding, n_retrieve)
                retrieved_indices.update(indices[0])
        
        initial_docs = [self.documents[i] for i in retrieved_indices]
        print(f"Retrieved {len(initial_docs)} unique documents for re-ranking.")

        if not initial_docs:
            return []

        pairs = [[query, doc.page_content] for doc in initial_docs]
        scores = self.reranker.predict(pairs)
        
        scored_docs = list(zip(scores, initial_docs))
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        
        reranked_docs = [doc for score, doc in scored_docs[:n_final]]
        return reranked_docs

    def get_response(self, query: str) -> Dict[str, Any]:
        relevant_docs = self.retrieve_and_rerank(query)
        
        context_with_metadata = [
            f"Source Page: {doc.metadata.get('page', 'N/A')}\nContent: {doc.page_content}"
            for doc in relevant_docs
        ]
        context = "\n---\n".join(context_with_metadata)

        parser = JsonOutputParser(pydantic_object=FinalResponse)
        prompt = self._make_prompt(parser)
        
        print("Generating response from LLM...")
        chain = prompt | self.llm | parser
        response = chain.invoke({
            "context": context,
            "query": query
        })
        
        return response

    def _make_prompt(self, parser: JsonOutputParser) -> PromptTemplate:
        return PromptTemplate(
            template=(
                "You are an expert policy and legal adjudicator AI. Your job is to analyze the provided document context and return a structured understanding of how the policy applies to a specific query.\n\n"
                "--- Your responsibilities ---\n"
                "1. **Query Interpretation:** Extract structured case facts from the query (age, procedure, etc.).\n"
                "2. **Clause Extraction & Mapping:** Reference every relevant clause from the context.\n"
                "3. **Outcome Generation:** Provide a definitive decision when possible. If not, provide all plausible outcomes with clear assumptions.\n"
                "4. **Total Transparency:** Always list all assumptions, amounts, limits, and waiting periods exactly as stated.\n\n"
                "--- Response Format ---\n"
                "Return your answer strictly in the following JSON format. Provide a high-level summary and decision first, followed by the detailed breakdown.\n"
                "```json\n"
                "{{\n"
                "  \"overall_summary\": \"A concise, one-sentence summary of the outcome.\",\n"
                "  \"overall_decision\": \"Approved | Denied | Needs More Info\",\n"
                "  \"decisions\": [\n"
                "    {{\n"
                "      \"scenario\": \"Description of scenario or inferred case\",\n"
                "      \"decision\": \"Approved | Denied | Needs More Info | Multiple Outcomes\",\n"
                "      \"amount\": \"₹100,000 or 'Up to Sum Insured'\",\n"
                "      \"justification\": {{\n"
                "        \"summary\": \"Why this decision was made\",\n"
                "        \"assumptions\": [\"If the user holds Imperial Plus Plan\"],\n"
                "        \"clauses_used\": [\n"
                "          {{\n"
                "            \"clause_title\": \"Coverage for Surgical Procedures\",\n"
                "            \"clause_text\": \"The policy covers procedures...\",\n"
                "            \"page_number\": 12,\n"
                "            \"matched_terms\": [\"surgery\", \"hospitalization\"]\n"
                "          }}\n"
                "        ],\n"
                "        \"alternate_outcomes\": [\n"
                "          \"Under Domestic Plan only, this would be denied...\"\n"
                "        ]\n"
                "      }}\n"
                "    }}\n"
                "  ]\n"
                "}}\n"
                "```\n\n"
                "**Policy Context:**\n---\n{context}\n---\n\n"
                "**User Query:** \"{query}\"\n\n"
                "**Your JSON Response (strictly following the schema and ensuring `page_number` is included for every clause):**\n"
                "{format_instructions}"
            ),
            input_variables=["context", "query"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )