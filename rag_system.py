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
from rank_bm25 import BM25Okapi

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

# **FIX:** The FinalResponse model now correctly includes all top-level fields.
class FinalResponse(BaseModel):
    overall_summary: str = Field(description="A concise, one-sentence summary of the outcome.")
    overall_decision: str = Field(description="The final, high-level decision: 'Approved', 'Denied', or 'Needs More Info'.")
    decisions: List[DecisionItem] = Field(description="A list of all plausible decisions and scenarios.")


class RAGSystem:
    def __init__(self, embedding_model_name='models/embedding-001', reranker_model_name='cross-encoder/ms-marco-MiniLM-L-6-v2'):
        print("Initializing models for RAG session...")
        self.embedding_model_name = embedding_model_name
        self.gemini_embeddings = GoogleGenerativeAIEmbeddings(model=self.embedding_model_name)
        self.reranker = CrossEncoder(reranker_model_name)
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0)
        self.db = None
        self.documents: List[Document] = []
        self.bm25 = None
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

        print("Creating BM25 index for keyword search...")
        tokenized_corpus = [doc.split(" ") for doc in doc_texts]
        self.bm25 = BM25Okapi(tokenized_corpus)
        print("BM25 index created successfully.")


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
                "You are an expert at analyzing legal and policy documents. Your task is to deconstruct a user's question into 3-5 more specific, diverse queries. "
                "These new queries should aim to find the core definitions, governing rules, specific conditions, exceptions, and related procedures to ensure all relevant context is retrieved.\n\n"
                "Deconstruct the following user query:\n"
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

        retrieved_docs_map = {} 

        print("Performing multi-layered hybrid search...")
        for q in all_queries:
            # 1a. Semantic Search
            query_embedding = self._embed_content([q], "retrieval_query")
            if query_embedding.size > 0:
                _, indices = self.db.search(query_embedding, n_retrieve)
                for i in indices[0]:
                    retrieved_docs_map[i] = self.documents[i]
            
            # 1b. Keyword Search
            tokenized_query = q.split(" ")
            bm25_scores = self.bm25.get_scores(tokenized_query)
            top_n_indices = np.argsort(bm25_scores)[::-1][:n_retrieve]
            for i in top_n_indices:
                retrieved_docs_map[i] = self.documents[i]

        initial_docs = list(retrieved_docs_map.values())
        print(f"Retrieved {len(initial_docs)} unique documents for re-ranking.")

        if not initial_docs:
            return []

        # 2. Re-rank the combined results with Gemini
        reranked_docs = self._rerank_with_gemini(query, initial_docs)
        
        return reranked_docs[:n_final]
        
    def _rerank_with_gemini(self, query: str, documents: List[Document]) -> List[Document]:
        """
        Re-ranks a list of documents using the Gemini LLM as a judge.
        """
        print(f"Re-ranking {len(documents)} documents with Gemini...")
        class RerankedDocument(BaseModel):
            document_index: int = Field(description="The original index of the document in the provided list.")
            relevance_score: float = Field(description="A score from 0.0 to 1.0 indicating the document's relevance to the query.")
            reasoning: str = Field(description="A brief justification for the assigned relevance score.")

        class RerankResponse(BaseModel):
            reranked_documents: List[RerankedDocument] = Field(description="An ordered list of documents, ranked by relevance.")

        parser = JsonOutputParser(pydantic_object=RerankResponse)
        
        doc_list_str = "\n".join([f"Doc [{i}]: {doc.page_content}" for i, doc in enumerate(documents)])
        
        prompt = PromptTemplate(
            template=(
                "You are a highly intelligent relevance ranking assistant. Your task is to re-rank the following documents based on their relevance to the user's query. "
                "Provide a relevance score from 0.0 (not relevant) to 1.0 (highly relevant) for each document. "
                "Return your response strictly as a JSON object that matches the specified format.\n\n"
                "USER QUERY: {query}\n\n"
                "DOCUMENTS:\n{documents}\n\n"
                "{format_instructions}"
            ),
            input_variables=["query", "documents"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )
        
        chain = prompt | self.llm | parser
        try:
            response = chain.invoke({"query": query, "documents": doc_list_str})
            
            reranked_indices = [item['document_index'] for item in sorted(response['reranked_documents'], key=lambda x: x['relevance_score'], reverse=True)]
            
            valid_indices = [i for i in reranked_indices if i < len(documents)]
            
            reranked_docs = [documents[i] for i in valid_indices]
            return reranked_docs
        except Exception as e:
            print(f"An error occurred during Gemini re-ranking: {e}. Returning original documents.")
            return documents

    def get_response(self, query: str) -> Dict[str, Any]:
        relevant_docs = self.retrieve_and_rerank(query)
        
        context_with_metadata = [
            f"Source Page: {doc.metadata.get('page', 'N/A')}\nSource Heading: {doc.metadata.get('heading', 'N/A')}\nContent: {doc.page_content}"
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
                "You are an expert policy and legal adjudicator AI. Your job is to analyze the provided document and return a structured understanding of how the policy applies to a specific query — or, if no clear query is given, return all applicable rules, scenarios, and exceptions.\n\n"
                "--- Your responsibilities ---\n\n"
                "1. **Query Interpretation**\n"
                "   - If the user provides a specific or vague query (e.g., \"Is knee surgery covered?\"), extract structured case facts such as age, procedure, geography, policy duration, or event type.\n"
                "   - If the query is completely **absent or ambiguous**, treat it as a request to enumerate **all possible covered or excluded cases** in the document.\n\n"
                "2. **Clause Extraction & Mapping**\n"
                "   - Retrieve and reference every relevant clause, definition, or rule in the document that:\n"
                "     - Defines eligibility or exclusions\n"
                "     - Describes payout logic or limits\n"
                "     - Contains timelines, waiting periods, or jurisdictional conditions\n"
                "     - Includes monetary amounts, thresholds, or ceilings\n\n"
                "3. **Outcome Generation**\n"
                "   - When possible, provide a **definitive decision** (e.g., *Approved*, *Denied*, *Conditionally Approved*, *Amount Payable*).\n"
                "   - If a decision cannot be made without further details, provide **all plausible outcomes**, each with:\n"
                "     - Associated clauses\n"
                "     - Clear assumptions\n"
                "     - Specific decision logic\n\n"
                "4. **Total Transparency**\n"
                "   - Always list all **assumptions** explicitly.\n"
                "   - Mention **amounts**, **limits**, or **waiting periods** exactly as stated.\n"
                "   - Never ignore edge cases — surface them clearly.\n\n"
                "--- Return your answer **strictly in the following JSON format**, structured for downstream systems and audits. Here is an example of the required format:\n\n"
                "```json\n"
                "{{\n"
                "  \"overall_summary\": \"A concise, one-sentence summary of the outcome.\",\n"
                "  \"overall_decision\": \"Approved | Denied | Needs More Info\",\n"
                "  \"decisions\": [\n"
                "    {{\n"
                "      \"scenario\": \"Description of scenario or inferred case (e.g., 'Knee surgery under international plan')\",\n"
                "      \"decision\": \"Approved | Denied | Needs More Info | Multiple Outcomes\",\n"
                "      \"amount\": \"₹100,000 or 'Up to annual limit of ₹5,00,000'\",\n"
                "      \"justification\": {{\n"
                "        \"summary\": \"Why this decision was made\",\n"
                "        \"assumptions\": [\"If the user holds Imperial Plus Plan\", \"If procedure is post-hospitalization\"],\n"
                "        \"clauses_used\": [\n"
                "          {{\n"
                "            \"clause_title\": \"Coverage for Surgical Procedures\",\n"
                "            \"clause_text\": \"The policy covers procedures conducted under inpatient hospitalization...\",\n"
                "            \"page_number\": 12,\n"
                "            \"matched_terms\": [\"surgery\", \"hospitalization\", \"waiting period\"]\n"
                "          }}\n"
                "        ],\n"
                "        \"alternate_outcomes\": [\n"
                "          \"Under Domestic Plan only, this would be denied due to lack of international coverage.\",\n"
                "          \"If the policy was under 30 days old, a waiting period would apply and this would be denied.\"\n"
                "        ]\n"
                "      }}\n"
                "    }}\n"
                "  ]\n"
                "}}\n"
                "```\n\n"
                "**Policy Context:**\n---\n{context}\n---\n\n"
                "**User Query:** \"{query}\"\n\n"
                "**Your JSON Response (strictly following the schema provided in your instructions):**\n"
                "{format_instructions}"
            ),
            input_variables=["context", "query"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )
