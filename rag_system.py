# rag_system.py

import os
import json
from typing import List, Dict, Any

import google.generativeai as genai
import faiss
import numpy as np

# --- Document Loading ---
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredEmailLoader
from sentence_transformers import CrossEncoder
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
# **FIX:** Import the LangChain wrapper for the Gemini model
from langchain_google_genai import ChatGoogleGenerativeAI

# --- Pydantic Models ---
from pydantic import BaseModel, Field

class DecomposedQuery(BaseModel):
    """Pydantic model for the decomposed queries."""
    decomposed_queries: List[str] = Field(description="A list of 3-5 diverse questions generated from the user's original query.")

# **FIX:** Re-added the final response models
class ClauseInfo(BaseModel):
    clause_title: str = Field(description="The title or heading of the relevant clause.")
    clause_text: str = Field(description="The exact text of the clause that was used for the decision.")
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
    decisions: List[DecisionItem] = Field(description="A list of all plausible decisions and scenarios.")


class RAGSystem:
    """
    A class that encapsulates the entire RAG system, from document processing to response generation.
    """
    def __init__(self, embedding_model_name='models/embedding-001', reranker_model_name='cross-encoder/ms-marco-MiniLM-L-6-v2'):
        print("Initializing models for RAG session...")
        self.embedding_model_name = embedding_model_name
        self.reranker = CrossEncoder(reranker_model_name)
        # **FIX:** Use the LangChain-compatible ChatGoogleGenerativeAI model
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0)
        self.db = None
        self.documents = []
        print("Models initialized.")

    def _embed_content(self, content: List[str], task_type: str) -> np.ndarray:
        """Helper function to embed content using the Gemini API."""
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
        """
        Loads, chunks, and indexes a document into a FAISS vector store.
        """
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
        
        pages = loader.load_and_split()
        self.documents = [page.page_content for page in pages]
        print(f"Document loaded and split into {len(self.documents)} chunks.")

        print("Creating FAISS index with Gemini embeddings...")
        embeddings = self._embed_content(self.documents, "retrieval_document")
        if embeddings.size == 0:
            raise ValueError("Failed to create document embeddings.")
            
        embedding_dim = embeddings.shape[1]
        self.db = faiss.IndexFlatL2(embedding_dim)
        self.db.add(embeddings)
        print("FAISS index created successfully.")

    def _decompose_query(self, query: str) -> List[str]:
        """
        Decomposes the user's query into multiple, more specific questions.
        """
        parser = JsonOutputParser(pydantic_object=DecomposedQuery)
        prompt = PromptTemplate(
            template=(
                "You are an expert at query analysis. Your task is to deconstruct a user's question into 3-5 more specific, diverse questions that cover different facets of the original query. "
                "For example, if the user asks 'Is my claim for an unregistered hospital covered?', you should generate questions like:\n"
                "- What is the policy's definition of a 'Hospital'?\n"
                "- Are there specific registration requirements for medical facilities?\n"
                "- What are the general conditions for claim admissibility?\n\n"
                "Deconstruct the following user query:\n"
                "QUERY: {query}\n\n{format_instructions}"
            ),
            input_variables=["query"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )
        
        # This LCEL chain will now work correctly
        chain = prompt | self.llm | parser
        response = chain.invoke({"query": query})
        return response['decomposed_queries']

    def retrieve_and_rerank(self, query: str, n_retrieve: int = 5, n_final: int = 5) -> List[str]:
        """
        Uses multi-query retrieval and then re-ranks the results for higher accuracy.
        """
        print("Decomposing query for multi-faceted retrieval...")
        decomposed_queries = self._decompose_query(query)
        all_queries = [query] + decomposed_queries
        print(f"Generated queries: {all_queries}")

        all_retrieved_docs = []
        for q in all_queries:
            query_embedding = self._embed_content([q], "retrieval_query")
            if query_embedding.size > 0:
                distances, indices = self.db.search(query_embedding, n_retrieve)
                all_retrieved_docs.extend([self.documents[i] for i in indices[0]])

        unique_docs = list(dict.fromkeys(all_retrieved_docs))
        print(f"Retrieved {len(unique_docs)} unique documents for re-ranking.")

        if not unique_docs:
            return []

        pairs = [[query, doc] for doc in unique_docs]
        scores = self.reranker.predict(pairs)
        
        scored_docs = list(zip(scores, unique_docs))
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        
        reranked_passages = [doc for score, doc in scored_docs[:n_final]]
        return reranked_passages

    def get_response(self, query: str) -> Dict[str, Any]:
        """
        Processes a query and returns a structured JSON response.
        """
        relevant_passages = self.retrieve_and_rerank(query)
        
        parser = JsonOutputParser(pydantic_object=FinalResponse)
        prompt = self._make_prompt(parser)
        
        print("Generating response from LLM...")
        # **FIX:** Use a proper LCEL chain for the final response generation
        chain = prompt | self.llm | parser
        response = chain.invoke({
            "context": "\n".join(relevant_passages),
            "query": query
        })
        
        return response

    def _make_prompt(self, parser: JsonOutputParser) -> PromptTemplate:
        """
        Creates the final prompt for the Gemini Pro model.
        """
        # **PROMPT UPGRADE: Using the detailed, expert adjudicator prompt with an explicit JSON example**
        return PromptTemplate(
            template=(
                "You are an expert policy and legal adjudicator AI operating over complex documents. Your job is to analyze the provided document and return a structured understanding of how the policy applies to a specific query — or, if no clear query is given, return all applicable rules, scenarios, and exceptions.\n\n"
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
