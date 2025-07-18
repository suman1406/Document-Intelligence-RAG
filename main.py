import os
import getpass
from dotenv import load_dotenv
from io import BytesIO
import requests
import re
import json

# Langchain core components
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

# Langchain community components (loaders, vectorstores)
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    Docx2txtLoader,
    UnstructuredEmailLoader
)
from langchain.chains.combine_documents import create_stuff_documents_chain

# Langchain text splitting
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Langchain Google integrations
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

# --- Constants for pretty printing ---
LINE_SEP = "=" * 80
SECTION_SEP = "-" * 80
SUBSECTION_SEP = "*" * 40
COLOR_BLUE = "\033[94m"
COLOR_GREEN = "\033[92m"
COLOR_RED = "\033[91m"
COLOR_END = "\033[0m"

# === Load Environment and API Key ===
load_dotenv()
if not os.environ.get("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter Google API Key: ")


# === STEP 1: Create RAG Chain ===
def get_conversational_chain():
    json_format_instructions = """
    Your output MUST be a JSON object with the following keys and format:
    'Decision': 'Approved', 'Rejected', or 'Cannot Determine' (Choose 'Cannot Determine' ONLY if absolutely no relevant information is found in the context to answer the question, or if the context directly states something is "Not applicable" or "Excluded" without providing a clear alternative path to coverage).
    'Amount': (If an amount or cost is mentioned or implied, state it, e.g., 'Determined by medical bills', 'Not applicable', 'INR 500,000'. Otherwise, 'Not applicable').
    'Justification': 'A clear and concise explanation based *only* on the provided policy clauses and definitions. Explain WHY a decision is made or why it cannot be determined. Directly reference specific policy sections or definitions that lead to your conclusion. If the decision is 'Rejected' due to an exclusion, clearly state the exclusion.'
    'RelevantClauses': ['List of specific clauses or definitions from the policy document that directly support your decision and justification. Include their approximate section/definition number and Page X where possible (e.g., "Section D, 1) Pre-Existing Diseases (Code -Exc101) a. [Page 20]", "37. Pre-Existing Disease [Page 5]").']
    """

    # MODIFIED PROMPT TEMPLATE - Made instructions more concise and direct
    prompt_template = f"""
    You are an expert AI assistant specialized in analyzing insurance policy documents.
    Your goal is to provide precise answers to user questions about policy coverage by strictly using the provided context.

    **Guidance for your response:**
    1.  **Decision (Mandatory):**
        * 'Approved': If context explicitly shows coverage.
        * 'Rejected': If context explicitly shows exclusion or limitations preventing coverage.
        * 'Cannot Determine': Only if the context offers NO information to approve or reject.
    2.  **Amount (Mandatory):** State specific amounts (e.g., 'INR 500,000'), 'Determined by medical bills', or 'Not applicable' based on context.
    3.  **Justification (Mandatory):** Concisely explain your decision. **ONLY use information from the provided context.** Clearly state which clauses/definitions support your conclusion, especially for rejections (mentioning the exclusion).
    4.  **RelevantClauses (Mandatory):** List exact clauses or definitions. Include approximate section/definition numbers and page numbers (e.g., "Section D, 1) Pre-Existing Diseases (Code -Exc101) [Page 20]").

    {json_format_instructions}

    Context:
    {{context}}

    Question:
    {{question}}

    Answer:
    """

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    model = ChatGoogleGenerativeAI(
        model="gemini-2.5-pro",
        temperature=0.1
    )

    return create_stuff_documents_chain(llm=model, prompt=prompt)


# === STEP 2: Load Document from Path or URL and Extract Sections ===
def load_document_and_extract_sections(file_path_or_link):
    print(f"\n{COLOR_BLUE}{SUBSECTION_SEP}\n Pre-processing document to identify definitions and exclusions...{COLOR_END}")
    
    raw_docs = []
    # Load raw documents
    if file_path_or_link.startswith("http://") or file_path_or_link.startswith("https://"):
        try:
            response = requests.get(file_path_or_link)
            response.raise_for_status()
            if "pdf" in response.headers.get("Content-Type", "") or file_path_or_link.lower().endswith(".pdf"):
                raw_docs = PyPDFLoader(BytesIO(response.content)).load()
            else:
                print(f"{COLOR_RED}Warning: Section extraction is optimized for PDFs. Unsupported URL file type: {file_path_or_link}. Continuing with standard loading.{COLOR_END}")
                if "officedocument.wordprocessingml.document" in response.headers.get("Content-Type", "") or file_path_or_link.lower().endswith(".docx"):
                    raw_docs = Docx2txtLoader(BytesIO(response.content)).load()
                elif "message" in response.headers.get("Content-Type", "") or file_path_or_link.lower().endswith(".eml"):
                    raw_docs = UnstructuredEmailLoader(BytesIO(response.content)).load()
                else:
                    raise ValueError(f"Unsupported URL file type for standard loading: {file_path_or_link}")
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Download error for {file_path_or_link}: {e}")
    else: # Local file loading
        if file_path_or_link.lower().endswith(".pdf"):
            raw_docs = PyPDFLoader(file_path_or_link).load()
        elif file_path_or_link.lower().endswith(".docx"):
            raw_docs = Docx2txtLoader(file_path_or_link).load()
        elif file_path_or_link.lower().endswith(".eml"):
            raw_docs = UnstructuredEmailLoader(file_path_or_link).load()
        elif file_path_or_link.lower().endswith(".txt"):
            raw_docs = TextLoader(file_path_or_link).load()
        else:
            raise ValueError(f"Unsupported local file extension: {file_path_or_link}. Supported: .pdf, .docx, .eml, .txt")

    # Add source information to each document page immediately after loading
    for i, doc in enumerate(raw_docs):
        page_num = doc.metadata.get('page', i + 1)
        doc.metadata["source"] = f"Page {page_num}"

    definitions_by_name = {}
    exclusions_by_code = {}
    
    definition_pattern = re.compile(r'^\s*(\d+)\.\s*(.+?):-?\s*$', re.IGNORECASE)
    exclusion_pattern = re.compile(r'^\s*(\d+)\)\s*([^()]+?)\s*(\(Code\s*-[^)]+\))$', re.IGNORECASE)

    full_text_content = "\n".join([doc.page_content for doc in raw_docs])
    
    # --- Extract Standard Definitions ---
    std_def_section_match = re.search(r'SECTION B\)\s*DEFINITIONS\s*-\s*STANDARD DEFINITIONS(.*?)SECTION B\)\s*DEFINITIONS\s*-\s*SPECIFIC DEFINITIONS', full_text_content, re.DOTALL)
    if std_def_section_match:
        section_text = std_def_section_match.group(1)
        current_def_name = None
        for line in section_text.split('\n'):
            line_stripped = line.strip()
            match = definition_pattern.match(line_stripped)
            if match:
                def_num = match.group(1)
                def_name_raw = match.group(2).strip()
                current_def_name = f"{def_num}. {def_name_raw}"
                def_source = "Unknown Page"
                for doc in raw_docs:
                    if line_stripped in doc.page_content:
                        def_source = doc.metadata.get("source", "Unknown Page")
                        break
                definitions_by_name[current_def_name] = {"content": [], "source": def_source}
            elif current_def_name and line_stripped and not re.match(r'^\s*\d+\.\s*', line_stripped) and not line_stripped.startswith('Global Health Care/ Policy Wordings/Page'):
                definitions_by_name[current_def_name]["content"].append(line_stripped)

    # --- Extract Specific Definitions ---
    spec_def_section_match = re.search(r'SECTION B\)\s*DEFINITIONS\s*-\s*SPECIFIC DEFINITIONS(.*?)SECTION C\)\s*BENEFITS COVERED UNDER THE POLICY', full_text_content, re.DOTALL)
    if spec_def_section_match:
        section_text = spec_def_section_match.group(1)
        current_def_name = None
        for line in section_text.split('\n'):
            line_stripped = line.strip()
            match = definition_pattern.match(line_stripped)
            if match:
                def_num = match.group(1)
                def_name_raw = match.group(2).strip()
                current_def_name = f"Specific Definition {def_num}. {def_name_raw}"
                def_source = "Unknown Page"
                for doc in raw_docs:
                    if line_stripped in doc.page_content:
                        def_source = doc.metadata.get("source", "Unknown Page")
                        break
                definitions_by_name[current_def_name] = {"content": [], "source": def_source}
            elif current_def_name and line_stripped and not re.match(r'^\s*\d+\.\s*', line_stripped) and not line_stripped.startswith('Global Health Care/ Policy Wordings/Page'):
                definitions_by_name[current_def_name]["content"].append(line_stripped)
    
    # --- Extract Standard Exclusions APPLICABLE TO PART A- DOMESTIC COVER ---
    domestic_exc_section_match = re.search(r'SECTION D\)\s*EXCLUSIONS-\s*STANDARD EXCLUSIONS APPLICABLE TO PART A- DOMESTIC COVER UNDER SECTION C\) BENEFITS COVERED UNDER THE POLICY(.*?)SECTION D\)\s*EXCLUSIONS-SPECIFIC EXCLUSIONS APPLICABLE TO PART A- DOMESTIC COVER', full_text_content, re.DOTALL)
    if domestic_exc_section_match:
        section_text = domestic_exc_section_match.group(1)
        current_exc_code_key = None
        for line in section_text.split('\n'):
            line_stripped = line.strip()
            match = exclusion_pattern.match(line_stripped)
            if match:
                exc_num = match.group(1)
                exc_name = match.group(2).strip()
                exc_code_part = match.group(3).strip()
                current_exc_code_key = f"Domestic Exclusion {exc_num}) {exc_name} {exc_code_part}"
                exc_source = "Unknown Page"
                for doc in raw_docs:
                    if line_stripped in doc.page_content:
                        exc_source = doc.metadata.get("source", "Unknown Page")
                        break
                exclusions_by_code[current_exc_code_key] = {"content": [], "source": exc_source}
            elif current_exc_code_key and line_stripped and not re.match(r'^\s*\d+\)\s*', line_stripped) and not line_stripped.startswith('Global Health Care/ Policy Wordings/Page') and not line_stripped.startswith('The following table:'):
                exclusions_by_code[current_exc_code_key]["content"].append(line_stripped)

    # --- Extract Standard Exclusions APPLICABLE TO PART B- INTERNATIONAL COVER ---
    international_exc_section_match = re.search(r'SECTION D\)\s*EXCLUSIONS-\s*STANDARD EXCLUSIONS APPLICABLE TO PART B- INTERNATIONAL COVER UNDER SECTION C\) BENEFITS COVERED UNDER THE POLICY(.*?)SECTION D\)\s*EXCLUSIONS-\s*SPECIFIC EXCLUSIONS APPLICABLE TO INTERNATIONAL COVER', full_text_content, re.DOTALL)
    if international_exc_section_match:
        section_text = international_exc_section_match.group(1)
        current_exc_code_key = None
        for line in section_text.split('\n'):
            line_stripped = line.strip()
            match = exclusion_pattern.match(line_stripped)
            if match:
                exc_num = match.group(1)
                exc_name = match.group(2).strip()
                exc_code_part = match.group(3).strip()
                current_exc_code_key = f"International Exclusion {exc_num}) {exc_name} {exc_code_part}"
                exc_source = "Unknown Page"
                for doc in raw_docs:
                    if line_stripped in doc.page_content:
                        exc_source = doc.metadata.get("source", "Unknown Page")
                        break
                exclusions_by_code[current_exc_code_key] = {"content": [], "source": exc_source}
            elif current_exc_code_key and line_stripped and not re.match(r'^\s*\d+\)\s*', line_stripped) and not line_stripped.startswith('Global Health Care/ Policy Wordings/Page') and not line_stripped.startswith('The following table:'):
                exclusions_by_code[current_exc_code_key]["content"].append(line_stripped)


    # Consolidate content for definitions and exclusions into Document objects
    final_definitions = {}
    for name, data in definitions_by_name.items():
        if data["content"]:
            full_content = "\n".join(data["content"])
            final_definitions[name] = Document(page_content=f"{name}: {full_content}", metadata={"source": data["source"], "type": "definition", "name": name})

    final_exclusions = {}
    for code, data in exclusions_by_code.items():
        if data["content"]:
            full_content = "\n".join(data["content"])
            final_exclusions[code] = Document(page_content=f"{code}: {full_content}", metadata={"source": data["source"], "type": "exclusion", "code": code})
    
    print(f"{COLOR_GREEN}Identified {len(final_definitions)} definitions and {len(final_exclusions)} exclusions.{COLOR_END}\n")
    return raw_docs, final_definitions, final_exclusions


# === STEP 3: Handle User Query ===
def user_input(user_question, file_path_or_link):
    print(f"\n{COLOR_BLUE}🔍 Loading and embedding document for query: '{user_question}'...{COLOR_END}")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    documents, definitions_map, exclusions_map = load_document_and_extract_sections(file_path_or_link)
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)

    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i

    vector_store = FAISS.from_documents(chunks, embedding=embeddings)
    
    semantic_similar_docs = vector_store.similarity_search(user_question, k=5) 

    additional_docs = []
    unique_combined_docs_content = set() 
    final_docs_for_llm = []

    # First, add semantic search results
    for doc in semantic_similar_docs:
        if doc.page_content not in unique_combined_docs_content:
            final_docs_for_llm.append(doc)
            unique_combined_docs_content.add(doc.page_content)


    # --- Hybrid Retrieval Logic ---
    # Prioritize specific definitions/exclusions if keywords are in the query
    
    # Check for Pre-Existing Disease keywords
    ped_keywords_query = ["pre-existing disease", "pre-existing condition", "ped", "health issue before", "old medical problem"]
    if any(keyword in user_question.lower() for keyword in ped_keywords_query):
        ped_def_key = "37. Pre-Existing Disease"
        if ped_def_key in definitions_map:
            doc_to_add = definitions_map[ped_def_key]
            if doc_to_add.page_content not in unique_combined_docs_content:
                additional_docs.append(doc_to_add)
                unique_combined_docs_content.add(doc_to_add.page_content)
                print(f"{COLOR_GREEN}Added targeted definition: {ped_def_key}{COLOR_END}")
        
        ped_exc_domestic_key = "Domestic Exclusion 1) Pre-Existing Diseases (Code -Exc101)"
        if ped_exc_domestic_key in exclusions_map:
            doc_to_add = exclusions_map[ped_exc_domestic_key]
            if doc_to_add.page_content not in unique_combined_docs_content:
                additional_docs.append(doc_to_add)
                unique_combined_docs_content.add(doc_to_add.page_content)
                print(f"{COLOR_GREEN}Added targeted exclusion: {ped_exc_domestic_key}{COLOR_END}")

        ped_exc_int_key = "International Exclusion 1) Pre-Existing Diseases (Code -Excl01)"
        if ped_exc_int_key in exclusions_map:
            doc_to_add = exclusions_map[ped_exc_int_key]
            if doc_to_add.page_content not in unique_combined_docs_content:
                additional_docs.append(doc_to_add)
                unique_combined_docs_content.add(doc_to_add.page_content)
                print(f"{COLOR_GREEN}Added targeted exclusion: {ped_exc_int_key}{COLOR_END}")

    # Check for Dental Implants
    if "dental implant" in user_question.lower() or "implants dental" in user_question.lower():
        for exc_key, exc_doc in exclusions_map.items():
            if "dental implant" in exc_doc.page_content.lower() and ("dental plan benefits" in exc_doc.metadata.get('code', '').lower() or "dental plan benefits" in exc_key.lower()):
                if exc_doc.page_content not in unique_combined_docs_content:
                    additional_docs.append(exc_doc)
                    unique_combined_docs_content.add(exc_doc.page_content)
                    print(f"{COLOR_GREEN}Added targeted exclusion: {exc_key}{COLOR_END}")
    
    # Check for cosmetic surgery / rhinoplasty keywords
    if "cosmetic" in user_question.lower() or "rhinoplasty" in user_question.lower() or "plastic surgery" in user_question.lower():
        cosmetic_exc_domestic_key = "Domestic Exclusion 8) Cosmetic or plastic Surgery (Code -Exc108)"
        if cosmetic_exc_domestic_key in exclusions_map:
            doc_to_add = exclusions_map[cosmetic_exc_domestic_key]
            if doc_to_add.page_content not in unique_combined_docs_content:
                additional_docs.append(doc_to_add)
                unique_combined_docs_content.add(doc_to_add.page_content)
                print(f"{COLOR_GREEN}Added targeted exclusion: {cosmetic_exc_domestic_key}{COLOR_END}")
        
        cosmetic_exc_int_key = "International Exclusion 8) Cosmetic or plastic Surgery (Code -Exc108)"
        if cosmetic_exc_int_key in exclusions_map:
            doc_to_add = exclusions_map[cosmetic_exc_int_key]
            if doc_to_add.page_content not in unique_combined_docs_content:
                additional_docs.append(doc_to_add)
                unique_combined_docs_content.add(doc_to_add.page_content)
                print(f"{COLOR_GREEN}Added targeted exclusion: {cosmetic_exc_int_key}{COLOR_END}")

    # Check for experimental / unproven treatment keywords
    if "experimental" in user_question.lower() or "unproven" in user_question.lower() or "gene therapy" in user_question.lower():
        unproven_def_key = "45. Unproven/Experimental Treatment"
        if unproven_def_key in definitions_map:
            doc_to_add = definitions_map[unproven_def_key]
            if doc_to_add.page_content not in unique_combined_docs_content:
                additional_docs.append(doc_to_add)
                unique_combined_docs_content.add(doc_to_add.page_content)
                print(f"{COLOR_GREEN}Added targeted definition: {unproven_def_key}{COLOR_END}")

        unproven_exc_domestic_key = "Domestic Exclusion 16) Unproven Treatments (Code -Excl16)"
        if unproven_exc_domestic_key in exclusions_map:
            doc_to_add = exclusions_map[unproven_exc_domestic_key]
            if doc_to_add.page_content not in unique_combined_docs_content:
                additional_docs.append(doc_to_add)
                unique_combined_docs_content.add(doc_to_add.page_content)
                print(f"{COLOR_GREEN}Added targeted exclusion: {unproven_exc_domestic_key}{COLOR_END}")
        
        unproven_exc_int_key = "International Exclusion 16) Unproven Treatments (Code -Excl16)"
        if unproven_exc_int_key in exclusions_map:
            doc_to_add = exclusions_map[unproven_exc_int_key]
            if doc_to_add.page_content not in unique_combined_docs_content:
                additional_docs.append(doc_to_add)
                unique_combined_docs_content.add(doc_to_add.page_content)
                print(f"{COLOR_GREEN}Added targeted exclusion: {unproven_exc_int_key}{COLOR_END}")


    # Add the additional, targeted documents to the front of the list for prioritization
    final_docs_for_llm = additional_docs + final_docs_for_llm

    # Trim if too many documents are combined (e.g., more than 15-20 for effective LLM processing)
    if len(final_docs_for_llm) > 15: # Keeping max 15 documents for the LLM context
        final_docs_for_llm = final_docs_for_llm[:15] 
        print(f"{COLOR_BLUE}Trimmed context to {len(final_docs_for_llm)} chunks to manage token limits.{COLOR_END}")


    print(f"\n{COLOR_BLUE}🤖 Generating answer for query: '{user_question}'...{COLOR_END}")
    chain = get_conversational_chain()

    raw_llm_response = chain.invoke({
        "context": final_docs_for_llm,
        "question": user_question
    })

    print(f"\n{SECTION_SEP}\n{COLOR_BLUE}--- LLM Raw Response (before JSON parsing attempt) ---{COLOR_END}")
    print(raw_llm_response)

    cleaned_response = raw_llm_response.strip()
    if cleaned_response.startswith("```json") and cleaned_response.endswith("```"):
        cleaned_response = cleaned_response[len("```json"): -len("```")].strip()
    elif cleaned_response.startswith("```") and cleaned_response.endswith("```"):
        cleaned_response = cleaned_response[len("```"): -len("```")].strip()
    if cleaned_response.lower().startswith("answer:"):
        cleaned_response = cleaned_response[len("answer:"):].strip()

    try:
        parsed_response = json.loads(cleaned_response)

        final_relevant_clauses = []
        if "RelevantClauses" in parsed_response and isinstance(parsed_response["RelevantClauses"], list):
            for llm_clause in parsed_response["RelevantClauses"]:
                found_citation = False
                for doc_chunk in final_docs_for_llm:
                    # Check if the LLM's cited clause is directly contained in a chunk or matches metadata
                    if (llm_clause.lower() in doc_chunk.page_content.lower() or 
                        llm_clause.strip() == doc_chunk.metadata.get('name', '').strip() or 
                        llm_clause.strip() == doc_chunk.metadata.get('code', '').strip()):
                        
                        # Only add if not already in final_relevant_clauses to avoid duplicates
                        full_clause_text = f"{llm_clause.strip()} [Source: {doc_chunk.metadata.get('source', 'Unknown Page')}]"
                        if full_clause_text not in final_relevant_clauses:
                            final_relevant_clauses.append(full_clause_text)
                        found_citation = True
                        break
                if not found_citation:
                    # If LLM cited something not found in chunks, add it as is (might be hallucinated or summary)
                    if llm_clause not in final_relevant_clauses: # Avoid adding direct duplicates if already added by exact match
                        final_relevant_clauses.append(llm_clause)

        parsed_response["RelevantClauses"] = final_relevant_clauses

        print(f"\n{SECTION_SEP}\n{COLOR_GREEN}📝 Structured Reply (JSON):{COLOR_END}\n")
        print(json.dumps(parsed_response, indent=4))

    except json.JSONDecodeError as e:
        print(f"\n{SECTION_SEP}\n{COLOR_RED}❌ Error parsing LLM response as JSON: {e}{COLOR_END}")
        print(f"{COLOR_RED}Cleaned LLM response that failed parsing:{COLOR_END}")
        print(cleaned_response)
        print(f"{COLOR_RED}Raw LLM response was:{COLOR_END}")
        print(raw_llm_response)

    print(f"\n{SECTION_SEP}\n{COLOR_BLUE}--- Retrieved Document Chunks (Context Provided to LLM) ---{COLOR_END}")
    for i, doc in enumerate(final_docs_for_llm):
        print(f"\n{SUBSECTION_SEP}\nChunk {i+1} from {doc.metadata.get('source', 'Unknown Source')}, Type: {doc.metadata.get('type', 'chunk')}:")
        print(doc.page_content)
        print(SUBSECTION_SEP)
    print(f"\n{LINE_SEP}\n")


# === MAIN ===
if __name__ == "__main__":
    # Ensure this PDF file exists at this exact path on your Windows system.
    file_path = "C:\\Users\\psuma\\Downloads\\BAJHLIP23020V012223.pdf" 

    if not os.path.exists(file_path):
        print(f"{COLOR_RED}Error: PDF file '{file_path}' not found. Please ensure it's at the specified path.{COLOR_END}")
    else:
        print(f"{LINE_SEP}\n{COLOR_BLUE}--- Starting RAG Model Tests ---{COLOR_END}\n")

        # Test Query 1
        print(f"{LINE_SEP}\n{COLOR_BLUE}--- Testing Query 1: Unregistered Hospital Treatment ---{COLOR_END}")
        user_input("Will my claim be accepted if treatment is taken in an unregistered hospital?", file_path)

        # Test Query 2
        print(f"{LINE_SEP}\n{COLOR_BLUE}--- Testing Query 2: Policy cancellation and refund schedule ---{COLOR_END}")
        user_input("How can I cancel my policy and what is the refund schedule?", file_path)

        # Test Query 3
        print(f"{LINE_SEP}\n{COLOR_BLUE}--- Testing Query 3: Coverage for cosmetic rhinoplasty (targeted retrieval for exclusion) ---{COLOR_END}")
        user_input("Does my policy cover cosmetic rhinoplasty surgery?", file_path)

        # Test Query 4
        print(f"{LINE_SEP}\n{COLOR_BLUE}--- Testing Query 4: Dental implants coverage (targeted retrieval for exclusion) ---{COLOR_END}")
        user_input("Do you cover dental implants?", file_path)

        # Test Query 5
        print(f"{LINE_SEP}\n{COLOR_BLUE}--- Testing Query 5: Coverage for experimental gene therapy (targeted retrieval for exclusion) ---{COLOR_END}")
        user_input("Is experimental gene therapy covered by my policy?", file_path)
        
        # Test Query 6
        print(f"{LINE_SEP}\n{COLOR_BLUE}--- Testing Query 6: Congenital Anomaly Definition ---{COLOR_END}")
        user_input("What qualifies as a 'Congenital Anomaly'?", file_path)

        print(f"{LINE_SEP}\n{COLOR_BLUE}--- All RAG Model Tests Completed. ---{COLOR_END}\n")