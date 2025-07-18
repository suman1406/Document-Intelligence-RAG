# app.py

import streamlit as st
import os
import tempfile
# Import the functions from your backend file
from rag_model import create_rag_retriever, process_claim_query

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="Insurance Policy RAG Assistant",
    page_icon="📄",
    layout="wide"
)

st.title("📄 Insurance Policy RAG Assistant")
st.write("Upload an insurance policy document (PDF, DOCX, or EML) and ask questions to get structured, justified answers.")

# --- State Management ---
if 'retriever' not in st.session_state:
    st.session_state.retriever = None
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'uploaded_filename' not in st.session_state:
    st.session_state.uploaded_filename = None

# --- File Uploader and Pipeline Creation ---
uploaded_file = st.file_uploader(
    "Upload your policy document",
    type=['pdf', 'docx', 'eml', 'msg']
)

if uploaded_file is not None:
    if st.session_state.uploaded_filename != uploaded_file.name:
        st.info(f"New file uploaded: **{uploaded_file.name}**. Processing...")
        st.session_state.uploaded_filename = uploaded_file.name
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        with st.spinner("Building RAG pipeline... This may take a moment."):
            retriever, vectorstore = create_rag_retriever(tmp_file_path)
            st.session_state.retriever = retriever
            st.session_state.vectorstore = vectorstore
        
        os.unlink(tmp_file_path)
        st.success("RAG pipeline is ready. You can now ask questions.")

# --- Function to display results cleanly ---
def display_results(result):
    """Formats and displays the RAG output in a user-friendly way."""
    st.subheader("Analysis Result")

    # Display the summary first
    summary = result.get("summary")
    if summary:
        st.markdown(f"> **Summary:** {summary}")
    
    decision = result.get("decision", "Error")

    # Display the main decision in a colored box
    if decision == "Approved":
        st.success(f"**Decision: {decision}**")
    elif decision == "Denied":
        st.error(f"**Decision: {decision}**")
    elif decision == "Further Information Required":
        st.warning(f"**Decision: {decision}**")
    else:
        st.error(f"**An Error Occurred:** {result.get('justification', 'No details provided.')}")
        return

    # Display the approved amount, if available
    amount = result.get("amount")
    if amount is not None:
        st.metric(label="Approved Amount", value=f"${amount:,.2f}")

    # **UI CHANGE: Moved Justification outside the expander**
    st.markdown("**Justification:**")
    st.info(result.get('justification', 'No justification provided.'))

    # Display grounding check result
    grounding_check = result.get("grounding_check", "Unknown")
    if grounding_check.lower() == 'yes':
        st.markdown("✅ **Grounded in Document:** The justification is supported by the retrieved text.")
    else:
        st.markdown("⚠️ **Potentially Ungrounded:** The justification may not be fully supported by the retrieved text.")

    # Display details in an expander
    with st.expander("Show Detailed Sources and Retrieved Context"):
        references = result.get("clause_references")
        if references:
            st.markdown("**Clause References:**")
            for ref in references:
                st.markdown(f"- `{ref}`")
        
        retrieved_context = result.get("retrieved_context")
        if retrieved_context:
            st.markdown("**Retrieved Context Used for Answer:**")
            for doc in retrieved_context:
                st.markdown(f"> {doc.page_content}")
                st.markdown(f"_(Source: {doc.metadata.get('source', 'N/A')}, Page: {doc.metadata.get('page', 'N/A')})_")
                st.markdown("---")

# --- Query Input and Processing ---
if st.session_state.retriever:
    st.write("---")
    query = st.text_input(
        "Ask a question about the policy:",
        placeholder="e.g., Am I covered for knee replacement surgery?"
    )

    if st.button("Get Answer"):
        if query:
            with st.spinner("Searching documents and generating answer..."):
                result = process_claim_query(
                    query=query,
                    retriever=st.session_state.retriever,
                    vectorstore=st.session_state.vectorstore
                )
                
            display_results(result)
        else:
            st.warning("Please enter a question.")
else:
    st.info("Please upload a document to begin.")
