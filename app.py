"""
Main Streamlit UI Orchestrator for the Smart Research Assistant.
Integrates the complete production-grade RAG pipeline (loaders, splitters, vector stores, retrievers, and LLMs).
"""

import os
import time
import datetime
import tempfile
import shutil
import streamlit as st

# Backend Imports
from src.config import get_config
from src.logger import setup_logger
from src.ingestion.pdf_loader import PDFLoader
from src.ingestion.txt_loader import TXTLoader
from src.ingestion.web_loader import WebLoader
from src.ingestion.splitter import DocumentSplitter
from src.embeddings.factory import EmbeddingFactory
from src.vectorstores.manager import VectorStoreFactory
from src.retrieval.pipeline import RetrievalPipeline
from src.llm.factory import LLMFactory
from src.llm.rag_pipeline import RAGPipeline

# Initialize Logger and Config
logger = setup_logger("streamlit_app")
config = get_config()

# 1. Setup Page Configurations with Premium styling
st.set_page_config(
    page_title="Smart Research Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium CSS injection for vibrant dashboard design
st.markdown(
    """
    <style>
    /* Dark-mode premium card styling */
    .dashboard-card {
        background-color: #1E293B;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.25);
        border: 1px solid #334155;
        text-align: center;
        transition: transform 0.2s;
    }
    .dashboard-card:hover {
        transform: translateY(-4px);
    }
    .dashboard-title {
        color: #94A3B8;
        font-size: 0.9rem;
        font-weight: 600;
        text-transform: uppercase;
        margin-bottom: 8px;
    }
    .dashboard-value {
        color: #F8FAFC;
        font-size: 2rem;
        font-weight: 700;
    }
    .stat-highlight {
        color: #38BDF8;
    }
    /* Citation pills styling */
    .citation-container {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-top: 10px;
    }
    .citation-badge {
        background: #0EA5E9;
        color: white;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 500;
        border: 1px solid #0284C7;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# 2. Maintain Session States
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {"role": "assistant", "content": "Welcome to your Smart Research Assistant! Upload research papers, documents, or websites in the sidebar to build your knowledge base, then ask me anything."}
    ]

if "last_rag_metadata" not in st.session_state:
    st.session_state.last_rag_metadata = None

# Temporary directory for file uploads
UPLOAD_DIR = os.path.join("data", "tmp_uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# 3. Sidebar UI Configuration Panel
st.sidebar.title("⚙️ RAG Configuration")

# Embedding Config
st.sidebar.subheader("1. Embedding Settings")
embedding_provider = st.sidebar.selectbox("Provider", ["huggingface", "mock"], index=0)
embedding_model = st.sidebar.selectbox("Model", ["BAAI/bge-small-en-v1.5", "all-MiniLM-L6-v2"], index=0)

# Vector Store Config
st.sidebar.subheader("2. Vector Database Settings")
vector_store_type = st.sidebar.selectbox("Vector Database", ["chroma", "faiss"], index=0)

# LLM Provider settings
st.sidebar.subheader("3. LLM Model Settings")
llm_provider = st.sidebar.selectbox("LLM Provider", ["gemini", "openai", "ollama", "mock"], index=0)

model_options = {
    "gemini": ["gemini-1.5-flash", "gemini-1.5-pro"],
    "openai": ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
    "ollama": ["llama3", "mistral", "phi3"],
    "mock": ["mock-model"]
}
llm_model = st.sidebar.selectbox("LLM Model", model_options.get(llm_provider, ["mock-model"]))
temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.0, step=0.1)

# Ingestion Parameters
st.sidebar.subheader("4. Chunking Parameters")
chunk_size = st.sidebar.slider("Chunk Size", min_value=200, max_value=2000, value=1000, step=100)
chunk_overlap = st.sidebar.slider("Chunk Overlap", min_value=0, max_value=500, value=200, step=50)

# Retrieval Strategy
st.sidebar.subheader("5. Retrieval Parameters")
retrieval_strategy = st.sidebar.selectbox("Retrieval Strategy", ["semantic", "mmr"], index=0)
top_k = st.sidebar.slider("Top-K Chunks", min_value=1, max_value=10, value=4, step=1)
score_threshold = st.sidebar.slider("Distance Threshold (L2)", min_value=0.0, max_value=3.0, value=1.5, step=0.1)

# API Keys management
st.sidebar.subheader("6. API Authorization Keys")
google_key = st.sidebar.text_input("Gemini API Key", value=os.environ.get("GOOGLE_API_KEY") or "", type="password")
openai_key = st.sidebar.text_input("OpenAI API Key", value=os.environ.get("OPENAI_API_KEY") or "", type="password")

if google_key:
    os.environ["GOOGLE_API_KEY"] = google_key
if openai_key:
    os.environ["OPENAI_API_KEY"] = openai_key

# 4. Initialize Core Backend Components
@st.cache_resource
def get_backend_resources(
    _emb_provider, _emb_model, _vs_type, _llm_provider, _llm_model, _temp
):
    """
    Constructs and caches backend factories. Swapping settings triggers rebuild.
    """
    try:
        embeddings = EmbeddingFactory.get_embeddings(
            provider=_emb_provider,
            model_name=_emb_model
        )
        
        vector_store = VectorStoreFactory.get_vector_store(
            store_type=_vs_type,
            embeddings=embeddings
        )
        
        # Instantiate Retriever Pipeline
        retriever_pipeline = RetrievalPipeline(
            vector_store=vector_store,
            default_strategy=retrieval_strategy
        )
        
        # Instantiate LLM Provider
        llm_provider_inst = LLMFactory.get_llm_provider(
            provider=_llm_provider,
            model_name=_llm_model,
            temperature=_temp
        )
        
        # Instantiate RAG Pipeline
        rag_pipeline = RAGPipeline(
            retrieval_pipeline=retriever_pipeline,
            llm_provider=llm_provider_inst
        )
        
        return embeddings, vector_store, retriever_pipeline, rag_pipeline
    except Exception as ex:
        st.error(f"Failed to load backend subsystems: {ex}")
        logger.error(f"Subsystems load error: {ex}")
        return None, None, None, None

embeddings_inst, vector_store_inst, retriever_pipeline_inst, rag_pipeline_inst = get_backend_resources(
    embedding_provider, embedding_model, vector_store_type, llm_provider, llm_model, temperature
)

# 5. Document Ingestion sidebar UI
st.sidebar.markdown("---")
st.sidebar.subheader("📤 Ingestion Source")
uploaded_files = st.sidebar.file_uploader("Upload files (PDF, TXT)", type=["pdf", "txt"], accept_multiple_files=True)
website_url = st.sidebar.text_input("Website URL", placeholder="https://example.com")

process_btn = st.sidebar.button("⚡ Process & Index", use_container_width=True)
reset_db_btn = st.sidebar.button("🗑️ Reset Database Index", use_container_width=True)

# Reset DB execution
if reset_db_btn and vector_store_inst:
    try:
        vector_store_inst.reset()
        if rag_pipeline_inst:
            rag_pipeline_inst.clear_memory()
            rag_pipeline_inst.retrieval_pipeline.invalidate_cache()
        st.sidebar.success("Vector Store index reset successfully!")
        st.rerun()
    except Exception as e:
        st.sidebar.error(f"Reset failed: {e}")

# Process documents execution
if process_btn:
    has_files = uploaded_files is not None and len(uploaded_files) > 0
    has_url = bool(website_url.strip())
    
    if not has_files and not has_url:
        st.sidebar.error("Please upload files or specify a website URL first.")
    else:
        # Progress metrics
        progress_bar = st.sidebar.progress(0)
        status_text = st.sidebar.empty()
        
        try:
            documents_to_split = []
            
            # Step 1: Loading
            if has_files:
                for idx, file in enumerate(uploaded_files):
                    status_text.text(f"📁 Loading {file.name}...")
                    progress_bar.progress(int((idx / len(uploaded_files)) * 40))
                    
                    # Save to temp path
                    temp_path = os.path.join(UPLOAD_DIR, file.name)
                    with open(temp_path, "wb") as f:
                        f.write(file.read())
                        
                    # Load depending on extension
                    if file.name.lower().endswith(".pdf"):
                        loader = PDFLoader(temp_path)
                    else:
                        loader = TXTLoader(temp_path)
                        
                    loaded_docs = loader.load()
                    documents_to_split.extend(loaded_docs)
                    
                    # Cleanup temp file
                    try:
                        os.remove(temp_path)
                    except:
                        pass

            if has_url:
                status_text.text("🌐 Fetching web content...")
                progress_bar.progress(50)
                web_loader = WebLoader(website_url)
                loaded_web = web_loader.load()
                documents_to_split.extend(loaded_web)
                
            # Step 2: Splitting
            status_text.text("✂️ Splitting document texts...")
            progress_bar.progress(70)
            splitter = DocumentSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            chunks = splitter.split_documents(documents_to_split)
            
            # Step 3: Embed & Index
            status_text.text(f"💾 Indexing {len(chunks)} text chunks...")
            progress_bar.progress(90)
            
            vector_store_inst.add_documents(chunks)
            if rag_pipeline_inst:
                # Invalidate retrieval cache on updates
                rag_pipeline_inst.retrieval_pipeline.invalidate_cache()
                
            progress_bar.progress(100)
            status_text.empty()
            progress_bar.empty()
            
            st.sidebar.success(f"Successfully indexed {len(chunks)} chunks!")
            time.sleep(1.0)
            st.rerun()
            
        except Exception as e:
            status_text.empty()
            progress_bar.empty()
            st.sidebar.error(f"Ingestion failed: {e}")
            logger.error(f"Ingestion runtime exception: {e}")

# 6. Fetch Vector Store details dynamically to populate stats
total_chunks_indexed = 0
unique_documents = {}

if vector_store_inst and vector_store_inst.db is not None:
    try:
        # Inspect database items depending on engine
        if vector_store_type == "chroma":
            all_meta = vector_store_inst.db._collection.get(include=["metadatas"])
            metadatas = all_meta.get("metadatas") or []
        else:
            metadatas = [doc.metadata for doc in vector_store_inst.db.docstore._dict.values()]
            
        total_chunks_indexed = len(metadatas)
        for meta in metadatas:
            filename = meta.get("filename") or meta.get("source") or "Unknown"
            doc_type = meta.get("document_type") or "unknown"
            doc_id = meta.get("document_id", "unknown")
            
            if filename not in unique_documents:
                unique_documents[filename] = {
                    "chunks": 0,
                    "type": doc_type,
                    "document_id": doc_id
                }
            unique_documents[filename]["chunks"] += 1
    except Exception as e:
        logger.warning(f"Failed to inspect vector store items: {e}")

# 7. Main Dashboard layout
st.title("🤖 Smart Research Assistant - Knowledge System")

# Top stats KPI Cards
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(
        f'<div class="dashboard-card"><div class="dashboard-title">Documents Loaded</div>'
        f'<div class="dashboard-value">{len(unique_documents)}</div></div>',
        unsafe_allow_html=True
    )
with col2:
    st.markdown(
        f'<div class="dashboard-card"><div class="dashboard-title">Total Text Chunks</div>'
        f'<div class="dashboard-value">{total_chunks_indexed}</div></div>',
        unsafe_allow_html=True
    )
with col3:
    latency_stat = st.session_state.last_rag_metadata.get("latency_seconds", 0.0) if st.session_state.last_rag_metadata else 0.0
    st.markdown(
        f'<div class="dashboard-card"><div class="dashboard-title">Last Query Latency</div>'
        f'<div class="dashboard-value stat-highlight">{latency_stat:.2f}s</div></div>',
        unsafe_allow_html=True
    )
with col4:
    conf_stat = st.session_state.last_rag_metadata.get("confidence_estimate", "N/A") if st.session_state.last_rag_metadata else "N/A"
    st.markdown(
        f'<div class="dashboard-card"><div class="dashboard-title">Retrieval Confidence</div>'
        f'<div class="dashboard-value">{conf_stat}</div></div>',
        unsafe_allow_html=True
    )

st.write("") # Spacer

# Main Area split: left col is Chat, right col is Document list + Retrieval Inspector
col_left, col_right = st.columns([0.6, 0.4])

with col_left:
    st.subheader("💬 Q&A Chatroom Console")
    
    # Render chat bubble history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
    # Capture user input
    user_query = st.chat_input("Ask a question about the indexed context files...")
    
    if user_query:
        # Display user bubble instantly
        with st.chat_message("user"):
            st.markdown(user_query)
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        
        # Generate Answer
        if rag_pipeline_inst:
            with st.chat_message("assistant"):
                with st.spinner("Retrieving facts and synthesizing response..."):
                    try:
                        # Call RAG Pipeline
                        response = rag_pipeline_inst.answer_question(
                            query=user_query,
                            limit=top_k,
                            strategy=retrieval_strategy,
                            score_threshold=score_threshold
                        )
                        st.markdown(response)
                        
                        st.session_state.chat_history.append({"role": "assistant", "content": response})
                        st.session_state.last_rag_metadata = rag_pipeline_inst.get_last_run_metadata()
                        
                        time.sleep(0.1)
                        st.rerun()
                    except Exception as ex:
                        st.error(f"Generation failure: {ex}")
                        logger.error(f"RAG Pipeline execution failed: {ex}")
        else:
            st.error("RAG pipeline is not loaded. Configure settings in the sidebar.")

with col_right:
    # 1. Document Management Panel
    st.subheader("📚 Managed Index Documents")
    if not unique_documents:
        st.info("No documents are currently indexed in the vector database.")
    else:
        for fname, info in unique_documents.items():
            col_doc, col_del = st.columns([0.8, 0.2])
            with col_doc:
                st.markdown(f"📄 **{fname}** ({info['chunks']} chunks | `{info['type']}`) ")
            with col_del:
                # Delete Document
                if st.button("🗑️", key=f"del_{fname}"):
                    try:
                        with st.spinner("Deleting document from index..."):
                            # Delete by checking chunk filename matches
                            if vector_store_type == "chroma":
                                # Query collection keys
                                all_items = vector_store_inst.db._collection.get(include=["metadatas"])
                                metadatas = all_items.get("metadatas") or []
                                ids = all_items.get("ids") or []
                                chunk_ids_to_del = [
                                    ids[i] for i, m in enumerate(metadatas)
                                    if m.get("filename") == fname or m.get("source") == fname
                                ]
                            else:
                                chunk_ids_to_del = [
                                    cid for cid, doc in vector_store_inst.db.docstore._dict.items()
                                    if doc.metadata.get("filename") == fname or doc.metadata.get("source") == fname
                                ]
                                
                            if chunk_ids_to_del:
                                vector_store_inst.remove_documents(chunk_ids_to_del)
                                if rag_pipeline_inst:
                                    rag_pipeline_inst.retrieval_pipeline.invalidate_cache()
                                st.success(f"Deleted {fname} successfully!")
                                time.sleep(0.5)
                                st.rerun()
                            else:
                                st.error("Could not find matching document IDs in index.")
                    except Exception as ex:
                        st.error(f"Deletion failed: {ex}")

    st.write("") # Spacer

    # 2. Retrieval Inspector Dashboard
    st.subheader("🔍 Diagnostics & Retrieval Inspector")
    inspector = st.session_state.last_rag_metadata.get("retrieval_inspector") if st.session_state.last_rag_metadata else None
    
    with st.expander("Expand Retrieval Metrics Inspector Console", expanded=True):
        if not inspector:
            st.info("Ask a question in the chat to view retrieval analytics details.")
        else:
            st.markdown(f"**Original Query:** `{inspector.get('original_query')}`")
            st.markdown(f"**Processed Query:** `{inspector.get('processed_query')}`")
            st.markdown(f"**Query Complexity (Intent):** `{inspector.get('query_complexity')}`")
            st.markdown(f"**Retrieval Strategy:** `{inspector.get('retrieval_strategy')}`")
            st.markdown(f"**Cache Hit:** `{inspector.get('cache_hit')}`")
            st.markdown(f"**Latency:** `{inspector.get('latency_seconds', 0.0):.4f} seconds`")
            st.markdown(f"**Confidence Est:** `{inspector.get('confidence_estimate')}`")
            
            st.markdown("---")
            st.markdown("**Retrieved Chunk Extracts:**")
            
            chunks = inspector.get("retrieved_chunks") or []
            for i, chunk in enumerate(chunks):
                meta = chunk.get("metadata") or {}
                source_lbl = meta.get("filename") or meta.get("source") or "unknown"
                page_lbl = f", Pg. {meta.get('page_number')}" if meta.get("page_number") else ""
                
                st.markdown(f"📌 **Chunk #{i+1}** (Source: *{source_lbl}{page_lbl}* | Score: `{chunk.get('score', 0.0):.4f}`)")
                st.code(chunk.get("content"), language="text")
                st.caption(f"*Selection Explanation:* {chunk.get('explanation')}")
