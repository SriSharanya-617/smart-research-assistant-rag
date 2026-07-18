"""
Modular components for the Streamlit dashboard layout.
Contains modular rendering methods for chat, dashboard, metrics, settings, and logs.
"""

from typing import Dict, Any, List
import streamlit as st

def render_dashboard_cards(stats: Dict[str, Any]) -> None:
    """
    Renders top-level KPI dashboard metrics cards using custom HTML styling.
    """
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(
            f"""
            <div class="dashboard-card">
                <div class="dashboard-title">Documents Loaded</div>
                <div class="dashboard-value">{stats.get("total_docs", 0)}</div>
            </div>
            """, 
            unsafe_allow_html=True
        )
        
    with col2:
        st.markdown(
            f"""
            <div class="dashboard-card">
                <div class="dashboard-title">Total Text Chunks</div>
                <div class="dashboard-value">{stats.get("total_chunks", 0)}</div>
            </div>
            """, 
            unsafe_allow_html=True
        )
        
    with col3:
        st.markdown(
            f"""
            <div class="dashboard-card">
                <div class="dashboard-title">Avg Response Time</div>
                <div class="dashboard-value">{stats.get("avg_latency_s", 0.0):.2f}s</div>
            </div>
            """, 
            unsafe_allow_html=True
        )
        
    with col4:
        st.markdown(
            f"""
            <div class="dashboard-card">
                <div class="dashboard-title">RAG Quality Index</div>
                <div class="dashboard-value stat-highlight">{stats.get("rag_score", 0.0):.1f}%</div>
            </div>
            """, 
            unsafe_allow_html=True
        )


def render_sidebar_controls() -> Dict[str, Any]:
    """
    Renders sidebar settings, sliders, dropdown selectors, and file uploads.
    
    Returns:
        Dict[str, Any]: User-selected parameters dict.
    """
    st.sidebar.header("🛠️ Configuration Center")
    
    # 1. Select RAG Mode
    st.sidebar.subheader("Retrieval Model Configuration")
    db_type = st.sidebar.selectbox("Vector Database", ["ChromaDB", "FAISS"], index=0)
    
    # 2. Select LLM Provider (Unified Config abstraction)
    llm_provider = st.sidebar.selectbox("LLM Provider", ["OpenAI", "Anthropic", "Ollama", "HuggingFace"], index=0)
    
    # 3. Model mapping skeleton
    model_options = {
        "OpenAI": ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
        "Anthropic": ["claude-3-5-sonnet-20240620", "claude-3-haiku-20240307"],
        "Ollama": ["llama3", "mistral", "phi3"],
        "HuggingFace": ["meta-llama/Meta-Llama-3-8B-Instruct", "mistralai/Mistral-7B-Instruct-v0.2"]
    }
    
    llm_model = st.sidebar.selectbox("LLM Model", model_options.get(llm_provider, ["default"]))

    # 4. Settings sliders
    st.sidebar.markdown("---")
    st.sidebar.subheader("Document Ingestion Knobs")
    chunk_size = st.sidebar.slider("Chunk Size (Tokens/Chars)", min_value=200, max_value=2000, value=1000, step=100)
    chunk_overlap = st.sidebar.slider("Chunk Overlap (Tokens/Chars)", min_value=0, max_value=500, value=200, step=50)

    # 5. File Uploader Placeholder
    st.sidebar.markdown("---")
    st.sidebar.subheader("Document Upload")
    uploaded_files = st.sidebar.file_uploader(
        "Upload files for parsing (PDF, TXT)", 
        type=["pdf", "txt"], 
        accept_multiple_files=True
    )
    
    web_url = st.sidebar.text_input("Ingest from Web URL:", placeholder="https://example.com/article")
    
    # 6. Action buttons
    trigger_ingestion = st.sidebar.button("⚡ Process & Index Documents", use_container_width=True)
    clear_database = st.sidebar.button("🗑️ Clear Vector Database", use_container_width=True)

    return {
        "db_type": db_type.lower(),
        "llm_provider": llm_provider.lower(),
        "llm_model": llm_model,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "uploaded_files": uploaded_files,
        "web_url": web_url,
        "trigger_ingestion": trigger_ingestion,
        "clear_database": clear_database
    }


def render_chat_interface(chat_messages: List[Dict[str, str]]) -> str:
    """
    Renders standard chat interface, presenting message bubbles.
    
    Returns:
        str: User query input string.
    """
    st.subheader("💬 Interactive QA Chatroom")
    
    # Render chat container using custom CSS styling classes
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    for msg in chat_messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        
        # Determine CSS class based on sender
        bubble_class = "user" if role == "user" else "assistant"
        avatar = "🧑‍💻" if role == "user" else "🤖"
        
        st.markdown(
            f"""
            <div style="display: flex; flex-direction: column; width: 100%;">
                <div class="chat-bubble {bubble_class}">
                    <strong>{avatar} {role.capitalize()}:</strong><br/>
                    {content}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    st.markdown('</div>', unsafe_allow_html=True)
    st.write("") # Spacer

    # Simple chat input
    user_query = st.chat_input("Ask a question about the indexed research papers/files...")
    return user_query or ""


def render_citations(citations: List[Dict[str, Any]]) -> None:
    """
    Displays nice citation pills/badges for source documentation mapping.
    """
    if not citations:
        return
        
    st.markdown("##### Source Citations:")
    st.markdown('<div class="citation-container">', unsafe_allow_html=True)
    
    cols = st.columns(len(citations) if len(citations) < 6 else 6)
    for idx, cite in enumerate(citations):
        col_idx = idx % len(cols)
        source_name = cite.get("file_name", cite.get("source", "Unknown"))
        page_num = cite.get("page", "")
        page_str = f" (Pg. {page_num})" if page_num else ""
        
        with cols[col_idx]:
            st.markdown(
                f'<span class="citation-badge" title="{cite.get("source", "")}">'
                f'📄 {source_name}{page_str}'
                f'</span>',
                unsafe_allow_html=True
            )
            
    st.markdown('</div>', unsafe_allow_html=True)
    st.write("")


def render_context_viewer(retrieved_contexts: List[Dict[str, Any]]) -> None:
    """
    Renders retrieved document snippets inside collapsible expanders.
    """
    with st.expander("🔍 Inspect Retrieved Context / Chunks Matches"):
        if not retrieved_contexts:
            st.info("No matching contexts fetched for the latest question.")
            return

        for idx, context in enumerate(retrieved_contexts):
            content = context.get("content", "")
            meta = context.get("metadata", {})
            source = meta.get("file_name", meta.get("source", "Unknown Source"))
            page = f" | Page {meta.get('page')}" if meta.get("page") else ""
            chunk_idx = f" | Chunk {meta.get('chunk_index')}" if "chunk_index" in meta else ""
            
            st.markdown(f"**Snippet #{idx+1}** - *Source: {source}{page}{chunk_idx}*")
            st.code(content, language="text")


def render_evaluation_panel(eval_metrics: Dict[str, float]) -> None:
    """
    Renders RAG verification metrics panel.
    """
    st.subheader("📊 RAG Pipeline Validation Evaluation")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        val = eval_metrics.get("faithfulness", 0.0)
        st.metric(
            label="Faithfulness (Groundedness)", 
            value=f"{val * 100:.1f}%", 
            help="Measures if the response is fully grounded in the retrieved sources, without hallucinations."
        )
        st.progress(val)
        
    with col2:
        val = eval_metrics.get("answer_relevance", 0.0)
        st.metric(
            label="Answer Relevance", 
            value=f"{val * 100:.1f}%",
            help="Measures if the generated answer directly addresses the user query."
        )
        st.progress(val)
        
    with col3:
        val = eval_metrics.get("context_recall", 0.0)
        st.metric(
            label="Context Recall", 
            value=f"{val * 100:.1f}%",
            help="Measures if the database retrieval successfully loaded all needed segments."
        )
        st.progress(val)


def render_statistics_panel(stats: Dict[str, Any]) -> None:
    """
    Renders general system status details.
    """
    st.subheader("🖥️ Local Database Status")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            f"""
            - **Vector Store Type**: `{stats.get("store_type", "Chroma")}`
            - **Current Embeddings Model**: `{stats.get("embeddings_model", "Hugging Face (all-MiniLM-L6-v2)")}`
            - **Configured LLM Model**: `{stats.get("llm_model", "OpenAI (gpt-4o-mini)")}`
            """
        )
    with col2:
        st.markdown(
            f"""
            - **Max upload size limit**: `{stats.get("max_upload_mb", 20)} MB`
            - **Chunk Overlap settings**: `{stats.get("chunk_overlap", 200)} chars`
            - **Total Tokens Ingested**: `~{stats.get("total_tokens", 0):,}`
            """
        )
