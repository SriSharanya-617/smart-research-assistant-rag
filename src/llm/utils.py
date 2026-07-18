"""
Utility classes: PromptBuilder, CitationFormatter, and ConversationMemory.
"""

import threading
from typing import Dict, Any, List, Optional
from src.ingestion.base import Document
from src.logger import setup_logger

logger = setup_logger("llm_utils")

# Default Prompt templates
DEFAULT_TEMPLATES = {
    "qa": (
        "Use only the following retrieved documents context to answer the question. "
        "If you do not know the answer based on the context, state clearly that 'I do not know'. "
        "Do not make up facts or extrapolate beyond the context.\n\n"
        "=== RETRIEVED CONTEXT ===\n"
        "{context}\n\n"
        "=== USER QUESTION ===\n"
        "{question}\n\n"
        "Response:"
    ),
    "summarization": (
        "Provide a clear summary of the retrieved documents context below. "
        "Ensure all summary points map directly to the context facts. Do not invent information.\n\n"
        "=== RETRIEVED CONTEXT ===\n"
        "{context}\n\n"
        "=== SUMMARIZE INSTRUCTION ===\n"
        "{question}\n\n"
        "Response:"
    ),
    "comparison": (
        "Compare and contrast the findings or terms presented in the retrieved context below. "
        "Be objective and stick strictly to the context.\n\n"
        "=== RETRIEVED CONTEXT ===\n"
        "{context}\n\n"
        "=== COMPARISON INSTRUCTION ===\n"
        "{question}\n\n"
        "Response:"
    )
}

class PromptBuilder:
    """
    Constructs prompts using templates. Separates system prompts, context, memory, and queries.
    """
    def __init__(self, templates: Optional[Dict[str, str]] = None):
        self.templates = templates or DEFAULT_TEMPLATES

    def build_prompt(
        self,
        query: str,
        documents: List[Document],
        intent: str = "question",
        history_exchanges: Optional[List[str]] = None
    ) -> str:
        """
        Merges query, context documents, and memory history into formatted prompt text.
        """
        # Select target template key
        template_key = "qa"
        if "comparison" in intent:
            template_key = "comparison"
        elif "summary" in intent:
            template_key = "summarization"

        template = self.templates.get(template_key, self.templates["qa"])

        # Format context block
        context_parts = []
        for idx, doc in enumerate(documents):
            source_info = doc.metadata.get("filename") or doc.metadata.get("source") or "Unknown source"
            page_info = f", Page {doc.metadata['page_number']}" if doc.metadata.get("page_number") else ""
            context_parts.append(
                f"[Doc #{idx+1} | Source: {source_info}{page_info}]\n"
                f"{doc.page_content}"
            )
            
        context_str = "\n\n".join(context_parts) if context_parts else "No context available."

        # Add history block if present
        question_str = query
        if history_exchanges:
            history_str = "\n".join(history_exchanges)
            question_str = (
                "=== CONVERSATION HISTORY ===\n"
                f"{history_str}\n\n"
                "=== USER QUESTION ===\n"
                f"{query}"
            )

        return template.format(context=context_str, question=question_str)


class CitationFormatter:
    """
    Aggregates, deduplicates, and formats document sources in markdown.
    """
    @staticmethod
    def format_citations(documents: List[Document]) -> str:
        """
        Creates a deduplicated source list, keeping original retrieval order.
        """
        if not documents:
            return ""

        seen_citations = set()
        citations_list = []

        for doc in documents:
            filename = doc.metadata.get("filename") or "Unknown File"
            page = doc.metadata.get("page_number")
            doc_id = doc.metadata.get("document_id") or "unknown"
            chunk_id = doc.metadata.get("chunk_id") or "unknown"

            page_str = f"Page {page}" if page is not None else "Unknown Page"
            
            # Key for deduplication
            citation_key = (filename, page, doc_id, chunk_id)
            if citation_key in seen_citations:
                continue
                
            seen_citations.add(citation_key)
            
            citations_list.append(
                f"- **{filename}** ({page_str}) | Document ID: {doc_id} | Chunk ID: {chunk_id}"
            )

        if not citations_list:
            return ""
            
        return "\n=== SOURCES & CITATIONS ===\n" + "\n".join(citations_list)


class ConversationMemory:
    """
    Thread-safe buffer storing conversation history exchanges.
    """
    def __init__(self, max_exchanges: int = 5, enabled: bool = True):
        self.max_exchanges = max_exchanges
        self.enabled = enabled
        self.history: List[str] = []
        self._lock = threading.Lock()

    def add_exchange(self, user_message: str, assistant_response: str) -> None:
        """
        Appends user query and assistant response to history.
        """
        if not self.enabled:
            return
            
        with self._lock:
            self.history.append(f"User: {user_message}")
            self.history.append(f"Assistant: {assistant_response}")
            
            # Keep within size limits (each exchange is 2 lines)
            max_lines = self.max_exchanges * 2
            if len(self.history) > max_lines:
                self.history = self.history[-max_lines:]

    def get_history(self) -> List[str]:
        """
        Retrieves history exchanges.
        """
        if not self.enabled:
            return []
        with self._lock:
            return list(self.history)

    def clear(self) -> None:
        """
        Resets memory buffer.
        """
        with self._lock:
            self.history.clear()
            logger.info("Conversation memory cleared.")
