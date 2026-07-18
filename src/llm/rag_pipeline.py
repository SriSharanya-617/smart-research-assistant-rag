"""
RAG Pipeline orchestrating document retrieval, prompts, LLM generation, and citations.
"""

import time
from typing import Dict, Any, List, Optional
from src.ingestion.base import Document
from src.retrieval.pipeline import RetrievalPipeline
from src.llm.base import BaseLLMProvider
from src.llm.utils import PromptBuilder, CitationFormatter, ConversationMemory
from src.logger import setup_logger

logger = setup_logger("rag_pipeline")

class RAGPipeline:
    """
    Retrieval-Augmented Generation (RAG) Orchestrator connecting RAG modules.
    """
    def __init__(
        self,
        retrieval_pipeline: RetrievalPipeline,
        llm_provider: BaseLLMProvider,
        system_prompt: Optional[str] = None,
        memory_enabled: bool = True,
        max_memory_exchanges: int = 5,
        safe_fallback_response: str = "I am sorry, but I could not find any relevant information in the uploaded documents to answer your question."
    ):
        self.retrieval_pipeline = retrieval_pipeline
        self.llm_provider = llm_provider
        self.system_prompt = system_prompt
        self.safe_fallback_response = safe_fallback_response
        
        self.prompt_builder = PromptBuilder()
        self.memory = ConversationMemory(max_exchanges=max_memory_exchanges, enabled=memory_enabled)

        # Diagnostics for debugging
        self.last_run_metadata: Dict[str, Any] = {}

    def set_llm_provider(self, new_provider: BaseLLMProvider) -> None:
        """
        Allows swapping LLM providers at runtime.
        """
        logger.info(f"Switching RAG pipeline LLM provider to: {new_provider.__class__.__name__}")
        self.llm_provider = new_provider

    def answer_question(
        self,
        query: str,
        limit: int = 4,
        strategy: Optional[str] = None,
        filter_dict: Optional[Dict[str, Any]] = None,
        score_threshold: Optional[float] = None
    ) -> str:
        """
        Executes complete RAG pipeline generation flow.
        """
        start_time = time.time()
        
        # 1. Retrieval
        retrieved_docs = self.retrieval_pipeline.retrieve(
            query=query,
            limit=limit,
            strategy=strategy,
            filter_dict=filter_dict,
            score_threshold=score_threshold
        )
        
        # 2. Check for empty context fallback
        if not retrieved_docs:
            logger.info("Retrieval returned zero documents. Returning safe fallback answer.")
            
            # Record diagnostic metadata
            self.last_run_metadata = {
                "query": query,
                "latency_seconds": time.time() - start_time,
                "retrieved_count": 0,
                "cache_hit": False,
                "llm_metadata": self.llm_provider.get_last_metadata(),
                "confidence_estimate": "Low",
                "retrieval_inspector": self.retrieval_pipeline.get_inspector_report()
            }
            return self.safe_fallback_response

        # 3. Retrieve intent classification
        inspector_report = self.retrieval_pipeline.get_inspector_report()
        intent = inspector_report.get("query_complexity", "question") if inspector_report else "question"
        cache_hit = inspector_report.get("cache_hit", False) if inspector_report else False

        # 4. Fetch history and build prompt instructions
        history = self.memory.get_history()
        prompt = self.prompt_builder.build_prompt(
            query=query,
            documents=retrieved_docs,
            intent=intent,
            history_exchanges=history
        )

        # 5. Run LLM generation
        generation_start = time.time()
        answer = self.llm_provider.generate(prompt, system_prompt=self.system_prompt)
        generation_latency = time.time() - generation_start

        # 6. Format Citations
        citations = CitationFormatter.format_citations(retrieved_docs)
        final_answer = answer + citations

        # 7. Update memory history
        self.memory.add_exchange(query, answer)

        # 8. Record metrics
        total_latency = time.time() - start_time
        
        # Merge provider metrics
        llm_metadata = self.llm_provider.get_last_metadata().copy()
        llm_metadata["generation_latency"] = generation_latency

        self.last_run_metadata = {
            "query": query,
            "latency_seconds": total_latency,
            "retrieved_count": len(retrieved_docs),
            "cache_hit": cache_hit,
            "llm_metadata": llm_metadata,
            "confidence_estimate": inspector_report.get("confidence_estimate", "Low") if inspector_report else "Low",
            "retrieval_inspector": inspector_report
        }

        logger.info(f"RAG query completed successfully in {total_latency:.3f}s. Answer size={len(answer)} chars.")
        return final_answer

    def get_last_run_metadata(self) -> Dict[str, Any]:
        """
        Exposes diagnostic metrics of the last execution block.
        """
        return self.last_run_metadata

    def clear_memory(self) -> None:
        """
        Clears conversation history.
        """
        self.memory.clear()
