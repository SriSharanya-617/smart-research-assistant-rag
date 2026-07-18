"""
Retrieval Inspector model class to monitor search metrics.
"""

import datetime
from typing import Dict, Any, List, Tuple, Optional
from src.ingestion.base import Document

class RetrievalInspector:
    """
    Exposes full retrieval results, query intent details, and confidence diagnostics.
    Powering debugging consoles.
    """
    def __init__(
        self,
        original_query: str,
        processed_query: str,
        retrieval_strategy: str,
        requested_k: int,
        returned_k: int,
        filters: Optional[Dict[str, Any]],
        results: List[Tuple[Document, float]],
        latency_seconds: float,
        query_complexity: str,
        confidence_estimate: str,
        cache_hit: bool
    ):
        self.timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
        self.original_query = original_query
        self.processed_query = processed_query
        self.retrieval_strategy = retrieval_strategy
        self.requested_k = requested_k
        self.returned_k = returned_k
        self.filters = filters
        self.results = results
        self.latency_seconds = latency_seconds
        self.query_complexity = query_complexity
        self.confidence_estimate = confidence_estimate
        self.cache_hit = cache_hit

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes structural parameters to dictionary.
        """
        # Map documents to serializable structures
        serialized_chunks = []
        for doc, score in self.results:
            serialized_chunks.append({
                "content": doc.page_content,
                "score": score,
                "metadata": doc.metadata,
                "explanation": doc.metadata.get("selection_explanation", "")
            })

        scores = [score for _, score in self.results]
        avg_score = sum(scores) / len(scores) if scores else 0.0
        max_score = max(scores) if scores else 0.0
        min_score = min(scores) if scores else 0.0

        return {
            "retrieval_timestamp": self.timestamp,
            "original_query": self.original_query,
            "processed_query": self.processed_query,
            "retrieval_strategy": self.retrieval_strategy,
            "requested_k": self.requested_k,
            "returned_k": self.returned_k,
            "applied_filters": self.filters,
            "cache_hit": self.cache_hit,
            "query_complexity": self.query_complexity,
            "confidence_estimate": self.confidence_estimate,
            "latency_seconds": self.latency_seconds,
            "retrieved_chunks": serialized_chunks,
            "statistics": {
                "average_score": avg_score,
                "highest_score": max_score,
                "lowest_score": min_score,
                "returned_count": len(self.results)
            }
        }
