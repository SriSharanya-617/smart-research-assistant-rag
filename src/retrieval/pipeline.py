"""
Main Retrieval Pipeline coordinating processors, caches, retrievers, and inspectors.
"""

import time
from typing import Dict, Any, List, Tuple, Optional
from src.ingestion.base import Document
from src.vectorstores.base import BaseVectorStore
from src.retrieval.exceptions import VectorStoreUnavailableError
from src.retrieval.query_processor import QueryProcessor, QueryComplexityAnalyzer
from src.retrieval.evaluators import ConfidenceEstimator, SelectionExplainer
from src.retrieval.cache import RetrievalCache
from src.retrieval.retriever import RetrieverFactory
from src.retrieval.inspector import RetrievalInspector
from src.logger import setup_logger

logger = setup_logger("retrieval_pipeline")

class RetrievalPipeline:
    """
    Retrieval Orchestrator managing caching, intent mapping, filtering, and metric inspects.
    """
    def __init__(
        self,
        vector_store: BaseVectorStore,
        default_strategy: str = "semantic",
        cache_ttl_seconds: Optional[int] = 300,
        max_query_length: int = 1000
    ):
        self.vector_store = vector_store
        self.default_strategy = default_strategy
        self.query_processor = QueryProcessor(max_length=max_query_length)
        self.cache = RetrievalCache(ttl_seconds=cache_ttl_seconds)

        # Active Inspector reference
        self.last_inspector: Optional[RetrievalInspector] = None

    def retrieve(
        self,
        query: str,
        limit: int = 4,
        strategy: Optional[str] = None,
        filter_dict: Optional[Dict[str, Any]] = None,
        score_threshold: Optional[float] = None
    ) -> List[Document]:
        """
        Executes query retrieval pipeline.
        
        Args:
            query: Input user query.
            limit: Top-K returned documents.
            strategy: 'semantic' or 'mmr'. Defaults to default_strategy.
            filter_dict: Metadata query filters.
            score_threshold: Excludes documents with scores exceeding threshold limit.
            
        Returns:
            List[Document]: Cleaned, filtered, sorted results.
        """
        if self.vector_store is None:
            raise VectorStoreUnavailableError("Cannot execute retrieval: Vector Store is not initialized.")

        start_time = time.time()
        active_strategy = strategy or self.default_strategy
        
        # 1. Query Preprocessing & Intent Analysis
        prep_results = self.query_processor.preprocess(query)
        processed_query = prep_results["processed_query"]
        complexity = QueryComplexityAnalyzer.analyze_complexity(processed_query)

        # 2. Cache Lookup
        cache_hit = True
        candidates = self.cache.get(processed_query, active_strategy, filter_dict)
        
        if candidates is None:
            cache_hit = False
            # Cache miss, execute database search
            retriever = RetrieverFactory.get_retriever(active_strategy, self.vector_store)
            candidates = retriever.retrieve(processed_query, limit=limit, filter=filter_dict)
            
            # Cache the raw retrieved results
            self.cache.set(processed_query, active_strategy, candidates, filter_dict)

        # 3. Apply Score Threshold Filters (Post-filtering)
        # Chroma/FAISS L2 distance: lower distance is better (e.g. 0.0 is perfect).
        # Cosine similarity: higher score is better.
        db_info = self.vector_store.get_index_info()
        db_type = db_info.get("database_type", "unknown")
        
        is_distance_based = db_type.lower() in ["chromadb", "faiss"]
        
        filtered_candidates = []
        for doc, score in candidates:
            if score_threshold is not None:
                if is_distance_based:
                    # Keep if distance is less than threshold
                    if score > score_threshold:
                        logger.debug(f"Doc excluded: L2 distance {score} exceeds threshold {score_threshold}.")
                        continue
                else:
                    # Keep if similarity is greater than threshold
                    if score < score_threshold:
                        logger.debug(f"Doc excluded: Similarity {score} is below threshold {score_threshold}.")
                        continue
            filtered_candidates.append((doc, score))

        # 4. Attach explanations to metadata for debugging
        final_docs = []
        scores_for_confidence = []
        
        for idx, (doc, score) in enumerate(filtered_candidates):
            rank = idx + 1
            # Mutates doc in place
            annotated_doc = SelectionExplainer.attach_explanation(
                document=doc,
                score=score,
                rank=rank,
                strategy=active_strategy,
                database_type=db_type
            )
            final_docs.append(annotated_doc)
            scores_for_confidence.append(score)

        # 5. Estimate Confidence
        confidence = ConfidenceEstimator.estimate_confidence(scores_for_confidence, db_type)
        
        # 6. Build Inspector
        latency = time.time() - start_time
        self.last_inspector = RetrievalInspector(
            original_query=query,
            processed_query=processed_query,
            retrieval_strategy=active_strategy,
            requested_k=limit,
            returned_k=len(final_docs),
            filters=filter_dict,
            results=filtered_candidates,
            latency_seconds=latency,
            query_complexity=complexity,
            confidence_estimate=confidence,
            cache_hit=cache_hit
        )

        logger.info(
            f"Retrieval complete. Intent='{complexity}', Confidence='{confidence}', "
            f"Latency={latency:.3f}s, Cache={cache_hit}."
        )

        return final_docs

    def get_inspector_report(self) -> Optional[Dict[str, Any]]:
        """
        Retrieves the metrics of the last executed search.
        """
        if self.last_inspector is None:
            return None
        return self.last_inspector.to_dict()

    def invalidate_cache(self) -> None:
        """
        Exposes cache invalidation.
        """
        self.cache.invalidate()
