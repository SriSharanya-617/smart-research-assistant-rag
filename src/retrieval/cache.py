"""
TTL-aware caching layer for query retrieval results.
Includes cache hit/miss analytics and cache invalidation.
"""

import time
from typing import Dict, Any, List, Tuple, Optional
from src.ingestion.base import Document
from src.logger import setup_logger

logger = setup_logger("retrieval_cache")

class RetrievalCache:
    """
    Caches document search results keyed by query + filters + strategy.
    Optimizes CPU/GPU load for identical repeat search queries.
    """
    def __init__(self, ttl_seconds: Optional[int] = 300):
        self.ttl_seconds = ttl_seconds
        # Structure: { key: (timestamp, results_list) }
        self._cache: Dict[Tuple[str, str, str], Tuple[float, List[Tuple[Document, float]]]] = {}
        
        # Stats counters
        self._hits = 0
        self._misses = 0

    def _generate_key(
        self,
        query: str,
        strategy: str,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> Tuple[str, str, str]:
        """
        Creates a hashable tuple representation of retrieval input.
        """
        filter_str = str(sorted(filter_dict.items())) if filter_dict else ""
        return (query.strip(), strategy.strip().lower(), filter_str)

    def get(
        self,
        query: str,
        strategy: str,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> Optional[List[Tuple[Document, float]]]:
        """
        Fetches query results from cache if within TTL bounds.
        """
        key = self._generate_key(query, strategy, filter_dict)
        
        if key not in self._cache:
            self._misses += 1
            return None

        timestamp, cached_results = self._cache[key]
        
        # Check TTL expiration
        if self.ttl_seconds is not None:
            age = time.time() - timestamp
            if age > self.ttl_seconds:
                logger.info(f"Cache expired for query '{query[:30]}...' (Age: {age:.1f}s). Evicting.")
                del self._cache[key]
                self._misses += 1
                return None

        self._hits += 1
        logger.info(f"Cache Hit for query '{query[:30]}...'. Retained {len(cached_results)} documents.")
        return cached_results

    def set(
        self,
        query: str,
        strategy: str,
        results: List[Tuple[Document, float]],
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Caches a query result with a current timestamp.
        """
        key = self._generate_key(query, strategy, filter_dict)
        self._cache[key] = (time.time(), results)
        logger.info(f"Cached results for query '{query[:30]}...' (strategy={strategy}).")

    def invalidate(self) -> None:
        """
        Clears cached entries. Call this when documents are modified/added.
        """
        logger.info("Invalidating all retrieval cache entries due to index update.")
        self._cache.clear()

    def get_statistics(self) -> Dict[str, Any]:
        """
        Exposes hit and miss analytics metrics.
        """
        total = self._hits + self._misses
        hit_ratio = (self._hits / total) if total > 0 else 0.0
        return {
            "cache_hits": self._hits,
            "cache_misses": self._misses,
            "cache_hit_ratio": hit_ratio,
            "cache_size": len(self._cache)
        }
