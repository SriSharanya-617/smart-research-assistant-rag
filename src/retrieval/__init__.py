"""
Retrieval package for the Smart Research Assistant.
Exposes pipeline orchestrators, query processors, and caches.
"""

from src.retrieval.base import BaseRetriever
from src.retrieval.retriever import SemanticRetriever, MMRRetriever, RetrieverFactory
from src.retrieval.query_processor import QueryProcessor, QueryComplexityAnalyzer
from src.retrieval.evaluators import ConfidenceEstimator, SelectionExplainer
from src.retrieval.cache import RetrievalCache
from src.retrieval.inspector import RetrievalInspector
from src.retrieval.pipeline import RetrievalPipeline
from src.retrieval.exceptions import (
    RetrievalError,
    EmptyQueryError,
    InvalidQueryError,
    VectorStoreUnavailableError
)

__all__ = [
    "BaseRetriever",
    "SemanticRetriever",
    "MMRRetriever",
    "RetrieverFactory",
    "QueryProcessor",
    "QueryComplexityAnalyzer",
    "ConfidenceEstimator",
    "SelectionExplainer",
    "RetrievalCache",
    "RetrievalInspector",
    "RetrievalPipeline",
    "RetrievalError",
    "EmptyQueryError",
    "InvalidQueryError",
    "VectorStoreUnavailableError"
]
