"""
Confidence estimation and selection explanations module.
Calculates categorical tiers and formats metadata diagnostic details.
"""

from typing import List, Tuple, Dict, Any
from src.ingestion.base import Document
from src.logger import setup_logger

logger = setup_logger("retrieval_evaluators")

class ConfidenceEstimator:
    """
    Estimates retrieval quality (High, Medium, Low) based on score signals.
    Does NOT depend on LLMs.
    """
    @staticmethod
    def estimate_confidence(scores: List[float], database_type: str) -> str:
        """
        Maps a list of raw similarity scores to a confidence category.
        Handles both distance formats:
          - ChromaDB: L2 Distance (lower is better, e.g. 0.0 is perfect)
          - FAISS / L2 distance: Lower is better
          - Normalized/Cosine similarity: Higher is better (e.g. 1.0 is perfect)
        """
        if not scores:
            return "Low"

        # Determine metric scale based on database type
        # For FAISS and Chroma: smaller values denote higher similarity (L2 distance is default)
        is_distance_based = database_type.lower() in ["chromadb", "faiss"]

        if is_distance_based:
            # Distance: 0.0 to 2.0+ (Cosine distance = 1 - similarity)
            top_score = scores[0]
            avg_score = sum(scores) / len(scores)
            
            # Lower distances = High confidence
            if top_score < 0.4:
                return "High"
            elif top_score < 0.85:
                return "Medium"
            return "Low"
        else:
            # Similarity score: 0.0 to 1.0 (Higher is better)
            top_score = scores[0]
            avg_score = sum(scores) / len(scores)

            if top_score > 0.8:
                return "High"
            elif top_score > 0.5:
                return "Medium"
            return "Low"


class SelectionExplainer:
    """
    Attaches diagnostic annotations to document metadata explaining the rationale behind retrieval selection.
    """
    @staticmethod
    def attach_explanation(
        document: Document,
        score: float,
        rank: int,
        strategy: str,
        database_type: str
    ) -> Document:
        """
        Injects a descriptive selection text string into metadata for debugging.
        """
        # Determine scale description
        is_distance = database_type.lower() in ["chromadb", "faiss"]
        metric_name = "distance" if is_distance else "similarity"
        
        explanation = (
            f"Chunk selected at rank #{rank} using '{strategy}' strategy. "
            f"Matches query with a {metric_name} score of {score:.4f} in the {database_type} index."
        )
        
        # Write to document metadata
        document.metadata["selection_explanation"] = explanation
        document.metadata["similarity_score"] = score
        document.metadata["retrieval_strategy"] = strategy
        
        return document
