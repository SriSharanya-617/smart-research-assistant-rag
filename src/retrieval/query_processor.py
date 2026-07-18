"""
Query processing, validation, and complexity analysis module.
"""

import re
import unicodedata
from typing import Dict, Any, Optional
from src.retrieval.exceptions import EmptyQueryError, InvalidQueryError
from src.logger import setup_logger

logger = setup_logger("query_processor")

class QueryProcessor:
    """
    Cleans, validates, and prepares query strings for embedding retrieval.
    Preserves original queries for debugging.
    """
    def __init__(self, max_length: int = 1000):
        self.max_length = max_length

    def clean_query(self, query: str) -> str:
        """
        Normalizes Unicode format, collapses spaces, and strips boundary whitespace.
        """
        if not query:
            return ""
            
        # Normalize Unicode to NFKC
        cleaned = unicodedata.normalize("NFKC", query)
        
        # Collapse multiple spaces and newlines to single space
        cleaned = re.sub(r"\s+", " ", cleaned)
        
        return cleaned.strip()

    def validate_query(self, query: str) -> None:
        """
        Asserts query bounds, raising custom exception on error.
        """
        if not query or not query.strip():
            raise EmptyQueryError("Query cannot be empty or whitespace only.")
            
        if len(query) > self.max_length:
            raise InvalidQueryError(
                f"Query length of {len(query)} characters exceeds maximum allowed limit of {self.max_length}."
            )

    def preprocess(self, query: str) -> Dict[str, str]:
        """
        Cleans and validates, returning original and processed query formats.
        """
        self.validate_query(query)
        processed = self.clean_query(query)
        # Final validation on cleaned query
        if not processed:
            raise EmptyQueryError("Query contains only blank characters after preprocessing.")
            
        return {
            "original_query": query,
            "processed_query": processed
        }


class QueryComplexityAnalyzer:
    """
    Categorizes query intent based on linguistic patterns.
    Helps guide prompt selection for LLM agents.
    """
    
    # Keyword list patterns for intents
    SUMMARY_PATTERNS = [
        r"\bsummariz", r"\bsummary\b", r"\btl;?dr\b", r"\boutline\b", 
        r"\boverview\b", r"\bsynopsis\b", r"\bcore concepts?\b"
    ]
    
    COMPARISON_PATTERNS = [
        r"\bcompar", r"\bcontrast\b", r"\bversus\b", r"\b\bvs\b\b", 
        r"\bdifferenc", r"\bsimilarit", r"\btrade-?offs?\b"
    ]
    
    DEFINITION_PATTERNS = [
        r"\bwhat is\b", r"\bwhat are\b", r"\bdefin", r"\bmeaning of\b", 
        r"\bexplain\b", r"\bconcept of\b"
    ]
    
    EXTRACTION_PATTERNS = [
        r"\blist\b", r"\bextract\b", r"\bparameters?\b", r"\bdetails of\b", 
        r"\bspecif", r"\bdata points?\b", r"\bnumbers?\b"
    ]

    @staticmethod
    def analyze_complexity(query: str) -> str:
        """
        Detects query type: comparison, summary, definition, information extraction, or question.
        """
        query_lower = query.lower()

        # Check in order of specificity
        for pattern in QueryComplexityAnalyzer.COMPARISON_PATTERNS:
            if re.search(pattern, query_lower):
                return "comparison request"

        for pattern in QueryComplexityAnalyzer.SUMMARY_PATTERNS:
            if re.search(pattern, query_lower):
                return "summary request"

        for pattern in QueryComplexityAnalyzer.DEFINITION_PATTERNS:
            if re.search(pattern, query_lower):
                return "definition request"

        for pattern in QueryComplexityAnalyzer.EXTRACTION_PATTERNS:
            if re.search(pattern, query_lower):
                return "information extraction request"

        return "question"
