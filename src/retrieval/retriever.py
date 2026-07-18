"""
Concrete Retriever implementations (Semantic, MMR) and a strategy factory.
"""

from typing import Dict, Any, List, Tuple, Optional
from src.ingestion.base import Document
from src.vectorstores.base import BaseVectorStore
from src.retrieval.base import BaseRetriever

class SemanticRetriever(BaseRetriever):
    """
    Standard Semantic search retriever returning top-K similar document chunks.
    """
    def retrieve(
        self,
        query: str,
        limit: int = 4,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Document, float]]:
        """
        Queries similarity scores from vector database.
        """
        return self.vector_store.similarity_search_with_score(
            query=query,
            k=limit,
            filter=filter
        )


class MMRRetriever(BaseRetriever):
    """
    Maximal Marginal Relevance (MMR) retriever to retrieve diverse candidate passages.
    """
    def retrieve(
        self,
        query: str,
        limit: int = 4,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Document, float]]:
        """
        Executes MMR search, converting lists to score-augmented tuples (scores are estimated).
        """
        # Fetch diverse documents
        diverse_docs = self.vector_store.max_marginal_relevance_search(
            query=query,
            k=limit,
            filter=filter
        )
        
        # In MMR, scores might not be returned directly. Let's run a similarity score pass
        # to fetch conformed similarity scores for all returned diverse docs
        results = []
        for doc in diverse_docs:
            chunk_id = doc.metadata.get("chunk_id")
            # Score lookup by querying this specific chunk metadata or matching
            score = 1.0
            if chunk_id:
                # Query index for chunk specifically
                single_match = self.vector_store.similarity_search_with_score(
                    query=query,
                    k=1,
                    filter={"chunk_id": chunk_id}
                )
                if single_match:
                    score = single_match[0][1]
            results.append((doc, score))
            
        # Re-sort MMR results by cosine similarity score
        results.sort(key=lambda x: x[1], reverse=False)  # distance-based sorting (lower distance first)
        return results


class RetrieverFactory:
    """
    Constructs concrete retrievers based on strategy string.
    """
    @staticmethod
    def get_retriever(
        strategy: str,
        vector_store: BaseVectorStore
    ) -> BaseRetriever:
        """
        Maps strategy to corresponding class.
        
        Args:
            strategy: 'semantic' or 'mmr'.
            vector_store: Active database backend.
        """
        strategy_clean = strategy.strip().lower()
        if strategy_clean == "mmr":
            return MMRRetriever(vector_store)
        elif strategy_clean == "semantic":
            return SemanticRetriever(vector_store)
        else:
            # Fallback to Semantic
            return SemanticRetriever(vector_store)
