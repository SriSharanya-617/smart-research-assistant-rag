"""
ChromaDB implementation of the BaseVectorStore interface.
Enforces vector validation, updates/deletes, similarity scoring statistics, and MMR.
"""

import os
import time
import shutil
import datetime
from typing import Dict, Any, List, Tuple, Optional
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document as LCDocument
from src.embeddings.base import BaseEmbeddingProvider
from src.ingestion.base import Document
from src.logger import setup_logger
from src.vectorstores.base import BaseVectorStore, validate_vector
from src.vectorstores.exceptions import (
    VectorStoreError,
    IndexLoadError,
    DimensionMismatchError,
    DuplicateVectorError,
    CorruptedIndexError
)

logger = setup_logger("chroma_vector_store")

class ChromaVectorStore(BaseVectorStore):
    """
    ChromaDB integration wrapper conformed to BaseVectorStore interface.
    """
    def __init__(self, embeddings: BaseEmbeddingProvider, persist_directory: str):
        self.embeddings = embeddings
        self.persist_directory = persist_directory
        self.db: Optional[Chroma] = None
        self.dimension: Optional[int] = None
        
        # Get underlying langchain embeddings instance
        if hasattr(embeddings, "lc_embeddings"):
            self.lc_embeddings = embeddings.lc_embeddings
        else:
            self.lc_embeddings = embeddings

        # Metadata tracking
        self.creation_timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
        self.last_updated_timestamp = self.creation_timestamp

        # Search Analytics
        self._total_queries_run = 0
        self._total_search_time = 0.0
        self._highest_score = 0.0
        self._lowest_score = 1.0
        self._score_sum = 0.0
        self._total_documents_searched = 0
        self._total_documents_returned = 0

        # Try auto loading if directory exists
        if os.path.exists(self.persist_directory):
            try:
                self.load(self.persist_directory)
            except Exception as e:
                logger.warning(f"Auto-load failed on init: {e}. Index will be created on document insertion.")

    def _ensure_dimension(self, vector: List[float]) -> None:
        """
        Extracts and asserts vector dimensions.
        """
        if self.dimension is None:
            self.dimension = len(vector)
        elif len(vector) != self.dimension:
            raise DimensionMismatchError(
                f"Vector dimension {len(vector)} does not match index dimension {self.dimension}."
            )

    def add_documents(self, documents: List[Document]) -> None:
        """
        Adds documents to Chroma, generating embeddings and validating vectors first.
        """
        if not documents:
            return

        logger.info(f"Preparing to insert {len(documents)} documents to Chroma.")
        
        # Extract and validate text contents
        texts = [doc.page_content for doc in documents]
        
        # Generate vectors
        vectors = self.embeddings.embed_documents(texts)
        
        # Validate vectors
        for vec in vectors:
            self._ensure_dimension(vec)
            validate_vector(vec, expected_dim=self.dimension)

        # Build LangChain document structures
        lc_docs = []
        doc_ids = []
        
        seen_batch_ids = set()
        for idx, doc in enumerate(documents):
            chunk_id = doc.metadata.get("chunk_id")
            if not chunk_id:
                raise ValueError(f"Document at index {idx} is missing 'chunk_id' metadata.")
            
            # Skip if duplicate within the current insertion batch
            if chunk_id in seen_batch_ids:
                logger.warning(f"Duplicate chunk ID in current batch: {chunk_id}. Skipping.")
                continue

            # Check for duplicate chunk_id in current index if loaded
            if self.db is not None:
                # Query index to verify ID uniqueness
                existing = self.db._collection.get(ids=[chunk_id])
                if existing and existing.get("ids"):
                    logger.warning(f"Duplicate chunk ID detected: {chunk_id}. Skipping.")
                    continue
                    
            seen_batch_ids.add(chunk_id)
            # Inject embedding model ID to metadata
            metadata = doc.metadata.copy()
            metadata["embedding_model"] = getattr(self.embeddings, "model_name", "unknown")
            
            lc_docs.append(LCDocument(page_content=doc.page_content, metadata=metadata))
            doc_ids.append(chunk_id)

        if not lc_docs:
            logger.info("No new non-duplicate documents to add to Chroma.")
            return

        try:
            start_time = time.time()
            if self.db is None:
                self.db = Chroma.from_documents(
                    documents=lc_docs,
                    embedding=self.lc_embeddings,
                    persist_directory=self.persist_directory,
                    ids=doc_ids
                )
            else:
                self.db.add_documents(documents=lc_docs, ids=doc_ids)
            
            self.last_updated_timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
            duration = time.time() - start_time
            logger.info(f"Successfully inserted {len(lc_docs)} vectors in {duration:.3f}s.")
            
        except Exception as e:
            logger.error(f"Failed to add documents to Chroma collection: {e}")
            raise VectorStoreError(f"Chroma insert transaction failed: {e}")

    def update_documents(self, documents: List[Document]) -> None:
        """
        Updates existing documents in Chroma matching metadata chunk_ids.
        """
        if not documents:
            return
            
        logger.info(f"Updating {len(documents)} documents in Chroma.")
        if self.db is None:
            raise IndexLoadError("Cannot update documents: index has not been loaded/created.")

        lc_docs = []
        doc_ids = []
        
        for doc in documents:
            chunk_id = doc.metadata.get("chunk_id")
            if not chunk_id:
                raise ValueError("Updated document is missing 'chunk_id' metadata.")
            
            metadata = doc.metadata.copy()
            metadata["embedding_model"] = getattr(self.embeddings, "model_name", "unknown")
            
            lc_docs.append(LCDocument(page_content=doc.page_content, metadata=metadata))
            doc_ids.append(chunk_id)

        try:
            # Enforce update
            self.db.update_documents(ids=doc_ids, documents=lc_docs)
            self.last_updated_timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
            logger.info("Chroma document updates committed.")
        except Exception as e:
            logger.error(f"Chroma update failed: {e}")
            raise VectorStoreError(f"Update failed: {e}")

    def remove_documents(self, document_ids: List[str]) -> None:
        """
        Removes documents matching chunk IDs.
        """
        if not document_ids:
            return
            
        logger.info(f"Removing {len(document_ids)} documents from Chroma.")
        if self.db is None:
            raise IndexLoadError("Cannot remove documents: index has not been loaded/created.")

        try:
            self.db.delete(ids=document_ids)
            self.last_updated_timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
            logger.info("Documents removed from Chroma collection.")
        except Exception as e:
            logger.error(f"Chroma delete transaction failed: {e}")
            raise VectorStoreError(f"Failed to delete vectors: {e}")

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        search_type: str = "dense",
        **kwargs
    ) -> List[Tuple[Document, float]]:
        """
        Runs similarity queries, tracking performance and scoring stats.
        """
        if self.db is None:
            logger.warning("Query run on uninitialized/empty Chroma database.")
            return []

        start_time = time.time()
        try:
            logger.info(f"Querying Chroma: '{query}' (k={k}, filter={filter})")
            results = self.db.similarity_search_with_score(query, k=k, filter=filter)
            
            duration = time.time() - start_time
            self._total_queries_run += 1
            self._total_search_time += duration
            
            # Map back to custom Document
            mapped_results = []
            
            for doc, score in results:
                # Track score analytics
                self._highest_score = max(self._highest_score, score)
                self._lowest_score = min(self._lowest_score, score)
                self._score_sum += score
                self._total_documents_returned += 1
                
                mapped_results.append(
                    (Document(page_content=doc.page_content, metadata=doc.metadata), score)
                )

            # Update count stats
            total_elements = self.db._collection.count()
            self._total_documents_searched += total_elements
            
            logger.info(f"Query completed in {duration:.3f}s. Fetched {len(mapped_results)} matches.")
            return mapped_results
            
        except Exception as e:
            logger.error(f"Chroma query failed: {e}")
            raise VectorStoreError(f"Failed to query database: {e}")

    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> List[Document]:
        """
        Executes MMR diverse ranking queries.
        """
        if self.db is None:
            return []

        logger.info(f"Querying Chroma MMR: '{query}' (k={k}, lambda={lambda_mult})")
        try:
            results = self.db.max_marginal_relevance_search(
                query=query,
                k=k,
                fetch_k=fetch_k,
                lambda_mult=lambda_mult,
                filter=filter
            )
            return [
                Document(page_content=doc.page_content, metadata=doc.metadata)
                for doc in results
            ]
        except Exception as e:
            logger.error(f"Chroma MMR query failed: {e}")
            raise VectorStoreError(f"Failed to query MMR: {e}")

    def save(self, path: str) -> None:
        """
        Chroma persists automatically on transaction commits.
        """
        logger.info("ChromaDB index is automatically saved/persisted.")

    def load(self, path: str) -> None:
        """
        Loads database from disk.
        """
        logger.info(f"Loading ChromaDB collection index from: {path}")
        self.persist_directory = path
        
        try:
            # Instantiate Chroma client
            self.db = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.lc_embeddings
            )
            
            # Fetch dimension dynamically if collection contains vectors
            count = self.db._collection.count()
            if count > 0:
                first_items = self.db._collection.get(limit=1, include=["embeddings"])
                if first_items and first_items.get("embeddings"):
                    self.dimension = len(first_items["embeddings"][0])
                    
            logger.info(f"ChromaDB loaded. Vector count={count}. Dimension={self.dimension}")
        except Exception as e:
            logger.error(f"Failed to load Chroma collection from disk: {e}")
            self.db = None
            raise IndexLoadError(f"Could not load Chroma database: {e}")

    def delete_index(self) -> None:
        """
        Clears client instances and deletes local directory files.
        """
        logger.info(f"Deleting ChromaDB collection files at: {self.persist_directory}")
        self.db = None
        self.dimension = None
        if os.path.exists(self.persist_directory):
            try:
                shutil.rmtree(self.persist_directory)
                logger.info("ChromaDB index files deleted.")
            except Exception as e:
                logger.error(f"Failed to delete Chroma index folder: {e}")
                raise VectorStoreError(f"Chroma index deletion failed: {e}")

    def reset(self) -> None:
        """
        Recreates/resets the current index collection database.
        """
        self.delete_index()
        logger.info("ChromaDB collection has been reset.")

    def get_statistics(self) -> Dict[str, Any]:
        """
        Gathers database analytics metrics.
        """
        vector_count = self.db._collection.count() if self.db is not None else 0
        avg_score = 0.0
        if self._total_documents_returned > 0:
            avg_score = self._score_sum / self._total_documents_returned

        avg_latency = 0.0
        if self._total_queries_run > 0:
            avg_latency = self._total_search_time / self._total_queries_run

        return {
            "total_queries_run": self._total_queries_run,
            "average_search_latency": avg_latency,
            "total_search_time": self._total_search_time,
            "total_vectors_searched": self._total_documents_searched,
            "total_vectors_returned": self._total_documents_returned,
            "average_similarity_score": avg_score,
            "highest_similarity_score": self._highest_score,
            "lowest_similarity_score": self._lowest_score,
            "number_of_vectors": vector_count
        }

    def get_index_info(self) -> Dict[str, Any]:
        """
        Gathers static database settings metadata.
        """
        vector_count = self.db._collection.count() if self.db is not None else 0
        return {
            "database_type": "chromadb",
            "index_size": vector_count,
            "embedding_dimension": self.dimension or 0,
            "embedding_model": getattr(self.embeddings, "model_name", "unknown"),
            "creation_timestamp": self.creation_timestamp,
            "last_updated_timestamp": self.last_updated_timestamp,
            "persist_directory": self.persist_directory
        }


# Backwards-compatible alias
ChromaStore = ChromaVectorStore
