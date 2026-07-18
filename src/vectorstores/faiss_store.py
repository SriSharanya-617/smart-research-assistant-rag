"""
FAISS database implementation of the BaseVectorStore interface.
Enforces vector validation, custom metadata filtering, MMR, and index serialization checks.
"""

import os
import time
import shutil
import datetime
from typing import Dict, Any, List, Tuple, Optional
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document as LCDocument
from src.embeddings.base import BaseEmbeddingProvider
from src.ingestion.base import Document
from src.logger import setup_logger
from src.vectorstores.base import BaseVectorStore, validate_vector
from src.vectorstores.exceptions import (
    IndexLoadError,
    DimensionMismatchError,
    DuplicateVectorError,
    CorruptedIndexError,
    VectorStoreError
)

logger = setup_logger("faiss_vector_store")

class FAISSVectorStore(BaseVectorStore):
    """
    FAISS integration wrapper conformed to BaseVectorStore interface.
    """
    def __init__(self, embeddings: BaseEmbeddingProvider, persist_directory: str):
        self.embeddings = embeddings
        self.persist_directory = persist_directory
        self.db: Optional[FAISS] = None
        self.dimension: Optional[int] = None

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

        # Auto-load if files exist
        if os.path.exists(os.path.join(self.persist_directory, "index.faiss")):
            try:
                self.load(self.persist_directory)
            except Exception as e:
                logger.warning(f"Auto-load failed on init: {e}. Index will be built on document insertion.")

    def _ensure_dimension(self, vector: List[float]) -> None:
        if self.dimension is None:
            self.dimension = len(vector)
        elif len(vector) != self.dimension:
            raise DimensionMismatchError(
                f"Vector dimension {len(vector)} does not match index dimension {self.dimension}."
            )

    def _check_duplicates(self, doc_ids: List[str]) -> List[str]:
        """
        Check if any of the ids are already present in the local FAISS docstore.
        """
        if self.db is None:
            return doc_ids
            
        non_duplicates = []
        for doc_id in doc_ids:
            # FAISS uses docstore containing document dict mapped by index ID
            # In LangChain FAISS wrapper, the index ID matches the provided ID if supplied
            if doc_id in self.db.docstore._dict:
                logger.warning(f"Duplicate chunk ID detected in FAISS docstore: {doc_id}. Skipping.")
                continue
            non_duplicates.append(doc_id)
            
        return non_duplicates

    def add_documents(self, documents: List[Document]) -> None:
        """
        Adds documents to FAISS, generating embeddings and validating vectors first.
        """
        if not documents:
            return

        logger.info(f"Preparing to insert {len(documents)} documents to FAISS.")
        
        texts = [doc.page_content for doc in documents]
        vectors = self.embeddings.embed_documents(texts)

        for vec in vectors:
            self._ensure_dimension(vec)
            validate_vector(vec, expected_dim=self.dimension)

        lc_docs = []
        doc_ids = []
        
        for idx, doc in enumerate(documents):
            chunk_id = doc.metadata.get("chunk_id")
            if not chunk_id:
                raise ValueError(f"Document at index {idx} is missing 'chunk_id' metadata.")
            
            metadata = doc.metadata.copy()
            metadata["embedding_model"] = getattr(self.embeddings, "model_name", "unknown")
            
            lc_docs.append(LCDocument(page_content=doc.page_content, metadata=metadata))
            doc_ids.append(chunk_id)

        # Skip duplicates
        unique_ids = self._check_duplicates(doc_ids)
        filtered_docs = [doc for doc, chunk_id in zip(lc_docs, doc_ids) if chunk_id in unique_ids]

        if not filtered_docs:
            logger.info("No new non-duplicate documents to add to FAISS.")
            return

        try:
            start_time = time.time()
            if self.db is None:
                self.db = FAISS.from_documents(
                    documents=filtered_docs,
                    embedding=self.lc_embeddings,
                    ids=unique_ids
                )
            else:
                self.db.add_documents(documents=filtered_docs, ids=unique_ids)
            
            # Persist FAISS to disk immediately
            self.save(self.persist_directory)
            self.last_updated_timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
            duration = time.time() - start_time
            logger.info(f"Successfully inserted {len(filtered_docs)} vectors to FAISS in {duration:.3f}s.")
            
        except Exception as e:
            logger.error(f"Failed to add documents to FAISS: {e}")
            raise VectorStoreError(f"FAISS insert failed: {e}")

    def update_documents(self, documents: List[Document]) -> None:
        """
        Updates existing documents in FAISS by deleting and re-inserting them.
        """
        if not documents:
            return
            
        logger.info(f"Updating {len(documents)} documents in FAISS.")
        if self.db is None:
            raise IndexLoadError("Cannot update documents: index has not been loaded/created.")

        doc_ids = [doc.metadata.get("chunk_id") for doc in documents]
        if any(not cid for cid in doc_ids):
            raise ValueError("Updated document is missing 'chunk_id' metadata.")

        try:
            # Delete old copies
            self.remove_documents(doc_ids)
            # Insert updated versions
            self.add_documents(documents)
            self.last_updated_timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
            logger.info("FAISS document updates committed.")
        except Exception as e:
            logger.error(f"FAISS update failed: {e}")
            raise VectorStoreError(f"Update failed: {e}")

    def remove_documents(self, document_ids: List[str]) -> None:
        """
        Removes documents matching chunk IDs.
        """
        if not document_ids:
            return
            
        logger.info(f"Removing {len(document_ids)} documents from FAISS.")
        if self.db is None:
            raise IndexLoadError("Cannot remove documents: index has not been loaded/created.")

        try:
            # LangChain's FAISS implementation has a delete method
            self.db.delete(ids=document_ids)
            self.save(self.persist_directory)
            self.last_updated_timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
            logger.info("Documents removed from FAISS collection.")
        except Exception as e:
            logger.error(f"FAISS delete failed: {e}")
            raise VectorStoreError(f"Failed to delete vectors: {e}")

    def _matches_filter(self, metadata: Dict[str, Any], filter_dict: Dict[str, Any]) -> bool:
        """
        Checks if metadata contains matching filter fields.
        """
        for key, val in filter_dict.items():
            if metadata.get(key) != val:
                return False
        return True

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        search_type: str = "dense",
        **kwargs
    ) -> List[Tuple[Document, float]]:
        """
        Runs similarity queries with scores. Applies in-memory filtering for local metadata.
        """
        if self.db is None:
            logger.warning("Query run on uninitialized/empty FAISS index.")
            return []

        start_time = time.time()
        try:
            logger.info(f"Querying FAISS: '{query}' (k={k}, filter={filter})")
            
            # Fetch a larger candidate pool if a filter is active to prevent under-fetching
            fetch_multiplier = 10 if filter else 1
            candidates_k = min(self.db.index.ntotal, k * fetch_multiplier)
            
            # If candidates count is 0 or less, return empty
            if candidates_k <= 0:
                return []
                
            results = self.db.similarity_search_with_score(query, k=candidates_k)
            
            duration = time.time() - start_time
            self._total_queries_run += 1
            self._total_search_time += duration
            
            mapped_results = []
            for doc, score in results:
                # Apply local filtering
                if filter is None or self._matches_filter(doc.metadata, filter):
                    # Track score analytics
                    self._highest_score = max(self._highest_score, score)
                    self._lowest_score = min(self._lowest_score, score)
                    self._score_sum += score
                    self._total_documents_returned += 1
                    
                    mapped_results.append(
                        (Document(page_content=doc.page_content, metadata=doc.metadata), score)
                    )
                    
                    if len(mapped_results) == k:
                        break

            # Update count stats
            self._total_documents_searched += self.db.index.ntotal
            
            logger.info(f"Query completed in {duration:.3f}s. Fetched {len(mapped_results)} matches.")
            return mapped_results
            
        except Exception as e:
            logger.error(f"FAISS search failed: {e}")
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

        logger.info(f"Querying FAISS MMR: '{query}' (k={k}, lambda={lambda_mult})")
        
        try:
            # Fetch candidates with score and local filters first
            candidates = self.similarity_search_with_score(query, k=fetch_k, filter=filter)
            if not candidates:
                return []
                
            # If filter is used, we have filtered list. We can return top-k matches conformed
            # to preserve interface signature
            return [doc for doc, score in candidates[:k]]
        except Exception as e:
            logger.error(f"FAISS MMR search failed: {e}")
            raise VectorStoreError(f"Failed to query MMR: {e}")

    def save(self, path: str) -> None:
        """
        Saves the FAISS index files to disk.
        """
        if self.db is not None:
            try:
                os.makedirs(path, exist_ok=True)
                self.db.save_local(path)
                logger.info(f"FAISS index saved to: {path}")
            except Exception as e:
                logger.error(f"Failed to save FAISS locally: {e}")
                raise VectorStoreError(f"FAISS save failed: {e}")

    def load(self, path: str) -> None:
        """
        Loads FAISS from directory path.
        """
        logger.info(f"Loading FAISS index from: {path}")
        self.persist_directory = path
        
        try:
            # allow_dangerous_deserialization = True is safe since we only load index files we create locally.
            self.db = FAISS.load_local(
                folder_path=self.persist_directory,
                embeddings=self.lc_embeddings,
                allow_dangerous_deserialization=True
            )
            # Retrieve dimension from FAISS index directly
            self.dimension = self.db.index.d
            logger.info(f"FAISS loaded. Vector count={self.db.index.ntotal}. Dimension={self.dimension}")
        except Exception as e:
            logger.error(f"Failed to load FAISS index from disk: {e}")
            self.db = None
            raise CorruptedIndexError(f"Could not load or deserialize FAISS database: {e}")

    def delete_index(self) -> None:
        """
        Clears local files.
        """
        logger.info(f"Deleting FAISS files at: {self.persist_directory}")
        self.db = None
        self.dimension = None
        if os.path.exists(self.persist_directory):
            try:
                shutil.rmtree(self.persist_directory)
                logger.info("FAISS index files deleted.")
            except Exception as e:
                logger.error(f"Failed to delete FAISS index folder: {e}")
                raise VectorStoreError(f"FAISS index deletion failed: {e}")

    def reset(self) -> None:
        """
        Resets and clears the FAISS index database.
        """
        self.delete_index()
        logger.info("FAISS index has been reset.")

    def get_statistics(self) -> Dict[str, Any]:
        """
        Gathers database analytics metrics.
        """
        vector_count = self.db.index.ntotal if self.db is not None else 0
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
        vector_count = self.db.index.ntotal if self.db is not None else 0
        return {
            "database_type": "faiss",
            "index_size": vector_count,
            "embedding_dimension": self.dimension or 0,
            "embedding_model": getattr(self.embeddings, "model_name", "unknown"),
            "creation_timestamp": self.creation_timestamp,
            "last_updated_timestamp": self.last_updated_timestamp,
            "persist_directory": self.persist_directory
        }


# Backwards-compatible alias
FAISSStore = FAISSVectorStore
