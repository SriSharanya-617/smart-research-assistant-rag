"""
FAISS implementation of the BaseVectorStore interface.
"""

import os
import shutil
from typing import List, Optional
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document as LCDocument
from src.embeddings.base import BaseEmbeddings
from src.ingestion.base import Document
from src.logger import setup_logger
from src.vectorstores.base import BaseVectorStore

logger = setup_logger("faiss_store")

class FAISSStore(BaseVectorStore):
    """
    Concrete wrapper for FAISS local database index.
    """
    def __init__(self, embeddings: BaseEmbeddings, persist_directory: str):
        self.embeddings = embeddings
        self.persist_directory = persist_directory
        self.db: Optional[FAISS] = None

        if hasattr(embeddings, "lc_embeddings"):
            self.lc_embeddings = embeddings.lc_embeddings
        else:
            self.lc_embeddings = embeddings

        logger.info(f"FAISSStore initialized. Save folder: {persist_directory}")

    def add_documents(self, documents: List[Document]) -> None:
        """
        Translates documents and builds/adds to the FAISS index.
        """
        lc_docs = [
            LCDocument(page_content=doc.page_content, metadata=doc.metadata)
            for doc in documents
        ]
        
        logger.info(f"Adding {len(lc_docs)} documents to FAISS index.")
        try:
            if self.db is None:
                # FAISS requires at least one document to construct using from_documents
                self.db = FAISS.from_documents(
                    documents=lc_docs,
                    embedding=self.lc_embeddings
                )
            else:
                self.db.add_documents(lc_docs)
            
            # Persist to disk immediately
            self.save(self.persist_directory)
            logger.info("Documents added to FAISS and saved.")
        except Exception as e:
            logger.error(f"Error adding documents to FAISS: {e}")
            raise RuntimeError(f"FAISS add failed: {e}")

    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """
        Queries FAISS and returns mapped documents.
        """
        if self.db is None:
            # Try to load from disk first
            if os.path.exists(os.path.join(self.persist_directory, "index.faiss")):
                self.load(self.persist_directory)
            else:
                logger.warning("FAISS DB index is empty and none found on disk. Returning empty.")
                return []

        try:
            logger.info(f"Searching FAISS index for: '{query}' (k={k})")
            results = self.db.similarity_search(query, k=k)
            return [
                Document(page_content=doc.page_content, metadata=doc.metadata)
                for doc in results
            ]
        except Exception as e:
            logger.error(f"FAISS similarity search failed: {e}")
            return []

    def delete_index(self) -> None:
        """
        Deletes FAISS index files on disk.
        """
        logger.info(f"Deleting FAISS index directory at: {self.persist_directory}")
        self.db = None
        if os.path.exists(self.persist_directory):
            try:
                shutil.rmtree(self.persist_directory)
                logger.info("FAISS index deleted from disk.")
            except Exception as e:
                logger.error(f"Failed to delete FAISS directory: {e}")

    def save(self, path: str) -> None:
        """
        Saves the FAISS index files.
        """
        if self.db is not None:
            try:
                os.makedirs(path, exist_ok=True)
                self.db.save_local(path)
                logger.info(f"FAISS index saved local to: {path}")
            except Exception as e:
                logger.error(f"Failed to save FAISS locally: {e}")
        else:
            logger.warning("Cannot save FAISS: index has not been built/loaded yet.")

    def load(self, path: str) -> None:
        """
        Loads FAISS from directory path.
        """
        try:
            self.persist_directory = path
            # allow_dangerous_deserialization = True is safe since we only load index files we create locally.
            self.db = FAISS.load_local(
                folder_path=self.persist_directory,
                embeddings=self.lc_embeddings,
                allow_dangerous_deserialization=True
            )
            logger.info(f"FAISS DB successfully loaded from: {path}")
        except Exception as e:
            logger.error(f"Failed to load FAISS DB: {e}")
            self.db = None
            raise
