"""
ChromaDB implementation of the BaseVectorStore interface.
"""

import os
import shutil
from typing import List, Optional
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document as LCDocument
from src.embeddings.base import BaseEmbeddings
from src.ingestion.base import Document
from src.logger import setup_logger
from src.vectorstores.base import BaseVectorStore

logger = setup_logger("chroma_store")

class ChromaStore(BaseVectorStore):
    """
    Concrete wrapper for ChromaDB.
    """
    def __init__(self, embeddings: BaseEmbeddings, persist_directory: str):
        self.embeddings = embeddings
        self.persist_directory = persist_directory
        self.db: Optional[Chroma] = None
        
        # Extract underlying langchain embeddings if wrapped
        # Chroma expects a LangChain embedding instance
        if hasattr(embeddings, "lc_embeddings"):
            self.lc_embeddings = embeddings.lc_embeddings
        else:
            # Fallback wrapper
            self.lc_embeddings = embeddings
            
        logger.info(f"ChromaStore initialized. Persisting to: {persist_directory}")

    def add_documents(self, documents: List[Document]) -> None:
        """
        Translates our custom Documents to LangChain format and inserts into Chroma.
        """
        lc_docs = [
            LCDocument(page_content=doc.page_content, metadata=doc.metadata)
            for doc in documents
        ]
        
        logger.info(f"Adding {len(lc_docs)} documents to Chroma index.")
        try:
            if self.db is None:
                self.db = Chroma.from_documents(
                    documents=lc_docs,
                    embedding=self.lc_embeddings,
                    persist_directory=self.persist_directory
                )
            else:
                self.db.add_documents(lc_docs)
            logger.info("Documents added and Chroma DB persisted.")
        except Exception as e:
            logger.error(f"Error adding documents to Chroma: {e}")
            raise RuntimeError(f"Chroma add failed: {e}")

    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """
        Queries Chroma and parses back to custom Documents.
        """
        if self.db is None:
            logger.warning("Chroma DB not loaded or index is empty. Returning empty list.")
            return []

        try:
            logger.info(f"Searching Chroma index for: '{query}' (k={k})")
            results = self.db.similarity_search(query, k=k)
            
            return [
                Document(page_content=doc.page_content, metadata=doc.metadata)
                for doc in results
            ]
        except Exception as e:
            logger.error(f"Chroma similarity search failed: {e}")
            return []

    def delete_index(self) -> None:
        """
        Deletes the Chroma database from local disk.
        """
        logger.info(f"Deleting Chroma database directory at: {self.persist_directory}")
        self.db = None
        if os.path.exists(self.persist_directory):
            try:
                shutil.rmtree(self.persist_directory)
                logger.info("Chroma index deleted from disk.")
            except Exception as e:
                logger.error(f"Failed to delete Chroma directory: {e}")

    def save(self, path: str) -> None:
        """
        Chroma automatically persists or saves when writing.
        """
        logger.info("Chroma saves automatically upon updates.")

    def load(self, path: str) -> None:
        """
        Instantiates Chroma from disk.
        """
        try:
            self.persist_directory = path
            self.db = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.lc_embeddings
            )
            logger.info(f"Chroma DB successfully loaded from: {path}")
        except Exception as e:
            logger.error(f"Failed to load Chroma DB: {e}")
            self.db = None
            raise
