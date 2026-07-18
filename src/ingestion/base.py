"""
Base loader interface and document data models.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List

class Document:
    """
    Representation of a single parsed block of text with associated metadata.
    Matches the schema structure of standard LangChain Documents.
    """
    def __init__(self, page_content: str, metadata: Dict[str, Any] = None):
        self.page_content = page_content
        self.metadata = metadata or {}

    def __repr__(self) -> str:
        return f"Document(page_content_len={len(self.page_content)}, metadata={self.metadata})"


class BaseDocumentLoader(ABC):
    """
    Abstract Base Class representing a document loader.
    """
    def __init__(self, source: str):
        """
        Args:
            source: Absolute path to a file or web page URL.
        """
        self.source = source

    @abstractmethod
    def load(self) -> List[Document]:
        """
        Parses the source and returns a list of Document objects.
        
        Returns:
            List[Document]: List of extracted documents.
        """
        pass
