"""
Web document loader using requests and BeautifulSoup4 to parse URLs.
"""

from typing import List
import requests
from bs4 import BeautifulSoup
from src.ingestion.base import BaseDocumentLoader, Document
from src.logger import setup_logger

logger = setup_logger("web_loader")

class WebLoader(BaseDocumentLoader):
    """
    Crawls and parses a single web page url, extracting text content.
    """
    def load(self) -> List[Document]:
        """
        Fetches webpage content via HTTP request and processes the raw text.
        
        Returns:
            List[Document]: List of extracted webpage documents.
        """
        logger.info(f"Initiating HTTP GET request to web URL: {self.source}")
        try:
            # Set user-agent header to avoid getting blocked by default protections
            headers = {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/91.0.4472.124 Safari/537.36"
                )
            }
            
            response = requests.get(self.source, headers=headers, timeout=15)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, "html.parser")
            
            # Remove script and style elements
            for script_or_style in soup(["script", "style", "meta", "noscript", "header", "footer"]):
                script_or_style.decompose()
                
            # Extract title
            title = soup.title.string.strip() if soup.title else "Web Page Content"
            
            # Extract and clean text
            text = soup.get_text(separator="\n")
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            clean_text = "\n".join(chunk for chunk in chunks if chunk)
            
            metadata = {
                "source": self.source,
                "title": title,
                "status_code": response.status_code
            }
            
            logger.info(f"Successfully crawled and parsed URL '{self.source}' (Title: {title})")
            return [Document(page_content=clean_text, metadata=metadata)]
            
        except Exception as e:
            logger.error(f"Error fetching web page {self.source}: {e}")
            raise RuntimeError(f"Failed to crawl web URL: {e}")
