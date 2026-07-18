"""
Production-quality web scraper and HTML parser loader.
Implements URL checks, robots.txt compliance, requests retry logic, timeout, and HTML tag filtering.
"""

import time
import hashlib
from typing import List, Callable, Optional
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser
import requests
from bs4 import BeautifulSoup
from src.ingestion.base import BaseDocumentLoader, Document
from src.ingestion.exceptions import WebScrapingError, IngestionCancelledError
from src.ingestion.preprocessing import TextPreprocessor
from src.logger import setup_logger

logger = setup_logger("web_loader")

class WebLoader(BaseDocumentLoader):
    """
    Crawls and extracts main content from a given web URL.
    Checks robots.txt compliance and handles network failures gracefully.
    """
    def __init__(
        self,
        source: str,
        timeout: int = 10,
        max_retries: int = 3,
        preprocessor: Optional[TextPreprocessor] = None
    ):
        super().__init__(source)
        self.timeout = timeout
        self.max_retries = max_retries
        self.preprocessor = preprocessor or TextPreprocessor()
        self.user_agent = "SmartResearchAssistantBot/1.0 (+https://github.com/SmartResearchAssistant)"

    def _validate_url(self) -> None:
        """
        Validates URL formatting.
        """
        try:
            parsed = urlparse(self.source)
            if not parsed.scheme or not parsed.netloc:
                raise ValueError("URL is missing scheme (http/https) or netloc domain.")
            if parsed.scheme.lower() not in ["http", "https"]:
                raise ValueError(f"Unsupported URL scheme '{parsed.scheme}'. Only http and https are allowed.")
        except Exception as e:
            logger.error(f"URL validation failed for '{self.source}': {e}")
            raise ValueError(f"Malformed or invalid URL: {self.source}. Details: {e}")

    def _check_robots_txt(self) -> bool:
        """
        Fetches and checks robots.txt rules for the domain.
        
        Returns:
            bool: True if allowed to fetch, False if disallowed.
        """
        try:
            parsed = urlparse(self.source)
            robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
            logger.debug(f"Checking robots.txt compliance at {robots_url}")
            
            rp = RobotFileParser()
            # Retrieve robots.txt with custom user agent and timeout
            response = requests.get(
                robots_url,
                headers={"User-Agent": self.user_agent},
                timeout=3
            )
            
            if response.status_code == 200:
                rp.parse(response.text.splitlines())
                allowed = rp.can_fetch(self.user_agent, self.source)
                logger.info(f"robots.txt check returned allowed={allowed} for URL: {self.source}")
                return allowed
            elif response.status_code == 404:
                # No robots file exists, default to allowed
                logger.debug("No robots.txt found (HTTP 404). Allowed by default.")
                return True
        except Exception as e:
            logger.warning(f"Unable to verify robots.txt restrictions for {self.source}: {e}. Proceeding.")
            
        return True

    def load(
        self,
        progress_callback: Optional[Callable[[float], None]] = None,
        cancellation_check: Optional[Callable[[], bool]] = None
    ) -> List[Document]:
        """
        Loads and parses the webpage.
        
        Args:
            progress_callback: Optional progress indicator.
            cancellation_check: Optional cancellation checker.
            
        Returns:
            List[Document]: Singleton list containing webpage document.
        """
        # 1. Validate URL structure
        self._validate_url()

        # Check cancellation
        if cancellation_check and cancellation_check():
            raise IngestionCancelledError("Ingestion cancelled prior to web crawl.")

        if progress_callback:
            progress_callback(0.1)

        # 2. Check robots.txt permissions
        if not self._check_robots_txt():
            logger.error(f"URL fetch blocked by robots.txt policy: {self.source}")
            raise WebScrapingError(f"Scraping denied by robots.txt instructions for URL: {self.source}")

        if progress_callback:
            progress_callback(0.2)

        # 3. HTTP Request with Exponential Backoff Retries
        headers = {"User-Agent": self.user_agent}
        response = None
        backoff = 1.0

        for attempt in range(self.max_retries):
            # Check cancellation in retry loop
            if cancellation_check and cancellation_check():
                raise IngestionCancelledError("Ingestion cancelled during HTTP retry loop.")

            try:
                logger.info(f"Fetching URL (Attempt {attempt+1}/{self.max_retries}): {self.source}")
                response = requests.get(self.source, headers=headers, timeout=self.timeout)
                response.raise_for_status()
                break
            except requests.exceptions.Timeout as e:
                logger.warning(f"Connection timeout on attempt {attempt+1}: {e}")
                if attempt == self.max_retries - 1:
                    raise WebScrapingError(f"HTTP request timed out after {self.max_retries} attempts.")
            except requests.exceptions.RequestException as e:
                logger.warning(f"Request failed on attempt {attempt+1}: {e}")
                if attempt == self.max_retries - 1:
                    raise WebScrapingError(f"HTTP request failed: {e}")
                
            time.sleep(backoff)
            backoff *= 2.0

        if progress_callback:
            progress_callback(0.6)

        # 4. Parse content using BeautifulSoup
        try:
            logger.info("Parsing page HTML structure.")
            soup = BeautifulSoup(response.content, "html.parser")
            
            # Decompose boilerplate, scripts, metadata, and advertisements
            for tag in soup(["script", "style", "header", "footer", "nav", "aside", "noscript", "iframe"]):
                tag.decompose()

            # Remove navigation blocks, share boxes, and ads classes/IDs
            ad_selectors = [
                ".ad", ".ads", ".sidebar", ".navigation", ".footer", 
                "#footer", "#header", "#sidebar", ".advertisement", 
                ".social-share", ".cookie-banner", "#nav"
            ]
            for sel in ad_selectors:
                for tag in soup.select(sel):
                    tag.decompose()

            title = soup.title.string.strip() if soup.title else "Web Page Content"
            
            # Extract text elements
            text_blocks = soup.get_text(separator="\n")
            
            # Clean text
            cleaned_text = self.preprocessor.clean_text(text_blocks)
            
            if not cleaned_text:
                raise WebScrapingError(f"Web scraper returned empty text content from URL: {self.source}")

            # 5. Generate SHA-256 based on extracted text contents
            text_bytes = cleaned_text.encode("utf-8")
            document_id = hashlib.sha256(text_bytes).hexdigest()

            # Parse domain
            parsed_url = urlparse(self.source)
            domain = parsed_url.netloc

            metadata = {
                "document_id": document_id,
                "filename": title,
                "page_number": 1,
                "total_pages": 1,
                "source": self.source,
                "domain": domain,
                "document_type": "web",
                "title": title
            }

            if progress_callback:
                progress_callback(1.0)

            logger.info(f"Successfully scraped and cleaned page content: {title}")
            return [Document(page_content=cleaned_text, metadata=metadata)]

        except IngestionError:
            raise
        except Exception as e:
            logger.error(f"HTML parsing failure: {e}")
            raise WebScrapingError(f"Failed to parse scraped webpage content: {e}")
